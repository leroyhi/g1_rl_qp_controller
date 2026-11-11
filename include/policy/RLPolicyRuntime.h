#pragma once

#include "observation/Observation.h"
#include "observation/ObservationManager.h"
#include "policy/FallSwitch.h"
#include "policy/PolicyConfig.h"
#include "policy/RLPolicyInterface.h"

#include <Eigen/Core>

#include <mc_rtc/Configuration.h>
#include <mc_rbdyn/Robot.h>
#include <mc_tasks/TorqueJointTask.h>

#include <memory>
#include <string>
#include <vector>

struct G1RLQPController;

namespace rlqp
{

  /**
 * @brief Runtime owner of the active RL policy session.
 *
 * Responsibilities:
 *
 * - Load and switch policies.
 * - Own the ONNX inference backend.
 * - Own the ObservationManager.
 * - Build observation vectors.
 * - Execute policy inference.
 * - Convert policy outputs into q_rl targets.
 * - Manage policy gains and action scaling.
 *
 * G1RLQPController remains responsible for the mc_rtc lifecycle.
 * RLPolicyRuntime owns everything that is policy-dependent.
 */
class RLPolicyRuntime
{
public:
  RLPolicyRuntime();

  void configure(const mc_rtc::Configuration & controllerConfig,
                 G1RLQPController & ctl,
                 const std::shared_ptr<mc_tasks::TorqueJointTask> & torqueTask);

  /** @brief Disarm and hold the current posture. A reset never resumes armed. */
  void reset(G1RLQPController & ctl);
  void runPolicyStepIfNeeded(G1RLQPController & ctl, double dt);

  /** @brief Start the policy from a clean state: clock, last action and fall switch. */
  void arm(G1RLQPController & ctl, const std::shared_ptr<mc_tasks::TorqueJointTask> & torqueTask);
  /** @brief Stop the policy. The target stays at its last output, so this is a stop and not a jump. */
  void hold(const std::shared_ptr<mc_tasks::TorqueJointTask> & torqueTask);
  bool armed() const { return armed_; }

  FallSwitch & fallSwitch() { return fallSwitch_; }
  const FallSwitch & fallSwitch() const { return fallSwitch_; }
  /** @brief The switch only applies to the walking policy it was trained with. */
  bool fallSwitchActive() const;
  /** @brief Largest command the policy can see, per axis (vx, vy, wz). */
  const Eigen::Vector3d & commandMax() const { return commandMax_; }
  /** @brief Name of the policy whose action is applied. */
  const std::string & activePolicyName() const;

  void reloadCurrentPolicy(G1RLQPController & ctl,
                           const std::shared_ptr<mc_tasks::TorqueJointTask> & torqueTask);

  void loadPolicyByName(const std::string & policyName,
                        G1RLQPController & ctl,
                        const std::shared_ptr<mc_tasks::TorqueJointTask> & torqueTask);

  void loadNextPolicy(G1RLQPController & ctl,
                      const std::shared_ptr<mc_tasks::TorqueJointTask> & torqueTask);

  bool policyLoaded() const { return policy_ && policy_->isLoaded(); }

  const std::string & currentPolicyName() const { return policyManager_.currentName(); }
  const std::vector<std::string> & availablePolicyNames() const { return policyManager_.names(); }
  const std::string & currentPolicyFolder() const { return policyManager_.current().folder; }
  const std::string & conventionName() const { return observationManager_.conventionName(); }

  int observationSize() const;
  int actionSize() const;

  double policyStepSize() const { return policyStepSize_; }
  double policyRate() const { return 1.0 / policyStepSize_; }
  size_t policyUpdateCount() const { return policyUpdateCount_; }
  /** @brief Policy-wide observation_source of the active observations.yaml; entries may override it. */
  const std::string & observationSource() const { return observationManager_.defaultSource(); }
  const std::string & baseBody() const { return baseBody_; }
  size_t controlledActionSize() const { return controlledActionControllerIndices_.size(); }
  double phase() const { return phaseNormalized_; }

  bool useQP() const { return useQP_; }
  void setUseQP(bool useQP) { useQP_ = useQP; }

  const Eigen::VectorXd & q_rl() const { return q_rl_; }
  const Eigen::VectorXd & q_zero() const { return q_zero_; }
  const Eigen::VectorXd & currentObservation() const { return currentObservation_; }
  const Eigen::VectorXd & currentAction() const { return currentAction_; }
  const Eigen::VectorXd & currentActionScaled() const { return currentActionScaled_; }
  const Eigen::VectorXd & actionScale() const { return actionScale_; }

  const Eigen::VectorXd & kp() const { return kp_; }
  const Eigen::VectorXd & kd() const { return kd_; }
  const Eigen::VectorXd & kpBase() const { return kpBase_; }
  const Eigen::VectorXd & kdBase() const { return kdBase_; }

  double pdGainsRatio() const { return pdGainsRatio_; }

  void setPDGainsRatio(double ratio,
                       const std::shared_ptr<mc_tasks::TorqueJointTask> & torqueTask);

  /** @brief Add one logger entry per configured observation/history element. Called on each policy load. */
  void addLogObs(G1RLQPController & ctl);

  Eigen::Vector3d & command() { return command_; }
  const Eigen::Vector3d & command() const { return command_; }
  void setCommand(Eigen::Vector3d new_cmd)
  {
    command_ = new_cmd;
  }
  /** @brief Hold the current posture of the control robot (target of the torque task while disarmed). */
  void holdCurrentPosture(G1RLQPController & ctl);

  /** @brief Command seen by the policy at its last update. command() remains the operator's value. */
  const Eigen::Vector3d & effectiveCommand() const { return effectiveCommand_; }

private:
  void loadPolicy(const std::string & policyName,
                  G1RLQPController & ctl,
                  const std::shared_ptr<mc_tasks::TorqueJointTask> & torqueTask);

  void configureControl(const PolicyConfig & policy,
                        G1RLQPController & ctl,
                        const std::shared_ptr<mc_tasks::TorqueJointTask> & torqueTask);

  void configureAction(const PolicyConfig & policy, G1RLQPController & ctl);
  void configureNetwork(const PolicyConfig & policy);
  void configureObservations(const PolicyConfig & policy, G1RLQPController & ctl);

  void resetObservationHistory(G1RLQPController & ctl);
  Eigen::VectorXd computeObservation(G1RLQPController & ctl);
  void updateEffectiveCommand();

  void validateObservationAgainstNetwork() const;
  ObservationContext makeObservationContext(G1RLQPController & ctl);

private:
  mc_rtc::Configuration controllerConfig_;

  /** Available policy configurations. */
  PolicyManager policyManager_;
  /** Observation type factory registry. */
  ObservationRegistry observationRegistry_;
  /** Active observation pipeline. */
  ObservationManager observationManager_;
  /** Active training-environment convention. */
  ObservationConvention activeConvention_;
  /** Active ONNX policy instance. */
  std::unique_ptr<RLPolicyInterface> policy_;
  /** Hands control over to the catch policy, which has its own ONNX session. */
  FallSwitch fallSwitch_;

  bool armed_ = false;
  bool fallSwitchFailed_ = false;
  /** Gain ratio while disarmed: the policy's gains are too low to keep the robot up without it. */
  double holdGainsRatio_ = 1.0;

  std::string robotName_;
  std::string baseBody_ = "base_link";

  std::vector<std::string> controllerJointOrder_;
  /** @brief Maps active policy action index to controllerJointOrder_ index. policyJointControllerIndices in ObservationContext. */
  std::vector<int> actionToControllerMap_;

  /** @brief Controller-order joint indices that are actually allowed to receive the policy action. */
  std::vector<int> controlledActionControllerIndices_;

  Eigen::VectorXd q_rl_;
  Eigen::VectorXd q_zero_;
  Eigen::VectorXd currentObservation_;
  Eigen::VectorXd currentAction_;
  Eigen::VectorXd currentActionScaled_;
  Eigen::VectorXd actionScale_;

  Eigen::VectorXd kp_;
  Eigen::VectorXd kd_;
  Eigen::VectorXd kpBase_;
  Eigen::VectorXd kdBase_;

  Eigen::Vector3d command_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d effectiveCommand_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d commandMin_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d commandMax_ = Eigen::Vector3d::Zero();

  bool useQP_ = true;
  double policyStepSize_ = 0.02;
  double policyTimer_ = 0.0;
  size_t policyUpdateCount_ = 0;

  double phasePeriod_ = 1.0;
  double phaseElapsedTime_ = 0.0;
  double phaseNormalized_ = 0.0;

  double pdGainsRatio_ = 1.0;
};

} // namespace rlqp
