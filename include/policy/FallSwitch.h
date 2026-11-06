#pragma once

#include "policy/RLPolicyInterface.h"

#include <Eigen/Core>

#include <mc_rbdyn/Robot.h>
#include <mc_rtc/Configuration.h>

#include <memory>
#include <string>
#include <vector>

namespace rlqp
{

/**
 * @brief Decides when the catch policy takes over from the walking policy, and when it gives control back.
 *
 * Runs on the policy tick, before the new action is chosen. It owns the two extra ONNX sessions (estimator and catch
 * policy), computes the 37 blind features of physics_features() in mjlab's scripts/fall_switch/common.py, keeps the
 * 5-frame window, and applies the rule of the bundle's README:
 *
 *   walking: switch when sigmoid(logits[switch horizon]) > thr
 *   holding: give control back after release_dwell_steps consecutive ticks inside the limit kernel
 *
 * p and the Mahalanobis distance are computed on every tick so that the logs are continuous. Each one only drives the
 * decision in its own state.
 */
class FallSwitch
{
public:
  static constexpr int FeatureCount = 37;

  FallSwitch();
  ~FallSwitch();

  /** @brief Load the networks and the bundle's sidecars. Throws if anything is missing or inconsistent. */
  void configure(const mc_rtc::Configuration & config,
                 const mc_rbdyn::Robot & robot,
                 const std::vector<std::string> & controllerJointOrder);

  /** @brief Back to walking, with an empty window. */
  void reset();

  /**
   * @brief One policy tick.
   * @param realRobot Measured state, up to date for this tick
   * @param previousTarget Joint targets in controller order, as applied since the previous tick: track_err compares
   * them with the measured joints, as closed_loop.py does by computing the features before choosing the action
   */
  void step(const mc_rbdyn::Robot & realRobot, const Eigen::VectorXd & previousTarget);

  bool configured() const { return configured_; }
  /** @brief Whether the automatic decisions are on. The scores are computed and the operator's choice applies anyway. */
  bool enabled() const { return configured_ && enabled_; }
  void setEnabled(bool enabled);

  /**
   * @brief Operator's choice: the catch policy holds control, with no automatic hand-back, until forceWalking().
   *
   * Unlike after an automatic switch, the operator's command still reaches the policy.
   */
  void forceRecovery();
  /** @brief Operator's choice: back to the walking policy, whose automatic supervision resumes. */
  void forceWalking();
  bool operatorHold() const { return operatorHold_; }

  const std::string & walkingPolicy() const { return walkingPolicy_; }
  const std::string & recoveryPolicy() const { return recoveryPolicy_; }
  RLPolicyInterface & recovery() { return *recovery_; }

  bool holdingRecovery() const { return holding_; }
  double p() const { return p_; }
  double threshold() const { return threshold_; }
  bool inside() const { return inside_; }
  double mahalanobis2() const { return mahalanobis2_; }
  double mahalanobis() const;
  double kernelThreshold() const { return kernelThreshold_; }
  int dwell() const { return dwell_; }
  int releaseDwellSteps() const { return releaseDwellSteps_; }
  const Eigen::VectorXd & features() const { return features_; }
  size_t switchCount() const { return switchCount_; }
  size_t handBackCount() const { return handBackCount_; }

private:
  struct Body
  {
    int index = 0;
    double mass = 0.0;
    Eigen::Vector3d com = Eigen::Vector3d::Zero();
  };

  /** A joint read by limit_margin, track_err and tau_sat. */
  struct LimbJoint
  {
    int controllerIndex = 0;
    int mbcIndex = 0;
    double lower = 0.0;
    double upper = 0.0;
    double effortLimit = 0.0;
  };

  struct Estimator;

  void computeFeatures(const mc_rbdyn::Robot & realRobot, const Eigen::VectorXd & previousTarget);
  void pushFrame();
  void updateScores();

  bool configured_ = false;
  bool enabled_ = true;

  std::string walkingPolicy_;
  std::string recoveryPolicy_;
  std::string gyroSensor_;
  std::unique_ptr<RLPolicyInterface> recovery_;
  std::unique_ptr<Estimator> estimator_;

  std::vector<Body> bodies_;
  std::vector<Eigen::Vector3d> bodyCom_;
  double totalMass_ = 0.0;
  size_t leftFoot_ = 0, rightFoot_ = 0, leftShoulder_ = 0, rightShoulder_ = 0, leftHip_ = 0, rightHip_ = 0;
  std::vector<LimbJoint> limbJoints_;

  int window_ = 0;
  int switchColumn_ = 0;
  Eigen::VectorXd mu_;
  Eigen::VectorXd sd_;
  double threshold_ = 0.0;
  Eigen::VectorXd kernelMu_;
  Eigen::MatrixXd kernelInvCov_;
  double kernelThreshold_ = 0.0;
  int releaseDwellSteps_ = 0;

  Eigen::VectorXd features_;
  std::vector<float> frames_; ///< Standardised window, oldest frame first
  bool windowEmpty_ = true;

  bool holding_ = false;
  bool operatorHold_ = false;
  double p_ = 0.0;
  bool inside_ = false;
  double mahalanobis2_ = 0.0;
  int dwell_ = 0;
  size_t switchCount_ = 0;
  size_t handBackCount_ = 0;
};

} // namespace rlqp
