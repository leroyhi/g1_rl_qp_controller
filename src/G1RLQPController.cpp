#include "G1RLQPController.h"

#include <RBDyn/MultiBodyConfig.h>

#include <mc_rtc/gui.h>
#include <mc_rtc/logging.h>
#include <mc_tvm/Robot.h>

#include <fcntl.h>
#include <mc_joystick_plugin/joystick_inputs.h>
#include <termios.h>

#include <algorithm>
#include <cmath>
#include <filesystem>

namespace
{

/**
 * Add the reflected rotor inertia of each joint to the robot module, which has none.
 *
 * mjlab sets the gains from the armature, as kp = armature * (2 pi f)^2, so plant.json gives it back. This has to happen
 * before the base controller creates the robots: the TVM robot computes this part of its mass matrix once, on creation.
 */
mc_rbdyn::RobotModulePtr withArmature(mc_rbdyn::RobotModulePtr rm, const mc_rtc::Configuration & config)
{
  if(!config.has("limits") || !config("limits").has("armature_natural_frequency_hz")) return rm;
  const double frequency = config("limits")("armature_natural_frequency_hz");
  const double omega = 2.0 * M_PI * frequency;
  mc_rtc::Configuration plant;
  plant.load(config("limits")("plant").operator std::string());
  const mc_rtc::Configuration joints = plant("joints");
  for(size_t i = 0; i < joints.size(); ++i)
  {
    const mc_rtc::Configuration joint = joints[i];
    const std::string name = joint("name");
    const double kp = joint("kp");
    const int index = rm->mb.jointIndexByName(name);
    rm->mb.setJointRotorInertia(index, kp / (omega * omega));
    rm->mb.setJointGearRatio(index, 1.0);
  }
  mc_rtc::log::info("[G1RLQPController] QP model: added the armature of {} joints, from kp at f = {} Hz", joints.size(),
                    frequency);
  return rm;
}

} // namespace

G1RLQPController::G1RLQPController(mc_rbdyn::RobotModulePtr rm,
                                     double dt,
                                     const mc_rtc::Configuration & config)
: mc_control::fsm::Controller(withArmature(rm, config), dt, config, Backend::TVM)
{
  config_ = config;

  applyActuatorLimits();
  loadCBFParameters();

  // Joint-limit and self-collision protection. The dynamics constraint reads the limits when it joins the solver.
  selfCollisionConstraint->setCollisionsDampers(solver(), {zeta_selfCollision_, lambda_selfCollision_});
  solver().removeConstraintSet(dynamicsConstraint);
  dynamicsConstraint = mc_rtc::unique_ptr<mc_solver::DynamicsConstraint>(
    new mc_solver::DynamicsConstraint(robots(), 0, {diPercent_, dsPercent_, 0.0, zeta_jointLimit_, lambda_jointLimit_}, velPercent_, true));
  solver().addConstraintSet(dynamicsConstraint);

  // Remove the default posture task created by the FSM
  solver().removeTask(getPostureTask(robot().name()));
  // The stiffness is a placeholder: every policy load sets the per-joint gains of its policy.yaml
  torqueJointTask = std::make_shared<mc_tasks::TorqueJointTask>(solver(), robot().robotIndex(), 100.0, 1);
  solver().addTask(torqueJointTask);
  initializeRobotBasics();

  rlRuntime_.configure(config_, *this, torqueJointTask);
  if(config_.has("fall_switch") && config_("fall_switch").has("threshold_buttons"))
  {
    const mc_rtc::Configuration buttons = config_("fall_switch")("threshold_buttons");
    // strToButtons falls back to A, cross, on an unknown name: check the round trip instead
    const auto button = [](const std::string & name)
    {
      const joystickButtonInputs out = strToButtons(name);
      if(buttons2str(out) != name)
        mc_rtc::log::error_and_throw("[G1RLQPController] fall_switch.threshold_buttons: unknown button '{}'", name);
      if(out == joystickButtonInputs::A || out == joystickButtonInputs::B)
        mc_rtc::log::error_and_throw("[G1RLQPController] fall_switch.threshold_buttons: {} already selects a policy",
                                     name);
      return out;
    };
    thresholdDownButton_ = button(buttons("down"));
    thresholdUpButton_ = button(buttons("up"));
  }

  addGui();
  addLog();

  mc_rtc::log::success("[G1RLQPController] init done");
}

bool G1RLQPController::run()
{
  // Use joystick plugin if present else jeyboard inputs
  controllerAvailable = datastore().has("Joystick::connected") && datastore().get<bool>("Joystick::connected");
  if(controllerAvailable && toggleJoystick)
    RLuseJoyStickInputs();
  else if(toggleJoystick && joystickSeen_)
    rlRuntime_.setCommand(Eigen::Vector3d::Zero()); // Lost in joystick mode: nothing else is driving the command
  else if(toggleKeyboard)
    RLuseKeyboardInputs();

  if(printLimits_) computeLimits();
  bool run = mc_control::fsm::Controller::run(
          mc_solver::FeedbackType::ClosedLoopIntegrateReal);
  if(byPassQPControl()) // Run RL without taking the QP into account
  {
    return true;
  }
  return run; // Return false if QP fails
}

void G1RLQPController::reset(const mc_control::ControllerResetData & reset_data)
{
  mc_control::fsm::Controller::reset(reset_data);
  rlRuntime_.reset(*this);
}

void G1RLQPController::RLuseJoyStickInputs()
{
  joystickSeen_ = true;
  const Eigen::Vector2d leftStick = datastore().call<Eigen::Vector2d>("Joystick::Stick", joystickAnalogicInputs::L_STICK);
  const Eigen::Vector2d rightStick =
      datastore().call<Eigen::Vector2d>("Joystick::Stick", joystickAnalogicInputs::R_STICK);

  if(--joystickCheckCountdown_ <= 0)
  {
    joystickCheckCountdown_ = joystickCheckPeriod_;
    std::error_code error;
    joystickDevicePresent_ = std::filesystem::exists("/dev/input/js0", error);
  }
  const Eigen::Vector4d sticks(leftStick(0), leftStick(1), rightStick(0), rightStick(1));
  if(!joystickDevicePresent_ && !joystickLost_)
  {
    joystickLost_ = true;
    joystickFrozen_ = sticks;
    mc_rtc::log::error("[G1RLQPController] Joystick gone: command forced to zero until it is back, the JoystickPlugin "
                       "Reset button pressed and a stick moved");
  }
  if(joystickLost_)
  {
    if(!joystickDevicePresent_ || (sticks - joystickFrozen_).cwiseAbs().maxCoeff() <= 1e-6)
    {
      rlRuntime_.setCommand(Eigen::Vector3d::Zero());
      return;
    }
    joystickLost_ = false;
    mc_rtc::log::warning("[G1RLQPController] Joystick live again: command re-enabled");
  }

  // On the rising edge: the plugin's ButtonEvent also fires on a release
  const bool cross = datastore().call<bool>("Joystick::Button", joystickButtonInputs::A);
  const bool circle = datastore().call<bool>("Joystick::Button", joystickButtonInputs::B);
  // By value: an lvalue would make the datastore look for a bool(joystickButtonInputs &) and throw
  const bool thresholdDown = datastore().call<bool>("Joystick::Button", joystickButtonInputs(thresholdDownButton_));
  const bool thresholdUp = datastore().call<bool>("Joystick::Button", joystickButtonInputs(thresholdUpButton_));
  if(cross && !crossPressed_) selectWalkingPolicy();
  if(circle && !circlePressed_) selectRecoveryPolicy();
  if(thresholdDown && !thresholdDownPressed_) adjustFallSwitchThreshold(-1);
  if(thresholdUp && !thresholdUpPressed_) adjustFallSwitchThreshold(1);
  crossPressed_ = cross;
  circlePressed_ = circle;
  thresholdDownPressed_ = thresholdDown;
  thresholdUpPressed_ = thresholdUp;

  // Sticks: x is up and y is left, in [0, 1] with their rest position at 0.5
  const double maxVelY = std::min(maxVelCmd, rlRuntime_.commandMax()(1));
  Eigen::Vector3d target = Eigen::Vector3d::Zero();
  if(std::abs(leftStick(0) - 0.5) > joystickDeadZone) target(0) = (leftStick(0) - 0.5) * 2.0 * maxVelCmd;
  if(std::abs(leftStick(1) - 0.5) > joystickDeadZone) target(1) = (leftStick(1) - 0.5) * 2.0 * maxVelY;
  if(std::abs(rightStick(1) - 0.5) > joystickDeadZone) target(2) = (rightStick(1) - 0.5) * 2.0 * maxYawCmd;

  const double maxDelta = joystickRampRate * timeStep;
  Eigen::Vector3d command = rlRuntime_.command();
  for(int i = 0; i < 3; ++i) command(i) += std::clamp(target(i) - command(i), -maxDelta, maxDelta);
  rlRuntime_.setCommand(command);
}

void G1RLQPController::RLuseKeyboardInputs()
{
  struct Ctx
  {
    bool ready = false;
    termios old{};
    bool seen[4] = {};
    std::chrono::steady_clock::time_point ts[4];
    std::array<char, 64> buf{};
    size_t sz = 0;
  };
  static Ctx k;

  if(!k.ready)
  {
    if(::isatty(STDIN_FILENO) != 1) return;
    k.ready = true;
    ::tcgetattr(STDIN_FILENO, &k.old);
    termios raw = k.old;
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VMIN] = raw.c_cc[VTIME] = 0;
    ::tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    ::fcntl(STDIN_FILENO, F_SETFL, ::fcntl(STDIN_FILENO, F_GETFL, 0) | O_NONBLOCK);
  }

  char tmp[32];
  ssize_t n = ::read(STDIN_FILENO, tmp, sizeof(tmp));
  if(n > 0 && k.sz + n < 64) std::copy(tmp, tmp + n, k.buf.begin() + k.sz), k.sz += n;

  auto now = std::chrono::steady_clock::now();
  for(size_t i = 0; i + 2 < k.sz; ++i)
    if(k.buf[i] == 27 && k.buf[i + 1] == '[')
    {
      int idx = k.buf[i + 2] == 'A'   ? 0
                : k.buf[i + 2] == 'B' ? 1
                : k.buf[i + 2] == 'D' ? 2
                : k.buf[i + 2] == 'C' ? 3
                                      : -1;
      if(idx >= 0) k.seen[idx] = true, k.ts[idx] = now;
      i += 2;
    }
  std::copy(k.buf.begin() + (k.sz > 2 ? k.sz - 2 : 0), k.buf.begin() + k.sz, k.buf.begin());
  k.sz = k.sz > 2 ? 2 : 0;

  const auto active = [&](int i)
  { return k.seen[i] && std::chrono::duration_cast<std::chrono::milliseconds>(now - k.ts[i]).count() < 500; };
  rlRuntime_.setCommand({(active(0) ? maxVelCmd : 0.0) - (active(1) ? maxVelCmd : 0.0),
                         (active(2) ? maxVelCmd : 0.0) - (active(3) ? maxVelCmd : 0.0), rlRuntime_.command()(2)});
}

void G1RLQPController::armPolicy()
{
  rlRuntime_.arm(*this, torqueJointTask);
}

void G1RLQPController::holdPolicy()
{
  rlRuntime_.hold(torqueJointTask);
}

void G1RLQPController::selectWalkingPolicy()
{
  if(!rlRuntime_.fallSwitchActive())
  {
    mc_rtc::log::error("[G1RLQPController] The current policy is not the walking policy of the fall switch");
    return;
  }
  armPolicy();
  rlRuntime_.fallSwitch().forceWalking();
  mc_rtc::log::info("[G1RLQPController] Operator: {} takes control", rlRuntime_.fallSwitch().walkingPolicy());
}

void G1RLQPController::adjustFallSwitchThreshold(int steps)
{
  rlqp::FallSwitch & fallSwitch = rlRuntime_.fallSwitch();
  if(!fallSwitch.configured()) return;
  const double before = fallSwitch.threshold();
  const double after = fallSwitch.adjustThreshold(steps * rlqp::FallSwitch::ThresholdStep);
  if(after == before)
    mc_rtc::log::warning("[G1RLQPController] Fall switch threshold already at its bound: {:.4f}", after);
  else
    mc_rtc::log::info("[G1RLQPController] Fall switch threshold: {:.4f} -> {:.4f} (bundle: {:.4f})", before, after,
                      fallSwitch.defaultThreshold());
}

void G1RLQPController::selectRecoveryPolicy()
{
  if(!rlRuntime_.fallSwitchActive())
  {
    mc_rtc::log::error("[G1RLQPController] The current policy is not the walking policy of the fall switch");
    return;
  }
  armPolicy();
  rlRuntime_.fallSwitch().forceRecovery();
  mc_rtc::log::info("[G1RLQPController] Operator: {} takes control", rlRuntime_.fallSwitch().recoveryPolicy());
}

void G1RLQPController::activateQPControl(bool activate)
{
  rlRuntime_.setUseQP(activate);
}

rlqp::RLPolicyRuntime & G1RLQPController::rlRuntime() { return rlRuntime_; }

const rlqp::RLPolicyRuntime & G1RLQPController::rlRuntime() const { return rlRuntime_; }

void G1RLQPController::applyActuatorLimits()
{
  if(!config_.has("limits"))
    mc_rtc::log::error_and_throw("[G1RLQPController] Missing 'limits' section: the robot would run with its URDF limits");
  const mc_rtc::Configuration limits = config_("limits");
  const std::string plantPath = limits("plant");
  const std::string band = limits("position", std::string("soft"));
  if(band != "soft" && band != "hard")
    mc_rtc::log::error_and_throw("[G1RLQPController] limits.position must be 'soft' or 'hard', got '{}'", band);

  mc_rtc::Configuration plant;
  plant.load(plantPath);
  const mc_rtc::Configuration joints = plant("joints");
  if(joints.size() != robot().refJointOrder().size())
    mc_rtc::log::error_and_throw("[G1RLQPController] {} has {} joints, the robot has {} actuated joints", plantPath,
                                 joints.size(), robot().refJointOrder().size());

  std::vector<std::string> names;
  std::vector<std::string> urdfNotes;
  for(size_t i = 0; i < joints.size(); ++i)
  {
    const mc_rtc::Configuration joint = joints[i];
    const std::string name = joint("name");
    if(!robot().hasJoint(name))
      mc_rtc::log::error_and_throw("[G1RLQPController] {} names joint '{}', which the robot does not have", plantPath,
                                   name);
    const double torque = joint("torque_limit");
    const double velocity = joint("velocity_limit");
    const std::vector<double> position = joint(band == "soft" ? "pos_limit_soft" : "pos_limit_hard");
    const size_t j = robot().jointIndexByName(name);

    const double urdfTorque = robot().tu()[j][0];
    const double urdfVelocity = robot().vu()[j][0];
    for(mc_rbdyn::Robot * r : {&robot(), &realRobot()})
    {
      r->tl()[j][0] = -torque;
      r->tu()[j][0] = torque;
      r->vl()[j][0] = -velocity;
      r->vu()[j][0] = velocity;
      r->ql()[j][0] = std::max(r->ql()[j][0], position[0]);
      r->qu()[j][0] = std::min(r->qu()[j][0], position[1]);
    }
    names.push_back(name);
    urdfNotes.push_back(urdfTorque != torque || urdfVelocity != velocity
                            ? fmt::format(" (URDF: {} Nm, {} rad/s)", urdfTorque, urdfVelocity)
                            : std::string());
  }

  // The TVM robot caches its limits when it is created, and these are the ones the QP reads
  const auto & mb = robot().mb();
  auto & tvmLimits = robot().tvmRobot().limits();
  tvmLimits.ql = rbd::paramToVector(mb, robot().ql());
  tvmLimits.qu = rbd::paramToVector(mb, robot().qu());
  tvmLimits.vl = rbd::dofToVector(mb, robot().vl());
  tvmLimits.vu = rbd::dofToVector(mb, robot().vu());
  tvmLimits.tl = rbd::dofToVector(mb, robot().tl());
  tvmLimits.tu = rbd::dofToVector(mb, robot().tu());

  mc_rtc::log::info("[G1RLQPController] QP joint limits, from {} with its {} position band:", plantPath, band);
  for(size_t i = 0; i < names.size(); ++i)
  {
    const int j = mb.jointIndexByName(names[i]);
    const int dof = mb.jointPosInDof(j);
    const int param = mb.jointPosInParam(j);
    mc_rtc::log::info("  {}: torque [{}, {}] Nm, velocity [{}, {}] rad/s, position [{:.3f}, {:.3f}] rad{}", names[i],
                      tvmLimits.tl(dof), tvmLimits.tu(dof), tvmLimits.vl(dof), tvmLimits.vu(dof), tvmLimits.ql(param),
                      tvmLimits.qu(param), urdfNotes[i]);
  }
}

void G1RLQPController::loadCBFParameters()
{
  config_("print_limits", printLimits_);
  if(!config_.has("cbf")) return;
  const mc_rtc::Configuration cbf = config_("cbf");
  cbf("velocity_percent", velPercent_);
  cbf("position_percent", dsPercent_);
  if(cbf.has("joint_limit"))
  {
    cbf("joint_limit")("zeta", zeta_jointLimit_);
    cbf("joint_limit")("lambda", lambda_jointLimit_);
  }
  if(cbf.has("self_collision"))
  {
    cbf("self_collision")("zeta", zeta_selfCollision_);
    cbf("self_collision")("lambda", lambda_selfCollision_);
  }
}

void G1RLQPController::initializeRobotBasics()
{
  mc_rtc::log::info("[G1RLQPController] Using torque control mode");
  if(!datastore().has("ControlMode"))
  {
    datastore().make<std::string>("ControlMode", "Torque");
  }
  else
  {
    datastore().assign<std::string>("ControlMode", "Torque");
  }

  robotName_ = robot().name();
  jointNames = robot().refJointOrder();
  nbActuatedJoints = static_cast<int>(jointNames.size());
}

bool G1RLQPController::byPassQPControl()
{
  if(rlRuntime_.useQP()) return false; // QP is not bypassed, do nothing

  robot().forwardKinematics();
  robot().forwardVelocity();
  robot().forwardAcceleration();

  // The measured state: after run(), robot() holds the state the QP integrated forward, with its own accelerations.
  Eigen::VectorXd tau_rl = Eigen::VectorXd::Zero(nbActuatedJoints);
  const std::vector<std::vector<double> > & q_mbc = realRobot().mbc().q;
  const std::vector<std::vector<double> > & q_dot_mbc = realRobot().mbc().alpha;

  int i = 0;
  for(const auto &joint_name : jointNames)
  {
    const size_t j = robot().jointIndexByName(joint_name);
    const double q = q_mbc[j][0];
    const double q_dot = q_dot_mbc[j][0];
    tau_rl(i) = std::clamp(rlRuntime_.kp()(i) * (rlRuntime_.q_rl()(i) - q) - rlRuntime_.kd()(i) * q_dot,
                           robot().tl()[j][0], robot().tu()[j][0]);
    robot().mbc().jointTorque[j][0] = tau_rl(i);
    i++;
  }

  return true;
}

void G1RLQPController::addLog()
{
  // Robot State variables
  logger().addLogEntry("G1RLQPController_kp_base", [this]() { return rlRuntime_.kpBase(); });
  logger().addLogEntry("G1RLQPController_kd_base", [this]() { return rlRuntime_.kdBase(); });
  logger().addLogEntry("G1RLQPController_kp_current", [this]() { return rlRuntime_.kp(); });
  logger().addLogEntry("G1RLQPController_kd_current", [this]() { return rlRuntime_.kd(); });
  logger().addLogEntry("G1RLQPController_pd_gains_ratio", [this]() { return rlRuntime_.pdGainsRatio(); });

  // RL variables
  logger().addLogEntry("G1RLQPController_RL_q", [this]() { return rlRuntime_.q_rl(); });
  logger().addLogEntry("G1RLQPController_RL_qZero", [this]() { return rlRuntime_.q_zero(); });
  logger().addLogEntry("G1RLQPController_RL_currentObservation", [this]() { return rlRuntime_.currentObservation(); });
  logger().addLogEntry("G1RLQPController_RL_currentAction", [this]() { return rlRuntime_.currentAction(); });
  logger().addLogEntry("G1RLQPController_RL_currentActionScaled", [this]() { return rlRuntime_.currentActionScaled(); });
  logger().addLogEntry("G1RLQPController_RL_actionScale", [this]() { return rlRuntime_.actionScale(); });
  logger().addLogEntry("G1RLQPController_RL_command", [this]() { return rlRuntime_.command(); });
  logger().addLogEntry("G1RLQPController_RL_effectiveCommand", [this]() { return rlRuntime_.effectiveCommand(); });

  // Controller state variables
  logger().addLogEntry("G1RLQPController_useQP", [this]() { return rlRuntime_.useQP(); });
  logger().addLogEntry("G1RLQPController_policyArmed", [this]() { return rlRuntime_.armed(); });

  // Fall switch: updated on each policy tick
  if(rlRuntime_.fallSwitch().configured())
  {
    const rlqp::FallSwitch & fallSwitch = rlRuntime_.fallSwitch();
    logger().addLogEntry("G1RLQPController_FallSwitch_p", [&fallSwitch]() { return fallSwitch.p(); });
    logger().addLogEntry("G1RLQPController_FallSwitch_threshold", [&fallSwitch]() { return fallSwitch.threshold(); });
    logger().addLogEntry("G1RLQPController_FallSwitch_inside", [&fallSwitch]() { return fallSwitch.inside(); });
    logger().addLogEntry("G1RLQPController_FallSwitch_mahalanobis", [&fallSwitch]() { return fallSwitch.mahalanobis(); });
    logger().addLogEntry("G1RLQPController_FallSwitch_mahalanobis2",
                         [&fallSwitch]() { return fallSwitch.mahalanobis2(); });
    logger().addLogEntry("G1RLQPController_FallSwitch_dwell", [&fallSwitch]() { return fallSwitch.dwell(); });
    logger().addLogEntry("G1RLQPController_FallSwitch_holdingRecovery",
                         [&fallSwitch]() { return fallSwitch.holdingRecovery(); });
    logger().addLogEntry("G1RLQPController_FallSwitch_operatorHold",
                         [&fallSwitch]() { return fallSwitch.operatorHold(); });
    logger().addLogEntry("G1RLQPController_FallSwitch_activePolicy",
                         [this]() { return rlRuntime_.activePolicyName(); });
    logger().addLogEntry("G1RLQPController_FallSwitch_features",
                         [&fallSwitch]() -> const Eigen::VectorXd & { return fallSwitch.features(); });
    logger().addLogEntry("G1RLQPController_FallSwitch_kernelFeatures",
                         [&fallSwitch]() -> const Eigen::VectorXd & { return fallSwitch.kernelFeatures(); });
  }

  // Log current policy (name and convention)
  logger().addLogEntry("G1RLQPController_currentPolicy", [this]() { return rlRuntime_.currentPolicyName(); });
  logger().addLogEntry("G1RLQPController_observationConvention", [this]() { return rlRuntime_.conventionName(); });
  logger().addLogEntry("G1RLQPController_RL_phase", [this]() { return rlRuntime_.phase(); });
  logger().addLogEntry("G1RLQPController_RL_policy_period_s", [this]() { return rlRuntime_.policyStepSize(); });
  logger().addLogEntry("G1RLQPController_RL_update_count", [this]() { return rlRuntime_.policyUpdateCount(); });
  // Observation entries are registered by RLPolicyRuntime on each policy load
}

void G1RLQPController::addGui()
{
  // At the top level: the first thing an operator sees
  gui()->addElement(
    {"G1RLQPController"},
    mc_rtc::gui::Label("Policy state", [this]() { return rlRuntime_.armed() ? "ARMED" : "held (disarmed)"; }),
    mc_rtc::gui::Button("ARM policy", [this]() { armPolicy(); }),
    mc_rtc::gui::Button("HOLD (disarm)", [this]() { holdPolicy(); }));

  if(rlRuntime_.fallSwitch().configured())
  {
    rlqp::FallSwitch & fallSwitch = rlRuntime_.fallSwitch();
    gui()->addElement(
      {"G1RLQPController", "Fall switch"},
      mc_rtc::gui::Checkbox(
        "Automatic switch", [&fallSwitch]() { return fallSwitch.enabled(); },
        [&fallSwitch]() { fallSwitch.setEnabled(!fallSwitch.enabled()); }),
      mc_rtc::gui::Button("Walking policy (cross)", [this]() { selectWalkingPolicy(); }),
      mc_rtc::gui::Button("Recovery policy (circle)", [this]() { selectRecoveryPolicy(); }),
      mc_rtc::gui::Label("Held by the operator", [&fallSwitch]() { return fallSwitch.operatorHold() ? "Yes" : "No"; }),
      mc_rtc::gui::Label("Active policy", [this]() { return rlRuntime_.activePolicyName(); }),
      mc_rtc::gui::Label("p", [&fallSwitch]() { return fallSwitch.p(); }),
      mc_rtc::gui::Label("p threshold", [&fallSwitch]() { return fallSwitch.threshold(); }),
      mc_rtc::gui::Label("p threshold of the bundle", [&fallSwitch]() { return fallSwitch.defaultThreshold(); }),
      mc_rtc::gui::Button("Lower the threshold (square)", [this]() { adjustFallSwitchThreshold(-1); }),
      mc_rtc::gui::Button("Raise the threshold (triangle)", [this]() { adjustFallSwitchThreshold(1); }),
      mc_rtc::gui::Label("Inside the limit kernel", [&fallSwitch]() { return fallSwitch.inside() ? "Yes" : "No"; }),
      mc_rtc::gui::Label("Mahalanobis squared", [&fallSwitch]() { return fallSwitch.mahalanobis2(); }),
      mc_rtc::gui::Label("Mahalanobis squared threshold", [&fallSwitch]() { return fallSwitch.kernelThreshold(); }),
      mc_rtc::gui::Label("Dwell", [&fallSwitch]() { return fallSwitch.dwell(); }),
      mc_rtc::gui::Label("Dwell required", [&fallSwitch]() { return fallSwitch.releaseDwellSteps(); }),
      mc_rtc::gui::Label("Switches since arming", [&fallSwitch]() { return fallSwitch.switchCount(); }),
      mc_rtc::gui::Label("Hand-backs since arming", [&fallSwitch]() { return fallSwitch.handBackCount(); }));
  }

  gui()->addElement(
    {"G1RLQPController", "Policy"},
    mc_rtc::gui::Label("Current policy", [this]() { return rlRuntime_.currentPolicyName(); }),
    mc_rtc::gui::Label("Current policy folder", [this]() { return rlRuntime_.currentPolicyFolder(); }),
    mc_rtc::gui::Label("Observation convention", [this]() { return rlRuntime_.conventionName(); }),
    mc_rtc::gui::Label("Observation source (default)", [this]() { return rlRuntime_.observationSource(); }),
    mc_rtc::gui::Label("Base body", [this]() { return rlRuntime_.baseBody(); }),
    mc_rtc::gui::Label("Observation size", [this]() { return rlRuntime_.observationSize(); }),
    mc_rtc::gui::Label("Action size", [this]() { return rlRuntime_.actionSize(); }),
    mc_rtc::gui::Label("Controlled action size", [this]() { return rlRuntime_.controlledActionSize(); }),
    mc_rtc::gui::ComboInput(
      "Select policy",
      rlRuntime_.availablePolicyNames(),
      [this]() { return rlRuntime_.currentPolicyName(); },
      [this](const std::string & policyName) {
        rlRuntime_.loadPolicyByName(policyName, *this, torqueJointTask);
      }),
    mc_rtc::gui::Button("Reload current policy", [this]() {
      rlRuntime_.reloadCurrentPolicy(*this, torqueJointTask);
    }));

  gui()->addElement(
    {"G1RLQPController", "PD Gains"},
    mc_rtc::gui::NumberSlider(
      "PD Gains Ratio",
      [this]() { return rlRuntime_.pdGainsRatio(); },
      [this](double v) { rlRuntime_.setPDGainsRatio(v, torqueJointTask); },
      0.0,
      2.0),
    mc_rtc::gui::Label("Current kp", [this]() { return rlRuntime_.kp(); }),
    mc_rtc::gui::Label("Current kd", [this]() { return rlRuntime_.kd(); }));

  gui()->addElement(
    {"G1RLQPController", "Control"},
    mc_rtc::gui::Button("Toggle QP Control", [this]() {
      rlRuntime_.setUseQP(!rlRuntime_.useQP());
    }),
    mc_rtc::gui::Label("QP Control", [this]() {
      return rlRuntime_.useQP() ? "Enforced" : "Bypassed";
    }),
    mc_rtc::gui::Button("Toggle print joint limits", [this]() {
      printLimits_ = !printLimits_;
    }),
    mc_rtc::gui::Label("Print joint limits", [this]() {
      return printLimits_ ? "Enabled" : "Disabled";
    }));

  gui()->addElement(
    {"G1RLQPController", "Runtime"},
    mc_rtc::gui::Label("Controller period [s]", [this]() { return timeStep; }),
    mc_rtc::gui::Label("Controller rate [Hz]", [this]() { return 1.0 / timeStep; }),
    mc_rtc::gui::Label("Policy period [s]", [this]() { return rlRuntime_.policyStepSize(); }),
    mc_rtc::gui::Label("Policy rate [Hz]", [this]() { return rlRuntime_.policyRate(); }),
    mc_rtc::gui::Label("Policy updates", [this]() { return rlRuntime_.policyUpdateCount(); }),
    mc_rtc::gui::Label("Phase", [this]() { return rlRuntime_.phase(); }));

  gui()->addElement(
    {"G1RLQPController", "Command"},
    mc_rtc::gui::Button(
        "Toggle Joystick Plugin",
        [this]()
        {
          toggleJoystick = !toggleJoystick;
          if(toggleJoystick)
            toggleKeyboard = false;
        }),
    mc_rtc::gui::Label(
        "Current velcity control mode",
        [this]()
        {
          if(toggleKeyboard)
            return std::string{"Keyboard"};
          if(toggleJoystick && controllerAvailable)
            return std::string{"mc_joystick_plugin"};
          return std::string{"GUI"};
        }),
    mc_rtc::gui::Button(
        "Toggle Keyboard",
        [this]()
        {
          toggleKeyboard = !toggleKeyboard;
          if(toggleKeyboard)
            toggleJoystick = false;
        }),
    mc_rtc::gui::Label(
        "Joystick plugin available",
        [this]() { return controllerAvailable ? "Yes" : "No"; }),
    mc_rtc::gui::Label("Command values", []() { return std::string(" "); }),
    mc_rtc::gui::NumberInput(
      "vx",
      [this]() { return rlRuntime_.command()(0); },
      [this](double v) { rlRuntime_.command()(0) = v; }),
    mc_rtc::gui::NumberInput(
      "vy",
      [this]() { return rlRuntime_.command()(1); },
      [this](double v) { rlRuntime_.command()(1) = v; }),
    mc_rtc::gui::NumberInput(
      "yaw_rate",
      [this]() { return rlRuntime_.command()(2); },
      [this](double v) { rlRuntime_.command()(2) = v; }),
    mc_rtc::gui::Label("Max values", []() { return std::string(" "); }),
    mc_rtc::gui::NumberInput(
        "max_vel_cmd",
        [this]() { return maxVelCmd; }, [this](double v) { maxVelCmd = v; }),
    mc_rtc::gui::NumberInput(
        "max_yaw_cmd",
        [this]() { return maxYawCmd; }, [this](double v) { maxYawCmd = v; }));
}

void G1RLQPController::computeLimits()
{
  const double epsilon = 1e-5;

  mc_rbdyn::Robot & real_robot = realRobot(robotName_);

  const std::vector<std::vector<double> > & currentPos = real_robot.q();
  const std::vector<std::vector<double> > & currentVel = real_robot.alpha();
  const std::vector<std::vector<double> > & currentTau = real_robot.jointTorque();

  const std::vector<std::vector<double> > & qLimLower = real_robot.ql();
  const std::vector<std::vector<double> > & qLimUpper = real_robot.qu();

  const std::vector<std::vector<double> > & qDotLimLower = real_robot.vl();
  const std::vector<std::vector<double> > & qDotLimUpper = real_robot.vu();

  const std::vector<std::vector<double> > & tauLimLower = real_robot.tl();
  const std::vector<std::vector<double> > & tauLimUpper = real_robot.tu();

  for(size_t j = 0; j < jointNames.size(); ++j)
  {
    const std::string & joint = jointNames[j];
    const int i = real_robot.jointIndexByName(joint);
    const size_t idx = static_cast<size_t>(i);

    const double ds = dsPercent_ * (qLimUpper[idx][0] - qLimLower[idx][0]);

    const double posLimitUp = qLimUpper[idx][0] - ds;
    const double posLimitLow = qLimLower[idx][0] + ds;

    const double velLimitUp = velPercent_ * qDotLimUpper[idx][0];
    const double velLimitLow = velPercent_ * qDotLimLower[idx][0];

    const double tauLimitUp = tauLimUpper[idx][0];
    const double tauLimitLow = tauLimLower[idx][0];

    if(currentPos[idx][0] > posLimitUp + epsilon)
    {
      mc_rtc::log::warning(
        "[G1RLQPController] Joint {} position upper limit breached: currentPos = {}, limit = {}",
        joint,
        currentPos[idx][0],
        posLimitUp);
    }

    if(currentPos[idx][0] < posLimitLow - epsilon)
    {
      mc_rtc::log::warning(
        "[G1RLQPController] Joint {} position lower limit breached: currentPos = {}, limit = {}",
        joint,
        currentPos[idx][0],
        posLimitLow);
    }

    if(currentVel[idx][0] > velLimitUp + epsilon)
    {
      mc_rtc::log::warning(
        "[G1RLQPController] Joint {} velocity upper limit breached: currentVel = {}, limit = {}",
        joint,
        currentVel[idx][0],
        velLimitUp);
    }

    if(currentVel[idx][0] < velLimitLow - epsilon)
    {
      mc_rtc::log::warning(
        "[G1RLQPController] Joint {} velocity lower limit breached: currentVel = {}, limit = {}",
        joint,
        currentVel[idx][0],
        velLimitLow);
    }

    if(currentTau[idx][0] > tauLimitUp + epsilon)
    {
      mc_rtc::log::warning(
        "[G1RLQPController] Joint {} torque upper limit breached: currentTau = {}, limit = {}",
        joint,
        currentTau[idx][0],
        tauLimitUp);
    }

    if(currentTau[idx][0] < tauLimitLow - epsilon)
    {
      mc_rtc::log::warning(
        "[G1RLQPController] Joint {} torque lower limit breached: currentTau = {}, limit = {}",
        joint,
        currentTau[idx][0],
        tauLimitLow);
    }
  }
}
