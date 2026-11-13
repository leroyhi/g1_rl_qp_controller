#include "policy/FallSwitch.h"

#include <mc_rtc/logging.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace rlqp
{

namespace
{

struct PhysicsGroup
{
  const char * name;
  int width;
  bool implemented;
};
// PHYS_GROUPS in mjlab's scripts/fall_switch/common.py, in its order and with its widths, so that a column of physics_
// is a column of mjlab's physics vector. com_vel and zmp are left out because nothing reads them, contact needs foot
// force sensing and dcm_margin and ang_mom_com are not used yet: their columns stay NaN.
const std::array<PhysicsGroup, 18> PhysicsGroups = {{{"tilt", 2, true},
                                                     {"feet", 4, true},
                                                     {"dcm", 3, true},
                                                     {"shoulders", 1, true},
                                                     {"ang_mom", 3, true},
                                                     {"com_pos", 3, true},
                                                     {"com_vel", 3, false},
                                                     {"zmp", 2, false},
                                                     {"tilt_rate", 2, true},
                                                     {"pelvis_h", 1, true},
                                                     {"leg_ext", 2, true},
                                                     {"limit_margin", 6, true},
                                                     {"track_err", 6, true},
                                                     {"tau_sat", 6, true},
                                                     {"support", 4, true},
                                                     {"contact", 3, false},
                                                     {"dcm_margin", 3, false},
                                                     {"ang_mom_com", 3, false}}};
const int PhysicsWidth = []()
{
  int width = 0;
  for(const PhysicsGroup & group : PhysicsGroups) width += group.width;
  return width;
}();

/** @return The first column of the group in physics_ */
int physicsOffset(const std::string & name)
{
  int offset = 0;
  for(const PhysicsGroup & group : PhysicsGroups)
  {
    if(name == group.name) return offset;
    offset += group.width;
  }
  mc_rtc::log::error_and_throw("[FallSwitch] Unknown feature group '{}'", name);
}

const double Gravity = 9.81;
// LIMB_JOINTS and FOOT_HALF in the same file
const std::array<const char *, 3> LimbJoints = {"hip_roll", "knee", "ankle_pitch"};
const Eigen::Vector2d FootHalf(0.10, 0.045);
const double TauSatClip = 4.0;
const double StandardisedClip = 8.0;

/** Signed distance from p to the boundary of the box [lo, hi]: positive inside, negative outside. */
double boxMargin(const Eigen::Vector2d & p, const Eigen::Vector2d & lo, const Eigen::Vector2d & hi)
{
  const Eigen::Vector2d center = 0.5 * (lo + hi);
  const Eigen::Vector2d half = 0.5 * (hi - lo);
  const Eigen::Vector2d q = (p - center).cwiseAbs() - half;
  const double outside = q.cwiseMax(0.0).norm();
  const double inside = std::min(q.maxCoeff(), 0.0);
  return -(outside + inside);
}

Eigen::VectorXd toVector(const std::vector<double> & values)
{
  return Eigen::Map<const Eigen::VectorXd>(values.data(), static_cast<Eigen::Index>(values.size()));
}

mc_rtc::Configuration loadFile(const std::string & path)
{
  mc_rtc::Configuration out;
  out.load(path);
  return out;
}

} // namespace

/** CPU session of the estimator. Its input has rank 3, which RLPolicyInterface does not handle. */
struct FallSwitch::Estimator
{
  explicit Estimator(const std::string & path)
  : env(ORT_LOGGING_LEVEL_WARNING, "FallSwitchEstimator"), session(nullptr),
    memory(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
  {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session = Ort::Session(env, path.c_str(), options);
    Ort::AllocatorWithDefaultOptions allocator;
    inputName = session.GetInputNameAllocated(0, allocator).get();
    outputName = session.GetOutputNameAllocated(0, allocator).get();
  }

  /** @return The logit of the given column */
  double run(std::vector<float> & frames, int window, int features, int column)
  {
    const std::array<int64_t, 3> shape = {1, window, features};
    Ort::Value input =
        Ort::Value::CreateTensor<float>(memory, frames.data(), frames.size(), shape.data(), shape.size());
    const char * inputNames[] = {inputName.c_str()};
    const char * outputNames[] = {outputName.c_str()};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, inputNames, &input, 1, outputNames, 1);
    return static_cast<double>(outputs[0].GetTensorData<float>()[column]);
  }

  Ort::Env env;
  Ort::Session session;
  Ort::MemoryInfo memory;
  std::string inputName;
  std::string outputName;
};

FallSwitch::FallSwitch() = default;
FallSwitch::~FallSwitch() = default;

void FallSwitch::configure(const mc_rtc::Configuration & config,
                           const mc_rbdyn::Robot & robot,
                           const std::vector<std::string> & controllerJointOrder)
{
  const std::string folder = config("folder");
  enabled_ = config("enabled", true);
  walkingPolicy_ = static_cast<std::string>(config("walking_policy"));
  recoveryPolicy_ = static_cast<std::string>(config("recovery_policy"));
  gyroSensor_ = static_cast<std::string>(config("gyro_sensor"));
  if(!robot.hasBodySensor(gyroSensor_))
    mc_rtc::log::error_and_throw("[FallSwitch] Robot has no body sensor named '{}'", gyroSensor_);

  // Not settled: the bundle's README says 25, its limit_kernel.json says 10. So it has no default here.
  releaseDwellSteps_ = config("release_dwell_steps");
  if(releaseDwellSteps_ <= 0)
    mc_rtc::log::error_and_throw("[FallSwitch] release_dwell_steps must be positive, got {}", releaseDwellSteps_);

  const mc_rtc::Configuration estimator = loadFile(folder + "/estimator.json");
  const mc_rtc::Configuration kernel = loadFile(folder + "/limit_kernel.json");
  const mc_rtc::Configuration plant = loadFile(folder + "/plant.json");
  const mc_rtc::Configuration massModel = loadFile(folder + "/training_mass_model.json");

  // The estimator and the kernel each read their own groups, in the order of their sidecar
  const std::vector<std::string> estimatorGroups = estimator("feature_groups");
  const std::vector<std::string> kernelGroups = kernel("feature_groups");
  estimatorColumns_ = columnsOf(estimatorGroups, "estimator.json");
  kernelColumns_ = columnsOf(kernelGroups, "limit_kernel.json");
  if(kernel.has("columns_into_phys_vector"))
  {
    const std::vector<int> expected = kernel("columns_into_phys_vector");
    if(expected != kernelColumns_)
      mc_rtc::log::error_and_throw("[FallSwitch] limit_kernel.json: columns_into_phys_vector {} do not match its groups "
                                   "{}, which give {}",
                                   expected, kernelGroups, kernelColumns_);
  }
  window_ = estimator("window");
  featureCount_ = static_cast<int>(estimatorColumns_.size());
  const int kernelCount = static_cast<int>(kernelColumns_.size());
  const int featureCount = estimator("n_features");
  if(featureCount != featureCount_)
    mc_rtc::log::error_and_throw("[FallSwitch] estimator.json has {} features, its groups {} give {}", featureCount,
                                 estimatorGroups, featureCount_);

  mu_ = toVector(estimator("mu"));
  sd_ = toVector(estimator("sd"));
  kernelMu_ = toVector(kernel("mu"));
  const std::vector<std::vector<double>> invCov = kernel("inv_cov");
  if(mu_.size() != featureCount_ || sd_.size() != featureCount_)
    mc_rtc::log::error_and_throw("[FallSwitch] estimator.json: mu and sd do not hold {} features", featureCount_);
  if(kernelMu_.size() != kernelCount || invCov.size() != static_cast<size_t>(kernelCount))
    mc_rtc::log::error_and_throw("[FallSwitch] limit_kernel.json: mu and inv_cov do not hold {} features", kernelCount);
  kernelInvCov_.resize(kernelCount, kernelCount);
  for(int i = 0; i < kernelCount; ++i)
  {
    if(invCov[static_cast<size_t>(i)].size() != static_cast<size_t>(kernelCount))
      mc_rtc::log::error_and_throw("[FallSwitch] limit_kernel.json: inv_cov is not square");
    kernelInvCov_.row(i) = toVector(invCov[static_cast<size_t>(i)]).transpose();
  }
  threshold_ = estimator("thr");
  defaultThreshold_ = threshold_;
  if(threshold_ < ThresholdMin || threshold_ > ThresholdMax)
    mc_rtc::log::error_and_throw("[FallSwitch] estimator.json: thr {} is outside [{}, {}]", threshold_, ThresholdMin,
                                 ThresholdMax);
  kernelThreshold_ = kernel("threshold");

  const std::vector<int> horizons = estimator("horizons_steps");
  const int switchHorizon = estimator("switch_horizon_steps");
  const auto horizon = std::find(horizons.begin(), horizons.end(), switchHorizon);
  if(horizon == horizons.end())
    mc_rtc::log::error_and_throw("[FallSwitch] Switch horizon {} is not one of {}", switchHorizon, horizons);
  switchColumn_ = static_cast<int>(std::distance(horizons.begin(), horizon));

  // Masses, centres of mass and inertias of training, kinematics of mc_rtc
  bodies_.clear();
  totalMass_ = 0.0;
  const mc_rtc::Configuration bodies = massModel("bodies");
  std::vector<std::string> names;
  for(size_t i = 0; i < bodies.size(); ++i)
  {
    const mc_rtc::Configuration entry = bodies[i];
    const std::string name = entry("name");
    if(!robot.hasBody(name))
      mc_rtc::log::error_and_throw("[FallSwitch] training_mass_model.json names body '{}', which the robot does not have",
                                   name);
    Body body;
    body.index = static_cast<int>(robot.bodyIndexByName(name));
    body.mass = entry("mass");
    body.com = entry("com");
    if(!entry.has("inertia"))
      mc_rtc::log::error_and_throw("[FallSwitch] training_mass_model.json: body '{}' has no inertia, regenerate it with "
                                   "scripts/gen_training_mass_model.py",
                                   name);
    const std::vector<std::vector<double>> inertia = entry("inertia");
    if(inertia.size() != 3)
      mc_rtc::log::error_and_throw("[FallSwitch] training_mass_model.json: inertia of '{}' is not 3x3", name);
    for(int r = 0; r < 3; ++r)
    {
      if(inertia[static_cast<size_t>(r)].size() != 3)
        mc_rtc::log::error_and_throw("[FallSwitch] training_mass_model.json: inertia of '{}' is not 3x3", name);
      for(int c = 0; c < 3; ++c) body.inertia(r, c) = inertia[static_cast<size_t>(r)][static_cast<size_t>(c)];
    }
    bodies_.push_back(body);
    names.push_back(name);
    totalMass_ += body.mass;
  }
  bodyCom_.assign(bodies_.size(), Eigen::Vector3d::Zero());
  bodyComVel_.assign(bodies_.size(), Eigen::Vector3d::Zero());
  const auto bodyPosition = [&names](const std::string & name)
  {
    const auto it = std::find(names.begin(), names.end(), name);
    if(it == names.end()) mc_rtc::log::error_and_throw("[FallSwitch] training_mass_model.json has no body '{}'", name);
    return static_cast<size_t>(std::distance(names.begin(), it));
  };
  leftFoot_ = bodyPosition("left_ankle_roll_link");
  rightFoot_ = bodyPosition("right_ankle_roll_link");
  leftShoulder_ = bodyPosition("left_shoulder_pitch_link");
  rightShoulder_ = bodyPosition("right_shoulder_pitch_link");
  leftHip_ = bodyPosition("left_hip_roll_link");
  rightHip_ = bodyPosition("right_hip_roll_link");

  // Limits of training, not the ones the controller enforces: the features must keep their training distribution
  limbJoints_.clear();
  const mc_rtc::Configuration plantJoints = plant("joints");
  for(const std::string side : {"left", "right"})
  {
    for(const char * limb : LimbJoints)
    {
      const std::string name = side + "_" + limb + "_joint";
      const auto order = std::find(controllerJointOrder.begin(), controllerJointOrder.end(), name);
      if(order == controllerJointOrder.end())
        mc_rtc::log::error_and_throw("[FallSwitch] Joint '{}' is not a controller joint", name);
      LimbJoint joint;
      joint.controllerIndex = static_cast<int>(std::distance(controllerJointOrder.begin(), order));
      joint.mbcIndex = static_cast<int>(robot.jointIndexByName(name));
      bool found = false;
      for(size_t i = 0; i < plantJoints.size() && !found; ++i)
      {
        const mc_rtc::Configuration entry = plantJoints[i];
        if(entry("name").operator std::string() != name) continue;
        const std::vector<double> hard = entry("pos_limit_hard");
        joint.lower = hard[0];
        joint.upper = hard[1];
        joint.effortLimit = entry("torque_limit");
        found = true;
      }
      if(!found) mc_rtc::log::error_and_throw("[FallSwitch] plant.json has no joint '{}'", name);
      limbJoints_.push_back(joint);
    }
  }

  estimator_ = std::make_unique<Estimator>(folder + "/estimator.onnx");
  recovery_ = std::make_unique<RLPolicyInterface>(static_cast<std::string>(config("recovery_onnx")));
  if(!recovery_->isLoaded()) mc_rtc::log::error_and_throw("[FallSwitch] Could not load the recovery policy");

  physics_ = Eigen::VectorXd::Constant(PhysicsWidth, std::numeric_limits<double>::quiet_NaN());
  features_ = Eigen::VectorXd::Zero(featureCount_);
  kernelFeatures_ = Eigen::VectorXd::Zero(kernelCount);
  frames_.assign(static_cast<size_t>(window_ * featureCount_), 0.0f);
  configured_ = true;
  reset();
  mc_rtc::log::success("[FallSwitch] Loaded: {} -> {} at p > {}, back after {} ticks inside the kernel ({}), {} kg in "
                       "{} bodies.\n  estimator: {} ({} features)\n  kernel: {} ({} features)",
                       walkingPolicy_, recoveryPolicy_, threshold_, releaseDwellSteps_,
                       enabled_ ? "enabled" : "disabled", totalMass_, bodies_.size(), estimatorGroups, featureCount_,
                       kernelGroups, kernelCount);
  const std::vector<std::string> needVelocity = {"dcm", "ang_mom"};
  for(const std::string & group : needVelocity)
  {
    if(std::find(estimatorGroups.begin(), estimatorGroups.end(), group) != estimatorGroups.end()
       || std::find(kernelGroups.begin(), kernelGroups.end(), group) != kernelGroups.end())
    {
      mc_rtc::log::warning("[FallSwitch] {} reads the base linear velocity of the real robot: it must be estimated on "
                           "hardware, an IMU alone gives zero",
                           group);
    }
  }
}

void FallSwitch::reset()
{
  windowEmpty_ = true;
  holding_ = false;
  operatorHold_ = false;
  inside_ = false;
  p_ = 0.0;
  mahalanobis2_ = 0.0;
  dwell_ = 0;
  switchCount_ = 0;
  handBackCount_ = 0;
}

void FallSwitch::setEnabled(bool enabled)
{
  enabled_ = enabled;
  // An automatic switch has nothing left to end it. The operator's choice stands.
  if(!enabled_ && holding_ && !operatorHold_)
  {
    holding_ = false;
    dwell_ = 0;
  }
}

void FallSwitch::forceRecovery()
{
  holding_ = true;
  operatorHold_ = true;
  dwell_ = 0;
}

void FallSwitch::forceWalking()
{
  holding_ = false;
  operatorHold_ = false;
  dwell_ = 0;
}

double FallSwitch::mahalanobis() const
{
  return std::sqrt(std::max(mahalanobis2_, 0.0));
}

double FallSwitch::adjustThreshold(double delta)
{
  threshold_ = std::clamp(threshold_ + delta, ThresholdMin, ThresholdMax);
  return threshold_;
}

std::vector<int> FallSwitch::columnsOf(const std::vector<std::string> & groups, const std::string & sidecar)
{
  std::vector<int> columns;
  for(const std::string & name : groups)
  {
    const auto group = std::find_if(PhysicsGroups.begin(), PhysicsGroups.end(),
                                    [&name](const PhysicsGroup & g) { return name == g.name; });
    if(group == PhysicsGroups.end() || !group->implemented)
      mc_rtc::log::error_and_throw("[FallSwitch] {} reads the feature group '{}', which is not implemented", sidecar,
                                   name);
    const int offset = physicsOffset(name);
    for(int i = 0; i < group->width; ++i) columns.push_back(offset + i);
  }
  return columns;
}

void FallSwitch::computePhysics(const mc_rbdyn::Robot & realRobot, const Eigen::VectorXd & previousTarget)
{
  const sva::PTransformd X_0_p = realRobot.posW();
  const Eigen::Matrix3d E_0_p = X_0_p.rotation(); // sva convention: world axes to pelvis axes
  const Eigen::Matrix3d R_0_p = E_0_p.transpose();
  const Eigen::Vector3d & pelvis = X_0_p.translation();

  // Body centres of mass, not link frames: d.body_com_pos_w and d.body_com_lin_vel_w. bodyVelW is the velocity of the
  // body origin, in world axes.
  const auto & bodyPosW = realRobot.mbc().bodyPosW;
  const auto & bodyVelW = realRobot.mbc().bodyVelW;
  Eigen::Vector3d weighted = Eigen::Vector3d::Zero();
  Eigen::Vector3d momentum = Eigen::Vector3d::Zero();
  Eigen::Vector3d spin = Eigen::Vector3d::Zero();
  for(size_t i = 0; i < bodies_.size(); ++i)
  {
    const size_t index = static_cast<size_t>(bodies_[i].index);
    const Eigen::Matrix3d R_0_b = bodyPosW[index].rotation().transpose();
    const Eigen::Vector3d lever = R_0_b * bodies_[i].com;
    const sva::MotionVecd & velocity = bodyVelW[index];
    bodyCom_[i] = bodyPosW[index].translation() + lever;
    bodyComVel_[i] = velocity.linear() + velocity.angular().cross(lever);
    weighted += bodies_[i].mass * bodyCom_[i];
    momentum += bodies_[i].mass * bodyComVel_[i];
    spin += R_0_b * bodies_[i].inertia * R_0_b.transpose() * velocity.angular();
  }
  const Eigen::Vector3d comWorld = weighted / totalMass_;
  const Eigen::Vector3d comVelWorld = momentum / totalMass_;
  const Eigen::Vector3d & leftFoot = bodyCom_[leftFoot_];
  const Eigen::Vector3d & rightFoot = bodyCom_[rightFoot_];
  const double footZ = std::min(leftFoot.z(), rightFoot.z());
  const Eigen::Vector3d comPelvis = E_0_p * (comWorld - pelvis);

  const auto put = [this](const char * group, const auto & values)
  { physics_.segment(physicsOffset(group), values.size()) = values; };

  // tilt: euler_xyz_from_quat of the root quaternion
  const Eigen::Quaterniond q(R_0_p);
  const double sinPitch = 2.0 * (q.w() * q.y() - q.z() * q.x());
  put("tilt", Eigen::Vector2d(
                  std::atan2(2.0 * (q.w() * q.x() + q.y() * q.z()), 1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y())),
                  std::abs(sinPitch) >= 1.0 ? std::copysign(M_PI / 2.0, sinPitch) : std::asin(sinPitch)));

  // feet: in the pelvis frame
  const Eigen::Vector3d leftFootPelvis = E_0_p * (leftFoot - pelvis);
  const Eigen::Vector3d rightFootPelvis = E_0_p * (rightFoot - pelvis);
  put("feet", Eigen::Vector4d(leftFootPelvis.x(), leftFootPelvis.y(), rightFootPelvis.x(), rightFootPelvis.y()));

  // dcm: com + com_v * sqrt(h / g), in the pelvis frame, h the height of the CoM above the lower foot
  const double height = std::max(comWorld.z() - footZ, 0.05);
  put("dcm", Eigen::Vector3d(comPelvis + E_0_p * comVelWorld * std::sqrt(height / Gravity)));

  // shoulders: height above the lower foot
  put("shoulders", Eigen::Matrix<double, 1, 1>(0.5 * (bodyCom_[leftShoulder_].z() + bodyCom_[rightShoulder_].z())
                                               - footZ));

  // ang_mom: sum_i I_i w_i + m_i r_i x v_i about the pelvis, in the pelvis frame
  Eigen::Vector3d angularMomentum = spin;
  for(size_t i = 0; i < bodies_.size(); ++i)
    angularMomentum += bodies_[i].mass * (bodyCom_[i] - pelvis).cross(bodyComVel_[i]);
  put("ang_mom", Eigen::Vector3d(E_0_p * angularMomentum));

  // com_pos: in the pelvis frame
  put("com_pos", comPelvis);

  // tilt_rate: the gyro, in the pelvis frame
  put("tilt_rate", Eigen::Vector2d(realRobot.bodySensor(gyroSensor_).angularVelocity().head<2>()));

  // pelvis_h
  put("pelvis_h", Eigen::Matrix<double, 1, 1>(pelvis.z() - footZ));

  // leg_ext
  put("leg_ext", Eigen::Vector2d((bodyCom_[leftHip_] - leftFoot).norm(), (bodyCom_[rightHip_] - rightFoot).norm()));

  const std::vector<double> & encoders = realRobot.encoderValues();
  const std::vector<double> & torques = realRobot.jointTorques();
  const size_t order = previousTarget.size() >= 0 ? static_cast<size_t>(previousTarget.size()) : 0;
  const auto jointPosition = [&](const LimbJoint & joint)
  {
    return encoders.size() == order ? encoders[static_cast<size_t>(joint.controllerIndex)]
                                    : realRobot.mbc().q[static_cast<size_t>(joint.mbcIndex)][0];
  };
  Eigen::Matrix<double, 6, 1> limitMargin, trackErr, tauSat;
  for(size_t i = 0; i < limbJoints_.size(); ++i)
  {
    const LimbJoint & joint = limbJoints_[i];
    const double position = jointPosition(joint);
    const double torque = torques.size() == order ? torques[static_cast<size_t>(joint.controllerIndex)] : 0.0;
    limitMargin(i) = std::min(position - joint.lower, joint.upper - position) / (joint.upper - joint.lower);
    trackErr(i) = previousTarget(joint.controllerIndex) - position;
    tauSat(i) = std::min(std::abs(torque) / joint.effortLimit, TauSatClip);
  }
  put("limit_margin", limitMargin);
  put("track_err", trackErr);
  put("tau_sat", tauSat);

  // support: in the heading frame, which only takes the yaw of the forward axis, not the pelvis frame
  const Eigen::Vector3d forward = R_0_p.col(0);
  const double heading = std::atan2(forward.y(), forward.x());
  const double cy = std::cos(heading);
  const double sy = std::sin(heading);
  const auto toHeading = [cy, sy](const Eigen::Vector3d & v)
  { return Eigen::Vector2d(cy * v.x() + sy * v.y(), -sy * v.x() + cy * v.y()); };
  const Eigen::Vector2d leftHeading = toHeading(leftFoot - pelvis);
  const Eigen::Vector2d rightHeading = toHeading(rightFoot - pelvis);
  const Eigen::Vector2d lo = leftHeading.cwiseMin(rightHeading) - FootHalf;
  const Eigen::Vector2d hi = leftHeading.cwiseMax(rightHeading) + FootHalf;
  const Eigen::Vector2d middle = 0.5 * (leftHeading + rightHeading);
  const Eigen::Vector2d comHeading = toHeading(comWorld - pelvis);
  const Eigen::Vector2d offset = comHeading - middle;
  put("support", Eigen::Vector4d(offset.x(), offset.y(), boxMargin(comHeading, lo, hi), (0.5 * (hi - lo)).norm()));

  for(size_t i = 0; i < estimatorColumns_.size(); ++i)
    features_(static_cast<Eigen::Index>(i)) = physics_(estimatorColumns_[i]);
  for(size_t i = 0; i < kernelColumns_.size(); ++i)
    kernelFeatures_(static_cast<Eigen::Index>(i)) = physics_(kernelColumns_[i]);
}

void FallSwitch::pushFrame()
{
  const size_t width = static_cast<size_t>(featureCount_);
  const auto newest = frames_.end() - static_cast<std::ptrdiff_t>(width);
  if(!windowEmpty_) std::copy(frames_.begin() + static_cast<std::ptrdiff_t>(width), frames_.end(), frames_.begin());
  for(int i = 0; i < featureCount_; ++i)
  {
    const double z = (features_(i) - mu_(i)) / sd_(i);
    *(newest + i) = static_cast<float>(std::clamp(z, -StandardisedClip, StandardisedClip));
  }
  if(windowEmpty_)
  {
    // Edge-pad with the first frame while the buffer fills
    for(int w = 0; w + 1 < window_; ++w)
      std::copy(newest, frames_.end(), frames_.begin() + static_cast<std::ptrdiff_t>(w) * featureCount_);
    windowEmpty_ = false;
  }
}

void FallSwitch::updateScores()
{
  const double logit = estimator_->run(frames_, window_, featureCount_, switchColumn_);
  p_ = 1.0 / (1.0 + std::exp(-logit));
  // On the raw features, unlike the estimator
  const Eigen::VectorXd offset = kernelFeatures_ - kernelMu_;
  mahalanobis2_ = offset.dot(kernelInvCov_ * offset);
  inside_ = mahalanobis2_ <= kernelThreshold_;
}

void FallSwitch::step(const mc_rbdyn::Robot & realRobot, const Eigen::VectorXd & previousTarget)
{
  computePhysics(realRobot, previousTarget);
  pushFrame();
  updateScores();

  if(holding_)
  {
    dwell_ = inside_ ? dwell_ + 1 : 0;
    if(!operatorHold_ && dwell_ >= releaseDwellSteps_)
    {
      holding_ = false;
      dwell_ = 0;
      ++handBackCount_;
    }
  }
  else if(enabled_ && p_ > threshold_)
  {
    holding_ = true;
    dwell_ = 0;
    ++switchCount_;
  }
}

} // namespace rlqp
