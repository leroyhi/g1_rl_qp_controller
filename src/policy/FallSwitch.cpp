#include "policy/FallSwitch.h"

#include <mc_rtc/logging.h>

#include <algorithm>
#include <array>
#include <cmath>

namespace rlqp
{

namespace
{

// PHYS_BLIND in mjlab's scripts/fall_switch/common.py, with the width of each group
const std::vector<std::string> FeatureGroups = {"tilt",     "feet",         "shoulders", "com_pos", "tilt_rate", "pelvis_h",
                                                "leg_ext",  "limit_margin", "track_err", "tau_sat", "support"};
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

  // Estimator and kernel read the same features, in the same order
  const std::vector<std::string> estimatorGroups = estimator("feature_groups");
  const std::vector<std::string> kernelGroups = kernel("feature_groups");
  if(estimatorGroups != FeatureGroups || kernelGroups != FeatureGroups)
    mc_rtc::log::error_and_throw("[FallSwitch] Unexpected feature groups.\n  estimator: {}\n  kernel: {}\n  implemented: {}",
                                 estimatorGroups, kernelGroups, FeatureGroups);
  window_ = estimator("window");
  const int featureCount = estimator("n_features");
  if(featureCount != FeatureCount)
    mc_rtc::log::error_and_throw("[FallSwitch] estimator.json has {} features, {} are implemented", featureCount,
                                 FeatureCount);

  mu_ = toVector(estimator("mu"));
  sd_ = toVector(estimator("sd"));
  kernelMu_ = toVector(kernel("mu"));
  const std::vector<std::vector<double>> invCov = kernel("inv_cov");
  if(mu_.size() != FeatureCount || sd_.size() != FeatureCount || kernelMu_.size() != FeatureCount
     || invCov.size() != static_cast<size_t>(FeatureCount))
    mc_rtc::log::error_and_throw("[FallSwitch] Sidecars do not hold {} features", FeatureCount);
  kernelInvCov_.resize(FeatureCount, FeatureCount);
  for(int i = 0; i < FeatureCount; ++i)
  {
    if(invCov[static_cast<size_t>(i)].size() != static_cast<size_t>(FeatureCount))
      mc_rtc::log::error_and_throw("[FallSwitch] limit_kernel.json: inv_cov is not square");
    kernelInvCov_.row(i) = toVector(invCov[static_cast<size_t>(i)]).transpose();
  }
  threshold_ = estimator("thr");
  kernelThreshold_ = kernel("threshold");

  const std::vector<int> horizons = estimator("horizons_steps");
  const int switchHorizon = estimator("switch_horizon_steps");
  const auto horizon = std::find(horizons.begin(), horizons.end(), switchHorizon);
  if(horizon == horizons.end())
    mc_rtc::log::error_and_throw("[FallSwitch] Switch horizon {} is not one of {}", switchHorizon, horizons);
  switchColumn_ = static_cast<int>(std::distance(horizons.begin(), horizon));

  // Masses and centres of mass of training, kinematics of mc_rtc
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
    bodies_.push_back(body);
    names.push_back(name);
    totalMass_ += body.mass;
  }
  bodyCom_.assign(bodies_.size(), Eigen::Vector3d::Zero());
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

  features_ = Eigen::VectorXd::Zero(FeatureCount);
  frames_.assign(static_cast<size_t>(window_ * FeatureCount), 0.0f);
  configured_ = true;
  reset();
  mc_rtc::log::success("[FallSwitch] Loaded: {} -> {} at p > {}, back after {} ticks inside the kernel ({}), {} kg in "
                       "{} bodies",
                       walkingPolicy_, recoveryPolicy_, threshold_, releaseDwellSteps_,
                       enabled_ ? "enabled" : "disabled", totalMass_, bodies_.size());
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

void FallSwitch::computeFeatures(const mc_rbdyn::Robot & realRobot, const Eigen::VectorXd & previousTarget)
{
  const sva::PTransformd X_0_p = realRobot.posW();
  const Eigen::Matrix3d E_0_p = X_0_p.rotation(); // sva convention: world axes to pelvis axes
  const Eigen::Matrix3d R_0_p = E_0_p.transpose();
  const Eigen::Vector3d & pelvis = X_0_p.translation();

  // Body centres of mass, not link frames: d.body_com_pos_w
  const auto & bodyPosW = realRobot.mbc().bodyPosW;
  Eigen::Vector3d weighted = Eigen::Vector3d::Zero();
  for(size_t i = 0; i < bodies_.size(); ++i)
  {
    const sva::PTransformd & X_0_b = bodyPosW[static_cast<size_t>(bodies_[i].index)];
    bodyCom_[i] = X_0_b.translation() + X_0_b.rotation().transpose() * bodies_[i].com;
    weighted += bodies_[i].mass * bodyCom_[i];
  }
  const Eigen::Vector3d comWorld = weighted / totalMass_;
  const Eigen::Vector3d & leftFoot = bodyCom_[leftFoot_];
  const Eigen::Vector3d & rightFoot = bodyCom_[rightFoot_];
  const double footZ = std::min(leftFoot.z(), rightFoot.z());

  int f = 0;

  // tilt: euler_xyz_from_quat of the root quaternion
  const Eigen::Quaterniond q(R_0_p);
  features_(f++) = std::atan2(2.0 * (q.w() * q.x() + q.y() * q.z()), 1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y()));
  const double sinPitch = 2.0 * (q.w() * q.y() - q.z() * q.x());
  features_(f++) = std::abs(sinPitch) >= 1.0 ? std::copysign(M_PI / 2.0, sinPitch) : std::asin(sinPitch);

  // feet: in the pelvis frame
  const Eigen::Vector3d leftFootPelvis = E_0_p * (leftFoot - pelvis);
  const Eigen::Vector3d rightFootPelvis = E_0_p * (rightFoot - pelvis);
  features_.segment<2>(f) = leftFootPelvis.head<2>();
  f += 2;
  features_.segment<2>(f) = rightFootPelvis.head<2>();
  f += 2;

  // shoulders: height above the lower foot
  features_(f++) = 0.5 * (bodyCom_[leftShoulder_].z() + bodyCom_[rightShoulder_].z()) - footZ;

  // com_pos: in the pelvis frame
  features_.segment<3>(f) = E_0_p * (comWorld - pelvis);
  f += 3;

  // tilt_rate: the gyro, in the pelvis frame
  features_.segment<2>(f) = realRobot.bodySensor(gyroSensor_).angularVelocity().head<2>();
  f += 2;

  // pelvis_h
  features_(f++) = pelvis.z() - footZ;

  // leg_ext
  features_(f++) = (bodyCom_[leftHip_] - leftFoot).norm();
  features_(f++) = (bodyCom_[rightHip_] - rightFoot).norm();

  const std::vector<double> & encoders = realRobot.encoderValues();
  const std::vector<double> & torques = realRobot.jointTorques();
  const size_t order = previousTarget.size() >= 0 ? static_cast<size_t>(previousTarget.size()) : 0;
  const auto jointPosition = [&](const LimbJoint & joint)
  {
    return encoders.size() == order ? encoders[static_cast<size_t>(joint.controllerIndex)]
                                    : realRobot.mbc().q[static_cast<size_t>(joint.mbcIndex)][0];
  };
  // limit_margin
  for(const LimbJoint & joint : limbJoints_)
  {
    const double position = jointPosition(joint);
    features_(f++) = std::min(position - joint.lower, joint.upper - position) / (joint.upper - joint.lower);
  }
  // track_err
  for(const LimbJoint & joint : limbJoints_)
  {
    features_(f++) = previousTarget(joint.controllerIndex) - jointPosition(joint);
  }
  // tau_sat
  for(const LimbJoint & joint : limbJoints_)
  {
    const double torque = torques.size() == order ? torques[static_cast<size_t>(joint.controllerIndex)] : 0.0;
    features_(f++) = std::min(std::abs(torque) / joint.effortLimit, TauSatClip);
  }

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
  features_.segment<2>(f) = comHeading - middle;
  f += 2;
  features_(f++) = boxMargin(comHeading, lo, hi);
  features_(f++) = (0.5 * (hi - lo)).norm();

  if(f != FeatureCount) mc_rtc::log::error_and_throw("[FallSwitch] Computed {} features instead of {}", f, FeatureCount);
}

void FallSwitch::pushFrame()
{
  std::array<float, FeatureCount> frame;
  for(int i = 0; i < FeatureCount; ++i)
  {
    const double z = (features_(i) - mu_(i)) / sd_(i);
    frame[static_cast<size_t>(i)] = static_cast<float>(std::clamp(z, -StandardisedClip, StandardisedClip));
  }
  if(windowEmpty_)
  {
    // Edge-pad with the first frame while the buffer fills
    for(int w = 0; w < window_; ++w) std::copy(frame.begin(), frame.end(), frames_.begin() + w * FeatureCount);
    windowEmpty_ = false;
    return;
  }
  std::copy(frames_.begin() + FeatureCount, frames_.end(), frames_.begin());
  std::copy(frame.begin(), frame.end(), frames_.end() - FeatureCount);
}

void FallSwitch::updateScores()
{
  const double logit = estimator_->run(frames_, window_, FeatureCount, switchColumn_);
  p_ = 1.0 / (1.0 + std::exp(-logit));
  // On the raw features, unlike the estimator
  const Eigen::VectorXd offset = features_ - kernelMu_;
  mahalanobis2_ = offset.dot(kernelInvCov_ * offset);
  inside_ = mahalanobis2_ <= kernelThreshold_;
}

void FallSwitch::step(const mc_rbdyn::Robot & realRobot, const Eigen::VectorXd & previousTarget)
{
  computeFeatures(realRobot, previousTarget);
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
