/**
 * Run the controller in mc_mujoco without a window, following a scripted scenario.
 *
 * Events are triggered on the simulation clock, so a scenario does not depend on how fast the machine runs:
 *
 *   duration: 60.0
 *   fall_height: 0.4        # optional: a pelvis below this height ends the run
 *   fall_tilt_deg: 80.0     # optional: so does a pelvis tilted beyond this angle
 *   start_armed: true       # optional, true by default: walking policy armed after the first step, else disarmed
 *   events:
 *     - {t: 5.0, command: [0.3, 0.0, 0.0]}
 *     - {t: 20.0, push: {body: torso_link, force: [150.0, 0.0, 0.0], duration: 0.1}}
 *     - {t: 25.0, velocity_push: [0.5, 0.0]}   # added to the world-frame base velocity, as mjlab's push()
 *     - {t: 30.0, use_qp: false}
 *     - {t: 35.0, arm: false}
 *     - {t: 40.0, fall_switch: false}
 *     - {t: 45.0, policy: recovery}             # or walking: what the gamepad's circle and cross do
 *
 * With fake_joystick: true, the run stands in for mc_joystick_plugin, which the scenario configuration does not load:
 *
 *     - {t: 3.0, pad: {left: [1.0, 0.5], right: [0.5, 0.5], cross: true}}
 *
 * Usage: g1_sim_scenario <mc_rtc configuration> <scenario>
 *
 * mc_mujoco's --without-visualization cannot replace this tool: mj_sim.updateScene() needs the window's camera.
 *
 * The last line of the output starts with [result] and holds the outcome as JSON.
 * Exit status: 0 if the run reached its duration, 1 if the robot fell, 2 if the controller failed.
 */

#include "G1RLQPController.h"

#include <mc_joystick_plugin/joystick_inputs.h>
#include <mc_mujoco/mj_sim.h>
#include <mc_rtc/logging.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace
{

struct Event
{
  double t = 0.0;
  mc_rtc::Configuration config;
};

/** What mc_joystick_plugin publishes: sticks at (0, 0) until the device's first events, as the real plugin does. */
struct FakeJoystick
{
  Eigen::Vector2d left = Eigen::Vector2d::Zero();
  Eigen::Vector2d right = Eigen::Vector2d::Zero();
  bool cross = false;
  bool circle = false;
  // As mc_joystick_plugin reports a PlayStation pad under Linux: triangle is X and square is Y
  bool triangle = false;
  bool square = false;

  void publish(mc_control::fsm::Controller & ctl)
  {
    auto & datastore = ctl.datastore();
    datastore.make<bool>("Joystick::connected", true);
    datastore.make_call("Joystick::Stick", [this](joystickAnalogicInputs stick) -> Eigen::Vector2d
                        { return stick == joystickAnalogicInputs::L_STICK ? left : right; });
    datastore.make_call("Joystick::Button",
                        [this](joystickButtonInputs button) -> bool
                        {
                          switch(button)
                          {
                            case joystickButtonInputs::A:
                              return cross;
                            case joystickButtonInputs::B:
                              return circle;
                            case joystickButtonInputs::X:
                              return triangle;
                            case joystickButtonInputs::Y:
                              return square;
                            default:
                              return false;
                          }
                        });
  }
};

std::vector<Event> loadEvents(const mc_rtc::Configuration & scenario)
{
  std::vector<Event> events;
  if(!scenario.has("events")) return events;
  const mc_rtc::Configuration list = scenario("events");
  for(size_t i = 0; i < list.size(); ++i)
  {
    Event event;
    event.config = list[i];
    event.t = event.config("t");
    events.push_back(event);
  }
  std::stable_sort(events.begin(), events.end(), [](const Event & a, const Event & b) { return a.t < b.t; });
  return events;
}

void apply(const Event & event,
           G1RLQPController & ctl,
           mc_mujoco::MjSim & sim,
           FakeJoystick & joystick,
           double & pushTime,
           double now)
{
  const mc_rtc::Configuration & config = event.config;
  if(config.has("pad"))
  {
    const mc_rtc::Configuration pad = config("pad");
    pad("left", joystick.left);
    pad("right", joystick.right);
    pad("cross", joystick.cross);
    pad("circle", joystick.circle);
    pad("triangle", joystick.triangle);
    pad("square", joystick.square);
    mc_rtc::log::info("[scenario] t={:.3f} pad: left {}, right {}, cross {}, circle {}, triangle {}, square {}", now,
                      joystick.left.transpose(), joystick.right.transpose(), joystick.cross, joystick.circle,
                      joystick.triangle, joystick.square);
  }
  if(config.has("policy"))
  {
    const std::string policy = config("policy");
    if(policy == "recovery")
      ctl.selectRecoveryPolicy();
    else
      ctl.selectWalkingPolicy();
  }
  if(config.has("arm"))
  {
    if(config("arm").operator bool())
      ctl.armPolicy();
    else
      ctl.holdPolicy();
  }
  if(config.has("fall_switch"))
  {
    const bool enabled = config("fall_switch");
    mc_rtc::log::info("[scenario] t={:.3f} fall switch {}", now, enabled);
    ctl.rlRuntime().fallSwitch().setEnabled(enabled);
  }
  if(config.has("velocity_push"))
  {
    const std::vector<double> push = config("velocity_push");
    // The free joint of the robot comes first among its joints; ground has none
    const std::string joint = ctl.robot().name() + "_floating_base_joint";
    const int id = mj_name2id(&sim.model(), mjOBJ_JOINT, joint.c_str());
    if(id < 0) mc_rtc::log::error_and_throw("[scenario] No joint named {}", joint);
    const int dof = sim.model().jnt_dofadr[id];
    sim.data().qvel[dof] += push[0];
    sim.data().qvel[dof + 1] += push[1];
    pushTime = now;
    mc_rtc::log::info("[scenario] t={:.3f} velocity push ({}, {}) m/s", now, push[0], push[1]);
  }
  if(config.has("command"))
  {
    const Eigen::Vector3d command = config("command");
    mc_rtc::log::info("[scenario] t={:.3f} command ({}, {}, {})", now, command.x(), command.y(), command.z());
    ctl.rlRuntime().setCommand(command);
  }
  if(config.has("use_qp"))
  {
    const bool useQP = config("use_qp");
    mc_rtc::log::info("[scenario] t={:.3f} use_qp {}", now, useQP);
    ctl.activateQPControl(useQP);
  }
  if(config.has("push"))
  {
    const mc_rtc::Configuration push = config("push");
    const std::string body = push("body");
    const Eigen::Vector3d force = push("force");
    const double duration = push("duration");
    ctl.datastore().call<void, const std::string &, const Eigen::Vector3d &, double>(ctl.robot().name() + "::ApplyForce",
                                                                                    body, force, duration);
  }
}

} // namespace

int main(int argc, char * argv[])
{
  if(argc != 3)
  {
    mc_rtc::log::error("Usage: {} <mc_rtc configuration> <scenario>", argv[0]);
    return 3;
  }

  mc_rtc::Configuration scenario;
  scenario.load(std::string(argv[2]));
  const double duration = scenario("duration");
  const double fallHeight = scenario("fall_height", 0.0);
  const double fallTilt = scenario("fall_tilt_deg", 180.0) * M_PI / 180.0;
  const bool startArmed = scenario("start_armed", true);
  const std::vector<Event> events = loadEvents(scenario);

  mc_mujoco::MjConfiguration config;
  config.mc_config = argv[1];
  config.with_visualization = false;
  config.with_mc_rtc_gui = false;
  config.torque_control = true;
  mc_mujoco::MjSim sim(config);

  auto & ctl = dynamic_cast<G1RLQPController &>(sim.controller()->controller());
  FakeJoystick joystick;
  if(scenario("fake_joystick", false)) joystick.publish(ctl);

  int status = 0;
  size_t next = 0;
  bool firstStep = true;
  double lowestPelvis = ctl.realRobot().posW().translation().z();
  double pushTime = -1.0;
  double fallTime = -1.0;
  double holdingTime = 0.0;
  double firstSwitch = -1.0;
  size_t switchesBeforePush = 0;
  size_t previousSwitches = 0;
  const rlqp::FallSwitch & fallSwitch = ctl.rlRuntime().fallSwitch();
  bool firstEventStep = true;
  while(sim.data().time < duration)
  {
    const double now = sim.data().time;
    // After the first step: the FSM state starts on the recovery policy, held by the operator. Scenarios start on the
    // walking policy, or disarmed holding their posture with start_armed: false.
    if(!firstStep)
    {
      if(firstEventStep)
      {
        firstEventStep = false;
        if(startArmed) ctl.selectWalkingPolicy();
        else
        {
          ctl.holdPolicy();
          ctl.rlRuntime().holdCurrentPosture(ctl);
        }
      }
      while(next < events.size() && events[next].t <= now) apply(events[next++], ctl, sim, joystick, pushTime, now);
    }
    firstStep = false;

    if(sim.stepSimulation())
    {
      status = 2;
      break;
    }

    if(fallSwitch.switchCount() > previousSwitches)
    {
      if(firstSwitch < 0.0) firstSwitch = now;
      if(pushTime < 0.0) switchesBeforePush += fallSwitch.switchCount() - previousSwitches;
      previousSwitches = fallSwitch.switchCount();
    }
    if(ctl.rlRuntime().activePolicyName() == fallSwitch.recoveryPolicy()) holdingTime += sim.model().opt.timestep;

    const sva::PTransformd pelvisPose = ctl.realRobot().posW();
    const double pelvis = pelvisPose.translation().z();
    const double tilt = std::acos(std::clamp(pelvisPose.rotation()(2, 2), -1.0, 1.0));
    lowestPelvis = std::min(lowestPelvis, pelvis);
    if(pelvis < fallHeight || tilt > fallTilt)
    {
      fallTime = now;
      status = 1;
      break;
    }
  }

  const char * outcome = status == 0 ? "reached its duration" : status == 1 ? "robot fell" : "controller failed";
  std::printf("[scenario] end t=%.3f: %s, lowest pelvis %.3f m\n", sim.data().time, outcome, lowestPelvis);
  std::printf("[result] {\"status\": %d, \"end\": %.3f, \"fall_time\": %.3f, \"push_time\": %.3f, \"switches\": %zu, "
              "\"switches_before_push\": %zu, \"first_switch\": %.3f, \"hand_backs\": %zu, \"holding_time\": %.3f, "
              "\"lowest_pelvis\": %.3f, \"pelvis_at_end\": %.3f}\n",
              status, sim.data().time, fallTime, pushTime, fallSwitch.switchCount(), switchesBeforePush, firstSwitch,
              fallSwitch.handBackCount(), holdingTime, lowestPelvis, ctl.realRobot().posW().translation().z());
  sim.stopSimulation();
  // Skip the static destructors: the ROS plugin of mc_rtc aborts while it is torn down. _Exit does not flush.
  std::fflush(nullptr);
  std::_Exit(status);
}
