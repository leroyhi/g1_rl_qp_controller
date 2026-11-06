/**
 * Run the controller in mc_mujoco without a window, following a scripted scenario.
 *
 * Events are triggered on the simulation clock, so a scenario does not depend on how fast the machine runs:
 *
 *   duration: 60.0
 *   fall_height: 0.4        # optional: a pelvis below this height ends the run
 *   events:
 *     - {t: 5.0, command: [0.3, 0.0, 0.0]}
 *     - {t: 20.0, push: {body: torso_link, force: [150.0, 0.0, 0.0], duration: 0.1}}
 *     - {t: 30.0, use_qp: false}
 *
 * Usage: g1_sim_scenario <mc_rtc configuration> <scenario>
 *
 * mc_mujoco's --without-visualization cannot replace this tool: mj_sim.updateScene() needs the window's camera.
 *
 * Exit status: 0 if the run reached its duration, 1 if the robot fell, 2 if the controller failed.
 */

#include "G1RLQPController.h"

#include <mc_mujoco/mj_sim.h>
#include <mc_rtc/logging.h>

#include <algorithm>
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

void apply(const Event & event, G1RLQPController & ctl, double now)
{
  const mc_rtc::Configuration & config = event.config;
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
  const std::vector<Event> events = loadEvents(scenario);

  mc_mujoco::MjConfiguration config;
  config.mc_config = argv[1];
  config.with_visualization = false;
  config.with_mc_rtc_gui = false;
  config.torque_control = true;
  mc_mujoco::MjSim sim(config);

  auto & ctl = dynamic_cast<G1RLQPController &>(sim.controller()->controller());

  int status = 0;
  size_t next = 0;
  double lowestPelvis = ctl.realRobot().posW().translation().z();
  while(sim.data().time < duration)
  {
    const double now = sim.data().time;
    while(next < events.size() && events[next].t <= now) apply(events[next++], ctl, now);

    if(sim.stepSimulation())
    {
      status = 2;
      break;
    }

    const double pelvis = ctl.realRobot().posW().translation().z();
    lowestPelvis = std::min(lowestPelvis, pelvis);
    if(pelvis < fallHeight)
    {
      status = 1;
      break;
    }
  }

  const char * outcome = status == 0 ? "reached its duration" : status == 1 ? "robot fell" : "controller failed";
  std::printf("[scenario] end t=%.3f: %s, lowest pelvis %.3f m\n", sim.data().time, outcome, lowestPelvis);
  sim.stopSimulation();
  // Skip the static destructors: the ROS plugin of mc_rtc aborts while it is torn down. _Exit does not flush.
  std::fflush(nullptr);
  std::_Exit(status);
}
