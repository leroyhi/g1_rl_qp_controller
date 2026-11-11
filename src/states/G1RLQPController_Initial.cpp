#include "G1RLQPController_Initial.h"

#include <G1RLQPController.h>

void G1RLQPController_Initial::configure(const mc_rtc::Configuration &) {}

void G1RLQPController_Initial::start(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<G1RLQPController &>(ctl_);
  ctl.rlStateRunner.start(ctl, "RL_State");
  // Start on the recovery policy, held until the operator gives control to the walking policy (cross). The disarmed
  // hold needs its gains x8 to keep a passive robot up, and those gains diverge when the robot is in the air.
  ctl.selectRecoveryPolicy();
}

bool G1RLQPController_Initial::run(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<G1RLQPController &>(ctl_);
  ctl.rlStateRunner.run(ctl);
  ctl.torqueJointTask->setPosTarget(ctl.rlRuntime().q_rl());
  return false;
}

void G1RLQPController_Initial::teardown(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<G1RLQPController &>(ctl_);
}

EXPORT_SINGLE_STATE("G1RLQPController_Initial", G1RLQPController_Initial)
