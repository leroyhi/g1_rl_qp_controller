<p align="center">
  <a >
    <img src="ext/image/logo.png" alt="logo" width="300">
  </a>
</p>

---

# G1 RL-QP controller

mc_rtc controller for the Unitree G1 29 DoF, forked from the RL-QP controller template. It deploys the fall-recovery bundle in `policies/fall_switch/`: a walking policy (`weak`), a catch policy (`recovery_v0`), and an estimator and limit kernel that decide when to switch between them.

## G1 setup

`plant.json` and `observation.json` in `policies/fall_switch/` are the source of truth. `scripts/gen_policy_from_plant.py` generates `policies/conventions.yaml` and each policy's `policy.yaml` and `observations.yaml` from them, so edit the generator rather than the generated files.

- **Robot module:** `G1_mode_5` from `mc_g1` (bastien-muraccioli fork, mc_g1 PR #4): 29 DoF rev 1.0 with the rubber hands, which matches the lab robot (`mode_machine` 5). `G1` is an alias of it; the deprecated `G1_29dof` is the mode 2 hardware.
- **Gains:** nominal `kp` and `kd` from `plant.json`, without the training randomisation.
- **Limits:** torque and velocity come from `plant.json`, the actuators' own limits, which the URDF underestimates on 8 joints. The position band is `pos_limit_soft`. The controller logs what the QP holds at startup. See `limits` in `etc/G1RLQPController.in.yaml`.
- **Run configuration:** `etc/mc_rtc_g1.yaml`, which leaves the mc_rtc user configuration untouched.

- **QP model:** the controller adds the armature to the robot module's mass matrix, and uses `lambda = 1/dt` for the joint-limit CBF. See the comments in `etc/G1RLQPController.in.yaml`.

### Dependencies

Install these three in the same prefix as mc_rtc, from the branches that bring the mode 5 robot (`G1_mode_5`, robot `g1_29dof_rev_1_0`):

- `g1_description`: [isri-aist/g1_description#3](https://github.com/isri-aist/g1_description/pull/3), URDF `g1_29dof_rev_1_0.urdf`
- `mc_g1`: [isri-aist/mc_g1#4](https://github.com/isri-aist/mc_g1/pull/4), robot module `G1_mode_5`
- `g1_mj_description`: [isri-aist/g1_mj_description#2](https://github.com/isri-aist/g1_mj_description/pull/2), mc_mujoco model `g1_29dof_rev_1_0.xml`

```bash
git fetch origin pull/<N>/head:pr-<N> && git checkout pr-<N>   # in each repository
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$HOME/workspace/install && cmake --build build -j && cmake --install build
```

`policies/fall_switch/training_mass_model.json` is generated for `G1_mode_5`, whose waist and torso frames are those of the training model. For `G1_29dof`, regenerate it with `scripts/gen_training_mass_model.py --robot G1_29dof`.

`g1_description` must be on `AMENT_PREFIX_PATH`, otherwise mc_rtc cannot resolve the `package://` URIs of the URDF. `scripts/run_sim.sh` does this for a prefix in `~/workspace/install`, or in `G1_INSTALL_PREFIX` if set.

### Run in simulation

```bash
scripts/run_sim.sh
```

The controller starts on the recovery policy, held until Cross gives control to the walking policy. This is how it starts with MCControlG1 too, when Enter is pressed on "Waiting on the ground". With the gamepad, through `mc_joystick_plugin`:

| Input | Effect |
|---|---|
| Left stick | `vx` and `vy` |
| Right stick | `yaw_rate` |
| Cross | Walking policy, arming it if needed |
| Circle | Recovery policy, arming it if needed. It holds until Cross, and the sticks still command it. |
| Square | Fall switch threshold on `p` lowered by 0.01, down to 0.01: switches earlier |
| Triangle | Fall switch threshold on `p` raised by 0.01, up to 0.99: switches later |

The threshold starts at the bundle's `thr` and is kept until the controller restarts. `G1RLQPController_FallSwitch_threshold` logs it. mc_joystick_plugin maps the raw button numbers of `/dev/input/js0`: with Linux's PlayStation driver, triangle is `X` and square is `Y`. If the buttons are swapped on your pad, change `fall_switch.threshold_buttons` in `etc/G1RLQPController.in.yaml`.

`HOLD (disarm)` stays in the GUI, under `G1RLQPController`. Without a gamepad, the same panel has `ARM policy`, and the commands are under `Command`. The fall switch can also take over by itself: untick `Fall switch > Automatic switch` to choose the policy by hand only.

### Scripted runs

`tests/sim/g1_sim_scenario` runs mc_mujoco without a window and triggers commands and pushes on the simulation clock. Build it with `-DBUILD_SIM_SCENARIO=ON`, then:

```bash
export AMENT_PREFIX_PATH=$HOME/workspace/install:$AMENT_PREFIX_PATH
build/tests/sim/g1_sim_scenario tests/sim/mc_rtc_scenario.yaml tests/sim/scenarios/walk.yaml
python3 scripts/analyze_sim_log.py /tmp/mc-control-G1RLQPController-latest.bin
```

The exit status is 0 if the run reached its duration, 1 if the robot fell and 2 if the controller failed.

### Fall switch

`src/policy/FallSwitch.cpp` runs on the 20 ms policy tick, before the action is chosen. It computes the feature groups of `physics_features()` in mjlab's `scripts/fall_switch/common.py` that the estimator and the limit kernel name in their `feature_groups`, holds the estimator's 5-frame window, and applies the rule of `policies/fall_switch/README.md`: switch to `recovery_v0` when `p` exceeds its threshold, give control back after `release_dwell_steps` consecutive ticks inside the limit kernel. During an automatic switch the policy sees a zero command and a stopped gait clock, and the `actions` term always stays the raw action of whichever policy acted last. To reproduce an automatic switch by hand, press Circle with the sticks at rest.

`release_dwell_steps` is set in `etc/G1RLQPController.in.yaml`: the bundle's README says 25, its `limit_kernel.json` says 10.

Since the v19 bundle (`weak_v19_tqnorm`), the estimator reads 13 features: `tilt`, `feet`, `dcm`, `shoulders` and `ang_mom`. The limit kernel still reads the 37 blind features. `dcm` and `ang_mom` use the body velocities, so the base linear velocity of the real robot: exact in mc_mujoco, where `BodySensor` takes the simulator's, but zero on hardware with an IMU only. Each `FallSwitch` warns about this at startup.

The logs are under `G1RLQPController_FallSwitch_`: `p`, `threshold`, `inside`, `mahalanobis`, `mahalanobis2`, `dwell`, `activePolicy`, `holdingRecovery`, `operatorHold`, `features` (the estimator's input) and `kernelFeatures` (the kernel's). To see when the policy changes in mc_log_ui, plot `holdingRecovery` with `operatorHold`: `activePolicy` is a string.

Three scripts check it:

```bash
# Features of the estimator and the kernel against mjlab's own function, on the states of a log
~/mjlab/.venv/bin/python scripts/check_fall_switch_features.py /tmp/mc-control-G1RLQPController-latest.bin
# Summary of a log
python3 scripts/analyze_sim_log.py /tmp/mc-control-G1RLQPController-latest.bin
# Falls, switches and false switches over many pushes, as in the bundle's README
python3 scripts/run_fall_campaign.py --episodes 300 --jobs 16
```

`scripts/gen_training_mass_model.py` generates `policies/fall_switch/training_mass_model.json`, the masses of training that the features need.

The rest of this README and `docs/` describe the template.

## Template

This project is based on a template for a new RL-QP controller project within [mc_rtc].

The goal of this template is to help users deploy reinforcement learning (RL) policies within a Quadratic Programming (QP) framework augmented with Control Barrier Functions (CBFs). This combination is enforcing physical and safety constraints, including:

* Joint position limits
* Joint velocity limits
* Torque limits
* Self-collision avoidance

Further details are available in:

[*Safe Execution of RL Policies via Acceleration-based CBF-QP Constraint Enforcement for Real-World Robotic Deployments*](https://hal.science/hal-05362571)

It comes with:
- a CMake project that can build a controller in [mc_rtc], the project can be put within [mc_rtc] source-tree for easier updates
- clang-format files
- automated GitHub Actions builds on three major platforms

Currently only ONNX format policies are supported.

This template has already been adapted to :
- [H1](https://github.com/Alhuuin/h1_rl_qp_controller)
- [HRP5P](https://github.com/bastien-muraccioli/hrp5p_rl_qp_controller)

If the target robot is already supported, please refer to the corresponding repository above. Otherwise, this template is intended as a starting point to integrate additional robots.

## Documentation

The complete documentation, including the controller architecture, observation system, policy configuration, adaptation guide and API reference, is available at:

**https://isri-aist.github.io/new-rl-qp-controller/**

The documentation also includes practical guides for adapting the controller to new robots and examples.

Its sources live in `docs/` (MkDocs Material) and are deployed to the `gh-pages` branch by the `Documentation` workflow on every push to `main`. To preview it locally with Docker:

```bash
cd docs
make            # serve on http://localhost:8000 with live reload
make build      # build the static website
```

## Template layout

- `policies/weak/`, `policies/recovery_v0/`: the two policies of the bundle.
- `policies/fall_switch/`: estimator, limit kernel and the bundle's JSON sidecars.
- `policies/conventions.yaml`: joint order of the bundle.
- `etc/G1RLQPController.in.yaml`: controller, observer and constraint configuration.
- `src/states/G1RLQPController_Initial.cpp`: policy-execution state.
- `src/observation/`: reusable observation implementations.
- `src/policy/`: policy loading, validation, ONNX inference and runtime state.

## Deploy your own policy

Policies are stored in `policies`. Each subdirectory describes everything the controller needs to know to load. run and use the policy.
See `docs/content/reference/policy-options.md` for every accepted field.

1. Create a new subdirectory in `policies`
1. Add the exported model in ONNX format inside
2. set the action and controlled-joint groups, provide training-consistent `kp` and `kd`, and add any required action scaling, default pose, phase period or command parameters in `policy.yaml`
5. reproduce the exact observation order and history in `observations.yaml`.

Set `default_policy` in `etc/G1RLQPController.in.yaml`.

## Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build . -j
sudo cmake --install .
```

Enable your controller in mc_rtc after completing all required adaptation points.

## Control flow summary

```
Every controller timestep (physics_step_size):
├── If syncTime >= policyStepSize:
│   ├── observation = getCurrentObservation()
│   ├── action      = rlPolicy->predict(observation)
│   ├── q_rl        = action * actionScale + q_zero
│   └── syncTime    = 0
│
├── τ = Kp*(q_rl - q) - Kd*q̇
│
├── useQP=true:  τ → TorqueJointTask → CBF-QP → robot
└── useQP=false: τ → robot (direct)
```

[mc_rtc]: https://jrl-umi3218.github.io/mc_rtc/
