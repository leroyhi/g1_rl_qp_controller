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

- **Robot module:** `G1_29dof` from `mc_g1`. `MainRobot: G1` loads the 23 DoF variant.
- **Gains:** nominal `kp` and `kd` from `plant.json`, without the training randomisation.
- **Limits:** each joint uses the more conservative of its URDF limits and `plant.json`, with the `pos_limit_soft` position band. See `limits` in `etc/G1RLQPController.in.yaml`.
- **Run configuration:** `etc/mc_rtc_g1.yaml`, which leaves the mc_rtc user configuration untouched.

`g1_description` must be on `AMENT_PREFIX_PATH`, otherwise mc_rtc cannot resolve the `package://` URIs of the URDF:

```bash
export AMENT_PREFIX_PATH=$HOME/workspace/install:$AMENT_PREFIX_PATH
```

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
