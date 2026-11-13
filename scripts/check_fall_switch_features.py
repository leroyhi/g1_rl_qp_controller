#!/usr/bin/env python3
"""Compare the features logged by FallSwitch with mjlab's physics_features() on the same states.

Replays a log through forward kinematics and velocities on the training model and through the reference
function itself, so nothing is reimplemented here. Checks the estimator's and the kernel's inputs, each
with the groups of its sidecar, recomputes p with the ONNX estimator, the kernel distance and the state
machine.
Run it with mjlab's interpreter, from anywhere:

  ~/mjlab/.venv/bin/python scripts/check_fall_switch_features.py /tmp/mc-control-G1RLQPController-latest.bin
"""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "policies/fall_switch"
# The training model, which the reference features assume: not the mc_mujoco model of g1_mj_description
DEFAULT_XML = Path.home() / "mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"
DEFAULT_MJLAB = Path.home() / "mjlab"


def load_log(path):
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "log.csv"
        subprocess.run(["mc_bin_to_log", str(path), str(target)], check=True, capture_output=True)
        with open(target, newline="") as f:
            rows = list(csv.reader(f, delimiter=";"))

    def number(text):
        try:
            return float(text)
        except ValueError:
            return 0.0

    names = {name: i for i, name in enumerate(rows[0])}
    return names, np.array([[number(x) for x in row] for row in rows[1:]])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--mjlab", type=Path, default=DEFAULT_MJLAB)
    args = parser.parse_args()

    sys.path.insert(0, str(args.mjlab / "scripts/fall_switch"))
    import common

    common.DEVICE = "cpu"

    names, data = load_log(args.log)

    def column(name):
        return data[:, names[name]]

    def columns(base, size):
        return data[:, [names[f"{base}_{i}"] for i in range(size)]]

    updates = column("G1RLQPController_RL_update_count")
    ticks = np.flatnonzero((np.diff(updates) != 0) & (updates[1:] > 0)) + 1
    plant = json.loads((BUNDLE / "plant.json").read_text())["joints"]
    joint_names = [j["name"] for j in plant]

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    state = mujoco.MjData(model)
    body_ids = list(range(1, model.nbody))
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in body_ids]
    xml_joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)]
    if xml_joints != joint_names:
        raise SystemExit("joint order of the model and of plant.json differ")

    # The log holds the sva rotation of the base, which maps world axes to base axes: conjugate it.
    quat = np.stack(
        [column("ff_real_qw"), -column("ff_real_qx"), -column("ff_real_qy"), -column("ff_real_qz")], axis=1
    )[ticks]
    position = np.stack([column(f"ff_real_t{a}") for a in "xyz"], axis=1)[ticks]
    # What the real robot was given: the BodySensor's base velocity, in world axes, and the encoder velocities
    base = "Observers_MainPipeline_BodySensor_FloatingBase_velW"
    angular_w = np.stack([column(f"{base}_w{a}") for a in "xyz"], axis=1)[ticks]
    linear_w = np.stack([column(f"{base}_v{a}") for a in "xyz"], axis=1)[ticks]
    alpha = columns("alphaIn", 29)[ticks]
    q = columns("qIn", 29)[ticks]
    tau = columns("tauIn", 29)[ticks]
    target_before = columns("G1RLQPController_RL_q", 29)[ticks - 1]
    gyro = np.stack([column(f"FloatingBase_angularVelocity_{a}") for a in "xyz"], axis=1)[ticks]

    n = len(ticks)
    com = np.zeros((n, len(body_ids), 3))
    com_quat = np.zeros((n, len(body_ids), 4))
    com_velocity = np.zeros((n, len(body_ids), 6))
    for i in range(n):
        state.qpos[:3] = position[i]
        state.qpos[3:7] = quat[i]
        state.qpos[7:] = q[i]
        # Free joint: linear velocity of the base origin in world axes, angular velocity in base axes
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, quat[i])
        state.qvel[:3] = linear_w[i]
        state.qvel[3:6] = rotation.reshape(3, 3).T @ angular_w[i]
        state.qvel[6:] = alpha[i]
        mujoco.mj_forward(model, state)
        com[i] = state.xipos[1:]
        for b in body_ids:
            mujoco.mju_mat2Quat(com_quat[i, b - 1], state.ximat[b])
            # At the centre of mass, in world axes: (angular, linear)
            mujoco.mj_objectVelocity(model, state, mujoco.mjtObj.mjOBJ_BODY, b, com_velocity[i, b - 1], 0)

    def tensor(array):
        return torch.as_tensor(np.asarray(array), dtype=torch.float32)

    forward = np.stack(
        [
            1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
            2 * (quat[:, 1] * quat[:, 2] + quat[:, 0] * quat[:, 3]),
        ],
        axis=1,
    )
    robot_data = SimpleNamespace(
        body_com_pos_w=tensor(com),
        body_com_quat_w=tensor(com_quat),
        body_com_lin_vel_w=tensor(com_velocity[..., 3:]),
        body_com_ang_vel_w=tensor(com_velocity[..., :3]),
        root_link_quat_w=tensor(quat),
        root_link_pos_w=tensor(position),
        root_link_ang_vel_b=tensor(gyro),
        joint_pos=tensor(q),
        joint_pos_limits=tensor(np.tile([j["pos_limit_hard"] for j in plant], (n, 1, 1))),
        qfrc_actuator=tensor(tau),
        heading_w=tensor(np.arctan2(forward[:, 1], forward[:, 0])),
    )
    force_limit = tensor([[j["torque_limit"] for j in plant]])
    robot = SimpleNamespace(
        data=robot_data,
        indexing=SimpleNamespace(body_ids=torch.tensor(body_ids)),
        body_names=body_names,
        joint_names=joint_names,
        actuators=[SimpleNamespace(target_names=joint_names, force_limit=force_limit)],
    )
    action_term = SimpleNamespace(_processed_actions=tensor(target_before), target_ids=torch.arange(29))
    raw = SimpleNamespace(
        scene={
            "robot": robot,
            "feet_ground_contact": SimpleNamespace(data=SimpleNamespace(force=torch.ones(n, 2, 3))),
        },
        num_envs=n,
        step_dt=0.02,
        sim=SimpleNamespace(
            model=SimpleNamespace(
                body_mass=tensor(model.body_mass[None]), body_inertia=tensor(model.body_inertia[None])
            )
        ),
        action_manager=SimpleNamespace(get_term=lambda name: action_term),
    )
    reference, _ = common.physics_features(raw, None)

    reference = reference.numpy()

    offsets, index = {}, 0
    for group, width in common.PHYS_GROUPS.items():
        offsets[group] = range(index, index + width)
        index += width

    def compare(title, groups, entry):
        cols = [c for g in groups for c in offsets[g]]
        expected = reference[:, cols]
        logged = columns(entry, len(cols))[ticks]
        error = np.abs(logged - expected)
        print(f"{title}: {len(cols)} features, {n} policy ticks compared with physics_features()")
        index = 0
        for group in groups:
            width = common.PHYS_GROUPS[group]
            sensor = common.SENSOR_SIGMA.get(group, float("nan"))
            spread = expected[:, index:index + width].std(axis=0).max()
            print(f"  {group:13s} max error {error[:, index:index + width].max():.2e}  (spread in the log {spread:.2e},"
                  f" sensor sigma in training {sensor:g})")
            index += width
        print(f"  largest error: {error.max():.2e}")
        return cols, expected, logged

    estimator = json.loads((BUNDLE / "estimator.json").read_text())
    kernel = json.loads((BUNDLE / "limit_kernel.json").read_text())
    _, estimator_reference, _ = compare("estimator", estimator["feature_groups"], "G1RLQPController_FallSwitch_features")
    kernel_cols, kernel_reference, _ = compare(
        "kernel", kernel["feature_groups"], "G1RLQPController_FallSwitch_kernelFeatures"
    )
    if kernel.get("columns_into_phys_vector", kernel_cols) != kernel_cols:
        raise SystemExit("columns_into_phys_vector differs from the columns of the kernel's groups")

    offset = kernel_reference - np.array(kernel["mu"])
    distance2 = np.einsum("ni,ij,nj->n", offset, np.array(kernel["inv_cov"]), offset)
    logged_distance2 = column("G1RLQPController_FallSwitch_mahalanobis2")[ticks]
    print(f"Mahalanobis squared, largest error against the reference features: {np.abs(distance2 - logged_distance2).max():.2e}"
          f" (threshold {kernel['threshold']:.1f})")

    # p from the reference features, through the same window as FallSwitch: edge-padded, oldest frame first
    try:
        import onnxruntime
    except ImportError:
        onnxruntime = None
    if onnxruntime is None:
        print("p not recomputed: no onnxruntime")
    else:
        session = onnxruntime.InferenceSession(str(BUNDLE / "estimator.onnx"))
        z = np.clip((estimator_reference - np.array(estimator["mu"])) / np.array(estimator["sd"]), -8, 8)
        window = estimator["window"]
        frames = np.stack([z[np.clip(np.arange(i - window + 1, i + 1), 0, None)] for i in range(n)]).astype(np.float32)
        logits = session.run(None, {session.get_inputs()[0].name: frames})[0]
        column_index = estimator["horizons_steps"].index(estimator["switch_horizon_steps"])
        p_reference = 1 / (1 + np.exp(-logits[:, column_index]))
        p_logged = column("G1RLQPController_FallSwitch_p")[ticks]
        # The window restarts on each reset of FallSwitch, so the first ticks can differ
        print(f"p, largest error against the reference features: {np.abs(p_reference - p_logged).max():.2e}"
              f" (p in the log: max {p_logged.max():.3f}, p > threshold on {(p_logged > estimator['thr']).sum()} ticks)")
        flipped = (p_reference > estimator["thr"]) != (p_logged > estimator["thr"])
        print(f"  ticks on which the reference p is on the other side of the threshold {estimator['thr']:.3f}:"
              f" {flipped.sum()} at t = {np.round(column('t')[ticks][flipped], 2).tolist()}")

    # State machine, recomputed from the logged scores
    # The operator can move the threshold: take the one in force at each tick when the log has it
    if "G1RLQPController_FallSwitch_threshold" in names:
        threshold = column("G1RLQPController_FallSwitch_threshold")[ticks]
    else:
        threshold = np.full(n, estimator["thr"])
    p = column("G1RLQPController_FallSwitch_p")[ticks]
    inside = logged_distance2 <= kernel["threshold"]
    holding = column("G1RLQPController_FallSwitch_holdingRecovery")[ticks] > 0.5
    dwell_required = column("G1RLQPController_FallSwitch_dwell")
    required = int(sys.argv[sys.argv.index("--dwell") + 1]) if "--dwell" in sys.argv else 25
    replay, dwell, now = [], 0, False
    for i in range(n):
        if now:
            dwell = dwell + 1 if inside[i] else 0
            if dwell >= required:
                now, dwell = False, 0
        elif p[i] > threshold[i]:
            now, dwell = True, 0
        replay.append(now)
    print(f"state machine replayed at dwell {required}: {(np.array(replay) != holding).sum()} ticks differ out of {n}")

    # The observation seen by the catch policy after an automatic switch. The operator's choice keeps the sticks.
    observation = columns("G1RLQPController_RL_currentObservation", 98)[ticks]
    action_before = columns("G1RLQPController_RL_currentAction", 29)[ticks - 1]
    automatic = holding & (column("G1RLQPController_FallSwitch_operatorHold")[ticks] < 0.5)
    print(f"while recovery holds after an automatic switch: largest |command| {np.abs(observation[automatic, 6:9]).max() if automatic.any() else 0:.3f},"
          f" largest |phase| {np.abs(observation[automatic, 9:11]).max() if automatic.any() else 0:.3f}")
    print(f"actions term against the raw action of the previous tick: {np.abs(observation[1:, 69:98] - action_before[1:]).max():.2e}")
    del dwell_required


if __name__ == "__main__":
    main()
