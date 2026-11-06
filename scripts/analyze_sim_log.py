#!/usr/bin/env python3
"""Summarize a G1RLQPController log: gait quality, torque limits, position limits, QP tracking, timing.

  python3 scripts/analyze_sim_log.py /tmp/mc-control-G1RLQPController-latest.bin
"""

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


class Log:
    def __init__(self, path):
        with open(path, newline="") as f:
            rows = list(csv.reader(f, delimiter=";"))
        self.names = {name: i for i, name in enumerate(rows[0])}
        self.data = np.array([[self._number(x) for x in row] for row in rows[1:]])

    @staticmethod
    def _number(text):
        try:
            return float(text)
        except ValueError:
            return 0.0

    def scalar(self, name):
        return self.data[:, self.names[name]]

    def vector(self, base, size):
        return self.data[:, [self.names[f"{base}_{i}"] for i in range(size)]]

    def xyz(self, base):
        return self.data[:, [self.names[f"{base}_{axis}"] for axis in "xyz"]]


def load(path):
    if path.suffix == ".bin":
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "log.csv"
            subprocess.run(["mc_bin_to_log", str(path), str(csv_path)], check=True, capture_output=True)
            return Log(csv_path)
    return Log(path)


def segments(command):
    """Index ranges over which the command stays constant."""
    changes = np.flatnonzero(np.any(np.diff(command, axis=0) != 0, axis=1)) + 1
    bounds = [0, *changes.tolist(), len(command)]
    return list(zip(bounds[:-1], bounds[1:]))


def report_fall_switch(log, t, ticks):
    holding = log.scalar("G1RLQPController_FallSwitch_holdingRecovery") > 0.5
    p = log.scalar("G1RLQPController_FallSwitch_p")
    inside = log.scalar("G1RLQPController_FallSwitch_inside") > 0.5
    distance2 = log.scalar("G1RLQPController_FallSwitch_mahalanobis2")
    print(f"fall switch: recovery held {holding.mean():.1%} of the time, p max {p[ticks].max():.3f},"
          f" inside {inside[ticks].mean():.1%} of the ticks, d2 median {np.median(distance2[ticks]):.1f}")
    print(f"  walking ticks only: p above 0.5 on {(p[ticks][~holding[ticks]] > 0.5).mean():.1%},"
          f" inside {inside[ticks][~holding[ticks]].mean():.1%}")
    changes = np.flatnonzero(np.diff(holding.astype(int))) + 1
    for i in changes:
        action = "switch to recovery" if holding[i] else "hand back to walking"
        print(f"  t {t[i]:6.2f} {action}: p {p[i]:.3f}, d2 {distance2[i]:.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path)
    parser.add_argument("--settle", type=float, default=1.5, help="seconds ignored after a command change")
    args = parser.parse_args()

    log = load(args.log)
    joints = json.loads((ROOT / "policies/fall_switch/plant.json").read_text())["joints"]
    names = [j["name"] for j in joints]
    t = log.scalar("t")
    dt = float(np.median(np.diff(t)))

    q = log.vector("qIn", 29)
    qd = log.vector("alphaIn", 29)
    tau = log.vector("tauOut", 29)
    target = log.vector("G1RLQPController_RL_q", 29)
    kp = log.vector("G1RLQPController_kp_current", 29)
    kd = log.vector("G1RLQPController_kd_current", 29)
    pd = kp * (target - q) - kd * qd

    print(f"duration {t[-1]:.2f} s, dt {dt:.4f} s, QP used {log.scalar('G1RLQPController_useQP').mean():.0%} of the time")

    pelvis_z = log.scalar("ff_real_tz")
    # The log holds the sva rotation of the base, which maps world axes to base axes: conjugate it.
    w, x, y, z = (log.scalar("ff_real_qw"), *(-log.scalar(f"ff_real_q{a}") for a in "xyz"))
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
    print(f"pelvis height {pelvis_z.min():.3f} to {pelvis_z.max():.3f} m, tilt max {tilt.max():.1f} deg")

    updates = log.scalar("G1RLQPController_RL_update_count")
    ticks = np.flatnonzero(np.diff(updates) != 0) + 1
    periods = np.diff(t[ticks])
    if len(periods) > 1:
        print(f"policy ticks {len(ticks)}, period min {periods.min() * 1e3:.1f} ms, max {periods[1:].max() * 1e3:.1f} ms")
    else:
        print(f"policy ticks {len(ticks)}")

    command = log.xyz("G1RLQPController_RL_effectiveCommand")
    velocity_world = log.xyz("FloatingBase_linearVelocity")
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    vx = np.cos(yaw) * velocity_world[:, 0] + np.sin(yaw) * velocity_world[:, 1]
    vy = -np.sin(yaw) * velocity_world[:, 0] + np.cos(yaw) * velocity_world[:, 1]
    wz = log.xyz("FloatingBase_angularVelocity")[:, 2]
    measured = np.stack([vx, vy, wz], axis=1)
    settle = int(args.settle / dt)
    print("velocity tracking (command -> mean measured, vx vy wz):")
    for begin, end in segments(command):
        if end - begin <= settle:
            continue
        mean = measured[begin + settle : end].mean(axis=0)
        print(f"  t {t[begin]:5.1f}-{t[end - 1]:5.1f}  {np.array2string(command[begin], precision=2, suppress_small=True)}"
              f" -> {np.array2string(mean, precision=3, suppress_small=True)}")

    tl = np.array([j["torque_limit"] for j in joints])
    # The first tick is the jump from the robot module's stance to the policy's first target.
    start = int(0.05 / dt)
    print("torque peaks above 50% of the plant limit, or clipped by the QP or the bypass (after 0.05 s):")
    for i, name in enumerate(names):
        peak = np.abs(tau[start:, i]).max()
        demand = np.abs(pd[start:, i]).max()
        if peak > 0.5 * tl[i] or demand > peak + 1.0:
            print(f"  {name:28s} peak {peak:6.2f} Nm, PD demand {demand:6.2f} Nm, plant limit {tl[i]:6.1f} Nm")
    error = np.abs(tau[start:] - pd[start:])
    print(f"  largest |tau - PD|: {error.max():.3f} Nm, rms {np.sqrt(np.mean(error ** 2)):.4f} Nm,"
          f" ticks with an error above 0.1 Nm: {np.any(error > 0.1, axis=1).mean():.2%}")

    soft = np.array([j["pos_limit_soft"] for j in joints])
    hard = np.array([j["pos_limit_hard"] for j in joints])
    for label, band in (("soft", soft), ("hard", hard)):
        outside = (q < band[:, 0]) | (q > band[:, 1])
        joints_outside = [f"{names[i]} ({outside[:, i].mean():.2%})" for i in np.flatnonzero(outside.any(axis=0))]
        print(f"outside the {label} position limits: {', '.join(joints_outside) or 'never'}")
    vel_limit = np.array([j["velocity_limit"] for j in joints])
    print(f"largest |qd| / plant velocity limit: {(np.abs(qd) / vel_limit).max():.2f}"
          f" ({names[int((np.abs(qd) / vel_limit).max(axis=0).argmax())]})")

    if "G1RLQPController_FallSwitch_p" in log.names and len(ticks) > 0:
        report_fall_switch(log, t, ticks)

    for name in ("perf_SolverBuildAndSolve", "perf_ControllerRun"):
        duration = log.scalar(name)
        print(f"{name}: mean {duration.mean():.2f} ms, p99 {np.percentile(duration, 99):.2f} ms, max {duration.max():.2f} ms")


if __name__ == "__main__":
    main()
