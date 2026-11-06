#!/usr/bin/env python3
"""Measure the fall switch over many simulated episodes, following the bundle's README.

Each episode lasts 9 s and takes one instantaneous push of 0.3 to 1.0 m/s in a random direction,
1.5 to 5 s after the start, under a command drawn from the training ranges. Every variant runs on the
same schedule, so differences between variants are not differences of luck. Undisturbed runs give
the false switch rate.

The numbers do not compare directly with the README: flat ground, no domain randomisation, no sensor
noise, a QP in the loop, and mc_mujoco instead of mjlab.

  python3 scripts/run_fall_campaign.py --episodes 200 --jobs 16 --output /tmp/campaign.json

Needs the controller installed, and build/tests/sim/g1_sim_scenario.
"""

import argparse
import json
import math
import os
import random
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "build/tests/sim/g1_sim_scenario"
INSTALL = Path(os.environ.get("G1_INSTALL_PREFIX", Path.home() / "workspace/install"))

COMMAND_RANGES = ((-0.5, 0.5), (-0.25, 0.25), (-0.7, 0.7))
EPISODE_S = 9.0
PUSH_BINS = 5

# Controller configuration overrides, and whether the QP is on.
VARIANTS = {
    "weak_alone": ("fall_switch: {enabled: false}\n", True),
    "switch_dwell25": ("fall_switch: {release_dwell_steps: 25}\n", True),
    "switch_dwell10": ("fall_switch: {release_dwell_steps: 10}\n", True),
    "switch_lambda200": ("cbf: {joint_limit: {lambda: 200.0}}\n", True),
    "switch_plant_limits": ("limits: {rule: plant}\n", True),
    "weak_plant_limits": ("limits: {rule: plant}\nfall_switch: {enabled: false}\n", True),
    "switch_no_qp": ("", False),
}


def global_config(path, module_dir):
    path.write_text(
        "MainRobot: G1_29dof\n"
        "Enabled: G1RLQPController\n"
        "Timestep: 0.002\n"
        "IncludeHalfSitController: false\n"
        "ClearControllerModulePath: true\n"
        f'ControllerModulePaths: ["{module_dir}", "{INSTALL / "lib/mc_controller"}"]\n'
        "Plugins: []\n"
        "Log: false\n"
        "GUIServer:\n"
        "  Enable: false\n"
    )


def draw_episodes(count, seed, push_max):
    generator = random.Random(seed)
    episodes = []
    for _ in range(count):
        magnitude = generator.uniform(0.3, push_max)
        angle = generator.uniform(0.0, 2.0 * math.pi)
        episodes.append(
            {
                "command": [generator.uniform(*r) for r in COMMAND_RANGES],
                "push_time": generator.uniform(1.5, 5.0),
                "magnitude": magnitude,
                "push": [magnitude * math.cos(angle), magnitude * math.sin(angle)],
            }
        )
    return episodes


def run(config, scenario, workdir, environment, duration):
    """Runs a scenario, once more if it hangs. About 1 run in 1800 stops making progress, for a reason not found yet."""
    hangs = 0
    # Far above the time a run takes, including with every core busy
    timeout = 60.0 + 3.0 * duration
    while True:
        try:
            result = subprocess.run(
                [str(RUNNER), str(config), str(scenario)],
                cwd=workdir,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            hangs += 1
            if hangs < 3:
                continue
            raise
        match = re.findall(r"^\[result\] (\{.*\})$", result.stdout, flags=re.M)
        if not match:
            raise RuntimeError(f"no result (exit {result.returncode}):\n{result.stdout[-2000:]}\n{result.stderr[-1000:]}")
        outcome = json.loads(match[-1])
        outcome["hangs"] = hangs
        return outcome


def episode_scenario(episode, use_qp):
    events = [f"  - {{t: 0.5, command: {episode['command']}}}\n"]
    if not use_qp:
        events.insert(0, "  - {t: 0.0, use_qp: false}\n")
    events.append(f"  - {{t: {episode['push_time']:.3f}, velocity_push: {episode['push']}}}\n")
    return f"duration: {EPISODE_S}\nfall_height: 0.4\nfall_tilt_deg: 80.0\nevents:\n" + "".join(events)


def undisturbed_scenario(seed, duration):
    generator = random.Random(seed)
    events, t = [], 0.5
    while t < duration:
        command = [generator.uniform(*r) for r in COMMAND_RANGES]
        events.append(f"  - {{t: {t:.3f}, command: {command}}}\n")
        t += generator.uniform(4.0, 10.0)
    return f"duration: {duration}\nfall_height: 0.4\nfall_tilt_deg: 80.0\nevents:\n" + "".join(events)


def summarize(results, episodes, push_max):
    n = len(results)
    # A QP failure stops the controller: the robot is lost as well
    falls = [r["status"] != 0 for r in results]
    fall_rate = sum(falls) / n
    bins = [0.3 + i * (push_max - 0.3) / PUSH_BINS for i in range(PUSH_BINS + 1)]
    bins[-1] += 1e-6
    failures = [r for r in results if r["status"] == 2]
    pre_exposure = sum(min(e["push_time"], r["first_switch"] if r["first_switch"] >= 0 else e["push_time"]) for r, e in zip(results, episodes))
    per_bin = []
    for low, high in zip(bins[:-1], bins[1:]):
        selected = [f for f, e in zip(falls, episodes) if low <= e["magnitude"] < high]
        per_bin.append(100.0 * sum(selected) / max(len(selected), 1))
    return {
        "episodes": n,
        "fall_pct": 100.0 * fall_rate,
        "qp_failure_pct": 100.0 * len(failures) / n,
        "qp_failure_pelvis_height": sorted(r["pelvis_at_end"] for r in failures),
        "hangs": sum(r["hangs"] for r in results),
        "fall_stderr_pct": 100.0 * math.sqrt(fall_rate * (1.0 - fall_rate) / n),
        "switched_pct": 100.0 * sum(r["switches"] > 0 for r in results) / n,
        "walking_lost_pct": 100.0 * sum(r["holding_time"] / max(r["end"], 1e-9) for r in results) / n,
        "hand_backs_per_episode": sum(r["hand_backs"] for r in results) / n,
        "switches_per_episode": sum(r["switches"] for r in results) / n,
        "false_switches_per_min": sum(r["switches_before_push"] for r in results) / max(pre_exposure / 60.0, 1e-9),
        "fall_pct_per_push_bin": per_bin,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--undisturbed", type=int, default=30, help="number of undisturbed runs of 60 s")
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--push-max", type=float, default=1.0, help="largest push in m/s; the README uses 1.0")
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    episodes = draw_episodes(args.episodes, args.seed, args.push_max)
    report = {}
    with tempfile.TemporaryDirectory(prefix="g1_campaign_") as tmp:
        tmp = Path(tmp)
        environment = dict(os.environ)
        environment["AMENT_PREFIX_PATH"] = f"{INSTALL}:{environment.get('AMENT_PREFIX_PATH', '')}"
        environment["TMPDIR"] = str(tmp)
        environment["OMP_NUM_THREADS"] = "1"

        def prepare(name, overrides):
            module_dir = tmp / name
            (module_dir / "etc").mkdir(parents=True)
            (module_dir / "etc/G1RLQPController.yaml").write_text(overrides or "{}\n")
            config = module_dir / "mc_rtc.yaml"
            global_config(config, module_dir)
            return module_dir, config

        with ThreadPoolExecutor(args.jobs) as pool:
            for name in args.variants:
                overrides, use_qp = VARIANTS[name]
                module_dir, config = prepare(name, overrides)
                jobs = []
                for i, episode in enumerate(episodes):
                    scenario = module_dir / f"episode_{i}.yaml"
                    scenario.write_text(episode_scenario(episode, use_qp))
                    jobs.append(pool.submit(run, config, scenario, module_dir, environment, EPISODE_S))
                results = [job.result() for job in jobs]
                report[name] = summarize(results, episodes, args.push_max)
                print(name, json.dumps(report[name]), flush=True)

            if args.undisturbed:
                module_dir, config = prepare("undisturbed", "")
                jobs = []
                for i in range(args.undisturbed):
                    scenario = module_dir / f"run_{i}.yaml"
                    scenario.write_text(undisturbed_scenario(args.seed + 1000 + i, 60.0))
                    jobs.append(pool.submit(run, config, scenario, module_dir, environment, 60.0))
                results = [job.result() for job in jobs]
                total = sum(r["end"] for r in results) / 60.0
                report["undisturbed"] = {
                    "minutes": total,
                    "falls": sum(r["status"] != 0 for r in results),
                    "hangs": sum(r["hangs"] for r in results),
                    "switches_per_min": sum(r["switches"] for r in results) / total,
                    "walking_lost_pct": 100.0 * sum(r["holding_time"] for r in results) / (total * 60.0),
                }
                print("undisturbed", json.dumps(report["undisturbed"]), flush=True)

    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
