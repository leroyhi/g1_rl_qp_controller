#!/usr/bin/env python3
"""Generate policies/fall_switch/training_mass_model.json from mjlab's g1.xml.

The estimator's features need body masses, centres of mass and inertias as they were in training.
The URDF that mc_rtc loads differs: 35.1 kg against 33.3 kg, with up to 49 mm between the
centres of mass of a body. So the kinematics come from mc_rtc and the masses from this file,
with each centre of mass expressed in the frame of the URDF link. The inertia is the full 3x3 matrix about the centre
of mass, in the axes of the link (the frame offsets below are translations only).

  python3 scripts/gen_training_mass_model.py [--g1-xml PATH] [--robot G1_mode_5|G1_29dof]
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_G1_XML = Path.home() / "mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"
OUTPUT = ROOT / "policies/fall_switch/training_mass_model.json"

# Where the frame of a training body sits in the frame of the same URDF link, per robot module of mc_g1.
# - G1_mode_5 (g1_29dof_rev_1_0.urdf of g1_description PR #3): the waist and torso frames are those of g1.xml.
# - G1_29dof (g1_29dof.urdf): waist_roll_link is 9 mm higher in g1.xml and torso_link 19 mm lower relative to it, so
#   10 mm lower overall. Without this, the 7.8 kg of the torso move the centre of mass by 2.3 mm.
# The wrong set moves com_pos by the same 2.3 mm: check_fall_switch_features.py shows it.
FRAME_OFFSETS = {
    "G1_mode_5": {},
    "G1_29dof": {"waist_roll_link": [0.0, 0.0, 0.009], "torso_link": [0.0, 0.0, -0.010]},
}


def inertia(inertial):
    """Inertia about the centre of mass in the body axes: R diag(I) R^T, R from the inertial quat (w, x, y, z)."""
    if inertial.get("fullinertia") is not None:
        raise SystemExit("gen_training_mass_model: fullinertia is not handled")
    w, x, y, z = (float(v) for v in inertial.get("quat", "1 0 0 0").split())
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    diagonal = np.diag([float(v) for v in inertial.get("diaginertia").split()])
    return (rotation @ diagonal @ rotation.T).round(12).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--g1-xml", type=Path, default=DEFAULT_G1_XML)
    parser.add_argument("--robot", choices=sorted(FRAME_OFFSETS), default="G1_mode_5", help="MainRobot of mc_rtc")
    args = parser.parse_args()
    offsets = FRAME_OFFSETS[args.robot]

    bodies = []
    for body in ET.parse(args.g1_xml).getroot().iter("body"):
        inertial = body.find("inertial")
        if inertial is None:
            raise SystemExit(f"gen_training_mass_model: body {body.get('name')} has no inertial")
        bodies.append(
            {
                "name": body.get("name"),
                "mass": float(inertial.get("mass")),
                "com": [
                    float(x) + offset
                    for x, offset in zip(inertial.get("pos").split(), offsets.get(body.get("name"), [0.0] * 3))
                ],
                "inertia": inertia(inertial),
            }
        )
    OUTPUT.write_text(
        json.dumps(
            {"source": "mjlab asset_zoo/robots/unitree_g1/xmls/g1.xml", "robot": args.robot, "bodies": bodies}, indent=2
        )
        + "\n"
    )
    print(f"wrote {OUTPUT.relative_to(ROOT)} for {args.robot}: {len(bodies)} bodies, {sum(b['mass'] for b in bodies):.6f} kg")


if __name__ == "__main__":
    main()
