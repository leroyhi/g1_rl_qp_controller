#!/usr/bin/env python3
"""Generate policies/fall_switch/training_mass_model.json from mjlab's g1.xml.

The estimator's features need body masses and centres of mass as they were in training.
The URDF that mc_rtc loads differs: 35.1 kg against 33.3 kg, with up to 49 mm between the
centres of mass of a body. So the kinematics come from mc_rtc and the masses from this file,
with each centre of mass expressed in the frame of the URDF link.

  python3 scripts/gen_training_mass_model.py [--g1-xml PATH]
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_G1_XML = Path.home() / "mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"
OUTPUT = ROOT / "policies/fall_switch/training_mass_model.json"

# Where the frame of a training body sits in the frame of the same URDF link. The two models differ on the waist:
# waist_roll_link is 9 mm higher in g1.xml and torso_link 19 mm lower relative to it, so 10 mm lower overall. Without
# this, the 7.8 kg of the torso move the centre of mass by 2.3 mm.
FRAME_OFFSETS = {"waist_roll_link": [0.0, 0.0, 0.009], "torso_link": [0.0, 0.0, -0.010]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--g1-xml", type=Path, default=DEFAULT_G1_XML)
    args = parser.parse_args()

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
                    for x, offset in zip(inertial.get("pos").split(), FRAME_OFFSETS.get(body.get("name"), [0.0] * 3))
                ],
            }
        )
    OUTPUT.write_text(json.dumps({"source": "mjlab asset_zoo/robots/unitree_g1/xmls/g1.xml", "bodies": bodies}, indent=2) + "\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)}: {len(bodies)} bodies, {sum(b['mass'] for b in bodies):.6f} kg")


if __name__ == "__main__":
    main()
