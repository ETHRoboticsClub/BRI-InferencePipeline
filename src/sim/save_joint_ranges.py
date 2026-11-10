import json
from pathlib import Path

import mujoco_menagerie as mm

from src.sim.config import HAND, MODEL

SPECS_PATH = Path(__file__).resolve().parent.parent.parent / "shadow_hand_specs.json"


def save_joint_ranges(out_path: Path = SPECS_PATH) -> None:
    model = mm.load(MODEL, HAND)
    data_to_save: dict[str, dict] = {"joints": {}, "actuators": {}}

    for i in range(model.njnt):
        name = model.joint(i).name
        if model.jnt_limited[i]:
            jnt_range = model.jnt_range[i].tolist()
            data_to_save["joints"][name] = {"min": jnt_range[0], "max": jnt_range[1]}
        else:
            data_to_save["joints"][name] = {"min": "unlimited", "max": "unlimited"}

    for i in range(model.nu):
        name = model.actuator(i).name
        if model.actuator_ctrllimited[i]:
            ctrl_range = model.actuator_ctrlrange[i].tolist()
            data_to_save["actuators"][name] = {"min": ctrl_range[0], "max": ctrl_range[1]}
        else:
            data_to_save["actuators"][name] = {"min": "unlimited", "max": "unlimited"}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4)


if __name__ == "__main__":
    save_joint_ranges()
