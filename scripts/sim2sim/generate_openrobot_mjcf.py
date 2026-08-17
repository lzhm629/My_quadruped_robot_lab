"""Generate the floating-base OpenRobot MJCF from the training URDF."""

from __future__ import annotations

import argparse
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from config import DEFAULT_MODEL_PATH, PROJECT_ROOT


DEFAULT_URDF = (
    PROJECT_ROOT
    / "source"
    / "My_quadruped_robot_lab"
    / "My_quadruped_robot_lab"
    / "assets"
    / "data"
    / "openrobot_wheelfixed"
    / "urdf"
    / "openrobot_wheelfixed.urdf"
)


def generate(urdf_path: Path, output_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(urdf_path.resolve()))
    with tempfile.TemporaryDirectory() as temporary_directory:
        compiled_path = Path(temporary_directory) / "compiled.xml"
        mujoco.mj_saveLastXML(str(compiled_path), model)
        root = ET.parse(compiled_path).getroot()

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", "../collision")

    option = ET.SubElement(root, "option", timestep="0.005", gravity="0 0 -9.81", integrator="implicitfast")
    option.set("cone", "elliptic")
    ET.SubElement(root, "size", njmax="2000", nconmax="500")

    asset = root.find("asset")
    if asset is None:
        raise RuntimeError("Converted MJCF has no asset section")
    for mesh in asset.findall("mesh"):
        mesh.set("file", Path(mesh.attrib["file"]).name)
    ET.SubElement(asset, "texture", name="skybox", type="skybox", builtin="gradient", rgb1="0.3 0.5 0.7", rgb2="0 0 0", width="512", height="3072")
    ET.SubElement(asset, "texture", name="ground", type="2d", builtin="checker", rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3", width="300", height="300")
    ET.SubElement(asset, "material", name="ground", texture="ground", texuniform="true", texrepeat="5 5", reflectance="0.1")

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Converted MJCF has no worldbody")
    converted_children = list(worldbody)
    for child in converted_children:
        worldbody.remove(child)

    ET.SubElement(worldbody, "light", pos="0 0 4", dir="0 0 -1", directional="true")
    ET.SubElement(worldbody, "geom", name="floor", type="plane", size="0 0 0.1", material="ground", friction="1 0.005 0.0001")
    base = ET.SubElement(worldbody, "body", name="base_link", pos="0 0 0.69")
    ET.SubElement(base, "freejoint", name="root")
    ET.SubElement(
        base,
        "inertial",
        pos="-0.0095784 0.0000043325 0.10347",
        mass="55.995",
        fullinertia="0.51509 0.90609 1.1068 0.000000012331 -0.0000021274 0.00000016669",
    )
    for child in converted_children:
        base.append(child)

    for joint in root.findall(".//joint"):
        name = joint.attrib.get("name", "")
        if name.endswith("_hip_joint") or name.endswith("_thigh_joint"):
            joint.set("armature", "0.133886")
        elif name.endswith("_calf_joint"):
            joint.set("armature", "0.695525")
        joint.attrib.pop("actuatorfrcrange", None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)
    validated = mujoco.MjModel.from_xml_path(str(output_path.resolve()))
    if (validated.nq, validated.nv, validated.njnt) != (19, 18, 13):
        raise RuntimeError(
            f"Unexpected generated model dimensions: nq={validated.nq}, nv={validated.nv}, njnt={validated.njnt}"
        )
    print(f"Generated and validated MJCF: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    generate(args.urdf, args.output)


if __name__ == "__main__":
    main()
