"""Install the OpenRobot Lab Isaac Lab extension."""

from pathlib import Path

import toml
from setuptools import find_packages, setup


EXTENSION_ROOT = Path(__file__).resolve().parent
EXTENSION_DATA = toml.load(EXTENSION_ROOT / "config" / "extension.toml")

setup(
    name="My_quadruped_robot_lab",
    version=EXTENSION_DATA["package"]["version"],
    description=EXTENSION_DATA["package"]["description"],
    author=EXTENSION_DATA["package"]["author"],
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "My_quadruped_robot_lab": [
            "assets/data/go2/urdf/*",
            "assets/data/go2/dae/*",
            "assets/data/openrobot_wheelfixed/urdf/*",
            "assets/data/openrobot_wheelfixed/meshes/*",
            "assets/data/openrobot_wheelfixed/collision/*",
        ]
    },
    python_requires=">=3.11",
    install_requires=["toml"],
    zip_safe=False,
)
