"""OpenRobot Lab Isaac Lab extension."""

from pathlib import Path


My_QUADRUPED_ROBOT_LAB_ROOT_DIR = Path(__file__).resolve().parent

# Register Gymnasium environments.
from .tasks import *  # noqa: F401, F403, E402

