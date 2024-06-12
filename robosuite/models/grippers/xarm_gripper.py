"""
Gripper for xArm (has two fingers).
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion

class XArmGripperBase(GripperModel):
    """
    Gripper for xArm (has two fingers).

    Args:
        xml_path (str): XML path to specify gripper model
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, xml_path, idn=0):
        super().__init__(xml_path, idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0, 0, 0., 0, 0., 0.])

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "left_outer_finger_collision",
                "left_inner_finger_collision",
                "left_fingertip_collision",
                "left_fingerpad_collision",
            ],
            "right_finger": [
                "right_outer_finger_collision",
                "right_inner_finger_collision",
                "right_fingertip_collision",
                "right_fingerpad_collision",
            ],
            "left_fingerpad": ["left_fingerpad_collision"],
            "right_fingerpad": ["right_fingerpad_collision"],
        }


class XArmGripper(XArmGripperBase):
    """
    Modifies XArmGripperBase to only take one action.
    """
    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/xarm_gripper.xml"), idn=idn)

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == 1
        self.current_action = np.clip(
            self.current_action + self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def speed(self):
        return 0.005

    @property
    def dof(self):
        return 1

