"""
Gripper for Franka's Panda (has two fingers).
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class PandaGripperBase(GripperModel):
    """
    Gripper for Franka's Panda (has two fingers).

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
        return np.array([0.020833, -0.020833])

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["finger1_collision", "finger1_pad_collision"],
            "right_finger": ["finger2_collision", "finger2_pad_collision"],
            "left_fingerpad": ["finger1_pad_collision"],
            "right_fingerpad": ["finger2_pad_collision"],
        }


class PandaGripper(PandaGripperBase):
    """
    Modifies PandaGripperBase to only take one action.
    """
    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/panda_gripper.xml"), idn=idn)

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == self.dof
        self.current_action = np.clip(
            self.current_action + np.array([-1.0, 1.0]) * self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1


class PandaTouchGripper(PandaGripperBase):
    """
    Modifies PandaGripperBase to only take one action.
    """
    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/panda_touch_gripper.xml"), idn=idn)

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == self.dof
        self.current_action = np.clip(
            self.current_action + np.array([-1.0, 1.0]) * self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1


    @property
    def _important_sensors(self):
        """
        Sensor names for each gripper (usually "force_ee" and "torque_ee")

        Returns:
            dict:

                :`'force_ee'`: Name of force eef sensor for this gripper
                :`'torque_ee'`: Name of torque eef sensor for this gripper
                :`'touch1'`: Name of touch sensor on left finger
                :`'touch2'`: Name of touch sensor on right finger
        """
        return {sensor: sensor for sensor in ["force_ee", "torque_ee", "touch1", "touch2"]}


class PandaTactileGripper(PandaGripperBase):
    """
    Modifies PandaGripperBase to only take one action.
    """
    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/panda_tactile_gripper.xml"), idn=idn)

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == self.dof
        self.current_action = np.clip(
            self.current_action + np.array([-1.0, 1.0]) * self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1


    @property
    def _important_sensors(self):
        """
        Sensor names for each gripper (usually "force_ee" and "torque_ee")

        Returns:
            dict:

                :`'force_ee'`: Name of force eef sensor for this gripper
                :`'torque_ee'`: Name of torque eef sensor for this gripper
        """
        return {sensor: sensor for sensor in ["force_ee", "torque_ee"]}