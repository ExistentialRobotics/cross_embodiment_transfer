import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class xArm6(ManipulatorModel):
    """
    xArm6 is a single arm robot created by UFactory.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/xarm6/robot.xml"), idn=idn)

        # Set joint damping
        self.set_joint_attribute(attrib="damping", 
            values=np.array((0.1, 0.1, 0.1, 0.1, 0.01, 0.01)))

        self.set_joint_attribute(attrib="frictionloss", 
            values=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.1)))

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        # return None
        # return "XArmGripper"
        return "Robotiq85Gripper"

    @property
    def default_controller_config(self):
        return "default_panda"

    @property
    def init_qpos(self):
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


    @property
    def base_xpos_offset(self):
        return {
            # "bins": (-0.5, -0.1, 0),
            "bins": (-0.407, 0.0, 0.171),    # PickPlace
            "empty": (-0.6, 0, 0),
            # "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "table": lambda table_length: (-0.407, 0, 0.171)      # Reach/Lift
            # "table": lambda table_length: (-0.437, 0, 0.171)
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
