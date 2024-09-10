from collections import OrderedDict

import numpy as np
from scipy.spatial.transform import Rotation as R

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, BallObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
import robosuite.utils.transform_utils as T


class Lift(SingleArmEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.
        
        use_touch_obs (bool): if True, include a grasping touch pressure information in the observation.

        use_tactile_obs (bool): if True, include a tactile sensor observation from depth camera


    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        table_offset=(0, 0, 0.8),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        use_touch_obs=False,
        use_tactile_obs=False,
        init_cube_pos=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array(table_offset)

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # whether to include touch sensor pressure in observations
        self.use_touch_obs = use_touch_obs
        self.use_tactile_obs = use_tactile_obs

        if self.use_touch_obs:
            assert gripper_types in ['PandaTouchGripper', 'Robotiq85TouchGripper', 'RethinkTouchGripper'], (
                "Must specify gripper_types in ['PandaTouchGripper', 'Robotiq85TouchGripper', 'RethinkTouchGripper']")

        elif self.use_tactile_obs:
            assert robots == "Panda", "Tactile sensor is only implemented on Panda gripper"
            gripper_types = "PandaTactileGripper" 

        self.init_cube_pos = init_cube_pos

        self.target_pos = np.array([0., 0., 1.0])

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if the cube is lifted

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube

        The sparse reward only consists of the lifting component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 2.25

        # use a shaping reward
        elif self.reward_shaping:

            # reaching reward
            cube_pos = self.sim.data.body_xpos[self.cube_body_id]
            gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
            dist = np.linalg.norm(gripper_site_pos - cube_pos)
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward

            # grasping reward
            if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube):
                reward += 0.25
            
            # move cube to target reward
            dist = np.linalg.norm(self.target_pos - cube_pos)
            reach_target_reward = 1 - np.tanh(10.0 * dist)
            reward += reach_target_reward

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cube = BoxObject(
            name="cube",
            # size_min=[0.025, 0.025, 0.025],  # [0.015, 0.015, 0.015],
            # size_max=[0.025, 0.025, 0.025],  # [0.018, 0.018, 0.018])
            size_min=[0.02, 0.02, 0.02],  # [0.015, 0.015, 0.015],
            size_max=[0.02, 0.02, 0.02],  # [0.018, 0.018, 0.018])
            rgba=[1, 0, 0, 1],
            material=redwood,
        )

        if self.init_cube_pos is None:
            x_range = [-0.05, 0.05]
            y_range = [-0.05, 0.05]     
        else:
            x_range = [self.init_cube_pos[0], self.init_cube_pos[0]]
            y_range = [self.init_cube_pos[1], self.init_cube_pos[1]]   

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=x_range,
                y_range=y_range,
                rotation=0,
                # rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        self.target = BallObject(
            name="target",
            size=[0.01],
            rgba=[0, 1, 0, 1],
            obj_type='visual',
            joints=None
        )
        self.target.get_obj().set("pos", " ".join([str(num) for num in self.target_pos]))

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.cube, self.target],
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)

        if self.robots[0].gripper.name.startswith("Robotiq85"):
            self.fingerpad_id1 = self.sim.model.geom_name2id('gripper0_left_fingerpad_collision')
            self.fingerpad_id2 = self.sim.model.geom_name2id('gripper0_right_fingerpad_collision')
            self.fingerpad_offset = 0.02
        elif self.robots[0].gripper.name.startswith("Panda"):
            self.fingerpad_id1 = self.sim.model.geom_name2id('gripper0_finger1_pad_collision')
            self.fingerpad_id2 = self.sim.model.geom_name2id('gripper0_finger2_pad_collision')
            self.fingerpad_offset = 0.007
        elif self.robots[0].gripper.name.startswith("Rethink"):
            self.fingerpad_id1 = self.sim.model.geom_name2id('gripper0_l_fingerpad_g0')
            self.fingerpad_id2 = self.sim.model.geom_name2id('gripper0_r_fingerpad_g0')
            self.fingerpad_offset = 0.02

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # Get prefix from robot model to avoid naming clashes for multiple robots
        pf = self.robots[0].robot_model.naming_prefix

        sensors = []
        names = []
        enableds = []
        actives = []

        # Add touch sensor observation
        if self.use_touch_obs:

            @sensor(modality=f"{pf}touch")
            def gripper_touch(obs_cache):
                touch_pressure = np.array([
                    self.robots[0].get_sensor_measurement('gripper0_touch1').item(),
                    self.robots[0].get_sensor_measurement('gripper0_touch2').item(),
                ])
                touch_pressure /= 10 
                touch_pressure[touch_pressure >= 1] = 1
                return touch_pressure

            sensors.append(gripper_touch)
            names.append(f"{pf}touch")
            enableds.append(True)
            actives.append(True)

        elif self.use_tactile_obs:

            @sensor(modality=f"{pf}tactile_depth")
            def gripper_tactile_depth(obs_cache):
                left_img, left_depth = self.sim.render(width=84, height=84, camera_name="gripper0_tactile_camera_left", depth=True)
                right_img, right_depth = self.sim.render(width=84, height=84, camera_name="gripper0_tactile_camera_right", depth=True)
                tactile_depth = np.stack([left_depth, right_depth], axis=-1)

                return tactile_depth

            sensors.append(gripper_tactile_depth)
            names.append(f"{pf}tactile_depth")
            enableds.append(True)
            actives.append(True)

        # Add gripper width observation
        gripper_name = self.robots[0].gripper.name
        if gripper_name.startswith("Panda") or gripper_name.startswith("Robotiq85") or gripper_name.startswith("Rethink"):

            @sensor(modality=f"{pf}gripper_width")
            def gripper_width(obs_cache):
                width = np.linalg.norm(self.sim.data.geom_xpos[self.fingerpad_id1] 
                    - self.sim.data.geom_xpos[self.fingerpad_id2]) - self.fingerpad_offset
                return width

            sensors.append(gripper_width)
            names.append(f"{pf}gripper_width")
            enableds.append(True)
            actives.append(True)

        # low-level object information
        if self.use_object_obs:
            modality = "object"

            # for conversion to relative gripper frame
            @sensor(modality=modality)
            def world_pose_in_gripper(obs_cache):
                return (
                    T.pose_inv(T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"])))
                    if f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache
                    else np.eye(4)
                )

            sensors.append(world_pose_in_gripper)
            names.append("world_pose_in_gripper")
            enableds.append(True)
            actives.append(False)

            # cube-related observables
            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                return T.convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")

            obj_name = 'cube'
            @sensor(modality=modality)
            def obj_to_eef_pos(obs_cache):
                # Immediately return default value if cache is empty
                if any(
                    [name not in obs_cache for name in [f"{obj_name}_pos", f"{obj_name}_quat", "world_pose_in_gripper"]]
                ):
                    return np.zeros(3)
                obj_pose = T.pose2mat((obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"]))
                rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache["world_pose_in_gripper"])
                rel_pos, rel_quat = T.mat2pose(rel_pose)
                obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
                rel_pos = obs_cache[f"{pf}eef_pos"] - obs_cache[f"{obj_name}_pos"]
                return rel_pos  

            @sensor(modality=modality)
            def obj_to_eef_quat(obs_cache):
                return (
                    obs_cache[f"{obj_name}_to_{pf}eef_quat"] if f"{obj_name}_to_{pf}eef_quat" in obs_cache else np.zeros(4)
                )

            @sensor(modality=modality)
            def eef_to_obj_yaw(obs_cache):
                """Computes the minimum yaw rotation to align with object"""
                if f"{pf}eef_quat" not in obs_cache or f"{obj_name}_quat" not in obs_cache:
                    return np.array([0.,])

                eef_quat = obs_cache[f'{pf}eef_quat']
                obj_quat = obs_cache[f'{obj_name}_quat']
                eef_az = R.from_quat(eef_quat).as_euler('zyx')[0]
                obj_az = R.from_quat(obj_quat).as_euler('zyx')[0]

                # Flip gripper yaw since it is pointing down
                eef_az = self.normalize_angle(2 * np.pi - eef_az)

                possible_az = self.normalize_angle(obj_az + np.pi * np.array([0, 0.5, 1, 1.5]))
                diff_az = self.normalize_angle(np.array([az - eef_az for az in possible_az]))
                i = np.argmin(np.abs(diff_az))
                return np.array([possible_az[i] - eef_az,])
            
            @sensor(modality=modality)
            def obj_to_target_pos(obs_cache):
                if 'cube_pos' not in obs_cache:
                    return np.zeros(3)
                else:
                    return self.target_pos - obs_cache['cube_pos']

            obj_sensors = [cube_pos, cube_quat, obj_to_eef_pos, obj_to_eef_quat, 
                eef_to_obj_yaw, obj_to_target_pos]
            obj_sensor_names = [
                f"{obj_name}_pos", f"{obj_name}_quat", 
                f"{obj_name}_to_{pf}eef_pos", 
                f"{obj_name}_to_{pf}eef_quat",
                f"{pf}eef_to_{obj_name}_yaw",
                f"{obj_name}_to_target_pos",
            ]
            sensors.extend(obj_sensors)
            names.extend(obj_sensor_names) 
            enableds += [True] * len(obj_sensors)
            actives += [True] * len(obj_sensor_names)


        # Create observables
        if len(names) > 0:
            for name, s, enabled, active in zip(names, sensors, enableds, actives):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    enabled=enabled,
                    active=active
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube)

    def _check_success(self):
        """
        Check if cube has been lifted.

        Returns:
            bool: True if cube has been lifted
        """
        # cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        # table_height = self.model.mujoco_arena.table_offset[2]

        # # cube is higher than the table top above a margin
        # return cube_height > table_height + 0.04

        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        dist = np.linalg.norm(self.target_pos - cube_pos)
        return dist < 0.02

    # @property
    # def _has_gripper_contact(self):
    #     """
    #     Determines whether the gripper is making contact with an object

    #     Returns:
    #         bool: True if there is contact between gripper fingers and cube object
    #     """
    #     return self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube)
