import os  

import robosuite as suite

def make_env(cfgs):

    config = suite.load_controller_config(default_controller="JOINT_VELOCITY")

    env = suite.make(
        env_name="Lift", # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        reward_shaping=True,
        controller_configs=config,
        has_renderer=False,
        **cfgs
    )
    return env

def main():

    touch_obs_cfgs = dict(
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_touch_obs=True,
        use_tactile_obs=False,
    )
    env = make_env(touch_obs_cfgs)
    obs = env.reset()

    print("Environment with touch observation on gripper fingertip")
    print("Observation modes and shapes are: ")
    for k, v in obs.items():
        print(k, v.shape)


    # You may need to unset LD_PRELOAD in order to turn on off-screen renderer
    tactile_obs_cfgs = dict(
        has_offscreen_renderer=True,
        use_camera_obs=False,
        use_touch_obs=False,
        use_tactile_obs=True,
    )
    env = make_env(tactile_obs_cfgs)
    obs = env.reset()

    print()
    print("Environment with tactile observation on gripper fingertip")
    print("Observation modes and shapes are: ")
    for k, v in obs.items():
        print(k, v.shape)

if __name__ == '__main__':
    main()