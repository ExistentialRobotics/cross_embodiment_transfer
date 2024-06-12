import time
import numpy as np
import robosuite as suite
from robosuite.wrappers import VisualizationWrapper

import cv2

def main():

    controller_configs = suite.load_controller_config(default_controller="JOINT_POSITION")
    env = suite.make(
        env_name="Lift", # try with other tasks like "Stack" and "Door"
        robots="xArm6",  # try with other robots like "Sawyer" and "Jaco"
        reward_shaping=True,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        # has_renderer=False,
        # has_offscreen_renderer=True,
        # use_camera_obs=True,
        use_object_obs=True,
        # horizon=100,
        controller_configs=controller_configs
    )
    env = VisualizationWrapper(env, indicator_configs="default")


    result = cv2.VideoWriter('xarm.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (256, 256))

    # reset the environment
    env.reset()

    # for i in range(1000):
        # action = np.random.randn(env.robots[0].dof) # sample random action


    dof = env.robots[0].dof

    for u in range(dof):

        for _ in range(50):
            action = np.zeros(dof)
            action[u] = 1
            obs, reward, done, info = env.step(action)  # take action in the environment
            env.render()  # render on display
            # result.write(obs['agentview_image'])
        
        for _ in range(50):
            action = np.zeros(dof)
            action[u] = -1
            obs, reward, done, info = env.step(action)  # take action in the environment    
            env.render()  # render on display
            # result.write(obs['agentview_image'])
        # time.sleep(0.1)

    result.release()
    env.close()

if __name__ == '__main__':
    main()