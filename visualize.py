import pathlib
import argparse
from ruamel.yaml import YAML

import numpy as np
import torch 

from align import ObsActAgent as Agent
import utils

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--config', help='train config file path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    yaml = YAML(typ='safe')
    params = yaml.load(open(args.config))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = utils.make_robosuite_env(
        params['env_name'], 
        robots=params['robots'],
        controller_type=params['controller_type'],
        **params['env_kwargs'],
    )

    obs = env.reset()
    robot_obs_shape = np.concatenate([obs[k] for k in params['robot_obs_keys']]).shape
    obj_obs_shape = np.concatenate([obs[k] for k in params['obj_obs_keys']]).shape

    env = utils.make(
        params['env_name'], 
        robots=params['robots'],
        controller_type=params['controller_type'],
        obs_keys=params['robot_obs_keys']+params['obj_obs_keys'], 
        seed=params['seed']+100,
        **params['env_kwargs'],
        render=True
    )

    obs_dims = {
        'robot_obs_dim': robot_obs_shape[0],
        'obs_dim': robot_obs_shape[0] + obj_obs_shape[0],
        'lat_obs_dim': params['lat_obs_dim'],
        'obj_obs_dim': obj_obs_shape[0],
    }
    act_dims = {
        'act_dim': env.action_space.shape[0],
        'lat_act_dim': params['lat_act_dim'],
    }

    agent = Agent(obs_dims, act_dims, device)
    agent.load(pathlib.Path(params['model_dir']))


    for i in range(params['num_episodes']):
        obs = env.reset()
        env.render()
        done = False
        episode_reward = 0
        while not done:
            action = agent.sample_action(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward   
            env.render()

        print(f"Episode {i}, return {episode_reward}")


if __name__ == '__main__':
	main()
