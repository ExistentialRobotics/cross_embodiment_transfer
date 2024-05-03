import pathlib
import argparse
import time
from ruamel.yaml import YAML

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import utils
import replay_buffer
from bc import BCObsActAgent as Agent


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

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################
    if params['expert_folder'] is not None:
        demo_dir = (pathlib.Path(params['expert_folder']) / 
            params['env_name'] / params['robots'] / params['controller_type']).resolve()

    if params['logdir_prefix'] is None:
        logdir_prefix = pathlib.Path(__file__).parent
    else:
        logdir_prefix = pathlib.Path(params['logdir_prefix'])
    data_path = logdir_prefix / 'logs' / time.strftime("%m.%d.%Y")
    logdir = '_'.join([
        time.strftime("%H-%M-%S"),
        params['env_name'],
        params['robots'],
        params['controller_type'],
        params['suffix']
    ])
    logdir = data_path / logdir
    params['logdir'] = str(logdir)
    print(params)

    # dump params
    logdir.mkdir(parents=True, exist_ok=True)
    import yaml
    with open(logdir / 'params.yml', 'w') as fp:
        yaml.safe_dump(params, fp, sort_keys=False)

    model_dir = logdir / 'models'
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

    ##################################
    ### SETUP ENV, AGENT
    ##################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = utils.make_robosuite_env(params['env_name'], robots=params['robots'],
        controller_type=params['controller_type'], **params['env_kwargs'])

    obs = env.reset()
    robot_obs_shape = np.concatenate([obs[k] for k in params['robot_obs_keys']]).shape
    obj_obs_shape = np.concatenate([obs[k] for k in params['obj_obs_keys']]).shape

    params['obs_keys'] = params['robot_obs_keys'] + params['obj_obs_keys']

    env = utils.make(
        params['env_name'], 
        robots=params['robots'],
        controller_type=params['controller_type'],
        obs_keys=params['obs_keys'], 
        seed=params['seed'],
        **params['env_kwargs'],
    )
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    print(f"Environment observation space shape {obs_shape}")
    print(f"Environment action space shape {act_shape}")

    eval_env = utils.make(
        params['env_name'], 
        robots=params['robots'],
        controller_type=params['controller_type'],
        obs_keys=params['obs_keys'], 
        seed=params['seed']+100,
        **params['env_kwargs'],
    )

    logger = SummaryWriter(log_dir=params['logdir'])

    obs_dims = {
        'obs_dim': obs_shape[0], 
        'robot_obs_dim': robot_obs_shape[0],
        'obj_obs_dim': obj_obs_shape[0],
        'lat_obs_dim': params['lat_obs_dim']
    }
    act_dims = {
        'act_dim': act_shape[0],
        'lat_act_dim': params['lat_act_dim']
    }
    agent = Agent(
        obs_dims,
        act_dims,
        device,
    )

    agent_replay_buffer = replay_buffer.ReplayBuffer(
        obs_shape=obs_shape,
        action_shape=act_shape,
        capacity=2000000,
        batch_size=params['batch_size'],
        device=device
    )
    demo_paths = utils.load_episodes(demo_dir, params['obs_keys'])
    agent_replay_buffer.add_rollouts(demo_paths)

    for step in range(params['total_timesteps']):
        if step % params['evaluation']['interval'] == 0:
            print(f"Evaluating at step {step}")
            agent.eval_mode()
            utils.evaluate(eval_env, agent, 4, logger, step) 
            agent.train_mode()
        if step % params['evaluation']['save_interval'] == 0:
            print(f"Saving model at step {step}")
            step_dir = model_dir / f"step_{step:07d}"
            pathlib.Path(step_dir).mkdir(parents=True, exist_ok=True)
            agent.save(step_dir)

        agent.update(agent_replay_buffer, logger, step)

    logger.close()

if __name__ == '__main__':
    main()


