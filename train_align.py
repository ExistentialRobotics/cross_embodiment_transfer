
import pathlib
import argparse
import time
from ruamel.yaml import YAML

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import utils
import replay_buffer
from align import ObsActAgent as Agent
from align import ObsActAligner as Aligner


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

    if params['logdir_prefix'] is None:
        logdir_prefix = pathlib.Path(__file__).parent
    else:
        logdir_prefix = pathlib.Path(params['logdir_prefix'])
    data_path = logdir_prefix / 'logs' / time.strftime("%m.%d.%Y")
    logdir = '_'.join([
        time.strftime("%H-%M-%S"),
        params['env_name'],
        params['src_env']['robot'],
        params['src_env']['controller_type'],
        params['tgt_env']['robot'],
        params['tgt_env']['controller_type'],
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
    params['model_dir'] = str(model_dir)
    params['src_model_dir'] = pathlib.Path(params['src_model_dir'])

    logger = SummaryWriter(log_dir=params['logdir'])

    ##################################
    ### SETUP ENV, AGENT
    ##################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    src_env = utils.make(
        params['env_name'], 
        robots=params['src_env']['robot'],
        controller_type=params['src_env']['controller_type'],
        obs_keys=params['src_env']['robot_obs_keys'], 
        seed=params['seed'],
        **params['env_kwargs'],
    )

    tgt_env = utils.make(
        params['env_name'], 
        robots=params['tgt_env']['robot'],
        controller_type=params['tgt_env']['controller_type'],
        obs_keys=params['tgt_env']['robot_obs_keys'], 
        seed=params['seed'],
        **params['env_kwargs'],
    )

    src_eval_env = utils.make_robosuite_env(
        params['env_name'], 
        robots=params['src_env']['robot'],
        controller_type=params['src_env']['controller_type'],
        **params['env_kwargs'],
    )

    tgt_eval_env = utils.make_robosuite_env(
        params['env_name'], 
        robots=params['tgt_env']['robot'],
        controller_type=params['tgt_env']['controller_type'],
        **params['env_kwargs'],
    )

    # Agent
    src_obs = src_eval_env.reset()
    src_robot_obs_shape = np.concatenate([src_obs[k] for k in params['src_env']['robot_obs_keys']]).shape
    src_obj_obs_shape = np.concatenate([src_obs[k] for k in params['src_env']['obj_obs_keys']]).shape
    tgt_obs = tgt_eval_env.reset()
    tgt_robot_obs_shape = np.concatenate([tgt_obs[k] for k in params['tgt_env']['robot_obs_keys']]).shape
    tgt_obj_obs_shape = np.concatenate([tgt_obs[k] for k in params['tgt_env']['obj_obs_keys']]).shape

    assert src_obj_obs_shape[0] == tgt_obj_obs_shape[0]

    env_params = params['src_env']
    src_eval_env = utils.make(
        params['env_name'], 
        robots=env_params['robot'],
        controller_type=env_params['controller_type'],
        obs_keys=env_params['robot_obs_keys']+env_params['obj_obs_keys'], 
        seed=params['seed'],
        **params['env_kwargs'],
    )

    env_params = params['tgt_env']
    tgt_eval_env = utils.make(
        params['env_name'], 
        robots=env_params['robot'],
        controller_type=env_params['controller_type'],
        obs_keys=env_params['robot_obs_keys']+env_params['obj_obs_keys'], 
        seed=params['seed'],
        **params['env_kwargs'],
    )


    src_obs_dims = {
        'robot_obs_dim': src_robot_obs_shape[0],
        'obs_dim': src_robot_obs_shape[0] + src_obj_obs_shape[0],
        'lat_obs_dim': params['lat_obs_dim'],
        'obj_obs_dim': src_obj_obs_shape[0],
    }
    src_act_dims = {
        'act_dim': src_eval_env.action_space.shape[0],
        'lat_act_dim': params['lat_act_dim'],
    }

    src_agent = Agent(src_obs_dims, src_act_dims, device)
    src_agent.load(params['src_model_dir'])
    src_agent.freeze()              # Freeze source agent

    # src_agent.eval_mode()
    # utils.evaluate(src_eval_env, src_agent, 10, logger, 0) 
    # import ipdb; ipdb.set_trace()

    tgt_obs_dims = {
        'robot_obs_dim': tgt_robot_obs_shape[0],
        'obs_dim': tgt_robot_obs_shape[0] + tgt_obj_obs_shape[0],
        'lat_obs_dim': params['lat_obs_dim'],
        'obj_obs_dim': tgt_obj_obs_shape[0],
    }
    tgt_act_dims = {
        'act_dim': tgt_eval_env.action_space.shape[0],
        'lat_act_dim': params['lat_act_dim'],
    }

    tgt_agent = Agent(tgt_obs_dims, tgt_act_dims, device)
    # Load latent policy for target and freeze
    tgt_agent.load_actor(params['src_model_dir'])       

    src_buffer = replay_buffer.ReplayBuffer(
        obs_shape=src_env.observation_space.shape,
        action_shape=src_env.action_space.shape,
        capacity=int(1e6),
        batch_size=params['batch_size'],
        device=device
    )
    demo_paths = utils.load_episodes(pathlib.Path(params['src_buffer']), params['src_env']['robot_obs_keys'])
    src_buffer.add_rollouts(demo_paths)

    tgt_buffer = replay_buffer.ReplayBuffer(
        obs_shape=tgt_env.observation_space.shape,
        action_shape=tgt_env.action_space.shape,
        capacity=int(1e6),
        batch_size=params['batch_size'],
        device=device
    )
    demo_paths = utils.load_episodes(pathlib.Path(params['tgt_buffer']), params['tgt_env']['robot_obs_keys'])
    tgt_buffer.add_rollouts(demo_paths)

    aligner = Aligner(src_agent, tgt_agent, device, log_freq=10)
    for step in range(params['tgt_align_timesteps']):
        for _ in range(5):
            src_obs, src_act, _, src_next_obs, _ = src_buffer.sample()
            tgt_obs, tgt_act, _, tgt_next_obs, _ = tgt_buffer.sample()
            src_act = src_act[:, :-1]
            tgt_act = tgt_act[:, :-1]
            aligner.update_disc(src_obs, src_act, tgt_obs, tgt_act, logger, step)
        aligner.update_gen(src_obs, src_act, src_next_obs, 
            tgt_obs, tgt_act, tgt_next_obs, logger, step)     

        if step % params['evaluation']['interval'] == 0:
            tgt_agent.eval_mode()
            utils.evaluate(tgt_eval_env, tgt_agent, 4, logger, step)
            tgt_agent.train_mode()

            print(f"Saving model at step {step}")
            step_dir = model_dir / f"step_{step:07d}"
            pathlib.Path(step_dir).mkdir(parents=True, exist_ok=True)
            tgt_agent.save(step_dir)



if __name__ == '__main__':
    main()