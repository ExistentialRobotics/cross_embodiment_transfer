from typing import Union, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as p

import robosuite as suite
from robosuite.wrappers import GymWrapper

Activation = Union[str, nn.Module]

ROBOSUITE_ENVS = ["Reach", "Door", "Lift", "PickPlaceCan", 
    "PickPlaceBread", "PickPlaceMilk", "PickPlaceCereal", "Stack"]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(0.2, inplace=True),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

def build_mlp(
    input_size: int,
    output_size: int,
    n_layers: int,
    size: int,
    activation: Optional[Activation] = 'relu',
    output_activation: Optional[Activation] = 'identity',
    spectral_norm: Optional[bool] = False,
    batch_norm: Optional[bool] = False
):
    """
        Builds a feedforward neural network
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer
            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer
        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layer = nn.Linear(in_size, size)
        if spectral_norm:
            layer = p.spectral_norm(layer)
        layers.append(layer)
        layers.append(activation)
        if batch_norm:
            layer = nn.BatchNorm1d(size)
            layers.append(layer)
        in_size = size
    layer = nn.Linear(in_size, output_size)
    if spectral_norm:
        layer = p.spectral_norm(layer)
    layers.append(layer)
    layers.append(output_activation)
    return nn.Sequential(*layers)

def make_robosuite_env(
    env_name, 
    robots="Panda", 
    controller_type='OSC_POSE', 
    render=False,
    offscreen_render=False,
    **kwargs
):
    controller_configs = suite.load_controller_config(default_controller=controller_type)
    env = suite.make(
        env_name=env_name, # try with other tasks like "Stack" and "Door"
        robots=robots,  # try with other robots like "Sawyer" and "Jaco"
        reward_shaping=True,
        has_renderer=render,
        has_offscreen_renderer=offscreen_render,
        use_camera_obs=offscreen_render,
        use_object_obs=True,
        controller_configs=controller_configs,
        initialization_noise=None,
        **kwargs,
    )
    env._max_episode_steps = env.horizon
    return env

def make(
    env_name, 
    robots="Panda", 
    controller_type='OSC_POSE', 
    obs_keys=None, 
    render=False,
    seed=1,
    **kwargs
):
    assert env_name in ROBOSUITE_ENVS, f'Task {env_name} not supported yet ...'
    env = make_robosuite_env(
        env_name, 
        robots=robots, 
        controller_type=controller_type, 
        render=render,
        **kwargs
    )
    if obs_keys is None:
        obs_keys = [
            'robot0_eef_pos',
            'robot0_eef_quat',
            'robot0_gripper_qpos',
            'object-state',
        ]

    env = GymWrapper(env, keys=obs_keys)
    env.seed(seed)
    return env


def load_episodes(directory, obs_keys, lat_obs_keys=None, capacity=None):
    # The returned directory from filenames to episodes is guaranteed to be in
    # temporally sorted order.
    filenames = sorted(directory.glob('*.npz'))
    if capacity:
        num_steps = 0
        num_episodes = 0
        for filename in reversed(filenames):
            length = int(str(filename).split('-')[-1][:-4])
            num_steps += length
            num_episodes += 1
            if num_steps >= capacity:
                break
        filenames = filenames[-num_episodes:]
    episodes = []
    for filename in filenames:
        try:
            with filename.open('rb') as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}
        except Exception as e:
            print(f'Could not load episode {str(filename)}: {e}')
            continue

        obs = np.concatenate([episode[k] for k in obs_keys], axis=-1)
        episode['obs'] = obs[:-1]
        episode['next_obs'] = obs[1:]

        if lat_obs_keys is not None:
            lat_obs = np.concatenate([episode[k] for k in lat_obs_keys], axis=-1)
            episode['lat_obs'] = lat_obs[:-1]
            episode['lat_next_obs'] = lat_obs[1:]
        episodes.append(episode)

    returns = [sum(ep['reward']) for ep in episodes]
    print(f"Loaded {len(returns)} episodes from {str(directory)}")
    print(f"Average return {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
    return episodes

def evaluate(env, agent, num_episodes, L, step):
    ret = []
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.sample_action(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward   

        ret.append(episode_reward)
    L.add_scalar('eval/episode_reward_mean', np.mean(ret), step)
    L.add_scalar('eval/episode_reward_std', np.std(ret), step)
    L.flush()

