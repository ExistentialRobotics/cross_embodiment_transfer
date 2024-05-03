import os
import numpy as np
import torch
from collections import deque

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, idxs=None, return_idxs=False):

        if isinstance(self.obses, np.ndarray):
            if idxs is None:
                idxs = np.random.randint(
                    0, self.capacity if self.full else self.idx, size=self.batch_size
                )   

            obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
            actions = torch.as_tensor(self.actions[idxs], device=self.device)
            rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
            next_obses = torch.as_tensor(
                self.next_obses[idxs], device=self.device
            ).float()
            not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        else:

            idxs = np.random.randint(
                0, self.capacity if self.full else self.idx, size=self.batch_size
            )
            obses = self.obses[idxs]
            actions = self.actions[idxs]
            rewards = self.rewards[idxs]
            next_obses = self.next_obses[idxs]
            not_dones = self.not_dones[idxs]

        return obses, actions, rewards, next_obses, not_dones


    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start, f"{start}, {end}, {self.idx}"            
            try:
                self.obses[start:end] = payload[0]
            except ValueError:
                print(f"{start}, {end}, {self.idx}")
                import ipdb; ipdb.set_trace()
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end


    def add_rollouts(self, episodes):
        for ep in episodes:
            T = len(ep['action'])
            start, end = self.idx, self.idx + T
            self.obses[start:end] = ep['obs']
            self.next_obses[start:end] = ep['next_obs']
            self.actions[start:end] = ep['action']
            self.rewards[start:end] = ep['reward'].reshape(-1, 1)
            self.not_dones[start:end] = np.ones((T, 1))
            self.idx = end

    def move_to_device(self, device):
        """
        Move the full dataset to device
        """
        self.obses = torch.from_numpy(self.obses).float().to(device)
        self.next_obses = torch.from_numpy(self.next_obses).float().to(device)
        self.actions = torch.from_numpy(self.actions).float().to(device)
        self.rewards = torch.from_numpy(self.rewards).float().to(device)
        self.not_dones = torch.from_numpy(self.not_dones).float().to(device)


