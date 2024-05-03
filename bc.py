import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_layers, hidden_dim):
        super().__init__()

        self.trunk = utils.build_mlp(state_dim, action_dim, 
            n_layers, hidden_dim, activation='relu', output_activation='tanh')
        self.apply(utils.weight_init)

    def forward(self, state):
        h = self.trunk(state)
        return h

class BCAgent:
    """Behavior cloning"""
    def __init__(self, obs_dims, act_dims, device, n_layers=3, hidden_dim=256, lr=3e-4):
        self.log_freq = 1000
        self.expl_noise = 0.1
        self.device = device
        self.dyn_cons_update_freq = 1

        self.obs_dim = obs_dims['obs_dim']
        self.act_dim = act_dims['act_dim']

        self.actor = Actor(self.obs_dim, self.act_dim, n_layers, hidden_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        self.modules = [self.actor]

    def sample_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().to(self.device)
            obs = obs.unsqueeze(0)
            act = self.actor(obs)

        act = act.cpu().data.numpy().flatten()
        if not deterministic:
            act += np.random.normal(0, 0.1, size=act.shape[0])
            act = np.clip(act, -1, 1)
        return act

    def update_actor(self, obs, act, L, step):
        pred_act = self.actor(obs)
        loss = F.mse_loss(pred_act, act)

        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()

        if step % self.log_freq == 0:
            L.add_scalar('train_actor/actor_loss', loss.item(), step)

    def update(self, replay_buffer, L, step):
        obs, act, rew, next_obs, not_done = replay_buffer.sample()
        if step % self.log_freq == 0:
            L.add_scalar('train/batch_reward', rew.mean().item(), step)
        self.update_actor(obs, act, L, step)

    def save(self, model_dir):
        torch.save(self.actor.state_dict(), f'{model_dir}/actor.pt')

    def load(self, model_dir):
        self.actor.load_state_dict(torch.load(f'{model_dir}/actor.pt'))

    def eval_mode(self):
        for m in self.modules:
            m.eval()

    def train_mode(self):
        for m in self.modules:
            m.train()

class BCObsAgent(BCAgent):
    def __init__(self, obs_dims, act_dims, device, n_layers=3, hidden_dim=256, lr=3e-4):
        super().__init__(obs_dims, act_dims, device, n_layers=n_layers, 
            hidden_dim=hidden_dim, lr=lr)
        self.batch_norm = False

        self.robot_obs_dim = obs_dims['robot_obs_dim']
        self.obj_obs_dim = obs_dims['obj_obs_dim']
        self.lat_obs_dim = obs_dims['lat_obs_dim']

        self.obs_enc = utils.build_mlp(self.robot_obs_dim, self.lat_obs_dim, n_layers, hidden_dim,
            activation='relu', output_activation='tanh', batch_norm=self.batch_norm).to(device)
        self.obs_enc_opt = torch.optim.Adam(self.obs_enc.parameters(), lr=3e-4)

        self.obs_dec = utils.build_mlp(self.lat_obs_dim, self.robot_obs_dim, n_layers, hidden_dim,
            activation='relu', output_activation='identity', batch_norm=self.batch_norm).to(device)
        self.obs_dec_opt = torch.optim.Adam(self.obs_dec.parameters(), lr=3e-4)
        
        self.inv_dyn = utils.build_mlp(self.lat_obs_dim*2, self.act_dim-1, 
            n_layers, hidden_dim, activation='relu', output_activation='identity', batch_norm=self.batch_norm).to(device)
        self.inv_dyn_opt = torch.optim.Adam(self.inv_dyn.parameters(), lr=3e-4)

        self.fwd_dyn = utils.build_mlp(self.lat_obs_dim+self.act_dim-1, self.lat_obs_dim, 
            n_layers, hidden_dim, activation='relu', output_activation='identity', batch_norm=self.batch_norm).to(device)
        self.fwd_dyn_opt = torch.optim.Adam(self.fwd_dyn.parameters(), lr=3e-4)

        self.actor = Actor(self.lat_obs_dim+self.obj_obs_dim, self.act_dim, n_layers, hidden_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.modules = [self.obs_enc, self.obs_dec, self.inv_dyn, self.fwd_dyn, self.actor]

    def sample_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().to(self.device)
            obs = obs.unsqueeze(0)
            robot_obs, obj_obs = obs[:, :self.robot_obs_dim], obs[:, self.robot_obs_dim:]
            lat_obs = self.obs_enc(robot_obs)
            act = self.actor(torch.cat([lat_obs, obj_obs], dim=-1))

        act = act.cpu().data.numpy().flatten()
        if not deterministic:
            act += np.random.normal(0, self.expl_noise, size=act.shape[0])
            act = np.clip(act, -1, 1)
        return act

    def update_actor(self, obs, act, L, step):

        robot_obs, obj_obs = obs[:, :self.robot_obs_dim], obs[:, self.robot_obs_dim:]
        lat_obs = self.obs_enc(robot_obs)
        pred_act = self.actor(torch.cat([lat_obs, obj_obs], dim=-1))
        loss = F.mse_loss(pred_act, act)

        self.actor_opt.zero_grad()
        self.obs_enc_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()
        self.obs_enc_opt.step()

        if step % self.log_freq == 0:
            L.add_scalar('train_actor/actor_loss', loss.item(), step)    

    def update_dyn_cons(self, obs, act, next_obs, L, step):

        robot_obs, robot_next_obs = obs[:, :self.robot_obs_dim], next_obs[:, :self.robot_obs_dim]
        act = act[:,:-1]
        lat_obs = self.obs_enc(robot_obs)
        lat_next_obs = self.obs_enc(robot_next_obs)
    
        pred_act = self.inv_dyn(torch.cat([lat_obs, lat_next_obs], dim=-1))
        inv_loss = F.mse_loss(pred_act, act)

        pred_next_obs = self.fwd_dyn(torch.cat([lat_obs, act], dim=-1))
        fwd_loss = F.mse_loss(pred_next_obs, lat_next_obs)    

        recon_robot_obs = self.obs_dec(lat_obs)
        recon_loss = F.mse_loss(recon_robot_obs, robot_obs)

        loss = fwd_loss + inv_loss*10 + recon_loss

        self.obs_enc_opt.zero_grad()
        self.obs_dec_opt.zero_grad()
        self.inv_dyn_opt.zero_grad()
        self.fwd_dyn_opt.zero_grad()
        loss.backward()
        self.obs_enc_opt.step()
        self.obs_dec_opt.step()
        self.inv_dyn_opt.step()    
        self.fwd_dyn_opt.step()

        if step % self.log_freq == 0:
            L.add_scalar('train_dyn_cons/inv_loss', inv_loss.item(), step)
            L.add_scalar('train_dyn_cons/fwd_loss', fwd_loss.item(), step)
            L.add_scalar('train_dyn_cons/recon_loss', recon_loss.item(), step)
            L.add_scalar('train_dyn_cons/lat_obs_sq', (lat_obs**2).mean().item(), step)
            L.add_scalar('train_dyn_cons/pred_act_sq', (pred_act**2).mean().item(), step)


    def update(self, replay_buffer, L, step):
        obs, act, rew, next_obs, not_done = replay_buffer.sample()
        if step % self.log_freq == 0:
            L.add_scalar('train/batch_reward', rew.mean().item(), step)
        if step % self.dyn_cons_update_freq == 0: 
            self.update_dyn_cons(obs, act, next_obs, L, step)
        self.update_actor(obs, act, L, step)


    def save(self, model_dir):
        super().save(model_dir)
        torch.save(self.obs_enc.state_dict(), f'{model_dir}/obs_enc.pt')        
        torch.save(self.obs_dec.state_dict(), f'{model_dir}/obs_dec.pt')
        torch.save(self.inv_dyn.state_dict(), f'{model_dir}/inv_dyn.pt')
        torch.save(self.fwd_dyn.state_dict(), f'{model_dir}/fwd_dyn.pt')

    def load(self, model_dir):
        super().load(model_dir)
        self.obs_enc.load_state_dict(torch.load(f'{model_dir}/obs_enc.pt'))
        self.obs_dec.load_state_dict(torch.load(f'{model_dir}/obs_dec.pt'))
        self.inv_dyn.load_state_dict(torch.load(f'{model_dir}/inv_dyn.pt'))
        self.fwd_dyn.load_state_dict(torch.load(f'{model_dir}/fwd_dyn.pt'))

class BCObsActAgent(BCObsAgent):
    def __init__(self, obs_dims, act_dims, device, n_layers=3, hidden_dim=256, lr=3e-4):
        super().__init__(obs_dims, act_dims, device, 
            n_layers=n_layers, hidden_dim=hidden_dim, lr=lr)

        self.lat_act_dim = act_dims['lat_act_dim']

        self.obs_enc = utils.build_mlp(self.robot_obs_dim, self.lat_obs_dim, n_layers, hidden_dim,
            activation='relu', output_activation='tanh', batch_norm=self.batch_norm).to(device)
        self.obs_enc_opt = torch.optim.Adam(self.obs_enc.parameters(), lr=lr)

        self.obs_dec = utils.build_mlp(self.lat_obs_dim, self.robot_obs_dim, n_layers, hidden_dim,
            activation='relu', output_activation='identity', batch_norm=self.batch_norm).to(device)
        self.obs_dec_opt = torch.optim.Adam(self.obs_dec.parameters(), lr=lr)
        
        self.act_enc = utils.build_mlp(self.robot_obs_dim+self.act_dim-1, self.lat_act_dim, n_layers, 
            hidden_dim, activation='relu', output_activation='tanh', batch_norm=self.batch_norm).to(device)
        self.act_enc_opt= torch.optim.Adam(self.act_enc.parameters(), lr=lr)

        self.act_dec = utils.build_mlp(self.robot_obs_dim+self.lat_act_dim, self.act_dim-1, n_layers,
            hidden_dim, activation='relu', output_activation='tanh', batch_norm=self.batch_norm).to(device)
        self.act_dec_opt = torch.optim.Adam(self.act_dec.parameters(), lr=lr)

        self.inv_dyn = utils.build_mlp(self.lat_obs_dim*2, self.lat_act_dim, n_layers, hidden_dim, 
            activation='relu', output_activation='tanh', batch_norm=self.batch_norm).to(device)
        self.inv_dyn_opt = torch.optim.Adam(self.inv_dyn.parameters(), lr=lr)

        self.fwd_dyn = utils.build_mlp(self.lat_obs_dim+self.lat_act_dim, self.lat_obs_dim, 
            n_layers, hidden_dim, activation='relu', output_activation='tanh', batch_norm=self.batch_norm).to(device)
        self.fwd_dyn_opt = torch.optim.Adam(self.fwd_dyn.parameters(), lr=lr)

        # One more dim for gripper action
        self.actor = Actor(self.lat_obs_dim+self.obj_obs_dim, self.lat_act_dim+1, n_layers, hidden_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.modules = [self.obs_enc, self.obs_dec, self.act_enc, 
            self.act_dec, self.inv_dyn, self.fwd_dyn, self.actor]
    
    def sample_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().to(self.device)
            obs = obs.unsqueeze(0)
            robot_obs, obj_obs = obs[:, :self.robot_obs_dim], obs[:, self.robot_obs_dim:]
            lat_obs = self.obs_enc(robot_obs)
            lat_act = self.actor(torch.cat([lat_obs, obj_obs], dim=-1))
            lat_act, gripper_act = lat_act[:, :-1], lat_act[:, -1].reshape(-1, 1)
            act = self.act_dec(torch.cat([robot_obs, lat_act], dim=-1))
            act = torch.cat([act, gripper_act], dim=-1)
        act = act.cpu().data.numpy().flatten()
        if not deterministic:
            act += np.random.normal(0, self.expl_noise, size=act.shape[0])
            act = np.clip(act, -1, 1)
        return act



    def update_actor(self, obs, act, L, step):

        robot_obs, obj_obs = obs[:, :self.robot_obs_dim], obs[:, self.robot_obs_dim:]
        lat_obs = self.obs_enc(robot_obs)
        lat_act = self.actor(torch.cat([lat_obs, obj_obs], dim=-1))
        lat_act, gripper_act = lat_act[:, :-1], lat_act[:, -1].reshape(-1, 1)
        pred_act = self.act_dec(torch.cat([robot_obs, lat_act], dim=-1))
        pred_act = torch.cat([pred_act, gripper_act], dim=-1)
        loss = F.mse_loss(pred_act, act)

        self.actor_opt.zero_grad()
        self.obs_enc_opt.zero_grad()
        self.act_dec_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()
        self.obs_enc_opt.step()
        self.act_dec_opt.step()

        if step % self.log_freq == 0:
            L.add_scalar('train_actor/actor_loss', loss.item(), step)    

    def update_dyn_cons(self, obs, act, next_obs, L, step):

        robot_obs, robot_next_obs = obs[:, :self.robot_obs_dim], next_obs[:, :self.robot_obs_dim]
        act = act[:,:-1]
        lat_obs = self.obs_enc(robot_obs)
        lat_next_obs = self.obs_enc(robot_next_obs)
    
        pred_lat_act = self.inv_dyn(torch.cat([lat_obs, lat_next_obs], dim=-1))
        pred_act = self.act_dec(torch.cat([robot_obs, pred_lat_act], dim=-1))
        inv_loss = F.mse_loss(pred_act, act)

        lat_act = self.act_enc(torch.cat([robot_obs, act], dim=-1))
        pred_next_obs = self.fwd_dyn(torch.cat([lat_obs, lat_act], dim=-1))
        fwd_loss = F.mse_loss(pred_next_obs, lat_next_obs)    

        recon_robot_obs = self.obs_dec(lat_obs)
        recon_obs_loss = F.mse_loss(recon_robot_obs, robot_obs)
        recon_act = self.act_dec(torch.cat([robot_obs, lat_act], dim=-1))
        recon_act_loss = F.mse_loss(recon_act, act)

        loss = fwd_loss + inv_loss + recon_obs_loss + recon_act_loss

        self.obs_enc_opt.zero_grad()
        self.obs_dec_opt.zero_grad()
        self.act_enc_opt.zero_grad()
        self.act_dec_opt.zero_grad()
        self.inv_dyn_opt.zero_grad()
        self.fwd_dyn_opt.zero_grad()
        loss.backward()
        self.obs_enc_opt.step()
        self.obs_dec_opt.step()
        self.act_enc_opt.step()
        self.act_dec_opt.step()
        self.inv_dyn_opt.step()    
        self.fwd_dyn_opt.step()

        if step % self.log_freq == 0:
            L.add_scalar('train_dyn_cons/inv_loss', inv_loss.item(), step)
            L.add_scalar('train_dyn_cons/fwd_loss', fwd_loss.item(), step)
            L.add_scalar('train_dyn_cons/recon_obs_loss', recon_obs_loss.item(), step)
            L.add_scalar('train_dyn_cons/recon_act_loss', recon_act_loss.item(), step)
            L.add_scalar('train_dyn_cons/lat_obs_sq', (lat_obs**2).mean().item(), step)
            L.add_scalar('train_dyn_cons/lat_act_sq', (lat_act**2).mean().item(), step)
            L.add_scalar('train_dyn_cons/pred_act_sq', (pred_act**2).mean().item(), step)


    def update(self, replay_buffer, L, step):
        obs, act, rew, next_obs, not_done = replay_buffer.sample()
        if step % self.log_freq == 0:
            L.add_scalar('train/batch_reward', rew.mean().item(), step)
        if step % self.dyn_cons_update_freq == 0: 
            self.update_dyn_cons(obs, act, next_obs, L, step)
        self.update_actor(obs, act, L, step)

    def save(self, model_dir):
        super().save(model_dir)
        torch.save(self.act_enc.state_dict(), f'{model_dir}/act_enc.pt')        
        torch.save(self.act_dec.state_dict(), f'{model_dir}/act_dec.pt')

    def load(self, model_dir):
        super().load(model_dir)
        self.act_enc.load_state_dict(torch.load(f'{model_dir}/act_enc.pt'))
        self.act_dec.load_state_dict(torch.load(f'{model_dir}/act_dec.pt'))

