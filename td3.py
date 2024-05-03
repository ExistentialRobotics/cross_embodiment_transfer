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


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, state_dim, action_dim, n_layers, hidden_dim):
        super().__init__()

        self.trunk = utils.build_mlp(state_dim+action_dim, 1, n_layers, hidden_dim)
        self.apply(utils.weight_init)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        return self.trunk(state_action)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_layers, hidden_dim):
        super().__init__()

        self.Q1 = QFunction(state_dim, action_dim, n_layers, hidden_dim)
        self.Q2 = QFunction(state_dim, action_dim, n_layers, hidden_dim)

    def forward(self, state, action):
        q1 = self.Q1(state, action)
        q2 = self.Q2(state, action)
        return q1, q2

class TD3Agent:
    """TD3 algorithm."""
    def __init__(
        self,
        obs_dim,
        act_dim,
        device,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        expl_noise=0.1,
        n_layers=3,
        hidden_dim=256,
        lr=3e-4,
    ):

        self.device = device
        self.tau = tau
        self.discount = discount
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.expl_noise = expl_noise
        self.log_freq = 1000

        self.critic_update_freq = 1
        self.actor_update_freq = 2

        self.actor = Actor(obs_dim, act_dim, n_layers, hidden_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = Critic(obs_dim, act_dim, n_layers, hidden_dim).to(device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Target actor critic
        self.actor_target = Actor(obs_dim, act_dim, n_layers, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())       
        self.critic_target = Critic(obs_dim, act_dim, n_layers, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())


    def sample_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().to(self.device)
            obs = obs.unsqueeze(0)
            act = self.actor(obs)

        act = act.cpu().data.numpy().flatten()
        if not deterministic:
            act += np.random.normal(0, self.expl_noise, size=act.shape[0])
            act = np.clip(act, -1, 1)
        return act

    def predict(self, obs):
        return self.sample_action(obs, deterministic=True), None

    def update_actor_critic(self, obs, act, reward, next_obs, not_done, L, step):

        # Optimize critic
        if self.critic_update_freq > 0 and step % self.critic_update_freq == 0:
            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (
                    torch.randn_like(act) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                
                next_act = self.actor_target(next_obs)
                next_act = (next_act + noise).clamp(-1, 1)  

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_obs, next_act)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + not_done * self.discount * target_Q

            current_Q1, current_Q2 = self.critic(obs, act)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            if step % self.log_freq == 0:
                L.add_scalar('train_critic/critic_loss', critic_loss.item(), step)
                L.add_scalar('train_critic/Q1', current_Q1.mean().item(), step)
                L.add_scalar('train_critic/Q2', current_Q2.mean().item(), step)
                L.add_scalar('train_critic/target_Q', target_Q.mean().item(), step)
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

        # Optimize actor
        if self.actor_update_freq > 0 and step % self.actor_update_freq == 0:
            pi = self.actor(obs)
            actor_loss = -self.critic(obs, pi)[0].mean()

            if step % self.log_freq == 0:
                L.add_scalar('train_actor/actor_loss', actor_loss.item(), step)
                L.add_scalar('train_actor/pi_norm', (pi**2).mean().item(), step)
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update(self, replay_buffer, L, step):
        obs, act, rew, next_obs, not_done = replay_buffer.sample()

        if step % self.log_freq == 0:
            L.add_scalar('train/batch_reward', rew.mean().item(), step)
        self.update_actor_critic(obs, act, rew, next_obs, not_done, L, step)

    def save(self, model_dir):
        torch.save(self.critic.state_dict(), f'{model_dir}/critic.pt')
        torch.save(self.actor.state_dict(), f'{model_dir}/actor.pt')

    def load(self, model_dir):
        self.critic.load_state_dict(torch.load(f'{model_dir}/critic.pt'))
        self.critic_target.load_state_dict(torch.load(f'{model_dir}/critic.pt'))
        self.actor.load_state_dict(torch.load(f'{model_dir}/actor.pt'))
        self.actor_target.load_state_dict(torch.load(f'{model_dir}/actor.pt'))


class TD3ObsAgent(TD3Agent):
    """TD3+AE algorithm."""
    def __init__(
        self,
        obs_dims,
        act_dims,
        device,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        expl_noise=0.1,
        n_layers=3,
        hidden_dim=256,
        lr=3e-4,
    ):

        self.device = device
        self.tau = tau
        self.discount = discount
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.expl_noise = expl_noise

        self.critic_update_freq = 1
        self.actor_update_freq = 2
        self.dyn_cons_update_freq = 1
        self.log_freq = 1000
        self.batch_norm = True
        
        self.obs_dim = obs_dims['obs_dim']
        self.robot_obs_dim = obs_dims['robot_obs_dim']
        self.obj_obs_dim = obs_dims['obj_obs_dim']
        self.lat_obs_dim = obs_dims['lat_obs_dim']
        self.act_dim = act_dims['act_dim']

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

        self.critic = Critic(self.obs_dim, self.act_dim, n_layers, hidden_dim).to(device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Target actor critic
        self.actor_target = Actor(self.lat_obs_dim+self.obj_obs_dim, self.act_dim, n_layers, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())       
        self.critic_target = Critic(self.obs_dim, self.act_dim, n_layers, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

    def sample_action(self, obs, deterministic=False):
        if self.batch_norm:
            self.obs_enc.eval()
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

        self.obs_enc.train()
        return act

    def update_actor_critic(self, obs, act, reward, next_obs, not_done, L, step):

        # Optimize critic
        if self.critic_update_freq > 0 and step % self.critic_update_freq == 0:
            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (
                    torch.randn_like(act) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                
                robot_obs, obj_obs = next_obs[:, :self.robot_obs_dim], next_obs[:, self.robot_obs_dim:]
                lat_next_obs = self.obs_enc(robot_obs)
                next_act = self.actor_target(torch.cat([lat_next_obs, obj_obs], dim=-1))
                next_act = (next_act + noise).clamp(-1, 1)  

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_obs, next_act)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + not_done * self.discount * target_Q

            current_Q1, current_Q2 = self.critic(obs, act)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            if step % self.log_freq == 0:
                L.add_scalar('train_critic/critic_loss', critic_loss.item(), step)
                L.add_scalar('train_critic/Q1', current_Q1.mean().item(), step)
                L.add_scalar('train_critic/Q2', current_Q2.mean().item(), step)
                L.add_scalar('train_critic/target_Q', target_Q.mean().item(), step)
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

        # Optimize actor
        # Update obs_encoder thru actor
        if self.actor_update_freq > 0 and step % self.actor_update_freq == 0:
            robot_obs, obj_obs = obs[:, :self.robot_obs_dim], obs[:, self.robot_obs_dim:]
            lat_obs = self.obs_enc(robot_obs)
            pi = self.actor(torch.cat([lat_obs, obj_obs], dim=-1))
            actor_loss = -self.critic(obs, pi)[0].mean()

            if step % self.log_freq == 0:
                L.add_scalar('train_actor/actor_loss', actor_loss.item(), step)
                L.add_scalar('train_actor/lat_obs_norm_sq', (lat_obs**2).mean().item(), step)
                L.add_scalar('train_actor/pi_norm_sq', (pi**2).mean().item(), step)       
            self.actor_opt.zero_grad()
            self.obs_enc_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            self.obs_enc_opt.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


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
        self.update_actor_critic(obs, act, rew, next_obs, not_done, L, step)

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



class TD3ObsActAgent(TD3Agent):
    """TD3+AE algorithm."""
    def __init__(
        self,
        obs_dims,
        act_dims,
        device,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        expl_noise=0.1,
        n_layers=3,
        hidden_dim=256,
        lr=3e-4,
    ):
        super().__init__(obs_dims, act_dims, device, n_layers=n_layers,
            hidden_dim=hidden_dim, lr=lr)

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

        self.critic = Critic(self.obs_dim, self.act_dim, n_layers, hidden_dim).to(device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Target actor critic
        self.actor_target = Actor(self.lat_obs_dim+self.obj_obs_dim, self.lat_act_dim+1, n_layers, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())       
        self.critic_target = Critic(self.obs_dim, self.act_dim, n_layers, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())


    def sample_action(self, obs, deterministic=False):
        if self.batch_norm:
            self.obs_enc.eval()
            self.act_dec.eval()

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

        if self.batch_norm:
            self.obs_enc.train()
            self.act_dec.train()
        return act

    def update_actor_critic(self, obs, act, reward, next_obs, not_done, L, step):

        # Optimize critic
        if self.critic_update_freq > 0 and step % self.critic_update_freq == 0:
            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (
                    torch.randn_like(act) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                
                robot_obs, obj_obs = next_obs[:, :self.robot_obs_dim], next_obs[:, self.robot_obs_dim:]
                lat_next_obs = self.obs_enc(robot_obs)
                next_lat_act = self.actor_target(torch.cat([lat_next_obs, obj_obs], dim=-1))
                next_lat_act, gripper_act = next_lat_act[:, :-1], next_lat_act[:, -1].reshape(-1, 1)
                next_act = self.act_dec(torch.cat([robot_obs, next_lat_act], dim=-1))

                next_act = torch.cat([next_act, gripper_act], dim=-1)
                next_act = (next_act + noise).clamp(-1, 1)  

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_obs, next_act)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + not_done * self.discount * target_Q

            current_Q1, current_Q2 = self.critic(obs, act)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            if step % self.log_freq == 0:
                L.add_scalar('train_critic/critic_loss', critic_loss.item(), step)
                L.add_scalar('train_critic/Q1', current_Q1.mean().item(), step)
                L.add_scalar('train_critic/Q2', current_Q2.mean().item(), step)
                L.add_scalar('train_critic/target_Q', target_Q.mean().item(), step)
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

        # Optimize actor
        # Update obs_encoder thru actor
        if self.actor_update_freq > 0 and step % self.actor_update_freq == 0:
            robot_obs, obj_obs = obs[:, :self.robot_obs_dim], obs[:, self.robot_obs_dim:]
            lat_obs = self.obs_enc(robot_obs)
            lat_act = self.actor(torch.cat([lat_obs, obj_obs], dim=-1))
            lat_act, gripper_act = lat_act[:, :-1], lat_act[:, -1].reshape(-1, 1)
            pi = self.act_dec(torch.cat([robot_obs, lat_act], dim=-1))
            pi = torch.cat([pi, gripper_act], dim=-1)
            actor_loss = -self.critic(obs, pi)[0].mean()

            if step % self.log_freq == 0:
                L.add_scalar('train_actor/actor_loss', actor_loss.item(), step)
                L.add_scalar('train_actor/lat_obs_sq', (lat_obs**2).mean().item(), step)
                L.add_scalar('train_actor/lat_act_sq', (lat_act**2).mean().item(), step)
                L.add_scalar('train_actor/pi_sq', (pi**2).mean().item(), step)       
            self.actor_opt.zero_grad()
            self.obs_enc_opt.zero_grad()
            self.act_dec_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            self.obs_enc_opt.step()
            self.act_dec_opt.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def update_dyn_cons(self, obs, act, next_obs, L, step):

        robot_obs, robot_next_obs = obs[:, :self.robot_obs_dim], next_obs[:, :self.robot_obs_dim]
        act = act[:,:-1]        # Remove gripper action
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

            obs_diff = (robot_obs-robot_next_obs)**2
            lat_obs_diff = (lat_obs-lat_next_obs)**2
            L.add_scalar('train_dyn_cons/obs_diff', obs_diff.mean().item(), step)
            L.add_scalar('train_dyn_cons/lat_obs_diff', lat_obs_diff.mean().item(), step)


    def update(self, replay_buffer, L, step):
        obs, act, rew, next_obs, not_done = replay_buffer.sample()
        if step % self.log_freq == 0:
            L.add_scalar('train/batch_reward', rew.mean().item(), step)
        if step % self.dyn_cons_update_freq == 0: 
            self.update_dyn_cons(obs, act, next_obs, L, step)
        self.update_actor_critic(obs, act, rew, next_obs, not_done, L, step)

    def save(self, model_dir):
        super().save(model_dir)
        torch.save(self.obs_enc.state_dict(), f'{model_dir}/obs_enc.pt')        
        torch.save(self.obs_dec.state_dict(), f'{model_dir}/obs_dec.pt')
        torch.save(self.act_enc.state_dict(), f'{model_dir}/act_enc.pt')        
        torch.save(self.act_dec.state_dict(), f'{model_dir}/act_dec.pt')
        torch.save(self.inv_dyn.state_dict(), f'{model_dir}/inv_dyn.pt')
        torch.save(self.fwd_dyn.state_dict(), f'{model_dir}/fwd_dyn.pt')

    def load(self, model_dir):
        super().load(model_dir)
        self.obs_enc.load_state_dict(torch.load(f'{model_dir}/obs_enc.pt'))
        self.obs_dec.load_state_dict(torch.load(f'{model_dir}/obs_dec.pt'))
        self.act_enc.load_state_dict(torch.load(f'{model_dir}/act_enc.pt'))
        self.act_dec.load_state_dict(torch.load(f'{model_dir}/act_dec.pt'))
        self.inv_dyn.load_state_dict(torch.load(f'{model_dir}/inv_dyn.pt'))
        self.fwd_dyn.load_state_dict(torch.load(f'{model_dir}/fwd_dyn.pt'))

