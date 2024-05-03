import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from td3 import Actor

class Agent:
    """Base class for adaptation"""
    def __init__(self, obs_dims, act_dims, device):
        self.obs_dim = obs_dims['obs_dim']
        self.robot_obs_dim = obs_dims['robot_obs_dim']
        self.obj_obs_dim = obs_dims['obj_obs_dim']
        self.act_dim = act_dims['act_dim']
        self.device = device
        self.batch_norm = False
        self.modules = []

        assert self.obs_dim == self.robot_obs_dim + self.obj_obs_dim

    def eval_mode(self):
        for m in self.modules:
            m.eval()

    def train_mode(self):
        for m in self.modules:
            m.train()

    def freeze(self):
        for m in self.modules:
            for p in m.parameters():
                p.requires_grad = False


class ObsAgent(Agent):
    def __init__(self, obs_dims, act_dims, device, n_layers=3, hidden_dim=256):
        super().__init__(obs_dims, act_dims, device)
        self.lat_obs_dim = obs_dims['lat_obs_dim']

        self.obs_enc = utils.build_mlp(self.robot_obs_dim, self.lat_obs_dim, n_layers, hidden_dim,
            activation='relu', output_activation='tanh', batch_norm=self.batch_norm).to(self.device)
        self.obs_dec = utils.build_mlp(self.lat_obs_dim, self.robot_obs_dim, n_layers, hidden_dim,
            activation='relu', output_activation='identity', batch_norm=self.batch_norm).to(self.device)
        self.inv_dyn = utils.build_mlp(self.lat_obs_dim*2, self.act_dim-1, n_layers, hidden_dim, 
            activation='relu', output_activation='identity', batch_norm=self.batch_norm).to(self.device)
        self.fwd_dyn = utils.build_mlp(self.lat_obs_dim+self.act_dim-1, self.lat_obs_dim, n_layers, hidden_dim, 
            activation='relu', output_activation='identity', batch_norm=self.batch_norm).to(self.device)
        self.actor = Actor(self.lat_obs_dim+self.obj_obs_dim, self.act_dim, n_layers, hidden_dim).to(self.device)

        self.modules = [self.obs_enc, self.obs_dec, self.inv_dyn, self.fwd_dyn, self.actor]

    def save(self, model_dir):
        torch.save(self.actor.state_dict(), f'{model_dir}/actor.pt')
        torch.save(self.obs_enc.state_dict(), f'{model_dir}/obs_enc.pt')        
        torch.save(self.obs_dec.state_dict(), f'{model_dir}/obs_dec.pt')
        torch.save(self.inv_dyn.state_dict(), f'{model_dir}/inv_dyn.pt')
        torch.save(self.fwd_dyn.state_dict(), f'{model_dir}/fwd_dyn.pt')

    def load(self, model_dir):
        self.obs_enc.load_state_dict(torch.load(model_dir/'obs_enc.pt'))
        self.obs_dec.load_state_dict(torch.load(model_dir/'obs_dec.pt'))
        self.fwd_dyn.load_state_dict(torch.load(model_dir/'fwd_dyn.pt'))
        self.inv_dyn.load_state_dict(torch.load(model_dir/'inv_dyn.pt'))
        self.actor.load_state_dict(torch.load(model_dir/'actor.pt'))

    def load_actor(self, model_dir):
        self.actor.load_state_dict(torch.load(model_dir/'actor.pt'))
        for p in self.actor.parameters():
            p.requires_grad = False

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


class ObsActAgent(ObsAgent):
    def __init__(self, obs_dims, act_dims, device, n_layers=3, hidden_dim=256):
        super().__init__(obs_dims, act_dims, device, n_layers=n_layers, hidden_dim=hidden_dim)

        self.lat_act_dim = act_dims['lat_act_dim']

        self.act_enc = utils.build_mlp(self.robot_obs_dim+self.act_dim-1, self.lat_act_dim, n_layers, 
            hidden_dim, activation='relu', output_activation='tanh', batch_norm=self.batch_norm).to(device)
        self.act_dec = utils.build_mlp(self.robot_obs_dim+self.lat_act_dim, self.act_dim-1, n_layers,
            hidden_dim, activation='relu', output_activation='tanh', batch_norm=self.batch_norm).to(device)
        self.inv_dyn = utils.build_mlp(self.lat_obs_dim*2, self.lat_act_dim, n_layers, hidden_dim, 
            activation='relu', output_activation='tanh', batch_norm=self.batch_norm).to(device)
        self.fwd_dyn = utils.build_mlp(self.lat_obs_dim+self.lat_act_dim, self.lat_obs_dim, 
            n_layers, hidden_dim, activation='relu', output_activation='tanh', batch_norm=self.batch_norm).to(device)
        self.actor = Actor(self.lat_obs_dim+self.obj_obs_dim, self.lat_act_dim+1, n_layers, hidden_dim).to(device)

        self.modules += [self.act_enc, self.act_dec]

    def save(self, model_dir):
        super().save(model_dir)
        torch.save(self.act_enc.state_dict(), f'{model_dir}/act_enc.pt')        
        torch.save(self.act_dec.state_dict(), f'{model_dir}/act_dec.pt')

    def load(self, model_dir):
        super().load(model_dir)
        self.act_enc.load_state_dict(torch.load(model_dir/'act_enc.pt'))
        self.act_dec.load_state_dict(torch.load(model_dir/'act_dec.pt'))

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


class ObsAligner:
    def __init__(
        self, 
        src_agent, 
        tgt_agent, 
        device, 
        n_layers=3, 
        hidden_dim=256,
        lr=3e-4,
        lmbd_gp=10,
        log_freq=1000
    ):
        
        self.device = device
        self.lmbd_gp = lmbd_gp
        self.lmbd_cyc = 10
        self.lmbd_dyn = 10
        self.log_freq = log_freq

        self.src_obs_enc = src_agent.obs_enc
        self.src_obs_dec = src_agent.obs_dec
        self.tgt_obs_enc = tgt_agent.obs_enc
        self.tgt_obs_dec = tgt_agent.obs_dec
        self.fwd_dyn = src_agent.fwd_dyn
        self.inv_dyn = src_agent.inv_dyn

        assert src_agent.lat_obs_dim == tgt_agent.lat_obs_dim
        self.lat_obs_dim = src_agent.lat_obs_dim
        self.src_obs_dim = src_agent.robot_obs_dim
        self.tgt_obs_dim = tgt_agent.robot_obs_dim

        self.lat_disc = utils.build_mlp(self.lat_obs_dim, 1, n_layers, hidden_dim,
            activation='leaky_relu', output_activation='identity').to(self.device)
        self.src_disc = utils.build_mlp(self.src_obs_dim, 1, n_layers, hidden_dim,
            activation='leaky_relu', output_activation='identity').to(self.device)
        self.tgt_disc = utils.build_mlp(self.tgt_obs_dim, 1, n_layers, hidden_dim,
            activation='leaky_relu', output_activation='identity').to(self.device)

        # Optimizers
        self.tgt_obs_enc_opt = torch.optim.Adam(self.tgt_obs_enc.parameters(), lr=lr)
        self.tgt_obs_dec_opt = torch.optim.Adam(self.tgt_obs_dec.parameters(), lr=lr)
        self.lat_disc_opt = torch.optim.Adam(self.lat_disc.parameters(), lr=lr)
        self.src_disc_opt = torch.optim.Adam(self.src_disc.parameters(), lr=lr)
        self.tgt_disc_opt = torch.optim.Adam(self.tgt_disc.parameters(), lr=lr)

    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((real_samples.size(0), 1)).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = torch.ones(real_samples.shape[0], 1, requires_grad=False, device=self.device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        # gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def update_disc(self, src_obs, src_act, tgt_obs, tgt_act, L, step):
        """
        Discriminator tries to separate source and target latent states and actions
        """
        
        fake_lat_obs = self.tgt_obs_enc(tgt_obs).detach()
        real_lat_obs = self.src_obs_enc(src_obs).detach()
        lat_disc_loss = self.lat_disc(fake_lat_obs).mean() - self.lat_disc(real_lat_obs).mean()

        fake_src_obs = self.src_obs_dec(fake_lat_obs).detach()
        src_disc_loss = self.src_disc(fake_src_obs).mean() - self.src_disc(src_obs).mean()
        
        real_tgt_obs = self.tgt_obs_dec(real_lat_obs).detach()
        tgt_disc_loss = self.tgt_disc(real_tgt_obs).mean() - self.tgt_disc(tgt_obs).mean()

        lat_gp = self.compute_gradient_penalty(self.lat_disc, real_lat_obs, fake_lat_obs)
        src_gp = self.compute_gradient_penalty(self.src_disc, src_obs, fake_src_obs)
        tgt_gp = self.compute_gradient_penalty(self.tgt_disc, tgt_obs, real_tgt_obs)

        disc_loss = lat_disc_loss + src_disc_loss + tgt_disc_loss + \
            self.lmbd_gp * (lat_gp + src_gp + tgt_gp)

        self.lat_disc_opt.zero_grad()
        self.src_disc_opt.zero_grad()
        self.tgt_disc_opt.zero_grad()
        disc_loss.backward()
        self.lat_disc_opt.step()
        self.src_disc_opt.step()
        self.tgt_disc_opt.step()

        if step % self.log_freq == 0:
            L.add_scalar('train_lat_disc/lat_disc_loss', lat_disc_loss.item(), step)
            L.add_scalar('train_lat_disc/lat_gp', lat_gp.item(), step)
            L.add_scalar('train_lat_disc/src_disc_loss', src_disc_loss.item(), step)
            L.add_scalar('train_lat_disc/src_gp', src_gp.item(), step)
            L.add_scalar('train_lat_disc/tgt_disc_loss', tgt_disc_loss.item(), step)
            L.add_scalar('train_lat_disc/tgt_gp', tgt_gp.item(), step)
            L.add_scalar('train_lat_disc/real_lat_obs_sq', (real_lat_obs**2).mean().item(), step)
            L.add_scalar('train_lat_disc/fake_lat_obs_sq', (fake_lat_obs**2).mean().item(), step)

    def update_gen(self, src_obs, src_act, src_next_obs, tgt_obs, tgt_act, tgt_next_obs, L, step):
        """
        Generator outputs more realistic latent states from target samples
        """

        # Generator
        fake_lat_obs = self.tgt_obs_enc(tgt_obs)
        lat_gen_loss = -self.lat_disc(fake_lat_obs).mean()

        fake_src_obs = self.src_obs_dec(fake_lat_obs)
        src_gen_loss = -self.src_disc(fake_src_obs).mean()

        real_tgt_obs = self.tgt_obs_dec(self.src_obs_enc(src_obs))
        tgt_gen_loss = -self.tgt_disc(real_tgt_obs).mean()

        # Cycle consistency
        pred_src_obs = self.src_obs_dec(self.tgt_obs_enc(real_tgt_obs))
        pred_tgt_obs = self.tgt_obs_dec(self.src_obs_enc(fake_src_obs))
        cycle_loss = F.l1_loss(pred_src_obs, src_obs) + F.l1_loss(pred_tgt_obs, tgt_obs)

        # Latent dynamics
        fake_lat_next_obs = self.tgt_obs_enc(tgt_next_obs)
        pred_act = self.inv_dyn(torch.cat([fake_lat_obs, fake_lat_next_obs], dim=-1))
        inv_loss = F.mse_loss(pred_act, tgt_act)
        pred_lat_next_obs = self.fwd_dyn(torch.cat([fake_lat_obs, tgt_act], dim=-1))
        fwd_loss = F.mse_loss(pred_lat_next_obs, fake_lat_next_obs)

        loss = lat_gen_loss + src_gen_loss + tgt_gen_loss + \
            self.lmbd_cyc * cycle_loss + self.lmbd_dyn * (inv_loss + fwd_loss)

        self.tgt_obs_enc_opt.zero_grad()
        self.tgt_obs_dec_opt.zero_grad()
        loss.backward()
        self.tgt_obs_enc_opt.step()
        self.tgt_obs_dec_opt.step()

        if step % self.log_freq == 0:
            L.add_scalar('train_lat_gen/lat_gen_loss', lat_gen_loss.item(), step)
            L.add_scalar('train_lat_gen/src_gen_loss', src_gen_loss.item(), step)
            L.add_scalar('train_lat_gen/tgt_gen_loss', tgt_gen_loss.item(), step)
            L.add_scalar('train_lat_gen/inv_loss', inv_loss.item(), step)
            L.add_scalar('train_lat_gen/fwd_loss', fwd_loss.item(), step)
            L.add_scalar('train_lat_gen/cycle_loss', cycle_loss.item(), step)
            L.add_scalar('train_lat_gen/lat_obs_diff', 
                F.l1_loss(self.src_obs_enc(tgt_obs), fake_lat_obs), step)
            src_lat_obs = self.src_obs_enc(src_obs)
            src_lat_next_obs = self.src_obs_enc(src_next_obs)
            pred_src_lat_next_obs = self.fwd_dyn(torch.cat([src_lat_obs, src_act], dim=-1))
            pred_src_act = self.inv_dyn(torch.cat([src_lat_obs, src_lat_next_obs], dim=-1))
            L.add_scalar('train_lat_gen/src_fwd_loss', 
                F.mse_loss(src_lat_next_obs, pred_src_lat_next_obs).item(), step)
            L.add_scalar('train_lat_gen/src_inv_loss',
                F.mse_loss(src_act, pred_src_act).item(), step)


class ObsActAligner(ObsAligner):
    def __init__(
        self, 
        src_agent, 
        tgt_agent, 
        device, 
        n_layers=3, 
        hidden_dim=256,
        lr=3e-4,
        lmbd_gp=10,
        log_freq=1000
    ):
        super().__init__(src_agent, tgt_agent, device, n_layers=n_layers, 
            hidden_dim=hidden_dim, lr=lr, lmbd_gp=lmbd_gp, log_freq=log_freq)

        assert src_agent.lat_act_dim == tgt_agent.lat_act_dim
        self.lat_act_dim = src_agent.lat_act_dim
        self.src_act_dim = src_agent.act_dim - 1
        self.tgt_act_dim = tgt_agent.act_dim - 1

        self.src_act_enc = src_agent.act_enc 
        self.src_act_dec = src_agent.act_dec
        self.tgt_act_enc = tgt_agent.act_enc 
        self.tgt_act_dec = tgt_agent.act_dec
        self.lat_disc = utils.build_mlp(self.lat_obs_dim + self.lat_act_dim, 1, n_layers, 
            hidden_dim, activation='leaky_relu', output_activation='identity').to(self.device)
        self.src_disc = utils.build_mlp(self.src_obs_dim + self.src_act_dim, 1, n_layers, 
            hidden_dim, activation='leaky_relu', output_activation='identity').to(self.device)
        self.tgt_disc = utils.build_mlp(self.tgt_obs_dim + self.tgt_act_dim, 1, n_layers, 
            hidden_dim, activation='leaky_relu', output_activation='identity').to(self.device)

        # Optimizers
        self.tgt_act_enc_opt = torch.optim.Adam(self.tgt_act_enc.parameters(), lr=lr)
        self.tgt_act_dec_opt = torch.optim.Adam(self.tgt_act_dec.parameters(), lr=lr)
        self.lat_disc_opt = torch.optim.Adam(self.lat_disc.parameters(), lr=lr)
        self.src_disc_opt = torch.optim.Adam(self.src_disc.parameters(), lr=lr)
        self.tgt_disc_opt = torch.optim.Adam(self.tgt_disc.parameters(), lr=lr)


    def update_disc(self, src_obs, src_act, tgt_obs, tgt_act, L, step):
        """
        Discriminator tries to separate source and target latent states and actions
        """
        
        fake_lat_obs = self.tgt_obs_enc(tgt_obs).detach()
        fake_lat_act = self.tgt_act_enc(torch.cat([tgt_obs, tgt_act], dim=-1)).detach()
        real_lat_obs = self.src_obs_enc(src_obs).detach()
        real_lat_act = self.src_act_enc(torch.cat([src_obs, src_act], dim=-1)).detach()
        fake_lat_input = torch.cat([fake_lat_obs, fake_lat_act], dim=-1)
        real_lat_input = torch.cat([real_lat_obs, real_lat_act], dim=-1)
        lat_disc_loss = self.lat_disc(fake_lat_input).mean() - self.lat_disc(real_lat_input).mean()

        fake_src_obs = self.src_obs_dec(fake_lat_obs).detach()
        fake_src_act = self.src_act_dec(torch.cat([fake_src_obs, fake_lat_act], dim=-1)).detach()
        fake_src_input = torch.cat([fake_src_obs, fake_src_act], dim=-1)
        src_input = torch.cat([src_obs, src_act], dim=-1)
        src_disc_loss = self.src_disc(fake_src_input).mean() - self.src_disc(src_input).mean()

        real_tgt_obs = self.tgt_obs_dec(real_lat_obs).detach()
        real_tgt_act = self.tgt_act_dec(torch.cat([real_tgt_obs, real_lat_act], dim=-1)).detach()
        real_tgt_input = torch.cat([real_tgt_obs, real_tgt_act], dim=-1)
        tgt_input = torch.cat([tgt_obs, tgt_act], dim=-1)
        tgt_disc_loss = self.tgt_disc(real_tgt_input).mean() - self.tgt_disc(tgt_input).mean()

        lat_gp = self.compute_gradient_penalty(self.lat_disc, real_lat_input, fake_lat_input)
        src_gp = self.compute_gradient_penalty(self.src_disc, src_input, fake_src_input)
        tgt_gp = self.compute_gradient_penalty(self.tgt_disc, tgt_input, real_tgt_input)

        disc_loss = lat_disc_loss + src_disc_loss + tgt_disc_loss + \
            self.lmbd_gp * (lat_gp + src_gp + tgt_gp)
        self.lat_disc_opt.zero_grad()
        self.src_disc_opt.zero_grad()
        self.tgt_disc_opt.zero_grad()
        disc_loss.backward()
        self.lat_disc_opt.step()
        self.src_disc_opt.step()
        self.tgt_disc_opt.step()

        if step % self.log_freq == 0:
            L.add_scalar('train_lat_disc/lat_disc_loss', lat_disc_loss.item(), step)
            L.add_scalar('train_lat_disc/lat_gp', lat_gp.item(), step)
            L.add_scalar('train_lat_disc/src_disc_loss', src_disc_loss.item(), step)
            L.add_scalar('train_lat_disc/src_gp', src_gp.item(), step)
            L.add_scalar('train_lat_disc/tgt_disc_loss', tgt_disc_loss.item(), step)
            L.add_scalar('train_lat_disc/tgt_gp', tgt_gp.item(), step)
            L.add_scalar('train_lat_disc/real_lat_obs_sq', (real_lat_obs**2).mean().item(), step)
            L.add_scalar('train_lat_disc/fake_lat_obs_sq', (fake_lat_obs**2).mean().item(), step)
            L.add_scalar('train_lat_disc/real_lat_act_sq', (real_lat_act**2).mean().item(), step)
            L.add_scalar('train_lat_disc/fake_lat_act_sq', (fake_lat_act**2).mean().item(), step)

    def update_gen(self, src_obs, src_act, src_next_obs, tgt_obs, tgt_act, tgt_next_obs, L, step):
        """
        Generator outputs more realistic latent states from target samples
        """

        # Generator
        fake_lat_obs = self.tgt_obs_enc(tgt_obs)
        fake_lat_act = self.tgt_act_enc(torch.cat([tgt_obs, tgt_act], dim=-1))
        real_lat_obs = self.src_obs_enc(src_obs)
        real_lat_act = self.src_act_enc(torch.cat([src_obs, src_act], dim=-1))
        fake_lat_input = torch.cat([fake_lat_obs, fake_lat_act], dim=-1)
        real_lat_input = torch.cat([real_lat_obs, real_lat_act], dim=-1)
        lat_gen_loss = -self.lat_disc(fake_lat_input).mean()

        fake_src_obs = self.src_obs_dec(fake_lat_obs)
        fake_src_act = self.src_act_dec(torch.cat([fake_src_obs, fake_lat_act], dim=-1))
        fake_src_input = torch.cat([fake_src_obs, fake_src_act], dim=-1)
        src_input = torch.cat([src_obs, src_act], dim=-1)
        src_gen_loss = -self.src_disc(fake_src_input).mean()

        real_tgt_obs = self.tgt_obs_dec(real_lat_obs)
        real_tgt_act = self.tgt_act_dec(torch.cat([real_tgt_obs, real_lat_act], dim=-1))
        real_tgt_input = torch.cat([real_tgt_obs, real_tgt_act], dim=-1)
        tgt_input = torch.cat([tgt_obs, tgt_act], dim=-1)
        tgt_gen_loss = -self.tgt_disc(real_tgt_input).mean()

        gen_loss = lat_gen_loss + src_gen_loss + tgt_gen_loss

        # Cycle consistency
        fake_src_obs = self.src_obs_dec(fake_lat_obs)
        fake_src_act = self.src_act_dec(torch.cat([fake_src_obs, fake_lat_act], dim=-1))
        fake_lat_obs_1 = self.src_obs_enc(fake_src_obs)
        fake_lat_act_1 = self.src_act_enc(torch.cat([fake_src_obs, fake_src_act], dim=-1))
        pred_tgt_obs = self.tgt_obs_dec(fake_lat_obs_1)
        pred_tgt_act = self.tgt_act_dec(torch.cat([pred_tgt_obs, fake_lat_act_1], dim=-1))
        tgt_obs_cycle_loss = F.l1_loss(pred_tgt_obs, tgt_obs)
        tgt_act_cycle_loss = F.l1_loss(pred_tgt_act, tgt_act)

        real_tgt_obs = self.tgt_obs_dec(real_lat_obs)
        real_tgt_act = self.tgt_act_dec(torch.cat([real_tgt_obs, real_lat_act], dim=-1))
        real_lat_obs_1 = self.tgt_obs_enc(real_tgt_obs)
        real_lat_act_1 = self.tgt_act_enc(torch.cat([real_tgt_obs, real_tgt_act], dim=-1))
        pred_src_obs = self.src_obs_dec(real_lat_obs_1)
        pred_src_act = self.src_act_dec(torch.cat([pred_src_obs, real_lat_act_1], dim=-1))
        src_obs_cycle_loss = F.l1_loss(pred_src_obs, src_obs)
        src_act_cycle_loss = F.l1_loss(pred_src_act, src_act)
        cycle_loss = tgt_obs_cycle_loss + tgt_act_cycle_loss + src_obs_cycle_loss + src_act_cycle_loss

        # Latent dynamics
        fake_lat_next_obs = self.tgt_obs_enc(tgt_next_obs)
        pred_lat_act = self.inv_dyn(torch.cat([fake_lat_obs, fake_lat_next_obs], dim=-1))
        pred_act = self.tgt_act_dec(torch.cat([tgt_obs, pred_lat_act], dim=-1))
        inv_loss = F.mse_loss(pred_act, tgt_act)
        pred_lat_next_obs = self.fwd_dyn(torch.cat([fake_lat_obs, fake_lat_act], dim=-1))
        fwd_loss = F.mse_loss(pred_lat_next_obs, fake_lat_next_obs)

        loss = gen_loss + self.lmbd_cyc * cycle_loss + self.lmbd_dyn * (inv_loss + fwd_loss)

        self.tgt_obs_enc_opt.zero_grad()
        self.tgt_obs_dec_opt.zero_grad()
        self.tgt_act_enc_opt.zero_grad()
        self.tgt_act_dec_opt.zero_grad()
        loss.backward()
        self.tgt_obs_enc_opt.step()
        self.tgt_obs_dec_opt.step()
        self.tgt_act_enc_opt.step()
        self.tgt_act_dec_opt.step()

        if step % self.log_freq == 0:
            L.add_scalar('train_lat_gen/lat_gen_loss', lat_gen_loss.item(), step)
            L.add_scalar('train_lat_gen/src_gen_loss', src_gen_loss.item(), step)
            L.add_scalar('train_lat_gen/tgt_gen_loss', tgt_gen_loss.item(), step)
            L.add_scalar('train_lat_gen/inv_loss', inv_loss.item(), step)
            L.add_scalar('train_lat_gen/fwd_loss', fwd_loss.item(), step)
            L.add_scalar('train_lat_gen/cycle_loss', cycle_loss.item(), step)
            L.add_scalar('train_lat_gen/tgt_obs_cycle_loss', tgt_obs_cycle_loss.item(), step)
            L.add_scalar('train_lat_gen/tgt_act_cycle_loss', tgt_act_cycle_loss.item(), step)
            L.add_scalar('train_lat_gen/src_obs_cycle_loss', src_obs_cycle_loss.item(), step)
            L.add_scalar('train_lat_gen/src_act_cycle_loss', src_act_cycle_loss.item(), step)
            
            real_lat_obs = self.src_obs_enc(tgt_obs)
            real_lat_act = self.src_act_enc(torch.cat([tgt_obs, tgt_act], dim=-1))
            L.add_scalar('valid_lat_gen/lat_obs_diff', F.l1_loss(real_lat_obs, fake_lat_obs), step)
            L.add_scalar('valid_lat_gen/lat_act_diff', F.l1_loss(real_lat_act, fake_lat_act), step)

            real_tgt_obs = self.src_obs_dec(real_lat_obs)
            real_tgt_act = self.src_act_dec(torch.cat([real_tgt_obs, real_lat_act], dim=-1))
            fake_tgt_obs = self.tgt_obs_dec(real_lat_obs)
            fake_tgt_act = self.tgt_act_dec(torch.cat([fake_tgt_obs, real_lat_act], dim=-1))
            fake_tgt_act_1 = self.tgt_act_dec(torch.cat([real_tgt_obs, real_lat_act], dim=-1))
            L.add_scalar('valid_lat_gen/tgt_obs_diff', F.l1_loss(real_tgt_obs, fake_tgt_obs), step)
            L.add_scalar('valid_lat_gen/tgt_act_diff', F.l1_loss(real_tgt_act, fake_tgt_act), step)
            L.add_scalar('valid_lat_gen/tgt_act_diff', F.l1_loss(real_tgt_act, fake_tgt_act_1), step)

