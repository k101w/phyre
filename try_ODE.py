#!/usr/bin/env python3

import os
import sys
import shutil
import math
import numpy as np
import pdb
import matplotlib.pyplot as plt
plt.style.use('bmh')

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad

import higher
import hydra
from omegaconf import DictConfig, OmegaConf
import csv
import copy
from torchdiffeq import odeint, odeint_event

from pymunk_balls import Balls
import utils
import pdb
import phyre

try:
    if os.isatty(sys.stdout.fileno()):
        from IPython.core import ultratb
        sys.excepthook = ultratb.FormattedTB(
            mode='Verbose', color_scheme='Linux', call_pdb=1)
except:
    pass

torch.set_default_dtype(torch.float64)
class PHYRE_simulation():
    def __init__(self,eval_setup,fold_id):
        super().__init__()
        self.tasks=tasks
        train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
        self.action_tier = phyre.eval_setup_to_action_tier(eval_setup)
        self.simulator=phyre.initialize_simulator(train_tasks, self.action_tier)
    def generate(self):


        for task_index in range(len(tasks)):
        task_id = simulator.task_ids[task_index]
        initial_scene = simulator.initial_scenes[task_index]
        plt.imshow(phyre.observations_to_float_rgb(initial_scene))
        print(initial_scene.shape)
        plt.title(f'Task {task_index}');
        plt.savefig('pic/ini{}.png'.format(task_index))

        initial_featurized_objects = simulator.initial_featurized_objects[task_index]
        print('Initial featurized objects shape=%s dtype=%s' % (initial_featurized_objects.features.shape, initial_featurized_objects.features.dtype))
        bar=np.array([x for x in initial_featurized_objects.features[0] if x[5]==1])
        bar=bar.reshape(1,bar.shape[0],-1)
        ball=np.array([x for x in initial_featurized_objects.features[0] if x[4]==1])
        ball=ball.reshape(1,ball.shape[0],-1)

        # bar=[x for x in initial_featurized_objects.features[0] if x[5]==1]
        # bar=bar.reshape(1,bar.shape[0],-1)
        # plt.imshow(phyre.observations_to_float_rgb(bar))
        # plt.savefig('ini_bar.png')
        # bar=[x for x in initial_featurized_objects.features[0] if x[5]==1]
        # bar=bar.reshape(1,bar.shape[0],-1)
        # plt.imshow(phyre.observations_to_float_rgb(bar))
        # plt.savefig('ini_bar.png')

        np.set_printoptions(precision=3)
        #print(initial_featurized_objects.features)

        actions = simulator.build_discrete_action_space(max_actions=100)
        print('A random action:', actions[0])

            # The simulator takes an index into simulator.task_ids.
        action = random.choice(actions)
        # Set need_images=False and need_featurized_objects=False to speed up simulation, when only statuses are needed.
        for i in range(len(actions)):
                action=actions[i]
                simulation = simulator.simulate_action(task_index, action, need_images=True, need_featurized_objects=True,stride=20)
                if(simulation.status!=0): break
        # May call is_* methods on the status to check the status.

        print('Result of taking action', action, 'on task', tasks[task_index], 'is:',
                simulation.status)

class HamiltonianDynamics(nn.Module):
    def __init__(self, n_balls):
        super().__init__()

    def forward(self, t, state):
        pos, vel, *rest = state
        dvel = torch.zeros_like(pos)
        dpos = vel
        dvel[:,1] = -17. # TODO: Not sure why this isn't -20. as in pymunk

        # Freeze anything going underground
        I = pos[:,-1] < -1.#Y方向不能超过地面
        dpos[I] = 0.
        dvel[I] = 0.

        return (dpos, dvel, *[torch.zeros_like(r) for r in rest])


class EventFn(nn.Module):
    def __init__(self, n_balls, n_event_latent, init_factor):
        super().__init__()
        self.mod = utils.mlp(
            input_dim=2*n_balls,
            hidden_dim=512,
            output_dim=n_event_latent,
            hidden_depth=2,
            output_mod=nn.Tanh(),
        )
        def init(m):
            if isinstance(m, nn.Linear):
                std = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-init_factor*std, init_factor*std)
                m.bias.data.zero_()
        self.mod.apply(init)
        #  TODO:
        self.fmod = higher.monkeypatch(self.mod, track_higher_grads=False)

    def parameters(self):
        return self.mod.parameters()

    def forward(self, t, state):
        pos, _, *params = state
        pos_flat = pos.view(-1)
        val = self.fmod(pos_flat, params=params)
        val = torch.prod(val)#返回所有的乘积

        # True event function for bounces:
        # val = min(pos[:,1]) - 0.35

        return val

    def __getstate__(self):
        d = copy.copy(self.__dict__)
        d['_modules'] = copy.copy(d['_modules'])
        del d['_modules']['fmod']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.fmod = higher.monkeypatch(self.mod, track_higher_grads=False)


class InstantaneousStateChange(nn.Module):
    def __init__(self, event_latents_fn, n_balls, n_event_latent):
        super().__init__()
        self.event_latents_fn = event_latents_fn
        self.net = utils.mlp(
            input_dim=2*2*n_balls + n_event_latent,
            hidden_dim=512,
            output_dim=2*n_balls,
            hidden_depth=2,
        )

    def forward(self, t, state):
        pos, vel, *rest = state

        event_latents = self.event_latents_fn(pos.view(-1)).detach()
        z = torch.cat((event_latents, pos.view(-1), vel.view(-1)))
        vel = self.net(z).reshape(pos.shape)

        return (pos, vel, *rest)


class NeuralPhysics(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device

        # Assume these are known and true:
        self.max_events = cfg.max_events
        self.n_balls = len(cfg.start_pos)
        cfg.n_balls = self.n_balls
        self.initial_pos = torch.tensor(cfg.start_pos).requires_grad_().to(self.device)
        self.initial_vel = torch.zeros_like(self.initial_pos).requires_grad_().to(self.device)
        #TODO:
        self.dynamics_fn = HamiltonianDynamics(self.n_balls)
        #self.event_fn = hydra.utils.instantiate(self.cfg.event_fn)
        self.event_fn = hydra.utils.instantiate(self.cfg.event_fn)
        self.inst_update = InstantaneousStateChange(
            self.event_fn.mod, self.n_balls, self.cfg.event_fn.n_event_latent)


    def event_fn_with_termination(self, t1):#t1是end time
        def event_fn(t, state):
            event_fval = self.event_fn(t, state)
            return event_fval * (t - (self.cfg.termination_factor*t1 + 1e-7))
        return event_fn
#TODO:
    def forward(self, times):
        t0 = torch.tensor([0.0]).to(times)
        state = (self.initial_pos, self.initial_vel, *self.event_fn.parameters())
        event_times = []
        traj_pos = [state[0].unsqueeze(0)]
        traj_vel = [state[1].unsqueeze(0)]#INITIAL
        event_fn_terminal = self.event_fn_with_termination(times[-1])#times[-1]是最终时间
        n_events = 0
        pdb.set_trace()

        while t0 < times[-1] and n_events < self.max_events:
            last = n_events == self.max_events - 1

            if not last:
                event_t, solution = odeint_event(
                    self.dynamics_fn, state, t0, event_fn=event_fn_terminal,
                    atol=1e-8, rtol=1e-8)
            else:
                event_t = times[-1]

            interval_ts = times[times > t0]
            interval_ts = interval_ts[interval_ts <= event_t]
            interval_ts = torch.cat([t0.reshape(-1), interval_ts.reshape(-1)])

            solution_ = odeint(self.dynamics_fn, state, interval_ts, atol=1e-8, rtol=1e-8)
            # [0] for position; [1:] to remove intial state.
            traj_pos.append(solution_[0][1:])
            traj_vel.append(solution_[1][1:])

            if not last:
                state = tuple(s[-1] for s in solution)
            else:#如果是last，就不用再通过odeint_event计算event_t了
                state = tuple(s[-1] for s in solution_)

            # update velocity instantaneously.
            state = self.inst_update(event_t, state)

            # step to avoid re-triggering the event fn.
            pos, *rest = state
            pos = pos + 1e-6 * self.dynamics_fn(event_t, state)[0]
            state = pos, *rest

            event_times.append(event_t)
            t0 = event_t
            n_events += 1

        traj_pos = torch.cat(traj_pos, dim=0)
        traj_vel = torch.cat(traj_vel, dim=0)
        return traj_pos, traj_vel, event_times


def cosine_decay(learning_rate, global_step, decay_steps, alpha=0.0):
    global_step = min(global_step, decay_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * global_step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return learning_rate * decayed


def learning_rate_schedule(global_step, warmup_steps, base_learning_rate, lr_scaling, train_steps):
    warmup_steps = int(round(warmup_steps))
    scaled_lr = base_learning_rate * lr_scaling
    if warmup_steps:
        learning_rate = global_step / warmup_steps * scaled_lr
    else:
        learning_rate = scaled_lr

    if global_step < warmup_steps:
        learning_rate = learning_rate
    else:
        learning_rate = cosine_decay(scaled_lr, global_step - warmup_steps, train_steps - warmup_steps)
    return learning_rate


def set_learning_rate(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

class Workspace(object):
    def __init__(self, cfg):
        # print(cfg)
        # pdb.set_trace()
        self.cfg = cfg
        self.device = cfg.device
        torch.manual_seed(self.cfg.seed)
        self.n_balls = len(self.cfg.start_pos)

        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.fig_dir = os.path.join(self.work_dir, 'figs')
        if os.path.exists(self.fig_dir):
            shutil.rmtree(self.fig_dir)
        os.makedirs(self.fig_dir)

        self.logf = open('log.csv', 'w')
        fieldnames = ['iter', 'loss']
        self.writer = csv.DictWriter(self.logf, fieldnames=fieldnames)
        self.writer.writeheader()

        self.model = NeuralPhysics(self.cfg).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.base_lr) #, betas=(0.5, 0.5))
        # self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.cfg.base_lr)

    # @profile
    def run(self):
        with torch.no_grad():
            system = Balls(start_pos=self.cfg.start_pos)
            obs_times, gt_pos, gt_vel = system.simulate(n_step=40)
            obs_times = torch.from_numpy(obs_times).requires_grad_().to(self.device)
            gt_pos = torch.from_numpy(gt_pos).requires_grad_().to(self.device)
            gt_vel = torch.from_numpy(gt_vel).requires_grad_().to(self.device)

        for itr in range(self.cfg.num_iterations):
            self.optimizer.zero_grad()
            pos, vel, event_times = self.model(obs_times)
            # loss = F.mse_loss(pos, gt_pos)
            loss = F.mse_loss(pos, gt_pos) + 0.1*F.mse_loss(vel, gt_vel)
            loss.backward()

            # lr = learning_rate_schedule(itr, 0, self.cfg.base_lr, 1.0,
            #                             self.cfg.num_iterations)
            # set_learning_rate(self.optimizer, lr)
            self.optimizer.step()

            self.writer.writerow({
                'iter': itr,
                'loss': loss.item(),
            })
            self.logf.flush()

            if itr % self.cfg.log_interval == 0:
                print(itr, loss.item(), len(event_times))

            if itr % self.cfg.plot_interval == 0:
                self.plot(itr, obs_times, pos, gt_pos)

            if itr % self.cfg.save_interval == 0:
                self.save('latest')

            del pos, vel, loss


    def plot(self, itr, obs_times, states, gt_states):
        nrow, ncol = 2, 2
        fig, axs = plt.subplots(nrow, ncol, figsize=(6*ncol, 4*nrow))
        axs = axs.ravel()

        colors = plt.style.library['bmh']['axes.prop_cycle'].by_key()['color']
        for state_dim in range(2):
            ax = axs[state_dim]
            for i in range(self.n_balls):
                ax.plot(utils.to_np(obs_times),
                        utils.to_np(states[:,i,state_dim]), color=colors[i])
                ax.plot(utils.to_np(obs_times),
                        utils.to_np(gt_states[:,i,state_dim]),
                        color=colors[i], alpha=0.3)
            ax.set_title(f'State Dim {state_dim}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Position')

        event_fn_terminal = self.model.event_fn_with_termination(obs_times[-1])
        event_vals = []
        event_vals_terminal = []
        ps = list(self.model.event_fn.mod.parameters())
        for t, state in zip(obs_times, states):
            event_vals.append(self.model.event_fn(t, [state, None] + ps))
            event_vals_terminal.append(event_fn_terminal(t, [state, None] + ps))
        ax = axs[2]
        # TODO:
        #obs_times=obs_times.cpu()
        #pdb.set_trace()
        ax.plot(obs_times.cpu().detach().numpy(), event_vals)
        ax.set_xlabel('Time')
        ax.set_title('Event Fn')

        ax = axs[3]
        ax.plot(utils.to_np(obs_times), event_vals_terminal)
        ax.set_xlabel('Time')
        ax.set_title('Event Fn with Termination')

        out = f'{self.fig_dir}/{itr:05d}.png'
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)

    def save(self, tag):
        path = os.path.join(self.work_dir, f'{tag}.pt')
        with open(path, 'wb') as f:
            torch.save(self, f)


    def __getstate__(self):
        d = copy.copy(self.__dict__)
        del d['logf'], d['writer']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        # TODO: Re-init logging if restarting

from try_ODE import Workspace as W # For saving/loading

@hydra.main(config_path='.', config_name='try_ODE')

#@hydra.main(config_path='learn_pymunk.yaml', strict=True)

def main(cfg):
    workspace = W(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
