#!/usr/bin/env python3
import os
import ptan
import time
import gym
import pybullet_envs
import argparse
from tensorboardX import SummaryWriter
import numpy as np

import torch.nn as nn
from dist_center import DistCenter

import torch
import torch.optim as optim
import torch.nn.functional as F


ENV_ID = "MinitaurBulletEnv-v0"
GAMMA = 0.99
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
REPLAY_SIZE = 10000
REPLAY_INITIAL = 1000
REWARD_STEPS = 25

TEST_ITERS = 1000

Vmax = 10
Vmin = 0
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)

class D4PGCritic(nn.Module):
    def __init__(self, obs_size, act_size, n_atoms, v_min, v_max):
        super(D4PGCritic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.ReLU(),
            nn.Linear(300, n_atoms)
        )

        delta = (v_max - v_min) / (n_atoms - 1)
        self.register_buffer("supports", torch.arange(v_min, v_max + delta, delta))

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))

    def distr_to_q(self, distr):
        weights = F.softmax(distr, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)
    
class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
    
class AgentD4PG(ptan.agent.BaseAgent):
    """
    Agent implementing noisy agent
    """
    def __init__(self, net, device="cpu", epsilon=0.3):
        self.net = net
        self.device = device
        self.epsilon = epsilon

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()
        actions += self.epsilon * np.random.normal(size=actions.shape)
        actions = np.clip(actions, 0, 1)
        return actions, agent_states


def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, 0, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


def distr_projection(next_distr_v, rewards_v, dones_mask_t, gamma, device="cpu"):
    next_distr = next_distr_v.data.cpu().numpy()
    rewards = rewards_v.data.cpu().numpy()
    dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, N_ATOMS), dtype=np.float32)

    for atom in range(N_ATOMS):
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * DELTA_Z) * gamma))
        b_j = (tz_j - Vmin) / DELTA_Z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

    if dones_mask.any():
        proj_distr[dones_mask] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones_mask]))
        b_j = (tz_j - Vmin) / DELTA_Z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones_mask.copy()
        eq_dones[dones_mask] = eq_mask
#         print(eq_dones.shape)
#         print(proj_distr.shape)
#         print(eq_dones.shape)
#         print(proj_distr.shape)
#         print(eq_dones)
#         print(eq_dones.reshape(-1,1).shape)
        if eq_dones.any():
            proj_distr[eq_dones,l[:,np.newaxis]] = 1.0
#         print(proj_distr)
#         if eq_dones.any():
#             proj_distr[eq_dones,l] = 1.0
        ne_mask = u != l
        ne_dones = dones_mask.copy()
        ne_dones[dones_mask] = ne_mask
#         print("----")
#         print(ne_dones)
#         print(ne_dones.shape)
#         print(proj_distr)
#         print(proj_distr.shape)
#         print(ne_mask)
#         print(ne_mask.shape)
#         print(l, u, b_j)
#         print(proj_distr[ne_dones, l])
        if ne_dones.any():
            proj_distr[ne_dones, l[:,np.newaxis]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[:,np.newaxis]] = (b_j - l)[ne_mask]
#         if ne_dones.any():
#             proj_distr[ne_dones, l] = (u - b_j)[ne_mask]
#             proj_distr[ne_dones, u] = (b_j - l)[ne_mask]
    return torch.FloatTensor(proj_distr).to(device)

def unpack_batch_ddqn(batch, device="cpu"):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    states_v = ptan.agent.float32_preprocessor(states).to(device)
    actions_v = ptan.agent.float32_preprocessor(actions).to(device)
    rewards_v = ptan.agent.float32_preprocessor(rewards).to(device)
    last_states_v = ptan.agent.float32_preprocessor(last_states).to(device)
    dones_t = torch.ByteTensor(dones).to(device)
    return states_v, actions_v, rewards_v, dones_t, last_states_v

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    save_path = os.path.join("saves", "d4pg-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    dist_config_path = "dist_center_config.json"
    invt_path = "data/starting_invt_serpentine_belts.csv"
    env = DistCenter("Charlotte", dist_config_path, invt_path)
    test_env = DistCenter("Charlotte", dist_config_path, invt_path)

    act_net = DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    crt_net = D4PGCritic(env.observation_space.shape[0], env.action_space.shape[0], N_ATOMS, Vmin, Vmax).to(device)
    print(act_net)
    print(crt_net)
    tgt_act_net = ptan.agent.TargetNet(act_net)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)

    writer = SummaryWriter(comment="-d4pg_" + args.name)
    agent = AgentD4PG(act_net, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)

    frame_idx = 0
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            while True:
                frame_idx += 1
                buffer.populate(1)
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], frame_idx)
                    tracker.reward(rewards[0], frame_idx)

                if len(buffer) < REPLAY_INITIAL:
                    continue

                batch = buffer.sample(BATCH_SIZE)
                states_v, actions_v, rewards_v, dones_mask, last_states_v = unpack_batch_ddqn(batch, device)

                # train critic
                crt_opt.zero_grad()
                crt_distr_v = crt_net(states_v, actions_v)
                last_act_v = tgt_act_net.target_model(last_states_v)
                last_distr_v = F.softmax(tgt_crt_net.target_model(last_states_v, last_act_v), dim=1)
#                 print(last_distr_v.shape)
#                 print(rewards_v.shape)
                proj_distr_v = distr_projection(last_distr_v, rewards_v, dones_mask,
                                                gamma=GAMMA**REWARD_STEPS, device=device)
                prob_dist_v = -F.log_softmax(crt_distr_v, dim=1) * proj_distr_v
                critic_loss_v = prob_dist_v.sum(dim=1).mean()
                critic_loss_v.backward()
                crt_opt.step()
                tb_tracker.track("loss_critic", critic_loss_v, frame_idx)

                # train actor
                act_opt.zero_grad()
                cur_actions_v = act_net(states_v)
                crt_distr_v = crt_net(states_v, cur_actions_v)
                actor_loss_v = -crt_net.distr_to_q(crt_distr_v)
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward()
                act_opt.step()
                tb_tracker.track("loss_actor", actor_loss_v, frame_idx)

                tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                if frame_idx % TEST_ITERS == 0:
                    print(cur_actions_v)
                    ts = time.time()
                    rewards, steps = test_net(act_net, test_env, device=device)
                    print("Test done in %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, frame_idx)
                    writer.add_scalar("test_steps", steps, frame_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, frame_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(act_net.state_dict(), fname)
                        best_reward = rewards

    pass
