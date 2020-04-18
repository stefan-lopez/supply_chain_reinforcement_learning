import os
import time
import math
import ptan
import argparse
from tensorboardX import SummaryWriter

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from dist_center import DistCenter
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
ENV_ID = "MinitaurBulletEnv-v0"
GAMMA = 0.999999
# BATCH_SIZE = 32
BATCH_SIZE = 64
# LEARNING_RATE = 5e-5
LEARNING_RATE = 1e-4
ENTROPY_BETA = 1e-4

TEST_ITERS = 1000

HID_SIZE = 128

REPLAY_SIZE = 1000
REPLAY_INITIAL = 1000

class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, act_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class DDPGCritic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGCritic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 200),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(200 + act_size, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))
    
class AgentDDPG(ptan.agent.BaseAgent):
    """
    Agent implementing Orstein-Uhlenbeck exploration process
    """
    def __init__(self, net, device="cpu", ou_enabled=True, ou_mu=0.0, ou_teta=0.15, ou_sigma=0.2, ou_epsilon=1.0):
        self.net = net
        self.device = device
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_teta = ou_teta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon

    def initial_state(self):
        return None

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()

        if self.ou_enabled and self.ou_epsilon > 0:
            new_a_states = []
            for a_state, action in zip(agent_states, actions):
                if a_state is None:
                    a_state = np.zeros(shape=action.shape, dtype=np.float32)
                a_state += self.ou_teta * (self.ou_mu - a_state)
                a_state += self.ou_sigma * np.random.normal(size=action.shape)

                action += self.ou_epsilon * a_state
                new_a_states.append(a_state)
        else:
            new_a_states = agent_states

        actions = np.clip(actions, 0, 1)
        return actions, new_a_states
    
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    save_path = os.path.join("saves", "ddpg-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    dist_config_path = "dist_center_config.json"
    invt_path = "data/starting_invt_serpentine_belts.csv"
    env = DistCenter("Charlotte", dist_config_path, invt_path)
    test_env = DistCenter("Charlotte", dist_config_path, invt_path)

    act_net = DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    crt_net = DDPGCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    print(act_net)
    print(crt_net)
    tgt_act_net = ptan.agent.TargetNet(act_net)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)

    writer = SummaryWriter(comment="-ddpg_" + args.name)
    agent = AgentDDPG(act_net, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)
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
                    try:
                        tracker.reward(rewards[0], frame_idx)
                    except:
                        pass

                if len(buffer) < REPLAY_INITIAL:
                    continue

                batch = buffer.sample(BATCH_SIZE)
                states_v, actions_v, rewards_v, dones_mask, last_states_v = unpack_batch_ddqn(batch, device)

                # train critic
                crt_opt.zero_grad()
                q_v = crt_net(states_v, actions_v)
                last_act_v = tgt_act_net.target_model(last_states_v)
                q_last_v = tgt_crt_net.target_model(last_states_v, last_act_v)
                q_last_v[dones_mask] = 0.0
                q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * GAMMA
                critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                critic_loss_v.backward()
                crt_opt.step()

                tb_tracker.track("loss_critic", critic_loss_v.tolist(), frame_idx)
                tb_tracker.track("critic_ref", q_ref_v.mean().tolist(), frame_idx)

                # train actor
                act_opt.zero_grad()
                cur_actions_v = act_net(states_v)
                actor_loss_v = -crt_net(states_v, cur_actions_v)
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward()
                act_opt.step()
                tb_tracker.track("loss_actor", actor_loss_v.tolist(), frame_idx)

                tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                tgt_crt_net.alpha_sync(alpha=1 - 1e-3)
                

                if frame_idx % TEST_ITERS == 0:
                    print(actions_v)
                    print(states_v[0])
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
