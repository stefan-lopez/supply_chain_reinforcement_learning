import math
import ptan
import numpy as np
import torch
import csv
import sys
sys.path.append('../')
from envs.reg_dist_center import RegDistCenter


def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v, var_v, _ = net(obs_v)
            mu = mu_v.data.cpu().numpy()
            sigma = torch.sqrt(var_v).data.cpu().numpy()
            action = np.random.normal(mu, sigma)
            action = np.clip(action, 0, 1)
            obs, reward, done, _ = env.step(action[0])
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count

def unpack_batch_a2c(batch, net3, last_val_gamma, device="cpu"):
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(exp.last_state)
    states_v = ptan.agent.float32_preprocessor(states).to(device)
    actions_v = torch.FloatTensor(actions).to(device)

    rewards_np = np.array(rewards, dtype=np.float32)

    if not_done_idx:
        last_states_v = ptan.agent.float32_preprocessor(last_states).to(device)
        last_vals_v = net3(last_states_v)[2]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    
    return states_v, actions_v, ref_vals_v

    
def calc_logprob(mu_v, var_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2 * var_v.clamp(min = 1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
    
    return p1 + p2

def create_regional_orders(dc_model_path, name, invt_path, order_path, save_path):
    env = RegDistCenter(dc_model_path = dc_model_path, name = name, invt_path = invt_path, order_path = order_path)
    master_orders = []
            
    for i in range(0, 1000):
        print("gathering generated data, iteration: " + str(i))
        order_list = []
        for _ in range(0, 1001 + len(env.incoming_orders)):
            env.step(np.array([0, 0, .05]))
            new_order = env.generate_dc_orders()
            order_list.append(new_order[0])
           
        master_orders.append(order_list)
      
    with open(save_path, "w", newline='') as file_out:
        writer = csv.writer(file_out)
        writer.writerows(master_orders)

    return master_orders