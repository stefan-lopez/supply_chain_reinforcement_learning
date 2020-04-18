import os
import time
import math
import ptan
import argparse
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.nn.functional as F
import sys
sys.path.append('../')
from common.file_operations import read_generated_orders
from envs.dist_center import DistCenter
from common.models import ModelA2C, AgentA2C
from common.utils import test_net, unpack_batch_a2c, calc_logprob
from envs.reg_dist_center import RegDistCenter


GAMMA = 0.9999999
REWARD_STEPS = 2
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
ENTROPY_BETA = 1e-5
TEST_ITERS = 1000
INVT_PATH = "../data/starting_invt_serpentine_belts.csv"
ORDER_PATH = "../data/incoming_orders.csv"
DC_MODEL_PATH = "../saves/a2c-pleasework/best_+1977.900_3692000.dat"
SAVE_PATH = "../data/dc_order_list.csv"
TRAINING_TYPE = "regional_dist_center"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    save_path = os.path.join("saves", "a2c-" + args.name)
    os.makedirs(save_path, exist_ok=True)
    
    training_types = {"dist_center" : DistCenter(name = "Charlotte", invt_path = INVT_PATH, order_path = ORDER_PATH),                                  
                     "regional_dist_center" : RegDistCenter(generated_orders = read_generated_orders(SAVE_PATH), name = "New York", invt_path = INVT_PATH, order_path = ORDER_PATH)}
    
    env = training_types[TRAINING_TYPE]
    test_env = training_types[TRAINING_TYPE]
    net = ModelA2C(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    writer = SummaryWriter(comment="-a2c_" + args.name)
    agent = AgentA2C(net, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    batch = []
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards1, steps = zip(*rewards_steps) 
                    tb_tracker.track("episode_steps", steps[0], step_idx)
                    try:
                        tracker.reward(rewards1[0], step_idx)
                    except:
                        pass  
                if step_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps = test_net(net, test_env, device=device)
                    print("Test done is %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, step_idx)
                    writer.add_scalar("test_steps", steps, step_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(net.state_dict(), fname)
                        best_reward = rewards            
                batch.append(exp)
                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_v, vals_ref_v = unpack_batch_a2c(batch, net, last_val_gamma=GAMMA ** REWARD_STEPS, device=device)
                    
                batch.clear()
                optimizer.zero_grad()
                mu_v, var_v, value_v = net(states_v)

                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
                log_prob_v = adv_v * calc_logprob(mu_v, var_v, actions_v)
                loss_policy_v = -log_prob_v.mean()
                entropy_loss_v = ENTROPY_BETA * (-(torch.log(2*math.pi*var_v) + 1)/2).mean()

                loss_v = loss_policy_v + entropy_loss_v + loss_value_v
                loss_v.backward()
                optimizer.step()

                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", value_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", loss_value_v, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)
