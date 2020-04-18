import sys
sys.path.append('../')
from envs.dist_center import DistCenter
from envs.reg_dist_center import RegDistCenter
import warnings
import torch
from common.models import ModelA2C
from common.utils import test_net

warnings.filterwarnings("ignore", category=UserWarning)
                                      
test_env_dict = {
                 "reg_dist_center" : {"env" : RegDistCenter(dc_model_path = "../saves/best/best_+1977.900_3692000.dat",
                                                            name = "New York", 
                                                            invt_path = "../data/starting_invt_brake_pads.csv", 
                                                            order_path = "../data/incoming_orders.csv"),  
                                      "model" : "../saves/best/best_+2178.800_4720000.dat"}, 
                               
                 "dist_center" :  {"env" : DistCenter(name = "Charlotte", 
                                                      invt_path = "../data/starting_invt_brake_pads.csv", 
                                                      order_path = "../data/incoming_orders.csv"),
                                   "model" : "../saves/best/best_+1977.900_3692000.dat"}  
                }

test_type = 'reg_dist_center'
env = test_env_dict[test_type]['env']
model_dict = torch.load(test_env_dict[test_type]['model'])
model = ModelA2C(env.observation_space.shape[0], env.action_space.shape[0]).to("cpu")
model.load_state_dict(model_dict)
rewards, steps = test_net(model, env, 1, "cpu")
print(rewards)
print(steps)