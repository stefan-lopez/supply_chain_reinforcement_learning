import torch
import ptan
import numpy as np
import random
import sys
sys.path.append('../')
from envs.dist_center import DistCenter
from common.models import ModelA2C

class RegDistCenter(DistCenter):
    def __init__(self, generated_orders = [], dc_model_path = "", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generated_orders = generated_orders 
        self.dc_model_path = dc_model_path
        self.initialize_generated_orders()
        self.incoming_orders = np.asarray([self.generate_dc_orders() for _ in range(self.ship_days)]).astype('float64')
        
    def initialize_generated_orders(self):
        if len(self.generated_orders) > 0:
            self.order_index = random.randint(0,9)
            self.order_list = self.generated_orders[self.order_index].copy()  
        else:
            self.dc_dict = self.create_dc_dict(["Charlotte", "Nashville", "Chicago", "Toronto"])
            self.min_stop_val = .1
            self.max_stop_val = 2      
        
        return self
    
    def reset(self):
        self.__init__(self.generated_orders, self.dc_model_path, self.name, self.invt_path, self.order_path)
            
        return self.observation()
        
    def update_incoming_order_queue(self):
        latest_order = self.generate_dc_orders()
        self.incoming_orders = np.append(self.incoming_orders, np.expand_dims(latest_order, axis=0), axis = 0)
        curr_order = self.incoming_orders[0]
        self.incoming_orders = np.delete(self.incoming_orders, 0, 0)
        self.apply_incoming_order(curr_order) 
        
    def create_dc_dict(self, dc_names_list):
        dc_dict = {}
        for name in dc_names_list:
            dc_dict[name] = {"env" : self.load_dc_env(name), "model" : self.load_dc_model(self.load_dc_env(name)), 
                             "obs" : self.load_dc_env(name).reset(), "curr_orders" : None}
              
        return dc_dict
      
    def load_dc_env(self, name):
        env = DistCenter(name, self.invt_path, self.order_path)  
 
        return env
                          
    def load_dc_model(self, env):
        model_dict = torch.load(self.dc_model_path)
        model = ModelA2C(env.observation_space.shape[0], env.action_space.shape[0]).to("cpu")
        model.load_state_dict(model_dict)
              
        return model
      
    def generate_dc_orders(self):
        if len(self.generated_orders) == 0:
            total_orders = np.asarray([0]).astype('float64')
            for dc in self.dc_dict.keys():
                action, obs = self.retrieve_actions(self.dc_dict[dc]["env"], self.dc_dict[dc]["model"], self.dc_dict[dc]["obs"])
                action_quantities = action * self.dc_dict[dc]["env"].init_invt_quantities
                total_orders = total_orders + np.array(action_quantities.sum())
                self.dc_dict[dc]["obs"] = obs
                self.dc_dict[dc]["curr_orders"] = action
        else:
            total_orders = [float(self.order_list.pop(0))]
  
        return total_orders
              
    def retrieve_actions(self, env, net, obs):
        obs_v = ptan.agent.float32_preprocessor([obs]).to("cpu")
        mu_v, var_v, _ = net(obs_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        action = np.random.normal(mu, sigma)
        action = np.clip(action, 0, 1)
        obs, _, _, _ = env.step(action[0])
          
        return action, obs
     

            
        
    

    