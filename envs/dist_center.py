import sys
sys.path.append('../')
from envs.inventory import Inventory
from common.file_operations import generate_orders
import numpy as np
import gym

class DistCenter(gym.Env):
    def __init__(self, name, invt_path, order_path):
        self.name = name
        self.invt_path = invt_path
        self.ship_days = 14
        self.freight_days = 7
        self.plane_days = 2
        self.invt = Inventory(name).load_invt_file(invt_path) 
        self.order_path = order_path          
        self.init_invt_quantities = self.invt.get_invt_quantities()
        self.action_space = gym.spaces.Box(low = 0, high = np.inf, shape = (len(self.init_invt_quantities) * 3,), dtype = np.float32)
        self.pending_refill_orders = np.asarray([[0] * len(self.init_invt_quantities) for _ in range(self.ship_days)]).astype('float64')
        self.observation_shape = ((len(self.pending_refill_orders) * len(self.init_invt_quantities)) + len(self.init_invt_quantities),)
        self.observation_space = gym.spaces.Box(low = 0, high = np.inf, shape = self.observation_shape, dtype = np.float32) 
        self.incoming_orders = np.asarray([self.reformat_incoming_orders(generate_orders(self.order_path)) for _ in range(self.ship_days)]).astype('float64')
        self.previous_orders = np.asarray([[0] for _ in range(self.ship_days)]).astype('float64')
        self.invt_history = np.asarray([[0] for _ in range(self.ship_days)]).astype('float64')
        self.min_stop_val = .2
        self.max_stop_val = 1.5
        self.step_count = 0 
        
    def reformat_incoming_orders(self, orders):
        order_array = [0] * len(self.init_invt_quantities)
        for idx, part in enumerate(self.invt.invt_records.keys()):
            for order in orders:
                if part == order[3]:
                    order_array[idx] = order[4]
                    
        return order_array
    
    def apply_refill_order(self, orders):
        for idx, part in enumerate(self.invt.invt_records.keys()):
            self.invt.invt_records[part]['qty'] += (orders[idx] * self.init_invt_quantities[idx])

    def apply_incoming_order(self, orders):
        for idx, part in enumerate(self.invt.invt_records.keys()):
            if self.invt.invt_records[part]['qty'] >= int(orders[idx]):
                self.invt.invt_records[part]['qty'] -= int(orders[idx])
                            
    def update_incoming_order_queue(self):
        latest_order = self.reformat_incoming_orders(generate_orders(self.order_path))
        self.incoming_orders = np.append(self.incoming_orders, np.expand_dims(latest_order, axis=0), axis = 0)
        curr_order = self.incoming_orders[0]
        self.incoming_orders = np.delete(self.incoming_orders, 0, 0)
        self.apply_incoming_order(curr_order)
        
    def update_previous_order_queue(self):
        latest_order = self.incoming_orders[0]
        self.previous_orders = np.append(self.previous_orders, np.expand_dims(latest_order, axis=0), axis = 0)        
        self.previous_orders = np.delete(self.previous_orders, 0, 0)
        
    def update_invt_history_queue(self):
        latest_invt = [self.invt.get_invt_quantities()[0]]
        self.invt_history = np.append(self.invt_history, np.expand_dims(latest_invt, axis=0), axis = 0)        
        self.invt_history = np.delete(self.invt_history, 0, 0)
        
    def update_refill_order_queue(self, refill_order, days_out):
        if days_out == self.ship_days:
            self.pending_refill_orders = np.append(self.pending_refill_orders, [[refill_order]], axis = 0)
            received_refill_orders = self.pending_refill_orders[0]
            self.pending_refill_orders = np.delete(self.pending_refill_orders, 0, 0)
            self.apply_refill_order(received_refill_orders)  
        else:
            self.pending_refill_orders[days_out] += np.expand_dims(refill_order, axis=0)     
                
    def observation(self):       
        invt_part_details = self.invt.get_invt_quantities()          
        invt_obs = invt_part_details / self.init_invt_quantities
        refill_obs = self.pending_refill_orders.flatten()
               
        return np.concatenate((invt_obs, refill_obs), axis = None)
                 
    def step(self, actions):
        self.step_count += 1
        self.update_refill_order_queue(actions[0], self.plane_days)
        self.update_refill_order_queue(actions[1], self.freight_days)
        self.update_refill_order_queue(actions[2], self.ship_days)
        self.update_incoming_order_queue()
        self.update_previous_order_queue()
        self.update_invt_history_queue()
        
        next_state = self.observation()
        done = False

        reward = self.calculate_reward(actions)
        if self.determine_terminate(self.invt.get_invt_quantities()) == True:
            done = True
        elif self.step_count == 1000:     
            done = True

        return next_state, reward, done, self.step_count
         
    def reset(self):
        self.__init__(self.name, self.invt_path, self.order_path)
        
        return self.observation()
    
    def determine_terminate(self, curr_invt):
        done = False
        projected_invt = curr_invt / self.init_invt_quantities
        if projected_invt.min() < self.min_stop_val or projected_invt.max() > self.max_stop_val:  
            done = True      
    
        return done
    
    def calculate_reward(self, actions):
        invt_copy = self.invt.get_invt_quantities()

        for i in range(self.ship_days):
            invt_copy += (self.pending_refill_orders[i] * self.init_invt_quantities)   
            invt_copy -= self.incoming_orders[i]
        
        if self.determine_terminate(invt_copy) == True:
            return -1
        
        reward = 1
        
        for idx, day in enumerate([self.plane_days, self.freight_days, self.ship_days]):
            if self.pending_refill_orders[day - 1] > 0 and actions[idx] > 0:
                if day == self.plane_days:
                    reward -= 5
                elif day == self.freight_days:
                    reward -= 2
                elif day == self.ship_days:
                    reward -= .5
            else:
                reward += 3
                
         
        return reward
        
                