import ptan
import torch
import numpy as np
from screen_display import DisplayObject

class TransitManager():
    def __init__(self, env, net, base_map, fps):
        self.env = env
        self.net = net
        self.dc_dict = base_map.dc_dict
        self.image_dict = base_map.image_dict
        self.fps = fps
        self.obs = env.reset()
        self.curr_routes = {}
        
    def gather_reg_actions(self):
        obs_v = ptan.agent.float32_preprocessor([self.obs]).to('cpu')
        mu_v, var_v, _ = self.net(obs_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        action = np.random.normal(mu, sigma)
        action = np.clip(action, 0, 1)
        obs, _, _, _ = self.env.step(action[0])
        
        return action, obs 
           
    def gather_dc_actions(self):
        action_dict = {}
        for name in self.env.dc_dict.keys():
            action_dict[name] = self.env.dc_dict[name]["curr_orders"]
    
        return action_dict
    
    def add_new_route(self, dc, action_type, frame_count):
        route_id = str(frame_count) + dc + action_type['mode']
        if dc == self.env.name:
            new_route = TransitRoute(self.dc_dict['Supplier'], self.dc_dict[self.env.name], self.image_dict, action_type['travel_days'], action_type['sea_mode'], frame_count, self.fps)
        else:
            new_route = TransitRoute(self.dc_dict[self.env.name], self.dc_dict[dc], self.image_dict, action_type['travel_days'], action_type['mode'], frame_count, self.fps)
            
        self.curr_routes[route_id] = new_route        
        
    def transform_actions_to_routes(self, frame_count):
        reg_actions, self.obs = self.gather_reg_actions()   
        dc_actions = self.gather_dc_actions()
        dc_actions[self.env.name] = reg_actions
        
        for dc in dc_actions.keys():
            for idx, action_val in enumerate(dc_actions[dc][0]):
                action_params = {'0' : {'mode' : 'plane', 'sea_mode' : 'plane', 'travel_days' : 2},
                                '1' : {'mode' : 'truck', 'sea_mode' : 'plane', 'travel_days' : 7},
                                '2' : {'mode' : 'train', 'sea_mode' : 'ship', 'travel_days' : 14}}
                
                action_type = action_params[str(idx)]
                if action_val > 0:
                    self.add_new_route(dc, action_type, frame_count)
                     
        
    def display_transit_routes(self, screen_display, action_duration):
        if screen_display.frame_count % action_duration == 0:            
            self.transform_actions_to_routes(screen_display.frame_count)
            
        for route in self.curr_routes.copy().keys():
            route_coordinates, done_flag = self.curr_routes[route].calculate_new_coordinates(screen_display.frame_count)
            if done_flag == False:
                screen_display.surface.blit(self.curr_routes[route].vessel.image, route_coordinates)    
            else:
                del self.curr_routes[route]

class TransitRoute():
    def __init__(self, source, dest, image_dict, travel_days, mode, frame_count, fps):
        self.source = source
        self.dest = dest
        self.image_dict = image_dict
        self.travel_days = travel_days
        self.travel_frames = travel_days * fps
        self.mode = mode
        self.init_frame_count = frame_count
        self.x_distance = dest.coordinates[0] - source.coordinates[0]
        self.y_distance = dest.coordinates[1] - source.coordinates[1]
        self.vessel = self.create_vessel()

    def create_vessel(self):
        mode_dict = {
        'plane' : DisplayObject(self.image_dict['plane'], self.source.coordinates).resize(35,35).flip(),
        'truck' : DisplayObject(self.image_dict['truck'], self.source.coordinates).resize(40,40).flip(),
        'ship' : DisplayObject(self.image_dict['ship'], self.source.coordinates).resize(45,45),
        'train' : DisplayObject(self.image_dict['train'], self.source.coordinates).resize(40,40).flip()
        }
        
        return mode_dict[self.mode]
        
    def calculate_new_coordinates(self, curr_frame_count):
        frames_elapsed = curr_frame_count - self.init_frame_count
        percent_completed = frames_elapsed / self.travel_frames
        x_distance_travelled = int(self.x_distance * percent_completed)
        y_distance_travelled = int(self.y_distance * percent_completed)
        new_x_coordinates = self.source.coordinates[0] + x_distance_travelled
        new_y_coordinates = self.source.coordinates[1] + y_distance_travelled
        done_flag = frames_elapsed > self.travel_frames
        
        return (new_x_coordinates, new_y_coordinates), done_flag
    