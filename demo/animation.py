import torch
import sys
sys.path.append('../')
from envs.reg_dist_center import RegDistCenter
from common.models import ModelA2C
from demo.transit_routes import TransitManager
from demo.screen_inset import ScreenInset
from demo.screen_display import ScreenDisplay
from demo.base_map import BaseMap

def main():
    WIN_WIDTH = 1239
    WIN_HEIGHT = 735
    FPS = 30
    SECONDS_PER_ACTION = 1
    REG_MODEL_PATH = "../saves/best/best_+2178.800_4720000.dat"
    DC_MODEL_PATH = "../saves/best/best_+1977.900_3692000.dat" 
    INVT_PATH = "../data/starting_invt_brake_pads.csv"
    ORDER_PATH = "../data/incoming_orders.csv" 
    ENV_NAME = "New York"
    action_duration = FPS * SECONDS_PER_ACTION
       
    env = RegDistCenter(dc_model_path = DC_MODEL_PATH, name = ENV_NAME, invt_path = INVT_PATH, order_path = ORDER_PATH)
    model_dict = torch.load(REG_MODEL_PATH)
    model = ModelA2C(env.observation_space.shape[0], env.action_space.shape[0]).to("cpu")
    model.load_state_dict(model_dict)
    
    screen = ScreenDisplay(WIN_WIDTH, WIN_HEIGHT, FPS) 
    base_map = BaseMap(WIN_WIDTH, WIN_HEIGHT)
    transit_manager = TransitManager(env, model, base_map, FPS)
    inset = ScreenInset(env, screen.surface)    
    curr_target_dc = env.name
    
    while not screen.done:
        curr_target_dc = screen.monitor_mouse(base_map.dc_dict, curr_target_dc)   
        base_map.display_background(screen.surface)       
        transit_manager.display_transit_routes(screen, action_duration)
        base_map.display_dc_locations(screen.surface)
        inset.draw(curr_target_dc)
        screen.update()

if __name__ == '__main__':
    main()