import sys
sys.path.append('../')
from common.file_operations import load_csv_as_dict
import numpy as np

class Inventory:
    def __init__(self, dist_center):
        self.owner = dist_center
        self.invt_records = {}
        
    def load_invt_file(self, filepath):
        init_invt_dict = load_csv_as_dict(filepath, "invt_id")
        for invt_details in init_invt_dict.values():
            if invt_details['dist_center'] == self.owner:
                invt_details['qty'] = int(invt_details['qty'])
                invt_details['unit_cost'] = float(invt_details['unit_cost'])
                invt_details['unit_price'] = float(invt_details['unit_price'])
                self.invt_records[invt_details['part_number']] = invt_details  
                
        return self  
    
    def calculate_total_price_on_hand(self):
        total_price_on_hand = 0
        for part_details in self.invt_records.values():
            if part_details['qty'] != None and part_details['unit_price'] != None:
                total_price_on_hand += part_details['qty'] * part_details['unit_price']
          
        return total_price_on_hand
          
    def get_invt_quantities(self):
        qty_list = []
        
        for part_details in self.invt_records.values():
            qty_list.append(part_details['qty'])
                
        return np.asarray(qty_list).astype('float64')