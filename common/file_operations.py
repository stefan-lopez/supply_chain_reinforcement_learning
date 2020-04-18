import csv
import random

def load_csv_as_dict(filepath, unique_key):
    with open(filepath) as file:
        reader = csv.reader(file)
        headers = next(reader)
        target_dict = {}
        for index, header in enumerate(headers):
            if header == unique_key:
                target_index = index
    
        for row in reader:
            target_dict[row[target_index]] = {}
            for index, value in enumerate(row):
                if index != target_index:
                    target_dict[row[target_index]][headers[index]] = value
                    
        return target_dict
    
def read_generated_orders(file_path):
    generated_orders = []
    with open(file_path, "r", newline='') as file_in:
        reader = csv.reader(file_in, )
        for row in reader:
            generated_orders.append(row)
            
    return generated_orders
    
    
def generate_orders(example_filepath):
    new_orders = []
    with open(example_filepath) as file:
        reader = csv.reader(file)
        new_orders.append(next(reader))
        counter = 1
        for row in reader:
            if random.random() < .1:
                pass
            elif random.random() > .9:
                row[4] = round(int(row[4]) * 2, 0)
                row[0] = counter
                counter += 1
                new_orders.append(row)
            else:
                multiply_factor = ((int(row[0]) % 4) + 1) * .2
                lower_bound = round(int(row[4]) * (1 - multiply_factor), 0)
                higher_bound = round(int(row[4]) * (1 + multiply_factor), 0)
                row[4] = random.randrange(lower_bound, higher_bound)
                row[0] = counter
                counter += 1
                new_orders.append(row)
         
    return new_orders
            
    
    