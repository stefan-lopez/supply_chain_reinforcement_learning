import pygame

class ScreenDisplay():
    def __init__(self, width, height, fps):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0
        self.create_display()  
        self.done = False 
               
    def create_display(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.surface = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Supply Chain AI')
        
        return self
            
    def update(self):
        self.clock.tick(self.fps) 
        pygame.display.update()
        self.frame_count += 1    
        
        return self    
            
    def monitor_mouse(self, location_dict, curr_target_dc):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
                break
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_coordinates = pygame.mouse.get_pos()
                    for dc_icon in location_dict:
                        if mouse_coordinates[0] - location_dict[dc_icon].coordinates[0] < 45 and mouse_coordinates[1] - location_dict[dc_icon].coordinates[1] < 45:
                            curr_target_dc = dc_icon            
                            
        return curr_target_dc
    

class DisplayObject():
    def __init__(self, image, coordinates):
        self.image = image
        self.coordinates = coordinates
    
    def resize(self, width, height):
        self.image = pygame.transform.scale(self.image, (width, height))
        return self
    
    def flip(self):
        self.image = pygame.transform.flip(self.image, True, False)
        return self
    