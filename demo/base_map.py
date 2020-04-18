import pygame
from transit_routes import DisplayObject

class BaseMap():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.image_dict = {
        'background' : pygame.image.load('../images/custom_map.png').convert_alpha(),
        'dc' : pygame.image.load('../images/dc-icon5.png').convert_alpha(),
        'plane' : pygame.image.load('../images/plane-icon.png').convert_alpha(),
        'truck' : pygame.image.load('../images/truck-icon.png').convert_alpha(),
        'ship' : pygame.image.load('../images/ship-icon.png').convert_alpha(),
        'train' : pygame.image.load('../images/train-icon.png').convert_alpha()
        }        
        self.dc_dict = {
        'Supplier' : DisplayObject(self.image_dict['dc'], (1560, 225)).resize(45,45),
        'New York' : DisplayObject(self.image_dict['dc'], (560, 275)).resize(45,45),
        'Charlotte' : DisplayObject(self.image_dict['dc'], (340, 505)).resize(45,45),
        'Nashville' : DisplayObject(self.image_dict['dc'], (150, 450)).resize(45,45),
        'Chicago' : DisplayObject(self.image_dict['dc'], (123, 278)).resize(45,45),
        'Toronto' : DisplayObject(self.image_dict['dc'], (360, 160)).resize(45,45)
        } 
        self.background = DisplayObject(self.image_dict['background'], (0,0)).resize(self.width, self.height)
        
    def display_background(self, surface):
        surface.blit(self.background.image, self.background.coordinates)
        
        return self
        
    def display_dc_locations(self, surface):
        for dc in self.dc_dict.keys():
            surface.blit(self.dc_dict[dc].image, self.dc_dict[dc].coordinates)