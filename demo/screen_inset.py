import numpy as np
import pygame
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg   
from matplotlib import rcParams
import matplotlib

class ScreenInset():
    def __init__(self, env, display_surface):
        self.env = env
        self.display_surface = display_surface
        self.order_data = None
        self.invt_data = None
        self.order_limit = None
        self.invt_limit = None
        self.avg_order_count = None
        self.avg_invt_count = None
        self.invt_turnover_days = None
        self.invt_turnover_ratio = None
        self.fig = None
        self.axs = None
                
    def gather_inset_data(self, dc_name):
        if dc_name == self.env.name:
            self.invt_data = self.env.invt_history
            self.order_data = self.env.previous_orders
            self.invt_limit = self.env.init_invt_quantities[0] * 2
            self.order_limit = 6000
        else:
            self.invt_data = self.env.dc_dict[dc_name]["env"].invt_history
            self.order_data = self.env.dc_dict[dc_name]["env"].previous_orders
            self.invt_limit = self.env.dc_dict[dc_name]["env"].init_invt_quantities[0] * 2
            self.order_limit = 1000
        
        self.avg_order_count = self.order_data.sum() / self.order_data.shape[0]  
        self.avg_invt_count = self.invt_data.sum() / self.invt_data.shape[0] 
        self.invt_turnover_days = self.avg_invt_count / self.avg_order_count    
        self.invt_turnover_ratio = 365 / self.invt_turnover_days
        
    def render_text(self, title):
        font = pygame.font.SysFont('courier' , 16, True)
        titlefont = pygame.font.SysFont('courier' , 22, True)
        text_dict = {}
        text_dict['avg_order_txt'] = font.render("Avg Order Count: " + str(round(self.avg_order_count, 1)), True, (0, 0, 0))
        text_dict['avg_invt_txt'] = font.render("Avg Invt Count: " + str(round(self.avg_invt_count, 1)), True, (0, 0, 0))
        text_dict['avg_turnover_days_txt'] = font.render("Invt Turnover Days: " + str(round(self.invt_turnover_days, 1)), True, (0, 0, 0))
        text_dict['avg_turnover_ratio_txt'] = font.render("Invt Turnover Ratio: " + str(round(self.invt_turnover_ratio, 1)), True, (0, 0, 0))
        text_dict['title_txt'] = titlefont.render(title, True, (0, 0, 0))
        
        return text_dict
    
    def display_text(self, text_dict):
        text_rect = text_dict['title_txt'].get_rect()
        text_rect.center = (900,400)
        self.display_surface.blit(text_dict['avg_order_txt'], (650,615))
        self.display_surface.blit(text_dict['avg_invt_txt'], (650,645))
        self.display_surface.blit(text_dict['avg_turnover_days_txt'], (920,615))
        self.display_surface.blit(text_dict['avg_turnover_ratio_txt'], (920,645))
        self.display_surface.blit(text_dict['title_txt'], text_rect)
            
    def draw(self, title):
        plt.close()
        self.gather_inset_data(title)
        self.create_figure()
        plot = self.create_order_plot()
        pygame.draw.rect(self.display_surface, (255,235,102), (610,370,590,330)) 
        text_dict = self.render_text(title)
        self.display_text(text_dict)
        self.display_surface.blit(plot, (625,405))
        plt.subplots_adjust(top=0.5)

    def create_figure(self):
        rcParams.update({'figure.autolayout' : True, "figure.facecolor" : (1,.92,.4), "font.weight" : "bold", "font.family" : "Courier New"})
        self.fig, self.axs = plt.subplots(nrows = 1, ncols = 2, figsize = (5.5, 2))
        self.axs[0].set_title('Orders', fontweight = "bold", fontsize = 12)
        self.axs[1].set_title('Inventory', fontweight = "bold", fontsize = 12)
                
    def create_order_plot(self):
        matplotlib.use("Agg")
        canvas = agg.FigureCanvasAgg(self.fig)
        self.axs[0].plot(self.order_data)
        self.axs[1].plot(self.invt_data)
        plt.locator_params(axis = 'y', nbins = 4)
        plt.locator_params(axis = 'x', nbins = 4)
        self.axs[0].set_xticklabels([])
        self.axs[1].set_xticklabels([])
        self.axs[0].set_yticks(np.arange(0, self.order_limit + 1, step = self.order_limit / 4))
        self.axs[1].set_yticks(np.arange(0, self.invt_limit + 1, step = self.invt_limit / 4))
        canvas.draw()
        renderer = canvas.get_renderer()
    
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        
        return pygame.image.fromstring(raw_data, size, "RGB")