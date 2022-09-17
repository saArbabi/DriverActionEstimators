import matplotlib.pyplot as plt
from importlib import reload
import numpy as np
from matplotlib.pyplot import cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Polygon
import dill

class Viewer():
    def __init__(self, config):
        self.config  = config
        # self.fig = plt.figure(figsize=(6, 2))

        self.end_merge_box = [Rectangle((config['merge_zone_end']+40, 0), \
                            200, config['lane_width'])]
        self.ramp_box = [Polygon(np.array([[config['merge_zone_end']-30, 0],
                                             [config['merge_zone_end']+40, config['lane_width']],
                                             [config['merge_zone_end']+40, 0]]))]
        self.ramp_bound = [Rectangle((0, config['lane_width']-0.25), \
                            config['merge_lane_start'], 0.5)]

        self.logged_var = {}

    def set_up_fig(self):
        self.fig, self.axs = plt.subplots(figsize=(30, 10), nrows=5, ncols=1)
        for ax in self.axs:
            ax.set_ylabel(r'Lat. pos. (m)')
            ax.set_yticks([1, 3, 5, 7])
            ax.set_xlim(0, self.config['lane_length'])

        self.axs[-1].set_xlabel(r'Longitudinal position (m)')

    def draw_road(self):
        for ax in self.axs:
            lane_cor = self.config['lane_width']*self.config['lanes_n']
            ax.hlines(0, 0, self.config['lane_length'], colors='k', linestyles='solid')

            ax.hlines(lane_cor, 0, self.config['lane_length'],
                                                        colors='k', linestyles='solid')

            # Create patch collection with specified colour/alpha
            pc_end = PatchCollection(self.end_merge_box, hatch='/', alpha=0.3, color='k')
            pc_ramp = PatchCollection(self.ramp_box, hatch='/', alpha=0.3, color='k')
            pc_ramp_bound = PatchCollection(self.ramp_bound, hatch='/', alpha=0.3, color='k')
            ax.add_collection(pc_end)
            ax.add_collection(pc_ramp)
            ax.add_collection(pc_ramp_bound)
            ax.hlines(self.config['lane_width'], self.config['merge_lane_start'], self.config['merge_zone_end']+40,
                                    colors='k', linestyles='--', linewidth=3)
