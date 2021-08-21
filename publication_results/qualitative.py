import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
from matplotlib import cm
from scipy.stats import beta

# %%
"""
Driver generation - beta distribution with aggressiveness levels
"""
35*0.2
35*0.3
x = np.linspace(0.01, 0.99, 100)
color_i = 0
range_means = np.linspace(0.01, 0.99, 10)
colors = [cm.rainbow(i) for i in np.linspace(0, 1, len(range_means))]
alpha_val_plot = 0.8
precision = 15

for mean in range_means:
    mean = 0.3
    alpha_param = mean*precision
    beta_param = precision*(1-mean)
    p = beta.pdf(x, alpha_param, beta_param)
    plt.plot(x, p, color=colors[color_i], linewidth=1)
    plt.fill_between(x, p, color=colors[color_i], alpha=alpha_val_plot)
    color_i += 1


# %%
