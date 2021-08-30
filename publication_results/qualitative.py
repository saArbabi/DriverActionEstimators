import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
from matplotlib import cm
from scipy.stats import beta
# %%

plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
params = {
          'font.size' : 20,
          'font.family' : 'EB Garamond',
          }
plt.rcParams.update(params)
plt.style.use(['science','ieee'])
# %%
"""
Driver generation - beta distribution with aggressiveness levels
"""
# plt.style.use('default')

x = np.linspace(0.01, 0.99, 100)
color_i = 0
range_means = np.linspace(0.01, 0.99, 10)
colors = [cm.rainbow(i) for i in np.linspace(0, 1, len(range_means))]
alpha_val_plot = 0.8
precision = 10
fig = plt.figure(figsize=(4, 3))
var = 0.03
for mean in range_means:
    # plt.figure((3, 3))
    alpha_param = (((1-mean)/var)-1/mean)*mean**2
    beta_param = alpha_param*((1/mean)-1)
    p = beta.pdf(x, alpha_param, beta_param)
    plt.plot(x, p, color=colors[color_i], linewidth=1)
    plt.fill_between(x, p, color=colors[color_i], alpha=alpha_val_plot)
    plt.xticks([0, 0.5, 1])
    plt.xlabel('$\psi$')
    plt.ylabel('density')
    plt.ylim(0, 8)
    # plt.xlim(0, 1)
    plt.minorticks_off()
    color_i += 1
# plt.savefig("beta_densities.png", dpi=500)



# %%
mean = 0.95
color_i = 0
var = 0.02
alpha_param = (((1-mean)/var)-1/mean)*mean**2
beta_param = alpha_param*((1/mean)-1)
p = beta.pdf(x, alpha_param, beta_param)
plt.plot(x*35, p, color=colors[color_i], linewidth=1)
