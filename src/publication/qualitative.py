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
MEDIUM_SIZE = 10
LARGE_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
# %%
"""
Driver generation - beta distribution with aggressiveness levels
"""
# plt.style.use('default')
x = np.linspace(0.0, 1, 100)
color_i = 0
range_means = np.linspace(0.1, 0.9, 10)
colors = [cm.rainbow(i) for i in np.linspace(0, 1, len(range_means))]
alpha_val_plot = 0.3
fig = plt.figure(figsize=(5, 3))
precision = 15
for mean in range_means:
    alpha_param = precision*mean
    beta_param = precision*(1-mean)

    p = beta.pdf(x, alpha_param, beta_param)
    plt.plot(x, p, color=colors[color_i], linewidth=2)
    plt.fill_between(x, p, color=colors[color_i], alpha=alpha_val_plot)
    plt.xticks([0, 0.5, 1])
    plt.xlabel('$\psi$')
    plt.ylabel('Density')
    plt.ylim(0, 8)
    # plt.xlim(0, 1)
    plt.minorticks_off()
    color_i += 1
plt.savefig("beta_densities.png", dpi=500)



# %%
precision = 15
mean = 0.34
color_i = 0
alpha_param = precision*mean
beta_param = precision*(1-mean)
p = beta.pdf(x, alpha_param, beta_param)
plt.plot(x*45, p, color=colors[color_i], linewidth=1)

# mean = 0.3
# precision = 10
#
# alpha_param = precision*mean
# beta_param = precision*(1-mean)
# p = beta.pdf(x, alpha_param, beta_param)
# plt.plot(x*35, p, color=colors[color_i], linewidth=1)
