import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
from matplotlib import cm
from scipy.stats import beta
# %%
""" plot setup
"""
fig = plt.figure(figsize=(5, 2))

plt.style.use('ieee')
# plt.style.use(['science','ieee'])

plt.rcParams["font.family"] = "Times New Roman"
MEDIUM_SIZE = 11
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels

"""
Driver generation - beta distribution with aggressiveness levels
"""
# plt.style.use('default')
x = np.linspace(0.0, 1, 1000)
color_i = 0
range_means = [0.1, 0.3, 0.5, 0.7, 0.9]
colors = [cm.rainbow(i) for i in np.linspace(0, 1, len(range_means))]
alpha_val_plot = 0.3
precision = 15
for mean in range_means:
    alpha_param = precision*mean
    beta_param = precision*(1-mean)

    p = beta.pdf(x, alpha_param, beta_param)
    plt.plot(x, p, color=colors[color_i], linewidth=2, linestyle='-')
    plt.fill_between(x, p, color=colors[color_i], alpha=alpha_val_plot)

    x_loc = x[np.where(p == p.max())][0]
    plt.text(x_loc-0.08, p.max()+0.2, '$\psi='+str(mean)+'$', color=colors[color_i])

    # plt.xlim(0, 1)
    color_i += 1
plt.ylim(0, 8)
plt.minorticks_off()
plt.xlabel('$\psi$')
plt.ylabel('Probability Density')
plt.xticks([0, 0.5, 1])
plt.yticks([0, 4, 8])

# plt.savefig("beta_densities.png", dpi=500)


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
