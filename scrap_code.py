x = np.linspace(-5, 0, 1000)
y = np.exp(2*x)
plt.plot(x, y)

y = np.exp(x)
plt.plot(x, y)
y = 1/(1+np.exp(-1*x))
plt.plot(x, y, color='red')
# %%

y = np.log(1+np.exp(x))
plt.plot(x, y)


# y = x
# plt.plot(x, y)
y = 1/(1+np.exp(-1*x))
plt.plot(x, y, color='red')
y = 1/(1+np.exp(-10*x))
plt.plot(x, y, color='red')
# y = x**2
# plt.plot(x, y)
plt.grid()


# %%
from scipy.stats import beta
mean = 0.5

precision = 5
alpha_param = precision*mean
beta_param = precision*(1-mean)
gen_samples = np.random.beta(alpha_param, beta_param, 50)*35
plt.xlim(0, 35)

_ = plt.hist(gen_samples, bins=150)
np.random.beta(alpha_param, beta_param, 50).std()
