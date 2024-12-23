# Monte-Carlo Simulation

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable
from tqdm import tqdm

# Stable distribution parameters
alpha = 1.7
beta = 0.0
gamma = 1.0
delta = 1.0

# Data generation
n_observations = 750
n_days = 10
num_trials = [100, 200, 500, 1000, 2000, 5000, 10000, 50000, 100000, 500000]

# Price P_i generation 
np.random.seed(42)
P = levy_stable.rvs(alpha, beta, loc=delta, scale=gamma, size=n_observations+1)

# Calculating 1-day returns
r1 = np.array([(P[i + 1] - P[i]) / P[i] for i in range(n_observations - 1)])

# Calculating 10-day returns
r10 = np.array([(P[i + n_days] - P[i]) / P[i] for i in range(n_observations - n_days)])

# Calculating 0.01-quantiles
percentile_1 = np.percentile(r10, 1)


# Plotting a histogram for r10
plt.hist(r10, bins=50, density=True, alpha=0.7, color='blue', label="Distribution of 10-day returns")
plt.axvline(x=percentile_1, color='red', linestyle='--', label=f"1% quantile = {percentile_1:.2f}")
plt.title("Distribution of 10-day returns")
plt.xlabel("10-day returns")
plt.ylabel("Density")
plt.xscale("symlog", linthresh=1)
plt.legend(loc="upper center", bbox_to_anchor=(0.45, -0.07), ncol=1, frameon=True)
plt.grid(True)
#plt.show()

# An array for storing standard errors
standard_errors = []

# Iterations for different N
for N in tqdm(num_trials):
    quantiles = []
    for _ in range(N):
        # Price generation
        P_sample = levy_stable.rvs(alpha, beta, loc=delta, scale=gamma, size=n_observations+1)
        # Calculating 10-day returns
        r10_sample = np.array([(P_sample[i + n_days] - P_sample[i]) / P_sample[i] for i in range(n_observations - n_days)])
        # Quantile
        quantiles.append(np.percentile(r10_sample, 1))
    
    # Average quantile and variance
    mean_quantile = np.mean(quantiles)
    variance = np.var(quantiles, ddof=1)
    se = np.sqrt(variance) / np.sqrt(N)  # Standard error
    standard_errors.append(se)

# Plotting the graph
plt.figure(figsize=(10, 6))
plt.plot(num_trials, standard_errors, marker='o', label="Standard error SE")
plt.axhline(y=np.abs(0.01 * np.mean(quantiles)), color='r', linestyle='--', label=r"$\epsilon \cdot q$")
plt.title("The dependence of the SE on the number of iterations N")
plt.xlabel("Number of iterations (N)")
plt.ylabel("Standard error SE")
plt.xscale("log")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Convergence check
mean_quantile = np.mean(quantiles)
std_error = np.std(quantiles) / np.sqrt(2000)

print(f"Score 0.01-quantiles: {mean_quantile:.6f}")
print(f"Average error: {std_error:.6f}")
print(f"< {abs(0.01 * np.mean(quantiles)):.6f}")

plt.show()
