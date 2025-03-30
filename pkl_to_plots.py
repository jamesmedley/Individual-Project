import matplotlib.pyplot as plt
from skopt import load
from skopt.plots import plot_convergence, plot_objective, plot_evaluations

result = load("hp_optim_results_F5.3.pkl")

# Best hyperparameters
best_hyperparameters = {
    "Learning Rate": result.x[0],
    "Weight Decay": result.x[1],
    "Gradient Clipping": result.x[2]
}

# Print the best hyperparameters
print("Best Hyperparameters:")
for param, value in best_hyperparameters.items():
    print(f"{param}: {value}")

# Save the best hyperparameters to a text file
with open('best_hyperparameters.txt', 'w') as f:
    f.write("Best Hyperparameters:\n")
    for param, value in best_hyperparameters.items():
        f.write(f"{param}: {value}\n")

# Define parameter names
param_names = ["Learning Rate", "Weight Decay", "Gradient Clipping"]

# Generate and save convergence plot
plt.figure(figsize=(12, 8))
plot_convergence(result)
plt.savefig('convergence_plot.png', dpi=300)
plt.close()

# Generate and save objective function plot
plt.figure(figsize=(25, 25))
plot_objective(result, dimensions=param_names, n_points=100, levels=30, size=5)
plt.savefig('objective_plot.png', dpi=300)
plt.close()

# Generate and save evaluations plot
plt.figure(figsize=(25, 25))
plot_evaluations(result, dimensions=param_names, size=5)
plt.savefig('evaluations_plot.png', dpi=300)
plt.close()
