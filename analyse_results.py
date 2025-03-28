import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# File path
file_path = "training_results.txt"

# Updated regex pattern to extract the two dictionaries
dict_pattern = re.compile(r"Run \d+: (\{.*?\}), Credible Intervals: (\{.*?\})")


# Function to clean up tensor values
def clean_tensor_values(text):
    return re.sub(r"tensor\(([\d\.]+), device='cuda:0'\)", r"\1", text)


# Storage for parsed data
runs = []

# Read and process file
with open(file_path, "r") as file:
    for line in file:
        matches = dict_pattern.findall(line)
        if matches:
            try:
                # Clean tensor values
                cleaned_metrics = clean_tensor_values(matches[0][0])
                cleaned_intervals = clean_tensor_values(matches[0][1])

                # Evaluate safely
                metric_dict = eval(cleaned_metrics)
                credible_intervals_dict = eval(cleaned_intervals)

                runs.append((metric_dict, credible_intervals_dict))
            except Exception as e:
                print(f"Error parsing line:\n{line.strip()}\n{e}")

if not runs:
    print("No valid data extracted from results.txt! Check file formatting.")
    exit()

metrics = {key: [] for key in runs[0][0].keys()}  # Initialize with keys from the first run

# Populate metric lists
for metric_dict, _ in runs:
    for metric, value in metric_dict.items():
        metrics[metric].append(float(value))


# Compute and print statistics
for metric, values in metrics.items():
    mean = np.mean(values)
    std_dev = np.std(values)
    print(f"{metric}: Mean = {mean:.4f}, Std Dev = {std_dev:.4f}")


def plot_boxplots(metrics, filename="boxplot.png"):
    """Generates boxplots for each metric and saves the plot as an image."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[metrics[key] for key in metrics.keys()])
    plt.xticks(ticks=range(len(metrics.keys())), labels=list(metrics.keys()), rotation=20)
    plt.ylabel("Metric Values")
    plt.title("Boxplots of Metrics for UNet")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    # Save the figure
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()


plot_boxplots(metrics)


def plot_credible_intervals(runs, filename="credible_intervals.png"):
    """
    Plots the evolution of credible intervals for each metric across runs,
    along with a horizontal line showing the mean for each metric.
    The mean line will match the colour of the credible interval plot for each metric.
    The mean values will be displayed next to the y-axis where the mean lines intersect.
    """
    # Dictionary to map original metric names to the new legend names
    metric_renaming = {
        'dice_score': 'Dice Score',
        'mIoU': 'Mean IoU',
        'precision': 'Precision',
        'recall': 'Recall'
    }

    metrics = runs[0][1].keys()  # Get metric names from the first run
    num_runs = len(runs)

    plt.figure(figsize=(12, 6))

    # Define a colour palette to match the plot colours
    palette = sns.color_palette("tab10", n_colors=len(metrics))

    for idx, metric in enumerate(metrics):
        lower_bounds = [run[1][metric][0] for run in runs]
        upper_bounds = [run[1][metric][1] for run in runs]
        midpoints = [(l + u) / 2 for l, u in zip(lower_bounds, upper_bounds)]

        # Select colour for the metric based on its index in the list
        colour = palette[idx]

        # Plot the midpoints and fill the credible interval with the selected colour
        plt.plot(range(1, num_runs + 1), midpoints, label=f'{metric_renaming.get(metric, metric)} (midpoint)', color=colour)
        plt.fill_between(range(1, num_runs + 1), lower_bounds, upper_bounds, color=colour, alpha=0.2)

        # Calculate and plot the mean of the metric
        metric_values = [run[0][metric] for run in runs]  # Extract values from the metric dictionary
        mean_value = np.mean(metric_values)
        plt.axhline(y=mean_value, color=colour, linestyle='--', label=f'{metric_renaming.get(metric, metric)} Mean')

        # Add the mean value on the y-axis next to the mean line
        plt.text(-5, mean_value, f'{mean_value:.4f}', color=colour, va='center', ha='right', fontsize=7)

    plt.xlabel("Run Number")
    plt.ylabel("Metric Value")
    plt.title("Bayesian Credible Intervals Over Training Runs with Mean Values")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()


plot_credible_intervals(runs)