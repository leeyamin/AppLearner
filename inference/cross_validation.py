"""
This file is part of inferential analysis.
As part of cross validation, we need to run the main.py file multiple times.
This file is used to compare the results of multiple runs, and perform a paired t-test to determine if the differences
between the runs are statistically significant. If the differences are statistically significant, we can conclude that
the results are not stable, and depend on the data split.
"""

import numpy as np
from scipy.stats import ttest_rel
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


model_dir_name = 'check_delete'
num_runs = 2


def get_runs_metrics(model_dir_name, num_runs):
    results_mae = []
    results_mape = []
    results_mse = []
    results_rmse = []

    for idx_run, run in enumerate(range(num_runs)):
        file_path = os.path.join('..', 'output', f'{model_dir_name}_{idx_run}', 'val_all_metrics_data.csv')

        best_metrics = pd.read_csv(file_path)

        mae = best_metrics['epoch_maes'].values
        mape = best_metrics['epoch_mapes'].values
        mse = best_metrics['epoch_mses'].values
        rmse = best_metrics['epoch_rmses'].values

        results_mae.append(mae)
        results_mape.append(mape)
        results_mse.append(mse)
        results_rmse.append(rmse)

    return results_mae, results_mape, results_mse, results_rmse


def calc_descriptive_statistics(results_metric):
    mean_metric = np.mean(results_metric)
    std_metric = np.std(results_metric)

    return mean_metric, std_metric


def run_paired_t_test(results_metrics):
    t_stat, p_value = ttest_rel(results_metrics[0], results_metrics[1])

    return t_stat, p_value


def print_colored_text(text, color):
    if color == 'red':
        color_code = "1;31"
    elif color == 'green':
        color_code = "1;32"
    else:
        raise NotImplementedError(f"Color {color} is not supported.")
    print(f"\033[{color_code}m{text}\033[0m")


if __name__ == '__main__':
    results_mae, results_mape, results_mse, results_rmse = get_runs_metrics(model_dir_name, num_runs)

    mean_mae, std_mae = calc_descriptive_statistics(results_mae)
    mean_mape, std_mape = calc_descriptive_statistics(results_mape)
    mean_mse, std_mse = calc_descriptive_statistics(results_mse)
    mean_rmse, std_rmse = calc_descriptive_statistics(results_rmse)

    t_stat_mae, p_value_mae = run_paired_t_test(results_mae)
    t_stat_mape, p_value_mape = run_paired_t_test(results_mape)
    t_stat_mse, p_value_mse = run_paired_t_test(results_mse)
    t_stat_rmse, p_value_rmse = run_paired_t_test(results_rmse)

    if any(p_value < 0.05 for p_value in [p_value_mae, p_value_mape, p_value_mse, p_value_rmse]):
        print_colored_text("At least one of the differences is statistically significant.", 'red')
    else:
        print_colored_text("No statistically significant differences observed.", 'green')
