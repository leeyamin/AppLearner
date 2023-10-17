import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import darts
from darts.metrics import mae, mape, mse, rmse

import src.dataset.utils as dataset_utils
import src.config as config


def format_number(number):
    return f"{number:.4f}"


def train_one_epoch(epoch_idx, model, train_dfs, scaler):
    epoch_train_maes = []
    epoch_train_mapes = []
    epoch_train_mses = []
    epoch_train_rmses = []

    # train on each df (one epoch = one pass through all dfs)
    for df_idx in train_dfs['source_df_idx'].unique():
        train_df_dataset = train_dfs[train_dfs['source_df_idx'] == df_idx]

        darts_train_dataset = darts.TimeSeries.from_dataframe(train_df_dataset,
                                                              time_col='time',
                                                              value_cols='sample')

        model.fit(series=darts_train_dataset, past_covariates=None, verbose=False)
        train_forecast = model.predict(n=len(darts_train_dataset) - config.look_back,
                                       series=darts_train_dataset[:config.look_back])

        train_forecast.values = (
            dataset_utils.re_transform_and_re_scale_data(darts_values=train_forecast.values(),
                                                         transformation_method=config.transformation_method,
                                                         scale_method=config.scale_method,
                                                         scaler=scaler)
        )
        darts_train_dataset.values = (
            dataset_utils.re_transform_and_re_scale_data(darts_values=darts_train_dataset.values(),
                                                         transformation_method=config.transformation_method,
                                                         scale_method=config.scale_method,
                                                         scaler=scaler)
        )

        assert len(darts_train_dataset[config.look_back:]) == len(train_forecast)
        train_mae = mae(darts_train_dataset[config.look_back:], train_forecast)
        train_mape = mape(darts_train_dataset[config.look_back:], train_forecast)
        train_mse = mse(darts_train_dataset[config.look_back:], train_forecast)
        train_rmse = rmse(darts_train_dataset[config.look_back:], train_forecast)

        epoch_train_maes.append(train_mae)
        epoch_train_mapes.append(train_mape)
        epoch_train_mses.append(train_mse)
        epoch_train_rmses.append(train_rmse)

        train_df_metrics_dict = {
            "mae": train_mae,
            "mape": train_mape,
            "mse": train_mse,
            "rmse": train_rmse
        }

        # plot every 500th df (TODO: create a plot function)
        if df_idx % 500 == 0:
            train_forecast.plot(label='Forecast', color='blue', alpha=0.5)
            darts_train_dataset.plot(label='Actual', color='black')
            plt.legend()
            plt.title(f'({config.model_name}) Epoch ({epoch_idx}) Train Forecasting (df={df_idx}):\n '
                      f'MAE = {train_df_metrics_dict["mae"]:.4f}'
                      f' MAPE = {train_df_metrics_dict["mape"]:.4f}'
                      f' MSE = {train_df_metrics_dict["mse"]:.4f}'
                      f' RMSE = {train_df_metrics_dict["rmse"]:.4f}'
                      )
            plt.show()
            plt.close()

    epoch_train_metrics_dict = {
        "mae": format_number(sum(epoch_train_maes) / len(epoch_train_maes)),
        "mape": format_number(sum(epoch_train_mapes) / len(epoch_train_mapes)),
        "mse": format_number(sum(epoch_train_mses) / len(epoch_train_mses)),
        "rmse": format_number(sum(epoch_train_rmses) / len(epoch_train_rmses))
    }

    return epoch_train_metrics_dict


def validate_one_epoch(epoch_idx, model, val_dfs, scaler):
    epoch_val_maes = []
    epoch_val_mapes = []
    epoch_val_mses = []
    epoch_val_rmses = []

    for df_idx in val_dfs['source_df_idx'].unique():
        val_df_dataset = val_dfs[val_dfs['source_df_idx'] == df_idx]

        darts_val_dataset = darts.TimeSeries.from_dataframe(val_df_dataset,
                                                            time_col='time',
                                                            value_cols='sample')

        val_forecast = model.predict(n=len(darts_val_dataset) - config.look_back,
                                     series=darts_val_dataset[:config.look_back])

        val_forecast.values = (
            dataset_utils.re_transform_and_re_scale_data(darts_values=val_forecast.values(),
                                                         transformation_method=config.transformation_method,
                                                         scale_method=config.scale_method,
                                                         scaler=scaler)
        )

        darts_val_dataset.values = (
            dataset_utils.re_transform_and_re_scale_data(darts_values=darts_val_dataset.values(),
                                                         transformation_method=config.transformation_method,
                                                         scale_method=config.scale_method,
                                                         scaler=scaler)
        )

        assert len(darts_val_dataset[config.look_back:]) == len(val_forecast)
        val_mae = mae(darts_val_dataset[config.look_back:], val_forecast)
        val_mape = mape(darts_val_dataset[config.look_back:], val_forecast)
        val_mse = mse(darts_val_dataset[config.look_back:], val_forecast)
        val_rmse = rmse(darts_val_dataset[config.look_back:], val_forecast)

        epoch_val_maes.append(val_mae)
        epoch_val_mapes.append(val_mape)
        epoch_val_mses.append(val_mse)
        epoch_val_rmses.append(val_rmse)

        if df_idx % 200 == 0:
            val_forecast.plot(label='Forecast', color='blue')
            darts_val_dataset.plot(label='Actual', color='black')
            plt.legend()
            plt.title(f'({config.model_name}) Epoch ({epoch_idx}) Validation Forecasting (df={df_idx}):\n '
                      f'MAE = {val_mae:.4f}'
                      f' MAPE = {val_mape:.4f}'
                      f' MSE = {val_mse:.4f}'
                      f' RMSE = {val_rmse:.4f}'
                      )
            plt.show()
            plt.close()

    epoch_val_metrics_dict = {
        "mae": format_number(sum(epoch_val_maes) / len(epoch_val_maes)),
        "mape": format_number(sum(epoch_val_mapes) / len(epoch_val_mapes)),
        "mse": format_number(sum(epoch_val_mses) / len(epoch_val_mses)),
        "rmse": format_number(sum(epoch_val_rmses) / len(epoch_val_rmses))
    }

    return epoch_val_metrics_dict


def print_metrics_in_table(epoch_idx, epoch_train_metrics_dict, epoch_val_metrics_dict):
    """
    Prints the metrics in a table.
    """
    results_data = [
        ["Train", epoch_train_metrics_dict["mae"], epoch_train_metrics_dict["mape"], epoch_train_metrics_dict["mse"],
         epoch_train_metrics_dict["rmse"]],
        ["Validation", epoch_val_metrics_dict["mae"], epoch_val_metrics_dict["mape"], epoch_val_metrics_dict["mse"],
         epoch_val_metrics_dict["rmse"]]
    ]
    headers = ["Data", "avg MAE", "avg MAPE", "avg MSE", "avg RMSE"]

    print(f"Epoch {epoch_idx + 1}/{config.num_epochs} Results:")
    print(tabulate(results_data, headers=headers, tablefmt="pretty"))


def plot_metrics(total_epochs_train_metrics_dict, total_epochs_val_metrics_dict, metric_name):
    train_values = [float(metrics[metric_name]) for metrics in total_epochs_train_metrics_dict.values()]
    val_values = [float(metrics[metric_name]) for metrics in total_epochs_val_metrics_dict.values()]

    plt.plot(train_values, color='blue', label='train')
    plt.plot(val_values, color='orange', label='val')
    plt.xticks(range(len(train_values)))
    plt.title(f'({config.model_name}) {metric_name.upper()}')
    plt.legend()
    plt.show()
    plt.close()


def train_and_validate(model, train_dfs, val_dfs, scaler):
    total_epochs_train_metrics_dict = {}
    total_epochs_val_metrics_dict = {}
    for epoch_idx in range(config.num_epochs):
        print(f"Epoch {epoch_idx + 1}/{config.num_epochs}")

        epoch_train_metrics_dict = train_one_epoch(epoch_idx, model, train_dfs, scaler)
        epoch_val_metrics_dict = validate_one_epoch(epoch_idx, model, val_dfs, scaler)

        print_metrics_in_table(epoch_idx, epoch_train_metrics_dict, epoch_val_metrics_dict)

        total_epochs_train_metrics_dict[epoch_idx] = epoch_train_metrics_dict
        total_epochs_val_metrics_dict[epoch_idx] = epoch_val_metrics_dict

    # TODO: define the metrics in config
    metrics_to_plot = ["mae", "mape", "mse", "rmse"]

    for metric in metrics_to_plot:
        plot_metrics(total_epochs_train_metrics_dict, total_epochs_val_metrics_dict, metric)
