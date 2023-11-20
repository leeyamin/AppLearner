import matplotlib.pyplot as plt
from tabulate import tabulate
import darts
from darts.metrics import mae, mape, mse, rmse
from torch.utils.tensorboard import SummaryWriter

from src.config import config
import src.utils as utils


def format_number(number):
    return f"{number:.4f}"


def plot_forecast_vs_actual(epoch_idx, forecast_data, actual_data, df_idx, df_metrics_dict, is_train):
    mode = "train" if is_train else "val"
    main_output_path = f"{config.output_path}/forecast_vs_actual/{mode}"
    df_idx_output_path = f"{main_output_path}/{df_idx}/"
    import os
    if not os.path.exists(df_idx_output_path):
        os.makedirs(df_idx_output_path)

    plt.figure(figsize=(20, 10))

    forecast_data.plot(label='Forecast', color='blue', alpha=0.5)
    actual_data.plot(label='Actual', color='black')
    plt.legend()
    plt.title(f'({config.model_name}) Epoch ({epoch_idx}) {mode} Forecasting (df={df_idx}):\n '
              f'MAE = {df_metrics_dict["mae"]:.4f}'
              f' MAPE = {df_metrics_dict["mape"]:.4f}'
              f' MSE = {df_metrics_dict["mse"]:.4f}'
              f' RMSE = {df_metrics_dict["rmse"]:.4f}'
              )
    plt.savefig(f"{df_idx_output_path}/{epoch_idx}.png")
    plt.close()


def train_one_epoch(epoch_idx, model, data):
    epoch_train_maes = []
    epoch_train_mapes = []
    epoch_train_mses = []
    epoch_train_rmses = []

    train_dfs = data.get_train_time_series_data()

    # train on each df (one epoch = one pass through all dfs)
    for df_idx in train_dfs['source_df_idx'].unique():
        train_df_dataset = train_dfs[train_dfs['source_df_idx'] == df_idx]

        darts_train_dataset = darts.TimeSeries.from_dataframe(train_df_dataset,
                                                              time_col='time',
                                                              value_cols='sample')

        model.fit(series=darts_train_dataset, past_covariates=None, verbose=False)
        train_forecast = model.predict(n=len(darts_train_dataset) - config.look_back,
                                       series=darts_train_dataset[:config.look_back])

        train_forecast.values = data.re_transform_and_re_scale_data(darts_values=train_forecast.values(),
                                                                    transformation_method=config.transformation_method,
                                                                    scale_method=config.scale_method)
        darts_train_dataset.values = data.re_transform_and_re_scale_data(darts_values=darts_train_dataset.values(),
                                                                         transformation_method=config.transformation_method,
                                                                         scale_method=config.scale_method)

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

        plot_forecast_vs_actual(epoch_idx, train_forecast, darts_train_dataset, df_idx, train_df_metrics_dict,
                                is_train=True)

    epoch_train_metrics_dict = {
        "mae": sum(epoch_train_maes) / len(epoch_train_maes),
        "mape": sum(epoch_train_mapes) / len(epoch_train_mapes),
        "mse": sum(epoch_train_mses) / len(epoch_train_mses),
        "rmse": sum(epoch_train_rmses) / len(epoch_train_rmses)
    }

    return epoch_train_metrics_dict


def validate_one_epoch(epoch_idx, model, data):
    epoch_val_maes = []
    epoch_val_mapes = []
    epoch_val_mses = []
    epoch_val_rmses = []

    val_dfs = data.get_val_time_series_data()

    for df_idx in val_dfs['source_df_idx'].unique():
        val_df_dataset = val_dfs[val_dfs['source_df_idx'] == df_idx]

        darts_val_dataset = darts.TimeSeries.from_dataframe(val_df_dataset,
                                                            time_col='time',
                                                            value_cols='sample')

        val_forecast = model.predict(n=len(darts_val_dataset) - config.look_back,
                                     series=darts_val_dataset[:config.look_back])

        val_forecast.values = data.re_transform_and_re_scale_data(darts_values=val_forecast.values(),
                                                                  transformation_method=config.transformation_method,
                                                                  scale_method=config.scale_method)

        darts_val_dataset.values = data.re_transform_and_re_scale_data(darts_values=darts_val_dataset.values(),
                                                                       transformation_method=config.transformation_method,
                                                                       scale_method=config.scale_method)

        assert len(darts_val_dataset[config.look_back:]) == len(val_forecast)
        val_mae = mae(darts_val_dataset[config.look_back:], val_forecast)
        val_mape = mape(darts_val_dataset[config.look_back:], val_forecast)
        val_mse = mse(darts_val_dataset[config.look_back:], val_forecast)
        val_rmse = rmse(darts_val_dataset[config.look_back:], val_forecast)

        epoch_val_maes.append(val_mae)
        epoch_val_mapes.append(val_mape)
        epoch_val_mses.append(val_mse)
        epoch_val_rmses.append(val_rmse)

        val_df_metrics_dict = {
            "mae": val_mae,
            "mape": val_mape,
            "mse": val_mse,
            "rmse": val_rmse
        }

        plot_forecast_vs_actual(epoch_idx, val_forecast, darts_val_dataset, df_idx, val_df_metrics_dict,
                                is_train=False)

    epoch_val_metrics_dict = {
        "mae": sum(epoch_val_maes) / len(epoch_val_maes),
        "mape": sum(epoch_val_mapes) / len(epoch_val_mapes),
        "mse": sum(epoch_val_mses) / len(epoch_val_mses),
        "rmse": sum(epoch_val_rmses) / len(epoch_val_rmses)
    }

    return epoch_val_metrics_dict


def write_metrics_to_tensorboard(epoch_idx, epoch_train_metrics_dict, epoch_val_metrics_dict,
                                 train_writer, val_writer):
    for metric in config.evaluation_metrics:
        train_writer.add_scalar(metric, epoch_train_metrics_dict[metric], epoch_idx)
        val_writer.add_scalar(metric, epoch_val_metrics_dict[metric], epoch_idx)

    train_writer.close()
    val_writer.close()


def print_metrics_in_table(epoch_idx, epoch_train_metrics_dict, epoch_val_metrics_dict):
    """
    Prints the metrics in a table.
    """
    results_data = [
        ["Train",
         format_number(epoch_train_metrics_dict["mae"]),
         format_number(epoch_train_metrics_dict["mape"]),
         format_number(epoch_train_metrics_dict["mse"]),
         format_number(epoch_train_metrics_dict["rmse"])],
        ["Validation",
         format_number(epoch_val_metrics_dict["mae"]),
         format_number(epoch_val_metrics_dict["mape"]),
         format_number(epoch_val_metrics_dict["mse"]),
         format_number(epoch_val_metrics_dict["rmse"])]
    ]
    headers = ["Data", "avg MAE", "avg MAPE", "avg MSE", "avg RMSE"]

    print(f"Epoch {epoch_idx + 1}/{config.num_epochs} Results:")
    print(tabulate(results_data, headers=headers, tablefmt="pretty"))
    utils.record_logs_to_txt(f"Epoch {epoch_idx + 1}/{config.num_epochs} Results:")
    utils.record_logs_to_txt(tabulate(results_data, headers=headers, tablefmt="pretty"))


def plot_metrics(total_epochs_train_metrics_dict, total_epochs_val_metrics_dict):
    for metric in config.evaluation_metrics:
        train_values = [float(epoch_metrics[metric]) for epoch_metrics in total_epochs_train_metrics_dict.values()]
        val_values = [float(epoch_metrics[metric]) for epoch_metrics in total_epochs_val_metrics_dict.values()]

        plt.plot(train_values, color='blue', label='train')
        plt.plot(val_values, color='orange', label='val')
        plt.xticks(range(len(train_values)))
        plt.title(f'({config.model_name}) {metric.upper()}')
        plt.legend()
        plt.show()
        plt.close()


def train_and_validate(model, data):
    total_epochs_train_metrics_dict = {}
    total_epochs_val_metrics_dict = {}
    val_mae_min = float('inf')
    for epoch_idx in range(config.num_epochs):
        print(f"Epoch {epoch_idx + 1}/{config.num_epochs}")

        train_writer = SummaryWriter(log_dir=f'{config.output_path}/tensorboard/train')
        val_writer = SummaryWriter(log_dir=f'{config.output_path}/tensorboard/val')

        epoch_train_metrics_dict = train_one_epoch(epoch_idx, model, data)
        epoch_val_metrics_dict = validate_one_epoch(epoch_idx, model, data)

        # save best model based on validation mae
        epoch_val_mae = epoch_val_metrics_dict["mae"]
        if epoch_val_mae < val_mae_min:
            val_mae_min = epoch_val_mae
            utils.save_model(model, model_name='model_best')

        write_metrics_to_tensorboard(epoch_idx, epoch_train_metrics_dict, epoch_val_metrics_dict,
                                     train_writer, val_writer)

        print_metrics_in_table(epoch_idx, epoch_train_metrics_dict, epoch_val_metrics_dict)

        total_epochs_train_metrics_dict[epoch_idx] = epoch_train_metrics_dict
        total_epochs_val_metrics_dict[epoch_idx] = epoch_val_metrics_dict

    plot_metrics(total_epochs_train_metrics_dict, total_epochs_val_metrics_dict)

    return model
