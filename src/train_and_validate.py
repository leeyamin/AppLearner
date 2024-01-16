import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import darts
from darts.models import TCNModel, NBEATSModel, RNNModel, BlockRNNModel
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Union, Optional
import os
from tqdm import tqdm
import time
import pandas as pd

import src.utils as utils
from src.utils import Config
from src.framework__data_set import TimeSeriesDataSet


def format_number(number: float) -> str:
    """Format a number to a string of 4 decimal places."""
    return f"{number:.4f}"


def plot_actual_vs_forecast(series, forecast, mode, epoch_idx, df_idx, mae, mape, mse, rmse, df_idx_output_path,
                            show_plots_flag):
    """
    Plot the actual vs. forecasted values of the series.
    @param series: the actual series (gt)
    @param forecast: the forecasted series (predicted)
    @param mode: train, validation or test
    @param epoch_idx: number of the epoch
    @param df_idx: the index of the dataframe
    @param mae: mean absolute error
    @param mape: mean absolute percentage error
    @param mse: mean squared error
    @param rmse: root mean squared error
    @param df_idx_output_path: path to the output directory of the dataframe
    @param show_plots_flag: whether to show plots or not
    @return None
    """
    forecast.plot(label='forecast', color='blue', low_quantile=0.05, high_quantile=0.95, alpha=0.3)
    series.plot(label='actual', color='black', linestyle='--')
    if mode == 'test':
        title = (f'{mode} forecasting (df={df_idx})\n'
                 f'MAE={mae:.3f} '
                 f'MAPE={mape:.3f} '
                 f'MSE={mse:.3f} '
                 f'RMSE={rmse:.3f}')
    else:
        title = (f'Epoch ({epoch_idx}) {mode} forecasting (df={df_idx})\n'
                 f'MAE={mae:.3f} '
                 f'MAPE={mape:.3f} '
                 f'MSE={mse:.3f} '
                 f'RMSE={rmse:.3f}')
    plt.title(title)
    plt.legend()
    if df_idx_output_path is not None:
        plt.savefig(f"{df_idx_output_path}/df_idx_{df_idx}_epoch_{epoch_idx}.png")
    if show_plots_flag:
        plt.show()
    plt.close()


def train_or_validate_one_epoch(epoch_idx: Optional[int],
                                model: Union[TCNModel, NBEATSModel, RNNModel, BlockRNNModel],
                                data: TimeSeriesDataSet, look_back: int,
                                mode: str, output_path: str, show_plots_flag: bool,
                                limit: Optional[int] = None) -> Dict:
    """
    Train or validate the model one epoch through each dataframe, and record the epoch metrics.
    @param epoch_idx: number of the epoch
    @param model: model to train on or predict from the darts library
    @param data: time series data object
    @param look_back: number of time steps to look back
    @param mode: train, validation or test
    @param output_path: path to the output directory
    @param show_plots_flag: whether to show plots or not
    @param limit: limit the number of dataframes to train on or predict from
    @return: dictionary of the computed metrics of the epoch
    """

    is_train = True if mode == 'train' else False
    print('Training...' if is_train else '\nValidating...')
    data_dfs = data.get_train_time_series_data() if is_train else data.get_val_time_series_data()

    epoch_maes = []
    epoch_mapes = []
    epoch_mses = []
    epoch_rmses = []

    series_dict = {}

    unique_values = data_dfs['source_df_idx'].unique()[:limit] if limit is not None else data_dfs[
        'source_df_idx'].unique()
    for df_idx in unique_values:
        df_dataset = data_dfs[data_dfs['source_df_idx'] == df_idx]
        darts_df_series = darts.TimeSeries.from_dataframe(df_dataset, time_col='time', value_cols='sample')
        series_dict[df_idx] = darts_df_series

    if is_train:
        model.fit(series=list(series_dict.values()), verbose=True, past_covariates=None)

        # TODO: if DeepTCN is implemented
        #  (see https://github.com/unit8co/darts/blob/master/examples/09-DeepTCN-examples.ipynb)
        # model.fit(series=darts_train_dataset, verbose=False, past_covariates=list(covariates_dict.values())

    for df_idx, series in tqdm(series_dict.items(), total=len(series_dict), leave=True, position=0, desc=mode):
        num_samples = 1 if model.model_name == 'LSTM' else 500
        forecast = model.predict(n=len(series) - look_back,  # number of time steps to predict
                                 series=series[:look_back],  # series to predict from
                                 num_samples=num_samples,  # number of samples to draw from the distribution
                                 num_loader_workers=4,
                                 verbose=False)
        gt = series[look_back:]
        assert len(gt) == len(forecast)
        assert (gt.time_index == forecast.time_index).all()

        series.values = data.re_transform_and_re_scale_data(series.values())
        forecast.values = data.re_transform_and_re_scale_data(forecast.values())
        gt.values = data.re_transform_and_re_scale_data(gt.values())

        mae = darts.metrics.mae(gt, forecast)
        mape = darts.metrics.mape(gt, forecast)
        mse = darts.metrics.mse(gt, forecast)
        rmse = darts.metrics.rmse(gt, forecast)

        if output_path is not None:
            main_output_path = f"{output_path}/forecast_vs_actual/{mode}"
            df_idx_output_path = f"{main_output_path}/df_idx_{df_idx}/"
            os.makedirs(df_idx_output_path, exist_ok=True)
        else:
            df_idx_output_path = None

        plot_actual_vs_forecast(series, forecast, mode, epoch_idx, df_idx, mae, mape, mse, rmse, df_idx_output_path,
                                show_plots_flag)
        epoch_maes.append(mae)
        epoch_mapes.append(mape)
        epoch_mses.append(mse)
        epoch_rmses.append(rmse)

    # compute epoch metrics (average of all dfs)
    epoch_metrics_dict = {
        "mae": sum(epoch_maes) / len(epoch_maes),
        "mape": sum(epoch_mapes) / len(epoch_mapes),
        "mse": sum(epoch_mses) / len(epoch_mses),
        "rmse": sum(epoch_rmses) / len(epoch_rmses)
    }

    return epoch_metrics_dict


def write_metrics_to_tensorboard(epoch_idx: int,
                                 epoch_train_metrics_dict: Dict, epoch_val_metrics_dict: Dict,
                                 train_writer: SummaryWriter, val_writer: SummaryWriter) -> None:
    """
    Write the metrics of the epoch to tensorboard.
    @param epoch_idx: number of the epoch
    @param epoch_train_metrics_dict: dictionary of the computed metrics of the train data
    @param epoch_val_metrics_dict: dictionary of the computed metrics of the validation data
    @param train_writer: SummaryWriter object of the train data
    @param val_writer: SummaryWriter object of the validation data
    @return None
    """
    evaluation_metrics = ['mae', 'mape', 'mse', 'rmse']
    for metric in evaluation_metrics:
        train_writer.add_scalar(metric, epoch_train_metrics_dict[metric], epoch_idx)
        val_writer.add_scalar(metric, epoch_val_metrics_dict[metric], epoch_idx)

    train_writer.close()
    val_writer.close()


def print_metrics_in_table(epoch_train_metrics_dict: Dict, epoch_val_metrics_dict: Dict,
                           epoch_idx: int, num_epochs: int, output_path: str) -> None:
    """
    Print the metrics of the epoch in a table.
    @param epoch_train_metrics_dict: dictionary of the computed metrics of the train data
    @param epoch_val_metrics_dict: dictionary of the computed metrics of the validation data
    @param epoch_idx: number of the epoch
    @param num_epochs: total number of epochs
    @param output_path: path to the output directory
    @return None
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

    print(f"Epoch {epoch_idx + 1}/{num_epochs} Results:")
    print(tabulate(results_data, headers=headers, tablefmt="pretty"))
    utils.record_logs_to_txt(f"Epoch {epoch_idx + 1}/{num_epochs} Results:", output_path)
    utils.record_logs_to_txt(tabulate(results_data, headers=headers, tablefmt="pretty"), output_path)


def plot_metrics(total_epochs_train_metrics_dict: Dict, total_epochs_val_metrics_dict: Dict,
                 output_path: str, model_name: str, mae_best_val: float, show_plots_flag: bool = True) -> None:
    """
    Plot the metrics convergence across epochs.
    @param total_epochs_train_metrics_dict: dictionary of the computed metrics of the train data across epochs
    @param total_epochs_val_metrics_dict: dictionary of the computed metrics of the validation data across epochs
    @param output_path: path to the output directory
    @param model_name: name of the model
    @param mae_best_val: best validation MAE
    @param show_plots_flag: whether to show plots or not
    @return None
    """
    evaluation_metrics = ['mae', 'mape', 'mse', 'rmse']
    for metric in evaluation_metrics:
        train_values = [float(epoch_metrics[metric]) for epoch_metrics in total_epochs_train_metrics_dict.values()]
        val_values = [float(epoch_metrics[metric]) for epoch_metrics in total_epochs_val_metrics_dict.values()]

        plt.plot(train_values, color='blue', marker='o', label='train')
        plt.plot(val_values, color='orange', marker='s', label='val')
        plt.xticks(range(len(train_values)))
        plt.ylim(0, None)
        if mae_best_val != float('inf'):
            best_epoch_idx = np.argmin(val_values)
            plt.axvline(x=best_epoch_idx, color='red', linestyle='--', label=f'best {metric} val')
        plt.title(f'({model_name}) {metric.upper()} across epochs')
        plt.legend()
        if output_path is not None:
            plt.savefig(f"{output_path}/{metric}_convergence.png")
        if show_plots_flag:
            plt.show()
        plt.close()


def save_metrics_to_csv(metrics_dict, epoch_idx, output_path, filename, is_best_val=False):
    if not is_best_val:
        for key, value in metrics_dict.items():
            for k, v in value.items():
                metrics_dict[key][k] = round(v, 4)

    if is_best_val:
        metrics_dict['epoch_idx'] = epoch_idx

    os.makedirs(output_path, exist_ok=True)
    csv_file_path = os.path.join(output_path, f'{filename}.csv')

    if is_best_val:
        df = pd.DataFrame([metrics_dict])
    else:
        df = pd.DataFrame(metrics_dict).T

    df.to_csv(csv_file_path, header=True)


def train_and_validate(model: Union[TCNModel, NBEATSModel, RNNModel],
                       data: TimeSeriesDataSet,
                       config: Config) -> Union[TCNModel, NBEATSModel, RNNModel, BlockRNNModel]:
    """
    Train and validate the model across epochs; record results and save best model.
    @param model: model to train and validate from the darts library
    @param data: time series data object
    @param config: configuration object of the run
    @return model: trained model of the last epoch
    """
    utils.disable_pytorch_lightning_logging()
    total_epochs_train_metrics_dict = {}
    total_epochs_val_metrics_dict = {}

    mae_best_val = float('inf')
    for epoch_idx in range(config.num_epochs):
        epoch_start_time = time.time()
        print(f"Epoch {epoch_idx + 1}/{config.num_epochs}")

        train_writer = SummaryWriter(log_dir=f'{config.output_path}/tensorboard/train')
        val_writer = SummaryWriter(log_dir=f'{config.output_path}/tensorboard/val')

        epoch_train_metrics_dict = train_or_validate_one_epoch(epoch_idx, model, data, look_back=config.look_back,
                                                               mode='train', output_path=config.output_path,
                                                               show_plots_flag=False, limit=1)
        epoch_val_metrics_dict = train_or_validate_one_epoch(epoch_idx, model, data, look_back=config.look_back,
                                                             mode='validation', output_path=config.output_path,
                                                             show_plots_flag=False, limit=1)

        write_metrics_to_tensorboard(epoch_idx, epoch_train_metrics_dict, epoch_val_metrics_dict,
                                     train_writer, val_writer)

        print_metrics_in_table(epoch_train_metrics_dict, epoch_val_metrics_dict,
                               epoch_idx, config.num_epochs, config.output_path)

        total_epochs_train_metrics_dict[epoch_idx] = epoch_train_metrics_dict
        total_epochs_val_metrics_dict[epoch_idx] = epoch_val_metrics_dict

        save_metrics_to_csv(total_epochs_train_metrics_dict, epoch_idx, config.output_path, 'train_metrics')
        save_metrics_to_csv(total_epochs_val_metrics_dict, epoch_idx, config.output_path, 'val_metrics')

        if float(epoch_val_metrics_dict['mae']) < mae_best_val:
            mae_best_val = float(epoch_val_metrics_dict['mae'])
            save_metrics_to_csv(epoch_val_metrics_dict, epoch_idx, config.output_path, 'best_val_metrics',
                                is_best_val=True)

        plot_metrics(total_epochs_train_metrics_dict, total_epochs_val_metrics_dict,
                     config.output_path, config.model_name, mae_best_val, show_plots_flag=False)

        print(f'Epoch time: {((time.time() - epoch_start_time) / 60):.3f} minutes')
        utils.record_logs_to_txt(f'Epoch time: {((time.time() - epoch_start_time) / 60):.3f} minutes',
                                 config.output_path)

    return model
