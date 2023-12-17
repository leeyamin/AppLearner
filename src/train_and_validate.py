import matplotlib.pyplot as plt
from tabulate import tabulate
import darts
from darts.models import TCNModel, NBEATSModel, RNNModel
import darts.utils.timeseries_generation as tg
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Union
import os
from tqdm import tqdm

import src.utils as utils
from src.utils import Config
from src.framework__data_set import TimeSeriesDataSet


def format_number(number: float) -> str:
    """Format a number to a string of 4 decimal places."""
    return f"{number:.4f}"


def train_or_validate_one_epoch(epoch_idx: int,
                                model: Union[TCNModel, NBEATSModel, RNNModel],
                                data: TimeSeriesDataSet,
                                look_back: int, transformation_method: str, scale_method: str,
                                is_train: bool, output_path: str, show_plots_flag: bool,
                                limit: int) -> Dict:
    """
    Train or validate the model one epoch through each dataframe, and record the epoch metrics.
    @param epoch_idx: number of the epoch
    @param model: model to train on or predict from the darts library
    @param data: time series data object
    @param look_back: number of time steps to look back
    @param transformation_method: transformation method for the data
    @param scale_method: scale method for the data
    @param is_train: whether the data is train or validation
    @param output_path: path to the output directory
    @param show_plots_flag: whether to show plots or not
    # TODO: delete this it is only for debugging
    @param limit: limit the number of dataframes to train on or predict from
    @return: dictionary of the computed metrics of the epoch
    """

    mode = "train" if is_train else "validation"
    data_dfs = data.get_train_time_series_data() if is_train else data.get_val_time_series_data()

    epoch_maes = []
    epoch_mapes = []
    epoch_mses = []
    epoch_rmses = []

    series_dict = {}
    covariates_dict = {}
    for df_idx in data_dfs['source_df_idx'].unique()[:limit]:

        df_dataset = data_dfs[data_dfs['source_df_idx'] == df_idx]
        darts_df_series = darts.TimeSeries.from_dataframe(df_dataset, time_col='time', value_cols='sample')
        if model.model_name == 'DeepAR':
            noise = tg.gaussian_timeseries(std=0.6, start=darts_df_series.time_index[0],
                                           end=darts_df_series.time_index[-1], freq=darts_df_series.freq)
            noise_modulator = (
                                      tg.sine_timeseries(value_frequency=0.02,
                                                         start=darts_df_series.time_index[0],
                                                         end=darts_df_series.time_index[-1],
                                                         freq=darts_df_series.freq)
                                      + tg.constant_timeseries(value=1,
                                                               start=darts_df_series.time_index[0],
                                                               end=darts_df_series.time_index[-1],
                                                               freq=darts_df_series.freq)
                              ) / 2
            noise = noise * noise_modulator
            darts_df_series = sum([noise, darts_df_series])
            covariates_dict[df_idx] = noise_modulator

        series_dict[df_idx] = darts_df_series

    if is_train:
        if model.model_name == 'DeepAR':
            assert sum(len(series) for series in series_dict.values()) == sum(
                len(series) for series in covariates_dict.values())
            model.fit(series=list(series_dict.values()), verbose=False,
                      future_covariates=list(covariates_dict.values()))
        else:
            model.fit(series=list(series_dict.values()), verbose=False, past_covariates=None)
        # TODO: if DeepTCN is implemented
        #  (see https://github.com/unit8co/darts/blob/master/examples/09-DeepTCN-examples.ipynb)
        # model.fit(series=darts_train_dataset, verbose=False, past_covariates=list(covariates_dict.values())

    for df_idx, series in tqdm(series_dict.items(), total=len(series_dict), leave=True, position=0, desc=mode):
        # n is the number of time steps to predict
        # num_samples is the number of samples to draw from the distribution
        # series is the series to predict from

        num_samples = 1 if model.model_name == 'LSTM' else 500
        if model.model_name == 'DeepAR':
            future_covariates = covariates_dict[df_idx]

            forecast = model.predict(n=len(series) - look_back,
                                     series=series[:look_back],
                                     future_covariates=future_covariates,
                                     num_samples=500)
        else:
            forecast = model.predict(n=len(series) - look_back,
                                     series=series[:look_back],
                                     num_samples=num_samples)
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

        forecast.plot(label='forecast', color='blue', low_quantile=0.05, high_quantile=0.95, alpha=0.2)
        series.plot(label='actual', color='black', linestyle='--', alpha=0.5)
        plt.title(f'Epoch ({epoch_idx}) {mode} Forecasting (df={df_idx})\n'
                  f'MAE = {mae:.3f}'
                  f' MAPE = {mape:.3f}'
                  f' MSE = {mse:.3f}'
                  f' RMSE = {rmse:.3f}')
        plt.legend()
        if output_path is not None:
            plt.savefig(f"{df_idx_output_path}/df_idx_{df_idx}_epoch_{epoch_idx}.png")
        if show_plots_flag:
            plt.show()
        plt.close()

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
                 output_path: str, model_name: str, show_plots_flag: bool = True) -> None:
    """
    Plot the metrics convergence across epochs.
    @param total_epochs_train_metrics_dict: dictionary of the computed metrics of the train data across epochs
    @param total_epochs_val_metrics_dict: dictionary of the computed metrics of the validation data across epochs
    @param output_path: path to the output directory
    @param model_name: name of the model
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
        plt.title(f'({model_name}) {metric.upper()} across epochs')
        plt.legend()
        # if output_path is not None:
        # plt.savefig(f"{output_path}/forecast_vs_actual/{metric}_convergence.png")
        if show_plots_flag:
            plt.show()
        plt.close()


def train_and_validate(model: Union[TCNModel, NBEATSModel, RNNModel],
                       data: TimeSeriesDataSet,
                       config: Config) -> Union[TCNModel, NBEATSModel, RNNModel]:
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
    val_mae_min = float('inf')
    for epoch_idx in range(config.num_epochs):
        print(f"Epoch {epoch_idx + 1}/{config.num_epochs}")

        train_writer = SummaryWriter(log_dir=f'{config.output_path}/tensorboard/train')
        val_writer = SummaryWriter(log_dir=f'{config.output_path}/tensorboard/val')

        epoch_train_metrics_dict = train_or_validate_one_epoch(epoch_idx, model, data, look_back=config.look_back,
                                                               transformation_method=config.transformation_method,
                                                               scale_method=config.scale_method,
                                                               is_train=True, output_path=config.output_path,
                                                               show_plots_flag=False, limit=1)
        epoch_val_metrics_dict = train_or_validate_one_epoch(epoch_idx, model, data, look_back=config.look_back,
                                                             transformation_method=config.transformation_method,
                                                             scale_method=config.scale_method,
                                                             is_train=False, output_path=config.output_path,
                                                             show_plots_flag=False, limit=10)

        # save best model based on validation mae
        # TODO: check if this is the best metric to save the model by
        epoch_val_mae = epoch_val_metrics_dict["mae"]
        if epoch_val_mae < val_mae_min:
            val_mae_min = epoch_val_mae
            utils.save_model(model, model_name='model_best', output_path=config.output_path)

        write_metrics_to_tensorboard(epoch_idx, epoch_train_metrics_dict, epoch_val_metrics_dict,
                                     train_writer, val_writer)

        print_metrics_in_table(epoch_train_metrics_dict, epoch_val_metrics_dict,
                               epoch_idx, config.num_epochs, config.output_path)

        total_epochs_train_metrics_dict[epoch_idx] = epoch_train_metrics_dict
        total_epochs_val_metrics_dict[epoch_idx] = epoch_val_metrics_dict

    plot_metrics(total_epochs_train_metrics_dict, total_epochs_val_metrics_dict,
                 output_path=config.output_path, model_name=config.model_name, show_plots_flag=False)

    return model
