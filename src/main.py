import matplotlib.pyplot as plt
import darts
from darts.models import TCNModel
from darts.utils.likelihood_models import GaussianLikelihood
from darts.metrics import mae, mape, mse, rmse

import src.framework__data_set as framework__data_set
import src.utils as utils
import src.config as config
import src.dataset.utils as dataset_utils

import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="MPS available but not used.")

if __name__ == '__main__':
    utils.seed_everything()

    data = framework__data_set.get_data_set(
        metric=config.metric,
        application_name=config.application_name,
        path_to_data='../data/'
    )

    data_set = dataset_utils.prepare_data_for_run(data)

    # TODO: shuffle between data frames (not within)

    train_ratio = 0.8
    num_dfs = len(data_set['source_df_idx'].unique())
    num_dfs_train = int(num_dfs * train_ratio)
    train_dfs = data_set[data_set['source_df_idx'] <= num_dfs_train]
    val_dfs = data_set[data_set['source_df_idx'] > num_dfs_train]

    look_back = 12
    horizon = 2

    model = (
        TCNModel(
            dropout=0.2,
            batch_size=16,
            n_epochs=1,
            num_filters=5,
            num_layers=3,
            optimizer_kwargs={"lr": 0.0003},
            random_state=0,
            input_chunk_length=look_back,
            output_chunk_length=horizon,
            kernel_size=5,
            weight_norm=True,
            dilation_base=4,
            likelihood=GaussianLikelihood(),
            pl_trainer_kwargs={"accelerator": "cpu"},
        ))

    num_epochs = 2

    train_maes = []
    train_mapes = []
    train_mses = []
    train_rmses = []

    val_maes = []
    val_mapes = []
    val_mses = []
    val_rmses = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_train_maes = []
        epoch_train_mapes = []
        epoch_train_mses = []
        epoch_train_rmses = []

        epoch_val_maes = []
        epoch_val_mapes = []
        epoch_val_mses = []
        epoch_val_rmses = []

        for df_idx in train_dfs['source_df_idx'].unique():
            train_df_dataset = data_set[data_set['source_df_idx'] == df_idx]

            darts_train_dataset = darts.TimeSeries.from_dataframe(train_df_dataset,
                                                                  time_col='time',
                                                                  value_cols='sample')

            model.fit(series=darts_train_dataset, past_covariates=None, verbose=False)
            train_forecast = model.predict(n=len(darts_train_dataset) - look_back, series=darts_train_dataset[:look_back])

            assert len(darts_train_dataset[look_back:]) == len(train_forecast)
            train_mae = mae(darts_train_dataset[look_back:], train_forecast)
            train_mape = mape(darts_train_dataset[look_back:], train_forecast)
            train_mse = mse(darts_train_dataset[look_back:], train_forecast)
            train_rmse = rmse(darts_train_dataset[look_back:], train_forecast)

            epoch_train_maes.append(train_mae)
            epoch_train_mapes.append(train_mape)
            epoch_train_mses.append(train_mse)
            epoch_train_rmses.append(train_rmse)

            if df_idx % 100 == 0:
                train_forecast.plot(label='Forecast', lw=2)
                darts_train_dataset.plot(label='Actual')
                plt.legend()
                plt.title(f'Epoch ({epoch}) Train Forecasting (df={df_idx}):\n '
                          f'MAE = {train_mae:.4f}'
                          f' MAPE = {train_mape:.4f}'
                          f' MSE = {train_mse:.4f}'
                          f' RMSE = {train_rmse:.4f}'
                          )
                plt.show()
                plt.close()

        for df_idx in val_dfs['source_df_idx'].unique():
            val_df_dataset = data_set[data_set['source_df_idx'] == df_idx]

            darts_val_dataset = darts.TimeSeries.from_dataframe(val_df_dataset,
                                                                time_col='time',
                                                                value_cols='sample')

            val_forecast = model.predict(n=len(darts_val_dataset) - look_back, series=darts_val_dataset[:look_back])

            assert len(darts_val_dataset[look_back:]) == len(val_forecast)
            val_mae = mae(darts_val_dataset[look_back:], val_forecast)
            val_mape = mape(darts_val_dataset[look_back:], val_forecast)
            val_mse = mse(darts_val_dataset[look_back:], val_forecast)
            val_rmse = rmse(darts_val_dataset[look_back:], val_forecast)

            epoch_val_maes.append(val_mae)
            epoch_val_mapes.append(val_mape)
            epoch_val_mses.append(val_mse)
            epoch_val_rmses.append(val_rmse)

            if df_idx % 20 == 0:
                val_forecast.plot(label='Forecast', lw=2)
                darts_val_dataset.plot(label='Actual')
                plt.legend()
                plt.title(f'Epoch ({epoch}) Validation Forecasting (df={df_idx}):\n '
                          f'MAE = {val_mae:.4f}'
                          f' MAPE = {val_mape:.4f}'
                          f' MSE = {val_mse:.4f}'
                          f' RMSE = {val_rmse:.4f}'
                          )
                plt.show()
                plt.close()

        print(f"Epoch {epoch + 1}/{num_epochs} TRAIN results:")
        print(f"avg MAE = {sum(epoch_train_maes) / len(epoch_train_maes):.4f}")
        print(f"avg MAPE = {sum(epoch_train_mapes) / len(epoch_train_mapes):.4f}")
        print(f"avg MSE = {sum(epoch_train_mses) / len(epoch_train_mses):.4f}")
        print(f"avg RMSE = {sum(epoch_train_rmses) / len(epoch_train_rmses):.4f}")

        print(f"Epoch {epoch + 1}/{num_epochs} VAL results:")
        print(f"avg MAE = {sum(epoch_val_maes) / len(epoch_val_maes):.4f}")
        print(f"avg MAPE = {sum(epoch_val_mapes) / len(epoch_val_mapes):.4f}")
        print(f"avg MSE = {sum(epoch_val_mses) / len(epoch_val_mses):.4f}")
        print(f"avg RMSE = {sum(epoch_val_rmses) / len(epoch_val_rmses):.4f}")

        train_maes.append(sum(epoch_train_maes) / len(epoch_train_maes))
        train_mapes.append(sum(epoch_train_mapes) / len(epoch_train_mapes))
        train_mses.append(sum(epoch_train_mses) / len(epoch_train_mses))
        train_rmses.append(sum(epoch_train_rmses) / len(epoch_train_rmses))

        val_maes.append(sum(epoch_val_maes) / len(epoch_val_maes))
        val_mapes.append(sum(epoch_val_mapes) / len(epoch_val_mapes))
        val_mses.append(sum(epoch_val_mses) / len(epoch_val_mses))
        val_rmses.append(sum(epoch_val_rmses) / len(epoch_val_rmses))

    # plot train vs val errors convergence
    # mae
    plt.plot(train_maes, label='train')
    plt.plot(val_maes, label='val')
    plt.title('MAE')
    plt.legend()
    plt.show()

    # mape
    plt.plot(train_mapes, label='train')
    plt.plot(val_mapes, label='val')
    plt.title('MAPE')
    plt.legend()
    plt.show()

    # mse
    plt.plot(train_mses, label='train')
    plt.plot(val_mses, label='val')
    plt.title('MSE')
    plt.legend()
    plt.show()







