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

    # train_ratio = 0.8
    # num_dfs = len(data_set['source_df_idx'].unique())
    # num_dfs_train = int(num_dfs * train_ratio)
    # train_data_set = data_set[data_set['source_df_idx'] <= num_dfs_train]
    # test_data_set = data_set[data_set['source_df_idx'] > num_dfs_train]

    # TODO: currently train and val for one df; then extend to all
    train_df_dataset = data_set[data_set['source_df_idx'] == 0]

    look_back = 12
    horizon = 2

    model = (
        TCNModel(
            dropout=0.2,
            batch_size=16,
            n_epochs=500,
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

    # TODO: change model num of epochs to 1 for each df and then add num_epochs to the loop
    # num_epochs = 2
    # for epoch in range(num_epochs):

    # TODO: currently for one df; then edit to all df in a loop
    # for df_idx in train_data_set['source_df_idx'].unique():
    #     subset_source_df_index = train_data_set['source_df_idx'].index[train_data_set['source_df_idx'] == df_idx].tolist()
    #     subset_data_set = train_data_set.loc[subset_source_df_index]

    darts_train_dataset = darts.TimeSeries.from_dataframe(train_df_dataset,
                                                          time_col='time',
                                                          value_cols='sample')

    model.fit(series=darts_train_dataset, past_covariates=None, verbose=True)
    train_forecast = model.predict(n=len(darts_train_dataset) - look_back, series=darts_train_dataset[:look_back])

    assert len(darts_train_dataset[look_back:]) == len(train_forecast)
    train_mae = mae(darts_train_dataset[look_back:], train_forecast)
    train_mape = mape(darts_train_dataset[look_back:], train_forecast)
    train_mse = mse(darts_train_dataset[look_back:], train_forecast)
    train_rmse = rmse(darts_train_dataset[look_back:], train_forecast)

    train_forecast.plot(label='Forecast', lw=2)
    darts_train_dataset.plot(label='Actual')
    plt.legend()
    # TODO: add to the plot the df idx
    plt.title(f'Train Forecasting: '
              f'MAE = {train_mae:.4f}'
              f'MAPE = {train_mape:.4f}'
              f'MSE = {train_mse:.4f}'
              f'RMSE = {train_rmse:.4f}'
              )
    plt.show()

    val_df_dataset = data_set[data_set['source_df_idx'] == 1]

    darts_val_dataset = darts.TimeSeries.from_dataframe(val_df_dataset,
                                                        time_col='time',
                                                        value_cols='sample')

    val_forecast = model.predict(n=len(darts_val_dataset) - look_back, series=darts_val_dataset[:look_back])
    assert len(darts_val_dataset[look_back:]) == len(val_forecast)
    val_mae = mae(darts_val_dataset[look_back:], val_forecast)
    val_mape = mape(darts_val_dataset[look_back:], val_forecast)
    val_mse = mse(darts_val_dataset[look_back:], val_forecast)
    val_rmse = rmse(darts_val_dataset[look_back:], val_forecast)

    val_forecast.plot(label='Forecast', lw=2)
    darts_val_dataset.plot(label='Actual')
    plt.legend()
    # TODO: add to the plot the df idx
    plt.title(f'Val Forecasting: '
              f'MAE = {val_mae:.4f}'
              f'MAPE = {val_mape:.4f}'
              f'MSE = {val_mse:.4f}'
              f'RMSE = {val_rmse:.4f}'
              )
    plt.show()
