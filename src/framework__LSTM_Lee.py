"""
***********************************************************************************************************************
    Import Libraries
***********************************************************************************************************************
"""
import os
import numpy as np
import random
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Tuple
from torch.utils.data import Dataset, DataLoader, Subset
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm

import src.framework__data_set as framework__data_set

"""
***********************************************************************************************************************
    Helper Functions
***********************************************************************************************************************
"""


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


"""
***********************************************************************************************************************
    Data
***********************************************************************************************************************
"""


def transform_data(dataset_as_np, transformation_method='log'):
    """
    Transform the data using the specified method. Currently only supports log transformation.
    """
    if transformation_method == 'log':
        epsilon = 1e-8
        dataset_as_np = np.log(dataset_as_np + epsilon)
        return dataset_as_np


def scale_data(dataset_as_np, scale_method='min-max', scaler=None):
    """
    Scale the data using the specified method. Currently only supports min-max scaling.
    """
    if scale_method == 'min-max':
        if scaler is None:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit_transform(dataset_as_np)
        return scaler.transform(dataset_as_np), scaler


def transform_and_scale_data(dataset_as_np: np.ndarray,
                             transformation_method: str = 'log',
                             scale_method: str = 'min-max',
                             scaler: Optional[MinMaxScaler] = None) -> Tuple[np.ndarray, Optional[MinMaxScaler]]:
    """
    Transform and scale the data using the specified methods. Currently only supports log transformation and min-max
    scaling.
    """
    if transformation_method not in ['log', None] or scale_method not in ['min-max', None]:
        raise NotImplementedError
    if transformation_method is None and scale_method is None:
        return dataset_as_np, None
    if transformation_method is not None:
        dataset_as_np = transform_data(dataset_as_np, transformation_method=transformation_method)
    if scale_method is not None:
        dataset_as_np, scaler = scale_data(dataset_as_np, scale_method=scale_method, scaler=scaler)
    return dataset_as_np, scaler


def split_data_into_x_and_y(data, look_back, horizon):
    x = data[:, horizon:horizon + look_back]
    y = data[:, 0:horizon]
    return x, y


def re_scale_data(dataset_as_np, scaler):
    """
    Rescale the data.
    """
    dataset_as_np = scaler.inverse_transform(dataset_as_np)
    return dataset_as_np


def re_transform_data(dataset_as_np, transformation_method='log'):
    """
    Reverse transforms the data using the specified method. Currently only supports log transformation.
    """
    if transformation_method == 'log':
        dataset_as_np = np.exp(dataset_as_np)
    return dataset_as_np


def re_transform_and_re_scale_data(dataset_as_np: np.ndarray,
                                   transformation_method: str = 'log',
                                   scale_method: str = 'min-max',
                                   scaler: Optional[MinMaxScaler] = None) -> np.ndarray:
    """
    Reverse transforms and scales the data using the specified methods. Currently only supports log transformation and min-max
    scaling.
    """
    if transformation_method not in ['log', None] or scale_method not in ['min-max', None]:
        raise NotImplementedError
    if transformation_method is None and scale_method is None:
        return dataset_as_np
    if scale_method is not None:
        dataset_as_np = re_scale_data(dataset_as_np, scaler=scaler)
    if transformation_method is not None:
        dataset_as_np = re_transform_data(dataset_as_np, transformation_method=transformation_method)
    return dataset_as_np


def re_format_y_to_df(y_df, look_back, horizon, transformation_method, scale_method, scaler):
    """
    To rescale the data, we need to reformat the data to the original shape. This funtion uses dummy variables to
    represent the x values and then rescales the y values using the specified methods.
    """
    y_df = y_df.numpy()
    dummies = np.zeros((y_df.shape[0], look_back + horizon))
    dummies[:, 0:horizon] = y_df.squeeze()
    dummies = re_transform_and_re_scale_data(dummies,
                                             transformation_method=transformation_method,
                                             scale_method=scale_method, scaler=scaler)
    return dummies[:, 0:horizon]


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y, df_idx, num_samples):
        self.x = x
        self.y = y
        self.df_idx = df_idx
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.x[index], self.y[index]


"""
***********************************************************************************************************************
    Models
***********************************************************************************************************************
"""


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, horizon, bidirectional=False, dropout_rate=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True, bidirectional=bidirectional)

        # Additional bidirectional LSTM layer
        self.num_directions = 2 if bidirectional else 1
        self.lstm2 = nn.LSTM(hidden_size * self.num_directions, hidden_size,
                             num_stacked_layers, batch_first=True, bidirectional=bidirectional)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc1 = nn.Linear(hidden_size * self.num_directions, 64)
        self.fc2 = nn.Linear(64, horizon)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers * self.num_directions, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_stacked_layers * self.num_directions, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out, _ = self.lstm2(out, (h0, c0))

        out = self.dropout(out)

        out = self.fc1(out[:, -1, :])
        out = self.dropout(out)
        out = self.fc2(out)
        out = out.unsqueeze(-1)
        return out


def get_model(model_name, horizon):
    if model_name == 'LSTM':
        # TODO: ablation study on architecture and bidirectional
        return LSTM(1, 4, 1, horizon=horizon)
    else:
        raise NotImplementedError


def train_df_epoch(train_loader, model, loss_function, optimizer):
    model.train()
    df_train_epoch_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        df_train_epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return df_train_epoch_loss


def validate_df_epoch(test_loader, model, loss_function):
    model.eval()
    df_test_epoch_loss = 0.0

    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader):
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            output = model(x_batch)
            loss = loss_function(output, y_batch)
            df_test_epoch_loss += loss.item()

    return df_test_epoch_loss


def train_or_validate_one_epoch(epoch, x_data, y_data, data_set_source_df_idx, model, loss_function, optimizer,
                                batch_size, train_flag):
    epoch_loss = 0.0
    str_train_or_val = 'Training' if train_flag else 'Validating'
    progress = tqdm(data_set_source_df_idx.unique(), desc=f'Epoch {epoch + 1} - {str_train_or_val}', leave=True)
    for df_index in progress:
        subset_source_df_index = torch.where(data_set_source_df_idx == df_index)[0]
        subset_df_dataset = Subset(
            TimeSeriesDataset(x_data, y_data, data_set_source_df_idx,
                              num_samples=len(subset_source_df_index)),
            indices=subset_source_df_index
        )
        dataloader_subset = DataLoader(subset_df_dataset, batch_size=batch_size, shuffle=False)
        if train_flag:
            df_epoch_loss = train_df_epoch(dataloader_subset, model, loss_function, optimizer)
        else:
            df_epoch_loss = validate_df_epoch(dataloader_subset, model, loss_function)
        epoch_loss += df_epoch_loss
    return epoch_loss


def plot_convergence(train_losses, val_losses, num_epochs):
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Convergence')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()


def plot_predictions(model, x_dataset, y_dataset, data_set_source_df_idx, modality):
    for df_index in data_set_source_df_idx.unique():
        train_subset_source_df_index = torch.where(data_set_source_df_idx == df_index)[0]
        subset_df_dataset = Subset(TimeSeriesDataset(x_dataset, y_dataset, data_set_source_df_idx,
                                                     num_samples=len(train_subset_source_df_index)),
                                   indices=train_subset_source_df_index)
        batch_size = 32
        dataloader_subset = DataLoader(subset_df_dataset, batch_size=batch_size, shuffle=False)

        model.eval()
        mae = nn.L1Loss(reduction='sum')
        rmse = nn.MSELoss(reduction='sum')
        mae_loss = 0.0
        rmse_loss = 0.0
        df_indices = torch.where(data_set_source_df_idx == df_index)[0]
        y_df = y_dataset[df_indices]
        predicted_df = torch.Tensor()
        for batch in dataloader_subset:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            with torch.no_grad():
                predicted = model(x_batch)

            mae_loss += mae(predicted, y_batch).item()
            rmse_loss += torch.sqrt(rmse(predicted, y_batch)).item()
            predicted_df = torch.cat([predicted_df, predicted.squeeze()], dim=0)

        mae_loss /= len(subset_df_dataset)
        rmse_loss /= len(subset_df_dataset)
        assert len(predicted_df) == len(subset_df_dataset) == len(y_df)

        rescaled_y_df = re_format_y_to_df(y_df, look_back, horizon, transformation_method, scale_method, scaler)
        rescaled_predicted_df = re_format_y_to_df(predicted_df, look_back, horizon, transformation_method, scale_method,
                                                  scaler)

        plt.plot(rescaled_y_df[:, 0], label=f'Actual', linestyle='solid', color='tab:green')
        for i in range(rescaled_predicted_df.shape[1]):
            max_value = max(np.max(rescaled_predicted_df[:, i]), np.max(rescaled_y_df[:, i]))
            plt.plot(rescaled_predicted_df[:, i], label=f'Predicted Time Step {i + 1}', linestyle='dashed')

        plt.xlabel('Time')  # TODO: revert to actual time
        plt.ylabel('Value')
        plt.ylim(0, max_value * 1.1)
        plt.title(f'{modality} DF: {df_index}; MAE: {mae_loss:.3f}; RMSE: {rmse_loss:.3f}')
        plt.legend()
        plt.show()
        plt.close()


"""
***********************************************************************************************************************
    Train and Test
***********************************************************************************************************************
"""


def train_and_test(model_name, x_train, y_train, x_test, y_test, train_set_source_df_idx, test_set_source_df_idx,
                   batch_size, num_epochs, learning_rate, loss_function, horizon, device):
    model = get_model(model_name, horizon)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        train_epoch_loss = train_or_validate_one_epoch(epoch, x_train, y_train, train_set_source_df_idx,
                                                       model, loss_function, optimizer, batch_size, train_flag=True)
        avg_train_epoch_loss = train_epoch_loss / len(train_set_source_df_idx)
        train_losses.append(avg_train_epoch_loss)

        test_epoch_loss = train_or_validate_one_epoch(epoch, x_test, y_test, test_set_source_df_idx,
                                                      model, loss_function, optimizer, batch_size, train_flag=False)
        avg_test_epoch_loss = test_epoch_loss / len(test_set_source_df_idx)
        val_losses.append(avg_test_epoch_loss)

        tqdm.write(f'\nEpoch {epoch + 1} - Average Train Loss: {avg_train_epoch_loss:.3f}')
        tqdm.write(f'Epoch {epoch + 1} - Average Validation Loss: {avg_test_epoch_loss:.3f}')
        tqdm.write('----------------------------------------------')

    plot_convergence(train_losses, val_losses, num_epochs)
    plot_predictions(model, x_train, y_train, train_set_source_df_idx, modality='Train')
    plot_predictions(model, x_test, y_test, test_set_source_df_idx, modality='Validation')


"""
***********************************************************************************************************************
    Main
***********************************************************************************************************************
"""
if __name__ == '__main__':
    seed_everything()

    cuda_available = 0
    device = torch.device(f"cuda:{cuda_available}" if torch.cuda.is_available() else "cpu")

    metric = 'container_cpu'
    application_name = 'collector'

    sub_sample_rate = 1
    aggregation_type = 'max'
    data_length_limit_in_minutes = 60

    look_back = 12
    horizon = 2

    model_name = 'LSTM'
    learning_rate = 0.001
    num_epochs = 10
    # TODO: determine the loss function
    loss_function = nn.L1Loss(reduction='sum')
    batch_size = 16

    transformation_method = 'log'
    scale_method = 'min-max'

    dataset = framework__data_set.get_data_set(
        metric=metric,
        application_name=application_name,
        path_to_data='../data/'
    )

    print(f"Throwing out data that is less than {data_length_limit_in_minutes / 60} hours long.")
    dataset.filter_data_that_is_too_short(data_length_limit=data_length_limit_in_minutes)

    print(f"Subsampling data from 1 sample per 1 minute to 1 sample per {sub_sample_rate} minutes.")
    dataset.sub_sample_data(sub_sample_rate=sub_sample_rate, aggregation_type=aggregation_type)
    print("Splitting data into train and test.")
    train_dataset, test_dataset = dataset.split_to_train_and_test_Lee()
    print(f"Amount of dataframes in train data is {len(train_dataset)}, "
          f"with {sum(len(df) for df in train_dataset)} time samples.")
    print(f"Amount of dataframes in test data is {len(test_dataset)}, "
          f"with {sum(len(df) for df in test_dataset)} time samples.")

    print("Preparing data for time series.")
    train_dataset.prepare_dataset_for_time_series(look_back=look_back, horizon=horizon)
    test_dataset.prepare_dataset_for_time_series(look_back=look_back, horizon=horizon)

    train_dataset.record_df_indices()
    test_dataset.record_df_indices()

    train_dataset.concatenate_dfs()
    test_dataset.concatenate_dfs()

    train_set = deepcopy(train_dataset.dfs_concatenated)
    test_set = deepcopy(test_dataset.dfs_concatenated)
    train_set_source_df_idx = deepcopy(train_set.source_df_idx).to_numpy()
    test_set_source_df_idx = deepcopy(test_set.source_df_idx).to_numpy()
    train_set.drop('source_df_idx', axis=1, inplace=True)
    test_set.drop('source_df_idx', axis=1, inplace=True)

    train_set_as_np = train_set.to_numpy()
    test_set_as_np = test_set.to_numpy()

    print(f"Transforming and scaling data using {transformation_method} transformation and {scale_method} scaling.")
    train_set_as_np, scaler = transform_and_scale_data(train_set_as_np,
                                                       transformation_method=transformation_method,
                                                       scale_method=scale_method, scaler=None)
    test_set_as_np, _ = transform_and_scale_data(test_set_as_np, transformation_method=transformation_method,
                                                 scale_method=scale_method, scaler=scaler)

    print("Preparing data for training.")
    x_train, y_train = split_data_into_x_and_y(train_set_as_np, look_back, horizon)
    x_test, y_test = split_data_into_x_and_y(test_set_as_np, look_back, horizon)

    # flip the data to be along the time axis
    x_train = deepcopy(np.flip(x_train, axis=1))
    x_test = deepcopy(np.flip(x_test, axis=1))

    # reshape the data to be in the format for torch
    x_train = x_train.reshape((-1, look_back, 1))
    x_test = x_test.reshape((-1, look_back, 1))
    y_train = y_train.reshape((-1, horizon, 1))
    y_test = y_test.reshape((-1, horizon, 1))
    train_set_source_df_idx = train_set_source_df_idx.reshape((-1, 1))
    test_set_source_df_idx = test_set_source_df_idx.reshape((-1, 1))

    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()
    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()
    train_set_source_df_idx = torch.tensor(train_set_source_df_idx)
    test_set_source_df_idx = torch.tensor(test_set_source_df_idx)

    train_data = TimeSeriesDataset(x_train, y_train, train_set_source_df_idx, num_samples=len(train_set_source_df_idx))
    test_data = TimeSeriesDataset(x_test, y_test, test_set_source_df_idx, num_samples=len(test_set_source_df_idx))

    print("Training and testing the model.")
    train_and_test(model_name, x_train, y_train, x_test, y_test, train_set_source_df_idx, test_set_source_df_idx,
                   batch_size, num_epochs, learning_rate, loss_function, horizon, device)

    train_dataset.recover_dfs_from_concatenated()
    test_dataset.recover_dfs_from_concatenated()

    print("Done!")
