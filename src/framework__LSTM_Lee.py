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


# TODO: delete this after converting to main
seed_everything()

cuda_available = 0
device = torch.device(f"cuda:{cuda_available}" if torch.cuda.is_available() else "cpu")

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


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y, df_idx):
        self.x = x
        self.y = y
        self.df_idx = df_idx

    def __len__(self):
        return len(self.df_idx)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.df_idx[idx]


"""
***********************************************************************************************************************
    Models
***********************************************************************************************************************
"""

learning_rate = 0.001
num_epochs = 10
# TODO: determine the loss function
loss_function = nn.L1Loss(reduction='sum')


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, horizon):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = out.unsqueeze(-1)
        return out


def train_df_epoch(train_loader, model, loss_function, optimizer):
    model.train()
    df_train_epoch_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch, df_idx = batch
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
            x_batch, y_batch, df_idx = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            output = model(x_batch)
            loss = loss_function(output, y_batch)
            df_test_epoch_loss += loss.item()

    return df_test_epoch_loss


metric = 'container_cpu'
application_name = 'collector'

sub_sample_rate = 1
aggregation_type = 'max'
data_length_limit_in_minutes = 60

look_back = 12
horizon = 2

transformation_method = 'log'
scale_method = 'min-max'

dataset = framework__data_set.get_data_set(
    metric=metric,
    application_name=application_name,
    path_to_data='../data/'
)

print(f"Throwing out data that is less than {data_length_limit_in_minutes / 60} hours long.")
dataset.filter_data_that_is_too_short(data_length_limit=data_length_limit_in_minutes)

# TODO: remove before commit, this is just for testing
dataset.list_of_df = dataset.list_of_df[:10]

print(f"Subsampling data from 1 sample per 1 minute to 1 sample per {sub_sample_rate} minutes.")
dataset.sub_sample_data(sub_sample_rate=sub_sample_rate, aggregation_type=aggregation_type)
print("Splitting data into train and test.")
train_dataset, test_dataset = dataset.split_to_train_and_test_Lee()
print(f"Amount of dataframes in train data is {len(train_dataset)}, "
      f"with {sum(len(df) for df in train_dataset)} time samples.")
print(f"Amount of dataframes in test data is {len(test_dataset)}, "
      f"with {sum(len(df) for df in test_dataset)} time samples.")
print("Scaling data.")

# TODO: ablation study: try different look_back values and horizon values
# TODO: edit this function to get also horizon
train_dataset.prepare_dataset_for_lstm(look_back=look_back, horizon=horizon)
test_dataset.prepare_dataset_for_lstm(look_back=look_back, horizon=horizon)
# TODO: assert the shape of the data after transformation

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

train_set_as_np, scaler = transform_and_scale_data(train_set_as_np,
                                                   transformation_method=transformation_method,
                                                   scale_method=scale_method, scaler=None)
test_set_as_np, _ = transform_and_scale_data(test_set_as_np, transformation_method=transformation_method,
                                             scale_method=scale_method, scaler=scaler)

# split the data into x and y
x_train = train_set_as_np[:, horizon:horizon+look_back]
y_train = train_set_as_np[:, 0:horizon]
x_test = test_set_as_np[:, horizon:horizon+look_back]
y_test = test_set_as_np[:, 0:horizon]

# flip the data to be along the time axis
x_train = deepcopy(np.flip(x_train, axis=1))
x_test = deepcopy(np.flip(x_test, axis=1))

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

train_data = TimeSeriesDataset(x_train, y_train, train_set_source_df_idx)
test_data = TimeSeriesDataset(x_test, y_test, test_set_source_df_idx)

# IM HERE: TODO: update the model output to match horizon
model = LSTM(1, 4, 1, horizon=horizon)
model.to(device)
batch_size = 16
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    print(f'Epoch: {epoch + 1}')
    train_epoch_loss = 0.0
    for df_index in train_set_source_df_idx.unique():
        train_subset_source_df_index = train_set_source_df_idx[train_set_source_df_idx == df_index]
        subset_df_train_dataset = Subset(
            TimeSeriesDataset(x_train, y_train, train_subset_source_df_index.reshape((-1, 1))),
            indices=train_subset_source_df_index
        )
        dataloader_subset = DataLoader(subset_df_train_dataset, batch_size=batch_size, shuffle=False)
        df_train_epoch_loss = train_df_epoch(dataloader_subset, model, loss_function, optimizer)
        train_epoch_loss += df_train_epoch_loss

    avg_train_epoch_loss = train_epoch_loss / len(train_set_source_df_idx)
    print('Train Loss: {0:.3f}'.format(avg_train_epoch_loss))
    print()
    test_epoch_loss = 0.0
    for df_index in test_set_source_df_idx.unique():
        test_subset_source_df_index = test_set_source_df_idx[test_set_source_df_idx == df_index]
        subset_df_test_dataset = Subset(
            TimeSeriesDataset(x_test, y_test, test_subset_source_df_index.reshape((-1, 1))),
            indices=test_subset_source_df_index
        )
        dataloader_subset = DataLoader(subset_df_test_dataset, batch_size=batch_size, shuffle=False)
        df_test_epoch_loss = validate_df_epoch(dataloader_subset, model, loss_function)
        test_epoch_loss += df_test_epoch_loss

    avg_test_epoch_loss = test_epoch_loss / len(test_set_source_df_idx)
    print('Val Loss: {0:.3f}'.format(avg_test_epoch_loss))
    print('----------------------------------------------')
    print()

# TODO: define as a function and input train and test sets
for df_index in train_set_source_df_idx.unique():
    train_subset_source_df_index = train_set_source_df_idx[train_set_source_df_idx == df_index]
    subset_df_dataset = Subset(TimeSeriesDataset(x_train, y_train, train_subset_source_df_index.reshape((-1, 1))),
                               indices=train_subset_source_df_index)
    batch_size = 32
    dataloader_subset = DataLoader(subset_df_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    mae = nn.L1Loss(reduction='sum')
    rmse = nn.MSELoss(reduction='sum')
    mae_loss = 0.0
    rmse_loss = 0.0
    df_indices = torch.where(train_set_source_df_idx == df_index)[0]
    y_df = y_train[df_indices]
    predicted_df = torch.Tensor()
    for batch in dataloader_subset:
        x_batch, y_batch, _ = batch
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        with torch.no_grad():
            predicted = model(x_batch)

        mae_loss += mae(predicted, y_batch).item()
        rmse_loss += torch.sqrt(rmse(predicted, y_batch)).item()
        predicted_df = torch.cat([predicted_df, predicted.squeeze()], dim=0)

    mae_loss /= len(subset_df_dataset)
    rmse_loss /= len(subset_df_dataset)
    assert len(predicted_df) == len(subset_df_dataset) == len(y_df)

    y_df = y_df.numpy()
    predicted_df = predicted_df.numpy()

    # TODO: define as a function
    # rescale the data
    dummies = np.zeros((y_df.shape[0], look_back + horizon))
    dummies[:, 0:horizon] = y_df.squeeze()
    dummies = re_transform_and_re_scale_data(dummies,
                                             transformation_method=transformation_method,
                                             scale_method=scale_method, scaler=scaler)
    rescaled_y_df = deepcopy(dummies[:, 0:horizon])

    dummies = np.zeros((predicted_df.shape[0], look_back + horizon))
    dummies[:, 0:horizon] = predicted_df.squeeze()
    dummies = re_transform_and_re_scale_data(dummies,
                                             transformation_method=transformation_method,
                                             scale_method=scale_method, scaler=scaler)
    rescaled_predicted_df = deepcopy(dummies[:, 0:horizon])

    for i in range(rescaled_predicted_df.shape[1]):
        max_value = max(np.max(rescaled_predicted_df[:, i]), np.max(rescaled_y_df[:, i]))
        plt.plot(rescaled_predicted_df[:, i], label=f'Predicted Time Step {i + 1}', linestyle='dashed')

    plt.plot(rescaled_y_df[:, 0], label=f'Actual', linestyle='solid')
    plt.xlabel('Time')  # TODO: change to actual time
    plt.ylabel('Value')
    plt.title(f'(Train) DF: {df_index}; MAE: {mae_loss:.3f}; RMSE: {rmse_loss:.3f}')
    plt.ylim(0, max_value)
    plt.legend()
    plt.show()

train_dataset.recover_dfs_from_concatenated()
test_dataset.recover_dfs_from_concatenated()

"""
***********************************************************************************************************************
    Train and Test
***********************************************************************************************************************
"""

"""
***********************************************************************************************************************
    Main
***********************************************************************************************************************
"""
if __name__ == '__main__':
    seed_everything()
    print("Done!")
    os._exit(0)
