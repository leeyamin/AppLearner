"""
***********************************************************************************************************************
    Import Libraries
***********************************************************************************************************************
"""
import os
import numpy as np
import random
import torch

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

"""
***********************************************************************************************************************
    Data
***********************************************************************************************************************
"""
metric = 'container_cpu'
application_name = 'collector'

sub_sample_rate = 1
aggregation_type = 'max'
data_length_limit_in_minutes = 60

look_back = 12

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
# TODO: determine the common technique to scale the data
# train_dataset.scale_data()
# test_dataset.scale_data()


# TODO: ablation study: try different look_back values and horizon values
# TODO: edit this function to get also horizon
train_dataset.prepare_dataset_for_lstm(look_back=look_back)
test_dataset.prepare_dataset_for_lstm(look_back=look_back)
# TODO: assert the shape of the data after transformation

train_dataset.record_df_indices()
test_dataset.record_df_indices()

train_dataset.concatenate_dfs()
test_dataset.concatenate_dfs()

from copy import deepcopy
train_set = deepcopy(train_dataset.dfs_concatenated)
test_set = deepcopy(test_dataset.dfs_concatenated)
train_set_source_df_idx = deepcopy(train_set.source_df_idx).to_numpy()
test_set_source_df_idx = deepcopy(test_set.source_df_idx).to_numpy()
train_set.drop('source_df_idx', axis=1, inplace=True)
test_set.drop('source_df_idx', axis=1, inplace=True)

train_set_as_np = train_set.to_numpy()
test_set_as_np = test_set.to_numpy()

# transform the data
epsilon = 1e-8
train_set_as_np = np.log(train_set_as_np + epsilon)
test_set_as_np = np.log(test_set_as_np + epsilon)

# scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
# TODO: check the normalization technique of train over test
scaler.fit_transform(train_set_as_np)
train_set_as_np = scaler.transform(train_set_as_np)
test_set_as_np = scaler.transform(test_set_as_np)

# split the data into x and y
x_train = train_set_as_np[:, 1:]
y_train = train_set_as_np[:, 0]
x_test = test_set_as_np[:, 1:]
y_test = test_set_as_np[:, 0]

from copy import deepcopy

# flip the data to be along the time axis
x_train = deepcopy(np.flip(x_train, axis=1))
x_test = deepcopy(np.flip(x_test, axis=1))

# TODO: update shape with horizon
x_train = x_train.reshape((-1, look_back, 1))
x_test = x_test.reshape((-1, look_back, 1))
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))
train_set_source_df_idx = train_set_source_df_idx.reshape((-1, 1))
test_set_source_df_idx = test_set_source_df_idx.reshape((-1, 1))

x_train = torch.tensor(x_train).float()
x_test = torch.tensor(x_test).float()
y_train = torch.tensor(y_train).float()
y_test = torch.tensor(y_test).float()
train_set_source_df_idx = torch.tensor(train_set_source_df_idx)
test_set_source_df_idx = torch.tensor(test_set_source_df_idx)

from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y, df_idx):
        self.x = x
        self.y = y
        self.df_idx = df_idx

    def __len__(self):
        return len(self.df_idx)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.df_idx[idx]


train_data = TimeSeriesDataset(x_train, y_train, train_set_source_df_idx)
test_data = TimeSeriesDataset(x_test, y_test, test_set_source_df_idx)

batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

cuda_available = 0
device = torch.device(f"cuda:{cuda_available}" if torch.cuda.is_available() else "cpu")

# sanity check
for batch in train_loader:
    x_batch, y_batch, df_idx = batch
    assert x_batch.shape == (batch_size, look_back, 1)
    assert y_batch.shape == (batch_size, 1)
    assert df_idx.shape == (batch_size, 1)
    break

import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


model = LSTM(1, 4, 1)
model.to(device)


def train_one_epoch(epoch_num, train_loader, model, loss_function, optimizer):
    model.train()
    print(f'Epoch: {epoch_num + 1}')
    epoch_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch, _ = batch
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_epoch_loss = epoch_loss / len(train_loader)
    print('Train Loss: {0:.3f}'.format(avg_epoch_loss))
    print()


def validate_one_epoch(test_loader, model, loss_function):
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader):
            x_batch, y_batch, _ = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            output = model(x_batch)
            loss = loss_function(output, y_batch)
            epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(test_loader)
    print('Val Loss: {0:.3f}'.format(avg_epoch_loss))
    print('----------------------------------------------')
    print()


learning_rate = 0.001
num_epochs = 10
# TODO: determine the loss function
loss_function = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    train_one_epoch(epoch, train_loader, model, loss_function, optimizer)
    validate_one_epoch(test_loader, model, loss_function)


from torch.utils.data import Subset

# TODO: define as a function and input train and test sets
for df_index in train_set_source_df_idx.unique():
    train_subset_source_df_index = train_set_source_df_idx[train_set_source_df_idx == df_index]
    subset_df_dataset = Subset(TimeSeriesDataset(x_train, y_train, train_subset_source_df_index.reshape((-1, 1))),
                               indices=train_subset_source_df_index)
    batch_size = 32
    dataloader_subset = DataLoader(subset_df_dataset, batch_size=batch_size, shuffle=False)
    # sanity check
    for batch in dataloader_subset:
        x_batch, y_batch, df_idx = batch
        assert x_batch.shape == (batch_size, look_back, 1)
        assert y_batch.shape == (batch_size, 1)
        assert df_idx.shape == (batch_size, 1)
        break

    model.eval()
    mae = nn.L1Loss(reduction='sum')
    rmse = nn.MSELoss(reduction='sum')
    mae_loss = 0.0
    rmse_loss = 0.0
    predicted_list = []
    y_df = y_train[train_set_source_df_idx == df_index]
    for batch in dataloader_subset:
        x_batch, y_batch, _ = batch
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        with torch.no_grad():
            predicted = model(x_batch)
            predicted = predicted
            y_batch = y_batch

        mae_loss += mae(predicted, y_batch).item()
        rmse_loss += torch.sqrt(rmse(predicted, y_batch)).item()
        predicted_list.extend(predicted.squeeze().cpu().numpy())

    mae_loss /= len(subset_df_dataset)
    rmse_loss /= len(subset_df_dataset)
    assert len(predicted_list) == len(subset_df_dataset) == len(y_df)

    # TODO: normalize back the data
    # log transformation
    # min-max scaler transformation

    import matplotlib.pyplot as plt
    plt.plot(y_df, label='Actual')
    plt.plot(predicted_list, label='Predicted')
    plt.title(f'DF: {df_index}; MAE: {mae_loss:.3f}; RMSE: {rmse_loss:.3f}')
    plt.xlabel('Time')  # TODO: change to time
    plt.ylabel('Usage')
    plt.legend()
    plt.show()

train_dataset.recover_dfs_from_concatenated()
test_dataset.recover_dfs_from_concatenated()


"""
***********************************************************************************************************************
    Model
***********************************************************************************************************************
"""

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
