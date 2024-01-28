"""
This script is a standalone as part of the project preprocessing.
This script is used to find the optimal learning rate for a given model.
Models are implemented from scratch to be used with fastai, and are essentially the same as implemented using darts.
The optimal learning rate is the one that is the steepest descent.
"""

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader
from fastai.data.core import DataLoaders
from torch.distributions import Normal
from fastai.callback.schedule import LRFinder  # do not delete this import
from fastai.learner import Learner
import matplotlib.pyplot as plt
import argparse

import src.framework__data_set as framework__data_set
import src.utils as utils

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of data series to use')
    parser.add_argument('--model_name', type=str, default='LSTM', help='LSTM/ DeepAR/ TCN')
    args = parser.parse_args()

    utils.seed_everything()
    config = utils.get_config()
    config.output_path = None

    data = framework__data_set.get_data_set(
        metric=config.metric,
        application_name=config.application_name,
        path_to_data='../data/'
    )

    data.set_configurations(config)
    data.prepare_data_for_run(config.output_path, record_logs_to_txt=True)
    data.split_to_train_and_test()
    data.transform_and_scale_data()

    train_dfs = data.get_train_time_series_data()
    val_dfs = data.get_val_time_series_data()
    train_series_dict = {}
    val_series_dict = {}

    if args.limit is not None:
        batch_size = args.limit * 100
    else:
        # determined as the largest batch size that does not cause a memory error
        batch_size = 10000

    unique_train_dfs = train_dfs['source_df_idx'].unique()[:args.limit] if args.limit is not None else (
        train_dfs['source_df_idx'].unique())
    for df_idx in unique_train_dfs:
        df_dataset = train_dfs[train_dfs['source_df_idx'] == df_idx]
        df_sample = df_dataset['sample']
        train_series_dict[df_idx] = df_sample

    unique_val_dfs = val_dfs['source_df_idx'].unique()[:args.limit] if args.limit is not None else (
        val_dfs['source_df_idx'].unique())
    for df_idx in unique_val_dfs:
        df_dataset = val_dfs[val_dfs['source_df_idx'] == df_idx]
        df_sample = df_dataset['sample']
        val_series_dict[df_idx] = df_sample

    train_series = list(train_series_dict.values())
    val_series = list(val_series_dict.values())

    train_sequences = [series.values for series in train_series]
    val_sequences = [series.values for series in val_series]


    class VariableLengthDataset(Dataset):
        def __init__(self, sequences, look_back, horizon):
            self.sequences = sequences
            self.look_back = look_back
            self.horizon = horizon

        def __len__(self):
            total_samples = 0
            for sequence in self.sequences:
                total_samples += len(sequence) - self.look_back - self.horizon
            return total_samples

        def __getitem__(self, idx):
            sequence_idx = 0
            while idx >= len(self.sequences[sequence_idx]) - self.look_back:
                idx -= len(self.sequences[sequence_idx]) - self.look_back
                sequence_idx += 1

            # Retrieve the sample from the selected sequence
            sequence = self.sequences[sequence_idx]

            x = sequence[idx:idx + self.look_back]
            y = sequence[idx + self.look_back:idx + self.look_back + self.horizon]

            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

            x = x.to('cpu')
            y = y.to('cpu')

            return x, y


    def collate_fn(batch):
        inputs, targets = zip(*batch)

        max_input_len = max(seq.shape[1] for seq in inputs)
        max_target_len = max(seq.shape[1] for seq in targets)

        padded_inputs = [torch.nn.functional.pad(input_seq, (0, max_input_len - input_seq.shape[1]), value=0) for
                         input_seq in inputs]
        padded_targets = [torch.nn.functional.pad(target_seq, (0, max_target_len - target_seq.shape[1]), value=0) for
                          target_seq in targets]

        batch_inputs = torch.stack(padded_inputs)
        batch_targets = torch.stack(padded_targets)

        return batch_inputs, batch_targets


    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_dim, output_size, n_rnn_layers, dropout=0):
            super(LSTMModel, self).__init__()

            self.lstm = nn.LSTM(input_size=input_size,
                                hidden_size=hidden_dim,
                                num_layers=n_rnn_layers,
                                dropout=dropout,
                                batch_first=True)

            self.linear = nn.Linear(hidden_dim, output_size)
            self.smooth_loss = torch.tensor(0.0)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            lstm_out = lstm_out[:, -1, :]
            output = self.linear(lstm_out)

            return output


    class DeepARModel(nn.Module):
        def __init__(self, input_size, hidden_dim, output_size, n_rnn_layers, dropout=0):
            super(DeepARModel, self).__init__()

            self.lstm = nn.LSTM(input_size=input_size,
                                hidden_size=hidden_dim,
                                num_layers=n_rnn_layers,
                                dropout=dropout,
                                batch_first=True)

            self.linear = nn.Linear(hidden_dim, output_size)
            self.smooth_loss = torch.tensor(0.0)
            self.hidden_dim = hidden_dim
            self.output_size = output_size

        def forward(self, x):
            lstm_out, _ = self.lstm(x)

            # Only take the output from the final time step
            lstm_out = lstm_out[:, -1, :]

            # Feed the output through the linear layer
            output = self.linear(lstm_out)

            return output


    class DeepARLoss(nn.Module):
        def __init__(self):
            super(DeepARLoss, self).__init__()

        def forward(self, output, target):
            # Compute Gaussian likelihood loss
            distribution = Normal(output, torch.ones_like(output))  # Assuming unit variance
            loss = -distribution.log_prob(target).mean()

            return loss


    class TCNBlock(nn.Module):
        def __init__(self, input_size, output_size, kernel_size, dilation, dropout):
            super(TCNBlock, self).__init__()
            self.conv = nn.Conv1d(input_size, output_size, kernel_size, dilation=dilation)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            out = self.conv(x)
            out = self.activation(out)
            out = self.dropout(out)
            return out


    class TCNModel(nn.Module):
        def __init__(self, input_size, output_size, num_layers, num_filters, kernel_size, dilation_base, dropout):
            super(TCNModel, self).__init__()
            self.num_layers = num_layers
            self.tcn_blocks = nn.ModuleList([
                TCNBlock(input_size if i == 0 else num_filters,
                         num_filters,
                         kernel_size,
                         dilation_base ** i,
                         dropout) for i in range(num_layers)
            ])
            self.linear = nn.Linear(num_filters, output_size)

        def forward(self, x):
            out = x.permute(0, 2, 1)  # Convert to (batch_size, input_size, sequence_length)
            for layer in self.tcn_blocks:
                out = layer(out)
            out = out[:, :, -1]  # Only take the output from the final time step
            out = self.linear(out)
            return out


    if args.model_name == 'LSTM':
        input_size = config.look_back
        output_size = config.horizon
        hidden_dim = 512
        n_rnn_layers = 2
        dropout = 0
        model = LSTMModel(input_size, hidden_dim, output_size, n_rnn_layers, dropout)
        loss_func = torch.nn.L1Loss()

    elif args.model_name == 'DeepAR':
        input_size = config.look_back
        output_size = config.horizon
        hidden_dim = 20
        n_rnn_layers = 2
        dropout = 0
        model = DeepARModel(input_size, hidden_dim, output_size, n_rnn_layers, dropout)
        loss_func = DeepARLoss()

    elif args.model_name == 'TCN':
        input_size = config.look_back
        output_size = config.horizon
        num_filters = 5
        num_layers = 3
        kernel_size = 1  # a larger kernel size would be better, but would require padding
        dilation_base = 4
        dropout = 0
        model = TCNModel(input_size, output_size, num_layers, num_filters, kernel_size, dilation_base, dropout)
        loss_func = torch.nn.L1Loss()

    else:
        raise NotImplementedError(f'Model {args.model_name} not implemented')

    model = model.to('cpu')

    train_ds = VariableLengthDataset(train_sequences, config.look_back, config.horizon)
    val_ds = VariableLengthDataset(val_sequences, config.look_back, config.horizon)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    dls = DataLoaders(train_dl, val_dl)

    learner = Learner(dls=dls, model=model, loss_func=loss_func)
    optimal_lr = learner.lr_find()  # the optimal lr is the one that is the steepest descent
    optimal_lr = optimal_lr[0]
    learner.recorder.plot_lr_find()
    plt.title(f'Fastai Learning Rate Finder: \n'
              f'({args.model_name}) optimal lr={optimal_lr}')
    plt.show()
    plt.close()

