import torch.nn
from darts.models import TCNModel, RNNModel, BlockRNNModel
from darts.utils.likelihood_models import GaussianLikelihood

import src.utils as utils


def tcn_model(look_back, horizon, gpu_idx, output_path):
    return (
        TCNModel(
            model_name='TCN',
            input_chunk_length=look_back,
            output_chunk_length=horizon,
            n_epochs=1,
            batch_size=16,
            dropout=0,
            num_filters=5,
            num_layers=3,
            kernel_size=5,
            weight_norm=True,
            dilation_base=4,
            likelihood=GaussianLikelihood(),
            optimizer_kwargs={"lr": 0.05},  # lr for TCN determined by lr_find.py
            loss_fn=torch.nn.L1Loss(),
            random_state=0,
            save_checkpoints=True,
            work_dir=output_path,
            **utils.generate_torch_kwargs(gpu_idx),
        )
    )


def deepar_model(look_back, horizon, gpu_idx, output_path):
    return (
        RNNModel(
            model='LSTM',
            model_name='DeepAR',
            input_chunk_length=look_back,
            # output_chunk_length=1,  # rnn outputs a single value at each step
            n_epochs=1,  # per dataframe
            batch_size=16,
            hidden_dim=20,
            n_rnn_layers=2,
            dropout=0,
            optimizer_kwargs={"lr": 0.03},  # lr for DeepAR determined by lr_find.py
            loss_fn=torch.nn.L1Loss(),
            likelihood=GaussianLikelihood(),
            random_state=0,
            save_checkpoints=True,
            work_dir=output_path,
            **utils.generate_torch_kwargs(gpu_idx)
        )
    )


def lstm_model(look_back, horizon, gpu_idx, output_path):
    return (
        BlockRNNModel(
            model='LSTM',
            model_name='LSTM',
            input_chunk_length=look_back,
            output_chunk_length=horizon,
            n_epochs=1,  # per dataframe
            batch_size=16,
            hidden_dim=512,
            n_rnn_layers=2,
            optimizer_kwargs={"lr": 0.00007},  # lr for LSTM determined by lr_find.py
            loss_fn=torch.nn.L1Loss(),
            dropout=0,
            random_state=0,
            save_checkpoints=True,
            work_dir=output_path,
            **utils.generate_torch_kwargs(gpu_idx)
        )
    )
