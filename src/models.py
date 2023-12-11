from darts.models import TCNModel, NBEATSModel, RNNModel, BlockRNNModel
from darts.utils.likelihood_models import GaussianLikelihood

import src.utils as utils


def tcn_model(look_back, horizon):
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
            optimizer_kwargs={"lr": 0.0003},
            kernel_size=5,
            weight_norm=True,
            dilation_base=4,
            likelihood=GaussianLikelihood(),
            random_state=0,
            **utils.generate_torch_kwargs()
        )
    )


def nbeats_model(look_back, horizon):
    return (
        NBEATSModel(
            model_name='NBEATS',
            input_chunk_length=look_back,
            output_chunk_length=horizon,
            n_epochs=1,
            generic_architecture=True,
            num_stacks=10,
            num_blocks=1,
            num_layers=4,
            layer_widths=512,
            activation='LeakyReLU',
            dropout=0,
            random_state=0,
            **utils.generate_torch_kwargs()
        )
    )


def deepar_model(look_back, horizon=None):
    return (
        RNNModel(
            model='LSTM',
            model_name='DeepAR',
            input_chunk_length=look_back,
            # TODO: check how to configure horizon
            # output_chunk_length=1,  # rnn outputs a single value at each step
            n_epochs=1,  # per dataframe
            batch_size=16,
            hidden_dim=20,
            n_rnn_layers=2,
            dropout=0,
            optimizer_kwargs={"lr": 1e-3},
            likelihood=GaussianLikelihood(),
            random_state=0,
            **utils.generate_torch_kwargs()
        )
    )


# TODO: add this model to the main.py
def lstm_model(look_back, horizon):
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
            dropout=0,
            random_state=0,
            **utils.generate_torch_kwargs()
        )
    )
