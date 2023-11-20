from darts.models import TCNModel, NBEATSModel, RNNModel
from darts.utils.likelihood_models import GaussianLikelihood


def tcn_model(look_back, horizon):
    return (
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
            model_name="TCN",
        )
    )


def nbeats_model(look_back, horizon):
    return (
        NBEATSModel(
            input_chunk_length=look_back,
            output_chunk_length=horizon,
            generic_architecture=True,
            num_stacks=30,
            num_blocks=1,
            n_epochs=1,  # default: 100
            num_layers=4,
            layer_widths=512,
            activation='LeakyReLU',
            pl_trainer_kwargs={"accelerator": "cpu"},
            model_name="NBEATS",
        )
    )


def deepar_model(look_back, horizon=None):
    return (
        RNNModel(
            model='LSTM',
            input_chunk_length=look_back,
            # TODO: check how to configure horizon
            # output_chunk_length=1,  # rnn outputs a single value at each step
            n_epochs=1,  # default: 100
            model_name="DeepAR",
            pl_trainer_kwargs={"accelerator": "cpu"},
        )
    )
