from darts.models import TCNModel, NBEATSModel, RNNModel
from darts.utils.likelihood_models import GaussianLikelihood

import src.config as config


def get_model(model_name):
    if model_name == "TCN":
        # TODO: define hyper parameters from config
        model = (
            TCNModel(
                dropout=0.2,
                batch_size=16,
                n_epochs=1,
                num_filters=5,
                num_layers=3,
                optimizer_kwargs={"lr": 0.0003},
                random_state=0,
                input_chunk_length=config.look_back,
                output_chunk_length=config.horizon,
                kernel_size=5,
                weight_norm=True,
                dilation_base=4,
                likelihood=GaussianLikelihood(),
                pl_trainer_kwargs={"accelerator": "cpu"},
                model_name="TCN",
                log_tensorboard=True,
                work_dir="../output",
            ))

    # TODO: define hyper parameters from config
    # TODO: configure the plot according to the model (quantiles)
    elif model_name == "NBEATS":
        model = (
            NBEATSModel(
                input_chunk_length=config.look_back,
                output_chunk_length=config.horizon,
                generic_architecture=True,
                num_stacks=30,
                num_blocks=1,
                n_epochs=1,  # default: 100
                num_layers=4,
                layer_widths=512,
                activation='LeakyReLU',
                pl_trainer_kwargs={"accelerator": "cpu"},
                model_name="NBEATS",
                log_tensorboard=True,
                work_dir="../output",
            ))
    # TODO: define hyper parameters from config
    # TODO: validate the model configuration is deepar
    elif model_name == "DeepAR":
        model = (
            RNNModel(
                model='LSTM',
                input_chunk_length=config.look_back,
                output_chunk_length=config.horizon,
                n_epochs=1,  # default: 100
                model_name="DeepAR",
                log_tensorboard=True,
                work_dir="../output",
                pl_trainer_kwargs={"accelerator": "cpu"},
            ))
    else:
        raise NotImplementedError

    return model
