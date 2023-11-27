import random
import numpy as np
import os
import torch
from darts.logging import get_logger
from argparse import ArgumentParser
from typing import Union
from darts.models import TCNModel, NBEATSModel, RNNModel

from src.config import config
import src.models as models


def seed_everything(seed: int = 42) -> None:
    """Seed parameters for reproducibility of results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_output_path(model_name: str = config.model_name) -> str:
    """
    Get the output path for a specific run as the first available index in the output folder based on the model name.
    @param model_name: the name of the model
    @return output_path: the path to the output folder
    """
    idx = 0
    while True:
        output_path = f"../output/{model_name}_{idx}"
        if not os.path.exists(output_path):
            return output_path
        idx += 1


def record_config(config_args: ArgumentParser) -> None:
    """
    Record the configuration of the run to a txt file.
    @param config_args: the configuration of the run as a parser object
    @return None
    """
    os.makedirs(f'{config.output_path}', exist_ok=True)
    txt_file = open(f'{config.output_path}/logs.txt', "w")

    for arg in vars(config_args):
        txt_file.write(f"config: {arg} = {getattr(config_args, arg)}\n")
    txt_file.write("\n")

    txt_file.close()


def record_logs_to_txt(txt: str, output_path: str = config.output_path) -> None:
    """
    Record the logs of the run to a txt file in the output folder.
    @param txt: the logs to record
    @param output_path: the path to the output folder
    @return None
    """
    txt_file = open(f'{output_path}/logs.txt', "a")
    txt_file.write(f'{txt}\n')


def get_model(model_name: str = config.model_name,
              look_back: int = config.look_back,
              horizon: int = config.horizon,
              output_path: str = config.output_path) -> Union[TCNModel, NBEATSModel, RNNModel]:
    """
    Get the configured model based on the model name.
    @param model_name: name of the model
    @param look_back: number of time steps to look back
    @param horizon: number of time steps to predict
    @param output_path: path of output folder
    @return model: darts model
    """
    if model_name == "TCN":
        model = models.tcn_model(look_back, horizon)
    elif model_name == "NBEATS":
        model = models.nbeats_model(look_back, horizon)
    elif model_name == "DeepAR":
        model = models.deepar_model(look_back, horizon=None)
    else:
        raise NotImplementedError

    record_logs_to_txt(model, output_path)
    record_logs_to_txt('\n', output_path)

    return model


def load_model_if_exists(model: Union[TCNModel, NBEATSModel, RNNModel],
                         trained_model_path: str = config.trained_model_path) -> Union[TCNModel, NBEATSModel, RNNModel]:
    """
    Load model weights if the model path exists.
    @param model: predefined darts model
    @param trained_model_path: path to the trained model weights
    @return model: trained darts model
    """

    logger = get_logger(__name__)

    if trained_model_path is not None:
        if os.path.exists(trained_model_path):
            if os.path.exists(trained_model_path):
                logger.info(f"Loading model weights from {trained_model_path}")
                model.load(trained_model_path)
            else:
                logger.warning("Weights file not found. Training from scratch.")
        else:
            logger.warning("Output path does not exist. Training from scratch.")

    return model


def save_model(model: Union[TCNModel, NBEATSModel, RNNModel], model_name: str = config.model_name,
               output_path: str = config.output_path) -> None:
    """Save model weights to the output folder."""
    logger = get_logger(__name__)
    model.save(os.path.join(f'{config.output_path}', f'{model_name}.pth'))
    logger.info(f"Model weights saved to {os.path.join(f'{output_path}', f'{model_name}.pth')}")


if __name__ == '__main__':
    seed_everything()
    print(get_output_path())
