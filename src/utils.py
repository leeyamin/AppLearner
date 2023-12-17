import torch
import os
from darts.logging import get_logger
from typing import Union, Optional
from darts.models import TCNModel, NBEATSModel, RNNModel
import logging
import yaml

import src.models as models


class Config:
    """convert a dictionary to a class"""
    def __init__(self, **entries):
        self.__dict__.update(entries)


def get_config():
    with open('config.yml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    config = Config(**config)

    return config


def get_output_path(model_name: str) -> str:
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


def save_config_to_file(output_path: str, config, filename='config.yml'):
    """
    Save the configuration to a YAML file in the specified output path.
    @param output_path: the path to the output folder
    @param config: the configuration object
    @param filename: the name of the configuration file
    """
    os.makedirs(output_path, exist_ok=True)
    config_file_path = os.path.join(output_path, filename)
    with open(config_file_path, 'w') as config_file:
        yaml.dump(vars(config), config_file, default_flow_style=False)


def record_logs_to_txt(txt: str, output_path: str) -> None:
    """
    Record the logs of the run to a txt file in the output folder.
    @param txt: the logs to record
    @param output_path: the path to the output folder
    @return None
    """
    if output_path is not None:
        txt_file = open(f'{output_path}/logs.txt', "a")
        txt_file.write(f'{txt}\n')


def generate_torch_kwargs(gpu_idx: int):
    """
    Get the kwargs for the pytorch lightning trainer based on the gpu index and disable pytorch lightning logging.
    @param gpu_idx: the index of the gpu to use
    @return pl_trainer_kwargs: the kwargs for the pytorch lightning trainer
    """
    if torch.cuda.is_available():
        if gpu_idx is not None:
            return {
                "pl_trainer_kwargs": {"accelerator": "gpu", "devices": [gpu_idx],
                                      "enable_progress_bar": False}
            }
        else:
            raise NotImplementedError("GPU is available but not used.")
            # return {
            #     "pl_trainer_kwargs": {"accelerator": "gpu", "devices": -1, "auto_select_gpus": True,
            #                      "enable_progress_bar": False}
            # }
    else:
        return {
            "pl_trainer_kwargs": {"accelerator": "cpu",
                                  "enable_progress_bar": False}
        }


def get_model(model_name: str, look_back: int, horizon: int, gpu_idx: Optional[int], output_path: str) \
        -> Union[TCNModel, NBEATSModel, RNNModel]:
    """
    Get the configured model based on the model name.
    @param model_name: name of the model
    @param look_back: number of time steps to look back
    @param horizon: number of time steps to predict
    @param gpu_idx: index of the gpu to use (if available)
    @param output_path: path of output folder
    @return model: darts model
    """
    if model_name == "TCN":
        model = models.tcn_model(look_back, horizon, gpu_idx)
    elif model_name == "NBEATS":
        model = models.nbeats_model(look_back, horizon, gpu_idx)
    elif model_name == "DeepAR":
        model = models.deepar_model(look_back, horizon=None, gpu_idx=gpu_idx)
    elif model_name == "LSTM":
        model = models.lstm_model(look_back, horizon, gpu_idx)
    else:
        raise NotImplementedError

    if output_path is not None:
        record_logs_to_txt('\nModel:', output_path)
        record_logs_to_txt(model, output_path)

    return model


def load_model_if_exists(model: Union[TCNModel, NBEATSModel, RNNModel], trained_model_path: str) \
        -> Union[TCNModel, NBEATSModel, RNNModel]:
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
            logger.warning("Trained model path does not exist. Training from scratch.")

    return model


def disable_pytorch_lightning_logging() -> None:
    """Disable pytorch lightning logging."""
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def save_model(model: Union[TCNModel, NBEATSModel, RNNModel], model_name: str, output_path: str) -> None:
    """Save model weights to the output folder."""
    model.save(os.path.join(f'{output_path}', f'{model_name}.pth'))
    print("Model weights saved.")
