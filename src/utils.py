import numpy as np
import pandas as pd
import torch
import os
from typing import Union, Optional
from darts.models import TCNModel, NBEATSModel, RNNModel, BlockRNNModel
import logging
import yaml
import warnings

import src.models as models

warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.callbacks.model_checkpoint")


class Config:
    """convert a dictionary to a class"""

    def __init__(self, **entries):
        self.__dict__.update(entries)


def get_config(config_path: str = 'config.yml'):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    config = Config(**config)

    return config


def get_output_path(run_name: Optional[str], model_name: str) -> str:
    """
    Get the output path for a specific run as the first available index in the output folder based on the model name.
    @param run_name: optionally predefined run name
    @param model_name: the name of the model
    @return output_path: the path to the output folder
    """
    base_path = os.path.join('..', 'output')
    idx = 0
    while True:
        if run_name is not None:
            output_path = os.path.join(base_path, f'{run_name}_{idx}')
        else:
            output_path = os.path.join(base_path, f'{model_name}_{idx}')
        if not os.path.exists(output_path):
            return output_path
        idx += 1


def seed_everything(seed: int = 42) -> None:
    """
    Seed all random number generators to make the results reproducible.
    @param seed: the seed to use
    @return None
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


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
    record_logs_to_txt('\nConfig:', output_path)
    for key, value in vars(config).items():
        record_logs_to_txt(f'{key}: {value}', output_path)
    record_logs_to_txt('\n', output_path)


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


def get_model(model_name: str, look_back: int, horizon: int, gpu_idx: Optional[int], output_path: str,
              trained_model_path: str) -> Union[TCNModel, NBEATSModel, RNNModel, BlockRNNModel]:
    """
    Get the configured model based on the model name.
    @param model_name: name of the model
    @param look_back: number of time steps to look back
    @param horizon: number of time steps to predict
    @param gpu_idx: index of the gpu to use (if available)
    @param output_path: path of output folder
    @param trained_model_path: path of the trained model
    @return model: darts model
    """

    if trained_model_path is not None:
        assert model_name == trained_model_path.split('_')[0], "Model name and trained model path do not match."
        main_dir = os.path.dirname(os.getcwd())
        work_dir = os.path.join(main_dir, 'output', trained_model_path)
        model = load_model(model_name=model_name, work_dir=work_dir)
        if model is not None:
            msg = f"Loading model from {os.path.join(work_dir, model_name, 'checkpoints', 'last.ckpt')}"
            print(msg)
            if output_path is not None:
                record_logs_to_txt(f'\n{msg}', output_path)
                record_logs_to_txt('\nModel:', output_path)
                record_logs_to_txt(model, output_path)
            return model

    msg = f"Training from scratch."
    print(msg)
    record_logs_to_txt(f'\n{msg}', output_path)
    if model_name == "TCN":
        model = models.tcn_model(look_back, horizon, gpu_idx, output_path)
    elif model_name == "NBEATS":
        model = models.nbeats_model(look_back, horizon, gpu_idx, output_path)
    elif model_name == "DeepAR":
        model = models.deepar_model(look_back, horizon=None, gpu_idx=gpu_idx, output_path=output_path)
    elif model_name == "LSTM":
        model = models.lstm_model(look_back, horizon, gpu_idx, output_path)
    else:
        raise NotImplementedError(f"Model {model_name} is not implemented.")

    if output_path is not None:
        record_logs_to_txt('\nModel:', output_path)
        record_logs_to_txt(model, output_path)

    return model


def disable_pytorch_lightning_logging() -> None:
    """Disable pytorch lightning logging."""
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def load_model(model_name, work_dir, file_name='last-epoch=0.ckpt'):
    """
    Load a model from a checkpoint that was automatically saved by pytorch lightning (darts).
    @param model_name: name of the model: TCN, NBEATS, RNN, LSTM
    @param work_dir: path of the output folder as set in work_dir in the model configuration
    @param file_name: name of the checkpoint file, default is 'last-epoch=0.ckpt' which is the last checkpoint
    """

    model_classes = {
        'TCN': TCNModel,
        'NBEATS': NBEATSModel,
        'DeepAR': RNNModel,
        'LSTM': BlockRNNModel
    }
    assert model_name in model_classes, f"Unsupported model name: {model_name}"
    assert os.path.exists(work_dir), f"Work directory does not exist: {work_dir}"
    assert os.path.exists(os.path.join(work_dir, model_name, 'checkpoints', file_name))

    print(f"Loading model from {os.path.join(work_dir, model_name, 'checkpoints', file_name)}")
    model_class = model_classes[model_name]
    model = model_class.load_from_checkpoint(model_name=model_name, work_dir=work_dir, file_name=file_name)

    return model


def save_dict_to_csv(dict, output_path, filename):
    """
    Save a dictionary to a csv file.
    @param dict: the dictionary to save
    @param output_path: path of the output folder
    @param filename: name of the csv file
    """
    df_dict = pd.DataFrame(dict)
    df_dict.to_csv(f'{output_path}/{filename}.csv', index=False)
