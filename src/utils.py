import random
import numpy as np
import os
import torch
from darts.logging import get_logger

from src.config import config
import src.models as models


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_output_path():
    idx = 0
    while True:
        output_path = f"../output/{config.model_name}_{idx}"
        if not os.path.exists(output_path):
            return output_path
        idx += 1


def record_config(config_args):
    os.makedirs(f'{config.output_path}', exist_ok=True)
    txt_file = open(f'{config.output_path}/logs.txt', "w")

    for arg in vars(config_args):
        txt_file.write(f"config: {arg} = {getattr(config_args, arg)}\n")
    txt_file.write("\n")

    txt_file.close()


def record_logs_to_txt(txt):
    txt_file = open(f'{config.output_path}/logs.txt', "a")
    txt_file.write(f'{txt}\n')


def get_model():
    if config.model_name == "TCN":
        # TODO: define hyper parameters from config
        model = models.tcn_model(look_back=config.look_back, horizon=config.horizon)
    # TODO: configure the plot according to the model (quantiles)
    elif config.model_name == "NBEATS":
        model = models.nbeats_model(look_back=config.look_back, horizon=config.horizon)
    # TODO: define hyper parameters from config
    # TODO: validate the model configuration is deepar
    elif config.model_name == "DeepAR":
        # TODO: check horizon configuration in deep ar model
        model = models.deepar_model(look_back=config.look_back, horizon=None)
    else:
        raise NotImplementedError

    record_logs_to_txt(model)
    record_logs_to_txt('\n')

    return model


def load_model_if_exists(model):
    logger = get_logger(__name__)

    if config.trained_model_path is not None:
        if os.path.exists(config.trained_model_path):
            if os.path.exists(config.trained_model_path):
                logger.info(f"Loading model weights from {config.trained_model_path}")
                model.load(config.trained_model_path)
            else:
                logger.warning("Weights file not found. Training from scratch.")
        else:
            logger.warning("Output path does not exist. Training from scratch.")

    return model


def save_model(model, model_name):
    logger = get_logger(__name__)
    model.save(os.path.join(f'{config.output_path}', f'{model_name}.pth'))
    logger.info(f"Model weights saved to {os.path.join(f'{config.output_path}', f'{model_name}.pth')}")


if __name__ == '__main__':
    seed_everything()
    print(get_output_path())
