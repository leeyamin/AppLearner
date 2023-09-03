import numpy as np
import torch
from copy import deepcopy
from typing import Optional, Tuple

import src.config as config


def transform_data(dataset_as_np, transformation_method='log'):
    """
    Transform the data using the specified method. Currently only supports log transformation.
    """
    if transformation_method == 'log':
        epsilon = 1e-8
        dataset_as_np = np.log(dataset_as_np + epsilon)
        return dataset_as_np


def scale_data(dataset_as_np, scale_method='min-max', scaling_attributes=None):
    """
    Scale the data using the specified method. Currently only supports min-max scaling.
    """
    if scale_method == 'min-max':
        if scaling_attributes is None:
            min = dataset_as_np.min()
            max = dataset_as_np.max()
            scaling_attributes = {'min': min, 'max': max}
        else:
            min = scaling_attributes['min']
            max = scaling_attributes['max']
        dataset_as_np = (dataset_as_np - min) / (max - min)
    return dataset_as_np, scaling_attributes


def transform_and_scale_data(dataset_as_np: np.ndarray,
                             transformation_method: str = 'log',
                             scale_method: str = 'min-max',
                             scaling_attributes: Optional[dict] = None) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Transform and scale the data using the specified methods. Currently only supports log transformation and min-max
    scaling.
    """
    if transformation_method not in ['log', None] or scale_method not in ['min-max', None]:
        raise NotImplementedError
    if transformation_method is None and scale_method is None:
        return dataset_as_np, None
    if transformation_method is not None:
        dataset_as_np = transform_data(dataset_as_np, transformation_method=transformation_method)
    if scale_method is not None:
        dataset_as_np, scaling_attributes = scale_data(dataset_as_np,
                                                       scale_method=scale_method, scaling_attributes=scaling_attributes)
    return dataset_as_np, scaling_attributes


def split_data_into_x_and_y(data, look_back, horizon):
    x = data[:, horizon:horizon + look_back]
    y = data[:, 0:horizon]
    return x, y


def re_scale_data(dataset_as_np, scale_method, scaling_attributes):
    """
    Rescale the data. Currently only supports min-max re-scaling.
    """
    if scale_method == 'min-max':
        min = scaling_attributes['min']
        max = scaling_attributes['max']
        dataset_as_np = dataset_as_np * (max - min) + min
    return dataset_as_np


def re_transform_data(dataset_as_np, transformation_method='log'):
    """
    Reverse transforms the data using the specified method. Currently only supports log transformation.
    """
    if transformation_method == 'log':
        dataset_as_np = np.exp(dataset_as_np)
    return dataset_as_np


def re_transform_and_re_scale_data(dataset_as_np: np.ndarray,
                                   transformation_method: str = 'log',
                                   scale_method: str = 'min-max',
                                   scaling_attributes: Optional[dict] = None) -> np.ndarray:
    """
    Reverse transforms and scales the data using the specified methods. Currently only supports log transformation and min-max
    scaling.
    """
    if transformation_method not in ['log', None] or scale_method not in ['min-max', None]:
        raise NotImplementedError
    if transformation_method is None and scale_method is None:
        return dataset_as_np
    if scale_method is not None:
        dataset_as_np = re_scale_data(dataset_as_np, scale_method, scaling_attributes)
    if transformation_method is not None:
        dataset_as_np = re_transform_data(dataset_as_np, transformation_method=transformation_method)
    return dataset_as_np


def prepare_data_for_run(data):
    print(f"Throwing out data that is less than {config.data_length_limit_in_minutes / 60} hours long.")
    data.filter_data_that_is_too_short(data_length_limit=config.data_length_limit_in_minutes)

    print(f"Subsampling data from 1 sample per 1 minute to 1 sample per {config.sub_sample_rate} minutes.")
    data.sub_sample_data(sub_sample_rate=config.sub_sample_rate, aggregation_type=config.aggregation_type)
    print("Splitting data into train and test.")
    train_dataset, test_dataset = data.split_to_train_and_test_Lee()
    print(f"Amount of dataframes in train data is {len(train_dataset)}, "
          f"with {sum(len(df) for df in train_dataset)} time samples.")
    print(f"Amount of dataframes in test data is {len(test_dataset)}, "
          f"with {sum(len(df) for df in test_dataset)} time samples.")

    print("Preparing data for time series.")
    train_dataset.prepare_dataset_for_time_series(look_back=config.look_back, horizon=config.horizon)
    test_dataset.prepare_dataset_for_time_series(look_back=config.look_back, horizon=config.horizon)

    train_dataset.record_df_indices()
    test_dataset.record_df_indices()

    train_dataset.concatenate_dfs()
    test_dataset.concatenate_dfs()

    train_set = deepcopy(train_dataset.dfs_concatenated)
    test_set = deepcopy(test_dataset.dfs_concatenated)
    train_set_source_df_idx = deepcopy(train_set.source_df_idx).to_numpy()
    test_set_source_df_idx = deepcopy(test_set.source_df_idx).to_numpy()
    train_set.drop('source_df_idx', axis=1, inplace=True)
    test_set.drop('source_df_idx', axis=1, inplace=True)

    train_set_as_np = train_set.to_numpy()
    test_set_as_np = test_set.to_numpy()

    print("Preparing data for training.")
    x_train, y_train = split_data_into_x_and_y(train_set_as_np, config.look_back, config.horizon)
    x_test, y_test = split_data_into_x_and_y(test_set_as_np, config.look_back, config.horizon)

    print(f"Transforming and scaling data using {config.transformation_method} transformation "
          f"and {config.scale_method} scaling.")
    x_train, scaling_attributes = transform_and_scale_data(x_train,
                                                           transformation_method=config.transformation_method,
                                                           scale_method=config.scale_method,
                                                           scaling_attributes=None)
    x_test, _ = transform_and_scale_data(x_test,
                                         transformation_method=config.transformation_method,
                                         scale_method=config.scale_method,
                                         scaling_attributes=scaling_attributes)
    y_train, _ = transform_and_scale_data(y_train,
                                          transformation_method=config.transformation_method,
                                          scale_method=config.scale_method,
                                          scaling_attributes=scaling_attributes)
    y_test, _ = transform_and_scale_data(y_test,
                                         transformation_method=config.transformation_method,
                                         scale_method=config.scale_method,
                                         scaling_attributes=scaling_attributes)

    # flip the data to be along the time axis
    x_train = deepcopy(np.flip(x_train, axis=1))
    x_test = deepcopy(np.flip(x_test, axis=1))

    # reshape the data to be in the format for torch
    x_train = x_train.reshape((-1, config.look_back, 1))
    x_test = x_test.reshape((-1, config.look_back, 1))
    y_train = y_train.reshape((-1, config.horizon, 1))
    y_test = y_test.reshape((-1, config.horizon, 1))
    train_set_source_df_idx = train_set_source_df_idx.reshape((-1, 1))
    test_set_source_df_idx = test_set_source_df_idx.reshape((-1, 1))

    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()
    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()
    train_set_source_df_idx = torch.tensor(train_set_source_df_idx)
    test_set_source_df_idx = torch.tensor(test_set_source_df_idx)

    return (train_dataset, test_dataset,
            x_train, y_train, x_test, y_test, train_set_source_df_idx, test_set_source_df_idx, scaling_attributes)
