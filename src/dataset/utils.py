import numpy as np
from copy import deepcopy
from typing import Optional, Tuple
from sklearn.preprocessing import MinMaxScaler

import src.config as config


def transform_data(samples, transformation_method='log'):
    """
    Transform the data using the specified method. Currently only supports log transformation.
    """
    if transformation_method == 'log':
        epsilon = 1e-8
        transformed_samples = np.log(samples + epsilon)
    else:
        raise NotImplementedError
    return transformed_samples


def scale_data(samples, scale_method='min-max', scaler=None):
    """
    Scale the data using the specified method. Currently only supports min-max scaling.
    """
    if scale_method == 'min-max':
        if scaler is None:
            scaler = MinMaxScaler()
            scaled_samples = scaler.fit_transform(samples)
        else:
            scaled_samples = scaler.transform(samples)
    else:
        raise NotImplementedError

    return scaled_samples, scaler


def transform_and_scale_data(samples,
                             transformation_method: str = 'log',
                             scale_method: str = 'min-max',
                             scaler: Optional[MinMaxScaler] = None):
    """
    Transform and scale the data using the specified methods. Currently only supports log transformation and min-max
    scaling.
    """
    if transformation_method is None and scale_method is None:
        return samples, None
    if transformation_method is not None:
        samples = transform_data(samples, transformation_method=transformation_method)
    if scale_method is not None:
        samples, scaler = scale_data(samples, scale_method=scale_method, scaler=scaler)
    return samples, scaler


def re_scale_data(darts_values, scale_method, scaler):
    """
    Rescale the data. Currently only supports min-max re-scaling.
    """
    if scale_method == 'min-max':
        darts_values = scaler.inverse_transform(darts_values)
    else:
        raise NotImplementedError
    return darts_values


def re_transform_data(darts_values, transformation_method='log'):
    """
    Reverse transforms the data using the specified method. Currently only supports log transformation.
    """
    if transformation_method == 'log':
        darts_values = np.exp(darts_values)
    else:
        raise NotImplementedError
    return darts_values


def re_transform_and_re_scale_data(darts_values, transformation_method, scale_method, scaler):
    """
    Reverse transforms and scales the data using the specified methods. Currently only supports log transformation and min-max
    scaling.
    """
    if transformation_method is None and scale_method is None:
        return darts_values
    if scale_method is not None:
        darts_values = re_scale_data(darts_values, scale_method, scaler)
    if transformation_method is not None:
        darts_values = re_transform_data(darts_values, transformation_method=transformation_method)
    return darts_values


def prepare_data_for_run(data):
    print(f"Throwing out data that is less than {config.data_length_limit_in_minutes / 60} hours long.")
    data.filter_data_that_is_too_short(data_length_limit=config.data_length_limit_in_minutes)

    print(f"Subsampling data from 1 sample per 1 minute to 1 sample per {config.sub_sample_rate} minutes.")
    data.sub_sample_data(sub_sample_rate=config.sub_sample_rate, aggregation_type=config.aggregation_type)

    data.record_df_indices()
    data.concatenate_dfs()

    # TODO: instead of deepcopy, save in the class
    data_set = deepcopy(data.dfs_concatenated)
    data_set = data_set[['sample', 'time', 'source_df_idx']]

    return data_set
