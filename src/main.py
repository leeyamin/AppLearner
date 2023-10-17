import numpy as np

import src.framework__data_set as framework__data_set
import src.utils as utils
import src.config as config
import src.dataset.utils as dataset_utils
import src.models as models
import src.train_and_validate as train_and_validate

import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="MPS available but not used.")

if __name__ == '__main__':
    utils.seed_everything()

    data = framework__data_set.get_data_set(
        metric=config.metric,
        application_name=config.application_name,
        path_to_data='../data/'
    )

    data_set = dataset_utils.prepare_data_for_run(data)

    # TODO: edit into prepare_data_for_run

    # split dfs into train and val
    num_dfs = len(data_set['source_df_idx'].unique())
    num_dfs_train = int(num_dfs * config.train_ratio)

    # shuffle between dfs indices (not within dfs)
    unique_indices = data_set['source_df_idx'].unique()
    np.random.shuffle(unique_indices)
    train_indices = unique_indices[:num_dfs_train]
    val_indices = unique_indices[num_dfs_train:]
    train_dfs = data_set[data_set['source_df_idx'].isin(train_indices)]
    val_dfs = data_set[data_set['source_df_idx'].isin(val_indices)]

    train_dfs['sample'], scaler = dataset_utils.transform_and_scale_data(
        samples=train_dfs[['sample']],
        transformation_method=config.transformation_method,
        scale_method=config.scale_method,
        scaler=None)

    val_dfs['sample'], _ = dataset_utils.transform_and_scale_data(
        samples=val_dfs[['sample']],
        transformation_method=config.transformation_method,
        scale_method=config.scale_method,
        scaler=scaler)

    model = models.get_model(model_name=config.model_name)
    model.log_every_n_steps = 1  # handles a warning. TODO: check this solution

    train_and_validate.train_and_validate(model, train_dfs, val_dfs, scaler)
