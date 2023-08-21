import sys

import src.framework__data_set as framework__data_set
import src.utils as utils
import src.config as config
import src.models.utils as models_utils
import src.train_and_validate as train_and_validate
import src.dataset.utils as dataset_utils

if __name__ == '__main__':
    utils.seed_everything()

    data = framework__data_set.get_data_set(
        metric=config.metric,
        application_name=config.application_name,
        path_to_data='../data/'
    )

    model = models_utils.get_model(config.model_name, config.hidden_size, config.num_stacked_layers, config.horizon)

    (train_dataset, test_dataset, x_train, y_train, x_test, y_test, train_set_source_df_idx, test_set_source_df_idx,
     scaler) = (dataset_utils.prepare_data_for_run(data))

    train_and_validate.train_and_validate(model, config.device,
                                          x_train, y_train, x_test, y_test, train_set_source_df_idx,
                                          test_set_source_df_idx,
                                          config.batch_size, config.num_epochs, config.learning_rate,
                                          config.loss_function, scaler)

    train_dataset.recover_dfs_from_concatenated()
    test_dataset.recover_dfs_from_concatenated()

    print("Done!")
    sys.exit()
