import src.framework__data_set as framework__data_set
import src.utils as utils
import src.config as config
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

    data.prepare_data_for_run()

    data_set = data.get_time_series_data()

    data.split_to_train_and_test_Lee()
    data.transform_and_scale_data()

    model = models.get_model(model_name=config.model_name)
    model.log_every_n_steps = 1  # handles a warning. TODO: check this solution

    train_and_validate.train_and_validate(model, data)
