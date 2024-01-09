import time

import src.framework__data_set as framework__data_set
import src.utils as utils
import src.train_and_validate as train_and_validate

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="MPS available but not used.")


if __name__ == '__main__':
    start_time = time.time()
    config = utils.get_config()
    config.output_path = utils.get_output_path(config.run_name, config.model_name)
    utils.save_config_to_file(config.output_path, config)

    data = framework__data_set.get_data_set(
        metric=config.metric,
        application_name=config.application_name,
        path_to_data='../data/'
    )

    data.set_configurations(config)
    data.prepare_data_for_run(config.output_path, record_logs_to_txt=True)
    data.split_to_train_and_test()
    data.transform_and_scale_data()

    model = utils.get_model(config.model_name, config.look_back, config.horizon, config.gpu_idx,
                            config.output_path, config.trained_model_path)
    model.log_every_n_steps = 1  # handles a warning
    model = train_and_validate.train_and_validate(model, data, config)
    print('Run time: ', time.time() - start_time)
    print('Done.')
