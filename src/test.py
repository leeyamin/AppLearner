import os
from tabulate import tabulate

import src.framework__data_set as framework__data_set
import src.utils as utils
from train_and_validate import train_or_validate_one_epoch, format_number

import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="MPS available but not used.")

model_dir_name = 'DeepAR_0'
metric = 'container_cpu'
application_name = 'collector'

if __name__ == '__main__':
    assert os.path.exists(os.path.join(os.path.dirname(os.getcwd()), 'output', model_dir_name, 'config.yml')), \
        f'No config file found in {os.path.join(os.path.dirname(os.getcwd()), "output", model_dir_name)}'
    config_path = os.path.join(os.path.dirname(os.getcwd()), 'output', model_dir_name, 'config.yml')
    config = utils.get_config(config_path)
    config.metric = metric
    config.application_name = application_name
    config.output_path = None

    main_dir = os.path.dirname(os.getcwd())
    work_dir = os.path.join(main_dir, 'output', model_dir_name)
    model = utils.load_model(model_name=config.model_name, work_dir=work_dir)

    data = framework__data_set.get_data_set(
        metric=config.metric,
        application_name=config.application_name,
        path_to_data='../data/'
    )

    data.set_configurations(config)
    data.prepare_data_for_run(config.output_path, record_logs_to_txt=True)
    data.split_to_train_and_test()
    data.transform_and_scale_data()

    utils.disable_pytorch_lightning_logging()

    test_metrics_dict, _ = train_or_validate_one_epoch(epoch_idx=None, model=model, data=data,
                                                       look_back=config.look_back,
                                                       output_path=None, mode='test', show_plots_flag=True, limit=1)

    results_data = [
        ["Test",
         format_number(test_metrics_dict["mae"]),
         format_number(test_metrics_dict["mape"]),
         format_number(test_metrics_dict["mse"]),
         format_number(test_metrics_dict["rmse"])]
    ]
    headers = ["Data", "avg MAE", "avg MAPE", "avg MSE", "avg RMSE"]

    print(tabulate(results_data, headers=headers, tablefmt="pretty"))
