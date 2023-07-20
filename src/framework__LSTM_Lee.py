"""
***********************************************************************************************************************
    Import Libraries
***********************************************************************************************************************
"""
import os
import numpy as np
import random
import torch

import src.framework__data_set as framework__data_set


"""
***********************************************************************************************************************
    Helper Functions
***********************************************************************************************************************
"""


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


"""
***********************************************************************************************************************
    Data
***********************************************************************************************************************
"""
metric = 'container_cpu'
application_name = 'collector'

sub_sample_rate = 5
aggregation_type = 'max'
data_length_limit_in_minutes = 60

look_back = 12

dataset = framework__data_set.get_data_set(
    metric=metric,
    application_name=application_name,
    path_to_data='../data/'
)


print(f"Throwing out data that is less than {data_length_limit_in_minutes / 60} hours long.")
dataset.filter_data_that_is_too_short(data_length_limit=data_length_limit_in_minutes)
print(f"Subsampling data from 1 sample per 1 minute to 1 sample per {sub_sample_rate} minutes.")
dataset.sub_sample_data(sub_sample_rate=sub_sample_rate, aggregation_type=aggregation_type)
print("Splitting data into train and test.")
train_dataset, test_dataset = dataset.split_to_train_and_test_Lee()
print(f"Amount of dataframes in train data is {len(train_dataset)}, "
      f"with {sum(len(df) for df in train_dataset)} time samples.")
print(f"Amount of dataframes in test data is {len(test_dataset)}, "
      f"with {sum(len(df) for df in test_dataset)} time samples.")
print("Scaling data.")
# TODO: determine the common technique to scale the data
train_dataset.scale_data()
test_dataset.scale_data()

train_dataset.prepare_dataset_for_lstm(look_back=look_back)
test_dataset.prepare_dataset_for_lstm(look_back=look_back)

"""
***********************************************************************************************************************
    Model
***********************************************************************************************************************
"""


"""
***********************************************************************************************************************
    Train and Test
***********************************************************************************************************************
"""


"""
***********************************************************************************************************************
    Main
***********************************************************************************************************************
"""
if __name__ == '__main__':
    seed_everything()
    print("Done!")
    os._exit(0)
