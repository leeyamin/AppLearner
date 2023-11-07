import torch
import torch.nn as nn

cuda_available = 0
device = torch.device(f"cuda:{cuda_available}" if torch.cuda.is_available() else "cpu")

metric = 'container_cpu'
application_name = 'collector'

sub_sample_rate = 1
aggregation_type = 'max'
data_length_limit_in_minutes = 60
train_ratio = 0.8

look_back = 12
horizon = 2

model_name = 'DeepAR'
hidden_size = 4
num_stacked_layers = 1
learning_rate = 0.0001
num_epochs = 4
# TODO: determine the loss function
loss_function = nn.L1Loss(reduction='sum')
batch_size = 16
evaluation_metrics = ["mae", "mape", "mse", "rmse"]

transformation_method = 'log'
scale_method = 'min-max'
