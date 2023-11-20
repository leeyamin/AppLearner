import argparse

parser = argparse.ArgumentParser()

# TODO: configure in model cuda device
parser.add_argument('--cuda_available', type=int, default=0, help='CUDA device index')

parser.add_argument('--metric', type=str, default='container_cpu', help='metric type')
parser.add_argument('--application_name', type=str, default='collector', help='application name')

parser.add_argument('--sub_sample_rate', type=int, default=1, help='sub-sample rate')
parser.add_argument('--aggregation_type', type=str, default='max', help='aggregation type')
parser.add_argument('--data_length_limit_in_minutes', type=int, default=60,
                    help='data length limit in minutes')
parser.add_argument('--train_ratio', type=float, default=0.8, help='training data ratio')
parser.add_argument('--transformation_method', type=str, default='log', help='transformation method')
parser.add_argument('--scale_method', type=str, default='min-max', help='scaling method')

parser.add_argument('--look_back', type=int, default=12, help='look back window size')
parser.add_argument('--horizon', type=int, default=2, help='prediction horizon')
parser.add_argument('--model_name', type=str, default='TCN', help='model name')
parser.add_argument('--trained_model_path', type=str, default=None, help='path to a pre-trained model')

parser.add_argument('--num_epochs', type=int, default=2, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--loss_function', type=str, default='L1Loss', help='loss function type')

parser.add_argument('--evaluation_metrics', default=["mae", "mape", "mse", "rmse"],
                    help='evaluation metrics')

parser.add_argument('--output_path', type=str, default=None, help='output path')

config = parser.parse_args()
