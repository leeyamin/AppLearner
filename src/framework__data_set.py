import json
import matplotlib.pyplot as plt
import pandas as pd
import random
from datetime import datetime, timedelta
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Dict

import src.utils as utils


class TimeSeriesDataSet:
    def __init__(self, list_of_df):
        self.list_of_df = list_of_df
        self.dfs_concatenated = None
        self.time_series_data = None
        self.train_time_series_data = None
        self.val_time_series_data = None
        self.scaler = None
        # configured parameters
        self.sub_sample_rate = None
        self.aggregation_type = None
        self.data_length_limit_in_minutes = None
        self.transformation_method = None
        self.scale_method = None
        self.train_ratio = None

    def get_list(self):
        return self.list_of_df

    def __getitem__(self, key):
        return self.list_of_df[key]

    def __len__(self):
        return len(self.list_of_df)

    def set_configurations(self, config):
        self.sub_sample_rate = config.sub_sample_rate
        self.aggregation_type = config.aggregation_type
        self.data_length_limit_in_minutes = config.data_length_limit_in_minutes
        self.transformation_method = config.transformation_method
        self.scale_method = config.scale_method
        self.train_ratio = config.train_ratio

    def sub_sample_data(self) -> None:
        """
        Subsample the data by a specified rate and aggregation type.
        """
        new_list_of_df = []

        assert self.sub_sample_rate > 0, f"sub_sample_rate = {self.sub_sample_rate} must be positive"
        assert self.aggregation_type in ['max', 'min', 'avg'], \
            f"aggregation_type = {self.aggregation_type} is not supported"
        if self.sub_sample_rate == 1:
            # no need to subsample
            return
        for df in self.list_of_df:
            if self.aggregation_type == 'max':
                sub_sampled_data = df.groupby(df.index // self.sub_sample_rate).max()
            elif self.aggregation_type == 'min':
                sub_sampled_data = df.groupby(df.index // self.sub_sample_rate).min()
            elif self.aggregation_type == 'avg':
                sub_sampled_data = df.groupby(df.index // self.sub_sample_rate).mean()

            assert len(sub_sampled_data) == ((len(df) + self.sub_sample_rate - 1) // self.sub_sample_rate)
            new_list_of_df.append(sub_sampled_data)

        self.list_of_df = new_list_of_df

    def filter_data_that_is_too_short(self) -> None:
        """
        Filter the data samples; remove all data samples of length lower than data_length_limit.
        @return None
        """
        new_list_of_df = []

        for df in self:
            if len(df) > self.data_length_limit_in_minutes:
                new_list_of_df.append(df)

        self.list_of_df = new_list_of_df

    def plot_dataset(self, number_of_samples: int = 3) -> None:
        """
        Randomly select samples from the data sets and plot; x-axis is time and y-axis is the value.
        @param number_of_samples: number of randomly selected samples
        @return None
        """
        samples = random.sample(self.list_of_df, k=number_of_samples)
        for df in samples:
            title = df.iloc[0, 2:5].str.cat(sep=', ')
            ts = df["sample"].copy()
            ts.index = [time for time in df["time"]]
            ts.plot()
            plt.title(title)
            plt.show()
            plt.close()

    def get_train_time_series_data(self):
        return self.train_time_series_data

    def get_val_time_series_data(self):
        return self.val_time_series_data

    def set_train_time_series_data_samples(self, train_time_series_data):
        self.train_time_series_data['sample'] = train_time_series_data.squeeze()

    def set_val_time_series_data_samples(self, val_time_series_data):
        self.val_time_series_data['sample'] = val_time_series_data.squeeze()

    def split_to_train_and_test(self):
        """Split the data into train and test sets according to a specified train ratio.
        The splitting is made upon the shuffled data frames, and not the samples."""
        num_dfs = len(self.time_series_data['source_df_idx'].unique())
        num_dfs_train = int(num_dfs * self.train_ratio)

        # shuffle between dfs indices (not within dfs)
        unique_indices = self.time_series_data['source_df_idx'].unique()
        np.random.seed(42)
        np.random.shuffle(unique_indices)
        train_indices = unique_indices[:num_dfs_train]
        val_indices = unique_indices[num_dfs_train:]

        train_data = self.time_series_data[self.time_series_data['source_df_idx'].isin(train_indices)].copy()
        val_data = self.time_series_data[self.time_series_data['source_df_idx'].isin(val_indices)].copy()

        self.train_time_series_data = train_data
        self.val_time_series_data = val_data

    def record_df_indices(self):
        """Add a column to each dataframe that indicates the index of the dataframe in the list of dataframes."""
        for idx, df in enumerate(self.list_of_df):
            df['source_df_idx'] = idx

    def concatenate_dfs(self):
        """Concatenate the data frames into one data frame."""
        self.dfs_concatenated = pd.concat(self.list_of_df)

    def set_time_series_data(self):
        """Set the time series data according to the concatenated data frames with the relevant columns."""
        self.time_series_data = self.dfs_concatenated[['sample', 'time', 'source_df_idx']]

    def prepare_data_for_run(self, output_path, record_logs_to_txt=True):
        """Prepare data for run; filter data that is too short,
        subsample the data, concatenate the data frames and set."""
        msg = f"Throwing out data that is less than {self.data_length_limit_in_minutes / 60} hours long."
        print(msg)
        utils.record_logs_to_txt(msg, output_path) if record_logs_to_txt else None
        self.filter_data_that_is_too_short()
        msg = f"Subsampling data from 1 sample per 1 minute to 1 sample per {self.sub_sample_rate} minutes."
        print(msg)
        utils.record_logs_to_txt(msg, output_path) if record_logs_to_txt else None
        self.sub_sample_data()

        self.record_df_indices()
        self.concatenate_dfs()

        self.set_time_series_data()

    def transform_data(self) -> None:
        """Transform the data using the specified method. Currently only supports log transformation."""
        if self.transformation_method == 'log':
            epsilon = 1e-8
            self.set_train_time_series_data_samples(np.log(self.train_time_series_data['sample'] + epsilon))
            self.set_val_time_series_data_samples(np.log(self.val_time_series_data['sample'] + epsilon))
        else:
            raise NotImplementedError(f"transformation_method = {self.transformation_method} is not supported")

    def scale_data(self, scale_method: str) -> None:
        """
        Scale the data using the specified method. Currently only supports min-max scaling.
        @param scale_method: the method of scaling to use. currently supports: min-max.
        @return None
        """
        if scale_method == 'min-max':
            if self.scaler is None:
                self.scaler = MinMaxScaler()
                self.scaler.fit(self.train_time_series_data['sample'].values.reshape(-1, 1))

            scaled_train_samples = self.scaler.transform(self.train_time_series_data['sample'].values.reshape(-1, 1))
            scaled_val_samples = self.scaler.transform(self.val_time_series_data['sample'].values.reshape(-1, 1))

            self.set_train_time_series_data_samples(scaled_train_samples)
            self.set_val_time_series_data_samples(scaled_val_samples)
        else:
            raise NotImplementedError(f"scale_method = {scale_method} is not supported")

    def transform_and_scale_data(self) -> None:
        """
        Transform and scale the data using the specified methods.
        """
        if self.transformation_method is None and self.scale_method is None:
            return
        if self.transformation_method is not None:
            self.transform_data(self.transformation_method)
        if self.scale_method is not None:
            self.scale_data(self.scale_method)

    def re_scale_data(self, darts_values: np.ndarray[np.float64]):
        """
        Rescale the data.
        @param darts_values: the values to rescale (in darts format)
        @return re_scaled_darts_values: rescaled data in darts format
        """
        if self.scale_method == 'min-max':
            re_scaled_darts_values = self.scaler.inverse_transform(darts_values)
        else:
            raise NotImplementedError(f"scale_method = {self.scale_method} is not supported")
        return re_scaled_darts_values

    def re_transform_data(self, darts_values: np.ndarray[np.float64]):
        """
        Reverse transforms the data using the specified method.
        @param darts_values: the values to rescale (in darts format)
        @return re_transformed_darts_values: rescaled data in darts format
        """
        if self.transformation_method == 'log':
            re_transformed_darts_values = np.exp(darts_values)
        else:
            raise NotImplementedError(f"transformation_method = {self.transformation_method} is not supported")
        return re_transformed_darts_values

    def re_transform_and_re_scale_data(self, darts_values: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        """
        Reverse transforms and scales the data using the specified methods. Currently only supports log transformation and min-max
        scaling.
        @param darts_values: the values to rescale (in darts format)
        @return re_scaled_darts_values: rescaled data in darts format
        """
        if self.transformation_method is None and self.scale_method is None:
            return darts_values
        if self.scale_method is not None:
            darts_values = self.re_scale_data(darts_values)
        if self.transformation_method is not None:
            darts_values = self.re_transform_data(darts_values)
        return darts_values


def __get_names_of_json_files_in_directory(directory_path: str) -> list[str]:
    """
    Get the names of the json files in the directory.
    @param directory_path: the path to the directory of the json files
    @return json_names: a list of the names of the json files in the directory
    """
    json_names = [f for f in listdir(directory_path) if (isfile(join(directory_path, f)) and ("json" in f))]
    return json_names


def __get_names_of_relevant_files(metric: str, path_to_data: str) -> list[str]:
    """
    Get the names of files that contain a specified metric in the directory.
    @param metric: specified data metric including: "container_cpu", "container_mem", "node_mem"
    @param path_to_data: directory of json files
    @return: a list of the files that contain the specified from each json file in the directory specified
    """
    list_of_files = __get_names_of_json_files_in_directory(path_to_data)
    relevant_files = [file for file in list_of_files if (metric in file)]
    relevant_files.sort()
    return relevant_files


def __get_app_name_from_key(key: str) -> list[str]:
    """
    Get the name of the app from the key of the data dictionary.
    @param key: column from the original data indicating name and other properties
    @return app_info: a list of the app name, namespace, node and pod
    """
    app_name = key.split(", ")[0]
    namespace = key.split(", ")[1]
    node = key.split(", ")[2]
    pod = key.split(", ")[3]

    app_info = [app_name, namespace, node, pod]
    return app_info


def __get_data_as_list_of_df_from_file(data_dict: Dict, application_name: str) -> list[pd.DataFrame]:
    """
    Get all the data associated with a specified application name from a specified file.
    @param data_dict: the data dictionary
    @param application_name: application name
    @return application_data_frames: a list of the data associated with the specified application name
    """
    application_data_frames = []
    relevant_keys = [k for k in data_dict.keys() if (application_name == __get_app_name_from_key(key=k)[0])]
    for k in relevant_keys:
        list_of_time_series = data_dict[k]
        for time_series in list_of_time_series:
            application_name, namespace, node, pod = __get_app_name_from_key(key=k)
            start_time = datetime.strptime(time_series["start"], "%Y-%m-%d %H:%M:%S")
            stop_time = datetime.strptime(time_series["stop"], "%Y-%m-%d %H:%M:%S")
            date_time_range = [start_time + timedelta(minutes=i) for i in range(len(time_series["data"]))]
            assert date_time_range[-1] == stop_time
            time_series_as_df = pd.DataFrame(
                {
                    "sample": time_series["data"],
                    "time": date_time_range,
                    "application_name": application_name,
                    "node": node,
                    "pod": pod,
                    "namespace": namespace
                },
            )
            application_data_frames.append(time_series_as_df)
    return application_data_frames


def __get_data_as_list_of_df(metric: str, application_name: str, path_to_data: str) -> list[pd.DataFrame]:
    """
    Get all the data associated with a specified application name from all the files in the directory.
    @param metric: specified data metric including: "container_cpu", "container_mem", "node_mem"
    @param application_name: application name
    @param path_to_data: directory of json files
    @return file_name: a list of all the data associated with the specified application name
    """
    file_names = __get_names_of_relevant_files(metric=metric, path_to_data=path_to_data)
    all_data_frames = []
    for file_name in file_names:
        with open(f'{path_to_data}{file_name}') as json_file:
            data_dict = json.load(json_file)
            application_data_frames = __get_data_as_list_of_df_from_file(
                data_dict=data_dict,
                application_name=application_name
            )
            all_data_frames.extend(application_data_frames)

    return all_data_frames


def get_data_set(metric: str, application_name: str, path_to_data: str) -> TimeSeriesDataSet:
    """
    Get a data set of a specified metric and application name.
    @param metric: specified metric including: "container_cpu", "container_mem", "node_mem"
    @param application_name: application name
    @param path_to_data: directory of json files
    @return ds: a TimeSeriesDataSet object

    """
    __supported_metrics = ["container_cpu", "container_mem", "node_mem"]
    assert metric in __supported_metrics, (f"Unsupported metric: {metric}. "
                                           f"Supported metrics are: {', '.join(__supported_metrics)}")

    list_of_df = __get_data_as_list_of_df(
        metric=metric,
        application_name=application_name,
        path_to_data=path_to_data
    )

    time_series_dataset = TimeSeriesDataSet(list_of_df=list_of_df)
    return time_series_dataset


def get_amount_of_data_per_application(metric: str, path_to_data: str) -> list[tuple[str, int]]:
    """
    Get the amount of data per application.
    @param metric: specified metric including: "container_cpu", "container_mem", "node_mem"
    @param path_to_data: directory of json files
    @return sorted_data_by_application: a list of tuples of the application name and the amount of data
    """
    __supported_metrics = ["container_cpu", "container_mem", "node_mem"]
    assert metric in __supported_metrics, (f"Unsupported metric: {metric}. "
                                           f"Supported metrics are: {', '.join(__supported_metrics)}")
    file_names = __get_names_of_relevant_files(metric=metric, path_to_data=path_to_data)
    application_names_histogram = {}
    for file_name in file_names:
        with open(f'{path_to_data}{file_name}') as json_file:
            data_dict = json.load(json_file)
            for k in data_dict.keys():
                app_name = __get_app_name_from_key(key=k)[0]
                # count number of time series samples
                amount_of_data = 0
                for ts in data_dict[k]:
                    amount_of_data += len(ts["data"])
                # add count to running count
                if app_name in application_names_histogram:
                    application_names_histogram[app_name] += amount_of_data
                else:
                    application_names_histogram[app_name] = amount_of_data
    sorted_data_by_application = sorted(application_names_histogram.items(), key=lambda item: - item[1])
    return sorted_data_by_application


def plot_top_ten_applications(hist: list[tuple[str, int]]) -> None:
    """
    Plot the top ten applications according to the amount of data.
    @param hist: a list of tuples of the application name and the amount of data
    @return None
    """
    hist = sorted(hist, key=lambda item: -item[1])
    hist = hist[:10]
    plt.figure(figsize=(10, 8))
    plt.bar(range(len(hist)), [val[1] for val in hist], align='center')
    plt.xticks(range(len(hist)), [val[0] for val in hist], rotation=45)
    plt.title("Top 10 applications according to the amount of data")
    plt.show()


if __name__ == "__main__":
    dataset = get_data_set(
        metric="container_mem",
        application_name="collector",
        path_to_data="../data/"
    )
    dataset.plot_dataset(number_of_samples=3)

    hist = get_amount_of_data_per_application(
        metric="container_mem",
        path_to_data="../data/"
    )
    print(f'Top 10 applications according to the amount of data:\n{hist}')
    plot_top_ten_applications(hist)
