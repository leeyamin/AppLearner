import json
import matplotlib.pyplot as plt
import pandas as pd
import random
from datetime import datetime, timedelta
from os import listdir
from os.path import isfile, join
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler

import src.config as config


class TimeSeriesDataSet:
    def __init__(self, list_of_df):
        self.list_of_df = list_of_df
        self.__is_data_scaled = False
        self.__mean = None
        self.__std = None
        self.dfs_concatenated = None
        self.time_series_data = None
        self.train_time_series_data = None
        self.val_time_series_data = None
        self.scaler = None

    def get_list(self):
        return self.list_of_df

    def __get_mean_and_std(self):
        """
        calculates mean and std of all samples
        @return: mean and std of all samples (type np_array)
        """
        np_array_list = []
        for df in self:
            np_array_list += [df["sample"].to_numpy()]
        flat_np_array = np.concatenate(np_array_list)
        return flat_np_array.mean(), flat_np_array.std()

    def __set_mean_and_std(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __getitem__(self, key):
        return self.list_of_df[key]

    def __len__(self):
        return len(self.list_of_df)

    def merge_df(self):
        """
        concat all the dataframes of the same application
        """
        # todo: fix the merging of same apps on different pods and namespaces
        self.merged_df = pd.concat(self.list_of_df)

    def sort_by_time(self):
        """
        concat all the dataframes of the same application
        and sort them by time
        """
        self.merge_df()
        self.merged_df = self.merged_df.sort_values(by="time")
        self.merged_df = self.merged_df.groupby(['time'], as_index=False).max().reset_index()

    def get_marged(self):
        return self.merged_df

    def sub_sample_data(self, sub_sample_rate, aggregation_type='max'):
        """
        creates sub sampling according to the rate (if for example rate = 5, then every 5 samples, the one with the
        maximal value is chosen to be in the data set).
        @param sub_sample_rate:
        """
        new_list_of_df = []

        assert sub_sample_rate > 0
        if sub_sample_rate == 1:
            return
        for df in self.list_of_df:
            if aggregation_type == 'max':
                sub_sampled_data = df.groupby(df.index // sub_sample_rate).max()
            elif aggregation_type == 'min':
                sub_sampled_data = df.groupby(df.index // sub_sample_rate).min()
            elif aggregation_type == 'avg':
                sub_sampled_data = df.groupby(df.index // sub_sample_rate).mean()
            else:
                raise ValueError(f"aggregation_type = {aggregation_type} is not supported")

            assert len(sub_sampled_data) == ((len(df) + sub_sample_rate - 1) // sub_sample_rate)
            new_list_of_df.append(sub_sampled_data)

        self.list_of_df = new_list_of_df

    def deepAR_sub_sample_data(self, sub_sample_rate, agg="max"):
        """
        creates sub sampling according to the rate (if for example rate = 5, then every 5 samples, the one with the
        maximal value is chosen to be in the data set).
        @param sub_sample_rate:
        """
        new_list_of_df = []

        for df in self:
            if agg == "max":
                sub_sampled_data = df.groupby(df.index // sub_sample_rate).max()
            if agg == "min":
                sub_sampled_data = df.groupby(df.index // sub_sample_rate).min()
            if agg == "avg":
                sub_sampled_data = df.groupby(df.index // sub_sample_rate).mean()

            assert len(sub_sampled_data) == ((len(df) + sub_sample_rate - 1) // sub_sample_rate)
            new_list_of_df.append(sub_sampled_data)

        self.list_of_df = new_list_of_df

    def add_features(self):  # 2022-04-21 02:50:00   - example
        """
        Adding to the DataFrame "hour" and "day of week" columns for using those columns as features later
        """
        # todo: figure out if one-hot encoding can be good here
        new_list_of_df = []
        for df in self:
            df['hour'] = df['time'].apply(lambda x: int((str(x).split(' ')[1].split(':')[0])))
            df['day'] = df['time'].apply(lambda x: pd.Timestamp(str(x).split(' ')[0]).day_of_week)  # or dayofweek
        self.list_of_df = new_list_of_df

    def mean_sub_sample_data(self, sub_sample_rate):
        """
        creates sub sampling according to the rate (if for example rate = 5, then every 5 samples, the one with the
        mean value is chosen to be in the data set).
        @param sub_sample_rate:
        """
        # todo: fix the bug where the "time" column is disappear
        new_list_of_df = []

        for df in self:
            sub_sampled_data = df.groupby(df.index // sub_sample_rate).mean()
            assert len(sub_sampled_data) == ((len(df) + sub_sample_rate - 1) // sub_sample_rate)
            new_list_of_df.append(sub_sampled_data)

        self.list_of_df = new_list_of_df

    def filter_data_that_is_too_short(self, data_length_limit):
        """
        filters the data samples. all data samples that have a length that is lower than data_length_limit will be
        removed from the dataset
        @param data_length_limit: minimal length of sample
        """
        new_list_of_df = []

        for df in self:
            if len(df) > data_length_limit:
                new_list_of_df.append(df)

        self.list_of_df = new_list_of_df

    def filter_series_with_zeros(self):
        """
        filters the data samples with zeros.
        """
        new_list_of_df = []

        for df in self:
            # check if there is sample in the dataframe that contains some zero value
            if not df.isin([0]).any().any():
                new_list_of_df.append(df)

        self.list_of_df = new_list_of_df

    def filter_series_extreme_values(self, n):
        """
        filter the first and last element from every dataframe
        """
        new_list_of_df = []

        for df in self:
            print(len(df))
            new_list_of_df.append(df.iloc[n:-n])
            print(len(df.iloc[n:-n]))
            assert len(new_list_of_df[-1]) == len(df) - 2 * n

        self.list_of_df = new_list_of_df

    def plot_dataset(self, number_of_samples):
        """
        randomly selects samples from the data sets and plots . x-axis is time and y-axis is the value
        @param number_of_samples: number of randomly selected samples
        """
        samples = random.sample(self.list_of_df, k=number_of_samples)
        for df in samples:
            title = df.iloc[0, 2:5].str.cat(sep=', ')
            # plt.close("all")
            ts = df["sample"].copy()
            ts.index = [time for time in df["time"]]
            ts.plot()
            plt.title(title)
            plt.show()

    def scale_data(self):
        """
        rescaling the distribution of values so that the mean of observed values is 0, and the std is 1.
        each sample is standardized (value - mean / std)
        """
        assert not self.__is_data_scaled
        self.__is_data_scaled = True
        if self.__mean is None or self.__std is None:
            self.__mean, self.__std = self.__get_mean_and_std()
        # print(f"self.__mean = {self.__mean}, self.__std = {self.__std}", )
        # print("max_sample = ", max_sample, " min_sample = ", min_sample)
        for df in self:
            standardized_sample_column = (df["sample"] - self.__mean) / self.__std
            # print("sample", df["sample"] , standardized_sample_column)
            df["sample"] = standardized_sample_column

    def deepAR_scale_data(self):
        """
        rescaling the distribution of values so that the mean of observed values is 0, and the std is 1.
        each sample is standardized (value - mean / std)
        """
        assert not self.__is_data_scaled
        self.__is_data_scaled = True
        self.__mean, self.__std = self.__get_mean_and_std()
        # print(f"self.__mean = {self.__mean}, self.__std = {self.__std}", )
        # print("max_sample = ", max_sample, " min_sample = ", min_sample)
        for df in self:
            standardized_sample_column = (df["sample"] - self.__mean) / self.__std
            # print("sample", df["sample"] , standardized_sample_column)
            df["sample"] = standardized_sample_column
        return self.__mean, self.__std

    def split_to_train_and_test(self, length_to_predict):
        """
        according to an input, length to predict, we split the entire data set to train set and test set.
        The test set will be the same as the dataset in self. The train set will have the same amount of samples,
        but they will be shorter samples with their "tips" cut off.
        @param length_to_predict: The length to cut off from the train set.
        @return: train data set and test data set with sizes according to the input percentage.
        """
        assert 0 < length_to_predict < min([len(df) for df in self])
        assert isinstance(length_to_predict, int)
        random.shuffle(self.list_of_df)
        # copy info to test
        test = TimeSeriesDataSet(list_of_df=self.list_of_df)
        test.__is_data_scaled = self.__is_data_scaled
        test.__mean = self.__mean
        test.__std = self.__std
        # copy info to train
        train = TimeSeriesDataSet(list_of_df=[df[:-length_to_predict] for df in self.list_of_df])
        train.__is_data_scaled = self.__is_data_scaled
        train.__mean = self.__mean
        train.__std = self.__std
        assert min(len(df) for df in train) == (min(len(df) for df in test) - length_to_predict)
        assert max(len(df) for df in train) == (max(len(df) for df in test) - length_to_predict)
        return train, test

    def get_train_time_series_data(self):
        return self.train_time_series_data

    def get_val_time_series_data(self):
        return self.val_time_series_data

    def set_train_time_series_data_samples(self, train_time_series_data):
        self.train_time_series_data['sample'] = train_time_series_data.squeeze()

    def set_val_time_series_data_samples(self, val_time_series_data):
        self.val_time_series_data['sample'] = val_time_series_data.squeeze()

    def split_to_train_and_test_Lee(self):
        """
        Split the data into train and test sets according to a specified train ratio.
        The splitting is made upon the shuffled dataframes, and not the samples.
        """
        num_dfs = len(self.time_series_data['source_df_idx'].unique())
        num_dfs_train = int(num_dfs * config.train_ratio)

        # shuffle between dfs indices (not within dfs)
        unique_indices = self.time_series_data['source_df_idx'].unique()
        np.random.shuffle(unique_indices)
        train_indices = unique_indices[:num_dfs_train]
        val_indices = unique_indices[num_dfs_train:]

        train_data = self.time_series_data[self.time_series_data['source_df_idx'].isin(train_indices)].copy()
        val_data = self.time_series_data[self.time_series_data['source_df_idx'].isin(val_indices)].copy()

        self.train_time_series_data = train_data
        self.val_time_series_data = val_data

    def prepare_dataset_for_time_series(self, look_back, horizon):
        """
        Prepare the dataset for time series such that for each dataframe, each sample will have the samples
        based on the look_back parameter. The output of each dataframe will be a dataframe with the columns: time,
        sample and sample(t-1), sample(t-2), ..., sample(t-look_back), sample(t+1), sample(t+2), ..., sample(t+horizon)
        """
        for idx, df in enumerate(self.list_of_df):
            df = deepcopy(df[['time', 'sample']])
            df.set_index('time', inplace=True)

            for i in range(1, horizon):
                df[f'sample(t+{str(i)})'] = df['sample'].shift(-i)

            for i in range(1, look_back + 1):
                df[f'sample(t-{str(i)})'] = df['sample'].shift(i)

            df.dropna(inplace=True)
            self.list_of_df[idx] = df

    def record_df_indices(self):
        """
        Add a column to each dataframe that indicates the index of the dataframe in the list of dataframes.
        """
        for idx, df in enumerate(self.list_of_df):
            df['source_df_idx'] = idx

    def concatenate_dfs(self):
        self.dfs_concatenated = pd.concat(self.list_of_df)

    def recover_dfs_from_concatenated(self):
        split_dfs = []

        for idx in self.dfs_concatenated['source_df_idx'].unique():
            split_dfs.append(
                self.dfs_concatenated[self.dfs_concatenated['source_df_idx'] == idx].drop('source_df_idx', axis=1))
        for df in self.list_of_df:
            df.drop('source_df_idx', axis=1, inplace=True)

        for idx in range(0, len(self.list_of_df)):
            assert self.list_of_df[idx].equals(split_dfs[idx])

    def set_time_series_data(self):
        self.time_series_data = self.dfs_concatenated[['sample', 'time', 'source_df_idx']]

    def get_time_series_data(self):
        return self.time_series_data

    def prepare_data_for_run(self):
        import src.config as config
        print(f"Throwing out data that is less than {config.data_length_limit_in_minutes / 60} hours long.")
        self.filter_data_that_is_too_short(data_length_limit=config.data_length_limit_in_minutes)

        print(f"Subsampling data from 1 sample per 1 minute to 1 sample per {config.sub_sample_rate} minutes.")
        self.sub_sample_data(sub_sample_rate=config.sub_sample_rate, aggregation_type=config.aggregation_type)

        self.record_df_indices()
        self.concatenate_dfs()

        self.set_time_series_data()

    def transform_data(self, transformation_method='log'):
        """
        Transform the data using the specified method. Currently only supports log transformation.
        """
        if transformation_method == 'log':
            epsilon = 1e-8
            self.set_train_time_series_data_samples(np.log(self.train_time_series_data['sample'] + epsilon))
            self.set_val_time_series_data_samples(np.log(self.val_time_series_data['sample'] + epsilon))
        else:
            raise NotImplementedError

    def scale_data(self, scale_method='min-max'):
        """
        Scale the data using the specified method. Currently only supports min-max scaling.
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
            raise NotImplementedError

    def transform_and_scale_data(self, transformation_method: str = 'log', scale_method: str = 'min-max'):
        """
        Transform and scale the data using the specified methods. Currently only supports log transformation and min-max
        scaling.
        """
        if transformation_method is None and scale_method is None:
            return
        if transformation_method is not None:
            self.transform_data(transformation_method=transformation_method)
        if scale_method is not None:
            self.scale_data(scale_method=scale_method)

    def re_scale_data(self, darts_values, scale_method):
        """
        Rescale the data. Currently only supports min-max re-scaling.
        """
        if scale_method == 'min-max':
            re_scaled_darts_values = self.scaler.inverse_transform(darts_values)
        else:
            raise NotImplementedError
        return re_scaled_darts_values

    @staticmethod
    def re_transform_data(darts_values, transformation_method='log'):
        """
        Reverse transforms the data using the specified method. Currently only supports log transformation.
        """
        if transformation_method == 'log':
            re_transformed_darts_values = np.exp(darts_values)
        else:
            raise NotImplementedError
        return re_transformed_darts_values

    def re_transform_and_re_scale_data(self, darts_values, transformation_method, scale_method):
        """
        Reverse transforms and scales the data using the specified methods. Currently only supports log transformation and min-max
        scaling.
        """
        if transformation_method is None and scale_method is None:
            return
        if scale_method is not None:
            darts_values = self.re_scale_data(darts_values, scale_method)
        if transformation_method is not None:
            darts_values = self.re_transform_data(darts_values, transformation_method=transformation_method)
        return darts_values


"""
***********************************************************************************************************************
    get train and test datasets
***********************************************************************************************************************
"""


def __get_names_of_json_files_in_directory(directory_path):
    """
    returns the names of the json files in the directory (specified by the param "directory_path"
    @param directory_path: the name of the directory
    @return: the names of json files in directory
    """
    json_names = [f for f in listdir(directory_path) if (isfile(join(directory_path, f)) and ("json" in f))]
    return json_names


def __get_names_of_relevant_files(metric, path_to_data):
    """
    find the names of files that contain a specified metric in the directory.
    @param metric: specified metric to get : "container_cpu", "container_mem", "node_mem"
    @param path_to_data: the path to the directory
    @return: a list of the files that contain the specified from each json file in the directory specified
    """
    list_of_files = __get_names_of_json_files_in_directory(path_to_data)
    relevant_files = [file for file in list_of_files if (metric in file)]
    relevant_files.sort()
    return relevant_files


def __get_app_name_from_key(key: str):
    """
    @param key: column from the original data indicating name and other properties
    @return: the name of the app
    """
    app_name = key.split(", ")[0]
    namespace = key.split(", ")[1]
    node = key.split(", ")[2]
    pod = key.split(", ")[3]
    return [app_name, namespace, node, pod]


def __get_data_as_list_of_df_from_file(data_dict, application_name):
    """
    given data dictionary and an application name, appends all of the data that is associated with the application name
    to create a list and returns it
    @param data_dict: dictionary of data
    @param application_name:
    @return: time series of a specified application name from a data dictionary
    """
    result_list = []
    relevant_keys = [k for k in data_dict.keys() if (application_name == __get_app_name_from_key(key=k)[0])]
    # relevant_keys = [k for k in data_dict.keys() if (application_name == __get_app_name_from_key(key=k)[0] and 'collector-krg9f' == __get_app_name_from_key(key=k)[3]) ]
    for k in relevant_keys:
        list_of_ts = data_dict[k]
        for time_series in list_of_ts:
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
                # index=date_time_range
            )
            result_list.append(time_series_as_df)
    return result_list


def __get_data_as_list_of_df(metric, application_name, path_to_data):
    """
    @param metric: specified metric to get : "container_cpu", "container_mem", "node_mem"
    @param application_name: application name
    @param path_to_data: directory of json files
    @return: a list of the metric data of a specified app from all the json files in the directory (found by
    path_to_data)
    """
    file_names = __get_names_of_relevant_files(metric=metric, path_to_data=path_to_data)
    result_list = []
    for file_name in file_names:
        with open(f'{path_to_data}{file_name}') as json_file:
            data_dict = json.load(json_file)
            result_list += __get_data_as_list_of_df_from_file(
                data_dict=data_dict,
                application_name=application_name
            )

    return result_list


def get_data_set(metric, application_name, path_to_data):
    """
    @param metric: specified metric to get : "container_cpu", "container_mem", "node_mem"
    @param application_name: application name
    @param path_to_data: directory of json files
    @return: a TimeSeriesDataSet according to an app with a specified metric.
    """
    # constants
    __supported_metrics = ["container_cpu", "container_mem", "node_mem"]

    # checks if metric is one of the supported metrics
    assert metric in __supported_metrics

    list_of_df = __get_data_as_list_of_df(
        metric=metric,
        application_name=application_name,
        path_to_data=path_to_data
    )

    ds = TimeSeriesDataSet(list_of_df=list_of_df)
    return ds


def get_amount_of_data_per_application(metric, path_to_data):
    """
    @param metric: specified metric to get : "container_cpu", "container_mem", "node_mem"
    @param path_to_data: directory of json files
    @return: a sorted list of amount of data per each app that has a specific metric in a specified file
    """
    __supported_metrics = ["container_cpu", "container_mem", "node_mem"]
    assert metric in __supported_metrics
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
    result = sorted(application_names_histogram.items(), key=lambda item: - item[1])
    return result


"""
***********************************************************************************************************************
    main function
***********************************************************************************************************************
"""


def main():
    print("Start.")
    length_to_predict = 4
    test = 0
    if test == 0:
        print("Getting DataSet.")
        dataset = get_data_set(
            metric="container_mem",
            application_name="bridge-marker",
            path_to_data="../data/"
        )
        print("Plotting.")
        dataset.plot_dataset(number_of_samples=3)
        print("Subsampling.")
        dataset.sub_sample_data(sub_sample_rate=60)
        print("Plotting.")
        dataset.plot_dataset(number_of_samples=3)
        print("Normalizing.")
        dataset.scale_data()
        print("Plotting.")
        dataset.plot_dataset(number_of_samples=3)
        print("Filtering time series that are too short.")
        dataset.filter_data_that_is_too_short(data_length_limit=2 * length_to_predict)
        print("Splitting.")
        train, test = dataset.split_to_train_and_test(length_to_predict=length_to_predict)
        print("Plotting.")
        train.plot_dataset(number_of_samples=10)
        test.plot_dataset(number_of_samples=10)
    else:
        hist = get_amount_of_data_per_application(
            metric="container_mem",
            path_to_data="../data/"
        )
        print(hist)


"""
***********************************************************************************************************************
    run main function
***********************************************************************************************************************
"""

if __name__ == "__main__":
    main()
