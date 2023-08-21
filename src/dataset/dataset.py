from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    Dataset class for time series data
    """
    def __init__(self, x, y, df_idx, num_samples):
        self.x = x
        self.y = y
        self.df_idx = df_idx
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.x[index], self.y[index]
