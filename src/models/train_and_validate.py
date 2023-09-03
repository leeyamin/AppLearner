import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm

import src.dataset.dataset as dataset
import src.dataset.utils as dataset_utils
import src.config as config


def calculate_mse(y_true, y_pred):
    """
    Mean Squared Error
    """
    return np.mean((y_true - y_pred) ** 2)


def calculate_mae(y_true, y_pred):
    """
    Mean Absolute Error
    """
    return np.mean(np.abs(y_true - y_pred))


def calculate_rmse(y_true, y_pred):
    """
    Root Mean Squared Error
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def calculate_smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error
    """
    return 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


def calculate_r_squared(y_true, y_pred):
    y_true_mean = np.mean(y_true)
    total_sum_of_squares = np.sum((y_true - y_true_mean) ** 2)
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
    return 1 - (residual_sum_of_squares / total_sum_of_squares)


def calculate_metrics(y_true, y_pred):
    mse = calculate_mse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    smape = calculate_smape(y_true, y_pred)
    r_squared = calculate_r_squared(y_true, y_pred)
    return mse, mae, rmse, mape, smape, r_squared


def train_df_epoch(train_loader, model, loss_function, optimizer, device):
    model.train()
    df_train_epoch_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        df_train_epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return df_train_epoch_loss


def validate_df_epoch(test_loader, model, loss_function, device):
    model.eval()
    df_test_epoch_loss = 0.0

    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader):
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            output = model(x_batch)
            loss = loss_function(output, y_batch)
            df_test_epoch_loss += loss.item()

    return df_test_epoch_loss


def train_or_validate_one_epoch(epoch, x_data, y_data, data_set_source_df_idx, model, loss_function, optimizer,
                                batch_size, device, train_flag):
    epoch_loss = 0.0
    str_train_or_val = 'Training' if train_flag else 'Validating'
    progress = tqdm(data_set_source_df_idx.unique(), desc=f'Epoch {epoch + 1} - {str_train_or_val}', leave=True)
    for df_index in progress:
        subset_source_df_index = torch.where(data_set_source_df_idx == df_index)[0]
        subset_df_dataset = Subset(
            dataset.TimeSeriesDataset(x_data, y_data, data_set_source_df_idx,
                                      num_samples=len(subset_source_df_index)),
            indices=subset_source_df_index
        )
        dataloader_subset = DataLoader(subset_df_dataset, batch_size=batch_size, shuffle=False)
        if train_flag:
            df_epoch_loss = train_df_epoch(dataloader_subset, model, loss_function, optimizer, device)
        else:
            df_epoch_loss = validate_df_epoch(dataloader_subset, model, loss_function, device)
        epoch_loss += df_epoch_loss
    return epoch_loss


def plot_convergence(train_losses, val_losses, num_epochs):
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Convergence')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, num_epochs + 1))
    plt.show()
    plt.close()


def test_df(data_set_source_df_idx, df_index, x_dataset, y_dataset, model, device):
    train_subset_source_df_index = torch.where(data_set_source_df_idx == df_index)[0]
    subset_df_dataset = Subset(dataset.TimeSeriesDataset(x_dataset, y_dataset, data_set_source_df_idx,
                                                         num_samples=len(train_subset_source_df_index)),
                               indices=train_subset_source_df_index)
    batch_size = 32
    dataloader_subset = DataLoader(subset_df_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    df_indices = torch.where(data_set_source_df_idx == df_index)[0]
    y_df = y_dataset[df_indices]
    predicted_df = torch.Tensor()
    for batch in dataloader_subset:
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        with torch.no_grad():
            predicted = model(x_batch)

        predicted_df = torch.cat([predicted_df, predicted.squeeze()], dim=0)

    return y_df, predicted_df


def plot_df_predictions(rescaled_y_df, rescaled_predicted_df, df_index, modality):
    """
    Plots the predictions for a single dataframe, based on the modality (train or validation).
    """
    max_value = max(rescaled_y_df.max(), rescaled_predicted_df.max())
    if rescaled_predicted_df.ndim == 1:
        rescaled_y_df = rescaled_y_df.reshape((-1, 1))
        rescaled_predicted_df = rescaled_predicted_df.reshape((-1, 1))
    num_time_steps = rescaled_predicted_df.shape[1]

    if num_time_steps == 1:
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))
        axs.plot(rescaled_y_df[:, 0], label=f'Actual', linestyle='solid')
        axs.plot(rescaled_predicted_df[:, 0], label=f'Predicted', linestyle='dashed')

        mae_loss = calculate_mae(rescaled_y_df[:, 0], rescaled_predicted_df[:, 0])
        rmse_loss = calculate_rmse(rescaled_y_df[:, 0], rescaled_predicted_df[:, 0])

        axs.set_ylabel('Value')
        axs.set_ylim(0, max_value * 1.1)
        axs.set_title(
            f'{modality} DF: {df_index}; MAE: {mae_loss:.3f}; RMSE: {rmse_loss:.3f}')
        axs.legend()
        axs.set_xlabel('Time')

    else:
        fig, axs = plt.subplots(num_time_steps, 1, figsize=(8, 6 * num_time_steps))
        for i in range(num_time_steps):
            axs[i].plot(rescaled_y_df[:, i], label=f'Actual Time Step {i + 1}', linestyle='solid')
            axs[i].plot(rescaled_predicted_df[:, i], label=f'Predicted Time Step {i + 1}', linestyle='dashed')

            mae_loss = calculate_mae(rescaled_y_df[:, i], rescaled_predicted_df[:, i])
            rmse_loss = calculate_rmse(rescaled_y_df[:, i], rescaled_predicted_df[:, i])

            axs[i].set_ylabel('Value')
            axs[i].set_ylim(0, max_value * 1.1)
            axs[i].set_title(
                f'{modality} DF: {df_index}; Time Step {i + 1}; MAE: {mae_loss:.3f}; RMSE: {rmse_loss:.3f}')
            axs[i].legend()

        axs[num_time_steps - 1].set_xlabel('Time')

    plt.tight_layout()
    plt.show()
    plt.close()


def test_and_plot(model, device, x_dataset, y_dataset, data_set_source_df_idx, scaling_attributes, modality):
    print(f'Testing the model on the {modality} set.')
    mse_total = 0.0
    mae_total = 0.0
    rmse_total = 0.0
    mape_total = 0.0
    for df_index in data_set_source_df_idx.unique():
        y_df, predicted_df = test_df(data_set_source_df_idx, df_index, x_dataset, y_dataset, model, device)

        rescaled_y_df = dataset_utils.re_transform_and_re_scale_data(y_df, config.transformation_method,
                                                                     config.scale_method, scaling_attributes)
        rescaled_predicted_df = dataset_utils.re_transform_and_re_scale_data(predicted_df, config.transformation_method,
                                                                             config.scale_method, scaling_attributes)

        rescaled_y_df = rescaled_y_df.squeeze().numpy()
        rescaled_predicted_df = rescaled_predicted_df.squeeze().numpy()

        mse, mae, rmse, mape, smape, r_squared = calculate_metrics(rescaled_y_df, rescaled_predicted_df)
        plot_df_predictions(rescaled_y_df, rescaled_predicted_df, df_index, modality)
        print(f'DF: {df_index}; MSE: {mse:.3f}; MAE: {mae:.3f}; RMSE: {rmse:.3f}; MAPE: {mape:.3f}; ')
        mse_total += mse
        mae_total += mae
        rmse_total += rmse
        mape_total += mape
    print(f'Total MSE: {mse_total:.3f}; Total MAE: {mae_total:.3f}; Total RMSE: {rmse_total:.3f}; ')


def train_and_validate(model, device,
                       x_train, y_train, x_test, y_test, train_set_source_df_idx, test_set_source_df_idx,
                       batch_size, num_epochs, learning_rate, loss_function, scaling_attributes):
    print("Training and testing the model.")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        train_epoch_loss = train_or_validate_one_epoch(epoch, x_train, y_train, train_set_source_df_idx,
                                                       model, loss_function, optimizer, batch_size, device,
                                                       train_flag=True)
        avg_train_epoch_loss = train_epoch_loss / len(train_set_source_df_idx)
        train_losses.append(avg_train_epoch_loss)

        test_epoch_loss = train_or_validate_one_epoch(epoch, x_test, y_test, test_set_source_df_idx,
                                                      model, loss_function, optimizer, batch_size, device,
                                                      train_flag=False)
        avg_test_epoch_loss = test_epoch_loss / len(test_set_source_df_idx)
        val_losses.append(avg_test_epoch_loss)

        tqdm.write(f'\nEpoch {epoch + 1} - Average Train Loss: {avg_train_epoch_loss:.3f}')
        tqdm.write(f'Epoch {epoch + 1} - Average Validation Loss: {avg_test_epoch_loss:.3f}')
        tqdm.write('----------------------------------------------')

    plot_convergence(train_losses, val_losses, num_epochs)
    test_and_plot(model, device, x_train, y_train, train_set_source_df_idx, scaling_attributes, modality='Train')
    test_and_plot(model, device, x_test, y_test, test_set_source_df_idx, scaling_attributes, modality='Validation')
