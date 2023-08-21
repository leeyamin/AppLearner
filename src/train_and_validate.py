import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm

import src.dataset.dataset as dataset
import src.dataset.utils as dataset_utils
import src.config as config


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
    plt.show()
    plt.close()


def plot_predictions(model, device, x_dataset, y_dataset, data_set_source_df_idx, scaler, modality):
    for df_index in data_set_source_df_idx.unique():
        train_subset_source_df_index = torch.where(data_set_source_df_idx == df_index)[0]
        subset_df_dataset = Subset(dataset.TimeSeriesDataset(x_dataset, y_dataset, data_set_source_df_idx,
                                                             num_samples=len(train_subset_source_df_index)),
                                   indices=train_subset_source_df_index)
        batch_size = 32
        dataloader_subset = DataLoader(subset_df_dataset, batch_size=batch_size, shuffle=False)

        model.eval()
        mae = nn.L1Loss(reduction='sum')
        rmse = nn.MSELoss(reduction='sum')
        mae_loss = 0.0
        rmse_loss = 0.0
        df_indices = torch.where(data_set_source_df_idx == df_index)[0]
        y_df = y_dataset[df_indices]
        predicted_df = torch.Tensor()
        for batch in dataloader_subset:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            with torch.no_grad():
                predicted = model(x_batch)

            mae_loss += mae(predicted, y_batch).item()
            rmse_loss += torch.sqrt(rmse(predicted, y_batch)).item()
            predicted_df = torch.cat([predicted_df, predicted.squeeze()], dim=0)

        mae_loss /= len(subset_df_dataset)
        rmse_loss /= len(subset_df_dataset)
        assert len(predicted_df) == len(subset_df_dataset) == len(y_df)

        rescaled_y_df = dataset_utils.re_format_y_to_df(y_df, config.look_back, config.horizon,
                                                        config.transformation_method,
                                                        config.scale_method, scaler)
        rescaled_predicted_df = dataset_utils.re_format_y_to_df(predicted_df, config.look_back, config.horizon,
                                                                config.transformation_method,
                                                                config.scale_method, scaler)

        plt.plot(rescaled_y_df[:, 0], label=f'Actual', linestyle='solid', color='tab:green')
        for i in range(rescaled_predicted_df.shape[1]):
            max_value = max(np.max(rescaled_predicted_df[:, i]), np.max(rescaled_y_df[:, i]))
            plt.plot(rescaled_predicted_df[:, i], label=f'Predicted Time Step {i + 1}', linestyle='dashed')

        plt.xlabel('Time')  # TODO: revert to actual time
        plt.ylabel('Value')
        plt.ylim(0, max_value * 1.1)
        plt.title(f'{modality} DF: {df_index}; MAE: {mae_loss:.3f}; RMSE: {rmse_loss:.3f}')
        plt.legend()
        plt.show()
        plt.close()


def train_and_validate(model, device,
                       x_train, y_train, x_test, y_test, train_set_source_df_idx, test_set_source_df_idx,
                       batch_size, num_epochs, learning_rate, loss_function, scaler):
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
    plot_predictions(model, device, x_train, y_train, train_set_source_df_idx, scaler, modality='Train')
    plot_predictions(model, device, x_test, y_test, test_set_source_df_idx, scaler, modality='Validation')
