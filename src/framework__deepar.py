import matplotlib.pyplot as plt
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import torch
from pytorch_forecasting import DeepAR
from pytorch_forecasting.metrics import NormalDistributionLoss


class DeepARTester:
    def __init__(self, training, learning_rate=0.1, hidden_size=30, rnn_layers=2):
        # prepare parameters
        self.__msg = "[DeepARTester]"
        self.__model = DeepAR.from_dataset(
            training,
            learning_rate=learning_rate,
            log_interval=10,
            log_val_interval=1,
            hidden_size=hidden_size,
            rnn_layers=rnn_layers,
            loss=NormalDistributionLoss(),
        )
        self.__best_model = None

    """
    *******************************************************************************************************************
        API functions
    *******************************************************************************************************************
    """

    def learn_from_data_set(self, train_dataloader, validation_dataloader, max_epochs=5):
        early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=1e-4, patience=10, verbose=False,
                                            mode="min")
        trainer = pl.Trainer(
            accelerator="cpu",
            max_epochs=max_epochs,
            # gpus=0,
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback],
            limit_train_batches=50,
            enable_checkpointing=True,
        )
        trainer.fit(
            self.__model,
            train_dataloaders=train_dataloader,
            val_dataloaders=validation_dataloader
        )
        best_model_path = trainer.checkpoint_callback.best_model_path
        best_model = DeepAR.load_from_checkpoint(best_model_path)
        self.__best_model = best_model
        return best_model

    def get_actuals(self, test_dataloader, mean, std):
        actuals = torch.add(torch.mul(torch.cat([y[0] for x, y in iter(test_dataloader)]), std), mean)
        return actuals

    def predict(self, data):
        pred = self.__best_model.predict(data)
        return pred

    def predictions(self, test_dataloader):
        trainer_kwargs = {'accelerator': 'cpu'}
        predictions = self.__best_model.predict(test_dataloader, mode="samples", return_x=True, n_samples=100,
                                                trainer_kwargs=trainer_kwargs)

        raw_predictions = predictions.output
        x = predictions.x
        return raw_predictions, x

    def plot_long_pred(self, i, actuals, predictions):

        time_axis = np.arange(len(actuals))
        # Compute mean and standard deviation of the predictions
        predictions = np.sort(predictions)
        mu = predictions.mean(axis=1)
        sigma = predictions.std(axis=1)
        raw_size = predictions.shape[1]

        # Plot the mean curve
        plt.plot(actuals, label="actuals", color='royalblue')
        plt.plot(time_axis, mu, label="preds", color='darkorange')

        # Set the confidence level (e.g., 95%)
        conf_level_and_corresponding_z = {"95%": 1.96, "70%": 1.036, "40%": 0.524, "10%": 0.125}
        org_alpha = 0.5 / (len(conf_level_and_corresponding_z) + 1)
        alpha = org_alpha
        for cof, z in conf_level_and_corresponding_z.items():
            # Compute the confidence interval
            upper = mu + z * sigma / np.sqrt(raw_size)
            lower = mu - z * sigma / np.sqrt(raw_size)
            # Shade the area between the upper and lower confidence bounds
            plt.fill_between(time_axis, upper, lower, color='orange', alpha=alpha, label=cof + ' Confidence Interval')
            alpha += org_alpha

        # Add axis labels and title
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title("TimeSeries " + str(i))

        # plt.plot(preds,label = "preds")
        plt.legend()
        plt.show()

    def preduce_long_pred(self, data_sets, max_prediction_length, std, mean):
        encoder_length = max_prediction_length * 4
        for i, data in data_sets.items():
            actuals = list(data['value'])[encoder_length:]
            actuals = [x * std + mean for x in actuals]
            preds = []
            for i in range(0, data.shape[0] - encoder_length):
                # stride = i*max_prediction_length
                # test_dataloader, mode="raw", return_x=True, n_samples=100
                trainer_kwargs = {'accelerator': 'cpu'}
                predictions = self.__best_model.predict(data.iloc[i:i + encoder_length, :], mode="samples",
                                                        return_x=True, n_samples=100,
                                                        trainer_kwargs=trainer_kwargs)
                raw_prediction = predictions.output
                x = predictions.x
                predictions = torch.add(torch.mul(raw_prediction[0, 0, :], std), mean).tolist()
                preds += [predictions]
            self.plot_long_pred(i, actuals, np.array(preds))

    def predict_unknown(self, dataset):
        length = dataset.shape[0]
        predicted = list(self.__best_model.predict(dataset)[0])
        actual = list(dataset["value"])
        print("#$@" * 30)
        plt.plot(actual, label='Actual')

        # Plot the predicted values
        plt.plot(predicted, label='Predicted')
        # Add a legend
        plt.legend()
        # Show the plot
        plt.show()

        return predicted

    def plot_predictions(self, raw_predictions, x, validation):
        device = validation.x_to_index(x)["device"]
        for idx in range(20):  # plot 20 examples
            self.__best_model.plot_prediction(x, raw_predictions, idx=idx)
            plt.suptitle(f"device: {device.iloc[idx]}")
            plt.show()
