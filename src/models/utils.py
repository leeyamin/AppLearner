import src.models.lstm as lstm


def get_model(model_name, hidden_size, num_stacked_layers, horizon):
    if model_name == 'lstm':
        # TODO: ablation study on architecture and bidirectional
        return lstm.LSTM(1, hidden_size, num_stacked_layers, horizon)
    else:
        raise NotImplementedError
