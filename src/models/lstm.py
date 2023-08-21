import sys

import torch.nn as nn
import torch


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, horizon, bidirectional=False, dropout_rate=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True, bidirectional=bidirectional)

        # Additional bidirectional LSTM layer
        self.num_directions = 2 if bidirectional else 1
        self.lstm2 = nn.LSTM(hidden_size * self.num_directions, hidden_size,
                             num_stacked_layers, batch_first=True, bidirectional=bidirectional)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc1 = nn.Linear(hidden_size * self.num_directions, 64)
        self.fc2 = nn.Linear(64, horizon)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers * self.num_directions, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_stacked_layers * self.num_directions, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out, _ = self.lstm2(out, (h0, c0))

        out = self.dropout(out)

        out = self.fc1(out[:, -1, :])
        out = self.dropout(out)
        out = self.fc2(out)
        out = out.unsqueeze(-1)
        return out


if __name__ == '__main__':
    model = LSTM(1, 4, 1, 2, bidirectional=True)
    print(model)
    x = torch.randn(16, 12, 1)
    y = model(x)
    print(y.shape)
    sys.exit()
