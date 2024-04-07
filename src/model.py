import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(QNetwork, self).__init__()
        self.n_observations = n_observations
        self.hidden_size = 64
        self.lstm = nn.LSTM(input_size=n_observations, hidden_size=self.hidden_size, num_layers=1)
        self.linear2 = nn.Linear(self.hidden_size, 128)
        self.linear3 = nn.Linear(128, 264)
        self.dropout = nn.Dropout(p=0.2)
        self.linear4 = nn.Linear(264, 128)
        self.linear5 = nn.Linear(128, 64)
        self.output = nn.Linear(64, n_actions)

    def forward(self, observation: torch.tensor, recurrent_cell: torch.tensor, sequence_length: int = 1):
        h = observation
        if sequence_length == 1:
            h, recurrent_cell = self.lstm(h.unsqueeze(1), recurrent_cell)
            h = h.squeeze(1)
        else:
            h_shape = tuple(h.size())
            h = h.reshape((h_shape[0] // sequence_length), sequence_length, h_shape[1])
            h, recurrent_cell = self.lstm(h, recurrent_cell)
            h_shape = tuple(h.size())
            h = h.reshape(h_shape[0] * h_shape[1], h_shape[2])

        h = F.relu(self.linear2(h))
        h = F.relu(self.linear3(h))
        h = self.dropout(h)
        h = F.relu(self.linear4(h))
        h = F.relu(self.linear5(h))
        return self.output(h), recurrent_cell

    def init_recurrent_cell_states(self, num_sequences: int, device: torch.device) -> tuple:
        hxs = torch.zeros(num_sequences, self.hidden_size,
                          dtype=torch.float32, device=device).unsqueeze(0)
        cxs = torch.zeros(num_sequences, self.hidden_size,
                          dtype=torch.float32, device=device).unsqueeze(0)

        return hxs, cxs
