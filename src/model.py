import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(QNetwork, self).__init__()
        self.n_observations = n_observations
        self.hidden_size = n_observations * 3
        self.linear1 = nn.Linear(n_observations, n_observations*2)
        self.lstm = nn.LSTM(input_size=n_observations*2, hidden_size=self.hidden_size, num_layers=1)
        self.linear2 = nn.Linear(self.hidden_size, n_observations*3 + n_observations // 2)
        self.linear3 = nn.Linear(self.linear2.out_features, self.linear2.out_features * 2)
        self.dropout1 = nn.Dropout(p=0.2)
        self.linear4 = nn.Linear(self.linear3.out_features, self.linear3.out_features * 3)
        self.dropout2 = nn.Dropout(p=0.2)
        self.linear5 = nn.Linear(self.linear4.out_features, self.linear4.out_features // 2)
        self.linear6 = nn.Linear(self.linear5.out_features, self.linear5.out_features // 2)
        self.dropout3 = nn.Dropout(p=0.1)
        self.linear7 = nn.Linear(self.linear6.out_features, self.linear6.out_features // 2)
        self.linear8 = nn.Linear(self.linear7.out_features, self.linear7.out_features // 2)
        self.output = nn.Linear(self.linear8.out_features, n_actions)

    def forward(self, observation: torch.tensor, recurrent_cell: torch.tensor, sequence_length: int = 1):
        h = observation
        h = F.relu(self.linear1(h))
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
        h = self.dropout1(h)
        h = F.relu(self.linear4(h))
        h = self.dropout2(h)
        h = F.relu(self.linear5(h))
        h = F.relu(self.linear6(h))
        h = self.dropout3(h)
        h = F.relu(self.linear7(h))
        h = F.relu(self.linear8(h))
        return self.output(h), recurrent_cell

    def init_recurrent_cell_states(self, num_sequences: int, device: torch.device) -> tuple:
        hxs = torch.zeros(num_sequences, self.hidden_size,
                          dtype=torch.float32, device=device).unsqueeze(0)
        cxs = torch.zeros(num_sequences, self.hidden_size,
                          dtype=torch.float32, device=device).unsqueeze(0)

        return hxs, cxs
