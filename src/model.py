import torch
import torch.nn as nn
import torch.nn.functional as F
from src.environment import ObservationShape


class QNetwork(nn.Module):
    def __init__(self, observation_shape: ObservationShape, n_actions):
        super(QNetwork, self).__init__()
        self.observation_shape = observation_shape
        self.hidden_size = observation_shape.window_size * 4
        self.linear_balances1 = nn.Linear(observation_shape.n_balances, 16)
        self.linear_balances2 = nn.Linear(self.linear_balances1.out_features, 32)
        self.lstm = nn.LSTM(input_size=observation_shape.window_size,
                            hidden_size=self.hidden_size,
                            num_layers=2,
                            batch_first=True, dropout=0.2)
        concat_in_size = self.observation_shape.n_window_features + self.linear_balances2.out_features
        self.linear_concat = nn.Linear(concat_in_size, concat_in_size // 2)
        self.linear1 = nn.Linear(self.linear_concat.out_features, self.linear_concat.out_features // 2)
        self.output = nn.Linear(self.linear1.out_features, n_actions)

    def forward(self, observation: torch.tensor):
        observation = observation.to(torch.float32)
        window_obs = observation[..., :-self.observation_shape.n_balances]
        if observation.size(0) > 1:
            window_obs = window_obs.view(-1, self.observation_shape.n_window_features,
                                         self.observation_shape.window_size)
        else:
            window_obs = window_obs.view(-1, self.observation_shape.window_size)
        balances_obs = observation[..., -self.observation_shape.n_balances:]
        balances_obs = balances_obs.view(balances_obs.size(0), -1)
        balances_x = F.relu(self.linear_balances1(balances_obs))
        balances_x = F.relu(self.linear_balances2(balances_x))
        window_x, _ = self.lstm(window_obs)

        if observation.size(0) > 1:
            window_x = window_x[:, :, -1]
            window_x = window_x.view(window_x.size(0), -1)
        else:
            window_x = window_x[:, -1]
            window_x = window_x.view(-1, window_x.size(0))
        x = torch.cat((window_x, balances_x.view(balances_x.size(0), -1)), dim=1)
        x = F.relu(self.linear_concat(x))
        x = F.relu(self.linear1(x))
        return self.output(x)

    def init_recurrent_cell_states(self) -> tuple:
        h0 = torch.zeros(self.lstm.num_layers, self.observation_shape.n_window_features, self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.lstm.num_layers, self.observation_shape.n_window_features, self.hidden_size).requires_grad_()

        return h0, c0
