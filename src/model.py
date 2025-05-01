import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from src.environment import ObservationShape

class QNetwork(nn.Module):
    def __init__(self, observation_shape: ObservationShape, n_actions):
        super(QNetwork, self).__init__()
        self.observation_shape = observation_shape

        self.hidden_size = observation_shape.window_size * 4

        self.lstm = nn.LSTM(
            input_size=observation_shape.n_window_features,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.4
        )

        # For the balance features (2 features in your case)
        self.linear_balances1 = nn.Linear(observation_shape.n_linear_features, 64)
        self.linear_balances2 = nn.Linear(64, 128)

        # For the concat part
        concat_in_size = self.hidden_size + self.linear_balances2.out_features
        self.linear_concat = nn.Linear(concat_in_size, n_actions * 8)
        self.linear1 = nn.Linear(self.linear_concat.out_features, self.linear_concat.out_features // 4)
        self.output = nn.Linear(self.linear1.out_features, n_actions)

        total_features = (
                observation_shape.n_window_features * observation_shape.window_size
                + observation_shape.n_linear_features
        )
        summary(self, (1, total_features))

    def forward(self, observation: torch.Tensor):
        observation = observation.to(torch.float32)
        batch_size = observation.size(0)

        # Split observation
        window_obs = observation[:, :-self.observation_shape.n_linear_features]
        balances_obs = observation[:, -self.observation_shape.n_linear_features:]

        # Reshape window_obs to (batch, n_window_features, window_size)
        window_obs = window_obs.view(batch_size, self.observation_shape.n_window_features, self.observation_shape.window_size)
        window_obs = window_obs.permute(0, 2, 1)

        # LSTM forward
        window_x, (h_n, c_n) = self.lstm(window_obs)

        # Get the final output from the last time step
        window_x = window_x[:, -1, :]  # (batch_size, hidden_size)

        # Process balances
        balances_x = F.relu(self.linear_balances1(balances_obs))
        balances_x = F.relu(self.linear_balances2(balances_x))

        # Concatenate
        x = torch.cat((window_x, balances_x), dim=1)

        # Final MLP
        x = F.relu(self.linear_concat(x))
        x = F.relu(self.linear1(x))
        return self.output(x)
