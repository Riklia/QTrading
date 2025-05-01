import gym
import numpy as np
import pandas as pd
import torch
import plotly.graph_objs as go
import plotly.offline as pyo
from sklearn.preprocessing import MinMaxScaler
from collections import namedtuple
from src.environment.env_parameters import EnvParameters
from src.environment.action import RangeSpace, MainActionTypes


YEAR = 365
ObservationShape = namedtuple('ObservationShape', ('n_classes', 'n_window_features', 'window_size', 'n_linear_features'))


class CryptoTradingEnvironment(gym.Env):
    # If the environment will be developed to handle the multi currency case, the self.price attribute must be
    # modified to represent rates of different currencies in different time points, also some functions which
    # work with Balance must be modified. For example, _get_overall_current_balance must use corresponding prices
    def __init__(self, configs: EnvParameters, device, render_directory):
        super(CryptoTradingEnvironment, self).__init__()
        self.window = configs.window
        # date processing
        data = pd.read_csv(configs.data_path)
        data = data[(configs.start_time <= data["date"]) & (data["date"] <= configs.end_time)]
        data["date"] = pd.to_datetime(data["date"], unit="s")
        data.sort_values(by="date", inplace=True)
        self.dates = data["date"].reset_index(drop=True)
        self.date_unit_in_seconds = (self.dates.iloc[1] - self.dates.iloc[0]).total_seconds()
        self.max_time_point = len(self.dates) - 1
        # initial point is window size, not 0
        self.time_point = self.window
        # device for tensors
        self.device = device
        # features
        self.prices = torch.tensor(data["close_default"].reset_index(drop=True).values, dtype=torch.float32, device=device)
        self.volume_default = torch.tensor(data["volume_default"].reset_index(drop=True).values, dtype=torch.float32, device=device)
        self.spread = torch.tensor(data["spread"].reset_index(drop=True).values, dtype=torch.float32, device=device)
        self.funding = torch.tensor(data["funding"].reset_index(drop=True).values, dtype=torch.float32, device=device)
        # 2 classes: 4 window features and 2 linear features
        self.observation_shape = ObservationShape(2, 4, self.window, 2)
        # transaction fee is used when transition from one state to another
        self.transaction_fee = configs.transaction_fee
        self.current_position = MainActionTypes.HOLD
        # action < 0 => short, action > 0 => long, action = 0 => hold
        # set step (0, 1) if you want to be able to take long/short position using some percentage of current balance.
        self.action_space = RangeSpace(-1, 1, 1)
        # action_history used for render
        self.action_history = []
        self.prices_numpy = self.prices.cpu().numpy()
        self.last_transition_price = self.prices[self.time_point]
        self._feature_cache = {
            'prices': {},
            'funding': {},
            'volume': {},
            'spread': {}
        }

    def reset(self, seed=None, options=None):
        # Reset the environment to its initial state
        # self.time_point = np.random.randint(self.window, self.max_time_point - self.window * 2)
        self.time_point = self.window
        self.action_history = []
        self.current_position = MainActionTypes.HOLD
        return self._get_observation(), {}

    def infeasible_step(self):
        self.action_history.append(self.current_position)
        terminated = False
        truncated = (self.time_point >= self.max_time_point - 1)
        reward = -1
        if not truncated and not terminated:
            self.time_point += 1
            scale = abs(self.prices[self.time_point] / self.prices[self.time_point - 1] - 1)
            reward *= scale * 100
        return self._get_observation(), reward, terminated, truncated, {}

    def step(self, action: int):
        action_value = self.action_space.range_value(action)
        action_type = MainActionTypes.percentage_to_type(action_value)

        # if not self._check_action_feasible(action_type):
        #     return self.infeasible_step()

        self.action_history.append(action_type)

        if action_type == MainActionTypes.LONG:
            transition = self._take_long()
        elif action_type == MainActionTypes.SHORT:
            transition = self._take_short()
        else:
            transition = self._take_hold()

        if transition:
            self.last_transition_price = self.prices[self.time_point]

        terminated = False
        truncated = (self.time_point >= self.max_time_point - 1)

        reward = torch.tensor(0.0, device=self.device)
        if not truncated and not terminated:
            self.time_point += 1
            reward = action_value * (self.prices[self.time_point] / self.prices[self.time_point - 1] - 1)
        reward -= self._calculate_transition_penalty(transition, reward)
        reward *= 100

        return self._get_observation(), reward, terminated, truncated, {}

    def _take_long(self) -> bool:
        transition = self.current_position != MainActionTypes.LONG
        self.current_position = MainActionTypes.LONG
        return transition

    def _take_short(self) -> bool:
        transition = self.current_position != MainActionTypes.SHORT
        self.current_position = MainActionTypes.SHORT
        return transition

    def _take_hold(self) -> bool:
        transition = self.current_position != MainActionTypes.HOLD
        self.current_position = MainActionTypes.HOLD
        return transition

    def _calculate_transition_penalty(self, transition: bool, reward: float) -> float:
        if not transition:
            return 0
        return self.transaction_fee * abs(reward)

    def _check_action_feasible(self, action: MainActionTypes) -> bool:
        if action == MainActionTypes.LONG or action == MainActionTypes.SHORT:
            if self.current_position != MainActionTypes.HOLD:
                return False
        return True

    def get_current_price(self):
        # again: if decide extend to multiple currency - add corresponding logic
        return self.prices[self.time_point]

    def get_current_timestamp(self):
        return self.dates[self.time_point]

    def _get_window_feature(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        start = self.time_point - self.window + 1
        stop = self.time_point + 1

        if start < 0:
            padding = abs(start)
            padded = torch.cat([torch.zeros(padding, dtype=feature_tensor.dtype), feature_tensor[0:stop]])
            return padded
        else:
            return feature_tensor[start:stop]

    def _get_observation(self) -> torch.Tensor:
        def cached_processed(feature_name, feature_data):
            index = self.time_point
            cache = self._feature_cache[feature_name]
            if index not in cache:
                cache[index] = self._get_normalized_feature(self._get_window_feature(feature_data))
            return cache[index]

        prices_in_window = cached_processed('prices', self.prices)
        funding_in_window = cached_processed('funding', self.funding)
        volume_in_window = cached_processed('volume', self.volume_default)
        spread_in_window = cached_processed('spread', self.spread)

        window_features = torch.cat([prices_in_window, funding_in_window, volume_in_window, spread_in_window])
        linear_features = torch.tensor([
            self.last_transition_price / self.prices[self.time_point],
            self.current_position.get_action_value()
        ], dtype=torch.float32, device=self.device)
        
        return torch.cat([window_features, linear_features])

    def _make_state_spans(self) -> list:
        """
        Create spans for LONG and SHORT states over time on the plot, handling transitions properly.
        """
        state_spans = []
        start_idx = None
        current_state = None

        for i, action in enumerate(self.action_history):
            idx = self.window + i

            if action in [MainActionTypes.LONG, MainActionTypes.SHORT]:
                if start_idx is None:
                    # Start a new span if no span is active
                    start_idx = idx
                    current_state = action
                elif current_state != action:
                    # End the current span if the state changes (e.g., LONG → SHORT)
                    state_spans.append(
                        dict(
                            type="rect",
                            xref="x", yref="paper",
                            x0=self.dates[start_idx], x1=self.dates[idx],
                            y0=0, y1=1,
                            fillcolor="green" if current_state == MainActionTypes.LONG else "red",
                            opacity=0.2,
                            layer="below",
                            line_width=0
                        )
                    )
                    # Start a new span for the new state
                    start_idx = idx
                    current_state = action
            else:
                # End the span if we encounter HOLD or an invalid state
                if start_idx is not None:
                    state_spans.append(
                        dict(
                            type="rect",
                            xref="x", yref="paper",
                            x0=self.dates[start_idx], x1=self.dates[idx],
                            y0=0, y1=1,
                            fillcolor="green" if current_state == MainActionTypes.LONG else "red",
                            opacity=0.2,
                            layer="below",
                            line_width=0
                        )
                    )
                    start_idx = None
                    current_state = None

        # Handle an ongoing state until the end of the timeline
        if start_idx is not None:
            state_spans.append(
                dict(
                    type="rect",
                    xref="x", yref="paper",
                    x0=self.dates[start_idx], x1=self.dates[self.time_point],
                    y0=0, y1=1,
                    fillcolor="green" if current_state == MainActionTypes.LONG else "red",
                    opacity=0.2,
                    layer="below",
                    line_width=0
                )
            )

        return state_spans

    def _make_state_transition_lines(self) -> list:
        transition_lines = []

        for i, action in enumerate(self.action_history):
            idx = self.window + i
            if i > 0 and self.action_history[i] != self.action_history[i - 1]:
                transition_lines.append(
                    dict(
                        type="line",
                        xref="x", yref="y",
                        x0=self.dates[idx], x1=self.dates[idx],
                        y0=min(self.prices_numpy),
                        y1=max(self.prices_numpy),
                        line=dict(color="blue", width=1, dash="dash")
                    )
                )
        return transition_lines

    def _make_action_line(self) -> list[go.Scatter]:
        long_x_coordinates = []
        long_y_coordinates = []
        short_x_coordinates = []
        short_y_coordinates = []
        hold_x_coordinates = []
        hold_y_coordinates = []

        for i, action in enumerate(self.action_history):
            idx = self.window + i
            if i == 0 or action != self.action_history[i - 1]:
                if action == MainActionTypes.LONG:
                    long_x_coordinates.append(self.dates[idx])
                    long_y_coordinates.append(self.prices_numpy[idx])
                elif action == MainActionTypes.SHORT:
                    short_x_coordinates.append(self.dates[idx])
                    short_y_coordinates.append(self.prices_numpy[idx])
                else:
                    hold_x_coordinates.append(self.dates[idx])
                    hold_y_coordinates.append(self.prices_numpy[idx])

        return [
            go.Scatter(
                x=long_x_coordinates,
                y=long_y_coordinates,
                mode='markers',
                marker=dict(size=10, color='green'),
                name='LONG Transitions'
            ),
            go.Scatter(
                x=short_x_coordinates,
                y=short_y_coordinates,
                mode='markers',
                marker=dict(size=10, color='red'),
                name='SHORT Transitions'
            ),
            go.Scatter(
                x=hold_x_coordinates,
                y=hold_y_coordinates,
                mode='markers',
                marker=dict(size=10, color='yellow'),
                name='HOLD Transitions'
            )
        ]

    def render(self):
        trace_btc_price = go.Scatter(
            x=self.dates[:self.time_point + 1],
            y=self.prices_numpy[:self.time_point + 1],
            mode='lines',
            name='BTC Price'
        )

        layout = go.Layout(
            title='Crypto Trading Environment',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Price (USD)'),
            margin=dict(r=200),
            legend=dict(x=0.01, y=0.98),
            shapes=self._make_state_spans(),
        )

        data_to_plot = [trace_btc_price]
        if self.time_point + 1 < len(self.dates):
            trace_current_time = go.Scatter(
                x=[self.dates[self.time_point + 1]],
                y=[self.prices_numpy[self.time_point + 1]],
                mode='markers',
                name='Current Time Point',
                marker=dict(color='red', size=10)
            )
            data_to_plot.append(trace_current_time)

        data_to_plot += self._make_action_line()

        fig = go.Figure(data=data_to_plot, layout=layout)
        pyo.plot(fig, filename=f"{directory}/crypto_trading_environment.html", auto_open=False, include_plotlyjs='cdn')

    @staticmethod
    def _get_normalized_feature(feature_tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return feature_tensor / (feature_tensor.mean() + eps)

    @staticmethod
    def _exponential_moving_average(prices, period, weighting_factor=0.2):
        ema = np.zeros(len(prices))
        sma = np.mean(prices[:period])
        ema[period - 1] = sma
        for i in range(period, len(prices)):
            ema[i] = (prices[i] * weighting_factor) + (ema[i - 1] * (1 - weighting_factor))
        return ema
