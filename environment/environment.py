import gym
import numpy as np
import pandas as pd
import copy
import plotly.graph_objs as go
import plotly.offline as pyo
from enum import Enum
from environment.balance import Balance
from configurations.config import EnvParameters


class MainActionTypes(Enum):
    HOLD = 1
    SELL = 2
    BUY = 3

    @staticmethod
    def percentage_to_type(val: float):
        if val > 0:
            return MainActionTypes.SELL
        elif val < 0:
            return MainActionTypes.BUY
        else:
            return MainActionTypes.HOLD


class RangeSpace(gym.spaces.Discrete):
    """
    RangeSpace allows to create a discrete space from a continuous range [low, high]
    """
    def __init__(self, low: float, high: float, step: float):
        num_actions = int((high - low) / step) + 1
        super(RangeSpace, self).__init__(num_actions)
        self.low = low
        self.high = high
        self.step = step

    def range_value(self, discrete_value: int):
        return self.low + self.step * discrete_value

    def __repr__(self):
        return f"RangeSpace(low={self.low}, high={self.high}, step={self.step})"


class CryptoTradingEnvironment(gym.Env):
    # now CryptoTradingEnvironment works only with USD and BTC. Doubt that it will be extended to use other currency,
    # but if so, the only problem is to establish the form of the input data (which now is csv with btc prices).
    # The input data in multi currencies case should map every other currency to one main currency.
    # However, for multi currencies case the action space must also be changed, as the agent must decide what to
    # sell and what to buy (even if there will be only one purpose currency, i.e. BTC, the agent decides what to sell).

    # If the environment will be developed to handle the multi currency case, the self.price attribute must be
    # modified to represent rates of different currencies in different time points, also some functions which
    # work with Balance must be modified. For example, _get_overall_current_balance must use corresponding prices
    def __init__(self, initial_balance: Balance, configs: EnvParameters):
        super(CryptoTradingEnvironment, self).__init__()

        self.time_point = 0
        self.window = configs.window
        data = pd.read_csv(configs.data_path)
        data['date'] = pd.to_datetime(data['date'])
        data.sort_values(by='date', inplace=True)
        self.prices = data['close'][-self.window:].reset_index(drop=True)
        self.dates = data['date'][-self.window:].reset_index(drop=True)

        self.initial_balance = initial_balance

        self.current_balance = copy.deepcopy(initial_balance)
        self.initial_overall_balance = self.get_overall_current_balance()

        self.TRANSACTION_FEE = 0.01
        # action < 0 => buy, action > 0 => sell, action = 0 => hold
        # set step (0, 1) if you want to be able to sell/buy using some percentage of current balance
        self.action_space = RangeSpace(-1, 1, configs.action_step_size)
        # action_history used for render
        self.action_history = []

    def reset(self):
        # Reset the environment to its initial state
        self.current_balance = copy.deepcopy(self.initial_balance)
        self.time_point = 0
        self.action_history = []
        return self._get_observation(), {}

    def step(self, action: int):
        percentage = self.action_space.range_value(action)
        action_type = MainActionTypes.percentage_to_type(percentage)
        self.action_history.append(action_type)
        if action_type == MainActionTypes.SELL:
            self._sell(abs(percentage))
        elif action_type == MainActionTypes.BUY:
            self._buy(abs(percentage))

        terminated = (self.get_overall_current_balance() <= 0.05 * self.initial_overall_balance)
        truncated = (self.time_point >= self.window - 1)

        overall_reward = self.get_overall_current_balance() - self.initial_overall_balance
        # usd_overall_reward - specify that in result we want to have more money in usd (worth to experiment
        # with this coefficient in the future)
        usd_overall_reward = 0
        if terminated or truncated:
            usd_overall_reward = 0.005 * (self.current_balance["USD"] - self.initial_balance["USD"])
        # reward function
        reward = overall_reward + usd_overall_reward - 10 * self.initial_overall_balance * terminated
        if self.current_balance["USD"] > self.initial_balance["USD"]:
            reward *= 3

        if not truncated:
            self.time_point += 1

        return self._get_observation(), reward, terminated, truncated, {}

    def _sell(self, sell_percentage: float):
        price = self.prices[self.time_point]
        if 0 <= sell_percentage <= 1:
            sell_proceeds = sell_percentage * self.current_balance['BTC'] * price
            trading_fees = self.TRANSACTION_FEE * sell_proceeds
            self.current_balance.update_balance("USD", sell_proceeds - trading_fees)
            self.current_balance.update_balance("BTC", -sell_percentage * self.current_balance["BTC"])

    def _buy(self, buy_percentage: float):
        price = self.prices[self.time_point]
        if 0 <= buy_percentage <= 1:
            buy_proceeds = buy_percentage * self.current_balance['USD'] / price
            trading_fees = self.TRANSACTION_FEE * buy_proceeds
            self.current_balance.update_balance("BTC", buy_proceeds - trading_fees)
            self.current_balance.update_balance("USD", -buy_percentage * self.current_balance["USD"])

    def get_overall_current_balance(self):
        price = self.prices[self.time_point]
        return self.current_balance["USD"] + self.current_balance["BTC"] * price

    def _get_observation(self):
        price = self.prices[self.time_point]
        # there is an idea to take time_point / (window - 1) into consideration.
        # It could make sense if the goal of the trading was "make me more money till date X". However,
        # the experiments on this topic needed. Example: we hold btc, it went down on the date X (seems like we lost
        # money), but went up on the date X + 6 month. So this state feature could help to specify that we want the best
        # yield close to some date.
        return np.array([self.time_point / (self.window - 1), price, self.current_balance["USD"], self.current_balance["BTC"]])
        # return np.array([price, self.current_balance["USD"], self.current_balance["BTC"]])

    def _make_action_line(self) -> list[go.Scatter]:
        x_coordinates = []
        y_coordinates = []
        marker_colors = []

        for i, action in enumerate(self.action_history):
            if action == MainActionTypes.SELL:
                x_coordinates.append(self.dates[i])
                y_coordinates.append(self.prices[i])
                marker_colors.append('red')
            elif action == MainActionTypes.BUY:
                x_coordinates.append(self.dates[i])
                y_coordinates.append(self.prices[i])
                marker_colors.append('green')

        scatter_traces = []
        for x, y, color in zip(x_coordinates, y_coordinates, marker_colors):
            scatter_traces.append(go.Scatter(x=[x], y=[y], mode='markers',
                                             marker=dict(color=color, size=10,
                                                         symbol='triangle-down' if color == 'red' else 'triangle-up')))

        return scatter_traces

    def render(self, directory):
        trace_btc_price = go.Scatter(x=self.dates[:self.time_point+2], y=self.prices[:self.time_point+2],
                                     mode='lines',
                                     name='BTC Price')

        layout = go.Layout(title='Crypto Trading Environment', xaxis=dict(title='Time'),
                           yaxis=dict(title='Price (USD)'), margin=dict(r=200),
                           legend=dict(x=0.01, y=0.98))

        data_to_plot = [trace_btc_price]
        if self.time_point + 1 < len(self.dates):
            trace_current_time = go.Scatter(x=[self.dates[self.time_point+1]], y=[self.prices[self.time_point+1]],
                                            mode='markers', name='Current Time Point', marker=dict(color='red', size=10))

            data_to_plot += [trace_current_time]

        data_to_plot += self._make_action_line()

        fig = go.Figure(data=data_to_plot, layout=layout)
        annotation_text = str(self.current_balance).replace("\n", "<br>")
        annotation_text += f"==========<br>Balance in USD:<br>{self.get_overall_current_balance():.2f}"
        fig.add_annotation(dict(x=1.02, y=max(self.prices[:self.time_point+1]),
                                text=annotation_text,
                                showarrow=False,
                                font=dict(color='green'),
                                xanchor='left',
                                xref="paper",
                                yref="y",
                                align="left"
                                ))
        pyo.plot(fig, filename=f"{directory}/crypto_trading_environment.html", auto_open=False)
