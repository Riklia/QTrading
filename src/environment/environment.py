import gymnasium as gym
import numpy as np
import pandas as pd
import copy
import plotly.graph_objs as go
import plotly.offline as pyo
from sklearn.preprocessing import MinMaxScaler
from src.environment.balance import Balance
from src.environment.env_parameters import EnvParameters
from src.environment.action import RangeSpace, MainActionTypes


class CryptoTradingEnvironment(gym.Env):
    # now CryptoTradingEnvironment works only with USD and BTC. Doubt that it will be extended to use other currency,
    # but if so, the only problem is to establish the form of the input data_for_test (which now is csv with btc prices).
    # The input data_for_test in multi currencies case should map every other currency to one main currency.
    # However, for multi currencies case the action space must also be changed, as the agent must decide what to
    # sell and what to buy (even if there will be only one purpose currency, i.e. BTC, the agent decides what to sell).

    # If the environment will be developed to handle the multi currency case, the self.price attribute must be
    # modified to represent rates of different currencies in different time points, also some functions which
    # work with Balance must be modified. For example, _get_overall_current_balance must use corresponding prices
    def __init__(self, initial_balance: Balance, configs: EnvParameters):
        super(CryptoTradingEnvironment, self).__init__()
        self.window = configs.window
        # date processing
        data = pd.read_csv(configs.data_path)
        data = data[(configs.start_time <= data['date']) & (data['date'] <= configs.end_time)]
        data['date'] = pd.to_datetime(data['date'], unit="s")
        data.sort_values(by='date', inplace=True)
        self.dates = data['date'].reset_index(drop=True)
        self.max_time_point = len(self.dates) - 1
        # initial point is window size, not 0
        self.time_point = self.window
        # features
        self.prices = data["close_default"].reset_index(drop=True)
        self.volume_default = data["volume_default"].reset_index(drop=True)
        self.spread = data["spread"].reset_index(drop=True)
        self.funding = data["funding"].reset_index(drop=True)

        self.initial_balance = initial_balance

        self.current_balance = copy.deepcopy(initial_balance)
        self.initial_overall_balance = self.get_overall_current_balance()

        self.transaction_fee = configs.transaction_fee
        # action < 0 => buy, action > 0 => sell, action = 0 => hold
        # set step (0, 1) if you want to be able to sell/buy using some percentage of current balance
        self.action_space = RangeSpace(-1, 1, configs.action_step_size)
        # action_history used for render
        self.action_history = []
        self.terminate_threshold = configs.terminate_threshold

    def reset(self):
        # Reset the environment to its initial state
        self.current_balance = copy.deepcopy(self.initial_balance)
        self.time_point = self.window
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

        terminated = (self.get_overall_current_balance() < self.terminate_threshold * self.initial_overall_balance)
        truncated = (self.time_point >= self.max_time_point - 1)

        overall_reward = (self.get_overall_current_balance() - self.initial_overall_balance) / self.initial_overall_balance
        time_penalty = 0
        if overall_reward <= 0:
            time_penalty = abs(0.001 * overall_reward) * self.time_point
        else:
            overall_reward *= 5
        # usd_overall_reward - specify that in result we want to have more money in usd (worth to experiment
        # with this coefficient in the future)
        usd_overall_reward = 0
        if terminated or truncated:
            # usd_overall_reward = 0.005 * (self.current_balance["USD"] - self.initial_balance["USD"]) / self.initial_balance["USD"]
            if truncated:
                print(f"END ON {self.time_point} Balance: {self.get_overall_current_balance()}")
            if terminated:
                print(f"LOST ON {self.time_point}")
        # reward function
        reward = overall_reward + usd_overall_reward - 5 * terminated - 0*time_penalty
        if not truncated:
            self.time_point += 1

        return self._get_observation(), reward, terminated, truncated, {}

    def _sell(self, sell_percentage: float):
        btc_price = self.get_current_price()
        if 0 <= sell_percentage <= 1:
            # to avoid restricting agent in actions, we include transaction fee in sell percentage and this way
            # always sure that we have enough money for the transaction
            btc_to_subtract = sell_percentage * self.current_balance["BTC"]
            btc_no_fee = btc_to_subtract - btc_to_subtract * self.transaction_fee
            sell_proceeds_usd = btc_no_fee * btc_price
            trading_fees_usd = self.transaction_fee * sell_proceeds_usd
            self.current_balance.update_balance("USD", sell_proceeds_usd - trading_fees_usd)
            self.current_balance.update_balance("BTC", -btc_to_subtract)

    def _buy(self, buy_percentage: float):
        btc_price = self.get_current_price()
        if 0 <= buy_percentage <= 1:
            # to avoid restricting agent in actions, we include transaction fee in buy percentage and this way
            # always sure that we have enough money for the transaction
            usd_to_subtract = buy_percentage * self.current_balance["USD"]
            usd_no_fee = usd_to_subtract - usd_to_subtract * self.transaction_fee
            buy_proceeds_btc = usd_no_fee / btc_price
            trading_fees_btc = self.transaction_fee * buy_proceeds_btc
            self.current_balance.update_balance("BTC", buy_proceeds_btc - trading_fees_btc)
            self.current_balance.update_balance("USD", -usd_to_subtract)

    def get_current_price(self):
        # again: if decide extend to multiple currency - add corresponding logic
        return self.prices[self.time_point]

    def get_current_timestamp(self):
        return self.dates[self.time_point]

    def get_overall_current_balance(self):
        price = self.get_current_price()
        return self.current_balance["USD"] + self.current_balance["BTC"] * price

    def _get_window_feature(self, feature_array, include_current: bool = True):
        start = self.time_point - self.window
        stop = self.time_point
        if include_current:
            stop += 1

        result = []
        if start < 0:
            result.extend([0] * abs(start))
            start = 0

        result.extend(feature_array[start:stop])
        return result

    def _get_observation(self):
        prices_in_window = self._get_window_feature(self.prices, True)
        funding_in_window = self._get_window_feature(self.funding, True)
        volume_in_window = self._get_window_feature(self.volume_default, True)
        spread_in_window = self._get_window_feature(self.spread, True)
        ema_price = self._exponential_moving_average(prices_in_window, self.window // 2)
        usd_state = self.current_balance["USD"] / (self.initial_balance["USD"] + 1e-7)
        btc_state = self.current_balance["BTC"] / (self.initial_balance["BTC"] + 1e-7)
        overall_balance_state = self.get_overall_current_balance() / (self.initial_overall_balance + 1e-7)
        return np.array(prices_in_window + funding_in_window + volume_in_window + spread_in_window + list(ema_price) +
                        [usd_state, btc_state, overall_balance_state])

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
                                                         symbol='triangle-down' if color == 'red' else 'triangle-up'),
                                             showlegend=False))

        return scatter_traces

    def render(self, directory):
        trace_btc_price = go.Scatter(x=self.dates[:self.time_point+1], y=self.prices[:self.time_point+1],
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
        pyo.plot(fig, filename=f"{directory}/crypto_trading_environment.html", auto_open=False,
                 include_plotlyjs='cdn')

    @staticmethod
    def _minmax_scale_feature(input_feature: pd.Series):
        scaler = MinMaxScaler()
        return pd.Series(scaler.fit_transform(input_feature.values.reshape(-1, 1)).flatten())

    @staticmethod
    def _exponential_moving_average(prices, period, weighting_factor=0.2):
        ema = np.zeros(len(prices))
        sma = np.mean(prices[:period])
        ema[period - 1] = sma
        for i in range(period, len(prices)):
            ema[i] = (prices[i] * weighting_factor) + (ema[i - 1] * (1 - weighting_factor))
        return ema
