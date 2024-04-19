import pytest
from src.environment import CryptoTradingEnvironment, Balance
from src.environment.env_parameters import EnvParameters


@pytest.fixture(scope="session")
def parameters_for_env() -> EnvParameters:
    params = EnvParameters(
        data_path="data_for_test/data_env_test.csv",
        start_time=1677865680,
        end_time=1677924600,
        window=0,
        initial_balance=10000,
        action_step_size=0.25,
        terminate_threshold=0.5,  # terminate balance is 5000
        transaction_fee=0.1
    )
    return params


@pytest.fixture(scope="session")
def initial_balance(parameters_for_env: EnvParameters) -> Balance:
    balance = Balance()
    balance.update_balance("USD", parameters_for_env.initial_balance)
    balance.register_currency("BTC")
    return balance


@pytest.fixture(scope="session")
def environment(initial_balance: Balance, parameters_for_env: EnvParameters) -> CryptoTradingEnvironment:
    env = CryptoTradingEnvironment(initial_balance, parameters_for_env)
    return env


def test_reset(environment: CryptoTradingEnvironment, parameters_for_env: EnvParameters):
    # given
    time_point_expected = 0
    current_balance_val_expected = parameters_for_env.initial_balance
    action_history_len_expected = 0
    # when
    environment.reset()
    # then
    assert environment.time_point == time_point_expected
    assert environment.current_balance["USD"] == current_balance_val_expected
    assert len(environment.action_history) == action_history_len_expected


def test_buy(environment: CryptoTradingEnvironment, parameters_for_env: EnvParameters):
    environment.reset()
    # given
    buy_percentage = 0.1
    usd_after_transaction_expected = 9000
    btc_after_transaction_expected = 0.81

    # when
    # buying BTC with price 1000 per unit having 10000 USD using 0.1 of out USD balance
    environment._buy(buy_percentage)

    # then
    assert environment.current_balance["USD"] == pytest.approx(usd_after_transaction_expected)
    assert environment.current_balance["BTC"] == pytest.approx(btc_after_transaction_expected)


def test_sell(environment: CryptoTradingEnvironment, parameters_for_env: EnvParameters):
    environment.reset()
    # given
    # 1 BTC on the account
    environment.current_balance.update_balance("BTC", 1)
    sell_percentage = 0.1
    usd_after_transaction_expected = 10081
    btc_after_transaction_expected = 0.9

    # when
    environment._sell(sell_percentage)

    # then
    assert environment.current_balance["USD"] == pytest.approx(usd_after_transaction_expected)
    assert environment.current_balance["BTC"] == pytest.approx(btc_after_transaction_expected)

