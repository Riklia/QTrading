import pytest
from src.environment import CryptoTradingEnvironment
from src.environment.action import MainActionTypes
from src.environment.env_parameters import EnvParameters


@pytest.fixture(scope="session")
def parameters_for_env() -> EnvParameters:
    params = EnvParameters(
        data_path="data_for_test/data_env_test.csv",
        start_time=1677865680,
        end_time=1677924600,
        window=1,
        transaction_fee=0.01
    )
    return params


@pytest.fixture(scope="session")
def environment(parameters_for_env: EnvParameters) -> CryptoTradingEnvironment:
    env = CryptoTradingEnvironment(parameters_for_env)
    return env


class TestEnvironment:

    def test_reset(self, environment: CryptoTradingEnvironment, parameters_for_env: EnvParameters):
        # given
        time_point_expected = 1
        action_history_len_expected = 0
        # when
        environment.reset()
        # then
        assert environment.time_point == time_point_expected
        assert len(environment.action_history) == action_history_len_expected

    def test_take_hold_position(self, environment: CryptoTradingEnvironment, parameters_for_env: EnvParameters):
        environment.reset()
        # given
        short_action = 0
        assert environment.current_position == MainActionTypes.HOLD
        environment.step(short_action)
        assert environment.current_position == MainActionTypes.SHORT

        # when
        hold_action = 1
        observation, reward, terminated, truncated, _ = environment.step(hold_action)

        # then
        assert environment.current_position == MainActionTypes.HOLD
        assert len(environment.action_history) == 2
        assert environment.time_point == 3
        assert reward == 0
        assert terminated is False
        assert truncated is False

    def test_take_long_position(self, environment: CryptoTradingEnvironment, parameters_for_env: EnvParameters):
        environment.reset()
        # given
        long_action = 2
        assert environment.current_position == MainActionTypes.HOLD

        # when
        observation, reward, terminated, truncated, _ = environment.step(long_action)

        # then
        assert environment.current_position == MainActionTypes.LONG
        assert len(environment.action_history) == 1
        assert environment.time_point == 2
        assert reward < 0
        assert terminated is False
        assert truncated is False

    def test_take_short_position(self, environment: CryptoTradingEnvironment, parameters_for_env: EnvParameters):
        environment.reset()
        # given
        short_action = 0
        assert environment.current_position == MainActionTypes.HOLD

        # when
        observation, reward, terminated, truncated, _ = environment.step(short_action)

        # then
        assert environment.current_position == MainActionTypes.SHORT
        assert len(environment.action_history) == 1
        assert environment.time_point == 2
        assert reward > 0
        assert terminated is False
        assert truncated is False

    def test_infeasible_action(self, environment: CryptoTradingEnvironment, parameters_for_env: EnvParameters):
        environment.reset()
        # given
        short_action = 0
        assert environment.current_position == MainActionTypes.HOLD
        environment.step(short_action)
        assert environment.current_position == MainActionTypes.SHORT

        # when
        long_action = 2
        observation, reward, terminated, truncated, _ = environment.step(long_action)

        # then
        assert environment.current_position == MainActionTypes.SHORT
        assert len(environment.action_history) == 2
        assert environment.time_point == 3
        assert reward < 0
        assert terminated is False
        assert truncated is False

