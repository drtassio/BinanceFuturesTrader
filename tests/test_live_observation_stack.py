from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from config.settings import AIConfig
from specialists.bull_specialist import BullSpecialist, BullTradingEnv
from specialists.bear_specialist import BearSpecialist, BearTradingEnv
from specialists.ranger_specialist import RangerSpecialist, RangerTradingEnv
from tests.test_reward_and_sizing_invariants import DATASET


@pytest.mark.skipif(not DATASET.exists(), reason='causal dataset required')
@pytest.mark.parametrize('name,agent_class,env_class', [
    ('bull', BullSpecialist, BullTradingEnv),
    ('bear', BearSpecialist, BearTradingEnv),
    ('ranger', RangerSpecialist, RangerTradingEnv),
])
def test_live_stack_matches_training_frames_and_does_not_shift_on_poll(name, agent_class, env_class):
    frame = pd.read_parquet(DATASET).iloc[-4000:-3800].ffill().fillna(0.0)
    probe = env_class(df=frame.copy(), config=AIConfig(), specialist_name=f'{name}_specialist')
    scaler = probe._feature_scaler
    probe.close()
    agent = object.__new__(agent_class)
    agent.config = AIConfig()
    agent.feature_scaler = scaler
    agent.feature_columns = list(scaler.feature_names_in_)
    agent.specialist_name = f'{name}_specialist'
    agent.profitability_predictor = None
    agent.reward_hyperparams = {}
    state = np.array([1.0, 0.02, 0.12], dtype=np.float32)

    def training_frame(data):
        env = env_class(df=data.tail(100).copy(), config=agent.config,
                             feature_scaler=scaler, feature_columns=agent.feature_columns,
                             specialist_name=f'{name}_specialist')
        env.start_idx = 0
        env.current_step = len(env.df) - 1
        env.position, env.pnl_since_entry = float(state[0]), float(state[1])
        env.steps_in_position = float(state[2]) * 100.0
        result = env._get_observation()
        env.close()
        return result

    first = training_frame(frame.iloc[:-1])
    agent.model = SimpleNamespace(observation_space=SimpleNamespace(shape=(first.size * 4,)))
    live1 = agent.prepare_live_observation(frame.iloc[:-1], state)
    np.testing.assert_allclose(live1, np.concatenate([np.zeros(first.size * 3), first]))
    np.testing.assert_array_equal(agent.prepare_live_observation(frame.iloc[:-1], state), live1)
    second = training_frame(frame)
    live2 = agent.prepare_live_observation(frame, state)
    np.testing.assert_allclose(live2, np.concatenate([np.zeros(first.size * 2), first, second]))
    np.testing.assert_array_equal(agent.prepare_live_observation(frame, state), live2)

    gapped = frame.copy()
    gapped.index = gapped.index + pd.Timedelta(days=1)
    reset = agent.prepare_live_observation(gapped, state)
    assert np.count_nonzero(reset[:first.size * 3]) == 0
    with pytest.raises(ValueError, match='Missing live features'):
        agent.prepare_live_observation(frame.drop(columns=agent.feature_columns[0]), state)
