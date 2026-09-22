"""Cloud evaluation must preserve each specialist's training environment."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("name", ["bull", "bear"])
def test_evaluation_uses_specialist_environment(monkeypatch, name):
    from cloud.train_agent import _episode_environment
    from specialists.bull_specialist import BullTradingEnv
    from specialists.bear_specialist import BearTradingEnv
    from stable_baselines3.common import vec_env

    expected = {"bull": BullTradingEnv, "bear": BearTradingEnv}[name]
    captured = {}
    raw = SimpleNamespace(max_steps=0,
                          set_phase3_runtime_tweaks=lambda: captured.update(phase3=True))

    def make_environment(frame, **kwargs):
        captured.update(kwargs)
        return raw

    monkeypatch.setattr(vec_env, "DummyVecEnv", lambda factories: factories[0]())
    monkeypatch.setattr(vec_env, "VecFrameStack", lambda env, **kwargs: env)
    monkeypatch.setattr(vec_env, "VecNormalize", lambda env, **kwargs: env)
    agent = SimpleNamespace(_make_trend_env=make_environment,
                            feature_columns=["signal"])
    result = _episode_environment(agent, [0] * 20, name)

    assert captured["env_class"] is expected
    assert captured["feature_columns"] == ["signal"]
    assert captured["mode"] == "training"
    assert captured["phase3"] is True
    assert result is raw
    assert raw.max_steps == 19
