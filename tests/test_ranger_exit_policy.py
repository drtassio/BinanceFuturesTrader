from types import SimpleNamespace

import pytest

from config.settings import AIConfig
from specialists.ranger_specialist import RangerTradingEnv
from learning.policy_imitation import decision_classes
import torch


@pytest.mark.parametrize('config', [None, SimpleNamespace(),
                                  SimpleNamespace(ENABLE_RANGER_RULE_BASED_EXITS=False)])
def test_ranger_cannot_replace_actor_exit_with_legacy_profit_rule(config):
    state = SimpleNamespace(config=config)
    assert RangerTradingEnv._should_exit_fast(state, .02) is False


def test_legacy_exit_requires_explicit_opt_in():
    state = SimpleNamespace(config=SimpleNamespace(ENABLE_RANGER_RULE_BASED_EXITS=True))
    assert RangerTradingEnv._should_exit_fast(state, .02) is True


def test_ranger_learns_both_entry_and_exit_directions():
    obs = torch.zeros((4, 40))
    obs[:, -11] = torch.tensor([0., 0., 1., -1.])
    assert decision_classes(obs, torch.tensor([.8, -.8, -.8, .8]), (1, -1)).tolist() == [1, 2, 5, 6]
