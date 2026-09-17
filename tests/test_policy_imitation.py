import pytest
import torch

from learning.policy_imitation import decision_classes, imitation_weights, imitation_fidelity


def observations(positions):
    result = torch.zeros((len(positions), 40))
    result[:, -11] = torch.tensor(positions)
    return result


def test_wait_and_exit_have_different_classes_despite_same_vote():
    obs = observations([0, 1, -1, 0, 1, -1])
    votes = torch.tensor([-.8, -.8, .8, .8, .8, -.8])
    assert decision_classes(obs, votes, (1, -1)).tolist() == [2, 5, 6, 1, 3, 4]
    assert decision_classes(obs[:2], votes[:2], (1,)).tolist() == [0, 5]


def test_rare_exits_receive_equal_total_training_mass():
    obs = observations([0] * 1000 + [1] * 2)
    actions = torch.zeros((1002, 3))
    actions[:, 0] = -.8
    weights = imitation_weights(obs, actions, (1,))
    assert weights[:1000].sum().item() == pytest.approx(weights[1000:].sum().item())
    assert weights.mean().item() == pytest.approx(1.)


def test_missing_all_exits_cannot_claim_perfect_macro_fidelity():
    obs = observations([0] * 1000 + [1] * 2)
    teacher = torch.full((1002,), -.8)
    predicted = teacher.clone()
    predicted[-2:] = .8
    result = imitation_fidelity(obs, teacher, predicted, (1,))
    assert result['exit_count'] == 2
    assert result['exit_recall'] == 0
    assert result['macro_recall'] == .5


def test_no_exit_examples_are_not_reported_as_perfect_exit_recall():
    obs = observations([0, 0])
    votes = torch.tensor([-.8, .8])
    assert imitation_fidelity(obs, votes, votes, (1,))['exit_recall'] is None


def test_neutral_vote_keeps_a_position_instead_of_claiming_exit():
    assert decision_classes(observations([1, -1]), torch.zeros(2), (1, -1)).tolist() == [3, 4]


def test_scaled_position_tail_is_rejected():
    with pytest.raises(ValueError):
        decision_classes(observations([.3]), torch.tensor([-.8]), (1,))
