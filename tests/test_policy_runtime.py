from types import SimpleNamespace

from trading.policy_runtime import runtime_fingerprint


def test_secrets_are_not_part_of_approval_fingerprint():
    before = runtime_fingerprint(SimpleNamespace(API_SECRET='one'),
                                 SimpleNamespace(API_KEY='one', MAX_LEVERAGE_PER_TRADE=3))
    after = runtime_fingerprint(SimpleNamespace(API_SECRET='two'),
                                SimpleNamespace(API_KEY='two', MAX_LEVERAGE_PER_TRADE=3))
    assert before == after
    assert all(before.values())


def test_effective_leverage_change_invalidates_fingerprint():
    assert runtime_fingerprint(SimpleNamespace(), SimpleNamespace(MAX_LEVERAGE_PER_TRADE=3)) != (
        runtime_fingerprint(SimpleNamespace(), SimpleNamespace(MAX_LEVERAGE_PER_TRADE=15)))


def test_missing_runtime_files_fail_closed(tmp_path):
    assert not all(runtime_fingerprint(SimpleNamespace(), root=tmp_path).values())
