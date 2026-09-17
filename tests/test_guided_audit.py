import sys

import pytest

from scripts.audit_guided_policy import main


def test_audit_cannot_inspect_holdout_before_training_finishes(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['audit', '--agent', 'bull', '--run-directory', str(tmp_path),
                                    '--splits', 'holdout', '--output', str(tmp_path / 'audit.json')])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not (tmp_path / 'audit.json').exists()


def test_holdout_cannot_be_used_to_compare_candidate_checkpoints(tmp_path, monkeypatch):
    (tmp_path / 'bull_guided_report.json').write_text('{}')
    monkeypatch.setattr(sys, 'argv', ['audit', '--agent', 'bull', '--run-directory', str(tmp_path),
                                    '--splits', 'holdout', '--checkpoint', 'candidate.zip',
                                    '--output', str(tmp_path / 'audit.json')])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
