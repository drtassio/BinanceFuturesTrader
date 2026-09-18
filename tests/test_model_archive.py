import io
import tarfile

import pytest

from scripts.promote_model import archive_run, safe_extract


def run_files(root):
    (root / 'models').mkdir(parents=True)
    (root / 'feature_contract.json').write_text('{}')
    (root / 'models' / 'bear_specialist_sac.zip').write_bytes(b'model')
    (root / 'models' / 'bear_specialist_scaler.joblib').write_bytes(b'scaler')


def test_nested_kaggle_archive_is_located(tmp_path):
    root = tmp_path / 'cloud' / 'artifacts' / 'bear_guided_123'
    run_files(root)
    assert archive_run(tmp_path, 'bear') == root


def test_ambiguous_runs_are_not_silently_selected(tmp_path):
    run_files(tmp_path / 'run1')
    run_files(tmp_path / 'run2')
    with pytest.raises(SystemExit):
        archive_run(tmp_path, 'bear')


@pytest.mark.parametrize('name,kind', [('../escape', tarfile.REGTYPE),
                                     ('/absolute', tarfile.REGTYPE),
                                     ('link', tarfile.SYMTYPE)])
def test_unsafe_archive_rejected_before_extraction(tmp_path, name, kind):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as tar:
        entry = tarfile.TarInfo(name)
        entry.type = kind
        tar.addfile(entry)
    stream.seek(0)
    with tarfile.open(fileobj=stream) as tar, pytest.raises(SystemExit):
        safe_extract(tar, tmp_path)
    assert not list(tmp_path.iterdir())
