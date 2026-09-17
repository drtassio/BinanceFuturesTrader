import hashlib
import json
from types import SimpleNamespace

import pytest

from trading.ai_controller import AIController


@pytest.mark.parametrize("case,approved", [
    ("complete", True), ("empty", False), ("partial", False),
    ("changed", False), ("missing", False), ("null", False),
    ("scaler_changed", False), ("contract_changed", False),
    ("artifacts_unrecorded", False),
    ("runtime_unrecorded", False), ("runtime_changed", False), ("settings_changed", False),
])
def test_approval_requires_all_current_policies(tmp_path, case, approved):
    controller = object.__new__(AIController)
    controller.config_ai = SimpleNamespace(MODEL_DIR=str(tmp_path),
                                          REQUIRE_OOS_POLICY_APPROVAL=True)
    controller.policy_validation_path = str(tmp_path / "approval.json")
    hashes = {}
    for name in controller._OOS_SPECIALISTS:
        content = name.encode()
        (tmp_path / f"{name}_specialist_sac.zip").write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
        (tmp_path / f"{name}_specialist_scaler.joblib").write_bytes(b"scaler")
        (tmp_path / f"{name}_feature_contract.json").write_text("{}")
    artifact_hashes = controller._specialist_artifact_hashes()
    from trading.policy_runtime import runtime_fingerprint
    runtime_hashes = runtime_fingerprint(controller.config_ai)
    if case == "empty":
        hashes = {}
    elif case == "partial":
        hashes.pop("bear")
    elif case == "changed":
        (tmp_path / "bear_specialist_sac.zip").write_bytes(b"different")
    elif case == "missing":
        (tmp_path / "bear_specialist_sac.zip").unlink()
    elif case == "null":
        hashes["bear"] = None
        (tmp_path / "bear_specialist_sac.zip").unlink()
    elif case == "scaler_changed":
        (tmp_path / "bull_specialist_scaler.joblib").write_bytes(b"different")
    elif case == "contract_changed":
        (tmp_path / "bull_feature_contract.json").write_text('{"changed": true}')
    elif case == "artifacts_unrecorded":
        artifact_hashes = {}
    elif case == "runtime_unrecorded":
        runtime_hashes = {}
    elif case == "runtime_changed":
        runtime_hashes['trading/execution_engine.py'] = 'old-code'
    elif case == "settings_changed":
        controller.config_ai.TRAINING_LEVERAGE_CAP = 15
    (tmp_path / "approval.json").write_text(json.dumps({
        "all_passed": True, "model_hashes": hashes, "artifact_hashes": artifact_hashes,
        "runtime_hashes": runtime_hashes,
    }), encoding="utf-8")

    assert controller._load_policy_oos_approval() is approved
