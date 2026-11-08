"""The batch normalization layers keep their statistics during a fine-tune.

This needs PyTorch and NeMo, which the default environment does not install, so
it is skipped there; run it with the environment of `scripts/nemo-venv.sh`:
`.venv-nemo/bin/python -m pytest tests/test_train_batch_norm.py`.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nemo")

from audiolab.train.finetune import FrozenBatchNormStatistics  # noqa: E402


def test_statistics_stay_and_scale_still_trains():
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.BatchNorm1d(3))
    batch_norm = model[1]
    model.train()
    before = (batch_norm.running_mean.clone(), batch_norm.running_var.clone())

    FrozenBatchNormStatistics().on_train_batch_start(None, model, None, 0)
    model(torch.randn(8, 4) * 5 + 3).sum().backward()

    assert not batch_norm.training
    assert model[0].training
    assert torch.equal(batch_norm.running_mean, before[0])
    assert torch.equal(batch_norm.running_var, before[1])
    assert batch_norm.weight.grad is not None
