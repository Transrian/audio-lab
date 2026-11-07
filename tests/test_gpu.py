import argparse

import pytest

from audiolab.core import gpu as gpu_module
from audiolab.core.gpu import (
    Gpu,
    GpuError,
    add_gpu_argument,
    list_gpus,
    parse_nvidia_smi,
    resolve_gpu,
    select_gpu,
)

SMI_OUTPUT = """\
0, GPU-aaaa1111-0000-0000-0000-000000000000, NVIDIA GeForce RTX 5090, 32607
1, GPU-bbbb2222-0000-0000-0000-000000000000, NVIDIA GeForce RTX 3090, 24576
"""

GPUS = parse_nvidia_smi(SMI_OUTPUT)


def test_parse_nvidia_smi():
    assert GPUS == [
        Gpu(0, "GPU-aaaa1111-0000-0000-0000-000000000000", "NVIDIA GeForce RTX 5090", 32607),
        Gpu(1, "GPU-bbbb2222-0000-0000-0000-000000000000", "NVIDIA GeForce RTX 3090", 24576),
    ]


def test_list_gpus_without_nvidia_smi_is_empty():
    assert list_gpus(run=lambda: None) == []


def test_resolve_by_index():
    assert resolve_gpu("1", GPUS).name == "NVIDIA GeForce RTX 3090"


def test_resolve_by_uuid_prefix():
    assert resolve_gpu("GPU-bbbb", GPUS).index == 1


def test_resolve_unknown_gpu_lists_available_ones():
    with pytest.raises(GpuError, match="matches no GPU; available: 0 .*, 1 "):
        resolve_gpu("7", GPUS)


def test_resolve_ambiguous_prefix():
    with pytest.raises(GpuError, match="matches several GPUs"):
        resolve_gpu("GPU-", GPUS)


def test_resolve_without_gpus():
    with pytest.raises(GpuError, match="no NVIDIA GPU found"):
        resolve_gpu("0", [])


def test_select_gpu_sets_cuda_visible_devices(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    chosen = select_gpu("1", GPUS)
    assert chosen.index == 1
    assert gpu_module.os.environ["CUDA_VISIBLE_DEVICES"] == chosen.uuid


def test_gpu_argument_defaults_to_first_gpu():
    parser = argparse.ArgumentParser()
    add_gpu_argument(parser)
    assert parser.parse_args([]).gpu == "0"
    assert parser.parse_args(["--gpu", "GPU-bbbb"]).gpu == "GPU-bbbb"
