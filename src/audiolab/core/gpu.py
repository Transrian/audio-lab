"""GPU discovery and selection.

Every command that uses a GPU selects exactly one, with a ``--gpu`` option that
accepts an index (``1``) or a UUID (``GPU-1a2b...``, a unique prefix is enough).

Selection works by setting ``CUDA_VISIBLE_DEVICES`` to the GPU's UUID, so the
chosen GPU appears as device 0 to CUDA libraries. It must happen before any
library (PyTorch, NeMo, ...) initializes CUDA.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

DEFAULT_GPU = "0"

_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,uuid,name,memory.total",
    "--format=csv,noheader,nounits",
]


class GpuError(ValueError):
    """Raised when a GPU specification does not match exactly one GPU."""


@dataclass(frozen=True)
class Gpu:
    index: int
    uuid: str
    name: str
    memory_mib: int


def parse_nvidia_smi(output: str) -> list[Gpu]:
    """Parse the CSV output of the ``nvidia-smi`` query used by :func:`list_gpus`."""
    gpus = []
    for line in output.strip().splitlines():
        index, uuid, name, memory = (field.strip() for field in line.split(","))
        gpus.append(Gpu(index=int(index), uuid=uuid, name=name, memory_mib=int(memory)))
    return gpus


def _run_nvidia_smi() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    result = subprocess.run(_QUERY, capture_output=True, text=True, check=False)
    return result.stdout if result.returncode == 0 else None


def list_gpus(run: Callable[[], str | None] = _run_nvidia_smi) -> list[Gpu]:
    """Return the NVIDIA GPUs of this machine, or an empty list if there are none."""
    output = run()
    return parse_nvidia_smi(output) if output else []


def resolve_gpu(spec: str, gpus: list[Gpu]) -> Gpu:
    """Return the single GPU matching ``spec`` (an index or a UUID prefix)."""
    spec = spec.strip()
    if not gpus:
        raise GpuError("no NVIDIA GPU found")
    if spec.isdigit():
        matches = [gpu for gpu in gpus if gpu.index == int(spec)]
    else:
        matches = [gpu for gpu in gpus if gpu.uuid.startswith(spec)]
    if len(matches) != 1:
        available = ", ".join(f"{gpu.index} ({gpu.uuid})" for gpu in gpus)
        problem = "matches no GPU" if not matches else "matches several GPUs"
        raise GpuError(f"--gpu {spec!r} {problem}; available: {available}")
    return matches[0]


def select_gpu(spec: str, gpus: list[Gpu] | None = None) -> Gpu:
    """Make the GPU matching ``spec`` the only one visible to this process."""
    gpu = resolve_gpu(spec, list_gpus() if gpus is None else gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu.uuid
    return gpu


def add_gpu_argument(parser: argparse.ArgumentParser) -> None:
    """Add the standard ``--gpu`` option to a command."""
    parser.add_argument(
        "--gpu",
        default=DEFAULT_GPU,
        help="GPU to use: an index or a UUID prefix (default: %(default)s). "
        "Run 'audiolab gpus' to list them.",
    )
