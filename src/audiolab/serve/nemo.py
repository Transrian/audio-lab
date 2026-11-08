"""The NeMo backend: a local checkpoint, decoded greedily on one GPU.

NeMo and PyTorch are imported here and nowhere else, so the rest of the
repository (and the whole test suite) works on a machine that has neither.

A restored archive carries the settings its own training run used, and several of
them are wrong for a server that answers one clip at a time. Each change below
says what breaks without it.
"""

from __future__ import annotations

import io
from collections.abc import Callable

from audiolab.core.audio import load_audio
from audiolab.serve.config import STRATEGY, ServeConfigError, ServeSpec

DEFAULT_DEVICE = "cuda"


class NemoBackend:
    """One NeMo ASR checkpoint, loaded on demand and reused for every request.

    ``device`` exists for the CPU smoke test that checks the plumbing on a
    machine without a GPU; the commands always serve on the GPU they selected
    ([D-008](../../../docs/owner/directives.md)).
    """

    def __init__(self, spec: ServeSpec, device: str = DEFAULT_DEVICE):
        self.spec = spec
        self.device = device
        self.model = None

    def load(self) -> None:
        """Read the checkpoint and put it in inference mode; fails loudly if it cannot."""
        from nemo.collections.asr.models import ASRModel

        checkpoint = self.spec.resolved()
        if checkpoint.path:
            model = ASRModel.restore_from(checkpoint.path, map_location="cpu")
        else:
            model = ASRModel.from_pretrained(model_name=checkpoint.hf_id, map_location="cpu")
        model.eval()
        self._configure_decoding(model)
        # Restored on the CPU and moved afterwards: that is what makes a
        # checkpoint saved on another machine loadable here, whatever the GPU.
        model.to(self.device)
        self.model = model

    def _configure_decoding(self, model) -> None:
        """The greedy decoding this server uses, whatever the archive carried.

        A checkpoint with no transducer joint is refused here, rather than failing
        into a transcript that looks like an answer.
        """
        from omegaconf import open_dict

        if getattr(model, "joint", None) is None:
            raise ServeConfigError(
                "this checkpoint has no transducer joint: the nemo serve backend "
                "only handles NeMo transducer ASR models"
            )
        with open_dict(model.cfg):
            model.cfg.decoding.strategy = STRATEGY
            # The fused batch paths assume one fixed batch shape, which is a
            # training-time speedup; a server decoding one clip at a time is
            # faster without them.
            model.cfg.decoding.fused_batch_size = None
            model.cfg.joint.fused_batch_size = None
            # Greedy decoding stops after this many symbols inside a blank run.
            model.cfg.decoding.greedy.max_symbols = self.spec.max_symbols
            # NeMo 2.6.1 asks the CUDA bindings for one value more than the
            # `cuda-python` it pins returns while capturing this decode, and a
            # decode that fails inside a capture comes back empty instead of
            # raising - an empty transcript that looks like an answer.
            model.cfg.decoding.greedy.use_cuda_graph_decoder = False
        # Build the decoder now: the one built while the archive was restored
        # would keep the archive's own settings, and building it later is what
        # the first request would otherwise pay for.
        model.change_decoding_strategy(model.cfg.decoding)

    def transcribe(self, audio: bytes, language: str = "") -> str:
        """One clip of any sample rate, as 16 kHz mono, decoded to text.

        ``language`` is accepted and ignored: Parakeet decides the language
        itself. It is part of the request the evaluation client sends, so a
        server that refused it would cost a second call per clip.
        """
        import torch

        if self.model is None:
            self.load()
        samples = load_audio(io.BytesIO(audio)).samples
        with torch.inference_mode():
            answers = self.model.transcribe(audio=[samples], batch_size=1, verbose=False)
        return _answer_text(answers[0])


def _answer_text(answer) -> str:
    """The text of one answer: a string, or something whose ``.text`` is one.

    Anything else means the model answered in a shape this server does not know.
    Turning that into ``str(...)`` would cache whatever it printed - a token list,
    a ``None`` - as a transcript, which is worse than failing the request.
    """
    text = answer if isinstance(answer, str) else getattr(answer, "text", None)
    if not isinstance(text, str):
        raise TypeError(f"the model answered with {type(answer).__name__}, which holds no text")
    return text


def transcriber(spec: ServeSpec, device: str = DEFAULT_DEVICE) -> Callable[[bytes, str], str]:
    """Load the checkpoint now and return the function the server calls.

    Loading here rather than at the first request means a checkpoint that cannot
    be read fails before a server exists, with the error of the loader rather
    than a 500 for every clip of a run.
    """
    backend = NemoBackend(spec, device=device)
    backend.load()
    return backend.transcribe
