"""Train one model on our own data, from a config file to a checkpoint.

The commands are ``audiolab train``; the pieces are the config
(`config.py`), the clips a run trains on (`data.py`), the run directory and its
records (`run.py`), the fine-tune itself (`finetune.py`) and the sequence that
ties them together (`runner.py`). Only `finetune.py` needs PyTorch and NeMo.
"""
