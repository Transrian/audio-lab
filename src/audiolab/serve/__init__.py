"""Serve a local model through an OpenAI-compatible API (D-016).

A local checkpoint and a remote server are evaluated the same way: the same
client, the same cache, the same report. ``audiolab serve`` runs one checkpoint
in the foreground; ``audiolab eval`` starts the same server itself and stops it
when the suite is done, so a normal run never needs a second terminal.
"""
