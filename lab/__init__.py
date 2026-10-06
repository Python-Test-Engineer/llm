"""LLM Lab -- an interactive companion app for mini_llm.py.

A student-facing lab: the same tiny language model, but every stage is
inspectable and every knob is turnable from a browser.

Modules:
    engine   -- the model itself (a faithful port of the maths in mini_llm.py)
    lessons  -- the guided tutorial content
    server   -- a stdlib-only HTTP server that serves the UI and JSON API
"""

__all__ = ["engine", "lessons", "server"]
