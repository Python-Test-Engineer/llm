"""Tests for the LLM Lab engine.

Run them either way:

    uv run --with pytest pytest tests/ -q
    uv run tests/test_engine.py

The point of these tests is to prove the lab's model really is the same model
as ``mini_llm.py`` -- not a lookalike with different numbers.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lab.engine import MiniLM, load_corpus  # noqa: E402

CORPUS_PATH = os.path.join(ROOT, "sample_text.txt")


def _fresh():
    return MiniLM(load_corpus(CORPUS_PATH))


# -- shape of the problem ---------------------------------------------------
def test_corpus_facts_match_the_tutorial():
    m = _fresh()
    assert len(m.text) == 2396
    assert m.vocab_size == 28
    assert len(m.examples) == 2393          # 2396 - context(3)
    assert m.n_params == 1324               # 28*6 + 24*18 + 24 + 28*24 + 28


def test_untrained_loss_equals_random_guessing():
    m = _fresh()
    # The tutorial's central sanity check: an untrained model must score
    # (almost exactly) like uniform guessing, ln(vocab_size).  Random weights
    # make the output nearly -- not perfectly -- uniform, so this is a tight
    # but not exact match, just as mini_llm.py reports 3.3320 vs ln(28)=3.3322.
    assert abs(m.average_loss() - math.log(28)) < 1e-3


def test_vocabulary_is_discovered_from_data():
    m = _fresh()
    # 'j' and 'z' never occur in the fox story, so they are not in the vocab.
    assert "j" not in m.char_to_id and "z" not in m.char_to_id
    assert m.char_to_id[" "] == 1
    assert m.char_to_id["\n"] == 0
    for c, i in m.char_to_id.items():
        assert m.id_to_char[i] == c


def test_softmax_is_a_probability_distribution():
    probs = MiniLM.softmax([1000.0, -1000.0, 0.0, 2.5])
    assert abs(sum(probs) - 1.0) < 1e-12
    assert all(p >= 0 for p in probs)
    # The huge score must dominate without overflowing.
    assert probs[0] > 0.99


# -- the sliding window -----------------------------------------------------
def test_examples_slide_by_one():
    m = MiniLM("the cat sat", context=3)
    first_context, first_target = m.examples[0]
    second_context, second_target = m.examples[1]
    assert "".join(m.id_to_char[c] for c in first_context) == "the"
    assert m.id_to_char[first_target] == " "
    # the window shifts by exactly one character
    assert first_context[1:] == second_context[:2]
    assert m.id_to_char[second_target] == "c"


# -- training actually learns ----------------------------------------------
def test_training_reduces_loss():
    m = _fresh()
    start = m.average_loss()
    last = None
    for ev in m.train(15, 5.0):
        last = ev
    assert math.isfinite(last["loss"])
    assert last["loss"] < start - 0.2
    # The loss reported for an epoch is measured with the weights *before*
    # that epoch's update (the same convention as mini_llm.py's "Final
    # loss"), so the model's true current loss is very slightly lower still.
    assert m.average_loss() < last["loss"]
    assert m.epochs_done == 15


def test_training_is_reproducible_with_a_seed():
    a = MiniLM(load_corpus(CORPUS_PATH), seed=0)
    b = MiniLM(load_corpus(CORPUS_PATH), seed=0)
    c = MiniLM(load_corpus(CORPUS_PATH), seed=7)
    la = [e["loss"] for e in a.train(3, 5.0)][-1]
    lb = [e["loss"] for e in b.train(3, 5.0)][-1]
    lc = [e["loss"] for e in c.train(3, 5.0)][-1]
    assert la == lb
    assert lc != la


def test_training_improves_the_markov_blind_spot():
    """After training, ' th' should strongly favour 'e' -- as the tutorial shows."""
    m = _fresh()
    for _ in m.train(60, 5.0):
        pass
    top = m.next_char_probs(" th", k=1)[0]
    assert top["char"] == "e"
    assert top["prob"] > 0.8


# -- generation -------------------------------------------------------------
def test_generation_respects_length_and_seed():
    m = _fresh()
    out = m.generate("the ", length=50, temperature=0.5)
    assert out.startswith("the ")
    assert len(out) == len("the ") + 50


def _greedy(model, seed, n):
    """Always take the single most likely next character."""
    ctx = model.to_context(seed)
    out = list(seed)
    for _ in range(n):
        _, _, probs = model.forward(ctx)
        nid = max(range(len(probs)), key=lambda i: probs[i])
        out.append(model.id_to_char[nid])
        ctx = ctx[1:] + [nid]
    return "".join(out)


def test_low_temperature_converges_on_the_argmax():
    """At a tiny temperature sampling is effectively 'always take the top one'."""
    m = _fresh()
    for _ in m.train(40, 5.0):
        pass
    assert m.generate(" th", length=60, temperature=0.05) == _greedy(m, " th", 60)


def test_high_temperature_is_wilder_than_low():
    m = _fresh()
    for _ in m.train(40, 5.0):
        pass
    low = m.generate(" the", length=150, temperature=0.05)
    high = m.generate(" the", length=150, temperature=1.2)
    assert len(set(low)) < len(set(high))


def test_short_context_is_padded_not_crashed():
    m = _fresh()
    ctx = m.to_context("")          # empty seed -> all spaces, length == context
    assert len(ctx) == m.context
    ctx = m.to_context("xy")        # unknown chars become spaces
    assert len(ctx) == m.context


# -- parity with the original script ---------------------------------------
def test_start_loss_matches_mini_llm_script():
    """The lab and mini_llm.py must agree on the untrained loss, to 4 dp."""
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "mini_llm.py"),
         "--epochs", "0", "--quiet"],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    match = re.search(r"Loss before training:\s*([0-9.]+)", result.stdout)
    assert match, result.stdout
    script_loss = float(match.group(1))
    lab_loss = _fresh().average_loss()
    assert abs(script_loss - lab_loss) < 5e-5, (script_loss, lab_loss)
    assert abs(script_loss - 3.3320) < 5e-5


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("PASS  " + fn.__name__)
        except AssertionError as exc:
            failed += 1
            print("FAIL  " + fn.__name__ + "  " + repr(exc))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("ERROR " + fn.__name__ + "  %s: %s" % (type(exc).__name__, exc))
    print("\n%d passed, %d failed" % (len(fns) - failed, failed))
    return failed


if __name__ == "__main__":
    raise SystemExit(_run_all())
