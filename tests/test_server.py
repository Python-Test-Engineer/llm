"""End-to-end tests for the LLM Lab HTTP API.

Starts the real server on a free port and talks to it exactly as the browser
does, so a broken endpoint fails here rather than in front of a student.

    uv run tests/test_server.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from http.server import ThreadingHTTPServer  # noqa: E402

import lab.server as server  # noqa: E402

BASE = None
HTTPD = None


def setup_module(_module=None):
    """pytest hook -- also called by _run_all below."""
    global BASE, HTTPD
    if HTTPD is not None:
        return
    server.STATE.reload_corpus()
    server.STATE.put_model("default", server.STATE.build({}))
    HTTPD = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    port = HTTPD.server_address[1]
    BASE = "http://127.0.0.1:%d" % port
    threading.Thread(target=HTTPD.serve_forever, daemon=True).start()


def _get(path, raw=False):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        data = r.read()
        return data if raw else json.loads(data)


def _post(path, payload):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


# -- static ----------------------------------------------------------------
def test_index_page_is_served():
    html = _get("/", raw=True).decode()
    assert "LLM Lab" in html
    assert "app.js" in html


def test_static_assets_are_served():
    js = _get("/app.js", raw=True).decode()
    css = _get("/style.css", raw=True).decode()
    assert "startTraining" in js
    assert "--accent" in css


def test_unknown_path_is_404():
    try:
        _get("/nope.js", raw=True)
        raise AssertionError("expected 404")
    except urllib.error.HTTPError as exc:
        assert exc.code == 404


# -- read endpoints --------------------------------------------------------
def test_lessons_endpoint():
    d = _get("/api/lessons")
    assert len(d["lessons"]) >= 7
    assert d["defaults"]["context"] == 3
    assert any(l["id"] == "train" for l in d["lessons"])


def test_corpus_endpoint():
    d = _get("/api/corpus")
    assert d["chars"] == 2396
    assert "little fox" in d["text"]


# -- model endpoints -------------------------------------------------------
def test_load_returns_the_tutorial_numbers():
    d = _post("/api/load", {})
    assert d["vocab_size"] == 28
    assert d["n_examples"] == 2393
    assert d["n_params"] == 1324
    assert len(d["vocab"]) == 28
    assert len(d["shapes"]) == 5


def test_examples_endpoint_uses_probe_text():
    d = _post("/api/examples", {"probe": "the cat sat", "context": 3, "count": 3})
    assert d["rows"][0]["context"] == "the"
    assert d["rows"][0]["target"] == " "
    assert d["rows"][1]["target"] == "c"
    assert d["total"] == len("the cat sat") - 3


def test_network_endpoint_scales_parameters():
    small = _post("/api/network", {"context": 3, "embed": 6, "hidden": 24})
    bigger = _post("/api/network", {"context": 3, "embed": 6, "hidden": 48})
    assert small["n_params"] == 1324
    assert bigger["n_params"] > small["n_params"]


def test_predict_returns_sorted_probabilities():
    _post("/api/load", {})
    d = _post("/api/predict", {"model_id": "default", "seed": "the", "k": 5})
    probs = [p["prob"] for p in d["predictions"]]
    assert probs == sorted(probs, reverse=True)
    assert abs(sum(d["all"]) - 1.0) < 1e-9
    assert len(d["all"]) == 28


# -- the training job lifecycle -------------------------------------------
def _wait_for_job(job_id, timeout=300):
    nxt, events, status = 0, [], "running"
    deadline = time.time() + timeout
    while time.time() < deadline:
        d = _get("/api/job?id=%s&from=%d" % (job_id, nxt))
        nxt = d["next"]
        events.extend(d["events"])
        status = d["status"]
        if status != "running":
            return status, events, d
        time.sleep(0.15)
    raise AssertionError("job did not finish in time")


def test_training_job_streams_events_and_then_serves_a_model():
    started = _post("/api/train", {"epochs": 4, "lr": 5, "context": 3})
    assert started["job_id"], started
    status, events, final = _wait_for_job(started["job_id"])
    assert status == "done", (status, events)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "start", kinds
    assert "milestone" in kinds, kinds
    assert kinds[-1] == "done", kinds
    # the loss genuinely fell
    assert final["loss"] < final["start_loss"] - 0.1, final

    # the trained model is now addressable by its job id
    mid = final["model_id"]
    pred = _post("/api/predict", {"model_id": mid, "seed": "the", "k": 3})
    assert len(pred["predictions"]) == 3

    gen = _post("/api/generate", {"model_id": mid, "seed": "the ",
                                  "length": 40, "temperature": 0.5})
    assert gen["outputs"][0].startswith("the ")
    assert len(gen["outputs"][0]) == 4 + 40, gen
    assert gen["trained_epochs"] == 4, gen

    emb = _post("/api/embeddings", {"model_id": mid})
    assert emb["trained"] is True
    assert len(emb["neighbours"]) == 28


def test_generate_validates_length():
    _post("/api/load", {})
    long = _post("/api/generate", {"model_id": "default", "seed": "the",
                                   "length": 99999, "temperature": 0.5})
    # clamped to MAX_LENGTH (600) rather than honoured literally
    assert len(long["outputs"][0]) == 3 + 600


def test_examples_rejects_a_too_short_corpus():
    try:
        _post("/api/examples", {"text": "ab", "context": 3})
        raise AssertionError("expected a 400")
    except urllib.error.HTTPError as exc:
        assert exc.code == 400


def test_high_learning_rate_is_flagged_as_diverging():
    """Exercise 1: a too-large lr must produce a visible warning, not silence."""
    started = _post("/api/train", {"epochs": 20, "lr": 15})
    status, events, final = _wait_for_job(started["job_id"])
    kinds = [e["type"] for e in events]
    assert "warning" in kinds or "error" in kinds, (status, kinds)
    assert final["loss"] > final["start_loss"], final


def _run_all():
    setup_module()
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
