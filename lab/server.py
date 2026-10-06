"""server.py -- a zero-dependency web server for LLM Lab.

Standard library only: ``http.server`` plus ``json``.  It serves the single
page app from ``static/`` and a small JSON API that the page calls.  Training
runs in a background thread and pushes progress events, so the browser can
show a live loss curve while the model learns.

Run it with::

    uv run llm_lab.py            # then open the printed URL

or directly::

    uv run lab/server.py
"""

from __future__ import annotations

import json
import math
import os
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# Make `lab` importable when this file is run directly.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lab.engine import MiniLM, DEFAULTS
    from lab import lessons as lessons_mod
else:
    from .engine import MiniLM, DEFAULTS
    from . import lessons as lessons_mod

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")
ROOT_DIR = os.path.dirname(HERE)
CORPUS_PATH = os.path.join(ROOT_DIR, "sample_text.txt")
HOST = os.environ.get("LLM_LAB_HOST", "127.0.0.1")
PORT = int(os.environ.get("LLM_LAB_PORT", "8000"))
MAX_EPOCHS = 2000
MAX_CONTEXT = 12
MAX_LENGTH = 600

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".json": "application/json; charset=utf-8",
}


class State:
    """Shared, thread-safe model and job registry."""

    def __init__(self):
        self.lock = threading.Lock()
        self.models: dict[str, MiniLM] = {}
        self.jobs: dict[str, dict] = {}
        self.default_text = ""
        self.corpus_path = ""
        self._local_models = threading.local()

    # -- corpus -------------------------------------------------------------
    def reload_corpus(self):
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            self.default_text = f.read()
        self.corpus_path = CORPUS_PATH

    def text_for(self, payload: dict) -> str:
        text = payload.get("text")
        if text is None or str(text).strip() == "":
            return self.default_text
        return str(text)

    # -- models -------------------------------------------------------------
    def build(self, payload: dict) -> MiniLM:
        """Construct a model from request parameters, with sane clamping."""
        text = self.text_for(payload)
        return MiniLM(
            text,
            context=_clamp_int(payload.get("context"), 1, MAX_CONTEXT,
                               DEFAULTS["context"]),
            embed=_clamp_int(payload.get("embed"), 1, 64, DEFAULTS["embed"]),
            hidden=_clamp_int(payload.get("hidden"), 1, 256, DEFAULTS["hidden"]),
            seed=_clamp_int(payload.get("seed"), 0, 10**9, DEFAULTS["seed"]),
        )

    def put_model(self, key: str, model: MiniLM):
        with self.lock:
            self.models[key] = model
            if len(self.models) > 64:  # keep memory bounded
                for k in list(self.models)[:-32]:
                    if k != key:
                        self.models.pop(k, None)

    def get_model(self, key: str | None):
        with self.lock:
            if key and key in self.models:
                return self.models[key]
            return self.models.get("default")

    # -- jobs ---------------------------------------------------------------
    def new_job(self) -> tuple[str, dict]:
        job = {
            "id": "", "status": "running", "epoch": 0, "epochs": 0,
            "loss": None, "start_loss": None, "elapsed": 0.0,
            "events": [], "error": None, "model_key": "",
        }
        with self.lock:
            job["id"] = "job_" + secrets.token_hex(6)
            job["model_key"] = job["id"]
            self.jobs[job["id"]] = job
            if len(self.jobs) > 32:
                for k in list(self.jobs)[:-16]:
                    if k != job["id"]:
                        self.jobs.pop(k, None)
        return job["id"], job

    def add_event(self, job: dict, event: dict):
        with self.lock:
            job["events"].append(event)

    def get_job(self, job_id: str):
        with self.lock:
            return self.jobs.get(job_id)


STATE = State()


def _clamp_int(value, lo, hi, default):
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _clamp_float(value, lo, hi, default):
    try:
        n = float(value)
    except (TypeError, ValueError):
        return default
    if n != n or n in (float("inf"), float("-inf")):  # NaN / inf guard
        return default
    return max(lo, min(hi, n))


# ---------------------------------------------------------------------------
# Training worker
# ---------------------------------------------------------------------------
def _train_worker(job: dict, model: MiniLM, epochs: int, lr: float):
    try:
        start_loss = model.average_loss()
        with STATE.lock:
            job["start_loss"] = start_loss
            job["loss"] = start_loss
            job["epochs"] = epochs
        STATE.add_event(job, {
            "type": "start", "start_loss": start_loss,
            "epochs": epochs, "lr": lr,
            "perplexity": math.exp(start_loss),
            "bits": start_loss / math.log(2),
        })

        milestone_epochs = sorted({max(1, int(round(epochs * f)))
                                   for f in (0.25, 0.5, 0.75, 1.0)})
        t0 = time.time()
        samples = {0.2: 0.3, 0.5: 0.7, 0.9: 1.0}
        for ev in model.train(epochs, lr):
            epoch, loss = ev["epoch"], ev["loss"]
            with STATE.lock:
                job["epoch"] = epoch
                job["loss"] = loss
                job["elapsed"] = time.time() - t0
            if not math.isfinite(loss):
                with STATE.lock:
                    job["events"].append({
                        "type": "error",
                        "message": ("The loss became %s -- the learning rate is "
                                    "too high and training diverged. Try a "
                                    "smaller lr."
                                    % ("nan" if loss != loss else loss)),
                    })
                    job["status"] = "diverged"
                break
            if epoch in milestone_epochs:
                STATE.add_event(job, {
                    "type": "milestone", "epoch": epoch,
                    "loss": loss, "elapsed": time.time() - t0,
                    "perplexity": math.exp(loss),
                    "bits": loss / math.log(2),
                    "sample": model.generate("the ", length=90, temperature=0.5),
                })
            if epoch % 5 == 0 or epoch == epochs:
                STATE.add_event(job, {
                    "type": "progress", "epoch": epoch, "loss": loss,
                    "elapsed": time.time() - t0,
                })
        else:
            elapsed = time.time() - t0
            final = job["loss"]
            samples = {str(t): model.generate("the ", length=160, temperature=t)
                       for t in (0.2, 0.5, 0.9)}
            with STATE.lock:
                job["elapsed"] = elapsed
                # Append the terminal event BEFORE flipping the status, so a
                # client can never observe "done" with the final event missing.
                job["events"].append({
                    "type": "done", "loss": final, "epochs": epochs,
                    "elapsed": elapsed,
                    "perplexity": math.exp(final) if math.isfinite(final) else None,
                    "bits": final / math.log(2) if math.isfinite(final) else None,
                    "samples": samples,
                })
                job["status"] = "done"
    except Exception as exc:  # pragma: no cover - defensive
        with STATE.lock:
            job["error"] = "%s: %s" % (type(exc).__name__, exc)
            job["events"].append({"type": "error", "message": job["error"]})
            job["status"] = "error"


# ---------------------------------------------------------------------------
# API handlers
# ---------------------------------------------------------------------------
def api_lessons(_payload):
    return {
        "lessons": lessons_mod.LESSONS,
        "exercises": lessons_mod.EXERCISES,
        "compare": [{"mini": a, "real": b} for a, b in lessons_mod.COMPARE],
        "defaults": DEFAULTS,
    }


def api_corpus(_payload):
    return {
        "text": STATE.default_text,
        "path": STATE.corpus_path,
        "chars": len(STATE.default_text.strip().lower()),
    }


def api_load(payload):
    model = STATE.build(payload)
    STATE.put_model("default", model)
    return _model_summary(model, "default")


def _model_summary(model: MiniLM, model_id: str):
    return {
        "model_id": model_id,
        "chars": len(model.text),
        "vocab_size": model.vocab_size,
        "n_examples": len(model.examples),
        "n_params": model.n_params,
        "context": model.context,
        "embed": model.embed,
        "hidden": model.hidden,
        "start_loss": None,
        "random_baseline": math.log(model.vocab_size),
        "vocab": model.vocab_table(),
        "char_counts": model.char_counts(),
        "counts": {
            "vocab": model.vocab_size,
            "params": model.n_params,
            "examples": len(model.examples),
            "chars": len(model.text),
        },
        "shapes": [
            {"name": "E", "label": "embeddings", "shape": [model.vocab_size, model.embed],
             "cols": "%d chars x %d numbers" % (model.vocab_size, model.embed)},
            {"name": "W1", "label": "input -> hidden", "shape": [model.hidden, model.context * model.embed],
             "cols": "%d hidden x %d inputs" % (model.hidden, model.context * model.embed)},
            {"name": "b1", "label": "hidden bias", "shape": [model.hidden],
             "cols": "%d numbers" % model.hidden},
            {"name": "W2", "label": "hidden -> output", "shape": [model.vocab_size, model.hidden],
             "cols": "%d scores x %d hidden" % (model.vocab_size, model.hidden)},
            {"name": "b2", "label": "output bias", "shape": [model.vocab_size],
             "cols": "%d numbers" % model.vocab_size},
        ],
    }


def api_examples(payload):
    context = _clamp_int(payload.get("context"), 1, MAX_CONTEXT,
                         DEFAULTS["context"])
    text = payload.get("text")
    if text is None or str(text).strip() == "":
        # Allow exploring examples for arbitrary typed text without retraining.
        text = payload.get("probe") or STATE.default_text
        model = MiniLM(text, context=context)
    else:
        model = MiniLM(str(text), context=context)
    count = _clamp_int(payload.get("count"), 1, 40, 8)
    offset = _clamp_int(payload.get("offset"), 0, max(0, len(model.examples) - 1), 0)
    return {
        "context": model.context,
        "total": len(model.examples),
        "rows": model.example_rows(count=count, offset=offset),
    }


def api_network(payload):
    model = STATE.build(payload)
    summary = _model_summary(model, "preview")
    return {
        "shapes": summary["shapes"],
        "n_params": model.n_params,
        "vocab_size": model.vocab_size,
        "context": model.context,
        "embed": model.embed,
        "hidden": model.hidden,
        "random_baseline": math.log(model.vocab_size),
    }


def api_embeddings(payload):
    model = STATE.get_model(payload.get("model_id"))
    if model is None or model.epochs_done == 0:
        return {"trained": False, "neighbours": []}
    return {"trained": True,
            "epochs_done": model.epochs_done,
            "neighbours": model.embedding_neighbours(k=3)}


def api_predict(payload):
    model = STATE.get_model(payload.get("model_id"))
    if model is None:
        return {"error": "no model"}
    seed = str(payload.get("seed", ""))[:200]
    k = _clamp_int(payload.get("k"), 1, 28, 6)
    return {"seed": seed, "context": model.to_context(seed),
            "context_label": "".join(
                model.id_to_char[c] for c in model.to_context(seed)),
            "all": model.forward(model.to_context(seed))[2],
            "predictions": model.next_char_probs(seed, k=k),
            "trained_epochs": model.epochs_done}


def api_generate(payload):
    model = STATE.get_model(payload.get("model_id"))
    if model is None:
        return {"error": "no model"}
    seed = str(payload.get("seed", "the "))[:200]
    length = _clamp_int(payload.get("length"), 1, MAX_LENGTH, 160)
    temperature = _clamp_float(payload.get("temperature"), 0.01, 5.0, 0.5)
    runs = _clamp_int(payload.get("runs"), 1, 3, 1)
    return {
        "seed": seed, "length": length, "temperature": temperature,
        "trained_epochs": model.epochs_done,
        "outputs": [model.generate(seed, length=length, temperature=temperature)
                    for _ in range(runs)],
    }


def api_train(payload):
    model = STATE.build(payload)
    epochs = _clamp_int(payload.get("epochs"), 1, MAX_EPOCHS, 60)
    lr = _clamp_float(payload.get("lr"), 0.01, 50.0, DEFAULTS["lr"])
    job_id, job = STATE.new_job()
    STATE.put_model(job["model_key"], model)
    thread = threading.Thread(target=_train_worker,
                              args=(job, model, epochs, lr), daemon=True)
    thread.start()
    return {"job_id": job_id, "epochs": epochs, "lr": lr,
            "n_examples": len(model.examples),
            "random_baseline": math.log(model.vocab_size)}


def api_job(query):
    job_id = (query.get("id") or [""])[0]
    frm = _clamp_int((query.get("from") or ["0"])[0], 0, 10**9, 0)
    job = STATE.get_job(job_id)
    if job is None:
        return {"error": "unknown job"}
    with STATE.lock:
        events = job["events"][frm:]
        nxt = len(job["events"])
        return {
            "id": job["id"], "status": job["status"],
            "epoch": job["epoch"], "epochs": job["epochs"],
            "loss": job["loss"], "start_loss": job["start_loss"],
            "elapsed": job["elapsed"], "next": nxt,
            "events": events, "error": job["error"],
            "model_id": job["model_key"],
        }


POST_ROUTES = {
    "load": api_load,
    "examples": api_examples,
    "network": api_network,
    "embeddings": api_embeddings,
    "predict": api_predict,
    "generate": api_generate,
    "train": api_train,
}
GET_ROUTES = {
    "lessons": api_lessons,
    "corpus": api_corpus,
    "job": api_job,
}


class Handler(BaseHTTPRequestHandler):
    server_version = "LLMLab/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the console clean
        if os.environ.get("LLM_LAB_VERBOSE"):
            super().log_message(fmt, *args)

    # -- helpers ------------------------------------------------------------
    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def _serve_static(self, path):
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        full = os.path.normpath(os.path.join(STATIC_DIR, rel))
        if not full.startswith(STATIC_DIR) or not os.path.isfile(full):
            self._send_json({"error": "not found"}, status=404)
            return
        ext = os.path.splitext(full)[1].lower()
        with open(full, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type",
                         CONTENT_TYPES.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # -- verbs --------------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            name = parsed.path[len("/api/"):]
            handler = GET_ROUTES.get(name)
            if handler is None:
                self._send_json({"error": "unknown endpoint"}, status=404)
                return
            try:
                if name == "job":
                    self._send_json(handler(parse_qs(parsed.query)))
                else:
                    self._send_json(handler(parse_qs(parsed.query)))
            except Exception as exc:
                self._send_json({"error": "%s: %s" % (type(exc).__name__, exc)},
                                status=500)
            return
        self._serve_static(parsed.path)

    def do_POST(self):
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            self._send_json({"error": "unknown endpoint"}, status=404)
            return
        name = parsed.path[len("/api/"):]
        handler = POST_ROUTES.get(name)
        if handler is None:
            self._send_json({"error": "unknown endpoint"}, status=404)
            return
        payload = self._read_json()
        try:
            self._send_json(handler(payload))
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json({"error": "%s: %s" % (type(exc).__name__, exc)},
                            status=500)


def main():
    STATE.reload_corpus()
    # A ready-to-use untrained model so the page has something immediately.
    STATE.put_model("default", STATE.build({}))
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = "http://%s:%d/" % (HOST, PORT)
    print("=" * 68)
    print("LLM Lab -- interactive companion for mini_llm.py")
    print("=" * 68)
    print("Open this in your browser:  " + url)
    print("Corpus: %s (%d characters)" % (CORPUS_PATH, len(STATE.default_text)))
    print("Press Ctrl+C to stop.")
    print("=" * 68)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
