"""engine.py -- the mini-LLM as a reusable, inspectable component.

This is a FAITHFUL port of the maths in ``mini_llm.py``.  Same embeddings,
same hidden layer, same tanh, same softmax, same cross-entropy loss, same
full-batch backpropagation, same temperature sampling.

The difference is what it is built *for*: instead of printing a fixed
walkthrough to the console, this module exposes the model's parts so a
learning app can look at them, step through training, and ask "what would
happen if ...?".

If you are reading this to learn, read ``mini_llm.py`` first -- it is the
intended teaching artefact, one screen per idea, with no plumbing in the way.
This file is the same algorithm with the plumbing attached.
"""

from __future__ import annotations

import math
import random

# Defaults chosen to match a default run of mini_llm.py.
DEFAULTS = {
    "context": 3,
    "embed": 6,
    "hidden": 24,
    "epochs": 250,
    "lr": 5.0,
    "seed": 0,
}


def show_char(c: str) -> str:
    """A readable label for a character (spaces and newlines are invisible)."""
    if c == " ":
        return "[space]"
    if c == "\n":
        return "\\n"
    if c == "\t":
        return "\\t"
    return "'" + c + "'"


class MiniLM:
    """A character-level language model, small enough to read completely.

    Train it with :meth:`train`, then use :meth:`predict` (what comes next?)
    and :meth:`generate` (write something).
    """

    def __init__(self, text: str, context: int = 3, embed: int = 6,
                 hidden: int = 24, seed: int = 0):
        # --- 1. the corpus -------------------------------------------------
        self.text = text.strip().lower()
        if len(self.text) <= context:
            raise ValueError(
                "corpus is too short: it needs more than 'context' characters")

        # --- 2. the vocabulary ---------------------------------------------
        self.chars = sorted(set(self.text))
        self.vocab_size = len(self.chars)
        self.char_to_id = {c: i for i, c in enumerate(self.chars)}
        self.id_to_char = {i: c for i, c in enumerate(self.chars)}

        # --- 3. the text as numbers + the sliding-window examples ----------
        self.context = context
        self.ids = [self.char_to_id[c] for c in self.text]
        self.examples = [
            (self.ids[i - context:i], self.ids[i])
            for i in range(context, len(self.ids))
        ]

        # --- 4. the learnable parameters (all random to start) -------------
        self.embed = embed
        self.hidden = hidden
        rng = random.Random(seed)
        # Draw order matches mini_llm.py exactly, so a fresh model has the
        # same starting loss as a fresh `python mini_llm.py` run.
        self.E = self._random_matrix(rng, self.vocab_size, embed)
        self.W1 = self._random_matrix(rng, hidden, context * embed)
        self.b1 = [0.0] * hidden
        self.W2 = self._random_matrix(rng, self.vocab_size, hidden)
        self.b2 = [0.0] * self.vocab_size

        # A separate RNG for generation so that sampling does not disturb the
        # reproducibility of training runs.
        self._sampler = random.Random(seed + 1)

        # Epochs trained so far (for the UI).
        self.epochs_done = 0

    # -- construction helpers ----------------------------------------------
    @staticmethod
    def _random_matrix(rng, rows, cols, scale=0.1):
        return [[rng.uniform(-scale, scale) for _ in range(cols)]
                for _ in range(rows)]

    @property
    def n_params(self) -> int:
        return (self.vocab_size * self.embed
                + self.hidden * self.context * self.embed + self.hidden
                + self.vocab_size * self.hidden + self.vocab_size)

    # -- the maths ----------------------------------------------------------
    @staticmethod
    def softmax(scores):
        """Raw scores -> probabilities that sum to 1 (max-subtracted for safety)."""
        biggest = max(scores)
        exps = [math.exp(s - biggest) for s in scores]
        total = sum(exps)
        return [e / total for e in exps]

    def forward(self, context):
        """characters -> embeddings -> hidden (tanh) -> scores -> probabilities."""
        # embeddings glued end to end
        x = []
        for c in context:
            x.extend(self.E[c])

        # hidden layer
        pre_h = []
        for h in range(self.hidden):
            total = self.b1[h]
            row = self.W1[h]
            for j in range(len(x)):
                total += row[j] * x[j]
            pre_h.append(total)
        h = [math.tanh(v) for v in pre_h]

        # output layer
        scores = []
        for v in range(self.vocab_size):
            total = self.b2[v]
            row = self.W2[v]
            for k in range(self.hidden):
                total += row[k] * h[k]
            scores.append(total)

        return x, h, self.softmax(scores)

    def loss_of(self, context, target) -> float:
        _, _, probs = self.forward(context)
        return -math.log(probs[target] + 1e-9)

    def average_loss(self) -> float:
        total = 0.0
        for context, target in self.examples:
            total += self.loss_of(context, target)
        return total / len(self.examples)

    # -- text <-> context ---------------------------------------------------
    def to_context(self, seed: str):
        """Turn any string into exactly `context` character ids."""
        space = self.char_to_id.get(" ", 0)
        ctx = [self.char_to_id.get(ch, space) for ch in seed.lower()]
        if len(ctx) < self.context:
            ctx = [space] * (self.context - len(ctx)) + ctx
        return ctx[-self.context:]

    # -- generation ---------------------------------------------------------
    def generate(self, seed: str, length: int = 120,
                 temperature: float = 0.5) -> str:
        """Autoregressive sampling: predict a char, append it, repeat."""
        if temperature <= 0:
            temperature = 1e-6
        context = self.to_context(seed)
        result = list(seed)
        for _ in range(length):
            _, _, probs = self.forward(context)
            adjusted = [p ** (1.0 / temperature) for p in probs]
            total = sum(adjusted)
            adjusted = [a / total for a in adjusted]
            next_id = self._sampler.choices(
                range(self.vocab_size), weights=adjusted)[0]
            result.append(self.id_to_char[next_id])
            context = context[1:] + [next_id]
        return "".join(result)

    def next_char_probs(self, seed: str, k: int = 8):
        """The model's top-k guesses for what comes after `seed`."""
        _, _, probs = self.forward(self.to_context(seed))
        ranked = sorted(range(self.vocab_size),
                        key=lambda i: probs[i], reverse=True)[:k]
        return [{"id": i, "char": self.id_to_char[i],
                 "label": show_char(self.id_to_char[i]),
                 "prob": probs[i]} for i in ranked]

    # -- training -----------------------------------------------------------
    def train(self, epochs: int, lr: float):
        """Full-batch gradient descent.

        A generator: after every epoch it yields a small progress dict, so a
        caller can stream a progress bar.  This is the same loop as STEP 5 of
        ``mini_llm.py`` -- read that for the annotated version.
        """
        n = len(self.examples)
        for epoch in range(epochs):
            dE = [[0.0] * self.embed for _ in range(self.vocab_size)]
            dW1 = [[0.0] * (self.context * self.embed)
                   for _ in range(self.hidden)]
            db1 = [0.0] * self.hidden
            dW2 = [[0.0] * self.hidden for _ in range(self.vocab_size)]
            db2 = [0.0] * self.vocab_size

            total_loss = 0.0
            for context, target in self.examples:
                x, h, probs = self.forward(context)
                total_loss += -math.log(probs[target] + 1e-9)

                # output layer: softmax + cross-entropy collapses to this
                dscores = probs[:]
                dscores[target] -= 1.0
                for v in range(self.vocab_size):
                    dv = dscores[v]
                    db2[v] += dv
                    drow2 = dW2[v]
                    for k in range(self.hidden):
                        drow2[k] += dv * h[k]

                # error flows back into the hidden layer
                dh = [0.0] * self.hidden
                for v in range(self.vocab_size):
                    dv = dscores[v]
                    row2 = self.W2[v]
                    for k in range(self.hidden):
                        dh[k] += dv * row2[k]
                # through tanh: d/dx tanh(x) = 1 - tanh(x)^2
                dpre_h = [dh[k] * (1 - h[k] * h[k])
                          for k in range(self.hidden)]

                # input -> hidden weights
                for k in range(self.hidden):
                    dk = dpre_h[k]
                    db1[k] += dk
                    drow1 = dW1[k]
                    for j in range(len(x)):
                        drow1[j] += dk * x[j]

                # and into each context character's embedding
                for j in range(len(x)):
                    dxj = 0.0
                    for k in range(self.hidden):
                        dxj += dpre_h[k] * self.W1[k][j]
                    pos = j // self.embed
                    dim = j % self.embed
                    dE[context[pos]][dim] += dxj

            # apply the averaged gradients
            for v in range(self.vocab_size):
                ev = self.E[v]
                dev = dE[v]
                for d in range(self.embed):
                    ev[d] -= lr * dev[d] / n
                w2v = self.W2[v]
                dw2v = dW2[v]
                for k in range(self.hidden):
                    w2v[k] -= lr * dw2v[k] / n
                self.b2[v] -= lr * db2[v] / n
            for k in range(self.hidden):
                w1k = self.W1[k]
                dw1k = dW1[k]
                for j in range(self.context * self.embed):
                    w1k[j] -= lr * dw1k[j] / n
                self.b1[k] -= lr * db1[k] / n

            self.epochs_done += 1
            yield {"epoch": epoch + 1, "loss": total_loss / n}

    # -- introspection for the UI ------------------------------------------
    def vocab_table(self):
        return [{"id": i, "char": self.id_to_char[i],
                 "label": show_char(self.id_to_char[i])}
                for i in range(self.vocab_size)]

    def char_counts(self):
        counts = {}
        for c in self.text:
            counts[c] = counts.get(c, 0) + 1
        total = len(self.text)
        return sorted(
            ({"id": self.char_to_id[c], "char": c, "label": show_char(c),
              "count": n, "freq": n / total}
             for c, n in counts.items()),
            key=lambda d: -d["count"])

    def example_rows(self, count: int = 8, offset: int = 0):
        """Readable rows for the first `count` sliding-window examples."""
        rows = []
        for context, target in self.examples[offset:offset + count]:
            ctx_str = "".join(self.id_to_char[c] for c in context)
            rows.append({
                "context": ctx_str,
                "context_label": repr(ctx_str),
                "target": self.id_to_char[target],
                "target_label": show_char(self.id_to_char[target]),
                "whole": (ctx_str + self.id_to_char[target]).replace("\n", "\\n"),
                "ids": context,
                "target_id": target,
            })
        return rows

    def embedding_neighbours(self, k: int = 3):
        """For each character, the characters whose learned embedding is closest.

        Nothing told the model that vowels are alike -- gradient descent
        discovered it.  This is the payoff of using embeddings at all.
        """
        def dot(a, b):
            return sum(x * y for x, y in zip(a, b))

        def norm(a):
            return math.sqrt(sum(x * x for x in a)) or 1e-9

        out = []
        for i in range(self.vocab_size):
            sims = []
            for j in range(self.vocab_size):
                if i == j:
                    continue
                sims.append((dot(self.E[i], self.E[j])
                             / (norm(self.E[i]) * norm(self.E[j])), j))
            sims.sort(reverse=True)
            out.append({
                "char": self.id_to_char[i],
                "label": show_char(self.id_to_char[i]),
                "neighbours": [show_char(self.id_to_char[j])
                               for _, j in sims[:k]],
            })
        return out


def load_corpus(path) -> str:
    """Read the training text from disk (same file mini_llm.py uses)."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
