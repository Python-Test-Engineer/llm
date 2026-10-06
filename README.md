# A very small LLM, from scratch in plain Python

This folder is a tiny, readable "language model" — the same core idea behind
GPT, just a billion times smaller. No numpy, no PyTorch: every multiply and
add is visible in `mini_llm.py`.

## Two ways to learn with this

- **`uv run llm_lab.py`** — **LLM Lab**, an interactive web app. Walk the seven
  stages as lessons, turn every knob (learning rate, context, hidden size),
  watch the loss curve fall, and see the model's next-character predictions
  and generated text. This is the recommended starting point.
- **`uv run mini_llm.py`** — the original console walkthrough, heavily
  commented, printing each step and ending in an interactive prompt. Read this
  to understand the code; `EXPLAINER.md` explains it line by line.

## Files

- `mini_llm.py` — the whole thing, heavily commented.
- `sample_text.txt` — the entire training corpus (a short story about a fox),
  2,396 characters. Edit this and re-run to train on your own text.
- `EXPLAINER.md` — a full student tutorial: how and why every part works.
- `llm_lab.py` — start the interactive lab app.
- `lab/` — the lab: `engine.py` (the model, a faithful port of
  `mini_llm.py`), `lessons.py` (tutorial content), `server.py` (stdlib-only
  HTTP server), `static/` (the page).
- `tests/` — checks that prove the lab's model is the *same* model as
  `mini_llm.py`, plus end-to-end tests of the web API.

## Run the interactive lab

```
uv run llm_lab.py
```

It prints a URL (default <http://127.0.0.1:8000/>) — open it in your browser.
Nothing to install: the server, the model and the page are all standard
library plus vanilla JavaScript.

Use `LLM_LAB_PORT=8123` to change the port. In the app:

1. **Corpus & vocabulary** — paste your own text and see the vocabulary change.
2. **Examples** — type a sentence and see exactly which input/target pairs the
   model would learn from.
3. **Network** — change `context` / `embed` / `hidden` and watch the parameter
   count; after training, see which characters the embeddings decided are alike.
4. **Train** — a live loss curve (with the `ln(vocab)` random-guess line), a
   progress bar, and the model's own writing at the 25 / 50 / 75 / 100% marks.
   Set the learning rate to 12 and watch the loss explode — the app tells you
   when training has diverged.
5. **Generate** — the model's top next-character probabilities for any context,
   and text generation with a temperature slider.

## Run the console version

From the `nn-master` folder:

```
uv run .\llm\mini_llm.py
```

(or from inside `llm/`: `python mini_llm.py`)

It takes about **160 seconds** by default and prints a guided walkthrough:

1. **STEP 1/6** — reads the text and shows the opening.
2. **STEP 2/6** — the vocabulary: every character and the number it maps to.
3. **STEP 3/6** — the first training examples, e.g. `'the' -> [space]`.
4. **STEP 4/6** — the network shape and how many learnable numbers it has.
5. **STEP 5/6** — a live progress bar with the loss falling, plus a sample of
   the model's own writing at 25% / 50% / 75% so you can *see* it improve.
6. **STEP 6/6** — finished samples at three temperatures, a table of the
   model's **top next-character guesses**, then an **interactive prompt**:

```
seed> start text: the little
```

Type a few characters and the model continues the sentence. Enter blank to quit.

Expected quality: the loss falls from ~3.33 to ~1.31. The starting value is
`ln(28) = 3.3322`, which is exactly the score of pure random guessing — a
useful sanity check explained in `EXPLAINER.md`. The model produces
fox-story-like text such as:

```
the fox was hed hith a will fall of the waseight. the river
and the fox was hed was river and was rose fox wats an
```

It has clearly learned the vocabulary, word shapes, and spacing — but it is a
tiny model, so it invents plausible-looking nonsense words rather than
perfect English. That gap is exactly what "bigger model + more data" fixes.

## Command-line options

```
uv run .\llm\mini_llm.py [--epochs 250] [--lr 5.0] [--hidden 24]
                         [--embed 6] [--context 3] [--quiet]
```

- `--epochs`   how many passes over the training data (more = better, slower)
- `--lr`       learning rate: how big each weight nudge is
- `--hidden`   neurons in the hidden layer (the model's "memory")
- `--embed`    numbers used to describe each character
- `--context`  how many previous characters the model may look at
- `--quiet`    skip the interactive prompt (useful for piping output)

Example — a fast, rough run:

```
uv run .\llm\mini_llm.py --epochs 60 --lr 5
```

## What a language model actually is

A language model answers one question, over and over:

> **Given the last few characters, what character comes next?**

If you can answer that well, you can *generate* text: predict a character,
append it, predict the next, repeat. That is literally all GPT does.

## What the code does, step by step

1. **Vocabulary** — every distinct character gets a number (`'a' -> 4`).
2. **Training examples** — sliding windows. From `"the cat"`:
   `t,h,e -> ' '`, `h,e,' ' -> 'c'`, `e,' ','c' -> 'a'`, ...
3. **The network** — four learnable pieces:
   - `E` embeddings: turn each character into a small list of numbers
   - `W1`/`b1`: input → hidden layer (with `tanh`)
   - `W2`/`b2`: hidden layer → one score per vocabulary character
   - `softmax`: scores → probabilities that sum to 1
4. **Loss** — `-log(probability of the correct character)`. Low = good.
5. **Backpropagation** — work out, for every weight, which direction reduces
   the loss, and nudge it a little. Repeat `--epochs` times.
6. **Generate** — start from a seed like `"the "`, sample the next character,
   feed it back in, repeat.

## Things to try (this is where the learning happens)

- **Learning rate is the sharpest knob.** On this corpus, `--lr 5` works well;
  `--lr 8` makes the loss *explode* (diverge). Try `--lr 1` to watch it learn
  slowly, then `--lr 8` to watch it break. (This is why the code divides the
  full-batch gradient by the number of examples — with a bigger corpus each
  step is smaller, so you need a bigger learning rate.)
- **`--epochs`** — 60 is rough, 250 is the default, 600 is cleaner but slower.
- **`--context`** — try 1 (just the previous character) vs 5 (more history).
- **`--hidden`** — more neurons = more capacity.
- **`--temperature`** (edit it in `generate`, or the interactive runs) — 0.2 is
  repetitive and safe, 0.9 is wilder. This is the same knob real chat models
  expose.
- **Your own text** — paste a short story into `sample_text.txt`. More text
  needs more epochs.

## Why it's "small"

Real LLMs differ in three ways only: they use **words/tokens** instead of
characters, a **transformer** (attention) instead of a tiny feed-forward
network, and **billions of parameters + huge data** instead of a few thousand
parameters + a few paragraphs. The training loop — predict, measure loss,
backpropagate, nudge — is the same one you're reading here.

## Run the tests

```
uv run --with pytest pytest tests/ -q      # or, with no installs:
uv run tests/test_engine.py
uv run tests/test_server.py
```

`tests/test_engine.py` checks the arithmetic and the tutorial's numbers, and
runs `mini_llm.py` itself to confirm the lab's model produces the *same*
untrained loss (3.3320) — so the app can never silently drift from the script
it teaches. `tests/test_server.py` starts the real web server and exercises
every API endpoint end to end, including a full training job.

## Editing the app

The model lives in exactly one place: `lab/engine.py`. `mini_llm.py` stays
untouched and independently readable on purpose — it is the teaching artefact,
so it is not refactored into the app. If you change the maths, change it in
both and run the tests; the parity test will catch a mismatch.
