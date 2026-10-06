r"""
mini_llm.py  --  a very small "language model" written in plain Python.

RUN IT WITH:   uv run .\llm\mini_llm.py        (from the nn-master folder)
           or: python mini_llm.py              (from inside the llm folder)

WHAT IS A LANGUAGE MODEL?
-------------------------
A language model predicts the NEXT character given the characters that came
before it.  That is the whole game.  A "Large Language Model" (LLM) like GPT
does exactly this -- it is just far bigger and trained on far more text.
If you can predict the next character well, you can *generate* text: predict
a character, stick it on the end, predict the next one, and repeat.

WHAT THIS FILE DOES (the big picture)
-------------------------------------
  1. Read a sample of text.
  2. Turn each character into a number (its index in a "vocabulary").
  3. Build training examples of the form:
         "given the last CONTEXT characters, predict the next one".
  4. Create a small neural network with learnable weights.
  5. TRAIN it: show it examples, measure how wrong it is, nudge the weights.
  6. GENERATE new text one character at a time.

Everything uses plain Python (no numpy, no PyTorch) so you can read every
single multiply and add.  That makes it slow, but it makes the idea clear.

The console output walks you through each step so you can watch the model
learn: it prints the vocabulary, a few example training pairs, the network
shape, a live progress bar with the loss falling, sample text at milestones,
the model's "top guesses" for some contexts, and finally an interactive
prompt where you can type your own starting text.
"""

import argparse
import math
import os
import random
import sys
import textwrap
import time

random.seed(0)  # so every run gives the same result while you are learning


# ==========================================================================
# 1. SETTINGS  --  change these and watch what happens
# ==========================================================================
# CONTEXT          how many previous characters the model is allowed to see
# EMBED_DIM        each character is turned into a list of this many numbers
# HIDDEN           how many "neurons" in the network's middle layer
# EPOCHS           how many times we sweep over all the training examples
# LEARNING_RATE    how big each weight nudge is
# These are the DEFAULTS. You can also set them from the command line, e.g.
#     uv run .\llm\mini_llm.py --epochs 200 --lr 3 --hidden 24
def parse_args():
    p = argparse.ArgumentParser(
        description="Train a tiny character-level language model from scratch.")
    p.add_argument("--epochs", type=int, default=250,
                   help="how many passes over all the training examples")
    p.add_argument("--lr", type=float, default=5.0,
                   help="learning rate: how big each weight nudge is")
    p.add_argument("--hidden", type=int, default=24,
                   help="number of neurons in the hidden layer")
    p.add_argument("--embed", type=int, default=6,
                   help="numbers used to describe each character")
    p.add_argument("--context", type=int, default=3,
                   help="how many previous characters the model sees")
    p.add_argument("--quiet", action="store_true",
                   help="skip the interactive prompt at the end")
    return p.parse_args()

ARGS = parse_args()
CONTEXT = ARGS.context
EMBED_DIM = ARGS.embed
HIDDEN = ARGS.hidden
EPOCHS = ARGS.epochs
LEARNING_RATE = ARGS.lr


# ==========================================================================
# Small helpers that make the console output easy to read
# ==========================================================================
LINE = "=" * 72

def section(title):
    """Print a big, obvious heading for a stage of the program."""
    print("\n" + LINE)
    print(title)
    print(LINE)

def info(msg):
    print("   " + msg)

def wrap(text, indent="      "):
    return textwrap.fill(text, width=68, initial_indent=indent,
                         subsequent_indent=indent)

def show_char(i):
    """Show a character id in a readable way (spaces/newlines are visible)."""
    c = id_to_char[i]
    if c == " ":
        return "[space]"
    if c == "\n":
        return "\\n"
    return "'" + c + "'"


# ==========================================================================
# 2. READ THE SAMPLE TEXT
# ==========================================================================
# This is our ENTIRE training corpus. Real LLMs use billions of characters;
# we use a few paragraphs so it is easy to follow. Edit sample_text.txt and
# re-run to train on your own writing.
section("STEP 1 / 6  --  Reading the training text")

# Build the path to sample_text.txt relative to THIS file, so the script
# works no matter which folder you run it from.
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "sample_text.txt"), "r", encoding="utf-8") as f:
    text = f.read().strip().lower()

info("Loaded sample_text.txt")
info("   %d characters, %d lines" % (len(text), text.count("\n") + 1))
info("")
info("Opening of the corpus:")
info(wrap(text[:150].replace("\n", " ")) + " ...")


# ==========================================================================
# 3. BUILD THE VOCABULARY  (character -> number, and back again)
# ==========================================================================
# A neural network only does maths on numbers, so we map every distinct
# character in the text to an integer id.
section("STEP 2 / 6  --  Building the vocabulary")

chars = sorted(set(text))               # every unique character, sorted
vocab_size = len(chars)                 # how many distinct characters
char_to_id = {c: i for i, c in enumerate(chars)}   # 'a' -> 5, etc.
id_to_char = {i: c for i, c in enumerate(chars)}   # 5 -> 'a', etc.

info("Found %d distinct characters. Each one gets a number:" % vocab_size)
info("")
for start in range(0, vocab_size, 5):
    row = ""
    for i in range(start, min(start + 5, vocab_size)):
        row += "%-10s= %-3d" % (show_char(i), i)
    info(row)

# Turn the whole text into a list of integer ids, ready for the maths.
ids = [char_to_id[c] for c in text]


# ==========================================================================
# 4. BUILD TRAINING EXAMPLES
# ==========================================================================
# Sliding window: for each position, the CONTEXT characters before it are the
# INPUT, and the character at that position is the TARGET we want to predict.
#
#   text:   "the cat"
#   input(3): 't','h','e'   target: ' '
#   input(3): 'h','e',' '   target: 'c'
#   input(3): 'e',' ','c'   target: 'a'
#   ...and so on.
section("STEP 3 / 6  --  Building training examples")

examples = []                      # each item is (context_ids, target_id)
for i in range(CONTEXT, len(ids)):
    context = ids[i - CONTEXT:i]   # the CONTEXT characters just before i
    target = ids[i]                # the character we want to predict
    examples.append((context, target))

info("Each example is %d characters of input plus the next character as the"
     % CONTEXT)
info("answer. Sliding a window across the text gives %d examples."
     % len(examples))
info("")
info("The first 8 examples the model will learn from:")
info("")
info("     %-14s %-9s %s" % ("INPUT", "TARGET", "SO THE SENTENCE..."))
info("     %-14s %-9s %s" % ("-" * 13, "-" * 8, "-" * 22))
for context, target in examples[:8]:
    ctx_str = "".join(id_to_char[c] for c in context)
    whole = ctx_str + id_to_char[target]
    info("     %-14s %-9s '%s'"
         % (repr(ctx_str), show_char(target), whole.replace("\n", "\\n")))


# ==========================================================================
# 5. CREATE THE NETWORK'S WEIGHTS
# ==========================================================================
section("STEP 4 / 6  --  Building the neural network")

# Helper: make a matrix of small random numbers (rows x cols).
def random_matrix(rows, cols, scale=0.1):
    return [[random.uniform(-scale, scale) for _ in range(cols)]
            for _ in range(rows)]

# The pieces we will learn (all start random, all get nudged during training):
#   E   : the "embeddings". One short list of numbers per character. Similar
#         characters end up with similar lists -- the network decides.
#   W1  : input layer  -> hidden layer   (HIDDEN rows, CONTEXT*EMBED_DIM cols)
#   b1  : hidden layer bias              (one number per hidden neuron)
#   W2  : hidden layer -> output layer   (vocab_size rows, HIDDEN cols)
#   b2  : output layer bias              (one number per vocabulary character)
E  = random_matrix(vocab_size, EMBED_DIM)
W1 = random_matrix(HIDDEN, CONTEXT * EMBED_DIM)
b1 = [0.0] * HIDDEN
W2 = random_matrix(vocab_size, HIDDEN)
b2 = [0.0] * vocab_size

n_params = (vocab_size * EMBED_DIM) + (HIDDEN * CONTEXT * EMBED_DIM) + HIDDEN \
           + (vocab_size * HIDDEN) + vocab_size

info("Architecture (a tiny feed-forward network):")
info("")
info("   %d chars -> embeddings (%d numbers each) -> %d numbers ->"
     % (CONTEXT, EMBED_DIM, CONTEXT * EMBED_DIM))
info("   %d hidden neurons (tanh) -> %d output scores -> softmax"
     % (HIDDEN, vocab_size))
info("")
info("   E  embeddings      : %d x %d" % (vocab_size, EMBED_DIM))
info("   W1 input->hidden   : %d x %d" % (HIDDEN, CONTEXT * EMBED_DIM))
info("   W2 hidden->output  : %d x %d" % (vocab_size, HIDDEN))
info("   total learnable numbers (parameters): %d" % n_params)
info("")
info("For comparison, real LLMs have billions to trillions of parameters.")


# ==========================================================================
# 6. THE MATHS HELPERS
# ==========================================================================
def softmax(scores):
    """Turn raw output scores into probabilities that sum to 1."""
    biggest = max(scores)                     # subtract max: keeps exp() safe
    exps = [math.exp(s - biggest) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def forward(context):
    """
    Push a context through the network and return everything we need to
    (a) measure the error and (b) compute gradients later.

    Think of it as a signal travelling left to right:
        characters -> embeddings -> hidden layer -> output scores -> probabilities
    """
    # 6a. EMBEDDING LOOKUP: glue the CONTEXT character embeddings into one list.
    x = []
    for c in context:
        x.extend(E[c])

    # 6b. HIDDEN LAYER: each hidden neuron looks at the whole of x.
    #     pre_h[h] = sum(W1[h][j] * x[j]) + b1[h]
    pre_h = []
    for h in range(HIDDEN):
        total = b1[h]
        row = W1[h]
        for j in range(len(x)):
            total += row[j] * x[j]
        pre_h.append(total)

    # 6c. ACTIVATION: squash each hidden value through tanh so it is between
    #     -1 and 1. This non-linearity lets the network learn patterns.
    h = [math.tanh(v) for v in pre_h]

    # 6d. OUTPUT LAYER: one score per character in the vocabulary.
    scores = []
    for v in range(vocab_size):
        total = b2[v]
        row = W2[v]
        for k in range(HIDDEN):
            total += row[k] * h[k]
        scores.append(total)

    # 6e. SOFTMAX: turn the scores into a probability for every character.
    probs = softmax(scores)
    return x, h, probs


def average_loss():
    """Measure how wrong the network currently is, over ALL examples."""
    total = 0.0
    for context, target in examples:
        _, _, probs = forward(context)
        total += -math.log(probs[target] + 1e-9)   # -log(prob we gave the answer)
    return total / len(examples)


def to_context(seed):
    """Turn any user string into exactly CONTEXT character ids.
    Unknown characters become spaces; short seeds are padded on the left."""
    space = char_to_id.get(" ", 0)
    ctx = [char_to_id.get(ch, space) for ch in seed.lower()]
    if len(ctx) < CONTEXT:
        ctx = [space] * (CONTEXT - len(ctx)) + ctx
    return ctx[-CONTEXT:]


def generate(seed, length=100, temperature=0.5):
    """
    Produce new text. Start with a `seed` string, then repeatedly ask the
    network "what comes next?" and feed the answer back in.

    temperature controls craziness:
      ~0.1 -> very safe and repetitive
      ~1.0 -> more varied and surprising
    """
    context = to_context(seed)
    result = list(seed)
    for _ in range(length):
        _, _, probs = forward(context)
        # Blend in temperature: smaller = safer, larger = wilder.
        adjusted = [p ** (1.0 / temperature) for p in probs]
        total = sum(adjusted)
        adjusted = [a / total for a in adjusted]
        # Pick the next character at random, weighted by those probabilities.
        next_id = random.choices(range(vocab_size), weights=adjusted)[0]
        result.append(id_to_char[next_id])
        context = context[1:] + [next_id]   # slide the window forward
    return "".join(result)


# ==========================================================================
# 7. TRAINING LOOP
# ==========================================================================
# We collect all the gradients for one full sweep, then apply them once.
# (This is called "full-batch" gradient descent -- simple and stable.)
section("STEP 5 / 6  --  Training the network")

# Show a sample from the UNtrained (random) network, for comparison later.
info("Before training the network is random. It generates gibberish:")
info(wrap(generate("the ", length=100, temperature=0.6)))
info("")

start_loss = average_loss()
info("Learning rate : %.2f   (how big each weight nudge is)" % LEARNING_RATE)
info("Epochs        : %d     (one epoch = one pass over all examples)" % EPOCHS)
info("Loss before training: %.4f   (lower is better)" % start_loss)
info("")

BAR_WIDTH = 34
MILESTONES = {max(1, int(EPOCHS * f)) - 1 for f in (0.25, 0.5, 0.75)}

def draw_progress(done, loss, elapsed):
    """Redraw the progress bar in place on one line."""
    frac = done / EPOCHS
    filled = int(BAR_WIDTH * frac)
    bar = "#" * filled + "." * (BAR_WIDTH - filled)
    sys.stdout.write("\r   [%s] %5.1f%%  epoch %d/%d  loss %.4f  eta %3.0fs"
                     % (bar, frac * 100, done, EPOCHS, loss, elapsed))
    sys.stdout.flush()

def clear_progress():
    sys.stdout.write("\r" + " " * 76 + "\r")
    sys.stdout.flush()

start_time = time.time()
avg = start_loss
for epoch in range(EPOCHS):
    # Fresh gradient buckets for this epoch.
    dE  = [[0.0] * EMBED_DIM for _ in range(vocab_size)]
    dW1 = [[0.0] * (CONTEXT * EMBED_DIM) for _ in range(HIDDEN)]
    db1 = [0.0] * HIDDEN
    dW2 = [[0.0] * HIDDEN for _ in range(vocab_size)]
    db2 = [0.0] * vocab_size

    total_loss = 0.0

    for context, target in examples:
        # --- forward pass: what does the network currently predict? ---
        x, h, probs = forward(context)

        # --- loss: how surprised are we by the correct character? ---
        total_loss += -math.log(probs[target] + 1e-9)

        # --- backward pass (backpropagation) ---
        # Work out, for every weight, which way to nudge it to lower the loss.
        # d... means "derivative of the loss with respect to ...".

        # 1) Output scores. With softmax + cross-entropy this simplifies to
        #    (predicted_probability - 1_for_correct_char).
        dscores = probs[:]
        dscores[target] -= 1.0

        for v in range(vocab_size):
            dv = dscores[v]
            db2[v] += dv
            drow2 = dW2[v]
            for k in range(HIDDEN):
                drow2[k] += dv * h[k]

        # 2) Hidden layer. Push the output error back through W2.
        dh = [0.0] * HIDDEN
        for v in range(vocab_size):
            dv = dscores[v]
            row2 = W2[v]
            for k in range(HIDDEN):
                dh[k] += dv * row2[k]

        # 3) Through the tanh activation: d/dx tanh(x) = 1 - tanh(x)^2.
        dpre_h = [dh[k] * (1 - h[k] * h[k]) for k in range(HIDDEN)]

        # 4) Input -> hidden weights and biases.
        for k in range(HIDDEN):
            dk = dpre_h[k]
            db1[k] += dk
            drow1 = dW1[k]
            for j in range(len(x)):
                drow1[j] += dk * x[j]

        # 5) Embeddings. Push the error back into x, then into each context
        #    character's embedding row.
        for j in range(len(x)):
            dxj = 0.0
            for k in range(HIDDEN):
                dxj += dpre_h[k] * W1[k][j]
            pos = j // EMBED_DIM        # which context character
            dim = j % EMBED_DIM         # which number inside its embedding
            dE[context[pos]][dim] += dxj

    # --- apply all the collected gradients (divide by count to average) ---
    n = len(examples)
    for v in range(vocab_size):
        for d in range(EMBED_DIM):
            E[v][d] -= LEARNING_RATE * dE[v][d] / n
        for k in range(HIDDEN):
            W2[v][k] -= LEARNING_RATE * dW2[v][k] / n
        b2[v] -= LEARNING_RATE * db2[v] / n
    for k in range(HIDDEN):
        for j in range(CONTEXT * EMBED_DIM):
            W1[k][j] -= LEARNING_RATE * dW1[k][j] / n
        b1[k] -= LEARNING_RATE * db1[k] / n

    avg = total_loss / n
    draw_progress(epoch + 1, avg, time.time() - start_time)

    # At a few milestones, pause and show what the model writes so far.
    if epoch in MILESTONES:
        clear_progress()
        info("   [after %d of %d epochs] \"%s\""
             % (epoch + 1, EPOCHS,
                generate("the ", length=70, temperature=0.5).replace("\n", " ")))

clear_progress()
info("")
info("Final loss: %.4f   (started at %.4f)" % (avg, start_loss))
info("Trained %d epochs in %.1f seconds." % (EPOCHS, time.time() - start_time))


# ==========================================================================
# 8. GENERATE TEXT
# ==========================================================================
section("STEP 6 / 6  --  Generating new text")

info("The model now writes text in the style of the sample, one character")
info("at a time. Lower temperature = safer; higher = wilder.")
info("")
for temp in (0.2, 0.5, 0.9):
    info("temperature %.1f:" % temp)
    info(wrap(generate("the ", length=110, temperature=temp)))
    info("")


# ==========================================================================
# 9. WHAT IS THE MODEL THINKING?  (top next-character guesses)
# ==========================================================================
def show_predictions(context_str, k=5):
    """Show the model's TOP next-character guesses for a short context."""
    _, _, probs = forward(to_context(context_str))
    ranked = sorted(range(vocab_size), key=lambda i: probs[i], reverse=True)[:k]
    info("   after %-9s the model expects:" % ("'" + context_str + "'"))
    for i in ranked:
        bars = "#" * int(round(probs[i] * 40))
        info("      %-9s %5.1f%%  %s" % (show_char(i), probs[i] * 100, bars))

info("What the model predicts next, for a few starting contexts:")
info("")
for ctx in ("the", "fox", " th", "riv"):
    show_predictions(ctx)
    info("")


# ==========================================================================
# 10. INTERACTIVE: let the user drive the model
# ==========================================================================
info("Interactive: type a few starting characters and the model continues")
info("the text. Press Enter on an empty line (or type 'quit') to stop.")

# Only run the interactive loop if the user is at a keyboard, so the script
# still works when its output is piped or run by another program.
interactive = (not ARGS.quiet) and sys.stdin is not None and sys.stdin.isatty()
if interactive:
    while True:
        try:
            seed = input("\nseed> start text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if seed == "" or seed.lower() in ("quit", "exit", "q"):
            info("Done. Try editing sample_text.txt or EPOCHS and running again!")
            break
        for temp in (0.3, 0.7):
            info("temperature %.1f:" % temp)
            info(wrap(generate(seed, length=120, temperature=temp), indent="   "))
else:
    info("")
    info("(Non-interactive session detected, so the prompt is skipped.)")
    info("Run this script in a terminal to try your own starting text.")
