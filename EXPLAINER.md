# How the mini-LLM works — a complete walkthrough for students

This is a full tutorial for `mini_llm.py` in this folder. By the end you will
understand, line by line, how a machine can learn to write text — starting
from nothing but a paragraph of English and some random numbers.

Everything here matches the actual code and the actual output of a default
run, so you can follow along by opening `mini_llm.py` in another window and
running it with:

```
uv run .\llm\mini_llm.py
```

Every number quoted below (2,396 characters, loss 3.3320 → 1.3112, 1,324
parameters, and so on) is real output from that command.

---

## 1. The one-sentence idea

> A language model is a function that says: **"given the last few characters,
> here is the probability of each possible next character."**

That's it. Everything else is a consequence.

Because the model can put a probability on every possible next character, you
can **generate** text: pick a character according to those probabilities, add
it to the end, then ask again — one character at a time. GPT, Claude and this
little script are doing the same thing. They differ only in *size* and in
*how the prediction is computed*.

Our model's input is **3 previous characters** and its output is a probability
for each of the **28 characters** it knows.

---

## 2. The big picture (the seven stages)

```
   corpus text            "the little fox lived at the edge of the wood..."
        |
        v
 [1] vocabulary           map each character to a number:  'a'->4, 'e'->8, ' '->1
        |
        v
 [2] training examples    sliding window:  ('the') -> ' '   ('he ') -> 'l'  ...
        |
        v
 [3] the network          characters -> embeddings -> hidden layer -> scores -> probabilities
        |
        v
 [4] loss                 how wrong was the prediction?   -ln(P(correct character))
        |
        v
 [5] backpropagation      for every weight, which way should it move? (calculus)
        |
        v
 [6] the training loop    nudge every weight a little; repeat 250 times
        |
        v
 [7] generation           sample a character, append it, repeat
```

The rest of this document takes each stage apart.

---

## 3. The data: `sample_text.txt`

The **entire** training set is one file:

```
the little fox lived at the edge of the wood, where the tall trees met the
open field. every morning the sun rose over the hills, and the fox woke to
the warm gold light. ...
```

A default run reports:

```
Loaded sample_text.txt
   2396 characters, 41 lines
```

Two things to notice straight away:

1. **It's small.** 2,396 characters. GPT-3 was trained on roughly 300
   *billion* tokens. This is a difference of about eight orders of magnitude —
   and yet the core algorithm below is the same.
2. **The code lowercases everything** (`text = f.read().strip().lower()`).
   That's a deliberate simplification: it halves the vocabulary and avoids
   having to learn that `T` and `t` behave similarly. Real tokenizers keep
   case; ours throws it away.

---

## 4. Stage 1 — the vocabulary

A neural network is arithmetic. It cannot add letters, only numbers. So the
first job is a dictionary from character to integer id.

The code builds it in three lines:

```python
chars = sorted(set(text))                            # every unique character, sorted
vocab_size = len(chars)                              # how many distinct characters
char_to_id = {c: i for i, c in enumerate(chars)}     # 'a' -> 4, etc.
id_to_char = {i: c for i, c in enumerate(chars)}     # 4 -> 'a', etc.
```

`sorted(set(text))` takes the *set* of characters (unique ones, no order) and
sorts it into a stable order. The real result for our corpus:

```
Found 28 distinct characters. Each one gets a number:

\n        = 0  [space]   = 1  ','       = 2  '.'       = 3  'a'       = 4
'b'       = 5  'c'       = 6  'd'       = 7  'e'       = 8  'f'       = 9
'g'       = 10 'h'       = 11 'i'       = 12 'k'       = 13 'l'       = 14
'm'       = 15 'n'       = 16 'o'       = 17 'p'       = 18 'q'       = 19
'r'       = 20 's'       = 21 't'       = 22 'u'       = 23 'v'       = 24
'w'       = 25 'x'       = 26 'y'       = 27
```

So the whole corpus becomes a list of integers. `"the "` becomes `[22, 11, 8, 1]`.

**Why 28?** There are 26 lowercase letters, but our text never uses `j` or
`z`, so 24 letters appear, plus space, comma, period and newline = 28. The
vocabulary is discovered from the data; you never hand-write it.

**Key idea:** `vocab_size = 28` is the size of the answer space. At every
prediction the model spreads 100% of its belief across these 28 characters.

---

## 5. Stage 2 — turning the text into training examples

We want to teach "given some characters, predict the next". So we slide a
window across the text. This is the whole of the code:

```python
examples = []
for i in range(CONTEXT, len(ids)):      # CONTEXT = 3
    context = ids[i - CONTEXT:i]        # the 3 characters before position i
    target  = ids[i]                    # the character at position i
    examples.append((context, target))
```

With `CONTEXT = 3`, and text `"the little fox"`, this produces:

| INPUT (context) | TARGET | the sentence so far |
|-----------------|--------|---------------------|
| `t h e`         | `␠`    | `the `              |
| `h e ␠`         | `l`    | `he l`              |
| `e ␠ l`         | `i`    | `e li`              |
| `␠ l i`         | `t`    | ` lit`              |
| `l i t`         | `t`    | `litt`              |
| `i t t`         | `l`    | `ittl`              |
| `t t l`         | `e`    | `ttle`              |

(`␠` is a space.)

The real run reports:

```
Each example is 3 characters of input plus the next character as the answer.
Sliding a window across the text gives 2393 examples.
```

That's `2396 - 3 = 2393` — one example for every position that has 3
characters before it.

> **This is supervised learning.** We already know the correct answer for
> every example, because the answer is just the real text shifted by one.
> There is no human labelling. The data labels itself. This trick — "predict
> the next thing" using the text itself — is called **self-supervised
> learning**, and it is how every large language model is trained.

---

## 6. Stage 3 — the network

### 6.1 What it has to learn

The network is four lumps of numbers ("parameters"), all starting as small
random values:

| Name | Shape         | Meaning                                              |
|------|---------------|------------------------------------------------------|
| `E`  | 28 × 6        | an **embedding** (6 numbers) for each character      |
| `W1` | 24 × 18       | weights from the input (18 numbers) to the hidden layer |
| `b1` | 24            | a bias for each of the 24 hidden neurons             |
| `W2` | 28 × 24       | weights from the hidden layer to the 28 output scores |
| `b2` | 28            | a bias for each output character                     |

Total learnable numbers: `28·6 + 24·18 + 24 + 28·24 + 28 = 1,324`.

For comparison: a real LLM has billions to trillions. Ours has 1,324. Same
idea, microscopically small.

### 6.2 Why an *embedding*?

You could feed the network the raw character id — but that would be a lie.
Id 27 (`y`) is not "27 times" id 1 (space); the numbers are arbitrary labels.

An **embedding** solves this: each character gets its own short list of
learnable numbers, and the network chooses them. If two characters end up
used in similar contexts, gradient descent naturally pushes their embeddings
close together. The model *discovers* that vowels are alike, that `.` and `,`
are alike, and so on. Nothing about that is hand-coded.

So a 3-character context becomes 3 embeddings glued together:

```
' t '  ->  [ 0.2, -0.4,  0.1,  0.7, -0.1,  0.3 ]
'h'    ->  [ 0.5,  0.1, -0.2,  0.0,  0.4, -0.6 ]
'e'    ->  [-0.3,  0.2,  0.6, -0.1,  0.1,  0.2 ]
--------------------------------------------------
x      =  [ ... 18 numbers in total ... ]
```

That 18-number vector `x` is everything the rest of the network is allowed to
know about the context.

### 6.3 The hidden layer

The hidden layer is 24 neurons. Each one looks at *all 18* input numbers and
produces one number:

```python
pre_h[h] = b1[h] + sum(W1[h][j] * x[j] for j in range(18))
h_act[h] = tanh(pre_h[h])
```

- `pre_h[h]` is a weighted sum plus bias — a straight line.
- `tanh` then squashes it into the range (−1, 1).

**Why the `tanh`?** Without a non-linear function between layers, stacking
layers would be pointless: a stack of linear operations is still one linear
operation, and the network could only ever learn straight-line relationships.
The bend introduced by `tanh` is what lets the model represent "if the context
looks like ` th`, strongly favour `e`" — a non-linear rule.

Think of each of the 24 hidden neurons as a small *feature detector*. Together
they form a 24-number summary of "what kind of context is this?".

### 6.4 The output layer and softmax

Finally, one score per vocabulary character:

```python
scores[v] = b2[v] + sum(W2[v][k] * h_act[k] for k in range(24))
probs = softmax(scores)
```

`softmax` turns arbitrary scores into probabilities:

```
softmax(s)[v] = exp(s[v]) / sum(exp(s[u]) for all u)
```

Three properties make it the right tool:

- every output is positive (`exp` is always positive);
- they all sum to 1 (it's a valid probability distribution);
- it's **smooth and differentiable**, which we will need for backpropagation.

Subtracting the maximum first (`biggest = max(scores)` in the code) is a
numerical trick only: it prevents `exp` of a large number from overflowing.
It changes nothing mathematically, because the same constant cancels top and
bottom.

### 6.5 The whole forward pass

Here is a real prediction from a trained model:

```
after 'the'     the model expects:
   [space]    93.5%  #####################################
   \n          4.3%  ##
   'r'         0.6%
   'd'         0.6%
   'n'         0.2%
```

Reading this: given the context `the`, the model is 93.5% sure the next
character is a space. That is exactly right — "the" is almost always
followed by a space in English. Other contexts:

```
after ' th'  ->  'e' 96.0%     (because " th" is nearly always " the")
after 'riv'  ->  'e' 99.3%     (because "riv" is nearly always "rive")
after 'fox'  ->  [space] 73.8%, \n 11.3%, '.' 7.4%, ',' 6.1%
```

The last one is interesting: after the word "fox" the model is fairly sure a
space follows, but it hedges — sometimes "fox." or "fox," or a line break. It
has genuinely learned the statistics of this little story, not memorised one
string.

---

## 7. Stage 4 — the loss (measuring wrongness)

Training needs a single number that says "how bad is the model right now?".
We use **cross-entropy loss**, which for one example is just:

```
loss = -ln( probability the model gave to the CORRECT next character )
```

The consequences:

- If the model gave the correct character probability 1.0 → loss is `-ln(1) = 0`. Perfect.
- If it gave it 0.5 → loss is `-ln(0.5) ≈ 0.69`.
- If it gave it 0.01 → loss is `-ln(0.01) ≈ 4.6`. Very bad.

We average this over all 2,393 examples to get the epoch's loss.

### 7.1 The number you should always check first

Before training, the weights are random, so the model has no idea. What loss
would pure guessing give? A uniform guess assigns `1/28` to every character,
so:

```
loss = -ln(1/28) = ln(28) = 3.3322
```

Now look at the real output of the run:

```
Loss before training: 3.3320
```

**That is not a coincidence — it is a check.** Our untrained model has
(almost exactly) the loss of random guessing. If you ever change the code and
see a *worse* starting loss than `ln(vocab_size)`, you have a bug. This
single comparison has caught countless real bugs in real projects.

### 7.2 What the final number means

After 250 epochs:

```
Final loss: 1.3112   (started at 3.3320)
```

Converting to a more intuitive scale:

- **Perplexity** = `exp(loss)` = `exp(1.3112) ≈ 3.71`. Loosely: "the model is
  as uncertain as if it were choosing evenly among about 3.7 characters."
  It started at `exp(3.3322) = 28`. So the model is roughly **7.5×** less
  confused than when it started.
- In **bits per character**: `1.3112 / ln(2) ≈ 1.89 bits/char`. For scale,
  good English text compressors reach about 1–1.5 bits/char, and our tiny
  model is already in that ballpark — on a corpus it has memorised a lot of.

---

## 8. Stage 5 — backpropagation (the heart of it)

Training means: *nudge every one of the 1,324 numbers so the loss goes down.*
Backpropagation is just the chain rule from calculus, applied efficiently.

`d<something>` in the code means "the derivative of the loss with respect to
that thing" — i.e. **if I increase this by a tiny bit, how much does the loss
change, and in which direction?**

### 8.1 Output layer — the beautiful simplification

Normally, differentiating softmax followed by cross-entropy is a messy
chain. But when you combine them (as we do), the derivative collapses to
something wonderfully simple:

```
dscores[v] = probs[v] - (1 if v is the correct character else 0)
```

In code:

```python
dscores = probs[:]            # copy of the probabilities
dscores[target] -= 1.0        # subtract 1 from the correct character
```

Read it as: *"push each probability down by the amount we over-predicted it,
and for the correct character, record the shortfall."* If the model gave the
right answer probability 1.0, every entry is 0 — no error, no learning needed.

The gradients for the output weights follow directly:

```
dW2[v][k] += dscores[v] * h_act[k]
db2[v]    += dscores[v]
```

### 8.2 Hidden layer — sending the error backwards

The hidden layer didn't cause the error alone, but it takes its share of the
blame in proportion to how strongly it was connected:

```
dh[k]     = sum over v of ( dscores[v] * W2[v][k] )
dpre_h[k] = dh[k] * (1 - h_act[k]**2)      # the tanh derivative
```

The second line is the chain rule through `tanh`. Note the identity
`d/dx tanh(x) = 1 - tanh(x)²` — you already have `tanh(x)` stored as
`h_act[k]`, so the derivative is nearly free. This is a recurring theme in
neural networks: activations are chosen partly for having cheap derivatives.

### 8.3 Input weights and embeddings

```
dW1[k][j] += dpre_h[k] * x[j]
db1[k]    += dpre_h[k]
```

And finally the error flows *into* the context characters, updating each
character's embedding:

```python
for j in range(len(x)):
    dxj = 0.0
    for k in range(HIDDEN):
        dxj += dpre_h[k] * W1[k][j]
    pos = j // EMBED_DIM          # which of the 3 context characters
    dim = j %  EMBED_DIM          # which of its 6 numbers
    dE[context[pos]][dim] += dxj
```

That division `j // EMBED_DIM` is just bookkeeping: `x` is three 6-number
embeddings glued end to end, so we work out which original character each of
the 18 slots came from.

### 8.4 The pattern to remember

Every layer does the same three things:

1. **Forward:** `output = activation(weights · input + bias)`
2. **Backward:** send the error back through the weights' transpose, and
   multiply by the activation's derivative.
3. **Gradient:** the weight's gradient is *its input* times *the error that
   reached its output*.

Learn that rhythm and you can derive any network, including transformers.

---

## 9. Stage 6 — the training loop

### 9.1 Full-batch gradient descent

Each epoch, the code does something deliberately simple:

1. set all gradients to zero;
2. loop over **all 2,393 examples**, computing the loss and *accumulating*
   gradients (never updating yet);
3. after the loop, update every weight **once**.

That is **full-batch gradient descent**. It's the most stable and the easiest
to reason about — but it means each step uses the average error over the whole
dataset, and so each step is small.

### 9.2 The update rule

```python
E[v][d] -= LEARNING_RATE * dE[v][d] / n
```

In words: *step a little way downhill, opposite the gradient.* The `/ n`
divides by the number of examples, converting the summed gradients into an
average.

### 9.3 A real lesson about the learning rate

Here is something that genuinely matters, and that you can reproduce:

> When this corpus was tiny (122 examples), a learning rate of `0.3` worked
> fine. After we grew the corpus to **2,393 examples**, that same `0.3` made
> training crawl — the final loss stalled around `2.55`.

Why? Because we now divide by `n = 2393` instead of `n = 122`. Every update is
**~20× smaller**, so 250 epochs is nowhere near enough to converge.

The fix was to raise the learning rate. Measured on this code:

| learning rate | result                                    |
|---------------|-------------------------------------------|
| 1.5           | learns, slowly                            |
| 3             | better                                    |
| 5             | **good (the default)**                    |
| 6             | slightly better in short runs             |
| 8 and above   | **diverges** — the loss explodes          |

So the learning rate isn't a magic constant you copy: it depends on your data
and your update scheme. Too small and you never arrive; too large and you
overshoot and blow up. This is exactly why, in real projects, people normalise
gradients (e.g. with optimisers like Adam) instead of picking a raw scale by
hand — so that the *same* learning rate works as the dataset grows.

### 9.4 Watching it learn

The script prints samples mid-training, and they show the progression clearly:

```
[after  62 of 250 epochs] "the overed sox woong and and and and watkerew and and fox wame and and won"
[after 125 of 250 epochs] "the cook the the do the soon the goon the ole to the the soun the soon to"
[after 187 of 250 epochs] "the fox woun warm and wame bit in and and waow was hos liver and the river"
[final,  250 epochs]      "the fox was hed hith a will fall of the waseight. the river and the fox..."
```

Watch the arc: early on it has learned *word shapes* and *spacing* but not
real words. By the end, real words appear (`fox`, `warm`, `river`, `wood`)
mixed with inventions (`woun`, `waseight`). Notice it learns easier, more
frequent patterns first — exactly what you'd expect.

The whole default run takes about **162 seconds** (250 epochs).

---

## 10. Stage 7 — generating text (and temperature)

Generation is a loop, and it is short:

```python
context = to_context(seed)          # e.g. "the " -> the last 3 character ids
result = list(seed)
for _ in range(length):
    _, _, probs = forward(context)          # 1. ask for the probabilities
    next_id = random.choices(range(vocab_size), weights=probs)[0]   # 2. sample
    result.append(id_to_char[next_id])      # 3. append the chosen character
    context = context[1:] + [next_id]       # 4. slide the window forward
```

Two details worth pausing on:

- **Sampling, not "argmax".** We could always pick the single most likely
  character, but that produces dull, looping text (and can get stuck). Instead
  we *roll a weighted die* using the probabilities. That one change is why the
  output varies between runs.
- **The window slides by one.** The character we just generated becomes part
  of the context for the next prediction. This is called *autoregressive*
  generation, and it's exactly how ChatGPT produces one token after another.

### 10.1 Temperature

```python
adjusted = [p ** (1.0 / temperature) for p in probs]   # then re-normalise
```

Temperature reshapes the distribution *before* sampling:

- `temperature = 0.2` → sharpens it. Top choices dominate; output is safe,
  repetitive, grammatically boring. Real output:
  `"the fox was hed hith a will fall of the waseight. the river and the fox..."`
- `temperature = 0.5` → balanced (the default-ish).
- `temperature = 0.9` → flattens it. Rare characters become more likely;
  output is creative but often nonsense:
  `"the lith tuing, on the wats and the like brightir.seight..."`

`temperature = 1.0` leaves the distribution untouched. This is the **same
temperature knob** that real chat APIs expose — now you know what it actually
does to the numbers.

### 10.2 Why the output isn't perfect English

Three reasons, and none of them are bugs:

1. **The model is tiny** — 1,324 parameters.
2. **The context is tiny** — it sees only 3 characters, so it cannot plan a
   sentence. It literally cannot "know" what it said 20 characters ago except
   indirectly, through the hidden layer's summary.
3. **The corpus is tiny** — 2,396 characters. It has seen the word "the" a few
   dozen times, and the word "wood" a handful of times.

What it *has* learned is impressive for its size: the vocabulary, which
letters can follow which, that spaces separate words, that sentences end in
`.`, and rough word shapes. It invents `waseight` because `was` + `eight` is
locally plausible — a real, well-known failure mode of character models.

---

## 11. Reading the code: a map

If you want to follow `mini_llm.py` top to bottom:

| Section in the file            | What it does                                    |
|--------------------------------|-------------------------------------------------|
| `parse_args()`                 | reads `--epochs`, `--lr`, `--hidden`, ...       |
| "STEP 1"                       | loads and lowercases `sample_text.txt`          |
| "STEP 2"                       | builds `chars`, `char_to_id`, `id_to_char`      |
| "STEP 3"                       | builds the `examples` list (sliding window)     |
| "STEP 4"                       | creates `E`, `W1`, `b1`, `W2`, `b2` randomly    |
| `softmax()`                    | scores → probabilities                          |
| `forward()`                    | the whole left-to-right computation             |
| `average_loss()`               | measures current wrongness over all examples    |
| `to_context()` / `generate()`  | turning a seed into generated text              |
| "STEP 5" (the `for epoch` loop)| the training loop and backpropagation           |
| "STEP 6"                       | prints samples at several temperatures          |
| `show_predictions()`           | prints the "top next character" tables          |
| the final `while True`         | the interactive prompt                          |

---

## 12. Where this differs from a real LLM

Everything above is the *actual* mechanism of a language model. The real ones
add scale and a few specific techniques. The important differences:

| This mini-LLM                        | Real LLM (e.g. GPT)                          |
|--------------------------------------|----------------------------------------------|
| characters                           | sub-word **tokens** (BPE) — ~50k–200k of them |
| 3-character context                  | thousands of tokens of context               |
| one hidden layer                     | dozens–hundreds of layers                    |
| feed-forward only                    | **attention** (transformers)                 |
| 1,324 parameters                     | billions–trillions                           |
| plain gradient descent, lr = 5       | Adam/AdamW, learning-rate schedules, warmup  |
| full-batch, 250 epochs               | mini-batches, one or a few passes over huge data |
| 2,396 characters of training data    | trillions of tokens                          |

What is **not** different: predict the next thing, measure the loss with
cross-entropy, backpropagate, nudge the weights. That loop is identical.

Two specific things worth knowing about, since you now understand the
foundation:

- **Tokens instead of characters.** Real models split words into pieces
  ("running" → "run" + "ning") so the vocabulary covers any word, including
  ones never seen. This makes sequences ~4× shorter and each prediction more
  meaningful.
- **Attention.** Our hidden layer must squeeze *everything* it knows about the
  context into 24 numbers. Attention instead lets the model look directly back
  at *any* previous position and decide which ones matter — which is why
  transformers can handle long, structured text.

---

## 13. Exercises (do these — this is where learning happens)

1. **Break it on purpose.** Run `--lr 12`. Watch the loss explode. Then find
   the highest value that still trains. You've just rediscovered why learning
   rates are chosen carefully.
2. **Starve the context.** Run `--context 1`. The model now only knows the
   previous character. Compare its output and its loss to the default. Why is
   it worse?
3. **Feed it more.** Run `--epochs 600` and compare the loss and samples to
   250. What improves first — rare words, or common ones?
4. **Change the corpus.** Replace `sample_text.txt` with a page of your own
   writing and re-run. Does it pick up your habits? (Keep it lowercase, or
   note how the vocabulary changes.)
5. **Get under the hood.** Add a `print` inside `forward()` to watch `probs`
   for a fixed context change during training. Watch `the ` → `[space]` climb
   from ~4% toward ~93%.
6. **Make it overfit.** Use a 200-character corpus and `--epochs 800`. It will
   start reproducing the training text almost exactly. That's **memorisation**,
   the enemy of generalisation — and it's exactly why real models are trained
   on enormous, varied corpora.

---

## 14. Glossary

- **Parameter / weight** — one learnable number. We have 1,324.
- **Embedding** — a short learned vector standing in for a discrete thing
  (here, a character).
- **Forward pass** — computing a prediction from inputs, left to right.
- **Loss** — one number measuring how wrong the prediction was. Lower is better.
- **Cross-entropy** — the standard loss for "pick one of N classes"; here,
  `-ln(P(correct character))`.
- **Softmax** — turns raw scores into probabilities summing to 1.
- **Backpropagation** — using the chain rule to find each weight's gradient.
- **Gradient** — the direction and steepness of the loss's slope with respect
  to a weight. We step *against* it.
- **Learning rate** — how big each step is.
- **Epoch** — one full pass over all training examples.
- **Full-batch** — updating once per epoch, using the average gradient.
- **Autoregressive** — generating one item, then feeding it back in to
  generate the next.
- **Temperature** — reshapes the probability distribution before sampling;
  low = safe, high = wild.
- **Perplexity** — `exp(loss)`; "how many equally likely choices is the model
  effectively torn between?"
- **Self-supervised** — learning from the data's own structure (predict the
  next character) with no human labels.

---

## 15. The one-paragraph summary

We turned text into numbers, slid a window over it to make "predict the next
character" examples, and built a small network of 1,324 numbers that maps a
3-character context to probabilities over 28 characters. We measured its
wrongness with cross-entropy (starting at the random-guess baseline of
`ln 28 = 3.33`), used backpropagation to find which way each weight should
move, and repeated that 250 times until the loss fell to `1.31` — about 1.9
bits per character. Then we generated text by sampling from its predictions
one character at a time, with a temperature knob to trade safety for
surprise. That is the entire idea behind a language model; everything in a
state-of-the-art system is a bigger, faster, more cleverly connected version
of exactly these steps.
