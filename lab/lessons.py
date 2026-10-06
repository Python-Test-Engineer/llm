"""lessons.py -- the guided tutorial content shown inside LLM Lab.

Each lesson maps to a stage of ``mini_llm.py`` (and a section of
``EXPLAINER.md``).  The app shows these in the left rail and links each one
to the control it explains, so a student can read an idea and immediately
turn the knob that demonstrates it.
"""

LESSONS = [
    {
        "id": "start",
        "tab": "corpus",
        "title": "0. The one-sentence idea",
        "tagline": "A language model predicts the next character.",
        "body": [
            "A language model is a function that says: given the last few "
            "characters, here is the probability of each possible next "
            "character. That is the whole game.",
            "Because it can put a probability on every possible next "
            "character, you can generate text: pick one according to those "
            "probabilities, add it to the end, and ask again. GPT, Claude and "
            "this little model are doing the same thing, one item at a time. "
            "They differ only in size and in how the prediction is computed.",
            "Everything below is that idea taken apart. Work down the "
            "lessons on the left; each one points at a panel on the right "
            "where you can try it yourself.",
        ],
        "points": [
            "Our model sees 3 previous characters as input.",
            "It outputs a probability for each of the characters it knows.",
            "Training = showing it the real next character and nudging the "
            "weights so that answer becomes more likely.",
        ],
        "try_this": [
            "Look at the Corpus panel: this short fox story is the entire "
            "training set. Everything the model will ever know comes from it.",
        ],
        "code": None,
    },
    {
        "id": "vocab",
        "tab": "corpus",
        "title": "1. The vocabulary",
        "tagline": "Turn text into numbers: 'a' -> 4, 'e' -> 8, ' ' -> 1",
        "body": [
            "A neural network is arithmetic. It cannot add letters, only "
            "numbers. So the first job is a dictionary from character to "
            "integer id, built from whatever characters actually appear.",
            "sorted(set(text)) takes the unique characters and sorts them "
            "into a stable order. The whole corpus then becomes a list of "
            "integers: 'the ' becomes [22, 11, 8, 1].",
            "The vocabulary is discovered from the data -- you never "
            "hand-write it. That is why the fox story has 28 characters "
            "rather than 27: it never uses 'j' or 'z', so those letters "
            "simply are not in the vocabulary.",
        ],
        "points": [
            "vocab_size is the size of the answer space. Each prediction "
            "spreads 100% of the model's belief across these characters.",
            "id 0 is a newline here, id 1 is a space -- the sort order "
            "decides, not you.",
            "Add or remove text and the vocabulary changes with it.",
        ],
        "try_this": [
            "Paste your own text into the Corpus box and press Load. Watch "
            "the character table and the vocabulary size change.",
            "The most common characters are the ones you would predict: "
            "space, e, t, a, o.",
        ],
        "code": "chars = sorted(set(text))\n"
                "vocab_size = len(chars)\n"
                "char_to_id = {c: i for i, c in enumerate(chars)}\n"
                "id_to_char = {i: c for i, c in enumerate(chars)}",
    },
    {
        "id": "examples",
        "tab": "examples",
        "title": "2. Training examples (sliding window)",
        "tagline": "Given these 3 characters, predict the next one.",
        "body": [
            "We want to teach 'given some characters, predict the next', so "
            "we slide a window across the text. For every position, the 3 "
            "characters before it are the INPUT and the character there is "
            "the TARGET.",
            "Notice there are no human labels. The answer is just the real "
            "text shifted by one, so the data labels itself. This trick -- "
            "predict the next thing -- is how every large language model is "
            "trained, and it is called self-supervised learning.",
            "With CONTEXT = 3 and 2,396 characters you get 2,393 examples: "
            "one for every position that has three characters before it.",
        ],
        "points": [
            "'the' -> ' ' is one example; 'he ' -> 'l' is the next.",
            "Overlapping windows mean the model sees the same text many "
            "ways, which is what makes so much training signal out of a "
            "small corpus.",
        ],
        "try_this": [
            "Type a sentence in the interactive tool and see exactly which "
            "input/target pairs the model would learn from.",
            "Change context in the Train panel and watch the windows widen.",
        ],
        "code": "examples = []\n"
                "for i in range(CONTEXT, len(ids)):\n"
                "    context = ids[i - CONTEXT:i]   # the characters before i\n"
                "    target  = ids[i]               # what we want to predict\n"
                "    examples.append((context, target))",
    },
    {
        "id": "network",
        "tab": "network",
        "title": "3. The network",
        "tagline": "embedding -> hidden layer (tanh) -> scores -> softmax",
        "body": [
            "The model is a handful of lists of numbers ('parameters'), all "
            "starting as small random values: an embedding per character, a "
            "weight matrix into a hidden layer, and a weight matrix out to "
            "one score per vocabulary character. Softmax turns those scores "
            "into probabilities.",
            "Why an embedding? You could feed the raw character id, but id "
            "27 is not '27 times' id 1 -- the numbers are arbitrary labels. "
            "An embedding gives each character its own short, learnable "
            "vector, and gradient descent pushes similar characters "
            "together. The model discovers that vowels are alike on its own.",
            "The hidden layer's tanh is what makes the network more than a "
            "line: a stack of purely linear layers is still one linear "
            "layer. The bend is what lets the model learn rules like 'if the "
            "context is \" th\", strongly favour e'.",
        ],
        "points": [
            "Every character -> 6 numbers (the embedding). 3 characters -> "
            "18 numbers. That vector is all the rest of the network sees.",
            "24 hidden neurons each look at all 18 numbers, then tanh "
            "squashes them into (-1, 1).",
            "28 output scores -> softmax -> 28 probabilities.",
            "Total learnable numbers: 28*6 + 24*18 + 24 + 28*24 + 28 = 1324.",
        ],
        "try_this": [
            "Change hidden and embed and watch the parameter count change. "
            "Real LLMs have billions to trillions.",
            "After training, look at the embedding neighbours panel: which "
            "characters did the model decide are alike?",
        ],
        "code": "pre_h[h] = b1[h] + sum(W1[h][j] * x[j] for j in range(18))\n"
                "h_act[h] = tanh(pre_h[h])\n"
                "scores[v] = b2[v] + sum(W2[v][k] * h_act[k] for k in range(24))\n"
                "probs = softmax(scores)",
    },
    {
        "id": "train",
        "tab": "train",
        "title": "4. Loss and training",
        "tagline": "Measure wrongness, nudge every weight, repeat.",
        "body": [
            "Training needs one number that says how bad the model is. We "
            "use cross-entropy: for each example, minus the log of the "
            "probability the model gave to the CORRECT next character. Right "
            "and confident -> 0. Wrong and confident -> large.",
            "Then backpropagation works out, for every one of the 1324 "
            "numbers, which way to nudge it to lower that loss. The update "
            "is just: step a little way downhill, opposite the gradient. "
            "Repeat for every epoch.",
            "The number to check first: before training, random weights "
            "should score exactly like random guessing, which is "
            "ln(vocab_size). For 28 characters that is 3.3322. If your "
            "untrained loss is ever worse than that, you have a bug.",
        ],
        "points": [
            "Full-batch: gradients are averaged over all examples, then one "
            "update per epoch. Simple and stable, but each step is small.",
            "Perplexity = exp(loss). It starts at 28 and should end near 4.",
            "Bits per character = loss / ln(2); near 1.9 after a good run.",
        ],
        "try_this": [
            "Train with the defaults and watch the loss curve fall and the "
            "sample text improve at each milestone.",
            "Now set lr to 8 or 12 and train: the loss explodes. This is "
            "'diverging', and it is why learning rates are chosen carefully.",
            "Set context to 1 and train: the model sees only one previous "
            "character, and both its loss and its writing get worse. Why?",
        ],
        "code": "loss   = -log(probs[target])            # cross-entropy\n"
                "dscores = probs[:]; dscores[target] -= 1.0\n"
                "E[v][d] -= LEARNING_RATE * dE[v][d] / n   # step downhill",
    },
    {
        "id": "generate",
        "tab": "generate",
        "title": "5. Generating text (and temperature)",
        "tagline": "Sample a character, append it, repeat.",
        "body": [
            "Generation is a short loop. Start with a seed like 'the ', ask "
            "the model for the probabilities of the next character, pick one "
            "at random weighted by those probabilities, append it, and slide "
            "the window forward. That last step -- feeding the output back "
            "in -- is called autoregressive generation, and it is exactly "
            "how ChatGPT produces one token after another.",
            "We sample rather than always taking the single most likely "
            "character. Always taking the top choice gives dull, looping "
            "text. Rolling a weighted die is why the output varies between "
            "runs.",
            "Temperature reshapes the distribution before sampling. Low "
            "sharpens it (safe, repetitive); high flattens it (creative, "
            "often nonsense). Temperature 1.0 leaves it untouched. This is "
            "the same knob real chat APIs expose -- now you know what it "
            "does to the numbers.",
        ],
        "points": [
            "It sees only 3 characters, so it cannot plan a sentence.",
            "It has learned the vocabulary, letter pairings, spacing, and "
            "rough word shapes -- but it invents plausible nonsense like "
            "'waseight' because 'was' + 'eight' is locally plausible.",
            "The three reasons the output is not perfect English: the model "
            "is tiny, the context is tiny, the corpus is tiny.",
        ],
        "try_this": [
            "Generate with temperature 0.2, then 0.9, and compare. Same "
            "model, same seed, very different text.",
            "Ask the model what comes next after 'riv' or ' th' in the "
            "Predict panel. Those should be nearly certain.",
        ],
        "code": "for _ in range(length):\n"
                "    _, _, probs = forward(context)              # ask\n"
                "    next_id = choices(range(vocab), weights=probs)  # sample\n"
                "    result.append(id_to_char[next_id])          # append\n"
                "    context = context[1:] + [next_id]           # slide",
    },
    {
        "id": "real",
        "tab": "compare",
        "title": "6. What a real LLM changes",
        "tagline": "Same loop; bigger scale and two extra techniques.",
        "body": [
            "Everything in this lab is the actual mechanism of a language "
            "model. Real ones add scale and a few specific techniques. What "
            "is NOT different: predict the next thing, measure the loss with "
            "cross-entropy, backpropagate, nudge the weights.",
            "Two techniques are worth knowing about now that you have the "
            "foundation. Tokens instead of characters: real models split "
            "words into pieces so the vocabulary covers any word, including "
            "ones never seen. And attention: our hidden layer must squeeze "
            "everything it knows into 24 numbers, while attention lets the "
            "model look directly back at any previous position.",
        ],
        "points": [
            "characters -> sub-word tokens (BPE), roughly 4x shorter sequences.",
            "3-character context -> thousands of tokens.",
            "one feed-forward hidden layer -> dozens of attention layers.",
            "1324 parameters -> billions to trillions.",
            "plain gradient descent -> Adam, warmup, learning-rate schedules.",
            "full-batch, 250 epochs -> mini-batches over trillions of tokens.",
        ],
        "try_this": [
            "Read EXPLAINER.md sections 12-14 for the full comparison and a "
            "glossary of every term used here.",
        ],
        "code": None,
    },
]

# Exercises from EXPLAINER.md section 13, surfaced as a checklist in the app.
EXERCISES = [
    {"id": "ex1", "text": "Break it on purpose: train with lr 12 and watch "
                          "the loss explode, then find the highest value that "
                          "still trains."},
    {"id": "ex2", "text": "Starve the context: set context to 1 and compare "
                          "its loss and output to the default. Why is it worse?"},
    {"id": "ex3", "text": "Feed it more: train for 600 epochs and compare to "
                          "250. What improves first -- rare or common words?"},
    {"id": "ex4", "text": "Change the corpus: paste a page of your own "
                          "writing and retrain. Does it pick up your habits?"},
    {"id": "ex5", "text": "Make it overfit: use a ~200-character corpus and "
                          "many epochs. Watch it reproduce the training text "
                          "almost exactly."},
    {"id": "ex6", "text": "Look under the hood: after training, check the "
                          "embedding neighbours. Did it group vowels? Punctuation?"},
]

# The comparison table from EXPLAINER.md, structured for rendering.
COMPARE = [
    ("characters", "sub-word tokens (BPE), ~50k-200k of them"),
    ("3-character context", "thousands of tokens of context"),
    ("one hidden layer", "dozens-hundreds of layers"),
    ("feed-forward only", "attention (transformers)"),
    ("1,324 parameters", "billions-trillions"),
    ("plain gradient descent, lr = 5", "Adam/AdamW, warmup, LR schedules"),
    ("full-batch, 250 epochs", "mini-batches, one or two passes over the data"),
    ("2,396 characters of training data", "trillions of tokens"),
]
