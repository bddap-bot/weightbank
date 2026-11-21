# weightbank

Does routing into a shared bank of weights beat a smaller plain model that stores the same
number of weights?

A transformer language model has M compute modules, and normally each owns its weights. Here
the model instead holds a bank of N < M weight chunks, and each module picks, for every input
token, one chunk from the bank to compute with. Stored weights shrink because N < M. Memory is
the resource treated as scarce; compute is not.

## Result

![Validation loss against stored weights](chart.png)

At equal stored weights, routing did not beat a smaller plain model on enwik8. At 8,000
training steps with one learning rate, routing only the feed-forward weights was level with
the plain curve (0.003 nats per byte below it, inside seed spread) and the other two routed
models were above it. At 32,000 steps with a learning rate chosen for every configuration, routing only the feed-forward weights lands 0.005
nats per byte above the plain curve, routing attention 0.021 above it, and routing both 0.012
above it. Sharing the same two banks with no router, each module fixed to one chunk, lands
0.008 below the plain curve, and sharing a single chunk across all 8 modules lands 0.038
below it. Two other routers for the model with both banks routed, one trained without
sampling noise and one that scores chunks by learned keys, do not close the distance to the
fixed map.

The full write-up, with every run of both experiments, is in [report.md](report.md).

## A second track: fitting a bank to a trained model

![Stored bytes against perplexity](compress-chart.png)

[compress.md](compress.md) asks the cheaper question: can a shared bank compress a model that
already exists? The 30 feed-forward steps of SmolLM2-135M are replaced by 8, 15 or 23 shared
ones plus low-rank corrections per layer, fitted from the weights and healed by brief
distillation. It did not pay: rounding the original to 5 bits cuts it from 269 to 93 MB at
perplexity 23.1, the best bank arm reaches 240 MB at 23.8, and every bank arm at 16 or 8 bits
is beaten by plain rounding at fewer bytes. The bank beats low rank on every layer
by 15 to 24% of perplexity at equal bytes, but a control that leaves the same layers whole
and shares nothing comes within 2% of it, and a per-input router added nothing. The layers of
that model have almost nothing in common to share, so this is a result about retrofitting,
not about banks a model is trained with.

## Design

- **Model.** A pre-norm, decoder-only transformer over bytes: 8 modules, width 256, 4 heads,
  256 bytes of context. A module is one layer: an attention step and a feed-forward step.
- **Bank.** A feed-forward chunk is the two linear maps of one feed-forward step. An attention
  chunk is the query/key/value map and the output map of one attention step. A routed model
  stores 4 chunks of the routed kind for its 8 modules.
- **Router.** One small network per routed kind, shared by all modules. It sees the token's
  normalised hidden state and a learned vector for the module that is asking, and scores the
  chunks: `score(tanh(data(x) + asker[module]))`.
- **Hard picks.** Every forward pass uses exactly one chunk per token per module. Training
  samples the pick with Gumbel noise and passes gradients straight through the softmax;
  evaluation takes the highest score with no noise.
- **Attention with per-token picks.** Each token writes its key and value once, under its own
  chunk, and queries read whatever keys are there (mixed keys). That is how inference with a
  single key/value cache runs, and it is what is trained.
- **Private weights.** Each module keeps its own two layer norms; they count as stored
  weights, as do the routers, embeddings and output head.
- **Controls.** The full unshared model; plain models narrowed to widths 96 to 224 and plain
  models cut to 1, 2, 4 and 6 modules, which draw loss against stored weights; and the fixed
  map, both banks shared with no router, module i always using chunk i mod N, at N = 1, 2, 4
  and 8 chunks. At N = 8 every module has its own chunk and the model must match the full one.
- **Other routers.** For the model with both banks routed: the same router trained with no
  sampling noise (the highest-scoring chunk, gradient through the softmax), and a router that
  scores each chunk by the dot product of a learned key with a query.
- **Data.** [enwik8](https://mattmahoney.net/dc/textdata.html), the first 100 MB of an English
  Wikipedia dump (text under CC BY-SA 3.0 / GFDL): 90 MB to train, the next 5 MB to validate.

## Run

The environment is pinned in `shell.nix` (PyTorch with CUDA); enter it with `nix-shell`.

```sh
python -m weightbank.train both-routed --seed 0 --steps 8000 --rate 1e-3
python -m weightbank.sweep                         # every planned run that runs/ lacks
python -m weightbank.report                        # report.md and chart.png from runs/
python -m weightbank.compress                      # compress.md and its charts from runs/compress/
python -m pytest && python -m mypy                 # tests and type check
```

The first command reproduces one run, any run of either experiment, and writes it to
`runs/<steps>/<configuration>-lr<rate>-s<seed>.json`; the table of every run in
[report.md](report.md) gives the four values of each. Configurations are named in
`weightbank/configs.py`. `weightbank/sweep.py` holds the plan: the first experiment, the
learning-rate sweeps, and the long runs at the rates the sweeps chose. The first run downloads
enwik8 into `data/`. The reported runs were trained on one RTX 2080, 8 GB.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
