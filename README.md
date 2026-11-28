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

The first round of the loop described below asked whether one chunk per bank plus a
rank-24 correction in each module beats two chunks per bank, which store 0.13% more
weights. It does not: 1.065 nats per byte against 1.058 at 32,000 steps, above at all
three seeds. The correction does recover nine tenths of what the second chunk buys over one
chunk alone (1.118).

The second and third rounds took sharing without a router into a modern transformer
(rotary positions, root-mean-square norms, a gated feed-forward) and set it against two
plain models at 32,000 steps. Eight modules sharing four chunks through the fixed map reach
1.001 nats per byte of validation loss. Four modules that own the same four chunks reach
1.040. Eight modules narrowed to width 184, the nearest buildable width, store 2.1% more
weights than the shared model and reach 1.013, above it at all three seeds. Stored weights
are held nearly equal and compute is not: the shared model multiplies by about twice the weights
per token of either plain model.

The fourth round held compute equal instead. The plain modern model of eight modules at
full width stores 6.4 million weights and reaches 0.982; the shared model does the same
arithmetic per token with 3.3 million and reaches 1.001, so halving the stored weights
costs 0.019 on average and at least 0.015 at every seed. Dropping to four modules costs 0.058 against that full model
and narrowing to width 184 costs 0.031.

Rounds five to eight replaced the mod map by four fixed maps drawn at random, at equal
stored weights. Two are worse than the mod map at every seed, by 0.020 and 0.011, and two
are level with it (0.002 above and 0.0001 below); none is separated from it on the better
side. Across the five maps the loss has a standard deviation of 0.009, against 0.0015 on
average across the training seeds of one map. Every map trained at the mod map's learning
rate.

Round nine froze each module of the routed model to the chunk its router picked most often
after a quarter of training. That reaches 1.068, against 1.013 for the mod map and 1.032 for
the model that keeps picking per token. The maps it froze to use only two or three of the
four chunks.

Round ten stopped the router after a quarter of training and let the modules go on picking
per token: 1.034, within 0.002 of the model whose router keeps learning. So what the router
learns late makes no measurable difference to the pick's 0.019 over the mod map, and
picking per token beats committing to the routers' own map by 0.036. Whether the 0.019 comes
from how the routers assign modules to chunks or from picking per token as such is open,
with two hypotheses in the ledger. Round eleven tried a fixed map whose first and last modules have chunks of their
own: 1.016 against the mod map's 1.001 at the same stored weights.

The full write-up, with every run, is in [report.md](report.md).

## A second track: fitting a bank to a trained model

![Stored bytes against perplexity](compress-chart.png)

[compress.md](compress.md) asks the cheaper question: can a shared bank compress a model that
already exists? The 30 feed-forward steps of SmolLM2-135M are replaced by 8, 15 or 23 shared
ones plus low-rank corrections per layer, fitted from the weights and healed by brief
distillation. It did not pay: rounding the original to 5 bits cuts it from 269 to 93 MB at
perplexity 23.1, the best bank arm reaches 240 MB at 23.8, and every bank arm at 16 or 8 bits
is beaten by plain rounding at fewer bytes. The bank has 15 to 24% lower perplexity than low
rank on every layer at equal bytes, but a control that leaves the same layers whole
and shares nothing comes within 2.2% of it, and a per-input router added nothing. The layers of
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
- **Other fixed maps.** Beside the mod map, a map drawn at random before training and kept
  for the whole run (each module's chunk drawn uniformly, the draw repeated until every chunk
  is used, its seed apart from the training seed), and a map that gives the first and the
  last module chunks of their own and cycles the modules between them through the rest. Each
  stores exactly what the mod map stores and trains at its learning rate.
- **Other routers.** For the model with both banks routed: the same router trained with no
  sampling noise (the highest-scoring chunk, gradient through the softmax), and a router that
  scores each chunk by the dot product of a learned key with a query.
- **Low-rank correction.** At a rank above 0, every module adds its own low-rank term to
  each of the two linear maps of the shared chunk it uses. The term starts at zero and its
  weights count as stored.
- **Settled routers.** Two diagnostics of a router. After a quarter of training, either each
  module keeps its router's most common pick on training text from then on (`frozen`), or
  the router's weights stop learning while picks stay per token (`stopped`).
- **Picks laid over the mod map.** `freed` is the mod map for a quarter of training with
  the routers idle; then the routers pick per token, the score of the module's mod-map chunk
  carrying a bonus of 6 that falls in a straight line to nothing by half of training, so the
  model that is evaluated is the router alone. `held` is a per-token pick from the first
  step whose score for the module's mod-map chunk carries a constant bonus of 2, which a
  trained router can outvote.
- **Architectures.** `classic` is the model above, with a learned position table, layer
  norms and a GELU feed-forward. `modern` has rotary positions on queries and keys in place
  of the position table, root-mean-square norms, and a gated feed-forward (SiLU gate, hidden
  width 8/3 of the model's, so a chunk stores the same weights to within 0.1%).
- **Data.** [enwik8](https://mattmahoney.net/dc/textdata.html), the first 100 MB of an English
  Wikipedia dump (text under CC BY-SA 3.0 / GFDL): 90 MB to train, the next 5 MB to validate.

## Run

The environment is pinned in `shell.nix` (PyTorch with CUDA); enter it with `nix-shell`.

```sh
python -m weightbank.train both-routed --seed 0 --steps 8000 --rate 1e-3
python -m weightbank.loop --gpu-hours 12           # one round: pick a cell, train it, record it
python -m weightbank.report                        # report.md, chart.png and curves.png from runs/ and the ledger
python -m weightbank.memory both-modern modern     # peak GPU memory of cells, into runs/memory.json
python -m weightbank.compress                      # compress.md and its charts from runs/compress/
python -m pytest && python -m mypy                 # tests and type check
```

The first command reproduces one run, any run in the ledger, and writes it to
`runs/<steps>/<name>-lr<rate>-s<seed>.json`; the table of every run in
[report.md](report.md) gives the four values of each. The first run downloads enwik8 into
`data/`. The reported runs were trained on one RTX 2080, 8 GB. On a shared card,
`WEIGHTBANK_GPU_MIB=4400` caps the memory a run's allocator may reserve; a run that needs
more stops with an out-of-memory error instead of changing its result.

## Exploring the design space

[ledger.json](ledger.json) is the record of what has been tried and what is open. It has
four lists:

- **`axes`.** What a model can vary: `shared`, `pick`, `chunks`, `rank`, `modules`, `width`,
  `steps` and `architecture`. Each axis has a meaning, a default and its values, each value
  with the tag it gets in a run's name and the plain words it gets in a chart's legend. A
  cell is one value on each axis. Its name is the
  tags of the values that differ from the default, in axis order (`both-c1-r24` shares both
  banks, one chunk each, with a rank-24 correction), and `full` when none differs; `steps`
  is the directory instead.
- **`cells`.** One row per cell tried, with its result and the run files behind it. The
  loop rewrites this list from `runs/`.
- **`hypotheses`.** Each has an `id`, a `statement`, a `priority` and `settle`: comparisons
  `{"lower": cell, "higher": cell}` claiming the first cell reaches the lower loss, a cell
  written as its non-default values. `status` and each comparison's `differences` are
  computed, never written by hand. A comparison is finished when both cells have three
  seeds at one learning rate. A hypothesis is `supported` when every comparison is finished
  and in each the lower cell wins at every seed by a mean margin above both cells' seed
  spread, `refuted` when it loses so in each, and `open` otherwise. One with no comparison
  stays listed and is never picked.
- **`rounds`.** What each round of the loop picked, why, and what it changed.

Each round of `python -m weightbank.loop --gpu-hours H` takes the open hypotheses with an
unfinished comparison, lowest priority number first and ledger order among equals, and
picks the first that has a comparison whose estimated cost is at most H GPU-hours, at its
cheapest such comparison. It trains what the two cells lack (a cell that has its runs costs
nothing), then rewrites the ledger, the report and the chart. With no such hypothesis it
trains nothing. Steering is editing: change a priority, or add a hypothesis with the cells
that would settle it. [report.md](report.md) states the cost estimate and shows the axes,
the hypotheses with their evidence, and every round.

Adding a value to an axis is one entry, `[value, tag, plain words]`, in that axis's `values`; adding an
axis is one entry in `axes`. Names, the cells hypotheses may name, the loop and the report's
tables follow, and cells already tried sit at a new axis's default. The model has to know
the value: an axis is a field of `ModelConfig` in `weightbank/model.py` of the same name,
or one of the three sharing axes that `weightbank/configs.py` turns into banks, and a cell
naming a value the model cannot build is refused. Defaults and tags are what run names are
spelled from: after changing one, the run files under the old spelling no longer match and
the ledger refuses them until they are renamed.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
