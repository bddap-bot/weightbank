# Routing into a shared weight bank against smaller plain models

Every table below is written by `python -m weightbank.report` from the run files in `runs/`.
Loss is cross-entropy on the validation split in nats per byte (divide by 0.693 for bits).

## Reading

Feed-forward routed (4,430,404 stored weights, loss 1.0977) sits 0.0032 nats per byte below the
better plain curve (fewer modules), inside the seed spread of the plain points on either side
of it (seed sd 0.0052 and 0.0080), and 0.0080 below the narrower plain curve, so it matches
the best plain model of its size and does not beat it. Attention routed (5,480,004, loss 1.1146) sits 0.0242 above the narrower
curve and 0.0263 above the shallower one, and both routed (3,394,952, loss 1.1452) sits 0.0184
and 0.0300 above them, each several seed spreads worse than a plain model storing the same
weights. The same two banks with no router, each module fixed to one chunk, reach 1.1004:
0.0448 below both routed and 0.0153 below the better plain curve, so within this budget
sharing weights across modules helped and choosing the chunk per token cost loss.

![Validation loss against stored weights](chart.png)

The dark line is the plain model narrowed from width 256 to 128 at 8 modules; the lighter line
is the plain model cut from 8 modules to 4 at width 256. Bands and bars run from the lowest to
the highest of three seeds. The square is a control outside the question: both banks shared
with no router, module i always using chunk i mod 4.

## Routed configurations against the plain curves

Each plain curve is read at the routed configuration's stored weights by interpolating mean
loss linearly in the logarithm of stored weights.

{{gaps}}

`both-routed` and `plain-l4` are an equal-storage pair that needs no interpolation: the same 4
feed-forward and 4 attention chunks, applied 8 times with routing or 4 times without. The
routed model adds 38,536 weights (two routers and four more modules' layer norms).

## Setup

- Data: enwik8, the first 100 MB of an English Wikipedia dump (text under CC BY-SA 3.0 and
  GFDL). Bytes are the tokens (vocabulary 256). The first 90 MB train, the next 5 MB validate;
  validation loss is over all 19,531 non-overlapping 256-byte windows of that split.
- Model: pre-norm decoder-only transformer, 8 modules, width 256, 4 heads, context 256,
  learned positions. A module is one layer: an attention step and a feed-forward step.
- Budget, the same for every run: 8,000 steps of 64 windows, AdamW (peak rate 1e-3, 200 warm-up
  steps, cosine decay to a tenth, weight decay 0.1 on matrices and embeddings), mixed precision (float16 autocast), gradient
  clipping at 1. Seeds 0, 1, 2 set both the initial weights and the batches.
- Bank: 4 chunks of each routed kind for 8 modules. A feed-forward chunk is both linear maps
  of a feed-forward step; an attention chunk is the query/key/value map and the output map.
- Router: one per routed kind, shared by all modules, 64 units wide. It scores the chunks from
  the token's normalised hidden state plus a learned vector for the asking module.
- Training the hard pick: straight-through Gumbel-softmax at temperature 1. The forward pass
  always computes with exactly one chunk per token per module, sampled with Gumbel noise, and
  the gradient flows through the softmax. This was chosen over a soft mixture annealed to a
  hard one because the network trained is the network evaluated from the first step: no
  schedule, and no gap between soft training and hard inference to close. Validation takes the
  highest-scoring chunk without noise.
- Attention with per-token picks: mixed keys. A token computes its query, key, value and
  output projection under its own chunk; a query reads keys and values that other tokens wrote
  under theirs. Inference with one key/value cache runs exactly this way, and training runs
  the same computation. Recomputing keys under the querying token's chunk would need one
  cache per chunk, spending the memory the bank saves.
- Private weights: each module keeps its own two layer norms (1,024 weights per module). No
  other per-module add-on. Stored weights count every parameter: bank chunks, routers and
  their module vectors, layer norms, embeddings and the output head.
- Hardware: one RTX 2080, 8 GB, shared with other work, so wall times are indicative. During
  training a routed step computes every chunk for every token (the straight-through gradient
  needs them), which is why routed runs take longer; a hard pick needs only the picked chunk
  at inference, which this code does not exploit. `own-chunk` and `both-no-router` pay a
  bookkeeping cost per chunk that `full` does not, so their times are not comparable to it.

## Sanity check: N = M, every module forced to its own chunk

`own-chunk` runs the routed code path with a bank of 8 chunks of each kind and no router:
module i always computes with chunk i. It starts from the same initial weights and sees the
same batches as `full` at the same seed, so the comparison is paired by seed.

{{sanity}}

The per-seed differences (+0.0004, -0.0001, +0.0000) are about a fifth or less of the full control's
seed sd (0.0021): the routed code path with N = M reproduces the full control.

## Does the per-token pick matter?

`both-no-router` stores the same two banks as `both-routed` but fixes module i to chunk
i mod 4, so it has no router and no per-token choice. At the same seed the two see the same
batches (their initial weights differ):

{{no-router}}

## Picks at validation against picks in training

Training samples each pick with Gumbel noise; the reported loss takes the highest-scoring
chunk. If the routers had not committed, the two would differ widely. Mean over seeds:

{{sampled}}

## During training

Mean over seeds of validation loss on the first {{log-windows}} validation windows, to show
whether the ordering at the end of the budget was still moving.

{{during-training}}

## Every run

{{runs}}

## Per configuration

{{summary}}

## Picks

Measured on the validation split with the picks inference makes. The first column of numbers
has one entry per module, first module to last: the share of tokens whose pick is not that
module's most common pick (0 means the module always picks the same chunk, 0.75 is the most
four chunks allow). The next has the mean probability the router's softmax gives the chunk it
picks (0.25 is indifference, 1 is certainty). The usage columns pool all modules; evenness is
the entropy of usage divided by its maximum (1 is even).

### Feed-forward routed

{{picks:ff-routed}}

### Attention routed

{{picks:attn-routed}}

### Both routed

{{picks:both-routed}}

## Reproducing

`nix-shell` gives the environment. `python -m weightbank.train <configuration> --seed <seed>`
reproduces one run, `python -m weightbank.sweep` runs every run missing from `runs/`, and
`python -m weightbank.report` rebuilds this file and the chart.
