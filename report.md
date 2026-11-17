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

| configuration | stored weights | mean loss | seed sd | narrower plain at equal stored weights | minus narrower | shallower plain at equal stored weights | minus shallower | minus the better plain |
|---|---|---|---|---|---|---|---|---|
| ff-routed | 4,430,404 | 1.0977 | 0.0047 | 1.1057 | -0.0080 | 1.1010 | -0.0032 | -0.0032 |
| attn-routed | 5,480,004 | 1.1146 | 0.0024 | 1.0903 | +0.0242 | 1.0882 | +0.0263 | +0.0263 |
| both-routed | 3,394,952 | 1.1452 | 0.0041 | 1.1268 | +0.0184 | 1.1152 | +0.0300 | +0.0300 |
| both-no-router | 3,360,512 | 1.1004 | 0.0049 | 1.1278 | -0.0274 | 1.1157 | -0.0153 | -0.0153 |

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

| configuration | stored weights | seed 0 | seed 1 | seed 2 | mean | seed sd |
|---|---|---|---|---|---|---|
| full | 6,515,456 | 1.0784 | 1.0744 | 1.0772 | 1.0767 | 0.0021 |
| own-chunk | 6,515,456 | 1.0788 | 1.0743 | 1.0773 | 1.0768 | 0.0023 |
| own-chunk minus full |  | +0.0004 | -0.0001 | +0.0000 | +0.0001 |  |

The per-seed differences (+0.0004, -0.0001, +0.0000) are about a fifth or less of the full control's
seed sd (0.0021): the routed code path with N = M reproduces the full control.

## Does the per-token pick matter?

`both-no-router` stores the same two banks as `both-routed` but fixes module i to chunk
i mod 4, so it has no router and no per-token choice. At the same seed the two see the same
batches (their initial weights differ):

| configuration | stored weights | seed 0 | seed 1 | seed 2 | mean | seed sd |
|---|---|---|---|---|---|---|
| both-no-router | 3,360,512 | 1.0976 | 1.0975 | 1.1061 | 1.1004 | 0.0049 |
| both-routed | 3,394,952 | 1.1455 | 1.1492 | 1.1409 | 1.1452 | 0.0041 |
| both-routed minus both-no-router |  | +0.0479 | +0.0516 | +0.0348 | +0.0448 |  |

## Picks at validation against picks in training

Training samples each pick with Gumbel noise; the reported loss takes the highest-scoring
chunk. If the routers had not committed, the two would differ widely. Mean over seeds:

| configuration | highest-scoring picks (reported) | picks sampled as in training |
|---|---|---|
| ff-routed | 1.0977 | 1.0995 |
| attn-routed | 1.1146 | 1.1163 |
| both-routed | 1.1452 | 1.1485 |

## During training

Mean over seeds of validation loss on the first 2,048 validation windows, to show
whether the ordering at the end of the budget was still moving.

| configuration | step 2,000 | step 4,000 | step 6,000 | step 8,000 |
|---|---|---|---|---|
| full | 1.3028 | 1.1850 | 1.1268 | 1.0989 |
| plain-w128 | 1.4490 | 1.3077 | 1.2459 | 1.2176 |
| plain-w160 | 1.3965 | 1.2646 | 1.2056 | 1.1778 |
| plain-w192 | 1.3498 | 1.2282 | 1.1678 | 1.1407 |
| plain-w224 | 1.3259 | 1.2055 | 1.1465 | 1.1195 |
| plain-l4 | 1.3335 | 1.2241 | 1.1655 | 1.1392 |
| plain-l6 | 1.3146 | 1.2039 | 1.1452 | 1.1180 |
| ff-routed | 1.3352 | 1.2123 | 1.1495 | 1.1211 |
| attn-routed | 1.3730 | 1.2289 | 1.1641 | 1.1334 |
| both-routed | 1.4193 | 1.2677 | 1.1959 | 1.1640 |
| own-chunk | 1.2995 | 1.1850 | 1.1258 | 1.0984 |
| both-no-router | 1.3358 | 1.2117 | 1.1512 | 1.1218 |

## Every run

| configuration | seed | stored weights | final validation loss | wall seconds (training) |
|---|---|---|---|---|
| full | 0 | 6,515,456 | 1.0784 | 492 |
| full | 1 | 6,515,456 | 1.0744 | 409 |
| full | 2 | 6,515,456 | 1.0772 | 426 |
| plain-w128 | 0 | 1,684,992 | 1.1949 | 261 |
| plain-w128 | 1 | 1,684,992 | 1.1867 | 207 |
| plain-w128 | 2 | 1,684,992 | 1.2000 | 243 |
| plain-w160 | 0 | 2,597,696 | 1.1503 | 342 |
| plain-w160 | 1 | 2,597,696 | 1.1540 | 282 |
| plain-w160 | 2 | 2,597,696 | 1.1589 | 310 |
| plain-w192 | 0 | 3,707,008 | 1.1194 | 443 |
| plain-w192 | 1 | 3,707,008 | 1.1191 | 321 |
| plain-w192 | 2 | 3,707,008 | 1.1146 | 348 |
| plain-w224 | 0 | 5,012,928 | 1.0915 | 479 |
| plain-w224 | 1 | 5,012,928 | 1.0979 | 373 |
| plain-w224 | 2 | 5,012,928 | 1.1027 | 373 |
| plain-l4 | 0 | 3,356,416 | 1.1113 | 260 |
| plain-l4 | 1 | 3,356,416 | 1.1146 | 218 |
| plain-l4 | 2 | 3,356,416 | 1.1214 | 208 |
| plain-l6 | 0 | 4,935,936 | 1.1021 | 565 |
| plain-l6 | 1 | 4,935,936 | 1.0864 | 304 |
| plain-l6 | 2 | 4,935,936 | 1.0971 | 304 |
| ff-routed | 0 | 4,430,404 | 1.1011 | 2010 |
| ff-routed | 1 | 4,430,404 | 1.0998 | 964 |
| ff-routed | 2 | 4,430,404 | 1.0923 | 923 |
| attn-routed | 0 | 5,480,004 | 1.1166 | 1050 |
| attn-routed | 1 | 5,480,004 | 1.1119 | 1011 |
| attn-routed | 2 | 5,480,004 | 1.1152 | 919 |
| both-routed | 0 | 3,394,952 | 1.1455 | 1602 |
| both-routed | 1 | 3,394,952 | 1.1492 | 1586 |
| both-routed | 2 | 3,394,952 | 1.1409 | 1454 |
| own-chunk | 0 | 6,515,456 | 1.0788 | 533 |
| own-chunk | 1 | 6,515,456 | 1.0743 | 550 |
| own-chunk | 2 | 6,515,456 | 1.0773 | 489 |
| both-no-router | 0 | 3,360,512 | 1.0976 | 494 |
| both-no-router | 1 | 3,360,512 | 1.0975 | 501 |
| both-no-router | 2 | 3,360,512 | 1.1061 | 504 |

## Per configuration

| configuration | stored weights | mean loss | seed sd | lowest | highest |
|---|---|---|---|---|---|
| full | 6,515,456 | 1.0767 | 0.0021 | 1.0744 | 1.0784 |
| plain-w128 | 1,684,992 | 1.1939 | 0.0067 | 1.1867 | 1.2000 |
| plain-w160 | 2,597,696 | 1.1544 | 0.0043 | 1.1503 | 1.1589 |
| plain-w192 | 3,707,008 | 1.1177 | 0.0027 | 1.1146 | 1.1194 |
| plain-w224 | 5,012,928 | 1.0974 | 0.0056 | 1.0915 | 1.1027 |
| plain-l4 | 3,356,416 | 1.1158 | 0.0052 | 1.1113 | 1.1214 |
| plain-l6 | 4,935,936 | 1.0952 | 0.0080 | 1.0864 | 1.1021 |
| ff-routed | 4,430,404 | 1.0977 | 0.0047 | 1.0923 | 1.1011 |
| attn-routed | 5,480,004 | 1.1146 | 0.0024 | 1.1119 | 1.1166 |
| both-routed | 3,394,952 | 1.1452 | 0.0041 | 1.1409 | 1.1492 |
| own-chunk | 6,515,456 | 1.0768 | 0.0023 | 1.0743 | 1.0788 |
| both-no-router | 3,360,512 | 1.1004 | 0.0049 | 1.0975 | 1.1061 |

## Picks

Measured on the validation split with the picks inference makes. The first column of numbers
has one entry per module, first module to last: the share of tokens whose pick is not that
module's most common pick (0 means the module always picks the same chunk, 0.75 is the most
four chunks allow). The next has the mean probability the router's softmax gives the chunk it
picks (0.25 is indifference, 1 is certainty). The usage columns pool all modules; evenness is
the entropy of usage divided by its maximum (1 is even).

### Feed-forward routed

| bank | seed | share of tokens off the module's most common pick, per module | router's mean probability for its pick, per module | share of all picks, per chunk | usage evenness |
|---|---|---|---|---|---|
| feed-forward | 0 | 0.48 0.66 0.64 0.64 0.61 0.66 0.63 0.65 | 0.96 0.96 0.94 0.92 0.91 0.91 0.92 0.89 | 0.23 0.31 0.22 0.24 | 0.99 |
| feed-forward | 1 | 0.20 0.68 0.70 0.71 0.72 0.70 0.69 0.69 | 0.99 0.96 0.93 0.92 0.92 0.91 0.93 0.91 | 0.21 0.22 0.21 0.36 | 0.98 |
| feed-forward | 2 | 0.41 0.63 0.66 0.66 0.65 0.66 0.68 0.67 | 0.98 0.98 0.96 0.95 0.94 0.95 0.95 0.93 | 0.30 0.20 0.21 0.29 | 0.99 |

### Attention routed

| bank | seed | share of tokens off the module's most common pick, per module | router's mean probability for its pick, per module | share of all picks, per chunk | usage evenness |
|---|---|---|---|---|---|
| attention | 0 | 0.38 0.28 0.30 0.38 0.48 0.51 0.52 0.52 | 0.45 0.89 0.91 0.95 0.96 0.96 0.95 0.94 | 0.20 0.08 0.58 0.15 | 0.81 |
| attention | 1 | 0.33 0.40 0.39 0.40 0.43 0.44 0.48 0.47 | 0.42 0.91 0.95 0.97 0.98 0.98 0.96 0.93 | 0.15 0.12 0.58 0.15 | 0.82 |
| attention | 2 | 0.54 0.48 0.49 0.47 0.47 0.43 0.46 0.53 | 0.44 0.74 0.79 0.94 0.97 0.98 0.97 0.94 | 0.15 0.23 0.12 0.50 | 0.88 |

### Both routed

| bank | seed | share of tokens off the module's most common pick, per module | router's mean probability for its pick, per module | share of all picks, per chunk | usage evenness |
|---|---|---|---|---|---|
| feed-forward | 0 | 0.24 0.22 0.65 0.68 0.69 0.69 0.69 0.66 | 0.56 0.89 0.91 0.92 0.94 0.94 0.93 0.91 | 0.20 0.26 0.42 0.13 | 0.94 |
| feed-forward | 1 | 0.00 0.27 0.54 0.58 0.61 0.63 0.66 0.66 | 0.73 0.86 0.94 0.95 0.95 0.95 0.93 0.91 | 0.34 0.23 0.12 0.32 | 0.95 |
| feed-forward | 2 | 0.00 0.33 0.54 0.55 0.60 0.62 0.65 0.65 | 1.00 0.89 0.94 0.95 0.95 0.95 0.94 0.92 | 0.38 0.13 0.27 0.22 | 0.95 |
| attention | 0 | 0.19 0.26 0.51 0.59 0.59 0.62 0.57 0.59 | 0.45 0.89 0.92 0.96 0.97 0.96 0.96 0.93 | 0.29 0.06 0.16 0.49 | 0.85 |
| attention | 1 | 0.62 0.42 0.46 0.45 0.46 0.49 0.58 0.57 | 0.38 0.86 0.94 0.98 0.98 0.97 0.95 0.92 | 0.20 0.45 0.27 0.09 | 0.90 |
| attention | 2 | 0.12 0.49 0.53 0.54 0.54 0.54 0.51 0.50 | 0.50 0.93 0.96 0.97 0.97 0.97 0.95 0.92 | 0.18 0.14 0.53 0.14 | 0.87 |

## Reproducing

`nix-shell` gives the environment. `python -m weightbank.train <configuration> --seed <seed>`
reproduces one run, `python -m weightbank.sweep` runs every run missing from `runs/`, and
`python -m weightbank.report` rebuilds this file and the chart.
