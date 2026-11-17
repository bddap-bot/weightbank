# Routing into a shared weight bank against smaller plain models

Every table below is written by `python -m weightbank.report` from the run files in `runs/`.
Loss is cross-entropy on the validation split in nats per byte (divide by 0.693 for bits).

Two experiments are reported together. The first trained every configuration for 8,000 steps
at one learning rate (1e-3), three seeds. The second asks which of the first's gaps belong to
the designs and which to a short or mis-tuned comparison: every configuration gets a learning
rate chosen for it from a sweep, then three seeds at four times the budget (32,000 steps);
[what was cut](#what-was-cut) lists the two configurations left out.
A gap is a configuration's mean validation loss minus the better of the two plain curves read
at its stored weights; negative means the configuration beats a plain model of its size.

## Reading

READING

![Validation loss against stored weights](chart.png)

The chart is the long budget. The dark line is the plain model narrowed from width 256 to 160
at 8 modules; the lighter line is the plain model cut from 8 modules to 4 at width 256. Bands
and bars run from the lowest to the highest of three seeds. The square is a control outside
the question: both banks shared with no router, module i always using chunk i mod 4.

## The gaps at both budgets

Each plain curve is read at the configuration's stored weights by interpolating mean loss
linearly in the logarithm of stored weights, and the gap is taken to whichever curve is lower
there. Seeds share batches across configurations, so the gap is also computed seed by seed
(the configuration at seed s against that same curve drawn from the plain runs of seed s);
the range of those three is beside each mean.

{{both-budgets}}

Between the two budgets both the learning rate and the number of steps change. The next
table changes the rate alone: the first budget, seed 0 only, with every configuration at 1e-3
and then at its chosen rate, both read against the plain curve the first-budget gap above
uses. These are the sweep's own runs, so each loss in the right-hand gap is the lowest of the
three or more that were tried, and a one-seed gap carries about a seed spread of noise (0.002
to 0.008 in the per-configuration table); it can show a shift of the size of the two large
gaps and not of the two small ones.

{{rate-alone}}

The plain curves the gaps are read against, at both budgets:

{{plain}}

### Long budget, both plain curves

{{gaps-long}}

### First budget, both plain curves

{{gaps-first}}

`both-routed` and `plain-l4` are an equal-storage pair that needs no interpolation: the same 4
feed-forward and 4 attention chunks, applied 8 times with routing or 4 times without. The
routed model adds 38,536 weights (two routers and four more modules' layer norms).

## Choosing the learning rate

Each configuration was trained once (seed 0, 8,000 steps) at peak rates 5e-4, 1e-3 and 2e-3;
the 1e-3 run is the first experiment's seed-0 run. When the lowest final validation loss fell
on the edge of the rates tried, the next rate a factor of two beyond it was added, until the
lowest loss had a tried rate on both sides. The rate with the lowest loss (bold) is the one
the configuration's three long runs use. The warm-up (200 steps) and the cosine decay to a
tenth of the peak are the same at every rate and budget.

{{sweep}}

Limits of this choice: it is made on one seed, so where two rates differ by less than the seed
spread the pick between them is arbitrary; and it is made at 8,000 steps, where a short
schedule rewards a high peak rate, on the assumption that the best peak rate does not move
much when the cosine schedule is stretched to 32,000.

That assumption is tested where it matters most, on the two configurations that sat above
the plain curve: one more long run each (seed 0, 32,000 steps) at half the chosen rate. If
the lower rate were better at the long budget, these gaps would still hold a mis-tuned part.

{{lower-rate}}

## What was cut

- `own-chunk`, the check that the routed code path with N = M reproduces the full control,
  was not repeated. It tests the code, which did not change, and not a design.
- `plain-w128` (1,684,992 stored weights) was left out of the sweep and the long runs. The
  smallest configuration compared against the curve stores 3,360,512 weights, which lies
  between `plain-w160` and `plain-w192`, so no gap reads the curve below `plain-w160`.
- The lower-rate check at the long budget covers the two configurations above the plain
  curve and not the plain models or the other two, to keep the experiment near a day of the
  GPU.
- Nothing else: every other configuration of the first experiment has its sweep and three
  long seeds.

## Setup

- Data: enwik8, the first 100 MB of an English Wikipedia dump (text under CC BY-SA 3.0 and
  GFDL). Bytes are the tokens (vocabulary 256). The first 90 MB train, the next 5 MB validate;
  validation loss is over all 19,531 non-overlapping 256-byte windows of that split.
- Model: pre-norm decoder-only transformer, 8 modules, width 256, 4 heads, context 256,
  learned positions. A module is one layer: an attention step and a feed-forward step.
- Budget: 8,000 steps of 64 windows in the first experiment and 32,000 in the second, AdamW
  (peak rate 1e-3 in the first experiment and the chosen rate in the second, 200 warm-up
  steps, cosine decay to a tenth of the peak over the run, weight decay 0.1 on matrices and
  embeddings), mixed precision (float16 autocast), gradient clipping at 1. Seeds 0, 1, 2 set
  both the initial weights and the batches.
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
- Hardware: one RTX 2080, 8 GB, shared with other work, so wall times are indicative. Runs
  were trained one at a time: one run saturates this GPU (three plain runs at once each took
  three times as long). During
  training a routed step computes every chunk for every token (the straight-through gradient
  needs them), which is why routed runs take longer; a hard pick needs only the picked chunk
  at inference, which this code does not exploit. `own-chunk` and `both-no-router` pay a
  bookkeeping cost per chunk that `full` does not, so their times are not comparable to it.

## Sanity check: N = M, every module forced to its own chunk

This check belongs to the first experiment (8,000 steps, rate 1e-3) and was not repeated at
the long budget. `own-chunk` runs the routed code path with a bank of 8 chunks of each kind and no router:
module i always computes with chunk i. It starts from the same initial weights and sees the
same batches as `full` at the same seed, so the comparison is paired by seed.

{{sanity}}

The per-seed differences (+0.0004, -0.0001, +0.0000) are about a fifth or less of the full control's
seed sd (0.0021): the routed code path with N = M reproduces the full control.

## Does the per-token pick matter?

`both-no-router` stores the same two banks as `both-routed` but fixes module i to chunk
i mod 4, so it has no router and no per-token choice. At the same seed the two see the same
batches (their initial weights differ). At the long budget, each at its chosen rate:

{{no-router-long}}

At the first budget:

{{no-router-first}}

## Picks at validation against picks in training

Training samples each pick with Gumbel noise; the reported loss takes the highest-scoring
chunk. If the routers had not committed, the two would differ widely. Mean over seeds, long
budget:

{{sampled}}

## During training

Long budget: mean over seeds of validation loss on the first {{log-windows}} validation
windows, to show whether the ordering at the end of the budget was still moving.

{{during-training}}

## How much picks vary by input

Measured on the validation split with the picks inference makes. For each module, the share
of tokens whose pick is not that module's most common pick says how much the pick depends on
the input: 0 means the module always picks the same chunk, 0.75 is the most four chunks
allow. The table gives the mean of that share over modules and seeds, its lowest value in any
module of any seed, and the mean probability the router's softmax gives the chunk it picks
(0.25 is indifference, 1 is certainty), at both budgets.

{{pick-variation}}

The tables below are the long budget in full. The first column of numbers has one entry per
module, first module to last: the share of tokens off that module's most common pick. The
next has the router's mean probability for its pick. The usage columns pool all modules;
evenness is the entropy of usage divided by its maximum (1 is even).

### Feed-forward routed

{{picks:ff-routed}}

### Attention routed

{{picks:attn-routed}}

### Both routed

{{picks:both-routed}}

## Per configuration

Long budget:

{{summary-long}}

First budget:

{{summary-first}}

## Every run

Both experiments and the learning-rate sweep, from the run files.

{{runs}}

## Reproducing

`nix-shell` gives the environment. One command reproduces any run in the table above:

```sh
python -m weightbank.train <configuration> --seed <seed> --steps <steps> --rate <learning rate>
```

It writes `runs/<steps>/<configuration>-lr<learning rate>-s<seed>.json`.
`python -m weightbank.sweep` trains every run of both experiments that `runs/` lacks (the
first experiment, the learning-rate sweep, the long runs at the rates the sweep chose, then
the lower-rate check), and `python -m weightbank.report` rebuilds this file and the
chart.
