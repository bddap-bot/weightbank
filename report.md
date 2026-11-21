# Routing into a shared weight bank against smaller plain models

Every table below is written by `python -m weightbank.report` from the run files in `runs/`.
Loss is cross-entropy on the validation split in nats per byte (divide by 0.693 for bits).

Two experiments are reported together. The first trained every configuration for 8,000 steps
at one learning rate (1e-3), three seeds. The second asks which of the first's gaps belong to
the designs and which to a short or mis-tuned comparison: every configuration gets a learning
rate chosen for it by sweeps, then three seeds at four times the budget (32,000 steps). The
second also adds two families: two other routers for the model with both banks routed, and
the fixed module-to-chunk map with no router at 1, 2, 4 and 8 chunks.
A gap is a configuration's mean validation loss minus the better of the two plain curves read
at its stored weights; negative means the configuration beats a plain model of its size.

## Reading

Three findings of the first experiment held in sign and shrank in size: attention routed
stays above the plain curve (+0.026 at the first budget, +0.021 at the long one), both routed
stays above it (+0.030, now +0.012), and the fixed map with no router stays below it (-0.015,
now -0.008) and still beats both routed on the same banks, by 0.019 where it was 0.045. One
moved: feed-forward routed went from -0.003, inside the seed spread, to +0.005 with all three
seeds above the curve (+0.003 to +0.008), so at the long budget it is slightly worse than a
plain model of its size and no longer level with it; part of that is the plain curve, since
at tuned rates the narrowed plain model became the better curve, and against the curve of
fewer modules, the better one at the first budget, feed-forward routed sits at -0.001. The
signs are therefore properties of the designs, while both routed lost three fifths of its
gap, the fixed map half and attention routed a fifth to the longer budget and tuned rates:
choosing a chunk per token costs loss against a plain model and against fixed sharing of the
same weights, and fixed sharing is the only arrangement here that beats a plain model of
equal storage.

![Validation loss against stored weights](chart.png)

The chart is the long budget, whole on the left and enlarged between 3 and 7 million stored
weights on the right. The dark line is the plain model narrowed from width 256 to 96 at 8
modules; the lighter line is the plain model cut from 8 modules to 1 at width 256. Bands and
bars run from the lowest to the highest of three seeds. The dashed line joins the fixed
module-to-chunk map with no router at 1, 2, 4 and 8 chunks. The fixed map at 4 chunks and the
three models with both banks routed store the same weights to within 1.2%; their four points
are drawn side by side so that none hides another.

## The gaps at both budgets

Each plain curve is read at the configuration's stored weights by interpolating mean loss
linearly in the logarithm of stored weights, and the gap is taken to whichever curve is lower
there. Seeds share batches across configurations, so the gap is also computed seed by seed
(the configuration at seed s against that same curve drawn from the plain runs of seed s);
the range of those three is beside each mean. At the long budget seed 0 is also the seed the
learning rates were chosen on, so each seed-0 loss is the lowest of the rates tried; the
column without seed 0 is the gap over the two seeds that took no part in the choice.

| configuration | stored weights | learning rate, 8,000 steps | gap, 8,000 steps | lowest to highest seed | learning rate, 32,000 steps | gap, 32,000 steps | lowest to highest seed | gap without seed 0 | long minus first |
|---|---|---|---|---|---|---|---|---|---|
| ff-routed | 4,430,404 | 0.001 | -0.0032 | -0.0116 to +0.0055 | 0.002 | +0.0046 | +0.0025 to +0.0078 | +0.0029 | +0.0078 |
| attn-routed | 5,480,004 | 0.001 | +0.0263 | +0.0234 to +0.0301 | 0.002 | +0.0207 | +0.0159 to +0.0247 | +0.0203 | -0.0056 |
| both-routed | 3,394,952 | 0.001 | +0.0300 | +0.0202 to +0.0354 | 0.002 | +0.0116 | +0.0107 to +0.0124 | +0.0111 | -0.0185 |
| both-routed-argmax | 3,394,952 | 0.001 | +0.0429 | +0.0389 to +0.0484 | 0.002 | +0.0253 | +0.0225 to +0.0291 | +0.0235 | -0.0176 |
| both-routed-keys | 3,398,016 | 0.001 | +0.0355 | +0.0281 to +0.0492 | 0.002 | +0.0115 | +0.0083 to +0.0154 | +0.0096 | -0.0240 |
| both-no-router-c1 | 994,304 | 0.001 | -0.0530 | -0.0550 to -0.0499 | 0.002 | -0.0384 | -0.0397 to -0.0371 | -0.0384 | +0.0146 |
| both-no-router-c2 | 1,783,040 | 0.001 | -0.0388 | -0.0433 to -0.0313 | 0.001 | -0.0147 | -0.0232 to -0.0053 | -0.0105 | +0.0240 |
| both-no-router | 3,360,512 | 0.001 | -0.0153 | -0.0170 to -0.0136 | 0.002 | -0.0080 | -0.0093 to -0.0067 | -0.0080 | +0.0073 |
| own-chunk | 6,515,456 | 0.001 | +0.0001 | -0.0001 to +0.0004 | 0.002 | +0.0008 | -0.0005 to +0.0025 | -0.0001 | +0.0007 |

Between the two budgets both the learning rate and the number of steps change. The next
table changes the rate alone: the first budget, seed 0 only, with every configuration at 1e-3
and then at the best of the three rates tried for it at that budget (the short sweep below),
both read against the plain curve the first-budget gap above uses. Each loss on the
right-hand side is the lowest of three, and a one-seed gap carries about a seed spread of
noise (0.002 to 0.008 in the per-configuration table); it can show a shift of the size of
the two large gaps and not of the two small ones.

| configuration | plain curve | gap at 0.001 everywhere | best short rate | gap at the best short rates | best minus first |
|---|---|---|---|---|---|
| ff-routed | shallower | -0.0036 | 0.002 | +0.0074 | +0.0110 |
| attn-routed | shallower | +0.0234 | 0.002 | +0.0295 | +0.0061 |
| both-routed | shallower | +0.0345 | 0.002 | +0.0232 | -0.0114 |
| both-routed-argmax | shallower | +0.0484 | 0.002 | +0.0384 | -0.0101 |
| both-routed-keys | shallower | +0.0492 | 0.002 | +0.0277 | -0.0215 |
| both-no-router-c1 | narrower | -0.0499 | 0.002 | -0.0479 | +0.0020 |
| both-no-router-c2 | shallower | -0.0433 | 0.002 | -0.0460 | -0.0027 |
| both-no-router | shallower | -0.0136 | 0.002 | -0.0180 | -0.0044 |

The plain curves the gaps are read against, at both budgets:

| configuration | stored weights | loss, 8,000 steps at 0.001 | seed sd | chosen learning rate | loss, 32,000 steps | seed sd | long minus first |
|---|---|---|---|---|---|---|---|
| plain-w96 | 968,896 | 1.2616 | 0.0040 | 0.002 | 1.1599 | 0.0010 | -0.1017 |
| plain-w128 | 1,684,992 | 1.1939 | 0.0067 | 0.004 | 1.0779 | 0.0060 | -0.1160 |
| plain-w160 | 2,597,696 | 1.1544 | 0.0043 | 0.004 | 1.0413 | 0.0017 | -0.1130 |
| plain-w192 | 3,707,008 | 1.1177 | 0.0027 | 0.002 | 1.0138 | 0.0013 | -0.1039 |
| plain-w224 | 5,012,928 | 1.0974 | 0.0056 | 0.002 | 0.9980 | 0.0038 | -0.0994 |
| plain-l1 | 987,136 | 1.4265 | 0.0017 | 0.002 | 1.3775 | 0.0013 | -0.0490 |
| plain-l2 | 1,776,896 | 1.1775 | 0.0025 | 0.002 | 1.1184 | 0.0004 | -0.0591 |
| plain-l4 | 3,356,416 | 1.1158 | 0.0052 | 0.002 | 1.0320 | 0.0013 | -0.0838 |
| plain-l6 | 4,935,936 | 1.0952 | 0.0080 | 0.002 | 1.0019 | 0.0022 | -0.0933 |
| full | 6,515,456 | 1.0767 | 0.0021 | 0.002 | 0.9833 | 0.0012 | -0.0933 |

### Long budget, both plain curves

| configuration | learning rate | stored weights | mean loss | seed sd | narrower plain at equal stored weights | minus narrower | shallower plain at equal stored weights | minus shallower | minus the better plain |
|---|---|---|---|---|---|---|---|---|---|
| ff-routed | 0.002 | 4,430,404 | 1.0090 | 0.0014 | 1.0045 | +0.0046 | 1.0103 | -0.0013 | +0.0046 |
| attn-routed | 0.002 | 5,480,004 | 1.0137 | 0.0032 | 0.9930 | +0.0207 | 0.9949 | +0.0188 | +0.0207 |
| both-routed | 0.002 | 3,394,952 | 1.0322 | 0.0013 | 1.0206 | +0.0116 | 1.0311 | +0.0011 | +0.0116 |
| both-routed-argmax | 0.002 | 3,394,952 | 1.0459 | 0.0021 | 1.0206 | +0.0253 | 1.0311 | +0.0148 | +0.0253 |
| both-routed-keys | 0.002 | 3,398,016 | 1.0321 | 0.0022 | 1.0205 | +0.0115 | 1.0310 | +0.0010 | +0.0115 |
| both-no-router-c1 | 0.002 | 994,304 | 1.1177 | 0.0008 | 1.1560 | -0.0384 | 1.3743 | -0.2567 | -0.0384 |
| both-no-router-c2 | 0.001 | 1,783,040 | 1.0583 | 0.0039 | 1.0731 | -0.0147 | 1.1179 | -0.0595 | -0.0147 |
| both-no-router | 0.002 | 3,360,512 | 1.0134 | 0.0012 | 1.0214 | -0.0080 | 1.0319 | -0.0185 | -0.0080 |
| own-chunk | 0.002 | 6,515,456 | 0.9841 | 0.0008 | 0.9833 | +0.0008 | 0.9833 | +0.0008 | +0.0008 |

### First budget, both plain curves

| configuration | learning rate | stored weights | mean loss | seed sd | narrower plain at equal stored weights | minus narrower | shallower plain at equal stored weights | minus shallower | minus the better plain |
|---|---|---|---|---|---|---|---|---|---|
| ff-routed | 0.001 | 4,430,404 | 1.0977 | 0.0047 | 1.1057 | -0.0080 | 1.1010 | -0.0032 | -0.0032 |
| attn-routed | 0.001 | 5,480,004 | 1.1146 | 0.0024 | 1.0903 | +0.0242 | 1.0882 | +0.0263 | +0.0263 |
| both-routed | 0.001 | 3,394,952 | 1.1452 | 0.0041 | 1.1268 | +0.0184 | 1.1152 | +0.0300 | +0.0300 |
| both-routed-argmax | 0.001 | 3,394,952 | 1.1581 | 0.0024 | 1.1268 | +0.0313 | 1.1152 | +0.0429 | +0.0429 |
| both-routed-keys | 0.001 | 3,398,016 | 1.1506 | 0.0088 | 1.1267 | +0.0239 | 1.1151 | +0.0355 | +0.0355 |
| both-no-router-c1 | 0.001 | 994,304 | 1.2054 | 0.0056 | 1.2584 | -0.0530 | 1.4235 | -0.2181 | -0.0530 |
| both-no-router-c2 | 0.001 | 1,783,040 | 1.1384 | 0.0073 | 1.1887 | -0.0503 | 1.1771 | -0.0388 | -0.0388 |
| both-no-router | 0.001 | 3,360,512 | 1.1004 | 0.0049 | 1.1278 | -0.0274 | 1.1157 | -0.0153 | -0.0153 |
| own-chunk | 0.001 | 6,515,456 | 1.0768 | 0.0023 | 1.0767 | +0.0001 | 1.0767 | +0.0001 | +0.0001 |

`both-routed` and `plain-l4` are an equal-storage pair that needs no interpolation: the same 4
feed-forward and 4 attention chunks, applied 8 times with routing or 4 times without. The
routed model adds 38,536 weights (two routers and four more modules' layer norms).

## Choosing the learning rate

The rate is chosen in two stages, each one seed (seed 0), and the choice is made at the
budget it is used at. The warm-up (200 steps) and the cosine decay to a tenth of the peak are
the same at every rate and budget.

Short sweep: 8,000 steps at peak rates 5e-4, 1e-3 and 2e-3. The 1e-3 run is the first
experiment's seed-0 run. Lowest loss in bold; the last column is how far the next-best rate
is behind it.

| configuration | 0.0005 | 0.001 | 0.002 | lowest | lead over the next |
|---|---|---|---|---|---|
| full | 1.1072 | 1.0784 | **1.0538** | 0.002 | 0.0246 |
| plain-w96 | 1.3421 | 1.2638 | **1.2306** | 0.002 | 0.0332 |
| plain-w128 | 1.2433 | 1.1949 | **1.1674** | 0.002 | 0.0274 |
| plain-w160 | 1.1878 | 1.1503 | **1.1185** | 0.002 | 0.0318 |
| plain-w192 | 1.1527 | 1.1194 | **1.0837** | 0.002 | 0.0357 |
| plain-w224 | 1.1208 | 1.0915 | **1.0648** | 0.002 | 0.0267 |
| plain-l1 | 1.4541 | 1.4254 | **1.4204** | 0.002 | 0.0050 |
| plain-l2 | 1.2171 | 1.1750 | **1.1659** | 0.002 | 0.0091 |
| plain-l4 | 1.1657 | 1.1113 | **1.0984** | 0.002 | 0.0129 |
| plain-l6 | 1.1317 | 1.1021 | **1.0743** | 0.002 | 0.0278 |
| ff-routed | 1.1460 | 1.1011 | **1.0884** | 0.002 | 0.0126 |
| attn-routed | 1.1429 | 1.1166 | **1.0961** | 0.002 | 0.0205 |
| both-routed | 1.1906 | 1.1455 | **1.1208** | 0.002 | 0.0247 |
| both-routed-argmax | 1.2087 | 1.1594 | **1.1360** | 0.002 | 0.0234 |
| both-routed-keys | 1.1991 | 1.1602 | **1.1253** | 0.002 | 0.0349 |
| both-no-router-c1 | 1.2404 | 1.2107 | **1.1797** | 0.002 | 0.0310 |
| both-no-router-c2 | 1.1882 | 1.1313 | **1.1195** | 0.002 | 0.0118 |
| both-no-router | 1.1466 | 1.0976 | **1.0802** | 0.002 | 0.0174 |

Every configuration's best short rate is 2e-3, the highest of the three, by 0.005 to 0.036
over 1e-3: a short budget rewards a high rate. A rate chosen here would therefore say more
about the budget than about the configuration, which is why the short sweep only names where
the long sweep starts, and why the first experiment's single rate of 1e-3 was low for every
configuration at once.

Long sweep: 32,000 steps at the best short rate and the better of its neighbours in the
short sweep. The rate with the lowest long loss is the configuration's rate, and its run is
the configuration's seed-0 long run; seeds 1 and 2 are then trained at that rate. A winner on
the edge of the rates tried gets the next rate a factor of two beyond it added, and the rule
is applied again, in two cases: when the winner is not the best short rate (the long budget
moved the optimum, so that side is followed until a tried rate lies beyond the winner), and
when it leads the next-best rate by more than 0.005. A run whose training loss stops being
finite is recorded with an infinite loss and loses.

| configuration | 0.0005 | 0.001 | 0.002 | 0.004 | 0.008 | lowest | lead over the next |
|---|---|---|---|---|---|---|---|
| full |  | 1.0003 | **0.9819** | 0.9840 |  | 0.002 | 0.0021 |
| plain-w96 |  | 1.1649 | **1.1602** |  |  | 0.002 | 0.0047 |
| plain-w128 |  | 1.1100 | 1.1017 | **1.0838** | 1.0862 | 0.004 | 0.0024 |
| plain-w160 |  | 1.0679 | 1.0484 | **1.0399** | 1.0499 | 0.004 | 0.0085 |
| plain-w192 |  | 1.0406 | **1.0125** | 1.0145 |  | 0.002 | 0.0020 |
| plain-w224 |  | 1.0124 | **0.9946** | 0.9991 |  | 0.002 | 0.0046 |
| plain-l1 |  | 1.3788 | **1.3764** |  |  | 0.002 | 0.0024 |
| plain-l2 |  | 1.1195 | **1.1182** |  |  | 0.002 | 0.0013 |
| plain-l4 |  | 1.0388 | **1.0317** | 1.0372 |  | 0.002 | 0.0055 |
| plain-l6 |  | 1.0215 | **1.0043** | 1.0072 |  | 0.002 | 0.0029 |
| ff-routed |  | 1.0123 | **1.0098** |  |  | 0.002 | 0.0025 |
| attn-routed |  | 1.0163 | **1.0117** |  |  | 0.002 | 0.0046 |
| both-routed |  | 1.0394 | **1.0316** | 1.0366 |  | 0.002 | 0.0049 |
| both-routed-argmax |  | 1.0593 | **1.0483** | 1.2190 |  | 0.002 | 0.0110 |
| both-routed-keys |  | 1.0477 | **1.0346** | 1.0362 |  | 0.002 | 0.0016 |
| both-no-router-c1 |  | 1.1267 | **1.1184** | 1.1207 |  | 0.002 | 0.0023 |
| both-no-router-c2 | 1.0752 | **1.0548** | 1.0553 |  |  | 0.001 | 0.0004 |
| both-no-router |  | 1.0167 | **1.0122** |  |  | 0.002 | 0.0045 |

Limits of this choice: it is made on one seed, so where two rates differ by less than the
seed spread the pick between them is arbitrary; and a winner that is the best short rate
and leads by 0.005 or less has no rate tried beyond it, so a rate beyond it could be better
by a similar amount. The two small gaps of the first experiment are of that size.

## What was cut

- The long sweep starts from two rates per configuration and adds more only under the rule
  above. Three long rates for every configuration would have added about six hours of the
  GPU.
- `own-chunk` has no sweep of its own: it is the full control computed through the bank code,
  so it trains at the full control's rate, which keeps the two paired.
- The two other routers were run for the model with both banks routed only. Feed-forward-only
  and attention-only versions of them would be four more configurations, about 21 hours of
  the GPU, and the both-routed pair already answers what they were added to ask.
- No configuration was trained under another optimizer. A check of three configurations under
  Muon, with its own rate sweeps, would have been about 12 more hours of the GPU.

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
  the token's normalised hidden state plus a learned vector for the asking module:
  `score(tanh(data(x) + asker[module]))`, where `score` is a linear map to one score per
  chunk.
- Key-per-chunk router (`both-routed-keys`): each chunk has a learned key of 64 numbers; the
  score of a chunk is the dot product of its key with a query, `query(x + asker[module])`, a
  linear map of the token's normalised hidden state plus the asking module's vector. It
  differs from the router above in having no nonlinearity and no per-chunk bias and in adding
  the module's vector before the linear map. Keys start at the scale of the other router's
  scoring weights, so both start equally close to indifferent. The pick is trained the same
  way.
- Noise-free router (`both-routed-argmax`): the router above, but training takes the
  highest-scoring chunk with no Gumbel noise; the gradient still flows through the softmax.
- Fixed map (`both-no-router-c1`, `both-no-router-c2`, `both-no-router`, `own-chunk`): no
  router; module i always computes with chunk i mod N, for N = 1, 2, 4, 8 chunks of each kind.
  Every module keeps its own layer norms, as in every other configuration. The equal-storage
  plain partner of N chunks is the plain model of N modules (`plain-l1`, `plain-l2`,
  `plain-l4`, `full`); the fixed map stores (8 - N) x 1,024 more weights than its partner,
  the layer norms of the extra modules.
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
- Hardware: one RTX 2080, 8 GB, shared with other work, so wall times are indicative: a
  run slowed or paused by other work on the GPU counts that time. Runs
  were trained one at a time: one run saturates this GPU (three plain runs at once each took
  three times as long). During
  training a routed step computes every chunk for every token (the straight-through gradient
  needs them), which is why routed runs take longer; a hard pick needs only the picked chunk
  at inference, which this code does not exploit. `own-chunk` and `both-no-router` pay a
  bookkeeping cost per chunk that `full` does not, so their times are not comparable to it.

## Sanity check: N = M, every module forced to its own chunk

`own-chunk` runs the routed code path with a bank of 8 chunks of each kind and no router:
module i always computes with chunk i. It starts from the same initial weights and sees the
same batches as `full` at the same seed, so the comparison is paired by seed. First budget:

| configuration | stored weights | seed 0 | seed 1 | seed 2 | mean | seed sd |
|---|---|---|---|---|---|---|
| full | 6,515,456 | 1.0784 | 1.0744 | 1.0772 | 1.0767 | 0.0021 |
| own-chunk | 6,515,456 | 1.0788 | 1.0743 | 1.0773 | 1.0768 | 0.0023 |
| own-chunk minus full |  | +0.0004 | -0.0001 | +0.0000 | +0.0001 |  |

Long budget, both at the full control's rate:

| configuration | stored weights | seed 0 | seed 1 | seed 2 | mean | seed sd |
|---|---|---|---|---|---|---|
| full | 6,515,456 | 0.9819 | 0.9843 | 0.9837 | 0.9833 | 0.0012 |
| own-chunk | 6,515,456 | 0.9845 | 0.9847 | 0.9832 | 0.9841 | 0.0008 |
| own-chunk minus full |  | +0.0025 | +0.0004 | -0.0005 | +0.0008 |  |

At the first budget the per-seed differences (+0.0004, -0.0001, +0.0000) are a fifth or less
of the full control's seed sd (0.0021). At the long budget they are +0.0025, +0.0004 and
-0.0005, mean +0.0008, against a seed sd of 0.0012: the two start from the same weights and
compute the same function in a different order of operations, and over 32,000 steps of
half-precision arithmetic their trajectories drift apart by about as much as two seeds do.
The routed code path with N = M reproduces the full control at both budgets.

## Why did the fixed map beat the learned router?

In the first experiment `both-no-router`, the same two banks with module i fixed to chunk
i mod 4, beat `both-routed` by 0.045. Two things separate them besides the choice itself: the
routed model trains under sampling noise (every pick is drawn with Gumbel noise), and its
router has one particular form. `both-routed-argmax` removes the noise and
`both-routed-keys` changes the form. All four store the same banks; at the same seed they
see the same batches. Long budget, each at its chosen rate:

| configuration | learning rate | stored weights | seed 0 | seed 1 | seed 2 | mean | seed sd | minus both-no-router | lowest to highest seed |
|---|---|---|---|---|---|---|---|---|---|
| both-no-router | 0.002 | 3,360,512 | 1.0122 | 1.0136 | 1.0145 | 1.0134 | 0.0012 |  |  |
| both-routed | 0.002 | 3,394,952 | 1.0316 | 1.0336 | 1.0312 | 1.0322 | 0.0013 | +0.0187 | +0.0167 to +0.0200 |
| both-routed-argmax | 0.002 | 3,394,952 | 1.0483 | 1.0446 | 1.0449 | 1.0459 | 0.0021 | +0.0325 | +0.0304 to +0.0361 |
| both-routed-keys | 0.002 | 3,398,016 | 1.0346 | 1.0304 | 1.0312 | 1.0321 | 0.0022 | +0.0186 | +0.0167 to +0.0225 |

First budget, every one at 1e-3:

| configuration | learning rate | stored weights | seed 0 | seed 1 | seed 2 | mean | seed sd | minus both-no-router | lowest to highest seed |
|---|---|---|---|---|---|---|---|---|---|
| both-no-router | 0.001 | 3,360,512 | 1.0976 | 1.0975 | 1.1061 | 1.1004 | 0.0049 |  |  |
| both-routed | 0.001 | 3,394,952 | 1.1455 | 1.1492 | 1.1409 | 1.1452 | 0.0041 | +0.0448 | +0.0348 to +0.0516 |
| both-routed-argmax | 0.001 | 3,394,952 | 1.1594 | 1.1553 | 1.1596 | 1.1581 | 0.0024 | +0.0577 | +0.0535 to +0.0618 |
| both-routed-keys | 0.001 | 3,398,016 | 1.1602 | 1.1429 | 1.1488 | 1.1506 | 0.0088 | +0.0502 | +0.0427 to +0.0625 |

Neither the sampling noise nor the router's form explains it. Removing the noise made the
model worse at both budgets: `both-routed-argmax` is 0.014 above `both-routed` at the long
budget (1.0459 against 1.0322, every seed) and 0.013 above it at the first, and its router
ends far less decided (it gives its pick a probability of 0.51 to 0.56, where the sampled
routers give 0.85 to 0.96), so the noise was helping the router commit and not costing loss.
Changing the form changed nothing: `both-routed-keys` ends at 1.0321, within 0.0001 of
`both-routed`, and at the first budget it is 0.005 above it, inside its own seed sd of 0.009.
All three routed models stay 0.019 to 0.033 above the fixed map at the long budget (0.045 to
0.058 at the first). What separates the routed models from the fixed map is therefore
something these two arms leave in place: every module choosing among all four chunks token by
token, trained through a hard pick. The longer budget and tuned rates closed three fifths of
the distance (0.045 to 0.019) without either change to the router.

## Reuse without a router, by number of chunks

The fixed map at 4 chunks is cycle sharing in the sense of Takase and Kiyono (2021, "Lessons
on Parameter Sharing across Layers in Transformers", arXiv:2104.06022): a stack of N layers
applied repeatedly. At 1 chunk it is ALBERT-style sharing (Lan et al., 2020, "ALBERT: A Lite
BERT for Self-supervised Learning of Language Representations"): one layer applied 8 times.
At 8 chunks it is the full control. Each is set beside the plain model of as many modules as
it has chunks, which stores the same chunks and applies each once, and beside the plain
curve at its stored weights.

| chunks | configuration | stored weights | plain partner | loss, 8,000 steps | partner's loss | minus partner | lowest to highest seed | gap to the plain curve | curve | loss, 32,000 steps | partner's loss | minus partner | lowest to highest seed | gap to the plain curve | curve |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | both-no-router-c1 | 994,304 | plain-l1 | 1.2054 | 1.4265 | -0.2211 | -0.2289 to -0.2147 | -0.0530 | narrower | 1.1177 | 1.3775 | -0.2599 | -0.2623 to -0.2580 | -0.0384 | narrower |
| 2 | both-no-router-c2 | 1,783,040 | plain-l2 | 1.1384 | 1.1775 | -0.0391 | -0.0437 to -0.0316 | -0.0388 | shallower | 1.0583 | 1.1184 | -0.0600 | -0.0634 to -0.0555 | -0.0147 | narrower |
| 4 | both-no-router | 3,360,512 | plain-l4 | 1.1004 | 1.1158 | -0.0153 | -0.0171 to -0.0136 | -0.0153 | shallower | 1.0134 | 1.0320 | -0.0186 | -0.0198 to -0.0164 | -0.0080 | narrower |
| 8 | own-chunk | 6,515,456 | full | 1.0768 | 1.0767 | +0.0001 | -0.0001 to +0.0004 | +0.0001 | narrower | 0.9841 | 0.9833 | +0.0008 | -0.0005 to +0.0025 | +0.0008 | narrower |

The gain from reuse grows as the chunks get fewer. Against the plain model that stores the
same chunks and applies each once, 8 modules reusing them are better by 0.260 at 1 chunk,
0.060 at 2 and 0.019 at 4 at the long budget (0.221, 0.039 and 0.015 at the first), and
level at 8, where nothing is reused; that comparison mostly measures depth, 8 applications
against N. Against the better plain model of equal storage, which may be wide and deep, the
fixed map still wins at every size below the full model, by 0.038 at 1 chunk, 0.015 at 2 and
0.008 at 4 at the long budget (0.053, 0.039 and 0.015 at the first): ALBERT-style sharing
gives the largest gain per stored weight and cycle sharing at 4 chunks the smallest, and the
longer budget widened the gain over the partner while narrowing the gain over the plain
curve, because tuned narrow plain models improved the most. The 2-chunk figure is the least
certain: its seeds range from -0.023 to -0.005 against the curve, and its rate was chosen by
a margin of 0.0004.

## Picks at validation against picks in training

Training samples each pick with Gumbel noise; the reported loss takes the highest-scoring
chunk. If the routers had not committed, the two would differ widely. Mean over seeds, long
budget:

| configuration | highest-scoring picks (reported) | picks sampled as in training |
|---|---|---|
| ff-routed | 1.0090 | 1.0102 |
| attn-routed | 1.0137 | 1.0155 |
| both-routed | 1.0322 | 1.0352 |
| both-routed-argmax | 1.0459 | 1.0459 |
| both-routed-keys | 1.0321 | 1.0345 |

## During training

Long budget: mean over seeds of validation loss on the first 2,048 validation
windows, to show whether the ordering at the end of the budget was still moving.

| configuration | step 4,000 | step 8,000 | step 12,000 | step 16,000 | step 20,000 | step 24,000 | step 28,000 | step 32,000 |
|---|---|---|---|---|---|---|---|---|
| full | 1.1941 | 1.1379 | 1.1062 | 1.0811 | 1.0549 | 1.0319 | 1.0127 | 1.0032 |
| plain-w96 | 1.3661 | 1.3057 | 1.2751 | 1.2531 | 1.2295 | 1.2089 | 1.1907 | 1.1829 |
| plain-w128 | 1.2915 | 1.2453 | 1.2173 | 1.1900 | 1.1621 | 1.1352 | 1.1117 | 1.1006 |
| plain-w160 | 1.2555 | 1.2095 | 1.1806 | 1.1531 | 1.1255 | 1.0978 | 1.0741 | 1.0635 |
| plain-w192 | 1.2213 | 1.1672 | 1.1368 | 1.1123 | 1.0862 | 1.0643 | 1.0448 | 1.0357 |
| plain-w224 | 1.2057 | 1.1508 | 1.1221 | 1.0952 | 1.0695 | 1.0471 | 1.0279 | 1.0182 |
| plain-l1 | 1.5115 | 1.4784 | 1.4573 | 1.4391 | 1.4205 | 1.4029 | 1.3876 | 1.3823 |
| plain-l2 | 1.2964 | 1.2511 | 1.2271 | 1.2069 | 1.1842 | 1.1647 | 1.1486 | 1.1412 |
| plain-l4 | 1.2340 | 1.1808 | 1.1520 | 1.1273 | 1.1036 | 1.0802 | 1.0625 | 1.0537 |
| plain-l6 | 1.2108 | 1.1545 | 1.1244 | 1.0997 | 1.0740 | 1.0502 | 1.0321 | 1.0235 |
| ff-routed | 1.2321 | 1.1707 | 1.1373 | 1.1098 | 1.0827 | 1.0592 | 1.0400 | 1.0300 |
| attn-routed | 1.2419 | 1.1793 | 1.1452 | 1.1177 | 1.0910 | 1.0661 | 1.0453 | 1.0355 |
| both-routed | 1.2727 | 1.2048 | 1.1670 | 1.1394 | 1.1101 | 1.0852 | 1.0645 | 1.0547 |
| both-routed-argmax | 1.2988 | 1.2272 | 1.1871 | 1.1573 | 1.1269 | 1.1026 | 1.0778 | 1.0686 |
| both-routed-keys | 1.2689 | 1.2004 | 1.1645 | 1.1356 | 1.1076 | 1.0841 | 1.0632 | 1.0537 |
| both-no-router-c1 | 1.3212 | 1.2677 | 1.2380 | 1.2131 | 1.1892 | 1.1671 | 1.1499 | 1.1414 |
| both-no-router-c2 | 1.2701 | 1.2052 | 1.1705 | 1.1441 | 1.1216 | 1.1025 | 1.0874 | 1.0803 |
| both-no-router | 1.2228 | 1.1680 | 1.1386 | 1.1129 | 1.0865 | 1.0637 | 1.0443 | 1.0346 |
| own-chunk | 1.1946 | 1.1377 | 1.1089 | 1.0834 | 1.0559 | 1.0339 | 1.0137 | 1.0054 |

## How much picks vary by input

Measured on the validation split with the picks inference makes. For each module, the share
of tokens whose pick is not that module's most common pick says how much the pick depends on
the input: 0 means the module always picks the same chunk, 0.75 is the most four chunks
allow. The table gives the mean of that share over modules and seeds, its lowest value in any
module of any seed, and the mean probability the router's softmax gives the chunk it picks
(0.25 is indifference, 1 is certainty), at both budgets.

| configuration | bank | off the common pick, mean, 8,000 steps | lowest module, 8,000 steps | router probability, 8,000 steps | off the common pick, mean, 32,000 steps | lowest module, 32,000 steps | router probability, 32,000 steps |
|---|---|---|---|---|---|---|---|
| ff-routed | feed-forward | 0.63 | 0.20 | 0.94 | 0.69 | 0.48 | 0.95 |
| attn-routed | attention | 0.44 | 0.28 | 0.87 | 0.40 | 0.00 | 0.82 |
| both-routed | feed-forward | 0.52 | 0.00 | 0.91 | 0.54 | 0.00 | 0.95 |
| both-routed | attention | 0.49 | 0.12 | 0.88 | 0.48 | 0.01 | 0.85 |
| both-routed-argmax | feed-forward | 0.59 | 0.34 | 0.54 | 0.63 | 0.00 | 0.56 |
| both-routed-argmax | attention | 0.52 | 0.31 | 0.60 | 0.59 | 0.35 | 0.51 |
| both-routed-keys | feed-forward | 0.54 | 0.00 | 0.91 | 0.56 | 0.00 | 0.96 |
| both-routed-keys | attention | 0.51 | 0.34 | 0.89 | 0.44 | 0.00 | 0.90 |

The tables below are the long budget in full. The first column of numbers has one entry per
module, first module to last: the share of tokens off that module's most common pick. The
next has the router's mean probability for its pick. The usage columns pool all modules;
evenness is the entropy of usage divided by its maximum (1 is even).

### Feed-forward routed

| bank | seed | share of tokens off the module's most common pick, per module | router's mean probability for its pick, per module | share of all picks, per chunk | usage evenness |
|---|---|---|---|---|---|
| feed-forward | 0 | 0.57 0.68 0.68 0.72 0.69 0.70 0.67 0.70 | 1.00 0.98 0.96 0.93 0.90 0.93 0.93 0.90 | 0.27 0.27 0.20 0.26 | 0.99 |
| feed-forward | 1 | 0.48 0.70 0.69 0.70 0.71 0.73 0.74 0.73 | 1.00 0.99 0.97 0.95 0.93 0.92 0.94 0.92 | 0.21 0.27 0.24 0.27 | 1.00 |
| feed-forward | 2 | 0.72 0.68 0.71 0.70 0.71 0.70 0.70 0.70 | 1.00 0.98 0.96 0.95 0.94 0.93 0.94 0.94 | 0.29 0.24 0.19 0.29 | 0.99 |

### Attention routed

| bank | seed | share of tokens off the module's most common pick, per module | router's mean probability for its pick, per module | share of all picks, per chunk | usage evenness |
|---|---|---|---|---|---|
| attention | 0 | 0.02 0.02 0.56 0.19 0.62 0.67 0.64 0.64 | 0.94 0.52 0.53 0.90 0.82 0.94 0.95 0.93 | 0.09 0.43 0.23 0.25 | 0.91 |
| attention | 1 | 0.01 0.27 0.26 0.43 0.28 0.63 0.62 0.63 | 0.94 0.57 0.75 0.77 0.91 0.96 0.95 0.93 | 0.08 0.53 0.33 0.06 | 0.77 |
| attention | 2 | 0.00 0.09 0.44 0.24 0.33 0.72 0.63 0.64 | 0.81 0.52 0.57 0.92 0.89 0.90 0.94 0.93 | 0.26 0.45 0.08 0.21 | 0.89 |

### Both routed

| bank | seed | share of tokens off the module's most common pick, per module | router's mean probability for its pick, per module | share of all picks, per chunk | usage evenness |
|---|---|---|---|---|---|
| feed-forward | 0 | 0.00 0.62 0.64 0.62 0.61 0.61 0.60 0.59 | 1.00 0.91 0.94 0.93 0.94 0.94 0.93 0.91 | 0.21 0.23 0.34 0.22 | 0.99 |
| feed-forward | 1 | 0.00 0.51 0.62 0.63 0.64 0.66 0.66 0.68 | 1.00 0.95 0.96 0.95 0.95 0.94 0.94 0.91 | 0.31 0.25 0.17 0.26 | 0.98 |
| feed-forward | 2 | 0.15 0.55 0.59 0.59 0.58 0.59 0.59 0.58 | 1.00 0.94 0.94 0.94 0.95 0.95 0.95 0.93 | 0.37 0.15 0.25 0.24 | 0.96 |
| attention | 0 | 0.01 0.51 0.36 0.58 0.59 0.59 0.55 0.60 | 0.90 0.60 0.88 0.89 0.91 0.92 0.91 0.88 | 0.08 0.47 0.12 0.34 | 0.85 |
| attention | 1 | 0.02 0.08 0.55 0.52 0.61 0.71 0.68 0.71 | 0.72 0.63 0.86 0.93 0.92 0.91 0.89 0.86 | 0.15 0.40 0.32 0.12 | 0.92 |
| attention | 2 | 0.16 0.11 0.47 0.57 0.64 0.64 0.67 0.68 | 0.57 0.68 0.90 0.93 0.93 0.92 0.90 0.86 | 0.46 0.14 0.31 0.09 | 0.88 |

### Both routed, no sampling noise

| bank | seed | share of tokens off the module's most common pick, per module | router's mean probability for its pick, per module | share of all picks, per chunk | usage evenness |
|---|---|---|---|---|---|
| feed-forward | 0 | 0.00 0.63 0.68 0.71 0.71 0.68 0.61 0.61 | 0.49 0.42 0.55 0.55 0.56 0.57 0.60 0.65 | 0.30 0.23 0.23 0.24 | 1.00 |
| feed-forward | 1 | 0.43 0.68 0.66 0.69 0.69 0.68 0.69 0.70 | 0.50 0.46 0.56 0.57 0.59 0.60 0.59 0.64 | 0.24 0.29 0.20 0.27 | 0.99 |
| feed-forward | 2 | 0.36 0.68 0.74 0.71 0.68 0.68 0.67 0.67 | 0.46 0.46 0.57 0.60 0.59 0.59 0.59 0.63 | 0.25 0.25 0.21 0.28 | 1.00 |
| attention | 0 | 0.59 0.60 0.65 0.54 0.58 0.63 0.65 0.70 | 0.25 0.31 0.43 0.65 0.65 0.62 0.60 0.58 | 0.38 0.20 0.20 0.22 | 0.97 |
| attention | 1 | 0.50 0.35 0.47 0.68 0.63 0.67 0.57 0.61 | 0.25 0.34 0.45 0.62 0.64 0.63 0.62 0.58 | 0.19 0.33 0.25 0.24 | 0.99 |
| attention | 2 | 0.55 0.52 0.50 0.62 0.70 0.70 0.56 0.60 | 0.25 0.32 0.47 0.57 0.62 0.62 0.67 0.58 | 0.20 0.30 0.22 0.28 | 0.99 |

### Both routed, key per chunk

| bank | seed | share of tokens off the module's most common pick, per module | router's mean probability for its pick, per module | share of all picks, per chunk | usage evenness |
|---|---|---|---|---|---|
| feed-forward | 0 | 0.00 0.51 0.58 0.60 0.60 0.59 0.58 0.58 | 1.00 0.96 0.96 0.96 0.96 0.95 0.95 0.94 | 0.22 0.19 0.22 0.37 | 0.97 |
| feed-forward | 1 | 0.00 0.62 0.69 0.71 0.69 0.66 0.65 0.65 | 0.87 0.97 0.96 0.96 0.95 0.96 0.96 0.94 | 0.33 0.15 0.26 0.26 | 0.97 |
| feed-forward | 2 | 0.00 0.59 0.68 0.67 0.68 0.68 0.69 0.70 | 1.00 0.98 0.97 0.96 0.96 0.96 0.95 0.93 | 0.41 0.15 0.25 0.19 | 0.94 |
| attention | 0 | 0.01 0.00 0.18 0.51 0.56 0.59 0.67 0.66 | 0.99 1.00 0.96 0.95 0.93 0.92 0.91 0.87 | 0.09 0.52 0.13 0.26 | 0.85 |
| attention | 1 | 0.38 0.01 0.52 0.60 0.58 0.60 0.67 0.68 | 0.53 0.64 0.81 0.95 0.95 0.94 0.91 0.87 | 0.11 0.34 0.46 0.09 | 0.85 |
| attention | 2 | 0.00 0.00 0.21 0.61 0.57 0.60 0.62 0.58 | 0.99 1.00 0.95 0.97 0.96 0.94 0.91 0.86 | 0.58 0.06 0.23 0.13 | 0.78 |

## Per configuration

Long budget:

| configuration | learning rate | stored weights | mean loss | seed sd | lowest | highest |
|---|---|---|---|---|---|---|
| full | 0.002 | 6,515,456 | 0.9833 | 0.0012 | 0.9819 | 0.9843 |
| plain-w96 | 0.002 | 968,896 | 1.1599 | 0.0010 | 1.1587 | 1.1606 |
| plain-w128 | 0.004 | 1,684,992 | 1.0779 | 0.0060 | 1.0719 | 1.0838 |
| plain-w160 | 0.004 | 2,597,696 | 1.0413 | 0.0017 | 1.0399 | 1.0432 |
| plain-w192 | 0.002 | 3,707,008 | 1.0138 | 0.0013 | 1.0125 | 1.0152 |
| plain-w224 | 0.002 | 5,012,928 | 0.9980 | 0.0038 | 0.9946 | 1.0021 |
| plain-l1 | 0.002 | 987,136 | 1.3775 | 0.0013 | 1.3764 | 1.3790 |
| plain-l2 | 0.002 | 1,776,896 | 1.1184 | 0.0004 | 1.1180 | 1.1188 |
| plain-l4 | 0.002 | 3,356,416 | 1.0320 | 0.0013 | 1.0309 | 1.0334 |
| plain-l6 | 0.002 | 4,935,936 | 1.0019 | 0.0022 | 1.0002 | 1.0043 |
| ff-routed | 0.002 | 4,430,404 | 1.0090 | 0.0014 | 1.0074 | 1.0099 |
| attn-routed | 0.002 | 5,480,004 | 1.0137 | 0.0032 | 1.0117 | 1.0174 |
| both-routed | 0.002 | 3,394,952 | 1.0322 | 0.0013 | 1.0312 | 1.0336 |
| both-routed-argmax | 0.002 | 3,394,952 | 1.0459 | 0.0021 | 1.0446 | 1.0483 |
| both-routed-keys | 0.002 | 3,398,016 | 1.0321 | 0.0022 | 1.0304 | 1.0346 |
| both-no-router-c1 | 0.002 | 994,304 | 1.1177 | 0.0008 | 1.1168 | 1.1184 |
| both-no-router-c2 | 0.001 | 1,783,040 | 1.0583 | 0.0039 | 1.0548 | 1.0625 |
| both-no-router | 0.002 | 3,360,512 | 1.0134 | 0.0012 | 1.0122 | 1.0145 |
| own-chunk | 0.002 | 6,515,456 | 0.9841 | 0.0008 | 0.9832 | 0.9847 |

First budget:

| configuration | learning rate | stored weights | mean loss | seed sd | lowest | highest |
|---|---|---|---|---|---|---|
| full | 0.001 | 6,515,456 | 1.0767 | 0.0021 | 1.0744 | 1.0784 |
| plain-w96 | 0.001 | 968,896 | 1.2616 | 0.0040 | 1.2569 | 1.2640 |
| plain-w128 | 0.001 | 1,684,992 | 1.1939 | 0.0067 | 1.1867 | 1.2000 |
| plain-w160 | 0.001 | 2,597,696 | 1.1544 | 0.0043 | 1.1503 | 1.1589 |
| plain-w192 | 0.001 | 3,707,008 | 1.1177 | 0.0027 | 1.1146 | 1.1194 |
| plain-w224 | 0.001 | 5,012,928 | 1.0974 | 0.0056 | 1.0915 | 1.1027 |
| plain-l1 | 0.001 | 987,136 | 1.4265 | 0.0017 | 1.4254 | 1.4285 |
| plain-l2 | 0.001 | 1,776,896 | 1.1775 | 0.0025 | 1.1750 | 1.1799 |
| plain-l4 | 0.001 | 3,356,416 | 1.1158 | 0.0052 | 1.1113 | 1.1214 |
| plain-l6 | 0.001 | 4,935,936 | 1.0952 | 0.0080 | 1.0864 | 1.1021 |
| ff-routed | 0.001 | 4,430,404 | 1.0977 | 0.0047 | 1.0923 | 1.1011 |
| attn-routed | 0.001 | 5,480,004 | 1.1146 | 0.0024 | 1.1119 | 1.1166 |
| both-routed | 0.001 | 3,394,952 | 1.1452 | 0.0041 | 1.1409 | 1.1492 |
| both-routed-argmax | 0.001 | 3,394,952 | 1.1581 | 0.0024 | 1.1553 | 1.1596 |
| both-routed-keys | 0.001 | 3,398,016 | 1.1506 | 0.0088 | 1.1429 | 1.1602 |
| both-no-router-c1 | 0.001 | 994,304 | 1.2054 | 0.0056 | 1.1996 | 1.2107 |
| both-no-router-c2 | 0.001 | 1,783,040 | 1.1384 | 0.0073 | 1.1313 | 1.1459 |
| both-no-router | 0.001 | 3,360,512 | 1.1004 | 0.0049 | 1.0975 | 1.1061 |
| own-chunk | 0.001 | 6,515,456 | 1.0768 | 0.0023 | 1.0743 | 1.0788 |

## Every run

Both experiments and the learning-rate sweep, from the run files.

| configuration | seed | steps | learning rate | stored weights | final validation loss | wall seconds (training) |
|---|---|---|---|---|---|---|
| full | 0 | 8,000 | 0.001 | 6,515,456 | 1.0784 | 492 |
| plain-w96 | 0 | 8,000 | 0.001 | 968,896 | 1.2638 | 182 |
| plain-w128 | 0 | 8,000 | 0.001 | 1,684,992 | 1.1949 | 261 |
| plain-w160 | 0 | 8,000 | 0.001 | 2,597,696 | 1.1503 | 342 |
| plain-w192 | 0 | 8,000 | 0.001 | 3,707,008 | 1.1194 | 443 |
| plain-w224 | 0 | 8,000 | 0.001 | 5,012,928 | 1.0915 | 479 |
| plain-l1 | 0 | 8,000 | 0.001 | 987,136 | 1.4254 | 78 |
| plain-l2 | 0 | 8,000 | 0.001 | 1,776,896 | 1.1750 | 117 |
| plain-l4 | 0 | 8,000 | 0.001 | 3,356,416 | 1.1113 | 260 |
| plain-l6 | 0 | 8,000 | 0.001 | 4,935,936 | 1.1021 | 565 |
| ff-routed | 0 | 8,000 | 0.001 | 4,430,404 | 1.1011 | 2010 |
| attn-routed | 0 | 8,000 | 0.001 | 5,480,004 | 1.1166 | 1050 |
| both-routed | 0 | 8,000 | 0.001 | 3,394,952 | 1.1455 | 1602 |
| both-routed-argmax | 0 | 8,000 | 0.001 | 3,394,952 | 1.1594 | 1440 |
| both-routed-keys | 0 | 8,000 | 0.001 | 3,398,016 | 1.1602 | 1438 |
| both-no-router-c1 | 0 | 8,000 | 0.001 | 994,304 | 1.2107 | 416 |
| both-no-router-c2 | 0 | 8,000 | 0.001 | 1,783,040 | 1.1313 | 430 |
| both-no-router | 0 | 8,000 | 0.001 | 3,360,512 | 1.0976 | 494 |
| own-chunk | 0 | 8,000 | 0.001 | 6,515,456 | 1.0788 | 533 |
| full | 1 | 8,000 | 0.001 | 6,515,456 | 1.0744 | 409 |
| plain-w96 | 1 | 8,000 | 0.001 | 968,896 | 1.2569 | 181 |
| plain-w128 | 1 | 8,000 | 0.001 | 1,684,992 | 1.1867 | 207 |
| plain-w160 | 1 | 8,000 | 0.001 | 2,597,696 | 1.1540 | 282 |
| plain-w192 | 1 | 8,000 | 0.001 | 3,707,008 | 1.1191 | 321 |
| plain-w224 | 1 | 8,000 | 0.001 | 5,012,928 | 1.0979 | 373 |
| plain-l1 | 1 | 8,000 | 0.001 | 987,136 | 1.4285 | 72 |
| plain-l2 | 1 | 8,000 | 0.001 | 1,776,896 | 1.1799 | 114 |
| plain-l4 | 1 | 8,000 | 0.001 | 3,356,416 | 1.1146 | 218 |
| plain-l6 | 1 | 8,000 | 0.001 | 4,935,936 | 1.0864 | 304 |
| ff-routed | 1 | 8,000 | 0.001 | 4,430,404 | 1.0998 | 964 |
| attn-routed | 1 | 8,000 | 0.001 | 5,480,004 | 1.1119 | 1011 |
| both-routed | 1 | 8,000 | 0.001 | 3,394,952 | 1.1492 | 1586 |
| both-routed-argmax | 1 | 8,000 | 0.001 | 3,394,952 | 1.1553 | 1446 |
| both-routed-keys | 1 | 8,000 | 0.001 | 3,398,016 | 1.1429 | 1439 |
| both-no-router-c1 | 1 | 8,000 | 0.001 | 994,304 | 1.1996 | 407 |
| both-no-router-c2 | 1 | 8,000 | 0.001 | 1,783,040 | 1.1379 | 418 |
| both-no-router | 1 | 8,000 | 0.001 | 3,360,512 | 1.0975 | 501 |
| own-chunk | 1 | 8,000 | 0.001 | 6,515,456 | 1.0743 | 550 |
| full | 2 | 8,000 | 0.001 | 6,515,456 | 1.0772 | 426 |
| plain-w96 | 2 | 8,000 | 0.001 | 968,896 | 1.2640 | 179 |
| plain-w128 | 2 | 8,000 | 0.001 | 1,684,992 | 1.2000 | 243 |
| plain-w160 | 2 | 8,000 | 0.001 | 2,597,696 | 1.1589 | 310 |
| plain-w192 | 2 | 8,000 | 0.001 | 3,707,008 | 1.1146 | 348 |
| plain-w224 | 2 | 8,000 | 0.001 | 5,012,928 | 1.1027 | 373 |
| plain-l1 | 2 | 8,000 | 0.001 | 987,136 | 1.4257 | 63 |
| plain-l2 | 2 | 8,000 | 0.001 | 1,776,896 | 1.1775 | 108 |
| plain-l4 | 2 | 8,000 | 0.001 | 3,356,416 | 1.1214 | 208 |
| plain-l6 | 2 | 8,000 | 0.001 | 4,935,936 | 1.0971 | 304 |
| ff-routed | 2 | 8,000 | 0.001 | 4,430,404 | 1.0923 | 923 |
| attn-routed | 2 | 8,000 | 0.001 | 5,480,004 | 1.1152 | 919 |
| both-routed | 2 | 8,000 | 0.001 | 3,394,952 | 1.1409 | 1454 |
| both-routed-argmax | 2 | 8,000 | 0.001 | 3,394,952 | 1.1596 | 1442 |
| both-routed-keys | 2 | 8,000 | 0.001 | 3,398,016 | 1.1488 | 1448 |
| both-no-router-c1 | 2 | 8,000 | 0.001 | 994,304 | 1.2059 | 410 |
| both-no-router-c2 | 2 | 8,000 | 0.001 | 1,783,040 | 1.1459 | 421 |
| both-no-router | 2 | 8,000 | 0.001 | 3,360,512 | 1.1061 | 504 |
| own-chunk | 2 | 8,000 | 0.001 | 6,515,456 | 1.0773 | 489 |
| full | 0 | 8,000 | 0.0005 | 6,515,456 | 1.1072 | 380 |
| full | 0 | 8,000 | 0.002 | 6,515,456 | 1.0538 | 397 |
| plain-w96 | 0 | 8,000 | 0.0005 | 968,896 | 1.3421 | 185 |
| plain-w96 | 0 | 8,000 | 0.002 | 968,896 | 1.2306 | 201 |
| plain-w128 | 0 | 8,000 | 0.0005 | 1,684,992 | 1.2433 | 198 |
| plain-w128 | 0 | 8,000 | 0.002 | 1,684,992 | 1.1674 | 198 |
| plain-w160 | 0 | 8,000 | 0.0005 | 2,597,696 | 1.1878 | 272 |
| plain-w160 | 0 | 8,000 | 0.002 | 2,597,696 | 1.1185 | 272 |
| plain-w192 | 0 | 8,000 | 0.0005 | 3,707,008 | 1.1527 | 303 |
| plain-w192 | 0 | 8,000 | 0.002 | 3,707,008 | 1.0837 | 304 |
| plain-w224 | 0 | 8,000 | 0.0005 | 5,012,928 | 1.1208 | 353 |
| plain-w224 | 0 | 8,000 | 0.002 | 5,012,928 | 1.0648 | 353 |
| plain-l1 | 0 | 8,000 | 0.0005 | 987,136 | 1.4541 | 63 |
| plain-l1 | 0 | 8,000 | 0.002 | 987,136 | 1.4204 | 293 |
| plain-l2 | 0 | 8,000 | 0.0005 | 1,776,896 | 1.2171 | 648 |
| plain-l2 | 0 | 8,000 | 0.002 | 1,776,896 | 1.1659 | 121 |
| plain-l4 | 0 | 8,000 | 0.0005 | 3,356,416 | 1.1657 | 199 |
| plain-l4 | 0 | 8,000 | 0.002 | 3,356,416 | 1.0984 | 199 |
| plain-l6 | 0 | 8,000 | 0.0005 | 4,935,936 | 1.1317 | 291 |
| plain-l6 | 0 | 8,000 | 0.002 | 4,935,936 | 1.0743 | 291 |
| ff-routed | 0 | 8,000 | 0.0005 | 4,430,404 | 1.1460 | 962 |
| ff-routed | 0 | 8,000 | 0.002 | 4,430,404 | 1.0884 | 916 |
| attn-routed | 0 | 8,000 | 0.0005 | 5,480,004 | 1.1429 | 910 |
| attn-routed | 0 | 8,000 | 0.002 | 5,480,004 | 1.0961 | 912 |
| both-routed | 0 | 8,000 | 0.0005 | 3,394,952 | 1.1906 | 1450 |
| both-routed | 0 | 8,000 | 0.002 | 3,394,952 | 1.1208 | 1468 |
| both-routed-argmax | 0 | 8,000 | 0.0005 | 3,394,952 | 1.2087 | 1622 |
| both-routed-argmax | 0 | 8,000 | 0.002 | 3,394,952 | 1.1360 | 1526 |
| both-routed-keys | 0 | 8,000 | 0.0005 | 3,398,016 | 1.1991 | 1545 |
| both-routed-keys | 0 | 8,000 | 0.002 | 3,398,016 | 1.1253 | 1492 |
| both-no-router-c1 | 0 | 8,000 | 0.0005 | 994,304 | 1.2404 | 469 |
| both-no-router-c1 | 0 | 8,000 | 0.002 | 994,304 | 1.1797 | 457 |
| both-no-router-c2 | 0 | 8,000 | 0.0005 | 1,783,040 | 1.1882 | 429 |
| both-no-router-c2 | 0 | 8,000 | 0.002 | 1,783,040 | 1.1195 | 436 |
| both-no-router | 0 | 8,000 | 0.0005 | 3,360,512 | 1.1466 | 446 |
| both-no-router | 0 | 8,000 | 0.002 | 3,360,512 | 1.0802 | 662 |
| full | 0 | 32,000 | 0.001 | 6,515,456 | 1.0003 | 2008 |
| full | 0 | 32,000 | 0.002 | 6,515,456 | 0.9819 | 2089 |
| full | 0 | 32,000 | 0.004 | 6,515,456 | 0.9840 | 1865 |
| plain-w96 | 0 | 32,000 | 0.001 | 968,896 | 1.1649 | 919 |
| plain-w96 | 0 | 32,000 | 0.002 | 968,896 | 1.1602 | 767 |
| plain-w128 | 0 | 32,000 | 0.001 | 1,684,992 | 1.1100 | 1124 |
| plain-w128 | 0 | 32,000 | 0.002 | 1,684,992 | 1.1017 | 1056 |
| plain-w128 | 0 | 32,000 | 0.004 | 1,684,992 | 1.0838 | 837 |
| plain-w128 | 0 | 32,000 | 0.008 | 1,684,992 | 1.0862 | 1090 |
| plain-w160 | 0 | 32,000 | 0.001 | 2,597,696 | 1.0679 | 1528 |
| plain-w160 | 0 | 32,000 | 0.002 | 2,597,696 | 1.0484 | 1298 |
| plain-w160 | 0 | 32,000 | 0.004 | 2,597,696 | 1.0399 | 1315 |
| plain-w160 | 0 | 32,000 | 0.008 | 2,597,696 | 1.0499 | 1303 |
| plain-w192 | 0 | 32,000 | 0.001 | 3,707,008 | 1.0406 | 1282 |
| plain-w192 | 0 | 32,000 | 0.002 | 3,707,008 | 1.0125 | 1431 |
| plain-w192 | 0 | 32,000 | 0.004 | 3,707,008 | 1.0145 | 1326 |
| plain-w224 | 0 | 32,000 | 0.001 | 5,012,928 | 1.0124 | 1440 |
| plain-w224 | 0 | 32,000 | 0.002 | 5,012,928 | 0.9946 | 1415 |
| plain-w224 | 0 | 32,000 | 0.004 | 5,012,928 | 0.9991 | 1414 |
| plain-l1 | 0 | 32,000 | 0.001 | 987,136 | 1.3788 | 254 |
| plain-l1 | 0 | 32,000 | 0.002 | 987,136 | 1.3764 | 252 |
| plain-l2 | 0 | 32,000 | 0.001 | 1,776,896 | 1.1195 | 439 |
| plain-l2 | 0 | 32,000 | 0.002 | 1,776,896 | 1.1182 | 538 |
| plain-l4 | 0 | 32,000 | 0.001 | 3,356,416 | 1.0388 | 901 |
| plain-l4 | 0 | 32,000 | 0.002 | 3,356,416 | 1.0317 | 822 |
| plain-l4 | 0 | 32,000 | 0.004 | 3,356,416 | 1.0372 | 867 |
| plain-l6 | 0 | 32,000 | 0.001 | 4,935,936 | 1.0215 | 1150 |
| plain-l6 | 0 | 32,000 | 0.002 | 4,935,936 | 1.0043 | 1146 |
| plain-l6 | 0 | 32,000 | 0.004 | 4,935,936 | 1.0072 | 1151 |
| ff-routed | 0 | 32,000 | 0.001 | 4,430,404 | 1.0123 | 3709 |
| ff-routed | 0 | 32,000 | 0.002 | 4,430,404 | 1.0098 | 8366 |
| attn-routed | 0 | 32,000 | 0.001 | 5,480,004 | 1.0163 | 5024 |
| attn-routed | 0 | 32,000 | 0.002 | 5,480,004 | 1.0117 | 3851 |
| both-routed | 0 | 32,000 | 0.001 | 3,394,952 | 1.0394 | 7041 |
| both-routed | 0 | 32,000 | 0.002 | 3,394,952 | 1.0316 | 5788 |
| both-routed | 0 | 32,000 | 0.004 | 3,394,952 | 1.0366 | 6741 |
| both-routed-argmax | 0 | 32,000 | 0.001 | 3,394,952 | 1.0593 | 5772 |
| both-routed-argmax | 0 | 32,000 | 0.002 | 3,394,952 | 1.0483 | 5771 |
| both-routed-argmax | 0 | 32,000 | 0.004 | 3,394,952 | 1.2190 | 5767 |
| both-routed-keys | 0 | 32,000 | 0.001 | 3,398,016 | 1.0477 | 5745 |
| both-routed-keys | 0 | 32,000 | 0.002 | 3,398,016 | 1.0346 | 5717 |
| both-routed-keys | 0 | 32,000 | 0.004 | 3,398,016 | 1.0362 | 5720 |
| both-no-router-c1 | 0 | 32,000 | 0.001 | 994,304 | 1.1267 | 1604 |
| both-no-router-c1 | 0 | 32,000 | 0.002 | 994,304 | 1.1184 | 1725 |
| both-no-router-c1 | 0 | 32,000 | 0.004 | 994,304 | 1.1207 | 1754 |
| both-no-router-c2 | 0 | 32,000 | 0.0005 | 1,783,040 | 1.0752 | 1765 |
| both-no-router-c2 | 0 | 32,000 | 0.001 | 1,783,040 | 1.0548 | 1792 |
| both-no-router-c2 | 0 | 32,000 | 0.002 | 1,783,040 | 1.0553 | 1778 |
| both-no-router | 0 | 32,000 | 0.001 | 3,360,512 | 1.0167 | 1868 |
| both-no-router | 0 | 32,000 | 0.002 | 3,360,512 | 1.0122 | 1852 |
| own-chunk | 0 | 32,000 | 0.002 | 6,515,456 | 0.9845 | 2059 |
| full | 1 | 32,000 | 0.002 | 6,515,456 | 0.9843 | 1569 |
| plain-w96 | 1 | 32,000 | 0.002 | 968,896 | 1.1587 | 771 |
| plain-w128 | 1 | 32,000 | 0.004 | 1,684,992 | 1.0780 | 854 |
| plain-w160 | 1 | 32,000 | 0.004 | 2,597,696 | 1.0432 | 1145 |
| plain-w192 | 1 | 32,000 | 0.002 | 3,707,008 | 1.0152 | 1223 |
| plain-w224 | 1 | 32,000 | 0.002 | 5,012,928 | 1.0021 | 1458 |
| plain-l1 | 1 | 32,000 | 0.002 | 987,136 | 1.3771 | 302 |
| plain-l2 | 1 | 32,000 | 0.002 | 1,776,896 | 1.1188 | 454 |
| plain-l4 | 1 | 32,000 | 0.002 | 3,356,416 | 1.0334 | 801 |
| plain-l6 | 1 | 32,000 | 0.002 | 4,935,936 | 1.0002 | 1161 |
| ff-routed | 1 | 32,000 | 0.002 | 4,430,404 | 1.0099 | 3872 |
| attn-routed | 1 | 32,000 | 0.002 | 5,480,004 | 1.0119 | 5449 |
| both-routed | 1 | 32,000 | 0.002 | 3,394,952 | 1.0336 | 5865 |
| both-routed-argmax | 1 | 32,000 | 0.002 | 3,394,952 | 1.0446 | 5803 |
| both-routed-keys | 1 | 32,000 | 0.002 | 3,398,016 | 1.0304 | 10529 |
| both-no-router-c1 | 1 | 32,000 | 0.002 | 994,304 | 1.1178 | 1622 |
| both-no-router-c2 | 1 | 32,000 | 0.001 | 1,783,040 | 1.0577 | 1731 |
| both-no-router | 1 | 32,000 | 0.002 | 3,360,512 | 1.0136 | 1743 |
| own-chunk | 1 | 32,000 | 0.002 | 6,515,456 | 0.9847 | 2103 |
| full | 2 | 32,000 | 0.002 | 6,515,456 | 0.9837 | 1509 |
| plain-w96 | 2 | 32,000 | 0.002 | 968,896 | 1.1606 | 688 |
| plain-w128 | 2 | 32,000 | 0.004 | 1,684,992 | 1.0719 | 815 |
| plain-w160 | 2 | 32,000 | 0.004 | 2,597,696 | 1.0409 | 1729 |
| plain-w192 | 2 | 32,000 | 0.002 | 3,707,008 | 1.0138 | 1679 |
| plain-w224 | 2 | 32,000 | 0.002 | 5,012,928 | 0.9973 | 1431 |
| plain-l1 | 2 | 32,000 | 0.002 | 987,136 | 1.3790 | 252 |
| plain-l2 | 2 | 32,000 | 0.002 | 1,776,896 | 1.1180 | 428 |
| plain-l4 | 2 | 32,000 | 0.002 | 3,356,416 | 1.0309 | 784 |
| plain-l6 | 2 | 32,000 | 0.002 | 4,935,936 | 1.0011 | 1251 |
| ff-routed | 2 | 32,000 | 0.002 | 4,430,404 | 1.0074 | 3728 |
| attn-routed | 2 | 32,000 | 0.002 | 5,480,004 | 1.0174 | 3644 |
| both-routed | 2 | 32,000 | 0.002 | 3,394,952 | 1.0312 | 5859 |
| both-routed-argmax | 2 | 32,000 | 0.002 | 3,394,952 | 1.0449 | 5827 |
| both-routed-keys | 2 | 32,000 | 0.002 | 3,398,016 | 1.0312 | 5773 |
| both-no-router-c1 | 2 | 32,000 | 0.002 | 994,304 | 1.1168 | 1693 |
| both-no-router-c2 | 2 | 32,000 | 0.001 | 1,783,040 | 1.0625 | 2168 |
| both-no-router | 2 | 32,000 | 0.002 | 3,360,512 | 1.0145 | 2197 |
| own-chunk | 2 | 32,000 | 0.002 | 6,515,456 | 0.9832 | 2487 |

## Reproducing

`nix-shell` gives the environment. One command reproduces any run in the table above:

```sh
python -m weightbank.train <configuration> --seed <seed> --steps <steps> --rate <learning rate>
```

It writes `runs/<steps>/<configuration>-lr<learning rate>-s<seed>.json`.
`python -m weightbank.sweep` trains every run of both experiments that `runs/` lacks (the
first experiment, the short sweep, the long sweep, then the remaining seeds at each
configuration's rate), and `python -m weightbank.report` rebuilds this file and the
chart.
