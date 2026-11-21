# Fitting a shared weight bank to a trained model

The aim of this project is an improvement other people can adopt in their own models. The
[first report](report.md) trains small models from scratch with a shared bank. This one is a
second, cheaper track beside it: take a published, already-trained language model, replace its
30 feed-forward steps with a bank of fewer shared ones, and ask what that costs in quality per
stored byte against the things people already do to shrink a model. An arm below is one
compressed model; healing is a brief fine-tune of the compressed weights to match the
original model's outputs.

Every table below is written by `python -m weightbank.compress` from the files in
`runs/compress/`. Quality is perplexity on the WikiText-2 test split; lower is better.

## Reading

1. Sharing did not pay for itself on this model. Rounding the original's weights to 5 bits
   cuts it from 269 to 93 MB at perplexity 23.1 (from 20.8), while the best bank arm reaches
   only 240 MB at 23.8; every bank arm at 16 or 8 bits is beaten by plain rounding at fewer
   bytes.
2. The bank does beat low rank on every layer, with 15 to 24% lower perplexity at equal
   bytes, but nearly all of that comes from leaving some layers whole: against a control that
   keeps the same layers whole and shares nothing, sharing is worth 0 to 2%. A per-input
   router over the bank, tried at one size, added nothing.
3. This model's layers have almost nothing in common to share: cosine 0.0007 as published,
   and 0.089 with hidden units matched against 0.079 for scrambled layers. This is a negative
   result for retrofitting a bank onto weights trained apart, and says nothing against a
   bank a model is trained with from the start.

![Stored bytes against perplexity for every arm](compress-chart.png)

Left: every arm before healing (hollow) and after (filled), at 16-bit weights, labelled by
bank size/rank; the weights-only fit alone leaves perplexity between 140 and 1.3 x 10^8, and healing
recovers most of it. Right: the healed arms stored at 16, 8 and 4 bits, with the original model rounded to
8, 6, 5, 4 and 3 bits as the black line. Arms that land on the same point are drawn as rings:
the shared bank is the small blue mark, and it sits inside the green same-layers control at
most sizes and inside the yellow routed bank at 8/64.

## What a negative result here does not show

This experiment asks whether weights that were trained to be separate can be folded together
afterwards. The layers of this model were never asked to share anything, and the measurement
below finds that they do not. A model trained from scratch with a shared bank can arrange its
layers so that sharing fits, which is what the fixed map in the [first report](report.md)
does, at a lower loss than a plain model of the same stored size. So a loss here is a
statement about retrofitting one trained model. It is not a verdict on shared weight banks.

## Setup

**Model.** [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M): 30 layers, width
576, 134.5 million weights, published as 16-bit floats (269 MB). It was chosen because its
Apache-2.0 licence allows redistributing derived weights, it downloads without a sign-in as a
safetensors file, it has 30 same-shaped layers to share among, and it is small enough to
fine-tune on one 8 GB GPU. Its feed-forward steps hold 59% of its weights, attention 20%, the
token embedding 21%.

**Bank.** One feed-forward step is three matrices (gate, up, down; 2.65 million weights). A
bank of N < 30 steps replaces the 30; each layer is assigned one bank step and, if it shares
that step with another layer, keeps a rank-r correction added to each of the three matrices.
A layer alone on its bank step is stored whole and gets no correction. Attention, norms and
the embedding are untouched.

**Fit, weights only.** The hidden units of a feed-forward step can be reordered without
changing what it computes, so two layers are compared, and later averaged, with their units
matched: the one-to-one pairing that maximises the inner product of their weights. Layers are
grouped by average-linkage clustering of their pairwise similarity after matching, then each
group alternates between its bank step (the mean of its members less
their corrections) and the corrections (the best rank-r matrix for what the bank step leaves),
minimising squared weight error.

**Healing.** The bank steps and corrections are then trained for 2,000 steps of 4 x 512
tokens to match the original model's next-token distribution (KL divergence) on WikiText-103
training text; everything else stays frozen. The learning rate is chosen per kind of arm by
500-step runs on one bank size, scored on the WikiText-2 validation split.

**Controls at equal stored bytes.** *Low rank, every layer*: no sharing, every feed-forward
matrix replaced by its best low-rank matrix at the one rank that stores what the bank arm
stores. *Low rank, same layers kept whole*: the layers the bank stores whole stay whole, and
every layer that shares a bank step is instead low-rank at the rank that spends that step's
bytes on its members separately. The second control differs from the bank only in the
sharing. Both are healed exactly as the bank is.

**Quantization.** Round to nearest on a uniform grid over every run of 64 consecutive weights,
with the run's lowest and highest weight kept as 16-bit floats: 8.5 and 4.5 bits per weight.
It is applied to every matrix of the model, the embedding included, with no calibration data,
and to the healed arms after healing with no further training.

The original is also rounded to 6, 5 and 3 bits, to fill in the line the arms are judged
against and to show where rounding alone stops working.

**Stored bytes** count every weight of the whole model once, at 16 bits or at the quantized
width. The table of which layer uses which bank step (30 bytes) is not counted.

| weights stored as | stored MB | perplexity |
|---|---|---|
| 16-bit floats, as published | 269.0 | 20.81 |
| 8-bit | 143.0 | 20.84 |
| 6-bit | 109.3 | 21.39 |
| 5-bit | 92.5 | 23.08 |
| 4-bit | 75.7 | 31.83 |
| 3-bit | 58.9 | 421.63 |

## How similar the layers are before any fit

![Layer similarity and shared energy](compress-similarity.png)

The three squares compare every pair of the 30 layers (a layer with itself is left blank).

| weights compared | mean cosine | lowest | highest |
|---|---|---|---|
| feed-forward step, hidden units as published | 0.0007 | -0.0017 | 0.0059 |
| feed-forward step, hidden units matched | 0.0891 | 0.0745 | 0.1331 |
| the same matching on scrambled layers | 0.0792 | 0.0752 | 0.0836 |
| attention key matrix, as published | -0.0000 | -0.0175 | 0.0206 |
| attention out matrix, as published | 0.0000 | -0.0050 | 0.0059 |
| attention query matrix, as published | 0.0001 | -0.0139 | 0.0152 |
| attention value matrix, as published | 0.0000 | -0.0077 | 0.0093 |

Before any fit, the layers of this model have nothing to share. With hidden units in the
order published, the cosine between two layers' feed-forward weights averages 0.0007 and never
exceeds 0.006; the four attention matrices are no closer (at most 0.021). The order of a layer's hidden units is
arbitrary, so the fair comparison first matches each layer's units to the other's (the
assignment that maximises their inner product). That lifts the mean to 0.089, but the same
matching applied to scrambled layers (every column's values shuffled among the hidden units,
so each layer keeps its values and loses any relation to the others) already gives 0.079: most of the similarity is made by the matching. What is left is a
band of neighbouring layers in the middle of the model, at most 0.133 between two layers. The
right panel says the same thing without choosing an order: the best k shared matrices can
hold barely more of the 30 layers' weights than k/30, the line for layers of equal size with
nothing in common. Sharing can therefore remove almost no weight error here that dropping
weights would not, and whatever the bank arms recover comes from healing.

## Every arm

Arms are named by bank size and correction rank; a control carries the name of the bank arm
whose bytes it matches. Weight error is squared error over the feed-forward weights as a share
of their squared size, before healing.

| arm, bank size/rank | feed-forward weights | weight error | MB at 16 bits | MB at 8 | MB at 4 | perplexity before healing | healed | healed, 8-bit | healed, 4-bit |
|---|---|---|---|---|---|---|---|---|---|
| shared bank, 8/16 | 8 matrices, rank 16 | 0.621 | 157.7 | 83.8 | 44.4 | 1.28e+08 | 41.64 | 41.72 | 57.68 |
| low rank, every layer, 8/16 | rank 126 | 0.508 | 157.7 | 83.8 | 44.4 | 449,922 | 55.01 | 55.08 | 74.13 |
| low rank, same layers kept whole, 8/16 | rank 62 to 120 | 0.547 | 157.5 | 83.7 | 44.3 | 194,429 | 42.55 | 42.55 | 64.09 |
| shared bank, 8/64 | 8 matrices, rank 64 | 0.468 | 174.1 | 92.5 | 49.0 | 1.10e+08 | 36.75 | 36.79 | 51.10 |
| low rank, every layer, 8/64 | rank 169 | 0.405 | 174.0 | 92.5 | 49.0 | 328,203 | 43.49 | 43.57 | 57.36 |
| low rank, same layers kept whole, 8/64 | rank 110 to 168 | 0.426 | 173.9 | 92.4 | 49.0 | 254,118 | 37.15 | 37.19 | 52.25 |
| shared bank, 15/16 | 15 matrices, rank 16 | 0.406 | 193.7 | 102.9 | 54.5 | 77,237 | 31.67 | 31.70 | 43.65 |
| low rank, every layer, 15/16 | rank 220 | 0.304 | 193.4 | 102.8 | 54.4 | 1.63e+06 | 37.48 | 37.54 | 47.42 |
| low rank, same layers kept whole, 15/16 | rank 62 to 225 | 0.353 | 193.5 | 102.8 | 54.5 | 111,369 | 31.93 | 31.92 | 45.52 |
| shared bank, 15/64 | 15 matrices, rank 64 | 0.294 | 206.4 | 109.7 | 58.1 | 13,600 | 29.49 | 29.50 | 38.64 |
| low rank, every layer, 15/64 | rank 254 | 0.247 | 206.3 | 109.7 | 58.1 | 913,541 | 35.05 | 35.08 | 43.59 |
| low rank, same layers kept whole, 15/64 | rank 110 to 273 | 0.271 | 206.3 | 109.6 | 58.1 | 105,552 | 30.01 | 30.03 | 40.43 |
| shared bank, 23/16 | 23 matrices, rank 16 | 0.185 | 233.9 | 124.3 | 65.8 | 154.93 | 24.27 | 24.29 | 33.60 |
| low rank, every layer, 23/16 | rank 326 | 0.151 | 233.7 | 124.2 | 65.8 | 832,722 | 30.51 | 30.54 | 37.80 |
| low rank, same layers kept whole, 23/16 | rank 99 to 225 | 0.152 | 233.8 | 124.2 | 65.8 | 418.56 | 24.37 | 24.39 | 34.00 |
| shared bank, 23/64 | 23 matrices, rank 64 | 0.131 | 240.0 | 127.5 | 67.5 | 140.01 | 23.78 | 23.81 | 33.58 |
| low rank, every layer, 23/64 | rank 342 | 0.134 | 239.8 | 127.4 | 67.5 | 524,844 | 30.17 | 30.19 | 37.59 |
| low rank, same layers kept whole, 23/64 | rank 147 to 273 | 0.116 | 239.9 | 127.5 | 67.5 | 304.00 | 23.85 | 23.91 | 33.07 |
| routed bank, 8/64 | 8 matrices, rank 64 |  | 174.2 | 92.6 | 49.1 | 1.10e+08 | 36.98 | 37.02 | 49.03 |
| routed bank, soft start, 8/64 | 8 matrices, rank 64 |  | 174.2 | 92.6 | 49.1 | 1.10e+08 | 41.53 | 41.57 | 55.49 |

## Sharing against the controls

Perplexity after healing at 16-bit weights.

| bank size/rank | layers stored whole | MB | shared bank | low rank, every layer | low rank, same layers kept whole | bank / same-layers control |
|---|---|---|---|---|---|---|
| 8/16 | 3 of 30 | 157.7 | 41.64 | 55.01 | 42.55 | 0.98 |
| 8/64 | 3 of 30 | 174.1 | 36.75 | 43.49 | 37.15 | 0.99 |
| 15/16 | 9 of 30 | 193.7 | 31.67 | 37.48 | 31.93 | 0.99 |
| 15/64 | 9 of 30 | 206.4 | 29.49 | 35.05 | 30.01 | 0.98 |
| 23/16 | 20 of 30 | 233.9 | 24.27 | 30.51 | 24.37 | 1.00 |
| 23/64 | 20 of 30 | 240.0 | 23.78 | 30.17 | 23.85 | 1.00 |

Against low rank on every layer the bank is ahead at every size, with 15 to 24% lower
perplexity at equal bytes. Almost all of that is the layout and not the sharing: the bank's
grouping leaves 3, 9 or 20 layers whole and concentrates the damage in the rest, and a model
that keeps the same layers whole and spends the same bytes on separate low-rank layers lands
within 2.2% of the bank at every size. What sharing itself adds is that last column: between
nothing and 2%.

**Stacked with quantization.** Rounding a healed arm to 8 bits costs it nothing, as it costs
the original nothing. At 4 bits the healed arms lose 24 to 51% of perplexity where the
original loses 53%. No arm at 4 bits is better than the original at 4 bits (31.8 at 76 MB),
but the stacked arms are the only models measured below the 4-bit original's 75.7 MB that
still work: the bank at
23/64 and 4 bits is 67.5 MB at 33.6, at 15/64 it is 58.1 MB at 38.6, while the original
rounded to 3 bits is 58.9 MB at 421.6. That comparison flatters the stacked arms, which were
healed, against a 3-bit original that was not; a calibrated 3-bit quantizer would be the fair
opponent and was not run. At 4 bits the same-layers control's perplexity runs from 2% below the
bank's to 11% above it, and that of low rank on every layer is 9 to 29% above it.

Three seeds of the 15/64 arms (the seed sets the healing batches):

| arm | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| shared bank | 29.49 | 29.36 | 29.37 | 29.40 |
| low rank, every layer | 35.05 | 34.44 | 34.98 | 34.82 |
| low rank, same layers kept whole | 30.01 | 30.06 | 29.97 | 30.02 |

At this size the bank's 2% over the same-layers control is outside the seed spread: the
bank's three seeds span 0.13 of perplexity, the control's 0.09, and the means differ by 0.61.
The other five sizes have one seed each.

Which layers share:

| bank size | layers that share a matrix (layers are numbered from 0) |
|---|---|
| 8 | 3, 4, 5, 6; 7, 8, 9, 10, 11; 12, 13, 14, 15, 16, 17, 18, 19, 20; 21, 22, 23, 24, 25; 26, 27, 28, 29 |
| 15 | 4, 5, 6; 8, 9; 10, 11; 12, 13, 14, 15, 16, 17, 18, 19, 20; 21, 22, 23; 28, 29 |
| 23 | 13, 14, 15, 16, 17; 18, 19, 20; 21, 22 |

## The routed variant

The routed variant was run at 8 bank steps and rank 64. A router, the score router of the first report, picks one of the 8 bank steps for
every token at every layer; each layer keeps its corrections. A per-layer prior starts every
layer on the bank step the fit assigned it, so before healing the routed model is exactly the
shared bank with its fixed assignment. It is healed the same way and at the same learning rate, sampling picks in
training and taking the highest score in evaluation. The router adds 0.08 MB.

Two starts were run. With a strong prior (8 added to the assigned step's score, so a sampled
pick leaves the assignment about once in 400) the router never moves: after healing, under one pick in a million is off the
fixed assignment and perplexity is 36.98 against the shared bank's 36.75. With a soft prior
(2, so training samples other bank steps about half the time) healing is worse, 41.53, and
the router ends back on the assignment, with 0.1% of picks off it. Both runs end with the router on
the fixed assignment, and sampling other bank steps on the way there costs perplexity. That
agrees with the first report, where a per-token pick lost to the fixed map on the same banks.

| arm | MB | perplexity before healing | healed | picks off the fixed assignment |
|---|---|---|---|---|
| shared bank | 174.1 | 1.10e+08 | 36.75 |  |
| routed bank | 174.2 | 1.10e+08 | 36.98 | 0.0% |
| routed bank, soft start | 174.2 | 1.10e+08 | 41.53 | 0.1% |
| low rank, same layers kept whole | 173.9 | 254,118 | 37.15 |  |
| low rank, every layer | 174.0 | 328,203 | 43.49 |  |

## What changed from the plan, and why

- **Hidden units are matched before sharing.** The first design fitted the bank to the
  weights as published. The similarity measurement showed those have nothing in common (cosine 0.0007),
  and that a feed-forward step's hidden units can be reordered freely without changing what it
  computes. Each layer's units are therefore reordered to line up with its bank step. It
  lowers the fit's weight error at every size, by 5 to 7%:

| bank size/rank | weight error, units as published | weight error, units matched |
|---|---|---|
| 8/16 | 0.652 | 0.621 |
| 8/64 | 0.491 | 0.468 |
| 15/16 | 0.433 | 0.406 |
| 15/64 | 0.312 | 0.294 |
| 23/16 | 0.200 | 0.185 |
| 23/64 | 0.141 | 0.131 |

- **A layer alone on its bank step gets no correction.** Its bank step is the layer itself, so
  a correction would store bytes that reconstruct nothing.
- **A second control.** The first design had one control, low rank on every layer. The bank turned out
  to beat it mostly by leaving some layers whole, which is not sharing, so the same-layers
  control was added to separate the two.
- **Layer groups come from the similarity measurement** (clustering) and not from a search
  over assignments. Every group it found is a run of neighbouring layers.
- **The quantized arms use plain rounding** written here, so that the same code quantizes the
  original and every healed arm and the bytes are counted the same way.
- **A second routed run with a softer start**, described in its section.

Learning rates tried, perplexity on the validation split after 500 steps, the chosen rate in
bold:

| arm | 1e-05 | 3e-05 | 0.0001 | 0.0003 | 0.001 | 0.003 | 0.01 |
|---|---|---|---|---|---|---|---|
| shared bank |  | 76.60 | 51.91 | 41.35 | **37.21** | 41.60 |  |
| low rank, every layer |  | 357.18 | 103.76 | 60.38 | **57.71** | 165.29 |  |
| low rank, same layers kept whole |  | 85.41 | 53.07 | 41.59 | **38.25** | 44.56 |  |

## Prior work

Sharing weights across layers is old when it is trained in: ALBERT (Lan et al., 2019) shares
one set of weights across all layers, and Takase and Kiyono (2021) assign N sets to M layers
in fixed patterns; the first report's fixed map is their cycle pattern. Deep Compression (Han
et al., 2015) shares scalar weight values by clustering, not matrices across layers.

Retrofitting sharing onto a trained model, which is what this report does, has been done at
larger scale. DeltaLLM (Mikaelyan et al., 2025) makes later blocks reuse an earlier block's
weights plus low-rank deltas and trains the deltas by distillation; it reports about 12% fewer
parameters at about 90% of base performance. Verma, Murray and Duh (2025, "Merging
Feed-Forward Sublayers for Compressed Transformers") match hidden units by permutation, then
merge and tie feed-forward steps of trained models. Basis Sharing (Wang et al., 2024) shares
an SVD basis across adjacent layers with per-layer coefficients and reports better perplexity
than per-layer SVD at 20 to 50% compression of LLaMA-7B, while leaving out the feed-forward
down matrix and the attention output matrix because sharing them raised error. MASA (Zhussip et al., 2025) and LISA (Mu et al.,
2024) share attention weights across layers. Matching hidden units by permutation comes from
model merging (Git Re-Basin, Ainsworth et al., 2022). Low rank alone is SVD-LLM (Wang et al.,
2024) and ASVD (Yuan et al., 2023), both of which weight the error by activations, which the
controls here do not. Stronger quantizers than the rounding used here exist (GPTQ, Frantar et
al., 2022). Gromov et al. (2024) remove whole layers and heal, finding late layers the most
expendable.

What this report adds to that is small: the same idea on a 135-million-weight model with a
weights-only start, a measurement of how little the layers have in common, and a control that
separates sharing from the choice of which layers to leave whole, which the retrofitting
papers above do not report. It does not add a better method. Its finding, that the gain
credited to sharing is mostly a gain from uneven treatment of layers, is from one small model.

## Limits

- One model, and a small one. The papers above report sharing working on models of billions
  of weights.
- One evaluation, perplexity on WikiText-2, with healing on text from the same source
  (WikiText-103). The healed arms may have adapted to that text in a way the unhealed
  quantized original could not, which favours the healed arms.
- Healing is short (2,000 steps, about 4 million tokens) with the learning rate annealed to
  zero. A longer budget was not tried and could reorder arms that are close.
- The learning rate was chosen once per kind of arm, at 500 steps on the 15/64 size,
  and used for every size at 2,000 steps. The routed arms use the shared bank's rate
  and were run at one size.
- One seed everywhere except the 15/64 size.
- The fit minimises weight error with no data. Activation-weighted fits, as in SVD-LLM, would
  help both the bank and the low-rank controls.
- The quantization baseline is rounding with no calibration; a stronger quantizer would lower
  the black line.
- Only the feed-forward steps are banked. Attention matrices were measured (they are as
  unrelated as the feed-forward ones) but not shared.
- Stored bytes are the measure. Speed and memory at run time were not measured; the low-rank
  corrections add work per token and the routed arm computes several bank steps in training.

## Reproduce

```sh
nix-shell --run 'python -m weightbank.compress'
```

It needs Nix (`shell.nix` pins the environment), a CUDA GPU (the runs here used one with 8 GB)
and about 430 MB of downloads, which go into `data/compress/` and are checked against pinned SHA-256 sums. It runs
every arm that `runs/compress/` lacks and rewrites this report and its two charts. With
`runs/compress/` as committed it only rewrites the report; delete that directory to run
everything again, about six and a half hours of healing on one RTX 2080.
