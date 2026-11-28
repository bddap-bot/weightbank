# Routing into a shared weight bank against smaller plain models

Every table below is written by `python -m weightbank.report` from the run files in `runs/`.
Loss is cross-entropy on the validation split in nats per byte (divide by 0.693 for bits).

Two experiments are reported together. The first trained every configuration for 8,000 steps
at one learning rate (1e-3), three seeds. The second asks which of the first's gaps belong to
the designs and which to a short or mis-tuned comparison: every configuration gets a learning
rate chosen for it by sweeps, then three seeds at four times the budget (32,000 steps). The
second also adds configurations, trained at both budgets: two other routers for the model
with both banks routed, the fixed module-to-chunk map with no router at 1 and 2 chunks beside
the first experiment's 4 and 8, and the plain models they are compared with.
A gap is a configuration's mean validation loss minus the better of the two plain curves read
at its stored weights; negative means the configuration beats a plain model of its size.

## How a run ends

A run ends after a fixed number of optimizer steps, 8,000 or 32,000, set before it starts.
Nothing watches the validation loss: no run stops early for having learned enough, and none
is trained until its loss stops falling. The one exception is a run whose training loss
stops being a finite number, which is ended and recorded as lost. The learning rate climbs for 200 steps and then follows a cosine down to a tenth of
its peak, which it reaches at the last step. The loss reported for a run is the validation
loss of the weights as they stand after that last step, over the whole validation split; no
earlier or best checkpoint is kept.

No run here is converged. The table sets each cell stopped at 8,000 steps (seed 0, the best
of the three rates tried) beside the same cell stopped at 32,000 (seed 0), and gives how far
the logged loss moved over the last 2,000 and the last 500 steps of the long run, averaged
over its three seeds. Four times the steps bought 0.064 to 0.089 at
every cell. Over the last 2,000 steps of the long run, by which point the rate is within a
tenth of its floor, the loss still falls by 0.002 to 0.004; over the last 500 it moves by
less than 0.001 in either direction. The curve is flat at the last step because the
schedule has brought the rate to its floor there, not because the model has stopped
learning: a longer run would end lower. Every comparison in this report is therefore
between runs of equal steps, and none is between converged models. Whether the order of
the model that picks per token and the fixed map survives four times the steps again is an
open hypothesis in the ledger (`picking-too-few-steps`).

{{run-end}}

![Validation loss along training](curves.png)

The curves are the four modern cells the reading leads with, logged every 500 steps on the
first 2,048 validation windows, which read about 0.02 above the loss over the whole split.
The plain model at full width is the lowest from about step 2,000 on and the plain model of
four modules the highest from step 1,000 on. The shared model starts below the plain model
narrowed to width 184, runs 0.002 to 0.005 above it from about step 8,000 to step 18,500,
crosses back between steps 19,000 and 20,000 and ends 0.012 below it. All four are flat
over the last 500 steps.

## Reading

The pair to judge sharing by holds compute equal. In the modern architecture, eight modules
at width 256 that each own their weights store 6,444,192 weights and reach 0.9819 nats per
byte at 32,000 steps. The same eight modules sharing four chunks through the fixed map store
3,289,936, 51% as many, and reach 1.0012. The two multiply by the same number of weights
per token, so the difference is what halving the stored weights costs at equal compute:
0.019 on average, and 0.015, 0.022 and 0.021 at the three seeds, where each model's seeds
have a standard deviation of 0.002 to 0.003. Sharing is not free. It is the cheapest of the
three ways tried here of storing half the weights: against the same full model, dropping to
four modules costs 0.058 and narrowing all eight to width 184 costs 0.031, and those two
also do about half the arithmetic per token, which sharing does not save. On the GPU the
shared model holds 13 MiB of weights against the full model's 25, and that is nearly all
it saves: a batch of 256 windows peaks at 508 MiB for one and 507 for the other, and a
training step at 1,879 and 1,916, where the 37 MiB between them are the gradient and
optimizer copies of the weights the shared model does not have (the table under "GPU
memory").

The fixed map is not new: sending module i to chunk i mod N is the cycle strategy of Takase
and Kiyono (2021, "Lessons on Parameter Sharing across Layers in Transformers",
arXiv:2104.06022), which stacks M layers with independent parameters and repeats them in
the same order, and which they present as relaxing the sharing of one layer's parameters
with all layers, as in Universal Transformers (Dehghani et al., arXiv:1807.03819). A
vendor's release page for a model of about 9 billion parameters, SenseNova Loop-9B,
describes reuse across layers at that scale, in a layout with ends of its own: by the
vendor's account it stores 32 decoder layers and computes 56, running the middle 24 twice
with the same weights and the first four and last four once. No number of theirs is checked here. A map with ends of its own is
in the ledger as an open hypothesis (`own-ends-beat-the-all-shared-map`).

Three findings of the first experiment held in sign and shrank in size: attention routed
stays above the plain curve (+0.026 at the first budget, +0.021 at the long one), both routed
stays above it (+0.030, now +0.012), and the fixed map with no router stays below it (-0.015,
now -0.008) and still beats both routed on the same banks, by 0.019 where it was 0.045. One
moved: feed-forward routed went from -0.003, inside the seed spread, to +0.005 with all three
seeds above the curve (+0.003 to +0.008), so at the long budget it is slightly worse than a
plain model of its size and no longer level with it; part of that is the plain curve, since
at tuned rates the narrowed plain model became the better curve, and against the curve of
fewer modules, the better one at the first budget, feed-forward routed sits at -0.001. The
signs of those three are therefore properties of the designs, while both routed lost three fifths of its
gap, the fixed map half and attention routed a fifth to the longer budget and tuned rates:
choosing a chunk per token costs loss against a plain model and against fixed sharing of the
same weights, and fixed sharing is the only arrangement here that beats a plain model of
equal storage.

The loop described below has since run eleven rounds at the long budget; the cells they
tried are the stars on the chart. The fourth is the equal-compute pair above. Rounds 5
to 8 trained four fixed maps drawn at random beside the mod map, in the modern architecture
at equal stored weights: two are worse than the mod map at every seed, by 0.020 and 0.011,
and two are level with it (0.002 above and 0.0001 below on average), so at the mod map's
learning rate which module uses which chunk matters, by more than the training seed does,
and no drawn map is separated from the mod map on the better side (the section "Fixed maps"
has each map). Round 9 froze each module to its router's most common pick after a
quarter of training: 1.0683 against the mod map's 1.0134 and the routed model's 1.0321.
The frozen maps use only two or three of the four chunks (the section "Freezing the
picks"). Round 10 stopped the router after a quarter of training instead and let every
module go on picking per token: 1.0336, 0.0015 above the routed model and inside its own
seed spread, so what the router learns after the first quarter makes no measurable
difference to the pick's 0.019 over the mod map. Of the three rival explanations one is
supported by its rule, one refuted and one left open with its comparison finished (the
section "Stopping the router"). Round 11 gave the
first and the last module chunks of their own and cycled the six modules between them
through the other two chunks, at the same stored weights: 1.0157 against the mod map's
1.0012, worse at every seed. A low-rank correction per module on one shared chunk does
not beat a second chunk at equal stored weights. Rounds 2 and 3 took the fixed map, eight
modules reusing four chunks with no router, into a modern transformer (rotary positions,
root-mean-square norms, a gated feed-forward) and set it against two plain models. The
shared model reaches 1.0012. Four modules that own the same four chunks reach 1.0403, which
is 0.039 above. The stronger control keeps all eight modules and is narrowed: at width 184
it reaches 1.0133, so the shared model is 0.012 below it on average and below it at every
seed, by 0.012, 0.010 and 0.015, where each model's seeds have a standard deviation of
0.003.

How equal stored weights was reached: with one width, not exactly. A width has to be a
multiple of 8, so that each of the four heads has an even number of dimensions to rotate in
pairs. Width 184 stores 2.1% more weights than the shared model and width 176 would store
6.4% fewer, so the control is the nearer of the two and the larger model. If a modern plain
model gains from width as the classic ones do, the margin at exactly equal stored weights is
a little above 0.012: on the classic narrower curve 2.1% of stored weights is worth 0.002.
No second modern width has been run to show that. In the classic architecture the shared
model of the same shape is 0.008 below the narrower plain curve at equal stored weights. So
the gain did not shrink in the modern design: 0.012 over a larger plain model, against 0.008
at equal stored weights in the classic one. One comparison in each architecture does not say
by how much the two gains differ.

That is the result nearest to something to adopt: in this small byte-level transformer with
the modern parts, a model limited by stored weights reaches a lower loss as eight modules
sharing four chunks than as either plain model of that storage that was run, four modules at
full width or eight modules narrowed. Round 2's 0.039 was measured against the weaker of the
two controls. The modern plain model of four modules reads 0.008 worse than the same cell in
the classic architecture, while the modern shared model and the modern narrowed model read
better than their classic counterparts, by 0.012 and 0.008 (the table under "The modern
cells beside the classic architecture"). Of the three modern plain cells run, then, one reads
worse than its classic counterpart, one reads better, against an interpolated classic curve
and not a run of its own width, and the full model reads level with it (0.001 lower, inside
its seed spread), so the plain model reading worse is not general to modern plain models
here. No run isolates why the cell of four modules reads
worse, and no cause is claimed. Against the control that does not read worse the margin is
0.012, under a third of 0.039.

What this does not show. It is one dataset of bytes, one size (3.3 million stored weights),
one bank size (four chunks for eight modules), one training budget and three seeds. The
modern plain control is one width and one depth: it is 2.1% larger than the shared model
instead of an interpolation between two widths, and no other mix of depth and width at that
storage, such as six narrower modules, has been run in the modern architecture. Learning
rates are chosen on seed 0, and the narrowed model's two best rates end 0.0003 apart in
loss. Against these two controls stored weights are held nearly equal and compute is not: the
shared model runs eight modules at width 256, which multiply by about 1.9 times the weights
per token of eight modules at width 184 and twice those of four modules at width 256; the
pair at the head of this reading is the one that holds compute equal and lets storage
differ. On this GPU a training step of the
shared model and of the narrowed one took the same time, about 110 ms. Whether a per-token
pick fares differently in the modern architecture has not been run.

![Validation loss against stored weights](chart.png)

The chart is the long budget, whole on the left and enlarged between 3 and 7 million stored
weights on the right. The dark line is the plain model narrowed from width 256 to 96 at 8
modules; the lighter line is the plain model cut from 8 modules to 1 at width 256. Bands and
bars run from the lowest to the highest of three seeds. The dashed line joins the fixed
module-to-chunk map with no router at 1, 2, 4 and 8 chunks. The fixed map at 4 chunks and the
three models with both banks routed store the same weights to within 1.2%; their four points
are drawn side by side so that none hides another, which the chart notes under its panels.
Points are means of three seeds. A star is a cell with three seeds at this budget beyond
those of the first two experiments, tried by the loop described next; it is drawn just
right of its stored weights so that it hides no point of equal size. The two lines and the
other points are the classic architecture; the four small stars are the maps drawn at
random, set just left of the shared modern model they are compared with, and the large stars of the modern architecture are
read against one another, not against the lines: at about 3.3 million stored weights the
shared model lowest, the plain model narrowed to width 184 above it and the plain model of
four modules highest, with the map with own ends just above the narrowed model, and at 6.4
million the plain model of eight modules at full width, below them all. The legend label of every point and
star is spelled from the plain words the ledger gives each axis value: what the cell
shares, the values it has off the default, and for a cell that shares weights also how a
module gets its chunk and how many chunks a bank holds. A line's label opens with the same
words for what its cells have in common.

## The design space and the loop

This section is generated from [ledger.json](ledger.json), the record of the design space:
its axes, every cell tried with the run files behind it, the hypotheses, and each round the
loop has run. The sections after it write up the two experiments that came before the loop. A cell is one value
on every axis. A cell's name is the tags of its values that differ from the default, in the
order of the axes, and `full` when none does; the training budget is the directory the run
file sits in. Learning rate and seed are not axes. A cell at the first budget is run at
three seeds at one rate and swept over three rates at seed 0; a cell at a longer budget gets
its rate from the sweep described under "Choosing the learning rate" and is then run at
three seeds at that rate.

### Axes

{{axes}}

Adding an axis or a value is one edit to the ledger's `axes`. Names, the cells a hypothesis
may name, the loop and the tables here follow from it; cells already tried sit at the new
axis's default. The model has to know the value: a cell naming one it cannot build is
refused. With nothing shared, `pick`, `chunks` and `rank` stay at their defaults.

### Hypotheses

A hypothesis claims that one cell reaches a lower validation loss than another, and lists
the comparisons that would settle it. A comparison is finished when each of its cells has
three seeds at one learning rate, its own. It favours the hypothesis when the lower cell is
below the higher one at every seed and the mean difference exceeds both cells' seed spread
(standard deviation), and goes against it when the lower cell is above by the same test. A
hypothesis is `supported` when all its comparisons are finished and favour it, `refuted`
when all go against it, and `open` otherwise; one with no comparison stays open and is
never picked. A lower priority number is run first. Priorities and hypotheses are edited by
hand in the ledger; statuses are computed.

{{hypotheses}}

### How a round picks its cell

`python -m weightbank.loop --gpu-hours H` runs one round:

1. Take the open hypotheses that have an unfinished comparison, lowest priority number
   first, ledger order among equals.
2. Cost each unfinished comparison: the steps its two cells still lack under the run
   protocol of "Choosing the learning rate" (three runs at the first budget, at least two
   at the cell's own budget to choose the rate, then the other two seeds), at the seconds
   per step of the tried cell that differs on the fewest axes, the slowest among ties. A
   tried cell's seconds per step are those of its fastest run with a finite loss at any
   budget: a run that sat stopped or shared the card only gained time, so the fastest is the
   nearest to training alone.
3. Pick the first hypothesis with a comparison costing at most H GPU-hours, and its
   cheapest such comparison. Hypotheses passed over, as too costly or for a rate sweep that
   cannot go on, are named in the round.
4. Train what the two cells lack, one run at a time; a cell that already has its three
   seeds costs nothing. Then rewrite the ledger's cells and statuses from the run files, and
   this report and chart from the ledger.

The estimate is a floor, since a rate sweep that lands on the edge of its grid adds runs,
and it prices a new value of an axis at the pace of the old one. A run's training seconds
are summed over its 500-step log windows, each window at the median pace of itself and the
two windows on either side, which leaves out a stretch of up to two windows that the
process sat stopped while other work held the card, keeps a pace that changes during
training, and does not count the validation pass after each window. The runs before round 2 recorded
the wall time inside their training loop instead, stops included; their files keep those
seconds, and the estimate reads past them by taking each cell's fastest run.
With no hypothesis to pick the loop trains nothing. A comparison whose runs end without
three finite seeds is marked abandoned in the ledger and not picked again.

### Rounds

{{rounds}}

### The modern cells beside the classic architecture

Each cell of the modern architecture that has three seeds at the long budget, beside the
same cell in the classic architecture where that cell has runs. A plain model narrowed in
width alone that has no classic twin is set beside the classic narrower plain curve read at
its stored weights, interpolated seed by seed as the gaps below are. One cell does not store
the same weights in both architectures: the modern one has no position table, its norms have
no bias, and its feed-forward hidden width is rounded down. Each cell is at its own chosen
learning rate.

{{beside-classic}}

In the modern architecture the shared model reads 0.012 lower than its classic counterpart
and the narrowed plain model 0.008 lower, the full plain model reads level with its own
(0.001 lower, with seeds on both sides), and the plain model of four modules reads 0.008
higher. That cell is above its classic twin at every seed, by 0.002 to 0.013, and its seeds
have a standard deviation of 0.007, the widest of the four modern cells. The two plain
cells that move differ in depth and width at once (four modules at width 256, eight at width 184), and
no run varies one without the other or changes one part of the modern design at a time, so
the table shows where the modern plain model reads worse and not why. What it changes is the
size of the gain from sharing: the shared model's margin over the plain model of four
modules, 0.039, is about three times its margin over the narrowed one, 0.012.

### GPU memory

Peak memory held by PyTorch's allocator on the GPU while a cell runs, measured by
`python -m weightbank.memory` on freshly initialised weights and random bytes: the weights
alone as 32-bit floats; one forward pass at inference over 1 window of 256 bytes and over
the 256 windows of a validation batch, with nothing kept for gradients; and three training
steps of 64 windows, with the gradients and the optimizer's two moments per weight. Every
peak includes the weights. It is the allocator's peak and not the process total, which adds
the CUDA context, the same for every cell, and it leaves out the 95 MiB
of text a training run also keeps on the GPU.

{{memory}}

At inference the shared model saves the weights it does not store and nothing else. Its weights take 13
MiB against the full plain model's 25. A pass over one window peaks 8 MiB above the weights
for the shared model and 14 MiB above them for the plain one; a 16-bit copy of the weights
would take 6 and 12 of those, and the pass runs as validation does, under 16-bit autocast
over 32-bit weights. Over 256 windows the activations are
nearly all of the peak and the two models are level, at 508 and 507 MiB. In training the
plain model's 37 MiB more is three further copies of its 12 MiB of extra weights, a
gradient and two optimizer moments. The plain model of four modules computes half as much
and trains in about half the memory. The model narrowed to width 184 peaks at 989 MiB over
256 windows, more than the full-width model. A one-off probe of plain modern models of
eight modules at seven widths, measured the same way and not kept in `runs/memory.json`,
peaks at 309, 373, 440 and 507 MiB at widths 160, 192, 224 and 256, whose four heads each
have a multiple of 8 dimensions, and at 968, 989 and 1,027 MiB at widths 176, 184 and 200,
whose heads do not. The probe shows where the larger peak falls and not what causes it.

### Fixed maps: the mod map, maps drawn at random, and own ends

A fixed map says which chunk each module computes with, for the whole run and for every
token. The mod map sends module i to chunk i mod N. The map with own ends gives the first
module and the last a chunk each that no other module uses and cycles the modules between
them through the remaining chunks: 0 1 2 1 2 1 2 3 for eight modules over four chunks. A drawn map is drawn before training
from a random generator seeded with the draw's number alone, so one draw is one map at
every training seed, and the training seed still sets the initial weights and the batches.
Each module's chunk is drawn uniformly and independently; a draw that leaves a chunk unused
is thrown away whole and drawn again, so a drawn map always uses the whole bank and stores
exactly the weights of the mod map. An unused chunk therefore never arises and never has to
be counted: a map that left one out would be a smaller bank, which the `chunks` axis
already covers. A drawn map may put neighbouring modules on one chunk and may load the
chunks unevenly. Which chunk is called 0 means nothing, so the table numbers each map's
chunks in order of first use; two draws that give the same row are the same map. A map
other than the mod map trains at the mod map's learning rate, so that the map is the only
thing that differs.

{{maps}}

{{map-spread}}

At this learning rate the map matters, and by more than the training seed does: the six
maps' means, the mod map's among them, have a standard deviation of 0.0084, against 0.0016
on average across the three seeds of one map (0.0087 and 0.0015 over the mod map and the
four drawn maps alone). The hypothesis that the mod map beats a map
drawn at random stays open by its own rule, because two of the four drawn maps are not
separated from it. Draws 1 and 2 are worse than the mod map at every seed, by 0.020 and
0.011. Draws 3 and 4 are level with it, 0.002 above and 0.0001 below on average; each is
below it at seed 1 alone, the mod map's own worst seed. No drawn map is separated from the
mod map on the better side. The two that lose each put three pairs of neighbouring modules on one chunk, and the
two that are level put two; the mod map puts none. Four draws cannot separate that from the
other ways the maps differ, such as how unevenly they load the chunks (draw 4 puts four
modules on one chunk and is level) or which modules share with the first and the last. Nor
do they separate the map from the learning rate: every map trained at the rate chosen for
the mod map, and a map that loses might lose less at a rate of its own. Draw 4 is the one
draw whose first attempt left a chunk unused and was drawn again. Four draws are a small
sample of the 1,701 distinct ways eight modules can use all of four chunks. The map with own ends, 0 1 2 1 2 1 2 3, is worse than the mod map at every seed, by
0.018, 0.012 and 0.013: 1.0157 against 1.0012, so `own-ends-beat-the-all-shared-map` is
refuted. At equal stored weights own ends leave two chunks for the six modules between
them, each used three times, and no neighbouring modules share a chunk, so what it loses
by is not the neighbour count the drawn maps suggested. This is not a test of the vendor's
model named in the reading, whose own ends are layers added around a loop at a far larger
size; it says only that in this model, with the bank held at four chunks, taking two of
them for the ends costs more than it returns. Sixteen modules
over the same four chunks, where a drawn map can differ from the mod map in more places, is
in the ledger as an open hypothesis (`the-map-matters-more-with-more-modules`) and has no
runs.

### Every cell

One row per cell and budget, from the run files; the rate, mean and seed spread are filled
where three seeds share a rate.

{{cells}}

## The gaps at both budgets

Each plain curve is read at the configuration's stored weights by interpolating mean loss
linearly in the logarithm of stored weights, and the gap is taken to whichever curve is lower
there. Seeds share batches across configurations, so the gap is also computed seed by seed
(the configuration at seed s against that same curve drawn from the plain runs of seed s);
the range of those three is beside each mean. At the long budget seed 0 is also the seed the
learning rates were chosen on, so each seed-0 loss is the lowest of the rates tried; the
column without seed 0 is the gap over the two seeds that took no part in the choice.

{{both-budgets}}

Between the two budgets both the learning rate and the number of steps change. The next
table changes the rate alone: the first budget, seed 0 only, with every configuration at 1e-3
and then at the best of the three rates tried for it at that budget (the short sweep below),
both read against the plain curve the first-budget gap above uses. Each loss on the
right-hand side is the lowest of three, and a one-seed gap carries about a seed spread of
noise (0.002 to 0.009 in the per-configuration table); it can show a shift of the size of
the attention-routed and both-routed gaps and not one of the size of the feed-forward routed
gap.

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

The rate is chosen in two stages, each one seed (seed 0), and the choice is made at the
budget it is used at. The warm-up (200 steps) and the cosine decay to a tenth of the peak are
the same at every rate and budget.

Short sweep: 8,000 steps at peak rates 5e-4, 1e-3 and 2e-3. The 1e-3 run is the first
experiment's seed-0 run. Lowest loss in bold; the last column is how far the next-best rate
is behind it.

{{short-sweep}}

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

{{long-sweep}}

Two kinds of cell have no sweep of their own and train at another cell's rate, so that
the pair differs in one thing: `both-c8`, the full control computed through the bank code,
at the full control's rate; and a fixed map other than the mod map at the mod map's rate.
At the first budget such a cell runs its three seeds at 1e-3 like any other.

Limits of this choice: it is made on one seed, so where two rates differ by less than the
seed spread the pick between them is arbitrary; and a winner that is the best short rate
and leads by 0.005 or less has no rate tried beyond it, so a rate beyond it could be better
by a similar amount. The feed-forward routed gap is of that size at both budgets.

## What was cut

- The long sweep starts from two rates per configuration and adds more only under the rule
  above. Six configurations ended with two long rates; a third for each would have added
  about three hours of the GPU.
- The two other routers were run for the model with both banks routed only. Feed-forward-only
  and attention-only versions of them would be four more configurations, about 21 hours of
  the GPU, and the both-routed pair already answers what they were added to ask.
- No configuration was trained under another optimizer: a check under Muon would need its own
  rate sweeps and long runs and was left out.

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
- Fixed map (`both-c1`, `both-c2`, `both`, `both-c8`): no
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
- Hardware: one RTX 2080, 8 GB, shared with other work, so the times of these two
  experiments are indicative: their runs recorded wall time, and one slowed or paused by
  other work on the GPU counts that time. Runs
  were trained one at a time. During
  training a routed step computes every chunk for every token (the straight-through gradient
  needs them), which is why routed runs take longer; a hard pick needs only the picked chunk
  at inference, which this code does not exploit. `both-c8` and `both` pay a
  bookkeeping cost per chunk that `full` does not, so their times are not comparable to it.

## Sanity check: N = M, every module forced to its own chunk

`both-c8` runs the routed code path with a bank of 8 chunks of each kind and no router:
module i always computes with chunk i. It starts from the same initial weights and sees the
same batches as `full` at the same seed, so the comparison is paired by seed. First budget:

{{sanity-first}}

Long budget, both at the full control's rate:

{{sanity-long}}

At the first budget the per-seed differences (+0.0004, -0.0001, +0.0000) are a fifth or less
of the full control's seed sd (0.0021). At the long budget they are +0.0025, +0.0004 and
-0.0005, mean +0.0008, against a seed sd of 0.0012: the two start from the same weights and
compute the same function in a different order of operations, and over 32,000 steps of
half-precision arithmetic their trajectories drift apart by about as much as two seeds do.
The routed code path with N = M reproduces the full control at both budgets.

## Why did the fixed map beat the learned router?

In the first experiment `both`, the same two banks with module i fixed to chunk
i mod 4, beat `both-routed` by 0.045. Two things separate them besides the choice itself: the
routed model trains under sampling noise (every pick is drawn with Gumbel noise), and its
router has one particular form. `both-routed-argmax` removes the noise and
`both-routed-keys` changes the form. All four store the same banks; at the same seed they
see the same batches. Long budget, each at its chosen rate:

{{routers-long}}

First budget, every one at 1e-3:

{{routers-first}}

Neither the sampling noise nor the router's form explains it. Removing the noise made the
model worse at both budgets: `both-routed-argmax` is 0.014 above `both-routed` at the long
budget (1.0459 against 1.0322, every seed) and 0.013 above it at the first, and its router
ends far less decided (it gives its pick a probability of 0.51 to 0.56, where the two sampled
both-routed models give 0.85 to 0.96), so the noise was helping the router commit and not costing loss.
Changing the form changed nothing: `both-routed-keys` ends at 1.0321, within 0.0001 of
`both-routed`, and at the first budget it is 0.005 above it, inside its own seed sd of 0.009.
All three routed models stay 0.019 to 0.033 above the fixed map at the long budget (0.045 to
0.058 at the first). What separates the routed models from the fixed map is therefore
something these two arms leave in place: every module choosing among all four chunks token by
token, trained through a hard pick. The longer budget and tuned rates closed three fifths of
the distance (0.045 to 0.019) without either change to the router.

### Freezing the picks

Round 9 of the loop tested one explanation of the distance: that routers lock in early
picks which make a poor map. `both-frozen-picks` trains as `both-routed` for a quarter of
the run, then fixes every module to the chunk its router picked most often on training text
and trains on with no router. It reaches 1.0683, against 1.0134 for the mod map over the
same banks and 1.0321 for the model that keeps picking per token: 0.055 above the mod map,
at every seed (0.047 to 0.063), and 0.036 above the routed model. The hypothesis
`picking-lock-in` is supported by its rule, which asks only that the frozen model lose to
the mod map.

The run files show what the frozen maps are. Numbering chunks as stored, the three seeds
froze the feed-forward bank to 3 2 1 1 2 2 2 2, 1 3 0 0 0 0 0 0 and 3 2 0 0 0 0 0 0, and
the attention bank to 3 1 1 3 3 3 3 1, 1 1 2 1 1 1 2 2 and 0 0 0 0 2 2 0 2. Every one
leaves chunks unused, attention two of its four and feed-forward one, and puts four to six
modules in a row on one feed-forward chunk. The frozen model still stores all four chunks
of each kind, 3,394,952 weights, and ends above the mod map with two chunks, which stores
1,783,040 and reaches 1.0583. The logged validation loss rises from 1.20 to 1.32 across
the freeze and ends 0.03 to 0.04 above the routed model's.

What this does and does not say. The map a router's common picks make after a quarter of
training is a poor one to commit to. It does not show that the routed model loses because
it is locked into that map: the routed model beats its frozen counterpart by 0.036, and in
five of the six banks the routed runs' most common picks at the end of training are not
the map frozen at the quarter point. Nor does it separate the layout from the switch: no
model was trained from the start on one of these collapsed maps, so some of the 0.055 may
be the cost of changing the model a quarter of the way in.

### Stopping the router, and what the three picking hypotheses show together

Round 10 trained `both-stopped-router`: routed as before, but after a quarter of training
the routers' weights are held still while every module goes on picking per token. It
reaches 1.0336, against 1.0321 for the model whose router learns throughout: 0.0015 above
it on average (0.0004, 0.0030 and 0.0011 at the three seeds), inside the stopped model's
seed standard deviation of 0.0026. That finishes the comparisons of all three hypotheses:

- `picking-lock-in`, supported by its rule: picks frozen to one chunk per module lose to the
  mod map, by 0.055.
- `picking-chase`, open: its comparison is finished and does not separate the two. With the
  router held still for the last three quarters of training the model ends a little higher
  at every seed, by less than its own seed spread. If the per-token pick lost because the router and the chunks kept
  chasing each other, stopping the router should have helped; the numbers lean against that
  explanation without refuting it by the rule.
- `picking-worse-function`, refuted: it claimed that under a router that no longer moves,
  one chunk per module (the frozen picks) beats mixing chunks token by token (the stopped
  router). The frozen model is worse at every seed, by 0.027 to 0.040. Under the routers
  these models learn, picking per token is the better use of them, not the worse.

In plain words. The per-token pick ends 0.019 above the mod map. What the router learns
after the first quarter of training makes no measurable difference to that: a router
stopped there ends within 0.002 of one that keeps learning, while the chunks go on
training under it. The distance is not sampling noise or the router's form either (the two
other routers above). The frozen picks show one more thing, and suggest a reading of it.
What they show: each module's most common pick falls on two or three of the four chunks,
with long runs of modules on one chunk, where the mod map spreads the modules evenly, and
a model committed to those picks ends 0.055 above the mod map, while the routed model, by
going on picking per token, ends 0.036 below the frozen one. No chunk goes unused in the
routed model: pooled over modules its feed-forward picks are close to even, and its
attention picks are not (one chunk gets under a tenth of them and another nearly half; the
usage tables under "How much picks vary by input"). The reading this suggests, and no run
tests: the routers arrive at a poor assignment of modules to chunks, and the per-token pick
pays for it. What stays open is whether the 0.019 is the price of that assignment or of
mixing chunks token by token as such. None of the runs above has a per-token pick laid
over a good assignment. The next section builds two.

### A pick laid over the mod map: freed and held

Two pick values put a per-token pick over the assignment known to be good, the mod map.

`freed` starts as the mod map. For the first quarter of training every module computes
through its mod-map chunk and the routers are idle, so the chunks specialise under the map.
From then on the routers pick per token, and the score of the module's mod-map chunk
carries a bonus that starts at {{freed-lean}} and falls in a straight line to nothing by half of
training. At that bonus an untrained router leaves the map on {{freed-off-map}} of sampled picks,
so the handover does not jolt the model; the second half of training has no bonus, and the
model evaluated is the router alone. The quarter is the point the frozen and stopped
diagnostics above use.

`held` is a per-token pick from the first step with a constant bonus of {{held-lean}} on the score of
the module's mod-map chunk, kept at evaluation. An untrained router then puts {{held-on-map}} of a
module's picks on its mod-map chunk. The bonus is a standing lean, not a lock: a trained
router can outvote it token by token, and could learn to cancel it for a module outright,
so the table below is what says whether the assignment stayed the mod map's.

Both store what `both-routed` stores (the banks and the routers, 1.0% above the mod map)
and choose their learning rate by the same protocol. How far the picks sit from the mod map
at the end of training, on validation text, is in the table below. A model whose picks
never leave the map is the mod map and shows nothing about picking. The free per-token pick
is listed for scale only: nothing ties its routers' numbering of the chunks to the mod
map's, so three quarters of its picks fall off the map by chance.

{{off-map}}

## Reuse without a router, by number of chunks

The fixed map at 4 chunks is cycle sharing in the sense of Takase and Kiyono (2021, "Lessons
on Parameter Sharing across Layers in Transformers", arXiv:2104.06022): a stack of N layers
applied repeatedly. At 1 chunk it is ALBERT-style sharing (Lan et al., 2020, "ALBERT: A Lite
BERT for Self-supervised Learning of Language Representations"): one layer applied 8 times.
At 8 chunks it is the full control. Unlike both, every module here keeps its own layer norms.
Each is set beside the plain model of as many modules as
it has chunks, which stores the same chunks and applies each once, and beside the plain
curve at its stored weights.

{{fixed-map}}

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

### Both routed, no sampling noise

{{picks:both-routed-argmax}}

### Both routed, key per chunk

{{picks:both-routed-keys}}

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
A configuration's name is its cell's name in the ledger. `python -m weightbank.loop
--gpu-hours <hours>` picks and trains the next cell, and `python -m weightbank.report`
rebuilds this file and the chart.
