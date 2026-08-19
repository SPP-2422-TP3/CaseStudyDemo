# Project details

The data, the models and the research this demo rests on. See the [README](../README.md) for
what the app is and how to run it, and [OPERATIONS.md](OPERATIONS.md) for deploying it and CI.

## The status board

The landing page's run is **assembled, not recorded**. Every stroke on screen is a real measured
stroke shown with its own model's prediction; what is authored is the order they arrive in. The
data holds nine production runs at *fixed* wear levels and seven feed series at *fixed* infeed —
snapshots of states, never a transition between them — so a run that degrades has to be scheduled.

Two are offered, picked in the machine bar, and each carries **one** fault, because a board that
showed both at once could never be seen to tell them apart:

- **Scenario 1 · Tool wear.** Ironing goes off first (A1 → A2 around stroke 90) and reaches A3 near
  the end; deep drawing follows much later and over twice the span (T1 → T2 around stroke 190). The
  strip stays centred throughout. The two tools are deliberately not on the same clock — all nine
  T × A combinations were measured, so a stroke at T1/A2 is as real as one at T1/A1.
- **Scenario 2 · Strip misalignment.** Both tools stay fresh while the feed ramps 60.00 → 60.30 mm
  in the 0.05 mm steps the campaign actually ran, taking the cup from centred to 0.90 mm off.

Wear levels crossfade over their transitions — strokes are drawn from both recorded runs with a
ramping probability — so the rolling means drift rather than step. Nothing interpolates between two
strokes and no curve is synthesised.

Two further compositions: misalignment and wear come from separate measurement campaigns on separate
tooling, and the first strokes of each block are skipped because a cold die reads as a briefly worn
one. The board itself is laid out as press-side equipment, without even the site's top bar, and
carries none of this — a shop-floor screen is not where a caveat gets read.

Tool wear is reported on the three stages the shop floor already names — **fresh, worn, critical**,
the T1–T3 and A1–A3 of the trials. The stage badge is the classifier's majority call over the last
20 strokes; the marker's position along the track is the mean of the same classifier's probabilities
over that window, so a tool drifting toward the next stage sits between the two instead of jumping
when the majority tips. One instrument, read coarsely and finely.

There is no percentage, deliberately. The levels are ordinal classes with no measured roughness
behind them, nothing in the data records when a tool was retired, and so no fraction of life
consumed can be computed — the board does not invent one. The continuous friction axis of
`wear_position.py` is a different question and stays in the detail window, labelled as such. Strip
misalignment is likewise **one axis only**: the campaign varied overfeed along the feed direction,
so there is no second axis to predict.

Each card opens into the same per-stroke views the research pages use. Strip misalignment shows the
full stroke and the line fitted across its plateau; tool wear shows the raw force curve against the
mean curve of each wear level, then the same stroke coloured by where the model found its evidence.
An engineer who has seen those pages recognises the board immediately.

### Operator feedback

*Report bad parts* is the one control that carries information back into the board, so it sits below
the cards as a full-width action rather than a button in a toolbar. It opens a form asking the two
things only the operator knows: **how far back** the parts were bad — 30, 60 or 120 strokes — and
**what was wrong** with them, from the defects this process actually produces, plus a free-text note.
Which defect matters, because it decides whether a force signal could ever have carried it: a burr
belongs to the cutting station, a thin wall to ironing, an off-centre cup to the strip feed.

The window is anchored to the stroke on screen when the form opened, not when it was submitted — the
operator is reporting on what they had just seen, and a form left open while the press runs on would
otherwise slide off the parts they meant. Submitting records what the monitor was saying over that
same window. That pair is what a label-collection loop needs. The interesting report is the one where
every signal read normal and the parts did not.

It **does not retrain anything** — the models are fixed, and one report is not a training set. What
it demonstrates is the capture step, which is precisely what is missing from every dataset in this
project: it is why the intermediate wear state has no labels to train on in the first place.

## The data

`data/curves.npz` (8.9 MB, committed) holds everything the app needs. `data/models/*.pkl`
(~5.8 MB, also committed) are the trained models and calibration built from it -- pickled
objects, so they are tied to the library versions in `uv.lock`. A code change that alters what
gets cached bumps `CACHE_VERSION` in `artifacts.py`, which rebuilds them automatically; a bare
dependency upgrade does not, and can make an old `.pkl` fail to load -- delete `data/models/` or
run `spp2422-demo prepare --force` if that happens.

| | |
|---|---|
| 4 500 measured strokes | 9 production runs × 500 strokes, one run per T × A combination |
| 44 simulated curves | finite element exports: 11 deep drawing, 33 ironing |

Both are **normalized force on a shared event-time axis** — 500 samples, 0 at the onset of the
forming event, 1 at its end. Amplitudes are comparable *within* a dataset but never across the
two: the measured signals are uncalibrated volts and the simulated ones kN, and no conversion
between them was ever recorded.

Wear state is two independent three-level labels, **T1–T3** for deep drawing and **A1–A3** for
ironing, recorded with the trials as tool roughness classes. No numerical roughness value was
logged alongside them, so the levels are ordinal, not a measurement. The simulated curves carry
friction coefficients instead; the app maps their terciles onto the same three levels, which
orders them correctly by severity but is a proxy, not a calibration.

The extract is derived from the research repository that produced this data. To rebuild it:

```bash
git clone git@github.com:felixdivo/sheet-metal-synthetic.git context-material/sheet-metal-synthetic
cd context-material/sheet-metal-synthetic
./scripts/download.sh          # ~14 GB measured + 250 KB simulated
uv run python scripts/prepare.py
cd ../..
uv run python scripts/extract_data.py
```

`context-material/` is git-ignored; nothing in it is needed to run the demo.

### Strip misalignment — `data/excentricity.npz`

A separate measurement campaign, behind the board's Strip Alignment card. `data/excentricity.npz`
(1.2 MB, committed) holds 343 measured strokes: seven series of 49, one per tool infeed from
60.00 to 60.30 mm in 0.05 mm steps, resampled to 912 samples and divided by the median of the
per-stroke maxima. `force_scale` carries the kN the normalization divided out, so the card can
plot physical force. Only the axial punch force of the deep-drawing module
(`K2_Ch2_Mod2AI4`) is used; the other eight force channels and four accelerometers are
recorded but unmodelled. No simulated curves appear here.

Labels are **hundredths of a millimetre of overfeed** past the 60 mm reference, which is what
the source folder names encode. Three progressive stages precede deep drawing, so the offset
accumulates roughly threefold: 0.30 mm of overfeed puts the cup ~0.9 mm off-centre. The
dashboard converts for display; nothing shows a raw label.

Rebuilding it needs the raw capture in `_excentricity_data/real_numisheet/` (~926 MB,
git-ignored — ask for it, it is not public):

```bash
uv run python scripts/extract_excentricity_data.py
```

## Models and explanations

Four classifiers, all mapping one curve to a probability over the three levels:

| Model | Input |
|---|---|
| Logistic regression | handcrafted shape features — the transparent baseline |
| Random forest | the same features |
| 1-D CNN | the raw 500-sample curve |
| Hybrid CNN | both — the conv stack's embedding concatenated with the features below one head |

The shape features (peak height and position, percentile rise/fall durations, per-segment slope
and R² over rise/plateau/fall, shape moments, per-tenth variance, and for ironing the
contact-transition burst and the draw-down that follows it) are ported from the research code and
extended: 45 for deep drawing, 67 for ironing.

Explanations are time-resolved, so they can be read straight off the force curve: **integrated
gradients** for the CNN, **occlusion sensitivity** for the other three. Both are smoothed over 4% of
the stroke — the question is which part of the stroke mattered, not which single sample.

## What the accuracies mean

Measured on the actual data, not rounded up:

| | Deep drawing (T) | | | | Ironing (A) | | | |
|---|---|---|---|---|---|---|---|---|
| | LogReg | Forest | CNN | Hybrid | LogReg | Forest | CNN | Hybrid |
| **Held-out strokes** | 100.00% | 100.00% | 100.00% | 100.00% | 96.56% | 99.22% | 96.89% | 99.11% |

*Held-out strokes* trains on strokes 0–399 of every run and tests on the rest. That is the split
the deployed model uses, and every wear level appears in training — so it measures monitoring a
tool that has already been characterised, not recognising a state the model has never met.

Those are not the same question, and the second one is the one production actually asks.

### Strip misalignment

A different model on a different dataset: a random forest (20 trees, depth 4) reading two numbers
per stroke — the slope and height of a line fitted across the force plateau. Held out over ten
random 80/20 splits it reaches **35.3 ± 3.2 µm** mean absolute error in infeed terms, reproducing
Table 1 of Moske et al. (0.0352 ± 0.0021 mm).

That average has a tail, and the page says so. Out of fold, a single stroke lands on the exact
infeed level **44%** of the time and within one level 91% — the median error, 28 µm, is larger
than the 25 µm half-spacing that rounding to the nearest level would need. Averaging ten
consecutive strokes cuts the error to ~21 µm, which is why the alarm watches the running mean by
default: on one pass through all seven series it fires **once**, at the correct series, where
watching single strokes fires 19 times and first stops on a series that is inside tolerance.

The bigger caveat is structural. Each infeed level is one uninterrupted run of 49 strokes, so a
random split puts strokes sharing a tool temperature, a lubrication state and one setup on both
sides of it. Holding a whole run out is impossible — that removes its label from training
entirely. The number therefore measures recognising a stroke from a run already seen. Repeated
runs per infeed level are what an honest generalisation estimate would need.

## Locating the state nobody can label

A tool does not step from good to scrap. It crosses a threshold where parts are still in tolerance
but the surface is going, and that is the state worth catching. It is also the one state that
cannot be put in a training set: wear passes through it uncontrolled, and the press cannot be held
there long enough to collect labelled strokes.

So the **Wear Threshold** page withholds it. The intermediate level is taken out of training
entirely; what remains is a pristine tool, a heavily worn one, and the FE friction sweep that spans
the middle continuously — 11 simulated deep-drawing curves and 33 ironing ones, already in
`data/curves.npz`. A Gaussian process maps curve descriptors to the friction coefficient using the
sweep alone, a second GP corrects it against a handful of real endpoint strokes, and their sum
places any measured stroke on the friction axis. The withheld state should land midway between the
two anchors: **0 is the pristine anchor, 1 the worn one, 0.5 exactly centred.**

Two controls decide whether a placement means anything. *Shuffled sweep* refits the prior on
permuted friction labels — same curves, no physical ordering — which is what separates a real
result from "any prior interpolates between two anchors". *Real endpoints only* drops the sweep
altogether, isolating what the simulation itself contributes.

| stage | best configuration | withheld state placed at | shuffled sweep | real endpoints only | p |
|---|---|---|---|---|---|
| Deep drawing (T2) | whole run, 25 strokes/endpoint | **0.410** | 0.618 | 0.646 | 0.0041 |
| Ironing (A2) | first 20 strokes, 10 strokes/endpoint | **0.332** | 0.240 | 0.239 | <0.0001 |

Both beat their control, and in both cases the control sits almost exactly on top of the
simulation-free baseline — the sweep is what moves the estimate, not the two real anchors.

**Read it narrowly.** Neither stage lands *on* 0.5, most window and budget combinations do not
separate from the control at all, and a placement on a friction axis is not a wear label: it says
the withheld state falls between the anchors, not that a given stroke can be classified. Ironing
also needs a cut the deep-drawing signal does not — its trace superimposes the upstream
deep-drawing die's response on its own, so a descriptor can track ironing friction in simulation
while its real variation follows the upstream state instead. Only descriptors whose simulated and
measured behaviour agree on which axis they follow survive; 9 of 67 do.

Restricting to early strokes matters because the die beds in: as a block runs, the surface smooths
back towards the fresh state, so late strokes carry a wear label their force curve no longer
supports. `tests/test_pipeline.py` pins the ironing result, including its direction, so if a future
change breaks it the tests fail rather than the page quietly becoming untrue.

## Background

Each content pillar rests on a publication of the project: force-based process state
classification ([Schumann et al. 2026](https://doi.org/10.1007/s12666-026-03839-4)) and
simulation-driven modelling of strip misalignment
([Moske et al. 2025](https://doi.org/10.1088/1742-6596/3104/1/012058)). The **More details** page
lists these and ten more, grouped by what they contribute.
