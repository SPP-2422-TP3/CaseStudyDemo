# SPP 2422 · Teilprojekt 3 — tool wear from forming force signals

A dashboard demo for an industrial colloquium, built on
[SPP 2422 Teilprojekt 3](https://www.ifu.uni-stuttgart.de/spp-2422/teilprojekte/teilprojekt-3/) —
*Optimierung des Wirkflächendesigns schnelllaufender Folgeverbundwerkzeuge unter Nutzung
maschineller Lernalgorithmen* (TU Darmstadt: PtU and the AI/ML Lab).

One press stroke of a progressive die drives three stations — shear cutting, **deep drawing** and
**ironing**. Each forming station carries a force sensor. The demo shows what those signals reveal
about the condition of the tools:

- **Deep drawing** and **Ironing** — pick a production run and a stroke, pick one of four
  models, and see the predicted wear state. A predicted critical state raises an alert, and the
  prediction can be opened up to show which part of the stroke the model actually read.
- **Product quality** — placeholder for the strip-misalignment work, contributed separately.
- **Help** — how to read the dashboard, and the papers behind it.

## Run it

On a machine with nothing but Docker installed:

```bash
git clone https://github.com/SPP-2422-TP3/CaseStudyDemo.git && cd CaseStudyDemo && docker compose up
```

Then open <http://localhost:8050>. In VS Code, *Reopen in Container* does the same through
`.devcontainer/`, and `uv run spp2422-demo` starts it. With [uv](https://docs.astral.sh/uv/)
already installed, that one command is the whole story.

The first start trains the models and caches them in `data/models/`; every later start is
instant. Force a rebuild with `uv run spp2422-demo prepare --force`.

## Show it on another machine

A [Cloudflare quick tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/)
puts the running server on a public URL without an account or any DNS setup:

```bash
curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared && chmod +x cloudflared
./cloudflared tunnel --url http://localhost:8050
```

It prints a `https://<random-words>.trycloudflare.com` address; open that from anywhere. The
link is **unauthenticated** — anyone who has it reaches the dashboard — and it disappears when
the process stops, so it suits a talk, not a deployment.

## The data

`data/curves.npz` (8.9 MB, committed) holds everything the app needs:

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

## Locating the state nobody can label

A tool does not step from good to scrap. It crosses a threshold where parts are still in tolerance
but the surface is going, and that is the state worth catching. It is also the one state that
cannot be put in a training set: wear passes through it uncontrolled, and the press cannot be held
there long enough to collect labelled strokes.

So the **Wear threshold** page withholds it. The intermediate level is taken out of training
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
([Moske et al. 2025](https://doi.org/10.1088/1742-6596/3104/1/012058)). The **Help** page lists
these and ten more, grouped by what they contribute.

## Layout

```
src/spp2422_demo/
  app.py             Dash shell: top bar, navigation, page container
  data.py            loads curves.npz, mu terciles, the train/test split
  features.py        handcrafted shape descriptors
  models.py          the four classifiers behind the held-out-stroke split
  calibration.py     GP calibration: placing the withheld wear state on the sweep
  explain.py         occlusion sensitivity and integrated gradients
  artifacts.py       trains, calibrates, caches to data/models/
  station_view.py    the wear page, shared by both forming stages
  pages/             one module per route, discovered by Dash
  components/        figures, cards, the alert
  assets/            stylesheet and the CAD animation of the die
scripts/extract_data.py       rebuilds data/curves.npz from the research pipeline
compose.yaml                  one-command start on a machine that only has Docker
```

Development: `uv run ruff check .`, `uv run ruff format .`, `uv run pytest`.

## Licence

MIT — © 2026 Felix Divo, Antonia Wüst, Jonas Moske and Markus Schumann.
