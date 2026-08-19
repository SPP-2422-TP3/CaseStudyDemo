# SPP 2422 · Teilprojekt 3 — tool wear from forming force signals

[![CI](https://github.com/SPP-2422-TP3/CaseStudyDemo/actions/workflows/ci.yml/badge.svg)](https://github.com/SPP-2422-TP3/CaseStudyDemo/actions/workflows/ci.yml)

A dashboard demo for an industrial colloquium, built on
[SPP 2422 Teilprojekt 3](https://www.ifu.uni-stuttgart.de/spp-2422/teilprojekte/teilprojekt-3/) —
*Optimierung des Wirkflächendesigns schnelllaufender Folgeverbundwerkzeuge unter Nutzung
maschineller Lernalgorithmen* (TU Darmstadt: PtU and the AI/ML Lab).

One press stroke of a progressive die drives three stations — shear cutting, **deep drawing** and
**ironing**. Each forming station carries a force sensor. The demo shows what those signals reveal
about the condition of the tools.

The landing page is the **Status** board, with **More details** — what the data is, what each
model is worth on it, how to read the dashboard, and the papers behind it — one click away. The
pages that take one model apart at a time sit behind that page and the **Model detail** menu:

- **Deep Drawing** and **Ironing** — pick a production run and a stroke, pick one of four
  models, and see the predicted wear state. A predicted critical state raises an alert, and the
  prediction can be opened up to show which part of the stroke the model actually read.
- **Wear Threshold** — the state nobody can label, located by anchoring on the simulated sweep.
- **Excentricity** — step through measured strokes of the deep-drawing station and read how
  far off-centre the strip was fed, from the slope of the force plateau alone. Crossing the
  alarm limit stops the stream and raises a warning.

See [docs/project.md](docs/project.md) for the data, the models, what the accuracies mean and the
research behind it, [docs/operations.md](docs/operations.md) for deploying it and CI, and
[docs/styleguide.md](docs/styleguide.md) for writing conventions.

## Run it

On a machine with nothing but Docker installed:

```bash
git clone https://github.com/SPP-2422-TP3/CaseStudyDemo.git && cd CaseStudyDemo && docker compose up
```

Then open <http://localhost:8050>. In VS Code, *Reopen in Container* does the same through
`.devcontainer/`, and `uv run spp2422-demo` starts it. With [uv](https://docs.astral.sh/uv/)
already installed, that one command is the whole story.

`compose.yaml` mounts the source and reloads with it, which is what you want while working on it.
`Dockerfile` builds the self-contained image that gets deployed, serving through gunicorn rather
than Flask's development server — `docker build -t spp2422-demo . && docker run -p 8050:8050
spp2422-demo`.

Trained models and the centre-state calibration are cached in `data/models/` and committed, so
even the first start is instant. Force a rebuild with `uv run spp2422-demo prepare --force`.

## Layout

```
src/spp2422_demo/
  app.py                  Dash shell: top bar, navigation, page container
  data.py                 loads curves.npz, mu terciles, the train/test split
  features.py             handcrafted shape descriptors
  models.py               the four classifiers behind the held-out-stroke split
  calibration.py          GP calibration: placing the withheld wear state on the sweep
  explain.py              occlusion sensitivity and integrated gradients
  artifacts.py            trains, calibrates, caches to data/models/
  station_view.py         the wear page, shared by both forming stages
  pages/                  one module per route, discovered by Dash
  components/             figures, cards, the alert
  assets/                 stylesheet and the CAD animation of the die
scripts/extract_data.py   rebuilds data/curves.npz from the research pipeline
compose.yaml              one-command start on a machine that only has Docker
docs/
  project.md              the data, models, evaluation and background
  operations.md           deployment options and CI
  styleguide.md           writing conventions for this repo's docs
```

Development: `uv run ruff check .`, `uv run ruff format .`, `uv run pytest`.

## Licence

MIT, see [LICENSE](LICENSE) — © 2026 Felix Divo, Antonia Wüst, Jonas Moske and Markus Schumann.
