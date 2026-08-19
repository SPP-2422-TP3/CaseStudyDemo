# Running it beyond your machine

Deployment options and CI. See the [README](../README.md) for running it locally, and
[PROJECT.md](PROJECT.md) for what the app actually shows.

## Putting it on the web

The dashboard needs a Python server — every card, modal and stream tick is a server-side callback —
so **GitHub Pages cannot host it**. What it needs is somewhere to run the `Dockerfile`.

The image idles at about 270 MB with one worker and peaks near 330 MB while it computes an
attribution, which is what decides where it fits. `WEB_CONCURRENCY` sets the worker count: two by
default, one on a small instance.

### Free, no credit card — Render

`render.yaml` is a blueprint for [Render](https://render.com)'s free instance: 512 MB, 0.1 CPU, a
permanent public URL. Sign up, *New → Blueprint*, point it at this repository, deploy. Nothing to
configure — the blueprint pins one worker and Render supplies `PORT`.

The free instance **spins down after 15 minutes without traffic** and takes about a minute to wake,
and 0.1 CPU makes it noticeably slower than a laptop — opening an attribution takes seconds rather
than being instant. Fine for a link someone can look at; not what you want mid-talk.

### Paid, and the least work — Hugging Face Spaces

`.github/workflows/deploy.yml` pushes to a [Space](https://huggingface.co/docs/hub/spaces) on every
push to `main`. Note that **Docker Spaces are not free**: static Spaces are, and Gradio Spaces on
ZeroGPU are, but a Docker Space needs PRO for a personal account or Team for an organisation. In
exchange you get 2 vCPU and 16 GB, which this runs comfortably on. **Three steps, once:**

1. A Space at <https://huggingface.co/new-space> — **SDK: Docker**, visibility Public.
2. A **Write** token at <https://huggingface.co/settings/tokens>, added to this repository under
   *Settings → Secrets and variables → Actions* as the secret `HF_TOKEN`.
3. On the *Variables* tab beside it, `HF_SPACE` = `<owner>/<space-name>`.

Until both exist the workflow skips rather than fails, so the repository is not red meanwhile.

Either way the link is public and unauthenticated. There is nothing private in the demo, but anyone
who has the URL reaches it.

### Just for a talk

A [Cloudflare quick tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/)
puts a *locally* running server on a public URL with no account and no DNS:

```bash
curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared && chmod +x cloudflared
./cloudflared tunnel --url http://localhost:8050
```

It prints a `https://<random-words>.trycloudflare.com` address. The link is **unauthenticated** and
disappears when the process stops, so it suits a demo from your own laptop — no cold start, and
nothing to set up in advance.

## Continuous integration

`.github/workflows/ci.yml` runs ruff and the test suite on every push and pull request, then two
checks that tests alone would miss: that the committed `data/models/*.pkl` still load without being
silently retrained — a stale cache passes every test while leaving the repository wrong — and that
the dashboard actually answers on `:8050`.
