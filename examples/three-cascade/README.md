# Settings Cascade (Three.js proof-of-concept)

A 3D take on Lemonade's settings resolution: `Intent / Preset` and `Model Tuning`
as stacked glass layers, a light beam converging through a funnel into an
`Effective Load Settings` card, and a transition where the layers withdraw and
that card rotates upright and becomes the chat panel.

Not a shipping feature — see [Why this isn't in the app](#why-this-isnt-in-the-app).

## Running it

Needs Node (for `three` only) and a running Lemonade Server.

```bash
cd examples/three-cascade
npm install
npm run serve            # http://localhost:8788
```

The page resolves the server as, in order: `localStorage['lemonade.base']`, a
`?server=` query parameter, then `<page-host>:8000`. So if you serve this from
the same box as `lemond`, it just works; otherwise pass
`?server=http://192.168.0.200:8000`.

`LEMONADE_ALLOWED_ORIGINS` must permit the page's origin, since the prototype is
served from a different port than `lemond`.

## What is real and what is faked

Real:

- The `Model Tuning` tiles come from `GET /api/v1/models`, filtered to
  completion recipes (`llamacpp`, `ryzenai-llm`, `flm`, `vllm`). If the server is
  unreachable the page falls back to a sample list and shows a banner.
- The chat streams from `POST /api/v1/chat/completions` with `stream: true`,
  parsing SSE frames incrementally. Partial frames are expected mid-chunk and are
  swallowed until the rest of the frame arrives.
- The preset tiles mirror the real starter presets in `src/app/src/presetStore.ts`
  (`s-creative`, `s-code`, `s-long-context`, `s-thorough`), including their
  `temperature_hint` and `context_hint`.

Faked:

- The hint-to-number mapping (`precise` -> 0.4, `max` -> 131072) is a local table
  in `main.js`, **not** the real cascade. The server resolves this properly in
  `Router::resolve_effective_recipe_options`.
- Flash Attention and KV-Cache are hardcoded display values, not read from
  `RecipeOptions`.

To make the card honest, point it at `POST /api/v1/load/command`, which returns
the genuinely resolved effective options rather than re-deriving them client-side.

## Why this isn't in the app

The browser app (`src/web-app/`) cannot take this dependency. Per invariant 12 in
`AGENTS.md`, its `package.json` is constrained to npm modules Debian ships in
`/usr/share/nodejs` so the `lemonade-server` `.deb` builds reproducibly, and
Debian has no `three` package. Adding it would break distro packaging.

The Tauri desktop app (`src/app/`) has no such constraint, so this could live
there — at the cost of diverging the two frontends, which currently share a
renderer. Note also that `AGENTS.md` reserves UI/frontend changes for core
maintainers.

## Poking at it

`window.__lemon` exposes `{ state, effective, layers, camera }` for driving the
scene from the console — useful because tiles are picked by raycast, so there is
no DOM node to click.

Hover state is resolved in the render loop rather than in the pointer handler, so
a synthetic `pointerdown` dispatched in the same tick as its `pointermove` will
use the previous frame's hover target. Real pointer input is unaffected.
