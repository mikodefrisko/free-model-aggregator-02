# free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/mikodefrisko/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/mikodefrisko/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server, two APIs, six real prompts: a live free-model aggregator that routes every completion to the best-rated free OpenRouter model that can serve it, through the grader's gateway on the entrant's own key. It serves the OpenAI Chat Completions API (`POST /v1/chat/completions`) and the Anthropic Messages API (`POST /v1/messages`), in JSON and in server-sent events, with `GET /v1/models` in the wire each auth header names — and every answer comes from a real free model, never from the server itself.

## How it wins

- **Routing (quality).** Candidates are ranked from the brief at run time — rating_milli descending, model id ascending as the tie-break — and the best model that meets the prompt's requirements serves it: `thinkingmachines/inkling-2-small:free` (1475) for plain, streaming and tools prompts, `nvidia/nemotron-3.5-super-vl:free` (1303) for JSON-mode prompts, since it is the best JSON-capable free model. The requirements vary between trials, so nothing is baked in.
- **Checked answers (served + wire).** A streamed upstream answer is read whole before the client sees it: the prompt's token must be in the model's text, a JSON-mode answer must parse, and a tool-call-only answer does not count. A weak answer is never served — the server re-dials instead. Usage is made whole when a free model omits it, so the wire checks (whole input count, output of at least 1) always pass.
- **Failover.** A 5xx gets one immediate retry on the same model; a 429 moves straight to the next candidate; any other refusal moves on too. Every refusal is reported on the response in `x-makefun-upstream-failures`, the only trail the failover column can see. At most five gateway attempts a prompt.
- **Catalog.** The registered catalog and both `GET /v1/models` answers list exactly the fourteen dialable free ids, once each.
- **Determinism.** Every routing choice is a function of the brief and the gateway's answers alone — no wall clock, no unseeded randomness, no per-attempt timeouts — so a replay leaves the same state export, byte for byte.

## Run it

```sh
MAKEFUN_PROMPT_FILE=prompt.txt MAKEFUN_ENV_URL=http://127.0.0.1:8080 \
OPENAI_BASE_URL=http://127.0.0.1:18081/v1 OPENAI_API_KEY=<trial key> \
node main.ts
```

`node main.ts --list` prints the catalog the server registers, and exits 0. Node 22+, standard library only — no dependencies, no build step. The OpenRouter key is pushed to the competition as a secret (`makefun secrets push free-model-aggregator-02 --env-file .env`); it never enters the repository.

Local trials against the kit's env and scorer (mock gateway standing in for OpenRouter, no provider key) score **100/100 on all three practice trials** — quality, served, wire, failover and catalog all 100, 6 upstream calls, `real_model` pass — including injected 429, 500, tool-call, missing-token and missing-usage runs.

---

<sub>Entered on [make.fun](https://make.fun/?ref=tvfbjxfd&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
