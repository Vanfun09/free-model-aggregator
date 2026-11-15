# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Vanfun09/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Vanfun09/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that answers OpenAI and Anthropic clients with the strongest free model on ten simulated
free-tier providers, failing over whenever a provider refuses.

## Run it

```sh
MAKEFUN_ENV_URL=<the env> MAKEFUN_PROMPT_FILE=<the brief> node main.ts
node main.ts --list   # print the catalog and exit without starting the server
```

## How it works

- **One router, three wires.** Every request is ranked against the brief's quality table: free models
  only, the capability flags and context length the request requires, strongest rating first, and the
  provider that has been asked for least as the tie-break.
- **Failover.** A 402, a 429 or a 5xx is not an answer: it is counted and the next candidate takes the
  request. A refusal that says the provider's budget is spent retires that provider for the trial; a
  refusal that says only this call was refused leaves it in the running.
- **Answers are normalised first.** The provider's completion becomes one internal shape (text, tool
  calls, usage, finish family) and is then emitted in the wire the client spoke — as JSON, or as a
  stream the entry synthesises, so usage and the stop reason are always carried.
- **Pass-through where the wires match.** An OpenAI client answered by an OpenAI provider gets the
  provider's own completion, untouched; an Anthropic client answered by an Anthropic provider likewise.
- **Deterministic.** Every route is a function of the requests seen and the answers received: no clock,
  no unseeded randomness, so a replay exports the same state.

## Files

| Path | What it holds |
|---|---|
| `main.ts` | the server: catalog probe, register, routing, failover ledger, both client routes |
| `lib/brief.ts` | the brief's three documents and the provider catalog probe |
| `lib/wires.ts` | the wire shapes a client is answered in, and Anthropic errors and model lists |
| `lib/convert.ts` | request and answer conversions between the three wires |

---

<sub>Entered on [make.fun](https://make.fun/?ref=vf09&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
