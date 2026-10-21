# RYO-CHAN Evidence Ledger

**Market-evidence interpreter and paper-trade agent with a tamper-evident reasoning trail.**
Built for the RYO-CHAN bounty (Autonomous Agents, Blockchain Legal Institute / BLI Legal Tech Hackathon 2).

The agent reads market-voice messages, decides whether a *narrative* is forming (several trusted
voices converging on the same token), applies transparent risk limits, records the **paper** trades
it would make, and keeps a **hash-chained ledger** of every step so the reasoning trail is
repeatable and provably untampered.

Everything runs offline and deterministically. There is no model call, no network access, no
credential, and no live trading anywhere in the code path.

---

## 1. What it does

```
message ──► signal ──► narrative ──► decision ──► paper trade
             │             │             │              │
             └─────────────┴─────────────┴──────────────┘
                    every step appended to a hash-chained ledger
```

**Signal** — one message interpreted into a symbol, sentiment, conviction, urgency, direction and
the cause clause the writer gave. Plain word lists, no black box:

| dimension | how it is derived |
|---|---|
| symbol | explicit `$TICKER` first, then an alias table (`bitcoin` → `BTC`, …) |
| sentiment | positive/negative lexicon, negated phrases (`not buying`) score zero |
| conviction | `high conviction`, `i am buying`, `allocating`, `doubling down`, … |
| urgency | `immediately`, `don't miss`, `front-run`, `time sensitive`, … |
| direction | explicit stance phrases (`I am buying` / `I am shorting`) |
| cause | the clause after `because`, `since`, `due to`, `triggered by`, … |

**Narrative** — signals grouped per symbol inside a time window. Convergence score =
`distinct voices × (0.5 + 0.5 × mean conviction) × (1 + 0.3 × max urgency)`. A narrative
*converged* when at least `min_voices` distinct voices carry a mean conviction above the floor.

**Decision** — the side comes from the explicit stance votes when they exist, otherwise from the
sentiment sign. Then every user risk limit is checked in a fixed order, and the reason for a skip
names the limit that blocked it.

**Risk limits** (all user-configurable): max position % of the book, max % in one symbol, max
open positions, max daily paper loss, minimum conviction, minimum distinct voices, and a per
symbol cooldown.

**Ledger** — an append-only JSONL file where each entry commits to the hash of the previous one:

```
hash_n = sha256(prev_hash | seq | ts | kind | payload)
```

`ryochan verify` re-computes the chain and reports the first mismatch, so a tampered or edited
entry is detectable. The trail can be replayed in order: `ryochan trail --ledger ledger.jsonl`.

---

## 2. Quick start

```bash
python -m pip install -e .          # no third-party dependencies
python -m pytest tests -q           # 14 tests
python -m ryochan demo               # runs the bundled sample messages end to end
```

Output of the demo (bundled sample messages, clearly labelled synthetic):

```json
{
  "signals": 20,
  "narratives": 4,
  "converged": ["SOL", "ETH"],
  "trades": [
    {"symbol": "SOL", "side": "long", "notional": 5000.0, "price": 145.0},
    {"symbol": "ETH", "side": "long", "notional": 5000.0, "price": 3100.0}
  ],
  "ledger_ok": true,
  "ledger_reason": "chain intact"
}
```

Run it on your own messages and prices:

```bash
python -m ryochan run --messages samples/signals.jsonl --out ledger.jsonl --prices prices.json
python -m ryochan verify --ledger ledger.jsonl
python -m ryochan trail --ledger ledger.jsonl
```

Message format (JSONL): `{"id": "m01", "voice": "alpha", "ts": 1759900000, "text": "..."}`.
Prices format: `{"BTC": 64000, "ETH": 3100}`.

---

## 3. Why this shape

* **Cause and effect, not just a score.** The ledger keeps the cause clause the writer gave, so a
  reviewer can read *why* the agent believed a narrative was forming, not just that it did.
* **Repeatable reasoning trail.** Every decision names the limits it passed and the limit that
  blocked it, in a fixed order, so two runs over the same inputs produce the same trail.
* **Bounded, auditable action.** The agent can only open positions inside the user's limits, and
  every refusal is logged with its reason instead of being silently dropped.
* **No hidden model.** The rules are word lists and arithmetic, so a compliance reviewer can read
  the whole decision procedure in one file.

## 4. Honest limitations

* The lexicon is English-only and tuned for short market-voice posts; it is not a sentiment model.
* Stance detection is phrase-based. A message that quotes the opposing view before stating its
  own ("everyone is bearish but I am buying") is handled by the direction votes, but longer
  discursive posts are only approximated.
* Message ingestion is a JSONL file. A live X / Discord / Telegram adapter is an interface point
  (`interpret_message` takes one dict), not a shipped connector; the sample corpus is synthetic.
* Prices come from a supplied map; there is no data vendor integration and no slippage model.

## 5. Layout

```
ryochan/core.py      models, extraction, convergence, risk limits, hash-chained ledger
ryochan/cli.py       demo / run / verify / trail
samples/signals.jsonl  20 synthetic market-voice messages from 10 voices
tests/               14 tests: extraction, negation, convergence, risk limits, tamper evidence
```

## 6. Licence

MIT.
