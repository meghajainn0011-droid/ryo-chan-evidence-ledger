"""RYO-CHAN evidence ledger: market-evidence interpreter and paper-trade agent.

The agent turns raw market voice messages into an auditable trail:

    message -> signal (token, sentiment, conviction, urgency, cause)
             -> narrative (voices converging on the same token)
             -> decision (transparent rules, user risk limits)
             -> paper trade (position) appended to a hash-chained ledger

Every step is deterministic and offline. No network calls, no live trading, no credentials.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# token aliasing: the agent only trades symbols it was given
# ---------------------------------------------------------------------------
DEFAULT_ALIASES: dict[str, str] = {
    "btc": "BTC", "bitcoin": "BTC", "xbt": "BTC",
    "eth": "ETH", "ethereum": "ETH",
    "sol": "SOL", "solana": "SOL",
    "xrp": "XRP", "ripple": "XRP",
    "arb": "ARB", "arbitrum": "ARB",
    "doge": "DOGE", "dogecoin": "DOGE",
    "bnb": "BNB", "binancecoin": "BNB",
    "link": "LINK", "chainlink": "LINK",
    "op": "OP", "optimism": "OP",
    "mstr": "MSTR", "microstrategy": "MSTR",
    "spy": "SPY", "qqq": "QQQ",
    "aapl": "AAPL", "nvda": "NVDA", "tsla": "TSLA",
}

# ---------------------------------------------------------------------------
# lexicons: plain, inspectable word lists. No model calls.
# ---------------------------------------------------------------------------
POSITIVE = {
    "bullish", "bull", "breaking", "breakout", "accumulating", "accumulation",
    "upside", "strong", "stronger", "soaring", "rally", "rallies", "demand",
    "inflows", "adoption", "upgrade", "approved", "partnership", "launch",
    "outperform", "undervalued", "greenlight", "moonshot",
}
NEGATIVE = {
    "bearish", "bear", "crash", "dump", "dumping", "downside", "weak", "weaker",
    "selling", "selloff", "outflows", "liquidation", "liquidated", "hack",
    "exploit", "downgrade", "lawsuit", "sec", "ban", "delisting", "risk",
    "overvalued", "bleed", "collapse", "scam",
}
CONVICTION = {
    "conviction": 0.8, "high conviction": 0.9, "strong conviction": 0.9,
    "i am buying": 0.8, "i'm buying": 0.8, "i bought": 0.7, "bought": 0.6,
    "adding": 0.6, "added": 0.6, "allocating": 0.7, "allocated": 0.7,
    "loading up": 0.8, "doubling down": 0.9, "all in": 1.0, "going long": 0.75,
    "long": 0.55, "shorting": 0.7, "i am short": 0.8, "sold": 0.6, "exit": 0.6,
    "trimmed": 0.6, "taking profit": 0.6,
}
URGENCY = {
    "now": 0.5, "immediately": 0.9, "right now": 0.7, "today": 0.5, "tonight": 0.6,
    "asap": 0.9, "don't miss": 0.9, "dont miss": 0.9, "front-run": 0.85,
    "front run": 0.85, "before it pumps": 0.9, "time sensitive": 0.85,
    "closing soon": 0.8, "last chance": 0.9, "urgent": 0.85, "quick": 0.6,
}
CAUSE_MARKERS = ("because", "since", "due to", "as a result of", "after", "if ",
                 "therefore", "so that", "triggered by", "driven by", "on the back of")
NEGATION = ("not ", "no ", "never ", "isn't ", "is not ", "aren't ", "won't ")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _contains_any(haystack: str, needles: Iterable[str]) -> bool:
    return any(n in haystack for n in needles)


@dataclass
class Signal:
    """One interpreted market message."""
    message_id: str
    voice: str
    ts: float
    text: str
    symbol: str | None = None
    sentiment: float = 0.0          # -1 .. 1
    conviction: float = 0.0         # 0 .. 1
    urgency: float = 0.0            # 0 .. 1
    cause: str | None = None        # the cause clause the writer gave
    direction: str = "none"         # long | short | none
    rejected: str | None = None     # why the message produced no signal

    def to_dict(self) -> dict:
        return asdict(self)


def extract_symbol(text: str, aliases: dict[str, str] | None = None) -> str | None:
    """Map a message to a tradable symbol using the alias table (explicit tickers win)."""
    aliases = aliases or DEFAULT_ALIASES
    for m in re.finditer(r"\$([A-Za-z]{1,6})\b", text):
        sym = m.group(1).upper()
        if sym in set(aliases.values()):
            return sym
    for t in _tokens(text):
        if t in aliases:
            return aliases[t]
    return None


def score_lexicon(text: str, lexicon: dict[str, float]) -> float:
    """Maximal phrase match; negated phrases score zero."""
    low = text.lower()
    best = 0.0
    for phrase, weight in lexicon.items():
        idx = low.find(phrase)
        while idx != -1:
            prefix = low[max(0, idx - 12):idx]
            if not any(n in prefix for n in NEGATION):
                best = max(best, weight)
            idx = low.find(phrase, idx + 1)
    return best


def sentiment_score(text: str) -> float:
    pos = score_lexicon(text, {w: 1.0 for w in POSITIVE})
    neg = score_lexicon(text, {w: 1.0 for w in NEGATIVE})
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / max(pos, neg)


def extract_cause(text: str) -> str | None:
    low = text.lower()
    for marker in CAUSE_MARKERS:
        idx = low.find(marker)
        if idx != -1:
            return text[idx:].strip()[:240]
    return None


def direction_of(text: str) -> str:
    low = text.lower()
    short = any(p in low for p in ("i am short", "i'm short", "shorting", "going short"))
    if short:
        return "short"
    long_words = ("i am buying", "i'm buying", "i bought", "loading up", "going long",
                  "all in", "doubling down", "adding", "accumulating")
    if any(p in low for p in long_words):
        return "long"
    return "none"


def interpret_message(message: dict, aliases: dict[str, str] | None = None) -> Signal:
    """Turn one raw message into a Signal. Pure function, no I/O."""
    text = str(message.get("text", ""))
    sig = Signal(
        message_id=str(message.get("id", "")),
        voice=str(message.get("voice", "unknown")),
        ts=float(message.get("ts", 0.0)),
        text=text,
    )
    sig.symbol = extract_symbol(text, aliases)
    if sig.symbol is None:
        sig.rejected = "no known symbol mentioned"
        return sig
    sig.sentiment = sentiment_score(text)
    sig.conviction = score_lexicon(text, CONVICTION)
    sig.urgency = score_lexicon(text, URGENCY)
    sig.cause = extract_cause(text)
    sig.direction = direction_of(text)
    if sig.sentiment == 0.0 and sig.conviction == 0.0:
        sig.rejected = "no sentiment or conviction signal"
    return sig


# ---------------------------------------------------------------------------
# narrative convergence
# ---------------------------------------------------------------------------
@dataclass
class Narrative:
    symbol: str
    voices: list[str] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    mean_conviction: float = 0.0
    mean_sentiment: float = 0.0
    max_urgency: float = 0.0
    long_votes: int = 0
    short_votes: int = 0
    causes: list[str] = field(default_factory=list)
    score: float = 0.0
    converged: bool = False

    def reasoning_trail(self) -> list[str]:
        lines = [
            f"symbol {self.symbol}: {len(self.voices)} distinct voices, "
            f"mean conviction {self.mean_conviction:.2f}, mean sentiment {self.mean_sentiment:+.2f}, "
            f"max urgency {self.max_urgency:.2f}, convergence score {self.score:.3f}"
        ]
        for s in self.signals:
            lines.append(
                f"  - {s.voice} @{s.ts:.0f} conv={s.conviction:.2f} sent={s.sentiment:+.2f} "
                f"urg={s.urgency:.2f} :: {s.text[:120]}"
                + (f" | cause: {s.cause[:80]}" if s.cause else "")
            )
        for c in self.causes:
            lines.append(f"  cause: {c[:160]}")
        return lines


def detect_narratives(signals: Sequence[Signal], min_voices: int = 2,
                      min_mean_conviction: float = 0.4,
                      window_s: float = 6 * 3600) -> list[Narrative]:
    """Group signals by symbol and score narrative convergence inside a time window.

    Convergence score: distinct voices in window, weighted by conviction and urgency.
    """
    by_symbol: dict[str, list[Signal]] = {}
    for s in signals:
        if s.symbol and not s.rejected:
            by_symbol.setdefault(s.symbol, []).append(s)
    out: list[Narrative] = []
    for symbol, group in by_symbol.items():
        group = sorted(group, key=lambda s: s.ts)
        # the narrative covers every signal of that symbol inside the window: the story is the
        # whole set of voices, not the best-scoring subset of it
        window = [s for s in group if s.ts - group[0].ts <= window_s]
        voices = sorted({w.voice for w in window})
        mc = sum(w.conviction for w in window) / len(window)
        ms = sum(w.sentiment for w in window) / len(window)
        mu = max(w.urgency for w in window)
        score = len(voices) * (0.5 + 0.5 * mc) * (1.0 + 0.3 * mu)
        best = Narrative(symbol=symbol, voices=voices, signals=list(window),
                         mean_conviction=mc, mean_sentiment=ms, max_urgency=mu,
                         long_votes=sum(1 for w in window if w.direction == "long"),
                         short_votes=sum(1 for w in window if w.direction == "short"),
                         causes=[w.cause for w in window if w.cause], score=score)
        best.converged = len(best.voices) >= min_voices and best.mean_conviction >= min_mean_conviction
        out.append(best)
    return sorted(out, key=lambda n: -n.score)


# ---------------------------------------------------------------------------
# risk limits
# ---------------------------------------------------------------------------
@dataclass
class RiskLimits:
    max_position_pct: float = 5.0        # percent of the paper book per position
    max_per_symbol_pct: float = 8.0       # percent of the book in one symbol
    max_open_positions: int = 5
    max_daily_loss_pct: float = 3.0       # paper drawdown per day that blocks new trades
    min_conviction: float = 0.4
    min_voices: int = 2
    cooldown_s: float = 3600              # no re-entry on the same symbol within this window
    cooldown_enabled: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PaperBook:
    cash: float = 100_000.0
    start_equity: float = 100_000.0
    positions: dict[str, float] = field(default_factory=dict)   # symbol -> signed notional
    last_trade_ts: dict[str, float] = field(default_factory=dict)
    day_pnl: dict[str, float] = field(default_factory=dict)     # YYYY-MM-DD -> pnl

    def equity(self, prices: dict[str, float] | None = None) -> float:
        prices = prices or {}
        pos = sum(self.positions.values())
        return self.cash + pos


@dataclass
class TradeDecision:
    action: str                 # open | add | close | skip
    symbol: str | None = None
    side: str | None = None
    notional: float = 0.0
    price: float = 0.0
    reason: str = ""
    limits_checked: list[str] = field(default_factory=list)
    blocked_by: str | None = None


def decide(narrative: Narrative, limits: RiskLimits, book: PaperBook,
           price: float, now: float) -> TradeDecision:
    """Transparent decision rules. Returns a decision with the reason spelled out."""
    checks: list[str] = []
    if not narrative.converged:
        return TradeDecision(action="skip", reason="narrative did not converge", limits_checked=checks,
                              blocked_by="convergence")
    checks.append("converged")
    if narrative.mean_conviction < limits.min_conviction:
        return TradeDecision(action="skip", reason="conviction below floor", limits_checked=checks,
                              blocked_by="conviction")
    checks.append("conviction floor")
    if len(narrative.voices) < limits.min_voices:
        return TradeDecision(action="skip", reason="not enough distinct voices", limits_checked=checks,
                              blocked_by="voices")
    checks.append("voice count")
    # an explicit stance ("I am buying", "I am shorting") outranks the lexicon sentiment,
    # because a message can talk about the opposite view before stating its own
    if narrative.long_votes or narrative.short_votes:
        side = "long" if narrative.long_votes >= narrative.short_votes else "short"
    else:
        side = "long" if narrative.mean_sentiment >= 0 else "short"
    if side == "short" and narrative.max_urgency < 0.5:
        return TradeDecision(action="skip", reason="short without urgency signal", limits_checked=checks,
                              blocked_by="urgency")
    checks.append("direction")
    last = book.last_trade_ts.get(narrative.symbol)
    if limits.cooldown_enabled and last is not None and now - last < limits.cooldown_s:
        return TradeDecision(action="skip", reason="cooldown active for symbol", limits_checked=checks,
                              blocked_by="cooldown")
    checks.append("cooldown")
    if len(book.positions) >= limits.max_open_positions and narrative.symbol not in book.positions:
        return TradeDecision(action="skip", reason="max open positions reached", limits_checked=checks,
                              blocked_by="max_open_positions")
    checks.append("max open positions")
    exposure = abs(book.positions.get(narrative.symbol, 0.0))
    if exposure >= book.equity() * limits.max_per_symbol_pct / 100.0:
        return TradeDecision(action="skip", reason="per-symbol exposure limit reached", limits_checked=checks,
                              blocked_by="max_per_symbol")
    checks.append("per-symbol exposure")
    day = time.strftime("%Y-%m-%d", time.gmtime(now))
    if abs(book.day_pnl.get(day, 0.0)) >= book.start_equity * limits.max_daily_loss_pct / 100.0:
        return TradeDecision(action="skip", reason="daily loss limit reached", limits_checked=checks,
                              blocked_by="daily_loss")
    checks.append("daily loss")
    size = book.equity() * limits.max_position_pct / 100.0
    signed = size if side == "long" else -size
    return TradeDecision(action="open" if narrative.symbol not in book.positions else "add",
                          symbol=narrative.symbol, side=side, notional=signed, price=price,
                          reason="all risk limits passed", limits_checked=checks)


# ---------------------------------------------------------------------------
# hash-chained paper-trade ledger
# ---------------------------------------------------------------------------
@dataclass
class LedgerEntry:
    seq: int
    ts: float
    kind: str                  # signal | narrative | decision | trade
    payload: dict
    prev_hash: str
    hash: str


class Ledger:
    """Append-only JSONL ledger where every entry commits to the previous one."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.entries: list[LedgerEntry] = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self.entries.append(LedgerEntry(**json.loads(line)))

    @staticmethod
    def _hash(prev_hash: str, seq: int, ts: float, kind: str, payload: dict) -> str:
        blob = json.dumps({"prev": prev_hash, "seq": seq, "ts": ts, "kind": kind, "payload": payload},
                          sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def append(self, kind: str, payload: dict, ts: float | None = None) -> LedgerEntry:
        ts = float(ts if ts is not None else time.time())
        prev = self.entries[-1].hash if self.entries else "0" * 64
        seq = len(self.entries)
        h = self._hash(prev, seq, ts, kind, payload)
        entry = LedgerEntry(seq=seq, ts=ts, kind=kind, payload=payload, prev_hash=prev, hash=h)
        self.entries.append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(entry.__dict__, sort_keys=True, separators=(",", ":")) + "\n")
        return entry

    def verify(self) -> tuple[bool, str]:
        prev = "0" * 64
        for e in self.entries:
            if e.prev_hash != prev:
                return False, f"entry {e.seq} prev_hash mismatch"
            if self._hash(e.prev_hash, e.seq, e.ts, e.kind, e.payload) != e.hash:
                return False, f"entry {e.seq} hash mismatch"
            prev = e.hash
        return True, "chain intact"

    def trail(self, kinds: Sequence[str] = ("signal", "narrative", "decision", "trade")) -> str:
        out: list[str] = []
        for e in self.entries:
            if e.kind not in kinds:
                continue
            p = e.payload
            if e.kind == "signal":
                out.append(f"[{e.seq}] signal {p.get('voice')} {p.get('symbol')} "
                           f"sent={p.get('sentiment'):+.2f} conv={p.get('conviction'):.2f} "
                           f"urg={p.get('urgency'):.2f} :: {str(p.get('text'))[:100]}")
            elif e.kind == "narrative":
                out.append(f"[{e.seq}] narrative {p.get('symbol')} voices={p.get('voices')} "
                           f"score={p.get('score'):.3f} converged={p.get('converged')}")
                for line in p.get("causes", []):
                    out.append(f"        cause: {str(line)[:140]}")
            elif e.kind == "decision":
                out.append(f"[{e.seq}] decision {p.get('action')} {p.get('symbol') or ''} "
                           f"{p.get('reason')} blocked_by={p.get('blocked_by')}")
            else:
                out.append(f"[{e.seq}] trade {p.get('side')} {p.get('symbol')} "
                           f"notional={p.get('notional'):.2f} @ {p.get('price')}")
        return "\n".join(out)
