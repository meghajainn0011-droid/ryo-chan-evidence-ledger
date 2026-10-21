"""Command line interface for the RYO-CHAN evidence ledger.

    python -m ryochan demo --out demo_ledger.jsonl
    python -m ryochan run --messages samples/signals.jsonl --out ledger.jsonl --prices prices.json
    python -m ryochan verify --ledger ledger.jsonl
    python -m ryochan trail --ledger ledger.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .core import (Ledger, PaperBook, RiskLimits, decide, detect_narratives,
                    interpret_message)

SAMPLE_PRICES = {"BTC": 64000.0, "ETH": 3100.0, "SOL": 145.0, "XRP": 0.62,
                 "ARB": 0.91, "DOGE": 0.14, "BNB": 580.0, "LINK": 14.2,
                 "OP": 1.85, "MSTR": 165.0, "SPY": 545.0, "QQQ": 480.0,
                 "AAPL": 228.0, "NVDA": 121.0, "TSLA": 245.0}


def _read_messages(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    text = Path(path).read_text()
    if path.suffix == ".jsonl" or path.suffix == ".ndjson":
        for line in text.splitlines():
            if line.strip():
                rows.append(json.loads(line))
    else:
        rows = json.loads(text)
    return rows


def _read_prices(path: str | None) -> dict[str, float]:
    if not path:
        return dict(SAMPLE_PRICES)
    data = json.loads(Path(path).read_text())
    return {k: float(v) for k, v in data.items()}


def run_agent(messages: list[dict], ledger_path: str | Path, limits: RiskLimits,
              prices: dict[str, float], now: float | None = None) -> dict:
    """Interpret messages, detect convergence, apply the risk limits, log paper trades."""
    ledger = Ledger(ledger_path)
    book = PaperBook()
    now = float(now if now is not None else time.time())

    signals = []
    for m in messages:
        sig = interpret_message(m)
        signals.append(sig)
        ledger.append("signal", sig.to_dict(), ts=sig.ts)

    narratives = detect_narratives(
        signals,
        min_voices=limits.min_voices,
        min_mean_conviction=limits.min_conviction,
    )
    for n in narratives:
        ledger.append("narrative", {
            "symbol": n.symbol, "voices": n.voices, "score": n.score,
            "converged": n.converged, "mean_conviction": n.mean_conviction,
            "mean_sentiment": n.mean_sentiment, "max_urgency": n.max_urgency,
            "causes": n.causes, "trail": n.reasoning_trail(),
        }, ts=now)

    trades = []
    for n in narratives:
        price = prices.get(n.symbol)
        if price is None:
            continue
        d = decide(n, limits, book, price, now)
        ledger.append("decision", {
            "action": d.action, "symbol": d.symbol, "side": d.side,
            "reason": d.reason, "blocked_by": d.blocked_by,
            "limits_checked": d.limits_checked, "notional": d.notional,
        }, ts=now)
        if d.action in ("open", "add"):
            book.positions[d.symbol] = book.positions.get(d.symbol, 0.0) + d.notional
            book.cash -= d.notional
            book.last_trade_ts[d.symbol] = now
            ledger.append("trade", {
                "symbol": d.symbol, "side": d.side, "notional": d.notional,
                "price": price, "book_cash": book.cash,
                "book_equity": book.equity(prices),
            }, ts=now)
            trades.append({"symbol": d.symbol, "side": d.side, "notional": d.notional,
                           "price": price})

    ok, why = ledger.verify()
    return {
        "signals": len(signals),
        "narratives": len(narratives),
        "converged": [n.symbol for n in narratives if n.converged],
        "trades": trades,
        "ledger_ok": ok,
        "ledger_reason": why,
        "book": {"cash": book.cash, "positions": book.positions},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ryochan", description="market evidence -> paper trades with an auditable trail")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the bundled sample messages end to end")
    d.add_argument("--out", default="demo_ledger.jsonl")

    r = sub.add_parser("run", help="run on a message file")
    r.add_argument("--messages", required=True)
    r.add_argument("--out", default="ledger.jsonl")
    r.add_argument("--prices", default=None)
    r.add_argument("--max-position-pct", type=float, default=5.0)
    r.add_argument("--max-open-positions", type=int, default=5)
    r.add_argument("--min-voice-count", type=int, default=2)
    r.add_argument("--min-conviction", type=float, default=0.4)

    v = sub.add_parser("verify", help="check the hash chain of a ledger")
    v.add_argument("--ledger", required=True)

    t = sub.add_parser("trail", help="print the reasoning trail of a ledger")
    t.add_argument("--ledger", required=True)

    args = ap.parse_args(argv)

    if args.cmd == "demo":
        here = Path(__file__).resolve().parents[1]
        messages = _read_messages(here / "samples" / "signals.jsonl")
        limits = RiskLimits()
        res = run_agent(messages, args.out, limits, dict(SAMPLE_PRICES), now=1760000000.0)
        print(json.dumps(res, indent=2))
        print()
        print(Ledger(args.out).trail())
        return 0

    if args.cmd == "run":
        messages = _read_messages(args.messages)
        limits = RiskLimits(max_position_pct=args.max_position_pct,
                            max_open_positions=args.max_open_positions,
                            min_voices=args.min_voice_count,
                            min_conviction=args.min_conviction)
        res = run_agent(messages, args.out, limits, _read_prices(args.prices))
        print(json.dumps(res, indent=2))
        return 0

    if args.cmd == "verify":
        ok, why = Ledger(args.ledger).verify()
        print(json.dumps({"ok": ok, "reason": why}))
        return 0 if ok else 1

    if args.cmd == "trail":
        print(Ledger(args.ledger).trail())
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
