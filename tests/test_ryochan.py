"""Tests for the RYO-CHAN evidence ledger."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ryochan.core import (Ledger, PaperBook, RiskLimits, decide, detect_narratives,
                          extract_cause, extract_symbol, interpret_message, sentiment_score)


def msg(i, voice, ts, text):
    return {"id": i, "voice": voice, "ts": ts, "text": text}


def test_symbol_extraction_prefers_explicit_ticker():
    assert extract_symbol("I like $ETH here") == "ETH"
    assert extract_symbol("bitcoin still works") == "BTC"
    assert extract_symbol("nothing relevant") is None


def test_sentiment_is_signed_and_bounded():
    assert sentiment_score("this is extremely bullish") > 0
    assert sentiment_score("a total crash and dump") < 0
    assert sentiment_score("neutral text") == 0.0


def test_negation_does_not_flip_into_a_signal():
    sig = interpret_message(msg("x", "v", 1, "I am not buying ETH, this is not a bullish setup"))
    assert sig.conviction == 0.0 or sig.direction != "long"


def test_cause_clause_is_captured():
    assert "because" in extract_cause("I am buying ETH because flows flipped")
    sig = interpret_message(msg("x", "v", 1, "I am buying ETH because the ETF inflows turned positive"))
    assert sig.cause and "ETF inflows" in sig.cause


def test_urgency_and_conviction_scoring():
    sig = interpret_message(msg("x", "v", 1, "Don't miss this SOL, I am buying with high conviction right now"))
    assert sig.conviction >= 0.8
    assert sig.urgency >= 0.7


def test_unknown_symbol_is_rejected_with_reason():
    sig = interpret_message(msg("x", "v", 1, "the weather is nice today"))
    assert sig.rejected
    assert sig.symbol is None


def test_narrative_detects_convergence_across_voices():
    sigs = [
        interpret_message(msg("1", "a", 100, "I am buying SOL with high conviction")),
        interpret_message(msg("2", "b", 200, "SOL breakout, going long, high conviction")),
        interpret_message(msg("3", "c", 300, "SOL is fine")),
    ]
    narratives = detect_narratives(sigs, min_voices=2, min_mean_conviction=0.4)
    sol = [n for n in narratives if n.symbol == "SOL"][0]
    assert sol.converged
    # a mention without sentiment or conviction is not an actionable voice
    assert set(sol.voices) == {"a", "b"}
    assert len(sol.signals) == 2
    assert sol.mean_conviction > 0.5


def test_narrative_does_not_converge_on_one_voice():
    sigs = [interpret_message(msg("1", "a", 100, "I am buying SOL with high conviction"))]
    narratives = detect_narratives(sigs, min_voices=2, min_mean_conviction=0.4)
    assert not narratives[0].converged


def test_explicit_stance_beats_lexicon_sentiment():
    # the writer says "everyone is bearish ... but I am buying": the stance is long even though
    # the sentiment lexicon sees the word bearish
    sig = interpret_message(msg("x", "v", 1, "Everyone is bearish on ETH but I am buying with high conviction"))
    assert sig.direction == "long"
    assert sig.sentiment <= 0.0


def test_side_prefers_stance_votes():
    n = type("N", (), {"converged": True, "mean_conviction": 0.9, "mean_sentiment": -0.4,
                       "max_urgency": 0.9, "voices": ["a", "b"], "symbol": "SOL",
                       "long_votes": 2, "short_votes": 0})()
    d = decide(n, RiskLimits(), PaperBook(), 100.0, 1000.0)
    assert d.side == "long"


def test_risk_limit_blocks_when_max_positions_reached():
    limits = RiskLimits(max_open_positions=1, min_voices=2, min_conviction=0.4)
    book = PaperBook(positions={"BTC": 1000.0})
    n = type("N", (), {"converged": True, "mean_conviction": 0.9, "mean_sentiment": 0.8,
                       "max_urgency": 0.9, "voices": ["a", "b"], "symbol": "SOL",
                       "long_votes": 1, "short_votes": 0})()
    d = decide(n, limits, book, 100.0, 1000.0)
    assert d.action == "skip"
    assert d.blocked_by == "max_open_positions"


def test_risk_limit_blocks_per_symbol_exposure():
    limits = RiskLimits(max_per_symbol_pct=1.0, max_position_pct=5.0)
    book = PaperBook(positions={"SOL": 5000.0}, cash=100_000.0)
    n = type("N", (), {"converged": True, "mean_conviction": 0.9, "mean_sentiment": 0.8,
                       "max_urgency": 0.9, "voices": ["a", "b"], "symbol": "SOL",
                       "long_votes": 1, "short_votes": 0})()
    d = decide(n, limits, book, 100.0, 1000.0)
    assert d.action == "skip"
    assert d.blocked_by == "max_per_symbol"


def test_cooldown_blocks_reentry():
    limits = RiskLimits(cooldown_s=3600)
    book = PaperBook(last_trade_ts={"SOL": 900.0})
    n = type("N", (), {"converged": True, "mean_conviction": 0.9, "mean_sentiment": 0.8,
                       "max_urgency": 0.9, "voices": ["a", "b"], "symbol": "SOL",
                       "long_votes": 1, "short_votes": 0})()
    d = decide(n, limits, book, 100.0, 1000.0)
    assert d.blocked_by == "cooldown"


def test_ledger_is_hash_chained_and_tamper_evident(tmp_path):
    path = tmp_path / "l.jsonl"
    led = Ledger(path)
    led.append("signal", {"voice": "a", "symbol": "ETH", "sentiment": 0.5, "conviction": 0.6,
                          "urgency": 0.2, "text": "hello"}, ts=1.0)
    led.append("trade", {"symbol": "ETH", "side": "long", "notional": 100.0, "price": 10.0}, ts=2.0)
    ok, why = led.verify()
    assert ok, why

    rows = [json.loads(x) for x in path.read_text().splitlines()]
    rows[0]["payload"]["text"] = "tampered"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ok2, why2 = Ledger(path).verify()
    assert not ok2
    assert "mismatch" in why2
