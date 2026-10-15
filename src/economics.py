"""
Economic significance: is a lead big enough to make money once you pay to trade?

A statistically significant lead can still be worthless: if the follower only
"catches up" by less than the cost of trading, nobody can profit from it, and
that is exactly how an efficient market should look (Grossman & Stiglitz, 1980).

The trade a lead-lag arbitrageur would make
-------------------------------------------
When the LEADER makes a big move in bar t, buy the FOLLOWER in the same direction
and hedge with the leader (sell beta units of it). Close both legs h bars later.
The hedge removes moves of the whole market (momentum, reversals), so what is
left is only the follower catching up with the leader:

    catch-up(h) = sign(r_L,t) * [ (p_F,t+h - p_F,t) - beta * (p_L,t+h - p_L,t) ]    (log prices)

    beta = cov(r_F, r_L) / var(r_L)    (minimum-variance hedge, same slice of data)

Costs: crossing the bid-ask spread (assumed one tick, its minimum) on both legs,
plus fees (config.FEE_PER_SIDE). Everything is in basis points (1 bp = 0.01%)
of the follower's notional value, per trade.

Caveat: bar closes are LAST TRADE prices. After a big up-move the last trade is
usually at the ask, so part of the spread is already inside the gross number
(bid-ask bounce). That makes the gross catch-up a slightly LOW estimate; the
1-tick cost assumption makes the cost a LOW estimate too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import log_returns
from .leadlag import _shift


def round_trip_cost_bp(name: str, price) -> float:
    """Cost of buying and later selling one unit at the touch, in bp of the price:
    one full bid-ask spread (assumed one tick) plus a fee on each side."""
    return (config.TICK_SIZE[name] + 2 * config.FEE_PER_SIDE[name] / config.MULTIPLIER[name]) / price * 1e4


def big_move_catchup(prices: pd.DataFrame, leader: str, follower: str, q: float = 0.99,
                     horizons: tuple[int, ...] = (1, 5), freq: str = "1min") -> dict:
    """Average outcome of the hedged trade after the leader's biggest moves.

    prices : bar closes of both markets on one slice of data (e.g. one month).
    q      : a "big move" is a bar whose |leader return| is at or above its
             q-quantile in this slice (q = 0.99 -> the top 1% of bars).
    Returns, in bp per trade: the leader's move, the follower's move in the SAME
    bar, the follower's unhedged next-h move, the hedged catch-up, and the costs.
    """
    p = prices[[leader, follower]]
    lp = np.log(p)
    r = log_returns(p, max_gap=freq)
    ok = r.notna().all(axis=1)
    rl, rf = r.loc[ok, leader], r.loc[ok, follower]
    beta = rf.cov(rl) / rl.var()

    events = rl.index[rl.abs() >= rl.abs().quantile(q)]
    move = {h: {c: (_shift(lp[c], -h, freq) - lp[c]).loc[events] for c in (leader, follower)}
            for h in horizons}
    full = np.ones(len(events), dtype=bool)                   # keep events with every horizon
    for h in horizons:
        full &= move[h][leader].notna().to_numpy() & move[h][follower].notna().to_numpy()
    events = events[full]
    s = np.sign(rl.loc[events])

    out = {"n_events": len(events), "beta": beta,
           "leader_move_bp": 1e4 * (s * rl.loc[events]).mean(),
           "follower_same_bar_bp": 1e4 * (s * rf.loc[events]).mean()}
    for h in horizons:
        dF, dL = move[h][follower].loc[events], move[h][leader].loc[events]
        out[f"follower_next{h}_bp"] = 1e4 * (s * dF).mean()             # unhedged
        out[f"catchup_{h}_bp"] = 1e4 * (s * (dF - beta * dL)).mean()   # hedged
    cost_f = round_trip_cost_bp(follower, p[follower].loc[events].mean())
    cost_l = round_trip_cost_bp(leader, p[leader].loc[events].mean())
    out["cost_follower_bp"] = cost_f
    out["cost_hedged_bp"] = cost_f + abs(beta) * cost_l
    return out


def seconds_catchup(logp: pd.DataFrame, traded: pd.DataFrame, leader: str, follower: str,
                    q: float = 0.999, horizons=range(0, 31), beta: pd.Series | None = None) -> pd.DataFrame:
    """The hedged catch-up trade at 1-second speed.

    Events: seconds where BOTH markets traded (fresh prices) and the leader's
    1-second |return| is in its top (1 - q) of that day. For each event and
    horizon h (seconds), the hedged gain from the close of second t to t+h:
        sign(r_L,t) * [ (p_F,t+h - p_F,t) - beta * (p_L,t+h - p_L,t) ]     in bp
    beta: hedge ratio per day (index = day). By default the 1-MINUTE beta of the
    same day - 1-second betas are biased toward zero by non-synchronous trading
    (the Epps effect), so they would under-hedge.
    Returns one row per event: day, leader move, catch_<h> (hedged) and
    follower_<h> (follower only, no hedge) columns.

    Note: after its biggest seconds the leader partly REVERSES (bid-ask bounce,
    overshoot). The hedge is short the leader, so over longer holds it books that
    reversal as profit - nothing to do with the lead. At horizons of a few
    seconds the leader is flat and both versions agree."""
    r = logp.diff()
    r[(logp.index.to_series().diff() > pd.Timedelta("1s")).to_numpy()] = np.nan   # no return across nights
    fresh = traded[leader] & traded[follower] & r[[leader, follower]].notna().all(axis=1)
    rl = r.loc[fresh, leader]
    thr = rl.abs().groupby(rl.index.normalize()).transform(lambda s: s.quantile(q))
    events = rl.index[(rl.abs() >= thr).to_numpy()]
    s = np.sign(rl.loc[events]).to_numpy()
    if beta is None:
        minute_end = logp[logp.index.second == 59]
        r60 = minute_end.diff()
        r60[(minute_end.index.to_series().diff() > pd.Timedelta("1min")).to_numpy()] = np.nan
        g = r60.groupby(r60.index.normalize())
        beta = g.apply(lambda d: d[follower].cov(d[leader]) / d[leader].var())
    b = beta.reindex(events.normalize()).to_numpy()
    at = lambda c, h: logp[c].reindex(events + pd.Timedelta(seconds=int(h))).to_numpy()
    out = {"day": events.normalize(), "leader_move_bp": 1e4 * s * rl.loc[events].to_numpy()}
    for h in horizons:
        d_f, d_l = at(follower, h) - at(follower, 0), at(leader, h) - at(leader, 0)
        out[f"catch_{h}"] = 1e4 * s * (d_f - b * d_l)          # hedged
        out[f"follower_{h}"] = 1e4 * s * d_f                   # follower only (the cheapest trade)
    return pd.DataFrame(out, index=events)
