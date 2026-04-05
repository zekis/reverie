"""Market prediction experiment on BTCUSDT 5m candles.

Does Reverie's answer margin carry information on noisy financial
time-series? Encode candles as sensorimotor-style scenes (features,
outcome), load into a brain, query the test set, bucket predictions by
margin, and measure hit rate per bucket.

If high-margin predictions hit above chance, geometric memory is
finding something in the noise. If they don't, this architecture has
hit a wall on this kind of data.

Fetches klines from Binance REST (no API key needed), caches locally.
"""

import os
import sys
import json
import time
import urllib.request
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brain import Brain


# ---- Config ----

SYMBOL = "BTCUSDT"
INTERVAL = "1d"
DAYS = 1460             # ~4 years of daily candles
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "memory", f"klines_{SYMBOL}_{INTERVAL}_{DAYS}d.npz")

LOOKBACK = 30           # candles used to compute each scene's features
HORIZON = 5             # predict direction at +5 candles (1 trading week)
USE_TERTILES = True     # tertile-based balanced outcomes (vs fixed threshold)
THRESHOLD = 0.03        # ±3% return threshold (used only if not tertiles)
TRAIN_FRAC = 0.70       # train/test split (time-ordered)

DIRECTIONS = ["down", "flat", "up"]  # class order
N_CLASSES = 3

# Online Hebbian: during training, adjust neighbour weights based on
# whether their outcome predicts the new scene's outcome correctly.
ONLINE_HEBBIAN = True
HEBBIAN_BOOST = 1.15    # neighbour weight multiplier when outcome matches
HEBBIAN_DECAY = 0.85    # neighbour weight multiplier when outcome mismatches
HEBBIAN_MAX_W = 3.0
HEBBIAN_MIN_W = 0.05
REBUILD_EVERY = 500     # rebuild expert matrices every N training scenes


# ---- Binance data pull ----

def fetch_klines(symbol, interval, days):
    """Pull N days of klines from Binance REST, paginated."""
    print(f"  fetching {days}d of {symbol} {interval} from Binance...")
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000
    all_rows = []
    cursor = start_ms
    while cursor < end_ms:
        url = (f"https://api.binance.com/api/v3/klines?"
               f"symbol={symbol}&interval={interval}"
               f"&startTime={cursor}&limit=1000")
        req = urllib.request.Request(url,
                                     headers={"User-Agent": "reverie-expt/0.1"})
        with urllib.request.urlopen(req, timeout=30) as r:
            batch = json.loads(r.read())
        if not batch:
            break
        all_rows.extend(batch)
        cursor = batch[-1][6] + 1  # close_time + 1ms
        if len(batch) < 1000:
            break
    # Columns we care about: open_time, open, high, low, close, volume
    data = np.array(
        [[row[0], float(row[1]), float(row[2]), float(row[3]),
          float(row[4]), float(row[5])] for row in all_rows],
        dtype=np.float64)
    print(f"  got {len(data)} candles")
    return data


def load_or_fetch():
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached klines: {CACHE_PATH}")
        data = np.load(CACHE_PATH)["klines"]
        print(f"  {len(data)} candles")
        return data
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    data = fetch_klines(SYMBOL, INTERVAL, DAYS)
    np.savez_compressed(CACHE_PATH, klines=data)
    return data


# ---- Feature engineering ----

def compute_features(klines):
    """For each candle, compute a 12-d feature vector using the LOOKBACK
    window ending at that candle. Features are stationary (returns,
    z-scores, ratios) — no raw prices enter the scene vectors."""
    opens = klines[:, 1]
    highs = klines[:, 2]
    lows = klines[:, 3]
    closes = klines[:, 4]
    volumes = klines[:, 5]
    n = len(klines)
    feats = np.zeros((n, 12), dtype=np.float32)

    log_ret = np.concatenate([[0.0], np.diff(np.log(closes))])

    for t in range(LOOKBACK, n):
        w_close = closes[t - LOOKBACK:t + 1]
        w_high = highs[t - LOOKBACK:t + 1]
        w_low = lows[t - LOOKBACK:t + 1]
        w_vol = volumes[t - LOOKBACK:t + 1]
        w_ret = log_ret[t - LOOKBACK:t + 1]

        sigma = w_ret.std() + 1e-8

        # Returns at multiple horizons (already stationary; normalize by σ)
        r1 = log_ret[t] / sigma
        r5 = (np.log(closes[t]) - np.log(closes[t - 5])) / (sigma * np.sqrt(5))
        r20 = (np.log(closes[t]) - np.log(closes[t - 20])) / (sigma * np.sqrt(20))

        # Volume z-score
        vol_z = (w_vol[-1] - w_vol.mean()) / (w_vol.std() + 1e-8)

        # Volatility (already sigma, but normalize across history)
        vol_ratio = sigma / (np.abs(w_ret[:-1]).mean() + 1e-8) - 1.0

        # Price position within window range
        rng = w_close.max() - w_close.min() + 1e-8
        pos = (w_close[-1] - w_close.min()) / rng

        # MA ratios
        ma5 = w_close[-5:].mean()
        ma20 = w_close.mean()
        ma_ratio = (ma5 / ma20) - 1.0

        # Trend slopes (normalized by sigma * len)
        slope5 = (w_close[-1] - w_close[-5]) / (w_close[-5] + 1e-8)
        slope20 = (w_close[-1] - w_close[0]) / (w_close[0] + 1e-8)

        # RSI-ish: up-days fraction
        ups = (np.diff(w_close) > 0).sum()
        rsi = ups / (LOOKBACK - 1) - 0.5  # centred

        # Range size z
        tr = (w_high - w_low)
        range_z = (tr[-1] - tr.mean()) / (tr.std() + 1e-8)

        # Upper-wick ratio
        body = abs(closes[t] - opens[t])
        upper = highs[t] - max(closes[t], opens[t])
        wick = upper / (tr[-1] + 1e-8)

        feats[t] = np.clip([
            r1, r5, r20, vol_z, vol_ratio, pos - 0.5,
            ma_ratio * 100, slope5 * 100, slope20 * 100,
            rsi, range_z, wick - 0.5,
        ], -5.0, 5.0)

    return feats


def compute_returns(klines):
    closes = klines[:, 4]
    n = len(klines)
    returns = np.full(n, np.nan, dtype=np.float64)
    for t in range(n - HORIZON):
        returns[t] = (closes[t + HORIZON] - closes[t]) / closes[t]
    return returns


def compute_outcomes(returns, low_thresh, high_thresh):
    """Outcome = 0/1/2 based on return < low, mid, > high. Thresholds
    are fitted on train data only and passed in — no test leakage."""
    n = len(returns)
    outcomes = np.full(n, -1, dtype=np.int32)
    for t in range(n):
        if np.isnan(returns[t]):
            continue
        r = returns[t]
        if r < low_thresh:
            outcomes[t] = 0   # down
        elif r > high_thresh:
            outcomes[t] = 2   # up
        else:
            outcomes[t] = 1   # flat
    return outcomes


def outcome_vec(cls: int) -> np.ndarray:
    v = np.zeros(N_CLASSES, dtype=np.float32)
    v[cls] = 1.0
    return v


# ---- Scene building ----

def regime_keys(features):
    """Partition scenes by coarse market regime. Each scene lands in
    multiple overlapping buckets — combined, they give the reconciler
    multiple voters at query time, which is what produces a meaningful
    margin across experts."""
    # Feature indices (from compute_features): 4=vol_ratio, 5=pos-0.5,
    # 7=slope5*100, 8=slope20*100, 9=rsi-centred.
    keys = ["market"]
    keys.append("vol_high" if features[4] > 0.0 else "vol_low")
    if features[7] > 0.3:
        keys.append("trend_up")
    elif features[7] < -0.3:
        keys.append("trend_down")
    else:
        keys.append("trend_flat")
    if features[8] > 0.5:
        keys.append("macro_up")
    elif features[8] < -0.5:
        keys.append("macro_down")
    else:
        keys.append("macro_flat")
    keys.append("pos_high" if features[5] > 0.0 else "pos_low")
    return keys


def build_scene(features, cls, t):
    return {
        "source_text": f"t={t} {DIRECTIONS[cls]}",
        "slots": [
            {"role": "features", "text": "",
             "vec": features.tobytes(), "is_gap": False},
            {"role": "outcome", "text": DIRECTIONS[cls],
             "vec": outcome_vec(cls).tobytes(), "is_gap": False},
        ],
        "vec": features.tobytes(),
        "keys": regime_keys(features),
    }


def build_gap(features):
    return {
        "query_vec": features.astype(np.float32),
        "subject_vec": np.zeros(len(features), dtype=np.float32),
        "role": "outcome",
    }, regime_keys(features)


def online_hebbian_update(brain, features, true_cls, keys):
    """For each relevant expert, find the current nearest-neighbour scene
    by feature-cosine. If that neighbour's outcome slot matches the new
    scene's true outcome, boost it; otherwise decay it. This compresses
    the pool toward scenes whose local geometry is predictive."""
    feat = features.astype(np.float32)
    fn = feat / (np.linalg.norm(feat) + 1e-8)
    for key in keys:
        if key not in brain.loader.warm:
            continue
        exp = brain.loader.warm[key]
        if exp.scene_matrix is None or len(exp.scene_vecs) == 0:
            continue
        # Direct matrix-multiply — sidesteps oscillator/reconciler for speed
        scores = exp.scene_matrix @ fn
        top_idx = int(np.argmax(scores))
        slots = exp.scene_slots[top_idx]
        outcome_slot = next(
            (s for s in slots if s.get("role") == "outcome"), None)
        if outcome_slot is None:
            continue
        ov = np.frombuffer(outcome_slot["vec"], dtype=np.float32)
        if len(ov) != N_CLASSES:
            continue
        pred_cls = int(np.argmax(ov))
        if pred_cls == true_cls:
            exp.scene_weights[top_idx] = min(
                HEBBIAN_MAX_W,
                exp.scene_weights[top_idx] * HEBBIAN_BOOST)
        else:
            exp.scene_weights[top_idx] = max(
                HEBBIAN_MIN_W,
                exp.scene_weights[top_idx] * HEBBIAN_DECAY)


# ---- Experiment ----

def main():
    klines = load_or_fetch()
    if len(klines) < LOOKBACK + HORIZON + 100:
        print("Not enough data")
        return

    print("\nComputing features...")
    feats = compute_features(klines)
    rets = compute_returns(klines)

    # Valid rows: have both features (t >= LOOKBACK) and outcomes (t < n-HORIZON)
    valid = np.arange(LOOKBACK, len(klines) - HORIZON)
    print(f"  valid samples: {len(valid)}")

    # Time-ordered split FIRST, then fit class thresholds on train only
    split = int(len(valid) * TRAIN_FRAC)
    train_idx = valid[:split]
    test_idx = valid[split:]

    train_rets = rets[train_idx]
    if USE_TERTILES:
        low_thresh = float(np.nanpercentile(train_rets, 100.0 / 3.0))
        high_thresh = float(np.nanpercentile(train_rets, 200.0 / 3.0))
        print(f"  tertile thresholds (train-fitted): "
              f"down<{low_thresh:+.4f}  flat  up>{high_thresh:+.4f}")
    else:
        low_thresh = -THRESHOLD
        high_thresh = THRESHOLD
        print(f"  fixed thresholds: down<{low_thresh:+.4f}  "
              f"up>{high_thresh:+.4f}")

    outs = compute_outcomes(rets, low_thresh, high_thresh)

    # Class distribution (train vs test)
    train_counts = np.bincount(outs[train_idx], minlength=N_CLASSES)
    test_counts = np.bincount(outs[test_idx], minlength=N_CLASSES)
    print(f"  train distribution: down={train_counts[0]} "
          f"flat={train_counts[1]} up={train_counts[2]}")
    print(f"  test distribution:  down={test_counts[0]} "
          f"flat={test_counts[1]} up={test_counts[2]}")
    majority = int(np.argmax(test_counts))
    majority_rate = test_counts[majority] / len(test_idx)
    print(f"  test majority class: {DIRECTIONS[majority]}  "
          f"(baseline hit rate: {majority_rate:.3f})")
    print(f"  train: {len(train_idx)}   test: {len(test_idx)}")

    mode = "online Hebbian" if ONLINE_HEBBIAN else "raw"
    print(f"\nBuilding brain + loading train scenes ({mode})...")
    brain = Brain(warm_cold=False, use_language=False, run_oscillator=False)
    t0 = time.time()
    for i, t in enumerate(train_idx):
        if ONLINE_HEBBIAN:
            keys = regime_keys(feats[t])
            online_hebbian_update(brain, feats[t], int(outs[t]), keys)
        scene = build_scene(feats[t], int(outs[t]), int(t))
        brain.learn_scene(scene, top_k=None, apply_hebbian=False)
        if ONLINE_HEBBIAN and (i + 1) % REBUILD_EVERY == 0:
            # Stale matrices use old weights; periodic rebuilds keep the
            # online updates visible to subsequent neighbour lookups.
            for exp in brain.loader.warm.values():
                if exp.scene_vecs:
                    exp._rebuild_matrix()
    # Final rebuild (weight changes since last rebuild)
    if ONLINE_HEBBIAN:
        for exp in brain.loader.warm.values():
            if exp.scene_vecs:
                exp._rebuild_matrix()
    print(f"  stored {brain.stats()['total_scenes']} scenes "
          f"in {time.time()-t0:.1f}s")

    if ONLINE_HEBBIAN:
        # Summarise weight shaping per expert
        print("  weight shape per expert:")
        for lem, exp in sorted(brain.loader.warm.items(),
                               key=lambda kv: -len(kv[1].scene_vecs)):
            w = np.array(exp.scene_weights)
            n_b = int((w > 1.1).sum())
            n_d = int((w < 0.9).sum())
            print(f"    {lem:12s} n={len(w):5d}  boosted={n_b:4d} "
                  f"decayed={n_d:4d}  w: mean={w.mean():.2f} "
                  f"min={w.min():.2f} max={w.max():.2f}")

    print("\nQuerying test set...")
    records = []  # (true_cls, pred_cls, margin)
    t0 = time.time()
    for i, t in enumerate(test_idx):
        if i % 500 == 0 and i > 0:
            dt = time.time() - t0
            print(f"  {i}/{len(test_idx)}  ({i/dt:.0f} q/s)")
        gap, keys = build_gap(feats[t])
        result = brain.query_scene(gap, keys)
        pred_cls = -1
        av = result.get("answer_vec")
        if av is not None:
            vec = np.frombuffer(av, dtype=np.float32)
            if len(vec) == N_CLASSES:
                pred_cls = int(np.argmax(vec))
        records.append((int(outs[t]), pred_cls, float(result.get("margin", 0.0))))
    print(f"  done in {time.time()-t0:.1f}s")

    records = np.array(records, dtype=[("true", "i4"), ("pred", "i4"),
                                        ("margin", "f4")])

    print("\n--- Results ---")
    n = len(records)
    n_answered = int((records["pred"] >= 0).sum())
    n_abstain = n - n_answered
    overall_hit = float(((records["pred"] == records["true"]) &
                          (records["pred"] >= 0)).sum()) / max(n_answered, 1)
    print(f"  total:    {n}")
    print(f"  answered: {n_answered}   abstain: {n_abstain}")
    print(f"  overall hit rate (answered): {overall_hit:.3f}")
    print(f"  chance (uniform):            {1.0/N_CLASSES:.3f}")
    print(f"  majority baseline:           {majority_rate:.3f}")

    # Hit rate per predicted class
    print("\n  hit rate by predicted class:")
    for c in range(N_CLASSES):
        mask = records["pred"] == c
        if mask.sum() == 0:
            continue
        hr = float((records["true"][mask] == c).sum()) / mask.sum()
        print(f"    {DIRECTIONS[c]:5s}: n={int(mask.sum()):5d}  hit={hr:.3f}")

    # Margin-bucketed hit rate — THE core question
    print("\n  hit rate by margin bucket:")
    buckets = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20),
               (0.20, 0.30), (0.30, 0.50), (0.50, 1.01)]
    answered = records[records["pred"] >= 0]
    print(f"    {'bucket':>15s}  {'n':>6s}  {'hit':>6s}  {'lift vs chance':>15s}")
    for lo, hi in buckets:
        mask = (answered["margin"] >= lo) & (answered["margin"] < hi)
        k = int(mask.sum())
        if k == 0:
            print(f"    [{lo:.2f},{hi:.2f}): {k:>6d}")
            continue
        hit = float((answered["true"][mask] == answered["pred"][mask]).sum()) / k
        lift = hit - (1.0 / N_CLASSES)
        print(f"    [{lo:.2f},{hi:.2f}): {k:>6d}  {hit:.3f}  {lift:+.3f}")

    # Confusion matrix
    print("\n  confusion matrix (rows=true, cols=pred):")
    print(f"    {'':>6s}  " + "  ".join(f"{d:>6s}" for d in DIRECTIONS))
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for t_cls, p_cls in zip(records["true"], records["pred"]):
        if p_cls >= 0:
            cm[t_cls, p_cls] += 1
    for i, d in enumerate(DIRECTIONS):
        print(f"    {d:>6s}  " + "  ".join(f"{cm[i,j]:>6d}" for j in range(N_CLASSES)))

    # Balanced accuracy = mean per-class recall (corrects for imbalance)
    per_class_recall = []
    for c in range(N_CLASSES):
        true_c = cm[c].sum()
        if true_c == 0:
            continue
        per_class_recall.append(cm[c, c] / true_c)
    bal_acc = np.mean(per_class_recall)
    print(f"\n  balanced accuracy (mean per-class recall): {bal_acc:.3f}")
    print(f"  chance for balanced accuracy:               {1.0/N_CLASSES:.3f}")

    # Directional-only: when the brain predicts up OR down (ignores flat),
    # how often is the SIGN correct? Useful-if-honest subset.
    dir_mask = (answered["pred"] == 0) | (answered["pred"] == 2)
    dir_n = int(dir_mask.sum())
    if dir_n > 0:
        sub = answered[dir_mask]
        dir_hit = float((sub["true"] == sub["pred"]).sum()) / dir_n
        # Sign correct (vs opposite direction, ignoring flat as "wrong")
        opp = np.where(sub["pred"] == 0, 2,
                       np.where(sub["pred"] == 2, 0, -1))
        opposite_n = int((sub["true"] == opp).sum())
        print(f"\n  directional predictions only (up/down, ignoring flat):")
        print(f"    n={dir_n}  exact hit={dir_hit:.3f}  "
              f"opposite direction={opposite_n/dir_n:.3f}")
        # Up-vs-down conditional on the brain being directional
        for c in (0, 2):
            m = sub["pred"] == c
            if m.sum() == 0:
                continue
            base = (records["true"] == c).sum() / len(records)
            hit = float((sub["true"][m] == c).sum()) / m.sum()
            print(f"    pred={DIRECTIONS[c]:5s}: n={int(m.sum()):4d}  "
                  f"hit={hit:.3f}  base_rate={base:.3f}  "
                  f"lift={hit-base:+.3f}")

    # Cumulative: "if I only trade when margin > X"
    print("\n  cumulative hit rate (margin >= X):")
    print(f"    {'min_margin':>12s}  {'n':>6s}  {'hit':>6s}  "
          f"{'coverage':>10s}  {'lift':>7s}")
    for thresh in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        mask = answered["margin"] >= thresh
        k = int(mask.sum())
        if k == 0:
            print(f"    >= {thresh:.2f}    : {k:>6d}")
            continue
        hit = float((answered["true"][mask] == answered["pred"][mask]).sum()) / k
        lift = hit - (1.0 / N_CLASSES)
        cov = k / n_answered
        print(f"    >= {thresh:.2f}    : {k:>6d}  {hit:.3f}  "
              f"{cov:>10.3f}  {lift:+.3f}")

    brain.oscillator.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
