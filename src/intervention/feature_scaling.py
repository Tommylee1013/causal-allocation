import numpy as np
import pandas as pd
import json

def month_end_range(df: pd.DataFrame, start="2015-01-01"):
    idx = pd.to_datetime(df.index)
    df = df.copy()
    df.index = idx
    df = df.loc[start:]
    me = pd.date_range(df.index.min(), df.index.max(), freq="M")
    me = [d for d in me if d in df.index or (df.index[df.index <= d].max() is not pd.NaT)]
    return pd.DatetimeIndex(me)

def _safe_last_leq_index(idx: pd.DatetimeIndex, t: pd.Timestamp):
    sub = idx[idx <= t]
    return None if len(sub) == 0 else sub.max()

def _zscore(x: pd.Series, win: int):
    mu = x.rolling(win).mean()
    sd = x.rolling(win).std(ddof=0)
    return (x - mu) / sd

def build_feature_template(
    macro_features: pd.DataFrame,
    asof: pd.Timestamp,
    feature: str,
    horizons=(21,),                # 1개월(거래일) 기준
    lookbacks=(21, 63, 126, 252),   # 1/3/6/12개월
) -> dict:
    df = macro_features.copy()
    df.index = pd.to_datetime(df.index)
    idx = df.index

    t0 = _safe_last_leq_index(idx, asof)
    if t0 is None or feature not in df.columns:
        return {}

    s = df[feature].dropna()
    s = s.loc[:t0]
    if len(s) < max(lookbacks) + 5:
        # 데이터 부족하면 가능한 범위로만 생성
        lookbacks = tuple(lb for lb in lookbacks if lb < len(s))

    last = float(s.iloc[-1])
    out = {
        "feature": feature,
        "asof": str(pd.Timestamp(t0).date()),
        "last_value": last,
        "changes": {},
        "vol": {},
        "zscore": {},
        "recent_path": {
            "last_5": [float(v) for v in s.tail(5).values],
            "last_21": [float(v) for v in s.tail(21).values] if len(s) >= 21 else [float(v) for v in s.values],
        },
    }

    for lb in lookbacks:
        if len(s) <= lb:
            continue
        delta = float(s.iloc[-1] - s.iloc[-1 - lb])
        pct = float(s.pct_change(lb).iloc[-1]) if np.isfinite(s.pct_change(lb).iloc[-1]) else None
        out["changes"][f"delta_{lb}d"] = delta
        out["changes"][f"pct_{lb}d"] = pct

        # 롤링 변동성(일간 변화 기준)
        d1 = s.diff()
        vol = float(d1.rolling(lb).std(ddof=0).iloc[-1]) if len(d1) >= lb else float(d1.std(ddof=0))
        out["vol"][f"std_diff1_{lb}d"] = vol

        z = _zscore(s, lb).iloc[-1]
        out["zscore"][f"z_{lb}d"] = float(z) if np.isfinite(z) else None

    # 목표: 1개월 ahead 변화량을 예측하도록 유도할 입력
    # horizon별로 “최근 horizon 수익/변화”를 추가
    for h in horizons:
        if len(s) > h:
            out["changes"][f"delta_{h}d"] = float(s.iloc[-1] - s.iloc[-1 - h])
            ph = s.pct_change(h).iloc[-1]
            out["changes"][f"pct_{h}d"] = float(ph) if np.isfinite(ph) else None

    return out