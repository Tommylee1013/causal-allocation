import numpy as np
import pandas as pd
import logging, sys
import time
import openai
import json

from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
import re

# constant values
root = logging.getLogger()
root.handlers.clear()  # 이미 설정된 핸들러 있으면 제거

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

root.addHandler(handler)
root.setLevel(logging.INFO)

client = openai.OpenAI(api_key=openai.api_key)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# class declaration
class SimpleVectorStore:
    def __init__(self, dim: int):
        self.embs = []
        self.meta = []
        self.dim = dim

    def add(self, emb: np.ndarray, meta: Dict):
        self.embs.append(emb.reshape(1, -1))
        self.meta.append(meta)

    def search(self, q: np.ndarray, k: int = 5):
        if len(self.embs) == 0:
            return []
        M = np.vstack(self.embs)
        sims = cosine_similarity(q.reshape(1, -1), M)[0]
        idx = np.argsort(-sims)[:k]
        return [(self.meta[i], sims[i]) for i in idx]

def safe_json_loads(txt: str):
    if txt is None:
        raise ValueError("Empty response")

    txt = txt.strip()

    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise ValueError(f"No JSON object found: {txt[:200]}")

    return json.loads(m.group(0))

def safe_json_extract(text: str):
    if text is None:
        return None
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start:end+1])
    except Exception:
        return None

def rag_decide(prompt: str, model="gpt-5-mini", max_retries=2):
    last_err = None

    for retry in range(max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=120,
            )

            content = None
            if hasattr(resp, "output_text") and resp.output_text:
                content = resp.output_text
            elif resp.output and resp.output[0].content:
                content = resp.output[0].content[0].text

            parsed = safe_json_extract(content)

            if parsed is None:
                raise ValueError("Invalid JSON")

            return bool(parsed["use"]), float(parsed["adjustment"]), parsed["reason"]

        except Exception as e:
            last_err = e
            time.sleep(0.5)

    raise RuntimeError(f"RAG JSON parse failed after retries: {repr(last_err)}")

def embed_texts(texts: List[str], model="text-embedding-3-large"):
    resp = client.embeddings.create(model=model, input=texts)
    return [np.array(x.embedding, dtype=float) for x in resp.data]

def build_rag_prompt(
        var, date,
        forecast,
        conf,
        retrieved_context: List[str]
    ) -> str:
    ctx = "\n\n".join(retrieved_context)

    prompt = f"""
You are a cautious macro portfolio manager.

You MUST return ONLY valid JSON.
Do NOT include any explanation or text outside JSON.
Do NOT use markdown.

Target variable: {var}
Date: {date}
Model forecast (1M): {forecast:.4f}
Model confidence: {conf:.2f}

Retrieved historical context:
{ctx}

Return STRICT JSON:
{{
  "use": true/false,
  "adjustment": float,
  "reason": "short text"
}}

Return JSON only.
No explanation.
No markdown.
No extra text.
The response must be a single JSON object.
"""
    return prompt.strip()

def rag_decide(prompt: str, model="gpt-5-mini"):
    resp = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=120,
    )
    content = resp.output_text
    if content is None and resp.output:
        content = resp.output[0].content[0].text

    parsed = json.loads(content)
    return bool(parsed["use"]), float(parsed["adjustment"]), parsed["reason"]

def build_vector_store_from_history(history_logs: pd.DataFrame):
    """
    history_logs columns:
    [date, variable, forecast_1m, realized_change_1m, confidence,
     error, abs_error, sq_error, hit_sign,
     rolling_mae_12m, rolling_rmse_12m, rolling_hit_12m, n_hist, has_realized, enough_history]
    """
    texts = []
    metas = []

    for _, r in history_logs.iterrows():
        txt = (
            f"Date={r['date']}, Var={r['variable']}, "
            f"Forecast={r['forecast_1m']:.3f}, "
            f"Realized={r['realized_change_1m']:.3f}, "
            f"Error={r['error']:.3f}, "
            f"HitRate12M={r['rolling_hit_12m'] if pd.notna(r['rolling_hit_12m']) else 'NA'}, "
            f"MAE12M={r['rolling_mae_12m'] if pd.notna(r['rolling_mae_12m']) else 'NA'}"
        )
        texts.append(txt)
        metas.append({
            "text": txt,
            "variable": r["variable"],
            "date": pd.to_datetime(r["date"])
        })

    embs = embed_texts(texts)
    store = SimpleVectorStore(dim=len(embs[0]))
    for e, m in zip(embs, metas):
        store.add(e, m)
    return store

def rag_filter_forecasts(
        forecast_1m_df: pd.DataFrame,
        confidence_df: pd.DataFrame,
        vector_store: SimpleVectorStore,
        k: int = 3,
        model="gpt-5-mini",
        sleep_sec=0.1,
    ) -> tuple :
    filtered = forecast_1m_df.copy()
    decision_log = []

    for dt in forecast_1m_df.index:
        for var in forecast_1m_df.columns:
            fcst = forecast_1m_df.loc[dt, var]
            conf = confidence_df.loc[dt, var]

            if pd.isna(fcst) or pd.isna(conf):
                filtered.loc[dt, var] = np.nan
                continue

            try:
                query_text = f"Var={var}, Date={dt}, Forecast={fcst:.3f}, Confidence={conf:.2f}"
                q_emb = embed_texts([query_text])[0]

                retrieved = vector_store.search(q_emb, k=k)
                retrieved_ctx = [m["text"] for m, _ in retrieved]

                prompt = build_rag_prompt(var, str(pd.to_datetime(dt).date()), float(fcst), float(conf), retrieved_ctx)
                use, adj, reason = rag_decide(prompt, model=model)

                filtered.loc[dt, var] = float(fcst) * float(adj) if use else 0.0

            except Exception as e:
                logging.error(f"[RAG_FAIL] {dt.date()} | {var} | {repr(e)}")
                filtered.loc[dt, var] = 0.0
                use, adj, reason = False, 0.0, "parse_fail_or_empty"

            decision_log.append({
                "date": dt,
                "variable": var,
                "use": use,
                "adjustment": adj,
                "reason": reason
            })

            time.sleep(sleep_sec)

    return filtered, pd.DataFrame(decision_log)

def build_history_logs(
        realized_df: pd.DataFrame,
        forecast_1m_df: pd.DataFrame,
        confidence_df: pd.DataFrame | None = None,
        horizon_months: int = 1,
        min_hist_months: int = 12,
        clip_conf: tuple[float, float] = (0.0, 1.0),
    ) -> pd.DataFrame:
    realized_df = realized_df.copy()
    forecast_1m_df = forecast_1m_df.copy()
    if confidence_df is not None:
        confidence_df = confidence_df.copy()

    # 인덱스를 date 컬럼으로 강제 변환
    realized_df = realized_df.reset_index().rename(columns={realized_df.index.name or "index": "date"})
    forecast_1m_df = forecast_1m_df.reset_index().rename(columns={forecast_1m_df.index.name or "index": "date"})
    if confidence_df is not None:
        confidence_df = confidence_df.reset_index().rename(columns={confidence_df.index.name or "index": "date"})

    realized_df["date"] = pd.to_datetime(realized_df["date"])
    forecast_1m_df["date"] = pd.to_datetime(forecast_1m_df["date"])
    if confidence_df is not None:
        confidence_df["date"] = pd.to_datetime(confidence_df["date"])

    vars_common = sorted(list(set(realized_df.columns) & set(forecast_1m_df.columns) - {"date"}))
    realized_df = realized_df[["date"] + vars_common]
    forecast_1m_df = forecast_1m_df[["date"] + vars_common]
    if confidence_df is not None:
        confidence_df = confidence_df[["date"] + vars_common]

    realized_df = realized_df.set_index("date")
    forecast_1m_df = forecast_1m_df.set_index("date")
    if confidence_df is not None:
        confidence_df = confidence_df.set_index("date")

    realized_change = realized_df.shift(-horizon_months) - realized_df

    fc_long = forecast_1m_df.stack().reset_index()
    fc_long.columns = ["date", "variable", "forecast_1m"]

    rl_long = realized_change.stack().reset_index()
    rl_long.columns = ["date", "variable", f"realized_change_{horizon_months}m"]

    out = fc_long.merge(rl_long, on=["date", "variable"], how="left")

    if confidence_df is not None:
        cf_long = confidence_df.stack().reset_index()
        cf_long.columns = ["date", "variable", "confidence"]
        out = out.merge(cf_long, on=["date", "variable"], how="left")
        out["confidence"] = out["confidence"].clip(*clip_conf)
    else:
        out["confidence"] = np.nan

    y = out[f"realized_change_{horizon_months}m"].astype(float)
    f = out["forecast_1m"].astype(float)

    out["error"] = y - f
    out["abs_error"] = out["error"].abs()
    out["sq_error"] = out["error"] ** 2
    out["hit_sign"] = ((np.sign(y) == np.sign(f)) & (y != 0) & (f != 0)).astype(float)

    out = out.sort_values(["variable", "date"]).reset_index(drop=True)

    def add_roll_stats(g):
        g["rolling_mae_12m"] = g["abs_error"].shift(1).rolling(min_hist_months).mean()
        g["rolling_rmse_12m"] = np.sqrt(g["sq_error"].shift(1).rolling(min_hist_months).mean())
        g["rolling_hit_12m"] = g["hit_sign"].shift(1).rolling(min_hist_months).mean()
        g["n_hist"] = g["abs_error"].shift(1).rolling(min_hist_months).count()
        return g

    out = out.groupby("variable", group_keys=False).apply(add_roll_stats)

    out["has_realized"] = out[f"realized_change_{horizon_months}m"].notna()
    out["enough_history"] = out["n_hist"].fillna(0) >= min_hist_months

    return out