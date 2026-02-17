import numpy as np
import pandas as pd
import logging, sys
import time
import openai
import json

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

def build_macro_prompt(
        var_name: str,
        history: pd.Series,
        spx_ret_3m: float
    ) -> str:
    """
    Open AI GPT API에 넣을 prompt를 생성하는 함수
    """
    hist_tail = history.tail(12).to_string()

    prompt = f"""
You are a professional macroeconomic forecaster.

Target variable: {var_name}

Recent 12 monthly observations:
{hist_tail}

S&P 500 3-month return: {spx_ret_3m:.4f}

Instructions:
- Produce a 1-month-ahead forecast for the target variable.
- The forecast must be a continuous numeric value (not discrete, not categorical).
- Provide a confidence score between 0 and 1.
- Provide a short rationale focusing only on macro drivers.
- Do NOT reveal chain-of-thought or internal reasoning steps.
- Do NOT include any text outside JSON.
- Output must strictly follow the JSON schema below.

Output JSON schema:
{{
  "forecast": float,
  "confidence": float,
  "rationale": "short text"
}}
"""
    return prompt.strip()

def gpt_forecast_from_template(
        prompt: str,
        model: str = "gpt-5-mini"
    ) -> tuple :
    """
    지정된 템플릿을 이용해 GPT API로부터 forecast 값과 confidence를 반환
    """
    resp = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=800,   # 충분히 크게
        reasoning={"effort": "low"},  # reasoning 최소화
        text={"format": {"type": "text"}},  # 최종 텍스트 강제
        store=False
    )

    content = None

    # Responses API는 output_text가 가장 안정적
    if hasattr(resp, "output_text") and resp.output_text:
        content = resp.output_text
    else:
        # fallback
        try:
            content = resp.output[0].content[0].text
        except Exception:
            pass

    if content is None:
        raise RuntimeError(f"Empty response from model={model}: {resp}")

    parsed = json.loads(content)
    forecast = float(parsed["forecast"])
    confidence = float(parsed["confidence"])
    rationale = parsed["rationale"]

    return forecast, confidence, rationale

def generate_monthly_macro_forecasts(
        macro_features: pd.DataFrame,
        spx: pd.Series,
        forecast_start="2010-01-31",
        history_start=None,
        model="gpt-5-mini",
        sleep_sec=0.2,
        max_retries=2,
    ) -> pd.DataFrame:
    """
    입력한 데이터로부터 각 Macro Feature의 GPT 예측값을 반환
    """
    data = macro_features.copy()
    if history_start is not None:
        data = data.loc[history_start:].copy()

    spx_ret_3m = spx.pct_change(63)
    month_end_idx = data.resample("ME").last().index

    # 실제 호출 대상 월만 카운트
    forecast_months = [dt for dt in month_end_idx if pd.to_datetime(dt) >= pd.to_datetime(forecast_start)]
    total_calls = len(forecast_months) * len(data.columns)

    logging.info(f"[INIT] forecast_start={forecast_start}, history_start={history_start}")
    logging.info(f"[INIT] Months to forecast: {len(forecast_months)}")
    logging.info(f"[INIT] Total API calls (planned): {total_calls}")

    records = []
    done_calls = 0

    for m_i, dt in enumerate(month_end_idx, 1):
        if pd.to_datetime(dt) < pd.to_datetime(forecast_start):
            continue

        spx_val = spx_ret_3m.loc[dt] if dt in spx_ret_3m.index else 0.0

        logging.info(
            f"[MONTH {m_i}/{len(month_end_idx)}] "
            f"date={dt.date()} | done={done_calls}/{total_calls} | remaining={total_calls - done_calls}"
        )

        for v_i, var in enumerate(data.columns, 1):
            hist = data[var].loc[:dt]
            if len(hist) < 12:
                done_calls += 1
                logging.warning(f"[SKIP] {dt.date()} | {var} | insufficient history")
                continue

            logging.info(
                f"[CALL] {done_calls+1}/{total_calls} | {dt.date()} | {var} | sending request..."
            )

            prompt = build_macro_prompt(var, hist, float(spx_val))

            success = False
            for retry in range(max_retries + 1):
                try:
                    fcst, conf, rationale = gpt_forecast_from_template(prompt, model=model)

                    records.append({
                        "date": dt,
                        "variable": var,
                        "forecast_1m": float(fcst),
                        "confidence": float(conf),
                        "rationale": rationale,
                    })

                    logging.info(
                        f"[OK] {done_calls+1}/{total_calls} | {dt.date()} | {var} "
                        f"| fcst={float(fcst):.4f}, conf={float(conf):.2f}"
                    )
                    success = True
                    break

                except Exception as e:
                    logging.error(
                        f"[FAIL] {dt.date()} | {var} | retry={retry}/{max_retries} | {repr(e)}"
                    )
                    time.sleep(1.0)

            if not success:
                records.append({
                    "date": dt,
                    "variable": var,
                    "forecast_1m": None,
                    "confidence": None,
                    "rationale": None,
                })
                logging.error(f"[DROP] {dt.date()} | {var} | all retries failed")

            done_calls += 1

            logging.info(
                f"[PROGRESS] done={done_calls}/{total_calls} | remaining={total_calls - done_calls}"
            )

            time.sleep(sleep_sec)

    df = pd.DataFrame(records)
    logging.info(f"[DONE] Total rows generated: {len(df)}")

    return df