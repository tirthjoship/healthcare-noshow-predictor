"""Healthcare no-show predictor — outreach call-list demo (ADR-010).

Renders precomputed artifacts (no model training at runtime) so it deploys
cleanly on Streamlit Community Cloud without the raw Kaggle dataset:

    demo/scored_appointments.csv   — calibrated risk + per-patient SHAP
    reports/fairness.json          — per-slice FPR/FNR + calibration
    reports/business_impact.json   — labeled what-if projection
    reports/shap/global_importance.png

Regenerate the artifacts with: python scripts/run_phase2.py && python scripts/export_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd
import streamlit as st

from adapters.visualization.call_list import precision_at_capacity, rank_call_list

ROOT = Path(__file__).resolve().parent
DEMO_CSV = ROOT / "demo" / "scored_appointments.csv"
REPORTS = ROOT / "reports"
SHAP_PNG = REPORTS / "shap" / "global_importance.png"
BEESWARM_PNG = REPORTS / "shap" / "shap_beeswarm.png"

FEATURE_LABELS = {
    "lead_time_days": "Lead time (days)",
    "age": "Age",
    "neighbourhood_enc": "Neighbourhood",
    "sms_received": "SMS received",
    "scholarship": "Scholarship (Bolsa Família)",
    "gender_enc": "Gender",
    "diabetes": "Diabetes",
    "alcoholism": "Alcoholism",
    "hypertension": "Hypertension",
    "handicap": "Handicap",
}


@st.cache_data
def load_scored() -> pd.DataFrame:
    return pd.read_csv(DEMO_CSV)


@st.cache_data
def load_json(path: str) -> dict[str, object]:
    return cast(dict[str, object], json.loads(Path(path).read_text()))


st.set_page_config(page_title="No-Show Call List", page_icon="🏥", layout="wide")

st.title("🏥 Appointment No-Show — Daily Call List")
st.caption(
    "Rank patients by calibrated no-show risk so an outreach team can call the "
    "highest-risk few. Public Kaggle benchmark (110K appointments, Vitória, Brazil) — "
    "not connected to any health system."
)

if not DEMO_CSV.exists():
    st.error(
        "Demo data not found. Run `python scripts/export_demo.py` to generate "
        "demo/scored_appointments.csv."
    )
    st.stop()

df = load_scored()

tab_calls, tab_shap, tab_fair, tab_impact = st.tabs(
    ["📞 Daily call list", "🔍 Why this patient?", "⚖️ Fairness", "💰 Business impact"]
)

with tab_calls:
    left, right = st.columns([1, 2])
    with left:
        days = sorted(df["appointment_day"].unique().tolist())
        day = st.selectbox("Appointment day", days, index=0)
        day_df = df[df["appointment_day"] == day]
        capacity = st.slider(
            "Daily call capacity (K)",
            min_value=1,
            max_value=int(min(50, len(day_df))),
            value=int(min(10, len(day_df))),
            help="How many reminder calls the team can make that day.",
        )
        candidates = day_df.to_dict("records")
        ranked = rank_call_list(candidates, capacity)
        hits = sum(bool(r["actual_no_show"]) for r in ranked)
        st.metric("Would-be no-shows caught", f"{hits} / {len(ranked)}")
        st.metric("Precision@K", f"{precision_at_capacity(ranked):.0%}")
        st.caption(f"{len(day_df)} appointments scheduled on {day}.")
    with right:
        st.subheader(f"Top {capacity} highest-risk patients — {day}")
        show_cols = [
            "appointment_id",
            "age",
            "gender",
            "lead_time_days",
            "sms_received",
            "risk_score",
            "risk_category",
            "actual_no_show",
        ]
        ranked_df = pd.DataFrame(ranked)[show_cols]
        st.dataframe(
            ranked_df,
            width="stretch",
            hide_index=True,
            column_config={
                "risk_score": st.column_config.ProgressColumn(
                    "Risk", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "actual_no_show": st.column_config.NumberColumn("No-show? (actual)"),
            },
        )
        st.caption(
            "`actual_no_show` is the true outcome (hindsight) — shown here only to "
            "illustrate hit rate; it is never a model input."
        )

with tab_shap:
    st.subheader("Global drivers — no-shows are an operations problem")
    if SHAP_PNG.exists():
        st.image(str(SHAP_PNG), width="stretch")
    st.markdown(
        "Lead time and age dominate; clinical comorbidities (diabetes, "
        "hypertension, handicap) barely move the prediction (ADR-012)."
    )
    st.divider()
    st.subheader("Per-patient explanation")
    day2 = st.selectbox(
        "Appointment day ", sorted(df["appointment_day"].unique()), key="shapday"
    )
    day2_df = df[df["appointment_day"] == day2].sort_values(
        "risk_score", ascending=False
    )
    options = day2_df["appointment_id"].tolist()
    appt_id = st.selectbox("Patient (appointment id)", options)
    row = day2_df[day2_df["appointment_id"] == appt_id].iloc[0]
    st.write(
        f"**Risk:** {row['risk_score']:.2f} ({row['risk_category']}) · "
        f"age {int(row['age'])} · lead time {int(row['lead_time_days'])} days · "
        f"SMS {'yes' if row['sms_received'] else 'no'}"
    )
    contribs = {
        FEATURE_LABELS.get(f, f): row[f"shap_{f}"]
        for f in FEATURE_LABELS
        if f"shap_{f}" in row
    }
    contrib_df = (
        pd.DataFrame(
            {"feature": list(contribs.keys()), "shap": list(contribs.values())}
        )
        .sort_values("shap")
        .set_index("feature")
    )
    st.bar_chart(contrib_df, horizontal=True)
    st.caption("Positive = pushes risk up, negative = pushes risk down (log-odds).")

with tab_fair:
    st.subheader("Fairness across demographic slices")
    if not (REPORTS / "fairness.json").exists():
        st.warning("Run `python scripts/run_phase2.py` to generate fairness.json.")
    else:
        fair = load_json(str(REPORTS / "fairness.json"))
        overall = cast("dict[str, object]", fair["overall"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall FPR", f"{overall['fpr']:.0%}")
        c2.metric("Overall FNR", f"{overall['fnr']:.0%}")
        c3.metric("Overall ECE (calibration)", f"{overall['ece']:.3f}")
        st.caption(
            f"Operating threshold {fair['threshold']} · June holdout "
            f"(n={fair['n_total']:,}). Calibration holds *within* every slice too."
        )
        slices = cast("dict[str, object]", fair["slices"])
        for attr, groups in slices.items():
            st.markdown(f"**{attr}**")
            group_metrics = cast("dict[str, dict[str, object]]", groups)
            st.dataframe(
                pd.DataFrame(group_metrics).T[
                    [
                        "n",
                        "base_rate",
                        "selection_rate",
                        "fpr",
                        "fnr",
                        "precision",
                        "ece",
                    ]
                ],
                width="stretch",
            )
        st.info(
            "Gender is essentially fair. The model over-flags younger patients and "
            "under-flags 55+, and the SMS gap reflects the documented SMS confound "
            "(ADR-006) — both are honest limitations, surfaced rather than hidden."
        )

with tab_impact:
    st.subheader("Business impact (labeled what-if)")
    if not (REPORTS / "business_impact.json").exists():
        st.warning(
            "Run `python scripts/run_phase2.py` to generate business_impact.json."
        )
    else:
        impact = load_json(str(REPORTS / "business_impact.json"))
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision@K", f"{impact['precision_at_k']:.0%}")
        c2.metric("Monthly value", f"${impact['monthly_value']:,.0f}")
        c3.metric("Annual value", f"${impact['annual_value']:,.0f}")
        st.caption(
            f"Assumes {impact['daily_call_capacity']:.0f} calls/day, "
            f"${impact['cost_per_noshow']:.0f}/recovered slot, "
            f"{impact['working_days_per_month']:.0f} working days/month. "
            "Upper bound — assumes every correctly-flagged, called no-show recovers "
            "a slot. Cost is a labeled literature estimate, not clinic data."
        )
