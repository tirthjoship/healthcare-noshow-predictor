"""Healthcare no-show predictor: outreach call-list demo (ADR-010, amended 2026-10-23).

Renders precomputed artifacts (no model training at runtime) so it deploys
cleanly on Streamlit Community Cloud without the raw Kaggle dataset:

    demo/scored_appointments.csv   : calibrated risk + per-patient SHAP
    reports/model_metrics.json     : validation metrics (GroupKFold + temporal)
    reports/fairness.json          : threshold view + top-K-per-day capacity view
    reports/business_impact.json   : slots-first what-if at the app's operating point
    reports/shap/global_importance.json : mean |SHAP| per feature

Regenerate the artifacts with:
    python scripts/run_training.py && python scripts/run_phase2.py && python scripts/export_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import altair as alt
import pandas as pd
import streamlit as st

from adapters.visualization.call_list import precision_at_capacity, rank_call_list

ROOT = Path(__file__).resolve().parent
DEMO_CSV = ROOT / "demo" / "scored_appointments.csv"
REPORTS = ROOT / "reports"
SHAP_JSON = REPORTS / "shap" / "global_importance.json"
REPO_URL = "https://github.com/tirthjoship/healthcare-noshow-predictor"
DECISION_TREE_URL = f"{REPO_URL}/blob/main/docs/DECISION_TREE.md"

FEATURE_LABELS = {
    "lead_time_days": "Days between booking and appointment",
    "age": "Age",
    "neighbourhood_enc": "Neighbourhood",
    "sms_received": "SMS reminder sent",
    "scholarship": "Welfare enrolment (Bolsa Família)",
    "gender_enc": "Gender",
    "diabetes": "Diabetes",
    "alcoholism": "Alcoholism",
    "hypertension": "Hypertension",
    "handicap": "Disability",
}

SLICE_LABELS = {
    "age_band": "Age band",
    "gender": "Gender",
    "neighbourhood": "Neighbourhood (top 5 by volume vs other)",
    "sms_received": "SMS reminder",
    "scholarship": "Welfare enrolment",
}


@st.cache_data
def load_scored() -> pd.DataFrame:
    return pd.read_csv(DEMO_CSV)


@st.cache_data
def load_json(path: str) -> dict[str, object]:
    return cast(dict[str, object], json.loads(Path(path).read_text()))


def fmt_day(iso: str) -> str:
    """ISO date string to 'Wed 1 Jun 2016' for display; values stay ISO."""
    ts = pd.Timestamp(iso)
    return f"{ts:%a} {ts.day} {ts:%b %Y}"


def metric_mean(value: object) -> float:
    """Model metrics store CV results as 'mean +/- sd' strings; return the mean."""
    return float(str(value).split("+/-")[0])


st.set_page_config(page_title="No-Show Call List", layout="wide")

st.title("Appointment No-Show: Daily Call List")
st.caption(
    "Public Kaggle benchmark (110K appointments, Vitória, Brazil, Apr to Jun 2016). "
    "Not connected to any health system; no real patients are identifiable."
)

missing = [
    p
    for p in (
        DEMO_CSV,
        REPORTS / "model_metrics.json",
        REPORTS / "fairness.json",
        REPORTS / "business_impact.json",
    )
    if not p.exists()
]
if missing:
    st.error(
        "Missing artifacts: "
        + ", ".join(str(p.relative_to(ROOT)) for p in missing)
        + ". Regenerate them with the scripts listed in this file's docstring."
    )
    st.stop()

df = load_scored()
metrics = load_json(str(REPORTS / "model_metrics.json"))
fair = load_json(str(REPORTS / "fairness.json"))
impact = load_json(str(REPORTS / "business_impact.json"))
capacity = cast("dict[str, object]", fair["capacity"])

tab_start, tab_calls, tab_why, tab_fair, tab_impact, tab_how = st.tabs(
    [
        "Start here",
        "Daily call list",
        "Why this patient?",
        "Fairness",
        "Impact",
        "How we built it",
    ]
)

# --------------------------------------------------------------------------- start
with tab_start:
    st.subheader(
        "The problem in one line",
        help="A no-show is a booked appointment where the patient does not come "
        "and the slot goes unused.",
    )
    st.markdown(
        "About **1 in 5** booked appointments end up as no-shows. A clinic outreach "
        "team can only make a limited number of reminder calls a day. **Who should "
        "they call first?**"
    )

    validation = cast("dict[str, dict[str, dict[str, object]]]", metrics["validation"])
    cv = validation["groupkfold"]["calibrated_xgboost"]
    temporal = validation["temporal_holdout"]["calibrated_xgboost"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Ranking quality (AUC)",
        f"{metric_mean(cv['auc']):.2f}",
        help="AUC is the chance that a patient who missed ranks above one who "
        "showed up. 0.5 = coin flip, 1.0 = perfect. 5-fold, grouped by patient.",
    )
    c2.metric(
        "Holds up on a later month",
        f"{float(cast(float, temporal['auc'])):.2f}",
        help="Trained on April to May, tested on unseen June appointments.",
    )
    c3.metric(
        "Calibration error (ECE)",
        f"{metric_mean(cv['ece']):.3f}",
        help="Calibration means a predicted 30% really happens about 30% of the "
        "time. This error is the average gap between predicted and actual. "
        "Lower is better.",
    )
    c4.metric(
        "Hit rate on the call list",
        f"{float(cast(float, impact['precision_at_k'])):.0%}",
        help="Hit rate is the share of the top 20 patients per day who really did not show up "
        "(June test data). The June no-show rate is about 18%.",
    )

    st.subheader("Three things we found")
    st.markdown("""
1. **No-shows are a scheduling problem, not a medical one.** The longer the gap between
   booking and the appointment, the more likely a no-show. Same-day bookings almost
   always show up (about 5% miss); next-day bookings miss about 21% of the time.
   Diabetes, hypertension and other conditions barely matter.
2. **SMS reminders *look* harmful, but they are not.** Patients who got an SMS missed more
   often, because the clinic sent SMS to long-lead (higher-risk) appointments. A
   selection effect, not cause and effect.
3. **Ranking by raw risk leaves older patients behind.** Patients 55+ are about a
   fifth of all no-shows but almost never make the daily call list. See the
   Fairness tab before using a list like this for real.
""")
    st.info(
        "**How to use this demo:** pick a day and a call capacity in *Daily call list*, "
        "click into a patient in *Why this patient?*, then check *Fairness* and "
        "*Impact*. *How we built it* shows every decision and why."
    )

# --------------------------------------------------------------------------- calls
with tab_calls:
    left, right = st.columns([1, 2])
    with left:
        days = sorted(df["appointment_day"].unique().tolist())
        day = st.selectbox("Appointment day", days, index=0, format_func=fmt_day)
        day_df = df[df["appointment_day"] == day]
        k = st.slider(
            "Calls the team can make today",
            min_value=1,
            max_value=int(min(50, len(day_df))),
            value=int(min(10, len(day_df))),
            help="Slots per day means how many reminder calls the team can make. "
            "The list shows that many patients, highest risk first.",
        )
        ranked = rank_call_list(day_df.to_dict("records"), k)
        hits = sum(bool(r["actual_no_show"]) for r in ranked)
        base_rate = float(day_df["actual_no_show"].mean())
        st.metric(
            "Would-be no-shows reached",
            f"{hits} of {len(ranked)} calls",
            help="In hindsight: how many of the people called really did not show up.",
        )
        st.metric(
            "Hit rate vs calling at random",
            f"{precision_at_capacity(ranked):.0%}",
            delta=f"{precision_at_capacity(ranked) - base_rate:+.0%} vs random",
            help="Hit rate is the share of called patients who really missed. "
            "Random calling would reach the day's overall no-show rate.",
        )
        st.caption(
            f"Demo sample: {len(day_df)} of that day's appointments. "
            f"Random calling would reach about {base_rate:.0%}."
        )
    with right:
        st.subheader(f"Who to call first on {fmt_day(day)}")
        ranked_df = pd.DataFrame(ranked)[
            [
                "age",
                "gender",
                "lead_time_days",
                "sms_received",
                "risk_score",
                "actual_no_show",
            ]
        ].rename(
            columns={
                "age": "Age",
                "gender": "Gender",
                "lead_time_days": "Days booked ahead",
                "sms_received": "SMS sent",
                "risk_score": "Risk",
                "actual_no_show": "Actually missed?",
            }
        )
        ranked_df["SMS sent"] = ranked_df["SMS sent"].map({1: "yes", 0: "no"})
        ranked_df["Actually missed?"] = ranked_df["Actually missed?"].map(
            {1: "Missed", 0: "Showed up"}
        )
        st.dataframe(
            ranked_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Risk": st.column_config.ProgressColumn(
                    "Risk", min_value=0.0, max_value=1.0, format="%.2f"
                ),
            },
        )
        st.caption(
            "'Actually missed?' is hindsight, shown only to judge the list. "
            "It is never a model input."
        )

# --------------------------------------------------------------------------- why
with tab_why:
    st.subheader(
        "What drives no-show risk overall",
        help="SHAP is a method that splits each prediction into the share "
        "contributed by each factor. Longer bars mean the factor moves "
        "predictions more, on average.",
    )
    if SHAP_JSON.exists():
        shap_raw = cast(
            "dict[str, float]",
            load_json(str(SHAP_JSON))["mean_abs_shap"],
        )
        shap_df = pd.DataFrame(
            {
                "Factor": [FEATURE_LABELS.get(f, f) for f in shap_raw],
                "Average influence": list(shap_raw.values()),
            }
        )
        st.altair_chart(
            alt.Chart(shap_df)
            .mark_bar()
            .encode(
                x=alt.X("Average influence:Q", title="Average influence on risk"),
                y=alt.Y("Factor:N", sort="-x", title=None),
                tooltip=["Factor", alt.Tooltip("Average influence:Q", format=".3f")],
            ),
            width="stretch",
        )
    st.markdown(
        "Booking lead time dominates, then age. Clinical conditions sit near zero "
        "(ADR-012)."
    )
    st.divider()
    st.subheader("Why is this patient on the list?")
    day2 = st.selectbox(
        "Appointment day",
        sorted(df["appointment_day"].unique()),
        key="whyday",
        format_func=fmt_day,
    )
    day2_df = df[df["appointment_day"] == day2].sort_values(
        "risk_score", ascending=False
    )
    labels = [
        f"#{i + 1}  ·  risk {r.risk_score:.2f}  ·  age {int(r.age)}  ·  "
        f"booked {int(r.lead_time_days)} days ahead"
        for i, r in enumerate(day2_df.itertuples())
    ]
    choice = st.selectbox(
        "Patient (ranked by risk)",
        range(len(labels)),
        format_func=labels.__getitem__,
        help="Patients on the chosen day, highest predicted risk first.",
    )
    row = day2_df.iloc[int(choice)]
    contribs = {
        FEATURE_LABELS.get(f, f): float(row[f"shap_{f}"])
        for f in FEATURE_LABELS
        if f"shap_{f}" in row
    }
    top_up = max(contribs, key=lambda f: contribs[f])
    top_down = min(contribs, key=lambda f: contribs[f])
    st.markdown(
        f"**Risk {row['risk_score']:.2f}.** Biggest push **up**: *{top_up}*. "
        f"Biggest push **down**: *{top_down}*."
    )
    contrib_df = (
        pd.DataFrame({"factor": list(contribs), "effect": list(contribs.values())})
        .sort_values("effect")
        .set_index("factor")
    )
    st.bar_chart(contrib_df, horizontal=True)
    st.caption(
        "Bars to the right raise this patient's risk, bars to the left lower it "
        "(SHAP values, log-odds scale)."
    )

# --------------------------------------------------------------------------- fairness
with tab_fair:
    st.subheader(
        "Who makes the daily call list?",
        help="This is the setting the app actually uses: the top "
        f"{capacity['k']} patients per day by risk, across all June appointments. "
        "Compare each group's share of the call list with its share of all "
        "no-shows. A big gap means the group is under-served by the list.",
    )
    st.caption(
        f"Top {capacity['k']} per day across all June appointments "
        f"({capacity['n_days']} days, {capacity['n_selected']} calls). "
        "Compare each group's share of the call list with its share of no-shows."
    )
    cap_slices = cast("dict[str, dict[str, dict[str, float]]]", capacity["slices"])
    attr = st.selectbox(
        "Group by",
        list(cap_slices),
        index=list(cap_slices).index("age_band") if "age_band" in cap_slices else 0,
        format_func=lambda a: SLICE_LABELS.get(a, a),
        help="Age band is where the main finding is. Other groups look closer "
        "to even.",
    )
    cap_df = pd.DataFrame(cap_slices[attr]).T
    chart_df = (
        cap_df[["share_of_no_shows", "share_of_list"]]
        .rename(
            columns={
                "share_of_no_shows": "Share of all no-shows",
                "share_of_list": "Share of the call list",
            }
        )
        .rename_axis("Group")
        .reset_index()
        .melt(id_vars="Group", var_name="Measure", value_name="Share")
    )
    group_order = [str(g) for g in cap_df.index]
    st.altair_chart(
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            y=alt.Y(
                "Group:N",
                sort=group_order,
                title=None,
                axis=alt.Axis(labelLimit=300, labelAngle=0),
            ),
            yOffset=alt.YOffset("Measure:N"),
            x=alt.X("Share:Q", title="Share", axis=alt.Axis(format="%")),
            color=alt.Color("Measure:N", title=None),
            tooltip=[
                "Group",
                "Measure",
                alt.Tooltip("Share:Q", format=".1%"),
            ],
        ),
        width="stretch",
    )
    age = cap_slices["age_band"]["55+"]
    st.warning(
        f"**Known limitation (ADR-013):** patients 55+ are "
        f"{age['share_of_no_shows']:.0%} of no-shows but only "
        f"{age['share_of_list']:.1%} of the call list. Their risk scores are honest "
        "(well calibrated), but their no-show rate is lower, so ranking by raw risk "
        "pushes them off a short list. Before real use: review per-group capacity or "
        "thresholds with clinical and equity input."
    )

    with st.expander("Screening view: a single probability threshold (more technical)"):
        threshold_rule = str(fair.get("threshold_rule", ""))
        st.caption(
            f"Threshold {fair['threshold']}: {threshold_rule}. "
            "FNR = share of a group's no-shows the model misses; "
            "FPR = share of a group's attenders flagged anyway."
        )
        slices = cast("dict[str, dict[str, dict[str, object]]]", fair["slices"])
        for name, groups in slices.items():
            st.markdown(f"**{SLICE_LABELS.get(name, name)}**")
            st.dataframe(
                pd.DataFrame(groups)
                .T[["n", "base_rate", "selection_rate", "fpr", "fnr", "ece"]]
                .rename(
                    columns={
                        "base_rate": "no-show rate",
                        "selection_rate": "flagged",
                        "fpr": "FPR",
                        "fnr": "FNR (missed)",
                        "ece": "calibration error",
                    }
                ),
                width="stretch",
            )

# --------------------------------------------------------------------------- impact
with tab_impact:
    st.subheader("What a 20-call list could recover (a labeled what-if)")
    slots_day = float(cast(float, impact["recoverable_slots_per_day"]))
    slots_month = float(cast(float, impact["recoverable_slots_per_month"]))
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Hit rate on the list",
        f"{float(cast(float, impact['precision_at_k'])):.0%}",
        help="Share of the called patients who really did not show up. It ranges "
        "from 43 to 45% depending on how tied scores are ordered.",
    )
    c2.metric(
        "Appointment slots per day",
        f"~{slots_day:.1f}",
        help="Slots per day = calls per day times the hit rate. A slot is one "
        "appointment time that could be refilled.",
    )
    c3.metric(
        "Appointment slots per month",
        f"~{slots_month:.0f}",
        help="Slots per day times the working days in a month.",
    )
    cost = st.number_input(
        "Value of one recovered slot (edit to your setting)",
        min_value=0.0,
        value=float(cast(float, impact["cost_per_noshow"])),
        step=25.0,
        help="Default USD 200 is a US primary-care literature estimate, not clinic data.",
    )
    st.metric("Monthly value at that rate", f"${slots_month * cost:,.0f}")
    st.caption(
        "Upper bound: assumes every correctly flagged, called patient keeps or "
        "rebooks their slot; real conversion is lower. "
        f"Hit rate measured on the top {int(float(cast(float, impact['daily_call_capacity'])))} "
        f"patients per day in the June 2016 test data ({capacity['n_days']} days, "
        f"{capacity['n_selected']} calls)."
    )


# --------------------------------------------------------------------------- how
with tab_how:
    st.subheader("From raw data to this app: every decision and why")
    st.markdown("""
| Step | What we saw | What we decided | Decision record |
|---|---|---|---|
| Problem | Outreach teams have limited calls | Rank patients for a daily call list | ADR-001 to 003, 005 |
| Data quality | 35% of lead times looked negative | They were same-day bookings: compute calendar-day gaps, drop 5 impossible rows | ADR-014 |
| Exploration | SMS linked to *more* no-shows | Keep it, document it as a selection effect | ADR-006 |
| Validation | 39% of patients book more than once | Split by patient, and test on a later month | ADR-011 |
| Models | Logistic AUC 0.66, XGBoost 0.73 | XGBoost, then calibrate so % means % | ADR-008 |
| Explanation | Lead time and age dominate | Frame as an operations problem | ADR-012 |
| Fairness | 55+ nearly absent from the list | Disclose now, review before real use | ADR-009, ADR-013 |
| Impact | Dollar headline oversold it | Lead with slots; dollars editable | ADR-004 |
""")
    st.caption(
        "ADR = architecture decision record, a short note explaining one decision. "
        "All of them are in the GitHub repo."
    )
    st.markdown(
        f"Full visual decision tree: [docs/DECISION_TREE.md]({DECISION_TREE_URL})"
    )
    st.markdown(f"Code, tests and ADRs: [{REPO_URL}]({REPO_URL})")
