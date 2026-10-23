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

import ui_theme as ui
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


GROUP_LABELS = {
    "F": "Female",
    "M": "Male",
    "no_sms": "No SMS",
    "sms": "SMS sent",
    "none": "Not enrolled",
    "scholarship": "Enrolled",
    "other": "All other",
}


def group_label(key: object) -> str:
    """Plain label for a slice group key; neighbourhood names are title-cased."""
    text = str(key)
    return GROUP_LABELS.get(text, text.title() if text.isupper() else text)


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
ui.inject_css()

st.markdown(
    ui.header(
        "HEALTHCARE OPERATIONS · APPOINTMENT NO-SHOWS",
        "Who should the clinic call first?",
        "A ranked daily call list for a reminder team that can only make a limited "
        "number of calls, tested on held-out appointments.",
        [
            ("110K appointments", None),
            (
                "Public data (Vitória, Brazil)",
                "Public Kaggle benchmark, April to June 2016.",
            ),
            (
                "Calibrated model",
                "Calibrated means a predicted 30% really happens about 30% of the time.",
            ),
            (
                "Not connected to any health system",
                "No real patients are identifiable.",
            ),
        ],
    ),
    unsafe_allow_html=True,
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
overall = cast("dict[str, float]", fair["overall"])
precision_k = float(cast(float, impact["precision_at_k"]))
june_base = float(overall["base_rate"])
cap_slices = cast("dict[str, dict[str, dict[str, float]]]", capacity["slices"])
age55 = cap_slices["age_band"]["55+"]

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
    validation = cast("dict[str, dict[str, dict[str, object]]]", metrics["validation"])
    cv = validation["groupkfold"]["calibrated_xgboost"]
    temporal = validation["temporal_holdout"]["calibrated_xgboost"]
    lift = round(precision_k / june_base, 1)

    st.markdown(
        ui.bottom_line(
            "Can a ranked call list beat calling at random?",
            f"Yes, about {lift:.1f} times better",
            "good",
            [
                f"The hit rate on the top 20 patients per day is {precision_k:.0%} "
                f"versus about {june_base:.0%} if the team called at random "
                "(June 2016 test data).",
            ],
            act="Pilot the list with one outreach team and track real call outcomes.",
            confidence="Moderate",
            confidence_why="Six test days, public data from one city, and the hit "
            "rate is 43 to 45% depending on how tied scores are ordered.",
            would_change="Real call outcomes, or a different clinic population.",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(ui.layer_label("Evidence"), unsafe_allow_html=True)
    st.markdown(
        ui.tiles(
            [
                ui.tile(
                    f"{metric_mean(cv['auc']):.2f}",
                    "Ranking quality (AUC)",
                    "good",
                    "AUC is the chance that a patient who missed ranks above one who "
                    "showed up. 0.5 = coin flip, 1.0 = perfect. 5-fold, grouped by patient.",
                ),
                ui.tile(
                    f"{float(cast(float, temporal['auc'])):.2f}",
                    "Holds up on a later month",
                    "good",
                    "Trained on April to May, tested on unseen June appointments.",
                ),
                ui.tile(
                    f"{metric_mean(cv['ece']):.3f}",
                    "Calibration error (ECE)",
                    "good",
                    "Calibration means a predicted 30% really happens about 30% of the "
                    "time. This error is the average gap between predicted and actual. "
                    "Lower is better.",
                ),
                ui.tile(
                    f"{precision_k:.0%}",
                    "Hit rate on the call list",
                    "good",
                    "Hit rate is the share of the top 20 patients per day who really "
                    "did not show up (June test data). The June no-show rate is "
                    f"about {june_base:.0%}.",
                ),
            ]
        ),
        unsafe_allow_html=True,
    )
    with ui.card("findings"):
        st.markdown(
            ui.card_title(
                "Three things we found",
                "A no-show is a booked appointment where the patient does not come "
                "and the slot goes unused.",
            ),
            unsafe_allow_html=True,
        )
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
        st.markdown(
            ui.takeaway(
                "About 1 in 5 booked appointments end up as no-shows, so the order "
                "of the calls matters."
            ),
            unsafe_allow_html=True,
        )
    st.info(
        "**How to use this demo:** pick a day and a call capacity in *Daily call list*, "
        "click into a patient in *Why this patient?*, then check *Fairness* and "
        "*Impact*. *How we built it* shows every decision and why."
    )

    st.markdown(ui.layer_label("Method"), unsafe_allow_html=True)
    with st.expander("How we know"):
        st.markdown(f"""
- **Ranking quality:** AUC {metric_mean(cv['auc']):.2f} in 5-fold validation grouped by
  patient, and {float(cast(float, temporal['auc'])):.2f} on a later month.
- **Calibration:** error {metric_mean(cv['ece']):.3f} (ECE), so a predicted risk can be
  read as a probability (ADR-008).
- **Split design:** each patient appears on only one side of every split, and the model
  was trained on April to May and tested on June (ADR-011).
- **Hit rate:** {precision_k:.0%} on the top 20 per day over {capacity['n_days']} June
  days ({capacity['n_selected']} calls); the June no-show rate is {june_base:.0%}.
- **Caveats:** one city, one clinic network, public data from 2016, and no real call
  outcomes yet. The hit rate moves between 43 and 45% depending on how tied scores
  are ordered.
- **Decision records:** ADR-010 (the app), ADR-011 (validation), ADR-008 (model).
""")

# --------------------------------------------------------------------------- calls
with tab_calls:
    days = sorted(df["appointment_day"].unique().tolist())
    left, right = st.columns([1, 2], gap="large")
    with left:
        day = st.selectbox("Appointment day", days, index=0, format_func=fmt_day)
        day_df = df[df["appointment_day"] == day]
        k = st.slider(
            "Calls the team can make today",
            min_value=1,
            max_value=int(min(50, len(day_df))),
            value=int(min(10, len(day_df))),
            help="How many reminder calls the team can make today. The list shows "
            "that many patients, highest risk first.",
        )
    ranked = rank_call_list(day_df.to_dict("records"), k)
    hits = sum(bool(r["actual_no_show"]) for r in ranked)
    base_rate = float(day_df["actual_no_show"].mean())
    hit_rate = precision_at_capacity(ranked)
    expected = round(float(sum(pd.DataFrame(ranked)["risk_score"])))

    with right:
        st.markdown(
            ui.bottom_line(
                "Who should the team call today?",
                f"Call these {len(ranked)} patients",
                "good",
                [f"About {expected} of them are likely to miss."],
                act="Work down the list from the top until the day's calls run out.",
                confidence="Moderate",
                confidence_why="Risk scores are calibrated, but one day of a "
                "demo sample is a small number of patients.",
                would_change="A different day or a different call capacity.",
            ),
            unsafe_allow_html=True,
        )
    with left:
        st.markdown(
            ui.tiles(
                [
                    ui.tile(
                        f"{hits} of {len(ranked)}",
                        "Would-be no-shows reached",
                        "good",
                        "In hindsight: how many of the people called really did not "
                        "show up.",
                    ),
                    ui.tile(
                        f"{hit_rate:.0%}",
                        "Hit rate vs calling at random",
                        "good" if hit_rate >= base_rate else "problem",
                        "Hit rate is the share of called patients who really missed. "
                        "Random calling would reach the day's overall no-show rate.",
                        sub=f"{(hit_rate - base_rate) * 100:+.0f} pts vs random",
                    ),
                ]
            ).replace("repeat(auto-fit, minmax(210px, 1fr))", "1fr"),
            unsafe_allow_html=True,
        )
        st.caption(
            f"Demo sample: {len(day_df)} of that day's appointments. "
            f"Random calling would reach about {base_rate:.0%}."
        )
    with right:
        with ui.card("calllist"):
            st.markdown(
                ui.card_title(f"Who to call first on {fmt_day(day)}"),
                unsafe_allow_html=True,
            )
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
            st.markdown(
                ui.takeaway(
                    f"In hindsight, {hits} of the {len(ranked)} patients called "
                    f"missed ({hit_rate:.0%}), against about {base_rate:.0%} "
                    "for calling at random."
                ),
                unsafe_allow_html=True,
            )
            st.caption(
                "'Actually missed?' is hindsight, shown only to judge the list. "
                "It is never a model input."
            )
        with st.expander("How we know"):
            st.markdown(
                "- **Risk:** a calibrated score, so 0.30 means about 30% likely to miss "
                "(ADR-008).\n"
                "- **Ranking:** highest risk first, then the top calls for the day.\n"
                "- **Hindsight column:** actual outcomes, shown only to judge the "
                "list, never used as a model input.\n"
                "- **Caveat:** the demo shows a sample of each day's appointments, "
                "so the hit rate on a single day is noisy."
            )

# --------------------------------------------------------------------------- why
with tab_why:
    shap_raw: dict[str, float] = {}
    if SHAP_JSON.exists():
        shap_raw = cast(
            "dict[str, float]",
            load_json(str(SHAP_JSON))["mean_abs_shap"],
        )
    order = sorted(shap_raw, key=lambda f: shap_raw[f], reverse=True)
    if len(order) >= 3:
        lead_ratio = shap_raw[order[0]] / shap_raw[order[1]]
        why_sentences = [
            f"{FEATURE_LABELS.get(order[0], order[0])} comes first, then "
            f"{FEATURE_LABELS.get(order[1], order[1]).lower()} and "
            f"{FEATURE_LABELS.get(order[2], order[2]).lower()}.",
            "Clinical conditions (diabetes, alcoholism, hypertension, disability) "
            "sit near zero (ADR-012).",
        ]
    else:
        lead_ratio = 0.0
        why_sentences = ["The factor ranking is not available in this deployment."]
    st.markdown(
        ui.bottom_line(
            "What drives no-show risk?",
            "Scheduling, not health conditions",
            "good",
            why_sentences,
            act="Shorten booking lead times or add reminders for long-lead bookings.",
            confidence="High",
            confidence_why="The ordering is consistent across the global SHAP ranking.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(ui.layer_label("Evidence"), unsafe_allow_html=True)
    if shap_raw:
        shap_df = pd.DataFrame(
            {
                "Factor": [FEATURE_LABELS.get(f, f) for f in shap_raw],
                "Average influence": list(shap_raw.values()),
            }
        )
        with ui.card("shap"):
            st.markdown(
                ui.card_title(
                    "What drives no-show risk overall",
                    "SHAP is a method that splits each prediction into the share "
                    "contributed by each factor. Longer bars mean the factor moves "
                    "predictions more, on average.",
                ),
                unsafe_allow_html=True,
            )
            st.altair_chart(
                ui.style_chart(
                    alt.Chart(shap_df)
                    .mark_bar(color=ui.ACCENT, cornerRadiusEnd=3)
                    .encode(
                        x=alt.X(
                            "Average influence:Q",
                            title="Average influence on risk",
                            axis=alt.Axis(tickCount=5, format=".2f"),
                        ),
                        y=alt.Y(
                            "Factor:N",
                            sort="-x",
                            title=None,
                            axis=alt.Axis(labelLimit=300),
                        ),
                        tooltip=[
                            "Factor",
                            alt.Tooltip("Average influence:Q", format=".3f"),
                        ],
                    ),
                    height=300,
                ),
                width="stretch",
                theme=None,
            )
            if lead_ratio:
                st.markdown(
                    ui.takeaway(
                        "Booking lead time moves predictions about "
                        f"{lead_ratio:.0f} times as much as the next factor, "
                        "age. Clinical conditions sit near zero (ADR-012)."
                    ),
                    unsafe_allow_html=True,
                )

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
    contrib_df = pd.DataFrame(
        {"factor": list(contribs), "effect": list(contribs.values())}
    )
    with ui.card("patient"):
        st.markdown(
            ui.card_title(
                "Why is this patient on the list?",
                "Bars to the right raise this patient's risk, bars to the left "
                "lower it (SHAP values, log-odds scale).",
            ),
            unsafe_allow_html=True,
        )
        st.altair_chart(
            ui.style_chart(
                alt.Chart(contrib_df)
                .mark_bar(cornerRadiusEnd=3)
                .encode(
                    y=alt.Y(
                        "factor:N", sort="-x", axis=alt.Axis(labelLimit=300), title=None
                    ),
                    x=alt.X("effect:Q", title="Push on risk (log-odds)"),
                    color=alt.condition(
                        "datum.effect > 0",
                        alt.value(ui.PROBLEM),
                        alt.value(ui.ACCENT),
                    ),
                    tooltip=[
                        alt.Tooltip("factor:N", title="Factor"),
                        alt.Tooltip("effect:Q", title="Push on risk", format="+.3f"),
                    ],
                ),
                height=300,
            ),
            width="stretch",
            theme=None,
        )
        st.markdown(
            ui.takeaway(
                f"Risk {row['risk_score']:.2f}. Biggest push up: {top_up}. "
                f"Biggest push down: {top_down}."
            ),
            unsafe_allow_html=True,
        )
    st.markdown(ui.layer_label("Method"), unsafe_allow_html=True)
    with st.expander("How we know"):
        st.markdown(
            "- **Method:** SHAP splits each prediction into a contribution per factor; "
            "the overall chart averages the size of those contributions over a "
            "sample of appointments.\n"
            "- **Scale:** contributions are in log-odds before calibration; "
            "calibration keeps the order of the scores.\n"
            "- **Caveat:** a factor that predicts no-shows is not necessarily a cause. "
            "SMS reminders are the example: they look harmful because the clinic "
            "sent them to long-lead bookings (ADR-006).\n"
            "- **Decision records:** ADR-012 (framing as an operations problem)."
        )

# --------------------------------------------------------------------------- fairness
with tab_fair:
    st.markdown(
        ui.bottom_line(
            "Does the list treat age groups fairly?",
            "No: patients 55+ are nearly left off",
            "problem",
            [
                f"Patients 55+ are {age55['share_of_no_shows']:.0%} of all no-shows "
                f"but only {age55['share_of_list']:.1%} of the call list.",
                "Their risk scores are honest, but their no-show rate is lower, so "
                "ranking by raw risk pushes them off a short list.",
            ],
            act="Review call capacity per age group with clinical and equity input "
            "before real use.",
            confidence="High for direction",
            confidence_why="The direction holds under every tie order; the exact "
            "percentages are noisy over 6 days.",
            would_change="A per-group capacity rule, or a longer test window.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(ui.layer_label("Evidence"), unsafe_allow_html=True)
    attr = st.selectbox(
        "Group by",
        list(cap_slices),
        index=list(cap_slices).index("age_band") if "age_band" in cap_slices else 0,
        format_func=lambda a: SLICE_LABELS.get(a, a),
        help="Age band is where the main finding is. Other groups look closer "
        "to even.",
    )
    cap_df = pd.DataFrame(cap_slices[attr]).T
    cap_df.index = pd.Index([group_label(g) for g in cap_df.index])
    coverage = cap_df["share_of_list"] / cap_df["share_of_no_shows"]
    worst = str(coverage.idxmin())
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
    with ui.card("fair"):
        st.markdown(
            ui.card_title(
                "Who makes the daily call list?",
                "This is the setting the app actually uses: the top "
                f"{capacity['k']} patients per day by risk, across all June "
                "appointments. Compare each group's share of the call list with its "
                "share of all no-shows. A big gap means the group is under-served "
                "by the list.",
            ),
            unsafe_allow_html=True,
        )
        st.altair_chart(
            ui.style_chart(
                alt.Chart(chart_df)
                .mark_bar(cornerRadiusEnd=3)
                .encode(
                    y=alt.Y(
                        "Group:N",
                        sort=group_order,
                        title=None,
                        axis=alt.Axis(labelLimit=300, labelAngle=0),
                    ),
                    yOffset=alt.YOffset("Measure:N"),
                    x=alt.X("Share:Q", title="Share", axis=alt.Axis(format="%")),
                    color=alt.Color(
                        "Measure:N",
                        title=None,
                        scale=alt.Scale(range=[ui.ACCENT, ui.ACCENT_SOFT]),
                    ),
                    tooltip=["Group", "Measure", alt.Tooltip("Share:Q", format=".1%")],
                ),
                height=max(180, 54 * len(group_order)),
            ),
            width="stretch",
            theme=None,
        )
        st.markdown(
            ui.takeaway(
                f"Most under-served: {worst} is {cap_df.loc[worst, 'share_of_no_shows']:.0%} "
                f"of no-shows and {cap_df.loc[worst, 'share_of_list']:.0%} of the "
                f"call list (top {capacity['k']} per day, {capacity['n_days']} days)."
            ),
            unsafe_allow_html=True,
        )
    st.markdown(
        ui.note(
            "Known limitation (ADR-013):",
            f"patients 55+ are {age55['share_of_no_shows']:.0%} of no-shows but only "
            f"{age55['share_of_list']:.1%} of the call list. Their risk scores are "
            "honest (well calibrated), but their no-show rate is lower, so ranking by "
            "raw risk pushes them off a short list. Before real use: review "
            "per-group capacity or thresholds with clinical and equity input.",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(ui.layer_label("Method"), unsafe_allow_html=True)
    with st.expander("How we know"):
        st.markdown(
            f"- **Setting:** top {capacity['k']} per day across all June appointments "
            f"({capacity['n_days']} days, {capacity['n_selected']} calls).\n"
            "- **Calibration:** risk scores are well calibrated in every group, so "
            "the gap comes from ranking, not from biased scores.\n"
            "- **Caveats:** six days is a short window, so exact percentages are "
            "noisy; the direction holds under every tie order.\n"
            "- **Decision records:** ADR-009 (fairness approach), ADR-013 (known "
            "limitation)."
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
            tbl = pd.DataFrame(groups).T[
                ["n", "base_rate", "selection_rate", "fpr", "fnr", "ece"]
            ]
            tbl.index = pd.Index([group_label(g) for g in tbl.index])
            for col in ("base_rate", "selection_rate", "fpr", "fnr"):
                tbl[col] = tbl[col].astype(float) * 100
            pct = st.column_config.NumberColumn(format="%.1f%%")
            st.dataframe(
                tbl.rename(
                    columns={
                        "n": "Appointments",
                        "base_rate": "No-show rate",
                        "selection_rate": "Flagged",
                        "fpr": "FPR",
                        "fnr": "FNR (missed)",
                        "ece": "Calibration error",
                    }
                ),
                width="stretch",
                column_config={
                    "No-show rate": pct,
                    "Flagged": pct,
                    "FPR": pct,
                    "FNR (missed)": pct,
                    "Calibration error": st.column_config.NumberColumn(format="%.3f"),
                },
            )

# --------------------------------------------------------------------------- impact
with tab_impact:
    slots_day = float(cast(float, impact["recoverable_slots_per_day"]))
    slots_month = float(cast(float, impact["recoverable_slots_per_month"]))
    calls_day = int(float(cast(float, impact["daily_call_capacity"])))
    st.markdown(
        ui.bottom_line(
            f"What could {calls_day} calls a day recover?",
            f"About {slots_day:.0f} slots a day, at most",
            "caution",
            [
                f"{calls_day} calls a day at a {precision_k:.0%} hit rate reach about "
                f"{slots_day:.1f} would-be no-shows a day, or about "
                f"{slots_month:.0f} a month.",
                "This assumes every reached patient keeps or rebooks the slot, so "
                "real recovery is lower.",
            ],
            act="Measure real rebooking after calls before trusting the dollar figure.",
            confidence="Low to moderate",
            confidence_why="The slot count is an upper bound, and the dollar value "
            "is an editable literature estimate, not clinic data.",
            would_change="Measured rebooking rates, or the clinic's own cost of a "
            "missed slot.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(ui.layer_label("Evidence"), unsafe_allow_html=True)
    st.markdown(
        ui.tiles(
            [
                ui.tile(
                    f"{precision_k:.0%}",
                    "Hit rate on the list",
                    "good",
                    "Share of the called patients who really did not show up. It "
                    "ranges from 43 to 45% depending on how tied scores are ordered.",
                ),
                ui.tile(
                    f"~{slots_day:.1f}",
                    "Appointment slots per day",
                    "caution",
                    "Slots per day = calls per day times the hit rate. A slot is "
                    "one appointment time that could be refilled.",
                ),
                ui.tile(
                    f"~{slots_month:.0f}",
                    "Appointment slots per month",
                    "caution",
                    "Slots per day times the working days in a month.",
                ),
            ]
        ),
        unsafe_allow_html=True,
    )
    with ui.card("value"):
        st.markdown(
            ui.card_title(
                "What a 20-call list could recover (a labeled what-if)",
                "Default USD 200 is a US primary-care literature estimate, not "
                "clinic data.",
            ),
            unsafe_allow_html=True,
        )
        cost = st.number_input(
            "Value of one recovered slot (edit to your setting)",
            min_value=0.0,
            value=float(cast(float, impact["cost_per_noshow"])),
            step=25.0,
            help="Default USD 200 is a US primary-care literature estimate, not "
            "clinic data.",
        )
        st.markdown(
            ui.tiles(
                [
                    ui.tile(
                        f"${slots_month * cost:,.0f}",
                        "Monthly value at that rate",
                        "caution",
                    )
                ]
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            ui.takeaway(
                f"At ${cost:,.0f} per slot, the upper bound is about "
                f"${slots_month * cost:,.0f} a month; real conversion is lower."
            ),
            unsafe_allow_html=True,
        )
    st.markdown(ui.layer_label("Method"), unsafe_allow_html=True)
    with st.expander("How we know"):
        st.markdown(
            "- **Upper bound:** assumes every correctly flagged, called patient keeps "
            "or rebooks their slot; real conversion is lower.\n"
            f"- **Hit rate:** measured on the top {calls_day} patients per day in the "
            f"June 2016 test data ({capacity['n_days']} days, "
            f"{capacity['n_selected']} calls).\n"
            "- **Dollar value:** a US primary-care literature estimate, editable "
            "above, not clinic data.\n"
            "- **Decision records:** ADR-004 (lead with slots, not dollars)."
        )

# --------------------------------------------------------------------------- how
with tab_how:
    st.markdown(
        ui.bottom_line(
            "Can we trust these numbers?",
            "Yes, within stated limits",
            "good",
            [
                "Models were validated with a split by patient, so no one appears in "
                "both training and test data, and then tested on a later month.",
                "A lead-time bug was found and fixed along the way (ADR-014).",
            ],
            act="Read the limits on each tab before relying on any single figure.",
            confidence="Moderate",
            confidence_why="Public data from one city and a six-day test window.",
        ),
        unsafe_allow_html=True,
    )
    with ui.card("decisions"):
        st.markdown(
            ui.card_title(
                "From raw data to this app: every decision and why",
                "ADR = architecture decision record, a short note explaining one "
                "decision. All of them are in the GitHub repo.",
            ),
            unsafe_allow_html=True,
        )
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
        st.markdown(
            ui.takeaway(
                "Every row pairs what the data showed with the decision it led to."
            ),
            unsafe_allow_html=True,
        )
    st.markdown(
        f"Full visual decision tree: [docs/DECISION_TREE.md]({DECISION_TREE_URL})"
    )
    st.markdown(f"Code, tests and ADRs: [{REPO_URL}]({REPO_URL})")
    st.markdown(ui.layer_label("Method"), unsafe_allow_html=True)
    with st.expander("How we know"):
        st.markdown(
            "- **Split design:** grouped by patient (39% of patients book more than "
            "once), then a later-month test (ADR-011).\n"
            "- **Data fix:** 35% of lead times looked negative; they were same-day "
            "bookings, so gaps are now counted in calendar days (ADR-014).\n"
            "- **Caveats:** public data from one city in 2016, no real call outcomes.\n"
            "- **Decision records:** all ADRs are in the GitHub repository."
        )
