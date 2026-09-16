"""
Maintenance Task Predictor — model integration test app.

The purpose of this app is to TEST whether the saved model can be used from
outside the notebook: is the bundle complete, is the input contract correct,
and do the results make sense.

Run it from the project folder:

    conda activate maintenance-ml
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

MODEL_PATH = Path("models/maintenance_task_model.joblib")
META_PATH = Path("models/model_metadata.json")

TRUTHY = {"Y", "YES", "1", "TRUE", "T", "YA"}

# Confidence thresholds — below these, a result is flagged for human review.
CONF_HIGH = 0.80
CONF_MID = 0.50

# Friendlier display labels for each target
TARGET_LABEL = {
    "PROPOSED_TASK": "Proposed Task",
    "DESC_PROPOSED_TASK": "Task Description",
    "INITIAL_INTERVAL": "Initial Interval",
}

# Explanations for the boolean codes, so non-technical users know what they mean
FLAG_GROUPS = {
    "Consequence": {
        "cols": ["H", "S", "E", "O"],
        "help": "Consequence evaluation: Hidden, Safety, Environment, Operational.",
    },
    "Operator view": {
        "cols": [
            "OP_H1", "OP_H2", "OP_H3", "OP_H4", "OP_H5",
            "OP_S1", "OP_S2", "OP_S3", "OP_S4",
            "OP_O1", "OP_O2", "OP_O3",
            "OP_N1", "OP_N2", "OP_N3",
        ],
        "help": "Default action from the operator's point of view.",
    },
    "Expert view": {
        "cols": [
            "EX_H1", "EX_H2", "EX_H3", "EX_H4", "EX_H5",
            "EX_S1", "EX_S2", "EX_S3", "EX_S4",
            "EX_O1", "EX_O2", "EX_O3",
            "EX_N1", "EX_N2", "EX_N3",
        ],
        "help": "Default action from the expert's point of view.",
    },
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_bundle(path: Path, mtime: float):
    """Load the model bundle. mtime is part of the cache key so a retrained model is picked up."""
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_meta(path: Path, mtime: float) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def known_categories(_bundle, cache_key: str) -> dict[str, list[str]]:
    """Get the categories the encoder knows about, per categorical column.

    A value outside this list is not an error — the encoder uses
    handle_unknown="ignore" — but the model knows nothing about it.
    """
    first = _bundle["models"][_bundle["target_cols"][0]]
    ohe = first.named_steps["prep"].named_transformers_["cat"]
    out = {}
    for col, cats in zip(_bundle["cat_cols"], ohe.categories_):
        vals = [str(c) for c in cats if not str(c).endswith("infrequent_sklearn")]
        out[col] = sorted(vals)
    return out


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------


def build_frame(records: list[dict], bundle) -> pd.DataFrame:
    """Build a DataFrame matching the model contract: all columns, correct order."""
    frame = pd.DataFrame(records)

    for col in bundle["cat_cols"]:
        if col not in frame:
            frame[col] = ""
        frame[col] = (
            frame[col].fillna("").astype(str)
            .str.strip().str.replace(r"\s+", " ", regex=True)
        )

    for col in bundle["bool_cols"]:
        if col not in frame:
            frame[col] = 0
        frame[col] = (
            frame[col].fillna(0).astype(str).str.strip().str.upper()
            .isin(TRUTHY).astype(int)
        )

    return frame[bundle["feature_cols"]]


def predict(records: list[dict], bundle) -> pd.DataFrame:
    frame = build_frame(records, bundle)

    out = {}
    for target in bundle["target_cols"]:
        model = bundle["models"][target]
        out[target] = model.predict(frame)
        if hasattr(model, "predict_proba"):
            out[f"{target}__conf"] = model.predict_proba(frame).max(axis=1)

    return pd.DataFrame(out)


def conf_bucket(value: float) -> tuple[str, str]:
    """Return (label, colour) for a confidence value."""
    if value >= CONF_HIGH:
        return "Confident", "normal"
    if value >= CONF_MID:
        return "Uncertain", "off"
    return "Guessing", "inverse"


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Maintenance Task Predictor",
    page_icon="🔧",
    layout="wide",
)

st.title("🔧 Maintenance Task Predictor")
st.caption("Integration test for the saved model — verifies the bundle works outside the notebook.")

# --- Guard: the model must exist ---
if not MODEL_PATH.exists():
    st.error(f"Model not found: `{MODEL_PATH}`")
    st.markdown(
        "Run `train_model.ipynb` (sections 1-10) first. It trains the model "
        "and saves it to the `models/` folder."
    )
    st.stop()

try:
    bundle = load_bundle(MODEL_PATH, MODEL_PATH.stat().st_mtime)
except Exception as exc:  # noqa: BLE001 - surface whatever went wrong
    st.error("The model file exists but could not be loaded.")
    st.exception(exc)
    st.stop()

meta = load_meta(META_PATH, META_PATH.stat().st_mtime) if META_PATH.exists() else {}
CATS = known_categories(bundle, str(MODEL_PATH.stat().st_mtime))

# ==========================================================================
# Sidebar — model status
# ==========================================================================

with st.sidebar:
    st.subheader("Model status")
    st.success("Loaded successfully")

    size_mb = MODEL_PATH.stat().st_size / 1_048_576
    st.caption(f"`{MODEL_PATH.name}` · {size_mb:.1f} MB")

    trained_at = meta.get("created") or meta.get("dibuat")
    if trained_at:
        st.caption(f"Trained: {trained_at.replace('T', ' ')}")

    st.divider()
    st.markdown("**Estimators**")
    for target in bundle["target_cols"]:
        clf = bundle["models"][target].named_steps["clf"]
        st.caption(f"{TARGET_LABEL.get(target, target)} — `{type(clf).__name__}`")

    test_scores = meta.get("test_scores") or meta.get("skor_test")
    if test_scores:
        st.divider()
        st.markdown("**Test scores**")
        for row in test_scores:
            label = TARGET_LABEL.get(row["target"], row["target"])
            st.caption(f"{label} — acc {row['accuracy']:.3f} · F1 {row['f1_macro']:.3f}")

    # Metadata key names differ between notebook versions, so accept either.
    n_groups = meta.get("n_unique_groups") or meta.get("n_grup_unik")
    n_rows = meta.get("n_rows") or meta.get("n_baris")
    if n_groups and n_rows:
        st.divider()
        st.warning(
            f"Trained on **{n_groups} distinct cases** "
            f"(from {n_rows} rows — many are duplicates). "
            "Treat every output as a suggestion, not a decision.",
            icon="⚠️",
        )

# ==========================================================================
# Tabs
# ==========================================================================

tab_single, tab_batch, tab_contract = st.tabs(
    ["Single prediction", "Batch from file", "Model contract"]
)

# --------------------------------------------------------------------------
# Tab 1 — single prediction
# --------------------------------------------------------------------------

with tab_single:
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Failure description")

        record: dict[str, object] = {}
        unknown_values: list[str] = []

        for col in bundle["cat_cols"]:
            options = CATS.get(col, [])
            label = col.replace("_", " ").title()

            choice = st.selectbox(
                label,
                options=options + ["── type a new value ──"],
                index=0 if options else None,
                key=f"sel_{col}",
            )

            if choice == "── type a new value ──":
                typed = st.text_input(
                    f"{label} (custom)",
                    key=f"txt_{col}",
                    placeholder="Value not seen during training",
                )
                record[col] = typed
                if typed.strip():
                    unknown_values.append(f"{label}: {typed.strip()}")
            else:
                record[col] = choice

        st.subheader("Assessment flags")
        st.caption("Tick what applies. Anything unticked counts as N.")

        for group_name, spec in FLAG_GROUPS.items():
            with st.expander(group_name, expanded=(group_name == "Consequence")):
                st.caption(spec["help"])
                cols_ui = st.columns(4)
                for i, flag in enumerate(spec["cols"]):
                    with cols_ui[i % 4]:
                        ticked = st.checkbox(
                            flag.replace("OP_", "").replace("EX_", ""),
                            key=f"flag_{flag}",
                        )
                        record[flag] = "Y" if ticked else "N"

    with right:
        st.subheader("Prediction")

        if unknown_values:
            st.info(
                "**Unseen category values**\n\n"
                + "\n".join(f"- {v}" for v in unknown_values)
                + "\n\nThe model has no information about these. "
                "It will still answer — check the confidence carefully.",
                icon="ℹ️",
            )

        run = st.button("Predict", type="primary", width="stretch")

        if run:
            try:
                result = predict([record], bundle)
            except Exception as exc:  # noqa: BLE001
                st.error("Prediction failed.")
                st.exception(exc)
            else:
                row = result.iloc[0]
                low_count = 0

                for target in bundle["target_cols"]:
                    conf = row.get(f"{target}__conf")
                    label = TARGET_LABEL.get(target, target)

                    st.markdown(f"**{label}**")
                    st.markdown(f"### {row[target]}")

                    if conf is not None and pd.notna(conf):
                        bucket, delta_color = conf_bucket(float(conf))
                        st.metric(
                            "Confidence",
                            f"{conf:.1%}",
                            delta=bucket,
                            delta_color=delta_color,
                            label_visibility="collapsed",
                        )
                        st.progress(min(float(conf), 1.0))
                        if conf < CONF_MID:
                            low_count += 1
                    st.divider()

                if low_count:
                    st.warning(
                        f"{low_count} of {len(bundle['target_cols'])} predictions "
                        f"are below {CONF_MID:.0%} confidence. Have an engineer review those.",
                        icon="⚠️",
                    )
                else:
                    st.success("All predictions above the review threshold.", icon="✅")
        else:
            st.caption("Fill the fields on the left, then press **Predict**.")

# --------------------------------------------------------------------------
# Tab 2 — batch
# --------------------------------------------------------------------------

with tab_batch:
    st.subheader("Predict many rows at once")
    st.markdown(
        "Upload a CSV or Excel file. Missing columns are filled with defaults "
        "(empty text, flags set to N), so a partial file still works."
    )

    template = pd.DataFrame([{c: "" for c in bundle["cat_cols"]}])
    for col in bundle["bool_cols"]:
        template[col] = "N"

    st.download_button(
        "Download blank template (CSV)",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="prediction_template.csv",
        mime="text/csv",
    )

    upload = st.file_uploader("Input file", type=["csv", "xlsx", "xls"])

    if upload is not None:
        try:
            if upload.name.lower().endswith(".csv"):
                data = pd.read_csv(upload)
            else:
                data = pd.read_excel(upload)
        except Exception as exc:  # noqa: BLE001
            st.error("Could not read that file.")
            st.exception(exc)
        else:
            st.caption(f"{len(data)} rows · {len(data.columns)} columns")

            missing = [
                c for c in bundle["cat_cols"] + bundle["bool_cols"]
                if c not in data.columns
            ]
            if missing:
                st.info(
                    f"{len(missing)} expected columns are absent and will use defaults: "
                    + ", ".join(f"`{c}`" for c in missing[:8])
                    + (" …" if len(missing) > 8 else ""),
                    icon="ℹ️",
                )

            if st.button("Run batch prediction", type="primary"):
                try:
                    preds = predict(data.to_dict("records"), bundle)
                except Exception as exc:  # noqa: BLE001
                    st.error("Batch prediction failed.")
                    st.exception(exc)
                else:
                    conf_cols = [
                        f"{t}__conf" for t in bundle["target_cols"]
                        if f"{t}__conf" in preds
                    ]
                    flagged = (
                        (preds[conf_cols] < CONF_MID).any(axis=1).sum()
                        if conf_cols else 0
                    )

                    m1, m2 = st.columns(2)
                    m1.metric("Rows predicted", len(preds))
                    m2.metric("Need review", int(flagged))

                    combined = pd.concat(
                        [data.reset_index(drop=True), preds], axis=1
                    )
                    st.dataframe(combined, width="stretch", height=340)

                    st.download_button(
                        "Download results (CSV)",
                        data=combined.to_csv(index=False).encode("utf-8"),
                        file_name="prediction_results.csv",
                        mime="text/csv",
                        type="primary",
                    )

                    if flagged:
                        st.warning(
                            f"{flagged} rows have at least one prediction below "
                            f"{CONF_MID:.0%} confidence.",
                            icon="⚠️",
                        )

# --------------------------------------------------------------------------
# Tab 3 — model contract
# --------------------------------------------------------------------------

with tab_contract:
    st.subheader("What this model expects and returns")
    st.markdown(
        "This tab exists so integration problems are diagnosable. Everything "
        "below is read from the saved bundle, not hard-coded in this app."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Feature columns", len(bundle["feature_cols"]))
    c2.metric("Targets", len(bundle["target_cols"]))
    c3.metric("Boolean flags", len(bundle["bool_cols"]))

    st.markdown("#### Categorical inputs")
    st.dataframe(
        pd.DataFrame(
            [
                {"Column": c, "Known values": len(CATS.get(c, []))}
                for c in bundle["cat_cols"]
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    picked = st.selectbox("Inspect known values for", bundle["cat_cols"])
    st.caption(
        f"{len(CATS.get(picked, []))} values seen during training. "
        "Anything else is accepted but carries no learned signal."
    )
    st.dataframe(
        pd.DataFrame({picked: CATS.get(picked, [])}),
        width="stretch",
        hide_index=True,
        height=240,
    )

    st.markdown("#### Boolean flags")
    st.code(", ".join(bundle["bool_cols"]), language=None)
    st.caption('Accepted as "Y"/"N", 1/0, or true/false. Absent flags default to N.')

    st.markdown("#### Outputs")
    for target in bundle["target_cols"]:
        model = bundle["models"][target]
        clf = model.named_steps["clf"]
        n_classes = len(getattr(clf, "classes_", []))
        st.markdown(
            f"- **{TARGET_LABEL.get(target, target)}** — "
            f"`{type(clf).__name__}`, {n_classes} possible values"
        )

    if meta:
        with st.expander("Raw metadata"):
            st.json(meta)
