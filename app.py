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


# ==========================================================================
# Constants
# ==========================================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_OPTIONS = {
    "V1 — Clustering": {
        "model": BASE_DIR / "models" / "V1 clustering" / "models V1" / "maintenance_task_model.joblib",
        "meta": BASE_DIR / "models" / "V1 clustering" / "models V1" / "model_metadata.json",
    },
    "V2 — Reduced data": {
        "model": BASE_DIR / "models" / "V2 reduced DATA" / "models" / "maintenance_task_model.joblib",
        "meta": BASE_DIR / "models" / "V2 reduced DATA" / "models" / "model_metadata.json",
    },
}
TRUTHY = {"Y", "YES", "1", "TRUE", "T", "YA"}

# Human-readable labels for model targets.
# If the model contains another target, the raw target name is used automatically.
TARGET_LABEL = {
    "PROPOSED_TASK": "Proposed task",
    "DESC_PROPOSED_TASK": "Task description",
}

# Confidence thresholds.
CONF_HIGH = 0.80
CONF_MID = 0.60


# ==========================================================================
# Loading
# ==========================================================================

@st.cache_resource(show_spinner=False)
def load_bundle(path: Path, mtime: float):
    """
    Load the model bundle.

    mtime is part of the cache key so that a newly trained model is picked up.
    """
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_meta(path: Path, mtime: float) -> dict:
    """Load model metadata."""
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def known_categories(_bundle, cache_key: str) -> dict[str, list[str]]:
    """
    Get the categories the encoder knows about, per categorical column.

    Columns processed through OneHotEncoder have a fixed category list.
    Text columns processed through TF-IDF accept free text and therefore
    return an empty list.
    """
    first_target = _bundle["target_cols"][0]
    first_model = _bundle["models"][first_target]

    prep = first_model.named_steps["prep"]
    ohe = prep.named_transformers_["cat"]

    out: dict[str, list[str]] = {}

    # The training pipeline currently uses these two columns as one-hot
    # categorical variables.
    one_hot_cols = ["OBJECT_TYPE", "COMPONEN"]

    for col in _bundle["cat_cols"]:
        if col in one_hot_cols:
            try:
                idx = one_hot_cols.index(col)
                vals = [
                    str(c)
                    for c in ohe.categories_[idx]
                    if not str(c).endswith("infrequent_sklearn")
                ]
                out[col] = sorted(vals)
            except (IndexError, AttributeError):
                out[col] = []
        else:
            # Free-text / TF-IDF column.
            out[col] = []

    return out


# ==========================================================================
# Inference
# ==========================================================================

def build_frame(records: list[dict], bundle) -> pd.DataFrame:
    """
    Build a DataFrame matching the model contract:
    all required columns, correct order and normalized values.
    """
    frame = pd.DataFrame(records)

    # Categorical / text columns
    for col in bundle["cat_cols"]:
        if col not in frame.columns:
            frame[col] = ""

        frame[col] = (
            frame[col]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    # Boolean flags
    for col in bundle["bool_cols"]:
        if col not in frame.columns:
            frame[col] = 0

        frame[col] = (
            frame[col]
            .fillna(0)
            .astype(str)
            .str.strip()
            .str.upper()
            .isin(TRUTHY)
            .astype(int)
        )

    # Return exactly the columns expected by the model.
    return frame[bundle["feature_cols"]]


def predict(records: list[dict], bundle) -> pd.DataFrame:
    """Run all target models and return predictions + confidence."""
    frame = build_frame(records, bundle)

    out = {}

    for target in bundle["target_cols"]:
        model = bundle["models"][target]

        # DESC_PROPOSED_TASK uses PROPOSED_TASK prediction as an extra input.
        if target == "DESC_PROPOSED_TASK":
            frame_t = frame.copy()
            frame_t["PROPOSED_TASK_hint"] = (
                bundle["models"]["PROPOSED_TASK"].predict(frame)
            )
        else:
            frame_t = frame

        out[target] = model.predict(frame_t)

        if hasattr(model, "predict_proba"):
            out[f"{target}__conf"] = (
                model.predict_proba(frame_t).max(axis=1)
            )

    return pd.DataFrame(out)


def conf_bucket(value: float) -> tuple[str, str]:
    """Return (label, Streamlit delta color) for a confidence value."""
    if value >= CONF_HIGH:
        return "Confident", "normal"

    if value >= CONF_MID:
        return "Uncertain", "off"

    return "Guessing", "inverse"


# ==========================================================================
# UI configuration
# ==========================================================================

st.set_page_config(
    page_title="Maintenance Task Predictor",
    page_icon="🔧",
    layout="wide",
)


# ==========================================================================
# Sidebar — model selection
# ==========================================================================

with st.sidebar:
    st.subheader("Model version")

    model_choice = st.radio(
        "Choose which trained model to use",
        options=list(MODEL_OPTIONS.keys()),
        key="model_choice",
    )


MODEL_PATH = MODEL_OPTIONS[model_choice]["model"]
META_PATH = MODEL_OPTIONS[model_choice]["meta"]


# ==========================================================================
# Main title
# ==========================================================================

st.title("🔧 Maintenance Task Predictor")

st.caption(
    f"Integration test for the saved model — currently using **{model_choice}**."
)


# ==========================================================================
# Guard — model existence
# ==========================================================================

if not MODEL_PATH.exists():
    st.error(f"Model not found: `{MODEL_PATH}`")

    st.markdown(
        "Run `train_model.ipynb` (sections 1–10) first. "
        "It trains the model and saves it to the `models/` folder."
    )

    st.stop()


# ==========================================================================
# Load model
# ==========================================================================

try:
    bundle = load_bundle(
        MODEL_PATH,
        MODEL_PATH.stat().st_mtime,
    )

except Exception as exc:
    st.error("The model file exists but could not be loaded.")
    st.exception(exc)
    st.stop()


# ==========================================================================
# Validate bundle structure
# ==========================================================================

required_keys = {
    "models",
    "target_cols",
    "cat_cols",
    "bool_cols",
    "feature_cols",
}

missing_keys = required_keys - set(bundle.keys())

if missing_keys:
    st.error(
        "The model bundle is incomplete. "
        f"Missing keys: {', '.join(sorted(missing_keys))}"
    )
    st.stop()


# ==========================================================================
# Load metadata
# ==========================================================================

if META_PATH.exists():
    try:
        meta = load_meta(
            META_PATH,
            META_PATH.stat().st_mtime,
        )
    except Exception as exc:
        st.warning("The metadata file exists but could not be read.")
        st.exception(exc)
        meta = {}
else:
    meta = {}


# ==========================================================================
# Build category information
# ==========================================================================

CATS = known_categories(
    bundle,
    str(MODEL_PATH.stat().st_mtime),
)


# ==========================================================================
# Build flag groups AFTER bundle exists
# ==========================================================================

FLAG_GROUPS = {
    "Assessment flags": {
        "help": (
            "Tick the flags that apply. "
            "Anything unticked is treated as N."
        ),
        "cols": bundle["bool_cols"],
    }
}


# ==========================================================================
# Sidebar — model status
# ==========================================================================

with st.sidebar:
    st.subheader("Model status")

    st.success("Loaded successfully")

    size_mb = MODEL_PATH.stat().st_size / 1_048_576

    st.caption(
        f"`{MODEL_PATH.name}` · {size_mb:.1f} MB"
    )

    trained_at = meta.get("created") or meta.get("dibuat")

    if trained_at:
        st.caption(
            f"Trained: {str(trained_at).replace('T', ' ')}"
        )

    st.divider()

    st.markdown("**Estimators**")

    for target in bundle["target_cols"]:
        model = bundle["models"][target]

        try:
            clf = model.named_steps["clf"]
            clf_name = type(clf).__name__
        except (AttributeError, KeyError):
            clf_name = type(model).__name__

        st.caption(
            f"{TARGET_LABEL.get(target, target)} — `{clf_name}`"
        )

    test_scores = (
        meta.get("test_scores")
        or meta.get("skor_test")
    )

    if test_scores:
        st.divider()
        st.markdown("**Test scores**")

        for row in test_scores:
            target = row.get("target", "Unknown")

            label = TARGET_LABEL.get(
                target,
                target,
            )

            accuracy = row.get("accuracy")
            f1_macro = row.get("f1_macro")

            if accuracy is not None and f1_macro is not None:
                st.caption(
                    f"{label} — "
                    f"acc {accuracy:.3f} · "
                    f"F1 {f1_macro:.3f}"
                )

    n_groups = (
        meta.get("n_unique_groups")
        or meta.get("n_grup_unik")
    )

    n_rows = (
        meta.get("n_rows")
        or meta.get("n_baris")
    )

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
    [
        "Single prediction",
        "Batch from file",
        "Model contract",
    ]
)


# ==========================================================================
# Tab 1 — Single prediction
# ==========================================================================

with tab_single:

    left, right = st.columns(
        [3, 2],
        gap="large",
    )

    # ----------------------------------------------------------------------
    # Input
    # ----------------------------------------------------------------------

    with left:

        st.subheader("Failure description")

        record: dict[str, object] = {}

        unknown_values: list[str] = []

        for col in bundle["cat_cols"]:

            options = CATS.get(col, [])

            label = col.replace("_", " ").title()

            if not options:

                typed = st.text_input(
                    label,
                    key=f"txt_{col}",
                )

                record[col] = typed

            else:

                choice = st.selectbox(
                    label,
                    options=options + [
                        "── type a new value ──"
                    ],
                    index=0,
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
                        unknown_values.append(
                            f"{label}: {typed.strip()}"
                        )

                else:

                    record[col] = choice

        st.subheader("Assessment flags")

        st.caption(
            "Tick what applies. Anything unticked counts as N."
        )

        for group_name, spec in FLAG_GROUPS.items():

            with st.expander(
                group_name,
                expanded=True,
            ):

                st.caption(spec["help"])

                flags = spec["cols"]

                if flags:

                    cols_ui = st.columns(4)

                    for i, flag in enumerate(flags):

                        with cols_ui[i % 4]:

                            ticked = st.checkbox(
                                flag.replace("OP_", "")
                                .replace("EX_", ""),
                                key=f"flag_{flag}",
                            )

                            record[flag] = (
                                "Y" if ticked else "N"
                            )

                else:

                    st.info(
                        "This model has no boolean assessment flags."
                    )


    # ----------------------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------------------

    with right:

        st.subheader("Prediction")

        if unknown_values:

            st.info(
                "**Unseen category values**\n\n"
                + "\n".join(
                    f"- {value}"
                    for value in unknown_values
                )
                + "\n\n"
                "The model has no information about these. "
                "It will still answer — check the confidence carefully.",
                icon="ℹ️",
            )

        run = st.button(
            "Predict",
            type="primary",
            width="stretch",
        )

        if run:

            try:

                result = predict(
                    [record],
                    bundle,
                )

            except Exception as exc:

                st.error("Prediction failed.")
                st.exception(exc)

            else:

                row = result.iloc[0]

                low_count = 0

                for target in bundle["target_cols"]:

                    conf = row.get(
                        f"{target}__conf"
                    )

                    label = TARGET_LABEL.get(
                        target,
                        target,
                    )

                    st.markdown(
                        f"**{label}**"
                    )

                    st.markdown(
                        f"### {row[target]}"
                    )

                    if conf is not None and pd.notna(conf):

                        conf = float(conf)

                        bucket, delta_color = conf_bucket(
                            conf
                        )

                        st.metric(
                            "Confidence",
                            f"{conf:.1%}",
                            delta=bucket,
                            delta_color=delta_color,
                            label_visibility="collapsed",
                        )

                        st.progress(
                            min(conf, 1.0)
                        )

                        if conf < CONF_MID:
                            low_count += 1

                    st.divider()

                if low_count:

                    st.warning(
                        f"{low_count} of "
                        f"{len(bundle['target_cols'])} predictions "
                        f"are below {CONF_MID:.0%} confidence. "
                        "Have an engineer review those.",
                        icon="⚠️",
                    )

                else:

                    st.success(
                        "All predictions above the review threshold.",
                        icon="✅",
                    )

        else:

            st.caption(
                "Fill the fields on the left, "
                "then press **Predict**."
            )


# ==========================================================================
# Tab 2 — Batch prediction
# ==========================================================================

with tab_batch:

    st.subheader("Predict many rows at once")

    st.markdown(
        "Upload a CSV or Excel file. Missing columns are filled "
        "with defaults (empty text, flags set to N), so a partial "
        "file still works."
    )

    template = pd.DataFrame(
        [
            {
                c: ""
                for c in bundle["cat_cols"]
            }
        ]
    )

    for col in bundle["bool_cols"]:
        template[col] = "N"

    st.download_button(
        "Download blank template (CSV)",
        data=template.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="prediction_template.csv",
        mime="text/csv",
    )

    upload = st.file_uploader(
        "Input file",
        type=["csv", "xlsx", "xls"],
    )

    if upload is not None:

        try:

            if upload.name.lower().endswith(".csv"):
                data = pd.read_csv(upload)
            else:
                data = pd.read_excel(upload)

        except Exception as exc:

            st.error("Could not read that file.")
            st.exception(exc)

        else:

            st.caption(
                f"{len(data)} rows · "
                f"{len(data.columns)} columns"
            )

            missing = [
                c
                for c in (
                    bundle["cat_cols"]
                    + bundle["bool_cols"]
                )
                if c not in data.columns
            ]

            if missing:

                st.info(
                    f"{len(missing)} expected columns are absent "
                    "and will use defaults: "
                    + ", ".join(
                        f"`{c}`"
                        for c in missing[:8]
                    )
                    + (
                        " …"
                        if len(missing) > 8
                        else ""
                    ),
                    icon="ℹ️",
                )

            if st.button(
                "Run batch prediction",
                type="primary",
            ):

                try:

                    preds = predict(
                        data.to_dict("records"),
                        bundle,
                    )

                except Exception as exc:

                    st.error(
                        "Batch prediction failed."
                    )
                    st.exception(exc)

                else:

                    conf_cols = [
                        f"{t}__conf"
                        for t in bundle["target_cols"]
                        if f"{t}__conf" in preds.columns
                    ]

                    if conf_cols:

                        flagged = int(
                            (
                                preds[conf_cols] < CONF_MID
                            )
                            .any(axis=1)
                            .sum()
                        )

                    else:

                        flagged = 0

                    m1, m2 = st.columns(2)

                    m1.metric(
                        "Rows predicted",
                        len(preds),
                    )

                    m2.metric(
                        "Need review",
                        flagged,
                    )

                    combined = pd.concat(
                        [
                            data.reset_index(drop=True),
                            preds,
                        ],
                        axis=1,
                    )

                    st.dataframe(
                        combined,
                        width="stretch",
                        height=340,
                    )

                    st.download_button(
                        "Download results (CSV)",
                        data=combined.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="prediction_results.csv",
                        mime="text/csv",
                        type="primary",
                    )

                    if flagged:

                        st.warning(
                            f"{flagged} rows have at least one "
                            f"prediction below "
                            f"{CONF_MID:.0%} confidence.",
                            icon="⚠️",
                        )


# ==========================================================================
# Tab 3 — Model contract
# ==========================================================================

with tab_contract:

    st.subheader(
        "What this model expects and returns"
    )

    st.markdown(
        "This tab exists so integration problems are diagnosable. "
        "Everything below is read from the saved bundle, "
        "not hard-coded in this app."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Feature columns",
        len(bundle["feature_cols"]),
    )

    c2.metric(
        "Targets",
        len(bundle["target_cols"]),
    )

    c3.metric(
        "Boolean flags",
        len(bundle["bool_cols"]),
    )

    st.markdown("#### Categorical inputs")

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Column": c,
                    "Known values": len(
                        CATS.get(c, [])
                    ),
                }
                for c in bundle["cat_cols"]
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    picked = st.selectbox(
        "Inspect known values for",
        bundle["cat_cols"],
    )

    st.caption(
        f"{len(CATS.get(picked, []))} values seen during training. "
        "Anything else is accepted but carries no learned signal."
    )

    st.dataframe(
        pd.DataFrame(
            {
                picked: CATS.get(picked, [])
            }
        ),
        width="stretch",
        hide_index=True,
        height=240,
    )

    st.markdown("#### Boolean flags")

    if bundle["bool_cols"]:

        st.code(
            ", ".join(bundle["bool_cols"]),
            language=None,
        )

        st.caption(
            'Accepted as "Y"/"N", 1/0, or true/false. '
            "Absent flags default to N."
        )

    else:

        st.info(
            "This model does not contain boolean flags."
        )

    st.markdown("#### Outputs")

    for target in bundle["target_cols"]:

        model = bundle["models"][target]

        try:

            clf = model.named_steps["clf"]

            clf_name = type(clf).__name__

            n_classes = len(
                getattr(
                    clf,
                    "classes_",
                    [],
                )
            )

        except (AttributeError, KeyError):

            clf_name = type(model).__name__
            n_classes = 0

        st.markdown(
            f"- **{TARGET_LABEL.get(target, target)}** — "
            f"`{clf_name}`, "
            f"{n_classes} possible values"
        )

    if meta:

        with st.expander("Raw metadata"):

            st.json(meta)
