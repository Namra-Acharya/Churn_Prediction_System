import json
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from ui_helpers import (
    CATEGORICAL_OPTIONS,
    FEATURE_COLUMNS,
    FIELD_LABELS,
    FORM_SECTIONS,
    INT_COLUMNS,
    NUMERIC_BOUNDS,
    batch_template_csv,
    validate_batch_frame,
)

st.set_page_config(page_title="ChurnIQ — Customer Intelligence", layout="wide")

PAGES = ["Overview", "Predict churn", "Batch predictions", "Model information"]
BATCH_SIZE = 100
DEFAULTS = {
    "CustomerAge": 35,
    "TenureMonths": 12,
    "MonthlyCharges": 70.0,
    "SupportTickets": 2,
    "UsageHours": 25.0,
}
CHURN_COLOR = "#B42318"
RETAIN_COLOR = "#067647"
NEUTRAL_COLOR = "#667085"
ACCENT_COLOR = "#2563EB"

MODEL_PATH = Path(__file__).resolve().parent / "churn_predictor.joblib"


@st.cache_resource
def load_model_artifact() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. Run train.py first."
        )
    return joblib.load(MODEL_PATH)


def get_model_info() -> dict:
    artifact = load_model_artifact()
    return {
        "model_name": artifact["model_name"],
        "threshold": artifact["threshold"],
        "feature_columns": artifact["feature_columns"],
        "numeric_columns": artifact["numeric_columns"],
        "categorical_columns": artifact["categorical_columns"],
        "best_params": artifact["best_params"],
        "cv_f1": artifact["cv_f1"],
        "test_metrics": artifact["test_metrics"],
        "training_rows": artifact["training_rows"],
        "test_rows": artifact["test_rows"],
    }


def predict(record: dict) -> dict:
    artifact = load_model_artifact()
    frame = pd.DataFrame([record])[artifact["feature_columns"]]
    probabilities = artifact["model"].predict_proba(frame)[:, artifact["churn_class_index"]]
    probability = float(probabilities[0])
    threshold = artifact["threshold"]
    prediction = int(probability >= threshold)
    return {
        "prediction": prediction,
        "label": "Churned" if prediction == 1 else "Not Churned",
        "churn_probability": round(probability, 4),
        "threshold": threshold,
    }


def predict_batch(records: list[dict]) -> list[dict]:
    artifact = load_model_artifact()
    frame = pd.DataFrame(records)[artifact["feature_columns"]]
    probabilities = artifact["model"].predict_proba(frame)[:, artifact["churn_class_index"]]
    threshold = artifact["threshold"]
    results = []
    for probability in probabilities:
        value = float(probability)
        prediction = int(value >= threshold)
        results.append({
            "prediction": prediction,
            "label": "Churned" if prediction == 1 else "Not Churned",
            "churn_probability": round(value, 4),
            "threshold": threshold,
        })
    return results


def render_connection_help() -> None:
    st.markdown(
        "The local model could not be loaded. Run `python train.py` and refresh the page."
    )


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("### ChurnIQ")
        st.caption("Customer Intelligence")
        page = st.radio("Navigation", PAGES, label_visibility="collapsed")
        st.divider()
        try:
            load_model_artifact()
            st.success("Model loaded")
        except Exception:
            st.error("Model unavailable")
    return page


def build_metrics_chart(test_metrics: dict) -> go.Figure:
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    values = [test_metrics[key] * 100 for key in ["accuracy", "precision", "recall", "f1"]]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=ACCENT_COLOR,
            text=[f"{value:.1f}%" for value in values],
            textposition="auto",
            hovertemplate="%{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis=dict(range=[0, 100], ticksuffix="%", title=None),
        height=320,
        margin=dict(l=10, r=10, t=30, b=30),
        showlegend=False,
    )
    return fig


def render_overview() -> None:
    st.title("Overview")
    st.caption(
        "Live status from the local prediction model. Model metrics are saved training and "
        "evaluation results, not live production metrics."
    )

    try:
        load_model_artifact()
        health = {"model_loaded": True}
    except Exception as err:
        st.error(f"Model could not be loaded. {err}")
        render_connection_help()
        return

    try:
        info = get_model_info()
    except Exception as err:
        st.error(str(err))
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model status", "Loaded")
    col2.metric("Active model", info["model_name"])
    col3.metric("Input features", len(info["feature_columns"]))
    col4.metric("Decision threshold", f"{info['threshold']:.0%}")

    if not health["model_loaded"]:
        st.warning("The backend is running, but no model is loaded.")

    col5, col6 = st.columns(2)
    col5.metric(
        "Saved cross-validation F1",
        f"{info['cv_f1']:.1%}",
        help="Measured on training folds during model selection.",
    )
    col6.metric(
        "Saved test F1",
        f"{info['test_metrics']['f1']:.1%}",
        help="Measured once on the held-out test split.",
    )

    st.subheader("Saved held-out test metrics")
    st.caption(
        f"Measured on {info['test_rows']} held-out rows after training. "
        "A test set this small gives noisy estimates."
    )
    st.plotly_chart(build_metrics_chart(info["test_metrics"]), use_container_width=True)


def render_input(container, column: str):
    label = FIELD_LABELS[column]
    if column in CATEGORICAL_OPTIONS:
        return container.selectbox(label, CATEGORICAL_OPTIONS[column])
    low, high = NUMERIC_BOUNDS[column]
    if column in INT_COLUMNS:
        return container.number_input(
            label, min_value=int(low), max_value=int(high), value=DEFAULTS[column], step=1
        )
    return container.number_input(
        label, min_value=float(low), max_value=float(high), value=DEFAULTS[column], step=0.5, format="%.2f"
    )


def build_record(values: dict) -> dict:
    record = {}
    for column in FEATURE_COLUMNS:
        value = values[column]
        if column in INT_COLUMNS:
            record[column] = int(value)
        elif column in CATEGORICAL_OPTIONS:
            record[column] = str(value)
        else:
            record[column] = float(value)
    return record


def build_probability_chart(probability: float, threshold: float, is_churn: bool) -> go.Figure:
    percent = probability * 100
    fig = go.Figure(
        go.Bar(
            x=[percent],
            y=["Churn probability"],
            orientation="h",
            marker_color=CHURN_COLOR if is_churn else RETAIN_COLOR,
            text=[f"{percent:.1f}%"],
            textposition="auto",
            hovertemplate="%{x:.1f}%<extra></extra>",
        )
    )
    fig.add_vline(
        x=threshold * 100,
        line_dash="dash",
        line_color=NEUTRAL_COLOR,
        annotation_text=f"Threshold {threshold:.0%}",
        annotation_position="top",
    )
    fig.update_layout(
        xaxis=dict(range=[0, 100], ticksuffix="%", title=None),
        yaxis=dict(showticklabels=False),
        height=180,
        margin=dict(l=10, r=10, t=40, b=30),
        showlegend=False,
    )
    return fig


def render_prediction(result: dict) -> None:
    probability = result["churn_probability"]
    threshold = result["threshold"]
    is_churn = result["prediction"] == 1
    outcome = "Churn" if is_churn else "No Churn"

    st.subheader("Prediction result")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Predicted outcome",
        outcome,
        help="The model's yes/no decision at the configured threshold.",
    )
    col2.metric(
        "Estimated churn probability",
        f"{probability:.1%}",
        help="The model's estimate, not a guarantee.",
    )
    col3.metric("Decision threshold", f"{threshold:.0%}", help="Cut-off used by the backend.")

    st.plotly_chart(
        build_probability_chart(probability, threshold, is_churn),
        use_container_width=True,
    )

    comparison = "at or above" if is_churn else "below"
    st.markdown(
        f"The estimated churn probability is **{probability:.1%}**. The model predicts "
        f"**{outcome}** because this is {comparison} the decision threshold of **{threshold:.0%}**."
    )
    st.info(
        "The probability is the model's estimate based on the submitted details. It is not a "
        "guarantee of what this customer will do, and it does not show which actions would change it."
    )
    st.caption(f"Backend label: {result['label']}")


def render_predict_error(err: Exception) -> None:
    st.error(f"Prediction failed. {err}")
    render_connection_help()


def render_predict_page() -> None:
    st.title("Predict churn")
    st.caption("Enter one customer's details. The saved model creates the prediction.")

    values: dict = {}
    with st.form("customer_form"):
        for title, columns in FORM_SECTIONS:
            st.subheader(title)
            left, right = st.columns(2)
            for index, column in enumerate(columns):
                container = left if index % 2 == 0 else right
                values[column] = render_input(container, column)
        submitted = st.form_submit_button("Predict churn", type="primary")

    if not submitted:
        return

    try:
        with st.spinner("Running prediction..."):
            result = predict(build_record(values))
    except Exception as err:
        render_predict_error(err)
        return

    render_prediction(result)


def run_batch_predictions(df: pd.DataFrame) -> list[dict]:
    records = json.loads(df[FEATURE_COLUMNS].to_json(orient="records"))
    progress = st.progress(0.0, text="Running predictions...")
    predictions: list[dict] = []
    for start in range(0, len(records), BATCH_SIZE):
        chunk = records[start:start + BATCH_SIZE]
        predictions.extend(predict_batch(chunk))
        done = min(start + len(chunk), len(records))
        progress.progress(done / len(records), text=f"Scored {done} of {len(records)} customers")
    progress.empty()
    return predictions


def render_batch_results(frame: pd.DataFrame) -> None:
    churn_count = int((frame["PredictedOutcome"] == "Churn").sum())
    st.subheader("Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Customers scored", f"{len(frame):,}")
    col2.metric("Predicted churn", f"{churn_count:,}")
    col3.metric("Predicted no churn", f"{len(frame) - churn_count:,}")
    st.dataframe(frame, hide_index=True)
    st.download_button(
        "Download predictions CSV",
        frame.to_csv(index=False).encode("utf-8"),
        file_name="churn_predictions.csv",
        mime="text/csv",
    )


def render_batch_page() -> None:
    st.title("Batch predictions")
    st.caption("Upload customers as a CSV. The saved model scores every row.")

    st.download_button(
        "Download CSV template",
        batch_template_csv(),
        file_name="churn_input_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Customer CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV with one customer per row. Columns must match the template.")
        return

    try:
        df = pd.read_csv(uploaded)
    except ValueError:
        st.error("The file could not be read as CSV.")
        return

    st.write(f"{len(df):,} rows uploaded.")

    errors, warnings = validate_batch_frame(df)
    for warning in warnings:
        st.warning(warning)
    if errors:
        st.error("The file cannot be used yet:")
        for error in errors:
            st.markdown(f"- {error}")
    else:
        st.success("All required columns are present and values are valid.")

    st.subheader("Preview")
    st.dataframe(df.head(10), hide_index=True)

    if not errors:
        if st.button("Run batch predictions", type="primary"):
            try:
                predictions = run_batch_predictions(df)
            except Exception as err:
                st.session_state.pop("batch_result", None)
                st.error(
                    "Batch prediction failed. No results are shown because the run is incomplete. "
                    f"{err}"
                )
                return
            result = df.copy()
            result["PredictedOutcome"] = [
                "Churn" if item["prediction"] == 1 else "No Churn" for item in predictions
            ]
            result["ChurnProbability"] = [item["churn_probability"] for item in predictions]
            result["Threshold"] = [item["threshold"] for item in predictions]
            st.session_state["batch_result"] = {"file_name": uploaded.name, "frame": result}

    stored = st.session_state.get("batch_result")
    if stored and stored["file_name"] == uploaded.name:
        render_batch_results(stored["frame"])


def render_model_page() -> None:
    st.title("Model information")
    st.caption("Training and evaluation information from the saved model artifact.")

    try:
        info = get_model_info()
    except Exception as err:
        st.error(f"Model could not be loaded. {err}")
        render_connection_help()
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Model type", info["model_name"])
    col2.metric("Decision threshold", f"{info['threshold']:.0%}")
    col3.metric("Held-out test rows", f"{info['test_rows']:,}")

    st.subheader("Input features")
    feature_types = pd.DataFrame({
        "Feature": info["feature_columns"],
        "Type": [
            "Numeric" if column in info["numeric_columns"] else "Categorical"
            for column in info["feature_columns"]
        ],
    })
    st.dataframe(feature_types, hide_index=True)

    st.subheader("Saved results")
    test_metrics = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1"],
            "Held-out test": [
                f"{info['test_metrics'][key]:.1%}" for key in ["accuracy", "precision", "recall", "f1"]
            ],
        }
    )
    st.dataframe(test_metrics, hide_index=True)
    st.write(f"Cross-validation F1: **{info['cv_f1']:.1%}**")
    st.write(f"Selected hyperparameters: `{info['best_params']}`")

    st.subheader("Cross-validation vs. held-out test results")
    st.markdown(
        "- **Cross-validation (CV)** scores come from splitting the training data into folds. "
        "Each fold is held out in turn while the model trains on the rest. CV is used to compare "
        "candidate settings.\n"
        "- **Held-out test results** come from a separate test split that is kept out of training "
        "and tuning. They estimate how the selected model may perform on data it has not seen."
    )
    st.caption("Saved metrics describe training and evaluation. They are not live production performance.")


def main() -> None:
    page = render_sidebar()
    if page == "Overview":
        render_overview()
    elif page == "Predict churn":
        render_predict_page()
    elif page == "Batch predictions":
        render_batch_page()
    else:
        render_model_page()


main()
