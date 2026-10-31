import json
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="NEUROBRIDGE",
    page_icon="🧠",
    layout="wide"
)

with open("results/experiment_results.json") as f:
    results = json.load(f)

st.title("NEUROBRIDGE")
st.caption("Neural Decoding & Research Intelligence Lab")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Explore",
    [
        "Research Overview",
        "Model Evaluation",
        "Calibration Lab"
    ]
)

if page == "Research Overview":

    st.header("Research Overview")

    st.write(
        "An experimental machine-learning framework "
        "for decoding motor imagery from human EEG recordings."
    )

    a, b, c = st.columns(3)

    a.metric("Participants", results["subject_count"])
    b.metric("EEG Channels", results["channels"])
    c.metric("Sampling Frequency", "250 Hz")

    st.subheader("Research Question")

    st.write(
        "How well can machine-learning models decode "
        "motor imagery, and how does subject-specific "
        "calibration influence performance?"
    )

    st.subheader("Dataset")

    st.write("BCI Competition IV Dataset 2a")

    st.subheader("Experimental Pipeline")

    st.code(
        "EEG Acquisition → Preprocessing → "
        "Feature Extraction → Neural Decoding → "
        "Evaluation"
    )

elif page == "Model Evaluation":

    st.header("Model Evaluation")

    experiments = results["experiments"]

    models = pd.DataFrame({
        "Model": ["CSP + LDA", "EEGNet"],
        "Balanced Accuracy": [
            experiments["csp_lda"]["balanced_accuracy"],
            experiments["eegnet"]["balanced_accuracy"]
        ]
    })

    fig = px.bar(
        models,
        x="Model",
        y="Balanced Accuracy",
        range_y=[0, 1],
        title="Classical ML vs Deep Learning"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.metric(
        "Cross-subject mean accuracy",
        f"{experiments['cross_subject']['mean_accuracy']:.2%}"
    )

    st.metric(
        "Cross-subject standard deviation",
        f"{experiments['cross_subject']['std']:.2%}"
    )

elif page == "Calibration Lab":

    st.header("Subject-specific Calibration")

    calibration = results["calibration"]

    df = pd.DataFrame({
        "Calibration": [
            "Zero-shot",
            "5 trials/class",
            "10 trials/class",
            "20 trials/class",
            "Subject-only"
        ],
        "Balanced Accuracy": list(calibration.values())
    })

    fig = px.line(
        df,
        x="Calibration",
        y="Balanced Accuracy",
        markers=True,
        range_y=[0, 1],
        title="Effect of Subject-specific Calibration"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df, use_container_width=True)

st.sidebar.divider()
st.sidebar.caption("NEUROBRIDGE | Research Prototype")