
import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Online Shopper Intent Analyzer",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #f5f7ff, #eef2ff);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #312e81;
    margin-bottom: 5px;
}

.subtitle {
    color: #64748b;
    font-size: 14px;
}

.hero {
    background: linear-gradient(120deg, #4338ca, #7c3aed);
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 32px;
    font-weight: 800;
}

.hero p {
    color: #e0e7ff;
    font-size: 15px;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.04);
    min-height: 110px;
}

.metric-label {
    color: #64748b;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #312e81;
    font-size: 27px;
    font-weight: 800;
    margin-top: 8px;
}

.section-title {
    color: #312e81;
    font-size: 23px;
    font-weight: 700;
    margin-top: 20px;
}

.info-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    margin-bottom: 12px;
}

.best-card {
    background: linear-gradient(120deg, #059669, #10b981);
    padding: 25px;
    border-radius: 18px;
    color: white;
    margin: 15px 0 25px 0;
}

.best-card h2 {
    color: white;
    font-size: 29px;
    font-weight: 800;
}

.best-card p {
    color: #ecfdf5;
}

.stButton > button {
    background: linear-gradient(120deg, #4f46e5, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 10px 20px;
}

.stButton > button:hover {
    background: linear-gradient(120deg, #3730a3, #6d28d9);
    color: white;
}

[data-testid="stSidebar"] {
    background: #ffffff;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# FEATURE CONFIGURATION
# =========================================================

FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
]

TARGET = "Revenue"

CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
]

NUMERIC_FEATURES = [
    col for col in FEATURES
    if col not in CATEGORICAL_FEATURES
]


# =========================================================
# DATASET LOADING
# =========================================================

def find_dataset():

    possible_paths = [
        "online_shoppers_intention.csv",
        "Online Shoppers Intention.csv",
        "online_shoppers_intention/online_shoppers_intention.csv"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    for root, dirs, files in os.walk("."):
        dirs[:] = [
            d for d in dirs
            if d not in [".git", ".venv", "__pycache__"]
        ]

        for file in files:
            if file.lower() == "online_shoppers_intention.csv":
                return os.path.join(root, file)

    return None


@st.cache_data
def load_data(path):

    data = pd.read_csv(path)

    missing_columns = [
        col for col in FEATURES + [TARGET]
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {missing_columns}"
        )

    data = data[FEATURES + [TARGET]].copy()

    # Numeric columns
    for col in NUMERIC_FEATURES:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    # Categorical columns
    # Convert all categories to consistent strings.
    for col in CATEGORICAL_FEATURES:
        data[col] = data[col].where(
            data[col].notna(),
            "__MISSING__"
        ).astype(str)

    # Target conversion
    if data[TARGET].dtype == object:
        data[TARGET] = (
            data[TARGET]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0
            })
        )

    else:
        data[TARGET] = data[TARGET].astype(int)

    data = data.dropna(subset=[TARGET])

    data[TARGET] = data[TARGET].astype(int)

    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Revenue must contain only True/False or 1/0.")

    return data


# =========================================================
# PREPROCESSING
# =========================================================

def make_preprocessor(scale_numeric=False):

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ]
    )

    return preprocessor


# =========================================================
# MODEL TRAINING AND EVALUATION
# =========================================================

@st.cache_resource(show_spinner="Training machine learning models...")

def train_and_evaluate(data):

    X = data[FEATURES].copy()
    y = data[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    }

    results = {}
    trained_models = {}

    for model_name, model in models.items():

        scale = model_name == "Logistic Regression"

        pipeline = Pipeline([
            (
                "preprocessor",
                make_preprocessor(scale_numeric=scale)
            ),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        probabilities = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        auc = roc_auc_score(
            y_test,
            probabilities
        )

        results[model_name] = {

            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": auc,

            "Predictions": predictions,
            "Probabilities": probabilities,

            "Confusion Matrix": confusion_matrix(
                y_test,
                predictions
            ),

            "Classification Report": classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0
            )
        }

        trained_models[model_name] = pipeline

    # Dynamic Best Model selection:
    # First priority: F1-Score
    # Second priority: ROC-AUC

    best_model_name = max(
        results,
        key=lambda name: (
            results[name]["F1-Score"],
            results[name]["ROC-AUC"]
        )
    )

    return (
        results,
        trained_models,
        best_model_name,
        X_test,
        y_test
    )


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def metric_card(label, value):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def page_heading(title, description):

    st.markdown(
        f"""
        <div class="main-title">{title}</div>
        <div class="subtitle">{description}</div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    """
    <h1 style="color:#4338ca;">🛍️ Shopper AI</h1>
    <p style="color:#64748b;">ANALYTICS DASHBOARD</p>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Overview",
        "📊 Dataset Explorer",
        "🔮 Purchase Predictor",
        "📈 Model Evaluation",
        "⭐ Feature Importance"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **PROJECT DETAILS**

    🤖 Supervised Learning

    📁 Online Shoppers Dataset

    🎯 Binary Classification

    🧠 Three ML Models
    """
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Online Shopper Intent Analyzer | ML Project"
)


# =========================================================
# LOAD DATASET
# =========================================================

dataset_path = find_dataset()

if dataset_path is None:

    st.error(
        "Dataset not found! Please place "
        "online_shoppers_intention.csv "
        "in the same folder as app.py."
    )

    st.stop()

try:

    df = load_data(dataset_path)

except Exception as e:

    st.error(f"Dataset loading error: {e}")
    st.stop()


# =========================================================
# TRAIN MODELS
# =========================================================

try:

    (
        results,
        trained_models,
        best_model_name,
        X_test,
        y_test
    ) = train_and_evaluate(df)

except Exception as e:

    st.error(f"Model training error: {e}")
    st.stop()


# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "🏠 Overview":

    st.markdown(
        """
        <div class="hero">
            <h1>🛍️ Online Shopper Intent Analyzer</h1>
            <p>
            Analyze online shopping sessions and predict
            whether a session results in a purchase.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">📊 Dataset Overview</div>',
        unsafe_allow_html=True
    )

    total_sessions = len(df)

    purchases = int(df[TARGET].sum())

    non_purchases = total_sessions - purchases

    purchase_rate = purchases / total_sessions * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "Total Sessions",
            f"{total_sessions:,}"
        )

    with col2:
        metric_card(
            "Purchase Sessions",
            f"{purchases:,}"
        )

    with col3:
        metric_card(
            "Non-Purchase Sessions",
            f"{non_purchases:,}"
        )

    with col4:
        metric_card(
            "Purchase Rate",
            f"{purchase_rate:.2f}%"
        )

    st.markdown(
        '<div class="section-title">🤖 Model Performance</div>',
        unsafe_allow_html=True
    )

    best = results[best_model_name]

    st.markdown(
        f"""
        <div class="best-card">
            <p>🏆 BEST PERFORMING MODEL</p>
            <h2>{best_model_name}</h2>
            <p>
            F1-Score: {best['F1-Score']:.4f}
            &nbsp; | &nbsp;
            ROC-AUC: {best['ROC-AUC']:.4f}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    comparison = pd.DataFrame(results).T[
        ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    ]

    st.dataframe(
        comparison.style.format("{:.4f}").highlight_max(
            axis=0,
            color="#bbf7d0"
        ),
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">📌 Project Objective</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    - Analyze online shopping session behavior.
    - Predict whether a session ends in a purchase.
    - Compare three supervised machine learning models.
    - Identify influential features.
    - Evaluate model performance using classification metrics.
    """)


# =========================================================
# DATASET EXPLORER PAGE
# =========================================================

elif page == "📊 Dataset Explorer":

    page_heading(
        "📊 Dataset Explorer",
        "Explore dataset structure, columns and shopping session information."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card("Total Rows", f"{df.shape[0]:,}")

    with col2:
        metric_card("Total Columns", f"{df.shape[1]}")

    with col3:
        metric_card("Missing Values", f"{df.isnull().sum().sum()}")

    st.markdown(
        '<div class="section-title">🔍 Dataset Preview</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">📋 Dataset Information</div>',
        unsafe_allow_html=True
    )

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">📈 Revenue Distribution</div>',
        unsafe_allow_html=True
    )

    revenue_counts = (
        df[TARGET]
        .map({0: "No Purchase", 1: "Purchase"})
        .value_counts()
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    revenue_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Purchase vs Non-Purchase Sessions")
    ax.set_xlabel("Session Outcome")
    ax.set_ylabel("Number of Sessions")

    plt.xticks(rotation=0)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        '<div class="section-title">📅 Monthly Session Distribution</div>',
        unsafe_allow_html=True
    )

    month_counts = df["Month"].value_counts()

    fig, ax = plt.subplots(figsize=(9, 4))

    month_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Sessions by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Sessions")

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "⬇️ Download Dataset",
        df.to_csv(index=False).encode("utf-8"),
        "online_shoppers_dataset.csv",
        "text/csv"
    )


# =========================================================
# PURCHASE PREDICTOR PAGE
# =========================================================

elif page == "🔮 Purchase Predictor":

    page_heading(
        "🔮 Purchase Predictor",
        "Enter all 17 session features to predict the purchase outcome."
    )

    st.markdown("---")

    st.info(
        "Please complete all input fields before clicking Predict Purchase."
    )

    with st.form("prediction_form"):

        st.markdown("### 📄 Administrative Information")

        c1, c2 = st.columns(2)

        with c1:
            administrative = st.number_input(
                "Administrative",
                min_value=0,
                value=0,
                step=1
            )

            administrative_duration = st.number_input(
                "Administrative Duration",
                min_value=0.0,
                value=0.0
            )

        with c2:
            informational = st.number_input(
                "Informational",
                min_value=0,
                value=0,
                step=1
            )

            informational_duration = st.number_input(
                "Informational Duration",
                min_value=0.0,
                value=0.0
            )

        st.markdown("### 🛒 Product Information")

        c1, c2 = st.columns(2)

        with c1:
            product_related = st.number_input(
                "Product Related",
                min_value=0,
                value=1,
                step=1
            )

        with c2:
            product_related_duration = st.number_input(
                "Product Related Duration",
                min_value=0.0,
                value=10.0
            )

        st.markdown("### 📊 Behavioral Information")

        c1, c2, c3 = st.columns(3)

        with c1:
            bounce_rates = st.number_input(
                "Bounce Rates",
                min_value=0.0,
                max_value=1.0,
                value=0.02,
                step=0.01,
                format="%.4f"
            )

        with c2:
            exit_rates = st.number_input(
                "Exit Rates",
                min_value=0.0,
                max_value=1.0,
                value=0.05,
                step=0.01,
                format="%.4f"
            )

        with c3:
            page_values = st.number_input(
                "Page Values",
                min_value=0.0,
                value=0.0,
                step=1.0
            )

        special_day = st.select_slider(
            "Special Day",
            options=[
                0.0, 0.2, 0.4, 0.6, 0.8, 1.0
            ],
            value=0.0
        )

        st.markdown("### 🌐 Visitor Information")

        c1, c2, c3 = st.columns(3)

        with c1:
            month = st.selectbox(
                "Month",
                [
                    "Feb", "Mar", "May", "June",
                    "Jul", "Aug", "Sep", "Oct",
                    "Nov", "Dec"
                ],
                index=8
            )

        with c2:
            operating_systems = st.selectbox(
                "Operating Systems",
                list(range(1, 9))
            )

        with c3:
            browser = st.selectbox(
                "Browser",
                list(range(1, 14))
            )

        c1, c2, c3 = st.columns(3)

        with c1:
            region = st.selectbox(
                "Region",
                list(range(1, 10))
            )

        with c2:
            traffic_type = st.selectbox(
                "Traffic Type",
                list(range(1, 21))
            )

        with c3:
            visitor_type = st.selectbox(
                "Visitor Type",
                [
                    "Returning_Visitor",
                    "New_Visitor",
                    "Other"
                ]
            )

        weekend = st.selectbox(
            "Weekend",
            ["False", "True"]
        )

        submitted = st.form_submit_button(
            "🔮 Predict Purchase",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{

            "Administrative": administrative,
            "Administrative_Duration": administrative_duration,

            "Informational": informational,
            "Informational_Duration": informational_duration,

            "ProductRelated": product_related,
            "ProductRelated_Duration": product_related_duration,

            "BounceRates": bounce_rates,
            "ExitRates": exit_rates,

            "PageValues": page_values,
            "SpecialDay": special_day,

            "Month": str(month),
            "OperatingSystems": str(operating_systems),
            "Browser": str(browser),
            "Region": str(region),
            "TrafficType": str(traffic_type),

            "VisitorType": str(visitor_type),
            "Weekend": str(weekend)

        }])

        try:

            # Use the dynamically selected best model.
            predictor = trained_models[best_model_name]

            prediction = int(
                predictor.predict(input_data)[0]
            )

            probability = float(
                predictor.predict_proba(input_data)[0][1]
            )

            st.markdown("---")

            st.markdown("### 🎯 Prediction Result")

            st.write(
                f"Model Used: **{best_model_name}**"
            )

            if prediction == 1:

                st.success(
                    "🛍️ Prediction: Purchase"
                )

            else:

                st.warning(
                    "📌 Prediction: No Purchase"
                )

            st.metric(
                "Predicted Purchase Probability",
                f"{probability * 100:.2f}%"
            )

            st.progress(probability)

            st.caption(
                "This is a machine learning prediction, "
                "not a guarantee of actual customer behavior."
            )

        except Exception as e:

            st.error(f"Prediction error: {e}")


# =========================================================
# MODEL EVALUATION PAGE
# =========================================================

elif page == "📈 Model Evaluation":

    page_heading(
        "📈 Model Evaluation",
        "Compare the three machine learning models and identify the best performer."
    )

    st.markdown("---")

    # Dynamic Best Model
    best_metrics = results[best_model_name]

    st.markdown(
        f"""
        <div class="best-card">
            <p>🏆 BEST PERFORMING MODEL</p>
            <h2>{best_model_name}</h2>
            <p>
            Selected using F1-Score as the primary metric
            and ROC-AUC as the tie-breaker.
            </p>
            <p>
            F1-Score: {best_metrics['F1-Score']:.4f}
            &nbsp; | &nbsp;
            ROC-AUC: {best_metrics['ROC-AUC']:.4f}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">📊 Performance Metrics</div>',
        unsafe_allow_html=True
    )

    metrics_df = pd.DataFrame(results).T[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC"
        ]
    ]

    st.dataframe(
        metrics_df.style.format("{:.4f}").highlight_max(
            axis=0,
            color="#bbf7d0"
        ),
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">📉 Model Comparison Chart</div>',
        unsafe_allow_html=True
    )

    selected_metric = st.selectbox(
        "Select Evaluation Metric",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC"
        ],
        index=3
    )

    chart_data = metrics_df[[selected_metric]]

    st.bar_chart(chart_data)

    st.markdown(
        '<div class="section-title">📈 ROC Curve Comparison</div>',
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, model_result in results.items():

        probabilities = model_result["Probabilities"]

        fpr, tpr, thresholds = roc_curve(
            y_test,
            probabilities
        )

        ax.plot(
            fpr,
            tpr,
            label=(
                f"{model_name} "
                f"(AUC={model_result['ROC-AUC']:.3f})"
            )
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")

    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        '<div class="section-title">🔍 Confusion Matrix</div>',
        unsafe_allow_html=True
    )

    selected_model = st.selectbox(
        "Select Model",
        list(results.keys())
    )

    cm = results[selected_model]["Confusion Matrix"]

    fig, ax = plt.subplots(figsize=(6, 4))

    image = ax.imshow(cm)

    ax.set_title(
        f"Confusion Matrix - {selected_model}"
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["No Purchase", "Purchase"])
    ax.set_yticklabels(["No Purchase", "Purchase"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=14
            )

    fig.colorbar(image, ax=ax)

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        '<div class="section-title">📋 Classification Report</div>',
        unsafe_allow_html=True
    )

    report = results[selected_model][
        "Classification Report"
    ]

    report_df = pd.DataFrame(report).T

    st.dataframe(
        report_df.style.format("{:.4f}"),
        use_container_width=True
    )

    st.download_button(
        "⬇️ Download Model Comparison",
        metrics_df.to_csv().encode("utf-8"),
        "model_comparison.csv",
        "text/csv"
    )


# =========================================================
# FEATURE IMPORTANCE PAGE
# =========================================================

elif page == "⭐ Feature Importance":

    page_heading(
        "⭐ Feature Importance",
        "Understand which input features influence predictions from the best model."
    )

    st.markdown("---")

    st.markdown(
        f"""
        <div class="best-card">
            <p>🏆 SELECTED BEST MODEL</p>
            <h2>{best_model_name}</h2>
            <p>
            Feature importance is calculated using
            permutation importance on the test dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.spinner("Calculating feature importance..."):

        best_pipeline = trained_models[best_model_name]

        importance_result = permutation_importance(
            best_pipeline,
            X_test,
            y_test,
            scoring="f1",
            n_repeats=5,
            random_state=42,
            n_jobs=-1
        )

    importance_df = pd.DataFrame({

        "Feature": FEATURES,

        "Importance": importance_result.importances_mean

    }).sort_values(
        by="Importance",
        ascending=False
    )

    st.markdown(
        '<div class="section-title">📊 Feature Importance Ranking</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        importance_df.style.format({
            "Importance": "{:.5f}"
        }),
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">📈 Feature Importance Chart</div>',
        unsafe_allow_html=True
    )

    top_features = importance_df.head(12).sort_values(
        by="Importance",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    ax.set_title(
        f"Top Features - {best_model_name}"
    )

    ax.set_xlabel("Mean Decrease in F1-Score")

    ax.set_ylabel("Features")

    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "Permutation importance measures how much the model's "
        "F1-Score changes when a feature is shuffled. "
        "It indicates predictive usefulness, not causation."
    )

    st.download_button(
        "⬇️ Download Feature Importance",
        importance_df.to_csv(index=False).encode("utf-8"),
        "feature_importance.csv",
        "text/csv"
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#64748b; padding:15px;">
        <b>🛍️ Online Shopper Intent Analyzer</b><br>
        Supervised Machine Learning Project<br>
        Logistic Regression | Random Forest | Gradient Boosting
    </div>
    """,
    unsafe_allow_html=True
)
