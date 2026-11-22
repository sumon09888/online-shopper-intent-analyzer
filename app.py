
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
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
    RocCurveDisplay
)

from sklearn.inspection import permutation_importance


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Online Shopper Intent Analyzer",
    page_icon="🛍️",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

DATA_PATH = BASE_DIR / "online_shoppers_intention.csv"
MODEL_PATH = BASE_DIR / "shopper_intent_model.pkl"


# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #172554;
}

.subtitle {
    color: #64748b;
    font-size: 15px;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 3px 12px rgba(0,0,0,0.04);
}

.metric-label {
    color: #64748b;
    font-size: 14px;
}

.metric-value {
    color: #172554;
    font-size: 27px;
    font-weight: 800;
}

.section-title {
    font-size: 23px;
    font-weight: 750;
    color: #172554;
}

div.stButton > button {
    background: #4f46e5;
    color: white;
    border-radius: 9px;
    border: none;
    font-weight: 700;
}

div.stButton > button:hover {
    background: #3730a3;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. FEATURE CONFIGURATION
# =====================================================

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


# =====================================================
# 4. LOAD DATA AND SAVED MODEL
# =====================================================

@st.cache_data
def load_data():

    data = pd.read_csv(DATA_PATH)

    data[TARGET] = data[TARGET].astype(bool)

    for col in NUMERIC_FEATURES:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    for col in CATEGORICAL_FEATURES:

        data[col] = data[col].astype(object)

        data[col] = data[col].where(
            pd.notna(data[col]),
            np.nan
        )

    return data


@st.cache_resource
def load_saved_model():

    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    return None


try:

    df = load_data()
    saved_model = load_saved_model()

except Exception as e:

    st.error(f"File loading error: {e}")
    st.stop()


missing_columns = [
    col for col in FEATURES + [TARGET]
    if col not in df.columns
]

if missing_columns:

    st.error(f"Missing columns: {missing_columns}")
    st.stop()


# =====================================================
# 5. PREPROCESSING
# =====================================================

def make_preprocessor(scale_numeric=False):

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]

    if scale_numeric:

        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(
        numeric_steps
    )

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent",
                missing_values=np.nan
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    return ColumnTransformer([
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
    ])


# =====================================================
# 6. MODEL TRAINING
# =====================================================

@st.cache_resource
def train_and_evaluate(data):

    X = data[FEATURES].copy()
    y = data[TARGET].astype(bool)

    for col in NUMERIC_FEATURES:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    for col in CATEGORICAL_FEATURES:

        X[col] = X[col].astype(object)

        X[col] = X[col].where(
            pd.notna(X[col]),
            np.nan
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {

        "Logistic Regression": Pipeline([
            (
                "preprocessor",
                make_preprocessor(scale_numeric=True)
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]),

        "Random Forest": Pipeline([
            (
                "preprocessor",
                make_preprocessor()
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]),

        "Gradient Boosting": Pipeline([
            (
                "preprocessor",
                make_preprocessor()
            ),
            (
                "classifier",
                GradientBoostingClassifier(
                    random_state=42
                )
            )
        ])

    }

    results = {}
    fitted_models = {}

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        results[name] = {

            "Accuracy": accuracy_score(
                y_test,
                predictions
            ),

            "Precision": precision_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "Recall": recall_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "F1 Score": f1_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "ROC-AUC": roc_auc_score(
                y_test,
                probabilities
            )

        }

        fitted_models[name] = model

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        results,
        fitted_models
    )


# =====================================================
# 7. TRAIN MODELS
# =====================================================

with st.spinner("Training three machine learning models..."):

    try:

        (
            X_train,
            X_test,
            y_train,
            y_test,
            model_results,
            fitted_models
        ) = train_and_evaluate(df)

    except Exception as e:

        st.error(f"Model training error: {e}")
        st.exception(e)
        st.stop()


# =====================================================
# 8. SIDEBAR
# =====================================================

st.sidebar.markdown("# 🛍️ Online Shopper")
st.sidebar.caption("ANALYTICS DASHBOARD")

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

st.sidebar.divider()

st.sidebar.markdown("### PROJECT DETAILS")

st.sidebar.write("🤖 Supervised Learning")
st.sidebar.write("📁 12,330 Sessions")
st.sidebar.write("🎯 Target: Revenue")
st.sidebar.write("🧩 17 Input Features")
st.sidebar.write("📊 Binary Classification")
st.sidebar.write("🧠 3 ML Models")

st.sidebar.divider()

st.sidebar.caption("Online Shopper Intent Analyzer")


# =====================================================
# 9. OVERVIEW
# =====================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="main-title">'
        '🛍️ Online Shopper Intent Analyzer'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Online Shopping Purchase Intention Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    total_sessions = len(df)

    purchases = int(df[TARGET].sum())

    non_purchases = total_sessions - purchases

    purchase_rate = (
        purchases / total_sessions
    ) * 100

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        ("Total Sessions", f"{total_sessions:,}"),
        ("Purchase Sessions", f"{purchases:,}"),
        ("Non-Purchase Sessions", f"{non_purchases:,}"),
        ("Purchase Rate", f"{purchase_rate:.2f}%")
    ]

    for col, (label, value) in zip(
        [c1, c2, c3, c4],
        metrics
    ):

        with col:

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    st.markdown(
        '<div class="section-title">'
        '📊 Purchase Distribution'
        '</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:

        counts = df[TARGET].value_counts().reindex(
            [False, True],
            fill_value=0
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.pie(
            counts.values,
            labels=["No Purchase", "Purchase"],
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title("Purchase vs Non-Purchase")

        st.pyplot(fig)
        plt.close(fig)

    with right:

        monthly = (
            df.groupby("Month")[TARGET]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        monthly.plot(kind="bar", ax=ax)

        ax.set_title("Purchase Rate by Month")
        ax.set_ylabel("Purchase Rate (%)")
        ax.set_xlabel("Month")

        plt.xticks(rotation=45)

        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    st.markdown("### 🎯 Project Objective")

    st.write(
        "This project uses historical browsing-session features "
        "to predict whether an online shopping session ends in a purchase."
    )

    st.write(
        "It compares Logistic Regression, Random Forest, "
        "and Gradient Boosting using held-out test data."
    )

    st.info(
        "Revenue represents the purchase outcome. "
        "A non-purchase session is not necessarily a confirmed "
        "cart-abandonment event."
    )


# =====================================================
# 10. DATASET EXPLORER
# =====================================================

elif page == "📊 Dataset Explorer":

    st.markdown(
        '<div class="main-title">📊 Dataset Explorer</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Explore the Online Shoppers Purchasing Intention dataset."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Rows", f"{len(df):,}")
    c2.metric("Total Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.divider()

    st.markdown("### Dataset Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    st.markdown("### Data Types and Missing Values")

    info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(
        info,
        use_container_width=True
    )

    st.markdown("### Statistical Summary")

    st.dataframe(
        df.describe(include="all"),
        use_container_width=True
    )

    st.divider()

    st.markdown("### Target Class Balance")

    class_counts = df[TARGET].value_counts().reindex(
        [False, True],
        fill_value=0
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    class_counts.plot(kind="bar", ax=ax)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["No Purchase", "Purchase"],
        rotation=0
    )

    ax.set_ylabel("Number of Sessions")
    ax.set_title("Target Class Distribution")

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Numerical Feature Correlation")

    numeric_df = df.select_dtypes(include=np.number)

    correlation = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 8))

    image = ax.imshow(correlation, aspect="auto")

    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.columns)))

    ax.set_xticklabels(
        correlation.columns,
        rotation=90
    )

    ax.set_yticklabels(correlation.columns)

    ax.set_title("Correlation Matrix")

    fig.colorbar(image, ax=ax)

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# =====================================================
# 11. PURCHASE PREDICTOR
# =====================================================

elif page == "🔮 Purchase Predictor":

    st.markdown(
        '<div class="main-title">🔮 Purchase Predictor</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter the 17 session features used by the trained model."
    )

    st.info(
        "Complete all 17 fields before clicking Predict Purchase."
    )

    if saved_model is None:

        st.error(
            "shopper_intent_model.pkl was not found in the project folder."
        )

        st.stop()

    with st.form("prediction_form"):

        st.markdown("### Session Input Features")

        col1, col2 = st.columns(2)

        with col1:

            administrative = st.number_input(
                "Administrative",
                min_value=0,
                value=None,
                step=1,
                placeholder="Enter page count"
            )

            administrative_duration = st.number_input(
                "Administrative_Duration",
                min_value=0.0,
                value=None,
                placeholder="Enter duration"
            )

            informational = st.number_input(
                "Informational",
                min_value=0,
                value=None,
                step=1,
                placeholder="Enter page count"
            )

            informational_duration = st.number_input(
                "Informational_Duration",
                min_value=0.0,
                value=None,
                placeholder="Enter duration"
            )

            product_related = st.number_input(
                "ProductRelated",
                min_value=0,
                value=None,
                step=1,
                placeholder="Enter page count"
            )

            product_related_duration = st.number_input(
                "ProductRelated_Duration",
                min_value=0.0,
                value=None,
                placeholder="Enter duration"
            )

            bounce_rates = st.number_input(
                "BounceRates",
                min_value=0.0,
                max_value=1.0,
                value=None,
                format="%.4f",
                placeholder="Enter 0 to 1"
            )

            exit_rates = st.number_input(
                "ExitRates",
                min_value=0.0,
                max_value=1.0,
                value=None,
                format="%.4f",
                placeholder="Enter 0 to 1"
            )

            page_values = st.number_input(
                "PageValues",
                min_value=0.0,
                value=None,
                placeholder="Enter page value"
            )

        with col2:

            special_options = sorted(
                df["SpecialDay"].dropna().unique().tolist()
            )

            special_day = st.selectbox(
                "SpecialDay",
                [None] + special_options,
                format_func=lambda x:
                "Select SpecialDay" if x is None else str(x)
            )

            month_options = sorted(
                df["Month"].dropna().unique().tolist()
            )

            month = st.selectbox(
                "Month",
                [None] + month_options,
                format_func=lambda x:
                "Select Month" if x is None else str(x)
            )

            operating_options = sorted(
                df["OperatingSystems"].dropna().unique().tolist()
            )

            operating_system = st.selectbox(
                "OperatingSystems",
                [None] + operating_options,
                format_func=lambda x:
                "Select Operating System" if x is None else str(x)
            )

            browser_options = sorted(
                df["Browser"].dropna().unique().tolist()
            )

            browser = st.selectbox(
                "Browser",
                [None] + browser_options,
                format_func=lambda x:
                "Select Browser" if x is None else str(x)
            )

            region_options = sorted(
                df["Region"].dropna().unique().tolist()
            )

            region = st.selectbox(
                "Region",
                [None] + region_options,
                format_func=lambda x:
                "Select Region" if x is None else str(x)
            )

            traffic_options = sorted(
                df["TrafficType"].dropna().unique().tolist()
            )

            traffic = st.selectbox(
                "TrafficType",
                [None] + traffic_options,
                format_func=lambda x:
                "Select Traffic Type" if x is None else str(x)
            )

            visitor_options = sorted(
                df["VisitorType"].dropna().unique().tolist()
            )

            visitor_type = st.selectbox(
                "VisitorType",
                [None] + visitor_options,
                format_func=lambda x:
                "Select Visitor Type" if x is None else str(x)
            )

            weekend = st.selectbox(
                "Weekend",
                [None, True, False],
                format_func=lambda x:
                "Select Weekend"
                if x is None
                else ("Yes" if x else "No")
            )

        submitted = st.form_submit_button(
            "🔮 Predict Purchase",
            use_container_width=True
        )

    if submitted:

        values = [
            administrative,
            administrative_duration,
            informational,
            informational_duration,
            product_related,
            product_related_duration,
            bounce_rates,
            exit_rates,
            page_values,
            special_day,
            month,
            operating_system,
            browser,
            region,
            traffic,
            visitor_type,
            weekend
        ]

        if any(value is None for value in values):

            st.error(
                "Please complete all 17 input fields before prediction."
            )

        else:

            input_data = pd.DataFrame([{

                "Administrative": int(administrative),

                "Administrative_Duration": float(
                    administrative_duration
                ),

                "Informational": int(informational),

                "Informational_Duration": float(
                    informational_duration
                ),

                "ProductRelated": int(product_related),

                "ProductRelated_Duration": float(
                    product_related_duration
                ),

                "BounceRates": float(bounce_rates),

                "ExitRates": float(exit_rates),

                "PageValues": float(page_values),

                "SpecialDay": float(special_day),

                "Month": str(month),

                "OperatingSystems": operating_system,

                "Browser": browser,

                "Region": region,

                "TrafficType": traffic,

                "VisitorType": visitor_type,

                "Weekend": bool(weekend)

            }], columns=FEATURES)

            try:

                prediction = saved_model.predict(
                    input_data
                )[0]

                probability = None

                if hasattr(saved_model, "predict_proba"):

                    probabilities = saved_model.predict_proba(
                        input_data
                    )[0]

                    classes = list(saved_model.classes_)

                    positive_index = next(
                        (
                            i for i, value in enumerate(classes)
                            if value in [True, 1]
                        ),
                        None
                    )

                    if positive_index is not None:

                        probability = (
                            probabilities[positive_index] * 100
                        )

                st.divider()

                st.markdown("### Prediction Result")

                if prediction in [True, 1]:

                    st.success("🛒 Purchase Predicted")

                else:

                    st.warning("No Purchase Predicted")

                if probability is not None:

                    st.metric(
                        "Estimated Purchase Probability",
                        f"{probability:.2f}%"
                    )

                    st.progress(
                        float(probability / 100)
                    )

                st.caption(
                    "This is a hypothetical prediction based on "
                    "historical session data, not a guarantee."
                )

                with st.expander("View Input Features"):

                    st.dataframe(
                        input_data,
                        use_container_width=True
                    )

            except Exception as e:

                st.error(f"Prediction error: {e}")
                st.exception(e)


# =====================================================
# 12. MODEL EVALUATION
# =====================================================

elif page == "📈 Model Evaluation":

    st.markdown(
        '<div class="main-title">📈 Model Evaluation</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Compare the performance of three machine learning models."
    )

    st.info(
        "All models use the same stratified train/test split. "
        "Accuracy is not considered alone because the target is imbalanced."
    )

    results_df = pd.DataFrame(model_results).T

    st.markdown("### Model Comparison")

    st.dataframe(
        results_df.style.format("{:.4f}"),
        use_container_width=True
    )

    st.markdown("### Performance Comparison")

    selected_metric = st.selectbox(
        "Select Evaluation Metric",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    results_df[selected_metric].plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        f"Model Comparison - {selected_metric}"
    )

    ax.set_ylabel(selected_metric)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Model")

    plt.xticks(rotation=15)

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.markdown("### Confusion Matrix")

    selected_model_name = st.selectbox(
        "Choose Model",
        list(fitted_models.keys())
    )

    selected_model = fitted_models[selected_model_name]

    predictions = selected_model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[False, True]
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.imshow(cm)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["No Purchase", "Purchase"]
    )

    ax.set_yticklabels(
        ["No Purchase", "Purchase"]
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_title(
        f"{selected_model_name} Confusion Matrix"
    )

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(report).T,
        use_container_width=True
    )

    st.markdown("### ROC Curve Comparison")

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in fitted_models.items():

        probabilities = model.predict_proba(X_test)[:, 1]

        RocCurveDisplay.from_predictions(
            y_test,
            probabilities,
            name=name,
            ax=ax
        )

    ax.set_title("ROC Curve Comparison")

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# =====================================================
# 13. FEATURE IMPORTANCE
# =====================================================

elif page == "⭐ Feature Importance":

    st.markdown(
        '<div class="main-title">⭐ Feature Importance</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Explore which features influence model predictions."
    )

    st.info(
        "Feature importance shows model reliance or association. "
        "It does not prove that a feature causes a purchase."
    )

    selected_name = st.selectbox(
        "Select Model",
        [
            "Logistic Regression",
            "Random Forest",
            "Gradient Boosting"
        ]
    )

    selected_model = fitted_models[selected_name]

    with st.spinner("Calculating feature importance..."):

        try:

            permutation = permutation_importance(
                selected_model,
                X_test,
                y_test,
                scoring="roc_auc",
                n_repeats=3,
                random_state=42,
                n_jobs=-1
            )

            importance_df = pd.DataFrame({

                "Feature": FEATURES,

                "Importance": permutation.importances_mean

            })

            importance_df = importance_df.sort_values(
                "Importance",
                ascending=False
            )

            st.markdown("### Feature Importance Ranking")

            st.dataframe(
                importance_df,
                use_container_width=True
            )

            fig, ax = plt.subplots(figsize=(10, 7))

            plot_data = importance_df.sort_values(
                "Importance",
                ascending=True
            )

            ax.barh(
                plot_data["Feature"],
                plot_data["Importance"]
            )

            ax.set_title(
                f"Permutation Feature Importance - {selected_name}"
            )

            ax.set_xlabel(
                "Decrease in ROC-AUC after shuffling"
            )

            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:

            st.error(f"Feature importance error: {e}")
            st.exception(e)


# =====================================================
# 14. FOOTER
# =====================================================

st.divider()

st.caption(
    "Online Shopper Intent Analyzer | "
    "Supervised Binary Classification"
)

st.caption(
    "Academic Machine Learning Project"
)
