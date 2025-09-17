# streamlit_app_custom.py
# Redesigned and polished version of the original streamlit_app.py
# - Modern layout (wide), nicer header, sidebar improvements
# - Background/styling via CSS
# - File uploader to load your own CSV
# - Caching for dataset loads
# - Small quality-of-life tweaks: logo support, metrics, About card
#
# Keep LICENSE in your repo (original project used MIT). If you want to change
# the displayed author name, edit AUTHOR_NAME below.

import warnings
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
import numpy as np
AUTHOR_NAME = "Puneet"  # <-- change this to your name to brand the app


# Streamlit warnings / caching
st.set_option("deprecation.showPyplotGlobalUse", False)
warnings.filterwarnings("ignore")

# --- Page config ---
st.set_page_config(
    page_title=f"HyperTuneML — {AUTHOR_NAME}",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Styling: background + simple theme polish ---
st.markdown(
    """
    <style>
    /* App background with overlay gradient for readability */
    .stApp {
        background: linear-gradient(rgba(10,25,41,0.65), rgba(10,25,41,0.65)), url('https://i.morioh.com/52c215bc5f.png');
        background-size: cover;
        background-attachment: fixed;
    }

    /* Header and card styling */
    .app-header{display:flex;align-items:center;gap:12px;margin-bottom:12px}
    .app-title{font-size:28px;font-weight:700;color:#fff;margin:0}
    .app-sub{color:#cbd5e1;margin:0}
    .card{background: rgba(255,255,255,0.03); padding:16px; border-radius:12px;}

    /* Sidebar inputs contrast */
    [data-testid="stSidebar"] .stSelectbox, [data-testid="stSidebar"] .stSlider, [data-testid="stSidebar"] .stTextInput{background:rgba(255,255,255,0.02)}

    /* Footer small text */
    .footer {color:#94a3b8; font-size:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Utility: cached dataset loader ---


@st.cache_data(show_spinner=False)
def load_dataset(Data):
    """Return either an sklearn bunch or a pandas DataFrame for CSV datasets."""
    if Data == "Iris":
        return datasets.load_iris()
    elif Data == "Wine":
        return datasets.load_wine()
    elif Data == "Breast Cancer":
        return datasets.load_breast_cancer()
    elif Data == "Diabetes":
        return datasets.load_diabetes()
    elif Data == "Digits":
        return datasets.load_digits()
    elif Data == "Salary":
        return pd.read_csv("Dataset/Salary_dataset.csv")
    elif Data == "Naive Bayes Classification":
        return pd.read_csv("Dataset/Naive-Bayes-Classification-Data.csv")
    elif Data == "Heart Disease Classification":
        return pd.read_csv("Dataset/Updated_heart_prediction.csv")
    elif Data == "Titanic":
        return pd.read_csv("Dataset/Preprocessed Titanic Dataset.csv")
    else:
        return pd.read_csv("Dataset/car_evaluation.csv")


# --- Input/Output extraction (supports sklearn bunch or pandas DataFrame) ---
def Input_output(data, data_name):
    # If sklearn bunch (has .data and .target)
    if hasattr(data, "data") and hasattr(data, "target"):
        X = data.data
        Y = data.target
        return X, Y

    # Else for custom DataFrame-based datasets
    df = data.copy()
    if data_name == "Salary":
        X = df["YearsExperience"].to_numpy().reshape(-1, 1)
        Y = df["Salary"].to_numpy().reshape(-1, 1)
    elif data_name == "Naive Bayes Classification":
        X = df.drop("diabetes", axis=1)
        Y = df["diabetes"]
    elif data_name == "Heart Disease Classification":
        X = df.drop("output", axis=1)
        Y = df["output"]
    elif data_name == "Titanic":
        X = df.drop(columns=["survived", "home.dest",
                    "last_name", "first_name", "title"], errors='ignore')
        Y = df["survived"]
    elif data_name == "Car Evaluation":
        le = LabelEncoder()
        for col in df.columns:
            try:
                df[col] = le.fit_transform(df[col])
            except Exception:
                pass
        X = df.drop(["unacc"], axis=1)
        Y = df["unacc"]
    else:
        # fallback: use all columns except last as X and last as Y
        X = df.iloc[:, :-1].values
        Y = df.iloc[:, -1].values
    return X, Y


# --- Parameter helpers (kept largely same, small polish) ---
def add_parameter_classifier_general(algorithm):
    params = dict()
    if algorithm == "SVM":
        c_regular = st.sidebar.slider(
            "C (Regularization)", 0.01, 10.0, value=1.0)
        kernel_custom = st.sidebar.selectbox(
            "Kernel", ("linear", "poly", "rbf", "sigmoid"))
        params["C"] = c_regular
        params["kernel"] = kernel_custom
    elif algorithm == "KNN":
        k_n = st.sidebar.slider(
            "Number of Neighbors (K)", 1, 20, key="k_n_slider")
        weights_custom = st.sidebar.selectbox(
            "Weights", ("uniform", "distance"))
        params["K"] = k_n
        params["weights"] = weights_custom
    elif algorithm == "Naive Bayes":
        st.sidebar.info("Naive Bayes has no hyperparameters in this UI.")
    elif algorithm == "Decision Tree":
        max_depth = st.sidebar.slider("Max Depth", 2, 17)
        criterion = st.sidebar.selectbox("Criterion", ("gini", "entropy"))
        splitter = st.sidebar.selectbox("Splitter", ("best", "random"))
        params["max_depth"] = max_depth
        params["criterion"] = criterion
        params["splitter"] = splitter
        try:
            random = st.sidebar.text_input("Enter Random State")
            params["random_state"] = int(random) if random != "" else 4567
        except:
            params["random_state"] = 4567
    elif algorithm == "Random Forest":
        max_depth = st.sidebar.slider("Max Depth", 2, 17)
        n_estimators = st.sidebar.slider("Number of Estimators", 1, 90)
        criterion = st.sidebar.selectbox(
            "Criterion", ("gini", "entropy", "log_loss"))
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["criterion"] = criterion
        try:
            random = st.sidebar.text_input("Enter Random State")
            params["random_state"] = int(random) if random != "" else 4567
        except:
            params["random_state"] = 4567
    else:
        c_regular = st.sidebar.slider(
            "C (Regularization)", 0.01, 10.0, value=1.0)
        fit_intercept = st.sidebar.selectbox(
            "Fit Intercept", ("True", "False"))
        penalty = st.sidebar.selectbox("Penalty", ("l2", None))
        n_jobs = st.sidebar.selectbox("Number of Jobs", (None, -1))
        params["C"] = c_regular
        params["fit_intercept"] = bool(fit_intercept == "True")
        params["penalty"] = penalty
        params["n_jobs"] = n_jobs
    return params


def add_parameter_regressor(algorithm):
    params = dict()
    if algorithm == "Decision Tree":
        max_depth = st.sidebar.slider("Max Depth", 2, 17)
        criterion = st.sidebar.selectbox(
            "Criterion", ("absolute_error", "squared_error", "poisson", "friedman_mse"))
        splitter = st.sidebar.selectbox("Splitter", ("best", "random"))
        params["max_depth"] = max_depth
        params["criterion"] = criterion
        params["splitter"] = splitter
        try:
            random = st.sidebar.text_input("Enter Random State")
            params["random_state"] = int(random) if random != "" else 4567
        except:
            params["random_state"] = 4567
    elif algorithm == "Linear Regression":
        fit_intercept = st.sidebar.selectbox(
            "Fit Intercept", ("True", "False"))
        n_jobs = st.sidebar.selectbox("Number of Jobs", (None, -1))
        params["fit_intercept"] = bool(fit_intercept == "True")
        params["n_jobs"] = n_jobs
    else:
        max_depth = st.sidebar.slider("Max Depth", 2, 17)
        n_estimators = st.sidebar.slider("Number of Estimators", 1, 90)
        criterion = st.sidebar.selectbox(
            "Criterion", ("absolute_error", "squared_error", "poisson", "friedman_mse"))
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["criterion"] = criterion
        try:
            random = st.sidebar.text_input("Enter Random State")
            params["random_state"] = int(random) if random != "" else 4567
        except:
            params["random_state"] = 4567
    return params


def model_classifier(algorithm, params):
    if algorithm == "KNN":
        return KNeighborsClassifier(n_neighbors=params["K"], weights=params["weights"])
    elif algorithm == "SVM":
        return SVC(C=params["C"], kernel=params["kernel"])
    elif algorithm == "Decision Tree":
        return DecisionTreeClassifier(criterion=params["criterion"], splitter=params["splitter"], random_state=params["random_state"])
    elif algorithm == "Naive Bayes":
        return GaussianNB()
    elif algorithm == "Random Forest":
        return RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], criterion=params["criterion"], random_state=params["random_state"])
    elif algorithm == "Linear Regression":
        return LinearRegression(fit_intercept=params["fit_intercept"], n_jobs=params["n_jobs"])
    else:
        return LogisticRegression(fit_intercept=params["fit_intercept"], penalty=params["penalty"], C=params["C"], n_jobs=params["n_jobs"])


def model_regressor(algorithm, params):
    if algorithm == "KNN":
        return KNeighborsRegressor(n_neighbors=params["K"], weights=params["weights"])
    elif algorithm == "SVM":
        return SVR(C=params["C"], kernel=params["kernel"])
    elif algorithm == "Decision Tree":
        return DecisionTreeRegressor(criterion=params["criterion"], splitter=params["splitter"], random_state=params["random_state"])
    elif algorithm == "Random Forest":
        return RandomForestRegressor(n_estimators=params["n_estimators"], max_depth=params["max_depth"], criterion=params["criterion"], random_state=params["random_state"])
    else:
        return LinearRegression(fit_intercept=params["fit_intercept"], n_jobs=params["n_jobs"])


# --- Improved info display that handles both sklearn datasets and dataframes ---
def info(data_name, algorithm, algorithm_type, data, X, Y):
    """Display dataset information in a Streamlit-friendly way.

    This function avoids passing DataFrame/Styler objects into st.markdown directly
    (which can trigger StreamlitAPIException: _repr_html_() is not a valid Streamlit command).
    Instead it uses st.table / st.write which are the supported display functions.
    """
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    # For sklearn-style datasets (Bunch with .data and .target)
    if hasattr(data, "data") and hasattr(data, "target"):
        if algorithm_type == "Classifier":
            st.markdown(f"### {data_name} — Classification Dataset")
        else:
            st.markdown(f"### {data_name} — Regression Dataset")

        st.write(f"**Algorithm:** {algorithm} {algorithm_type}")
        st.write("Shape of dataset:", X.shape)

        if hasattr(data, "target_names"):
            df = pd.DataFrame({"Target Value": list(
                np.unique(Y)), "Target Name": data.target_names})
            st.write("Values and names of classes:")
            # Use st.table (safe) instead of st.markdown with HTML
            st.table(df)

    else:
        # Generic DataFrame-backed dataset
        st.markdown(f"### {data_name}")
        st.write(f"**Algorithm:** {algorithm} {algorithm_type}")
        st.write("Shape of dataset:", X.shape)

        # If a DataFrame-like object is passed in, show a small preview safely.
        try:
            df_preview = pd.DataFrame(data).head(10)
            st.write("Preview of dataset (first 10 rows):")
            st.dataframe(df_preview)
        except Exception:
            # fallback: nothing to preview
            pass

    st.markdown("</div>", unsafe_allow_html=True)

# --- Plotting helpers (kept similar to original but with sizing) ---


def choice_classifier(data, data_name, X, Y):
    fig = plt.figure(figsize=(7, 5))
    if data_name == "Diabetes":
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap="viridis", alpha=0.8)
        plt.title("Scatter Classification Plot of Dataset")
        plt.colorbar()
    elif data_name == "Digits":
        colors = sns.color_palette("tab10", 10)
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y, palette=colors, alpha=0.6)
        plt.title("Digits dataset (PCA)")
    elif data_name == "Salary":
        sns.scatterplot(x=data["YearsExperience"], y=data["Salary"], data=data)
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.title("Salary vs Experience")
    elif data_name == "Naive Bayes Classification":
        colors = sns.color_palette("Set2", 2)
        sns.scatterplot(x=data["glucose"], y=data["bloodpressure"],
                        data=data, hue=Y, palette=colors, alpha=0.6)
        plt.xlabel("Glucose")
        plt.ylabel("Blood Pressure")
        plt.title("Naive Bayes data")
    else:
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y, palette=sns.color_palette(
            ["#7c3aed", "#06b6d4", "#f59e0b", "#ef4444"]), alpha=0.6)
        plt.title("Scatter Plot (PCA)")
    return fig


def choice_regressor(X, x_test, predict, data, data_name, Y, fig):
    fig = plt.figure(figsize=(7, 5))
    if data_name == "Diabetes":
        plt.scatter(X[:, 0], Y, c=Y, cmap="viridis", alpha=0.4)
        plt.plot(x_test, predict, color="red")
        plt.title("Regression: Diabetes")
        plt.legend(["Actual Values", "Prediction"])
    elif data_name == "Salary":
        sns.scatterplot(x=data["YearsExperience"], y=data["Salary"], data=data)
        plt.plot(x_test, predict, color="red")
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.legend(["Actual Values", "Prediction"])
    else:
        plt.scatter(X[:, 0], X[:, 1], c=Y, alpha=0.5)
        plt.plot(x_test, predict, color="red")
        plt.title("Regression (PCA)")
    return fig


def pca_plot(data_name, X):
    pca = PCA(2)
    if data_name != "Salary":
        try:
            X = pca.fit_transform(X)
        except Exception:
            # if PCA fails because X is not numeric or shape issues, leave X as-is
            pass
    return X


# --- Main pipeline (keeps original behaviour with nicer layout) ---
def data_model_description(algorithm, algorithm_type, data_name, data, X, Y):
    # display info
    info(data_name, algorithm, algorithm_type, data, X, Y)

    if (algorithm_type == "Regressor") and (algorithm in ["Decision Tree", "Random Forest", "Linear Regression"]):
        params = add_parameter_regressor(algorithm)
    else:
        params = add_parameter_classifier_general(algorithm)

    if algorithm_type == "Regressor":
        algo_model = model_regressor(algorithm, params)
    else:
        algo_model = model_classifier(algorithm, params)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)
    algo_model.fit(x_train, y_train)

    predict = algo_model.predict(x_test)
    X_pca = pca_plot(data_name, X)

    if algorithm_type == "Regressor":
        fig = choice_regressor(X_pca, x_test, predict,
                               data, data_name, Y, None)
    else:
        fig = choice_classifier(data, data_name, X_pca, Y)

    st.pyplot(fig)

    # Metrics area
    col1, col2 = st.columns(2)
    with col1:
        if algorithm != "Linear Regression" and algorithm_type != "Regressor":
            st.metric("Train Accuracy",
                      f"{algo_model.score(x_train, y_train) * 100:.2f}%")
            st.metric("Test Accuracy",
                      f"{accuracy_score(y_test, predict) * 100:.2f}%")
        else:
            st.write("Mean Squared Error:",
                     mean_squared_error(y_test, predict))
            st.write("Mean Absolute Error:",
                     mean_absolute_error(y_test, predict))

    with col2:
        st.write("Model: ", type(algo_model).__name__)
        st.write("Parameters:")
        st.json(params)


# --- Header + sidebar layout and user upload option ---

def main():
    # top header
    logo_col1, logo_col2 = st.columns([0.08, 0.92])
    with logo_col1:
        # optional: user can add their own logo file named 'logo.png' in repo or use a URL
        try:
            st.image("logo.png", width=64)
        except Exception:
            # fallback emoji
            st.markdown("<div style='font-size:42px'>🚀</div>",
                        unsafe_allow_html=True)
    with logo_col2:
        st.markdown("<div class='app-header'><div><h1 class='app-title'>HyperTuneML</h1><div class='app-sub'>Interactive ML explorer — customize & experiment</div></div></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Sidebar controls
    st.sidebar.header("Configuration")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV to use as dataset (optional)", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.sidebar.success("Loaded uploaded CSV")
            custom_dataset_name = st.sidebar.text_input(
                "Name for uploaded dataset", value="Uploaded CSV")
            data_name = custom_dataset_name
        except Exception as e:
            st.sidebar.error("Failed to read CSV: " + str(e))
            data = None
            data_name = None
    else:
        data_name = st.sidebar.selectbox(
            "Select Dataset",
            (
                "Iris",
                "Breast Cancer",
                "Wine",
                "Diabetes",
                "Digits",
                "Salary",
                "Naive Bayes Classification",
                "Car Evaluation",
                "Heart Disease Classification",
                "Titanic",
            ),
        )
        data = load_dataset(data_name)

    algorithm = st.sidebar.selectbox(
        "Select Algorithm",
        (
            "KNN",
            "SVM",
            "Decision Tree",
            "Naive Bayes",
            "Random Forest",
            "Linear Regression",
            "Logistic Regression",
        ),
    )

    if algorithm not in ["Linear Regression", "Logistic Regression", "Naive Bayes"]:
        algorithm_type = st.sidebar.selectbox(
            "Select Algorithm Type", ("Classifier", "Regressor"))
    else:
        if algorithm == "Linear Regression":
            algorithm_type = "Regressor"
        else:
            algorithm_type = "Classifier"

    # Show quick help and credits
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Author:** {AUTHOR_NAME}")
    st.sidebar.markdown(
        "Based on an original demo project (MIT licensed). Keep LICENSE in repo.")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Tip: upload a CSV with the target in the last column to quickly test your own dataset.")

    if data is None:
        st.error(
            "No dataset loaded — upload a CSV or choose a dataset from the sidebar.")
        return

    # compute X, Y
    X, Y = Input_output(data, data_name)

    # main content: run experiment
    data_model_description(algorithm, algorithm_type, data_name, data, X, Y)

    st.markdown("---")
    st.markdown(
        f"<div class='footer'>Made by {AUTHOR_NAME}</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
