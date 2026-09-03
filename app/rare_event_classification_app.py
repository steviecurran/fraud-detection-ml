from pathlib import Path
import sys
import matplotlib.ticker as ticker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier


# ---------------------------------------------------------------------
# Streamlit setup
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Rare-Event Classification Workbench",
    page_icon="🎯",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Match the typography used across the portfolio apps. */
    div[data-testid="stWidgetLabel"] p {
        color: #111111 !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }
    [data-testid="stRadio"] label {
        color: #222222 !important;
        font-size: 16px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 24px !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 13px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Rare-Event Classification Workbench")

st.markdown(
    """
### How do you find rare events without overwhelming investigators with false alarms?

This application demonstrates a fraud-detection workflow for an extremely imbalanced
classification problem. The original case study contains only **1.5% suspect cases**, so
raw accuracy is a poor measure of usefulness.

The workbench focuses instead on the operational trade-off between **precision, recall,
F1 score and alert rate**, and shows how class-imbalance handling and the classification
threshold change the number of cases an investigation team would need to review.
    """
)


# ---------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_csv(source):
    return pd.read_csv(source)


def normalise_remote_url(url: str) -> str:
    cleaned = url.strip()
    marker = "drive.google.com/file/d/"
    if marker in cleaned:
        file_id = cleaned.split(marker, 1)[1].split("/", 1)[0]
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return cleaned


def dataframe_memory_mb(dataframe: pd.DataFrame) -> float:
    return float(dataframe.memory_usage(index=True, deep=True).sum() / 1024**2)


def class_summary(target: pd.Series, positive_value) -> dict:
    positive = int((target == positive_value).sum())
    total = int(len(target))
    negative = total - positive
    rate = positive / total if total else np.nan
    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "rate": rate,
    }


def prepare_xy(
    dataframe: pd.DataFrame,
    target_column: str,
    positive_value,
    excluded_columns: tuple[str, ...],
):
    working = dataframe.copy()
    y = (working[target_column] == positive_value).astype(int)

    candidate = working.drop(
        columns=[target_column, *excluded_columns],
        errors="ignore",
    )
    X = candidate.select_dtypes(include="number").copy()

    # Keep the workflow robust for user-supplied data.
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    constant_columns = [
        column for column in X.columns if X[column].nunique(dropna=False) <= 1
    ]
    if constant_columns:
        X = X.drop(columns=constant_columns)

    return X, y, constant_columns


# ---------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------

MODEL_NAMES = [
    "Gradient Boost", # Since index=0 points at position 0 ("XGBoost"),
                      #just reorder the list so Gradient Boost is first:
    "XGBoost",
    "Logistic Regression",
    "Decision Tree",
]

IMBALANCE_METHODS = [
    "Class weighting",
    "SMOTE",
    "Under-sampling",
]


def build_classifier(model_name: str, class_ratio: float):
    if model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=2,
            eval_metric="logloss",
            scale_pos_weight=class_ratio,
        )

    if model_name == "Gradient Boost":
        # GradientBoostingClassifier has no class_weight parameter --
        # class balancing is applied via sample_weight at fit time
        # instead (see fit_and_score_model below).
        return GradientBoostingClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
        )

    if model_name == "Logistic Regression":
        return LogisticRegression(
            C=0.08858667,
            solver="newton-cg",
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )

    return DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42,
    )


@st.cache_resource(show_spinner=False)
def fit_and_score_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    imbalance_method: str,
    test_fraction: float,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_fraction,
        stratify=y,
        random_state=42,
    )

    ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    classifier = build_classifier(model_name, ratio)

    steps = [("scale", StandardScaler())]

    if imbalance_method == "SMOTE":
        # Remove class weights when explicit resampling is used.
        if model_name == "XGBoost":
            classifier.set_params(scale_pos_weight=1.0)
        elif model_name != "Gradient Boost":
            classifier.set_params(class_weight=None)
        steps.append(("sampler", SMOTE(random_state=42)))

    elif imbalance_method == "Under-sampling":
        if model_name == "XGBoost":
            classifier.set_params(scale_pos_weight=1.0)
        elif model_name != "Gradient Boost":
            classifier.set_params(class_weight=None)
        steps.append(
            (
                "sampler",
                RandomUnderSampler(
                    sampling_strategy="majority",
                    random_state=42,
                ),
            )
        )
    
    steps.append(("classifier", classifier))
    model = Pipeline(steps)

    fit_params = {}
    if model_name == "Gradient Boost" and imbalance_method == "Class weighting":
        # Pipeline fit-param convention: "<step_name>__<param_name>"
        fit_params["classifier__sample_weight"] = compute_sample_weight(
            class_weight="balanced", y=y_train
        )

    model.fit(X_train, y_train, **fit_params)
    

    y_prob = model.predict_proba(X_test)[:, 1]

    fitted_classifier = model.named_steps["classifier"]
    feature_importance = None

    if hasattr(fitted_classifier, "feature_importances_"):
        feature_importance = np.asarray(
            fitted_classifier.feature_importances_, dtype=float
        )
    elif hasattr(fitted_classifier, "coef_"):
        feature_importance = np.abs(
            np.asarray(fitted_classifier.coef_[0], dtype=float)
        )

    return {
        "model": model,
        "y_test": np.asarray(y_test),
        "y_prob": np.asarray(y_prob),
        "feature_names": tuple(X.columns),
        "feature_importance": feature_importance,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_positive": int((y_train == 1).sum()),
        "test_positive": int((y_test == 1).sum()),
    }


def metrics_at_threshold(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    alert_rate = float(y_pred.mean())
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "alert_rate": alert_rate,
        "alerts": int(y_pred.sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "y_pred": y_pred,
    }


@st.cache_data(show_spinner=False)
def threshold_table(y_true, y_prob):
    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        result = metrics_at_threshold(y_true, y_prob, float(threshold))
        rows.append(
            {
                "threshold": threshold,
                "f1": result["f1"],
                "precision": result["precision"],
                "recall": result["recall"],
                "alert_rate": result["alert_rate"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------

def style_plot_axes(ax, PLOT_FONT,ax_width,grid=False):
    """Apply the scientific plot style used across the portfolio apps."""
    for spine in ax.spines.values():
        spine.set_linewidth(ax_width)

    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True,
        pad=7, length=6, width=1.5, labelsize=PLOT_FONT)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True,
        length=3, width=1.2)
    ax.minorticks_on()
    if grid:
        ax.grid(True, linestyle="--", alpha=0.30, linewidth=0.8)

    return PLOT_FONT
# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def fix_log(axis, ticks, start, end):
    """Format symlog count ticks without Matplotlib math-text markup."""

    def update_ticks(value, pos):
        if not np.isfinite(value):
            return ""
        if np.isclose(value, 0):
            return "0"

        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 1:
            return f"{value:.0f}"
        return f"{value:g}"

    upper = max(float(start), float(end), 0.0)
    minor = []

    if upper >= 1:
        max_exponent = int(np.floor(np.log10(upper)))
        for exponent in range(max_exponent + 1):
            base = 10**exponent
            for multiplier in range(2, 10):
                tick = multiplier * base
                if tick <= upper:
                    minor.append(tick)

    if ticks == "xticks":
        axis.set_xticks(minor, minor=True)
        axis.xaxis.set_major_formatter(ticker.FuncFormatter(update_ticks))
    else:
        axis.set_yticks(minor, minor=True)
        axis.yaxis.set_major_formatter(ticker.FuncFormatter(update_ticks))
        

def make_class_histogram(
    dataframe,
    feature,
    y,
    bins,
    negative_label="Negative class",
    positive_label="Positive class",
):
    negative = pd.to_numeric(
        dataframe.loc[y == 0, feature], errors="coerce"
    ).dropna()
    positive = pd.to_numeric(
        dataframe.loc[y == 1, feature], errors="coerce"
    ).dropna()

    combined = pd.concat([negative, positive])
    if combined.empty:
        return None

    minimum = float(combined.min())
    maximum = float(combined.max())
    if minimum == maximum:
        minimum -= 0.5
        maximum += 0.5

    edges = np.linspace(minimum, maximum, bins + 1)
    fig, ax = plt.subplots(figsize=(8, 4.8))

    # Side-by-side bars keep class colours distinct rather than blending.
    ax.hist(
        [negative, positive],
        bins=edges,
        color=["#4C78A8", "#E45756"],
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
        label=[
            f"{negative_label} (negative, n={len(negative):,})",
            f"{positive_label} (positive, n={len(positive):,})",
        ],
    )
    ax.set_yscale("symlog")

    PLOT_FONT = style_plot_axes(ax, 13, 1.5)
    ax.set_xlabel(feature, size=PLOT_FONT)
    ax.set_ylabel("Number", size=PLOT_FONT)
    y1, y2 = ax.get_ylim()
    fix_log(ax, "yticks", y1, y2)

    ax.legend(fontsize=0.8 * PLOT_FONT, frameon=False)
    fig.tight_layout()
    return fig

def make_confusion_plot(result):
    matrix = np.array(
        [[result["tn"], result["fp"]], [result["fn"], result["tp"]]],
        dtype=float,
    )
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals != 0)

    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    image = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    labels = [["TN", "FP"], ["FN", "TP"]]

    for row in range(2):
        for col in range(2):
            cell_value = normalised[row, col]
            text_colour = "white" if cell_value >= 0.50 else "black"
            ax.text(
                col, row,
                f"{labels[row][col]}\n{int(matrix[row, col]):,}\n{cell_value:.3f}",
                ha="center", va="center", fontsize=14,
                color=text_colour,
            )

    ax.set_xticks([0, 1], ["Non-suspect", "Suspect"])
    ax.set_yticks([0, 1], ["Non-suspect", "Suspect"])
    PLOT_FONT = style_plot_axes(ax,17,2)
    ax.set_title("Confusion matrix", fontsize=PLOT_FONT, pad=12)
    ax.tick_params(axis="both", labelsize=PLOT_FONT)
    colourbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colourbar.ax.tick_params(labelsize=PLOT_FONT)
    fig.tight_layout()
    return fig

def make_pr_plot(y_true, y_prob, threshold):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    result = metrics_at_threshold(y_true, y_prob, threshold)

    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    ax.plot(recall, precision, color="#6F4E9C", linewidth=2.6, label="Precision-recall curve")
    ax.scatter(
        result["recall"], result["precision"], s=90, color="#F28E2B",
        edgecolor="black", linewidth=0.7, zorder=5,
        label=f"Threshold = {threshold:.2f}",
    )
    PLOT_FONT = style_plot_axes(ax,17,2)
    ax.set_xlabel("Recall", size=PLOT_FONT)
    ax.set_ylabel("Precision", size=PLOT_FONT)
    ax.set_title("Precision-recall curve", fontsize=PLOT_FONT, fontweight="normal", pad=12)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    
    ax.legend(fontsize=0.8*PLOT_FONT, frameon=False, loc="best")
    fig.tight_layout()
    return fig

def make_roc_plot(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    ax.plot(fpr, tpr, color="r", linewidth=2.6, label=f"ROC AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="k", linestyle="dotted", linewidth=1.8,
            label="Random classifier")
    PLOT_FONT = style_plot_axes(ax,17,2)
    ax.set_xlabel("False-positive rate", fontsize=PLOT_FONT)
    ax.set_ylabel("True-positive rate", fontsize=PLOT_FONT)
    ax.set_title("Receiver operating characteristic", fontsize=PLOT_FONT, pad=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=0.8*PLOT_FONT, frameon=False, loc="lower right")
    fig.tight_layout()
    return fig

def make_probability_separation_plot(y_true, y_prob, confidence_percent):
    rare = y_prob[y_true == 1]
    non = y_prob[y_true == 0]
    z_value = norm.ppf(0.5 + confidence_percent / 200.0)

    def mean_ci(values):
        mean = float(np.mean(values))
        if len(values) < 2:
            return mean, 0.0
        se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
        return mean, z_value * se

    rare_mean, rare_ci = mean_ci(rare)
    non_mean, non_ci = mean_ci(non)

    fig, ax = plt.subplots(figsize=(6.3, 2.45))
    ax.errorbar(rare_mean, 0, xerr=rare_ci, fmt="o", markersize=9,
        color="#E45756", ecolor="#E45756", elinewidth=2.2,
        capsize=6, capthick=1.8, markeredgecolor="black", markeredgewidth=0.7,
        label="Suspect",
    )
    ax.errorbar(non_mean, 1, xerr=non_ci, fmt="o", markersize=9,
        color="#4C78A8", ecolor="#4C78A8", elinewidth=2.2,
        capsize=6, capthick=1.8, markeredgecolor="black", markeredgewidth=0.7,
        label="Non-suspect",
    )

    PLOT_FONT = style_plot_axes(ax,17,2)
    ax.set_xlabel(
        f"Mean predicted fraud probability ({confidence_percent:.1f}% CI)",fontsize=PLOT_FONT)
    ax.set_title("Predicted-probability separation",size=PLOT_FONT,pad=7)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.75, 1.75)
    

    # The y positions are arbitrary and used only to separate the two classes visually.
    ax.set_yticks([])
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
    )

    ax.legend(fontsize=0.8*PLOT_FONT, frameon=False, loc="best", ncol=1)
    fig.tight_layout(pad=0.8)
    return fig

def make_threshold_plot(table, selected_threshold):
    fig, ax = plt.subplots(figsize=(8, 5.0))
    #ax.plot(table["threshold"], table["f1"], color="silver",lw=3,zorder=1)
    ax.plot(table["threshold"], table["f1"], color="dimgrey", lw=2, label="F1 score",zorder=2)
    ax.plot(table["threshold"], table["precision"], color="b", ls="--", lw=2, label="Precision")
    ax.plot(table["threshold"], table["recall"], color="lime", ls="--",lw=2, label="Recall")
    ax.plot(table["threshold"], table["alert_rate"], color="orange", ls="--",lw=2, label="Alert rate")
    ax.axvline(selected_threshold, color="r", ls="dotted", lw=2, label="Selected threshold")
    
    PLOT_FONT = style_plot_axes(ax,13,1.5)
    ax.set_xlabel("Classification threshold",size = PLOT_FONT);ax.set_ylabel("Metric value",size = PLOT_FONT)
    ax.set_xlim(0.01, 0.99)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=0.8*PLOT_FONT, ncol=2) #, frameon=False)
    fig.tight_layout()
    return fig

def make_feature_importance_plot(feature_names, importance):
    frame = pd.DataFrame(
        {"Feature": list(feature_names), "Importance": importance}
    ).sort_values("Importance", ascending=True)

    fig_height = max(4.0, 0.3 * len(frame))
    fig, ax = plt.subplots(figsize=(7, fig_height))
    ax.barh(frame["Feature"], frame["Importance"], color="silver",edgecolor="black", linewidth=1.5, alpha=0.85)
    PLOT_FONT = style_plot_axes(ax,10,1.0)
    ax.tick_params(axis="both", which="major", direction="in", top=False, right=False,
                   pad=7, length=3, width=1.5, labelsize=PLOT_FONT)
    ax.minorticks_off()
    ax.set_xlabel("Importance",size = PLOT_FONT)
     
    fig.tight_layout()
    return fig, frame.sort_values("Importance", ascending=False)


# ---------------------------------------------------------------------
# Sidebar data source
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("Data source")

    source_type = st.radio(
        "Choose source",
        [
            "Built-in fraud case study",
            "Upload your own CSV",
            "Remote CSV URL",
        ],
    )

    data = None
    source_label = None

    if source_type == "Built-in fraud case study":
        built_in_path = DATA_DIR / "synthetic_gambling_aml.csv"
        if built_in_path.exists():
            data = load_csv(built_in_path)
            source_label = "Built-in fraud-detection case study"
        else:
            st.error(
                "Could not find `data/synthetic_gambling_aml.csv`. Add the merged modelling table "
                "to the repository's data directory."
            )

    elif source_type == "Upload your own CSV":
        uploaded = st.file_uploader("CSV file", type=["csv"])
        if uploaded is not None:
            try:
                data = load_csv(uploaded)
                source_label = f"Uploaded file: {uploaded.name}"
            except Exception as error:
                st.error(f"Could not read the uploaded file. {error}")

    else:
        remote_url = st.text_input(
            "Public CSV URL",
            placeholder="Paste a Google Drive or other public CSV URL",
        )
        if remote_url:
            try:
                data = load_csv(normalise_remote_url(remote_url))
                source_label = "Remote CSV"
            except Exception as error:
                st.error(f"Could not load the remote CSV. {error}")

    if data is not None:
        st.markdown("### Loaded dataset")
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{len(data):,}")
        c2.metric("Memory", f"{dataframe_memory_mb(data):.1f} MB")
        st.caption(source_label or "Loaded dataset")


if data is None:
    st.info("Choose or load a dataset from the sidebar to begin.")
    st.stop()


# ---------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------

all_columns = data.columns.tolist()

if source_type == "Built-in fraud case study" and "Target_ml" in data.columns:
    target_column = "Target_ml"
    positive_value = 1
    excluded_default = ["ID"] if "ID" in data.columns else []
else:
    st.sidebar.divider()
    st.sidebar.subheader("Configure classification")
    target_column = st.sidebar.selectbox(
        "Target column",
        all_columns,
        index=(all_columns.index("Target_ml") if "Target_ml" in all_columns else len(all_columns) - 1),
    )
    if (
        pd.api.types.is_object_dtype(data[target_column])
        or pd.api.types.is_string_dtype(data[target_column])
    ):
        data[target_column] = data[target_column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    target_values = data[target_column].dropna().unique().tolist()
    if len(target_values) != 2:
        st.error(
            "The selected target must contain exactly two classes for this binary "
            "rare-event classification workbench."
        )
        st.stop()
    positive_value = st.sidebar.selectbox(
        "Rare / positive class",
        target_values,
        index=1 if len(target_values) > 1 else 0,
    )
    identifier_guess = [column for column in all_columns if column.lower() in {"id", "index"}]
    excluded_default = identifier_guess[:1]

_target_values = data[target_column].dropna().unique().tolist()
_negative_values = [value for value in _target_values if value != positive_value]
negative_value = _negative_values[0] if _negative_values else None

if source_type == "Built-in fraud case study" and target_column == "Target_ml":
    negative_class_label = "Non-suspect"
    positive_class_label = "Suspect"
else:
    negative_class_label = str(negative_value)
    positive_class_label = str(positive_value)

excluded_columns = st.sidebar.multiselect(
    "Exclude identifier / non-predictor columns",
    [column for column in all_columns if column != target_column],
    default=excluded_default,
)

X, y, constant_columns = prepare_xy(
    data,
    target_column,
    positive_value,
    tuple(excluded_columns),
)

if X.empty:
    st.error("No usable numeric predictor columns remain after configuration.")
    st.stop()

summary = class_summary(data[target_column], positive_value)


# ---------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------

tabs = st.tabs(
    [
        "Dataset overview",
        "Feature Explorer",
        "Model Workbench",
        "Threshold Tuning",
        "Interpretation",
    ]
)


with tabs[0]:
    st.subheader("Dataset overview")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{summary['total']:,}")
    m2.metric("Suspect / positive", f"{summary['positive']:,}")
    m3.metric("Non-suspect / negative", f"{summary['negative']:,}")
    m4.metric("Rare-event rate", f"{100 * summary['rate']:.2f}%")

    naive_accuracy = 1 - summary["rate"]
    st.markdown(
        f"""
If every case were simply predicted as **non-suspect**, the model would appear to achieve
**{100 * naive_accuracy:.2f}% accuracy** while detecting **none** of the rare events.
That is why this project evaluates precision, recall, F1 and alert workload instead of
optimising raw accuracy.
        """
    )

    if constant_columns:
        st.caption(
            "Constant columns automatically excluded from modelling: "
            + ", ".join(constant_columns)
        )

    with st.expander("Preview data"):
      
        total_rows = len(data)
        preview_cap = 20_000

        st.markdown("**Choose number of rows to display**")

        # Very small datasets: just show everything
        if total_rows <= 10:
            preview_rows = total_rows
            st.caption(f"Showing all {total_rows:,} rows.")

        else:
            slider_max = min(total_rows, preview_cap)

            # Generate logarithmically spaced row counts
            row_options = np.geomspace(
                10,
                slider_max,
                num=50,
            )

            # Convert to integers, remove duplicates and ensure max is included
            row_options = sorted(
                set(int(round(value)) for value in row_options)
            )

            if slider_max not in row_options:
                row_options.append(slider_max)

            # Default as close as possible to 100 rows
            default_rows = min(100, slider_max)

            default_value = min(
                row_options,
                key=lambda value: abs(value - default_rows),
            )

            preview_rows = st.select_slider(
                "Rows to preview",
                options=row_options,
                value=default_value,
                format_func=lambda value: f"{value:,}",
                label_visibility="collapsed",
            )

            # For very large datasets, optionally display everything
            if total_rows > preview_cap:
                show_all = st.checkbox(
                    f"Show all {total_rows:,} rows"
                )

                if show_all:
                    preview_rows = total_rows

            st.caption(
                f"Showing {preview_rows:,} of {total_rows:,} rows."
            )

        st.dataframe(
            data.head(preview_rows),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Descriptive statistics"):
        st.dataframe(X.describe().T, use_container_width=True)


with tabs[1]:
    st.subheader("Feature Explorer")
    st.markdown(
        "Compare the distribution of any numeric predictor between the rare and majority classes. "
        "The y-axis uses a symlog scale because the classes are extremely imbalanced."
    )

    f1, f2 = st.columns([1, 1])
    with f1:
        feature = st.selectbox("Feature", list(X.columns), index=0)
    with f2:
        number_of_bins = st.select_slider("Number of histogram bins",
            options=[5, 10, 20, 30, 50, 75, 100],value=30,
        )

    explorer = pd.DataFrame({feature: X[feature], "_class": y})
    figure = make_class_histogram(
        explorer,
        feature,
        explorer["_class"],
        number_of_bins,
        negative_class_label,
        positive_class_label,
    )
    if figure is not None:
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)

    stats_rows = []
    for class_value, label in [
        (0, negative_class_label),
        (1, positive_class_label),
    ]:
        values = X.loc[y == class_value, feature]
        stats_rows.append(
            {
                "Class": label,
                "n": len(values),
                "Mean": values.mean(),
                "Std. dev.": values.std(ddof=1),
                "Median": values.median(),
            }
        )
    st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)


with tabs[2]:
    st.subheader("Model Workbench")
    st.markdown(
        "Fit a classifier on a stratified 80/20 split. Resampling is applied only to the "
        "training data; the held-out test set retains the real rare-event rate."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        model_name = st.selectbox(
            "Classifier",
            MODEL_NAMES,
            index=0,
            key="model_name",
        )
        with st.popover("ℹ️ About classifiers"):
            st.markdown(
                """
                **XGBoost**  
                An ensemble of decision trees built sequentially to improve prediction errors.
                Often performs very well on structured tabular data and can model complex
                non-linear relationships.

               **Gradient Boost**  
                Similar in spirit to XGBoost — trees are added sequentially, each correcting
                the previous
                ensemble's errors. Achieves the strongest recall and PR AUC of the models
                compared in this
                workbench's case study, at the cost of a higher alert rate.

               **Logistic Regression**  
               A simple, interpretable linear classifier that estimates the probability of
                the positive class.
               Useful as a strong baseline and when transparency matters.

               **Decision Tree**  
                Splits the data into rule-based branches to make predictions. Easy to interpret,
                but can overfit more readily than ensemble methods.
                """
            )
    with c2:
        imbalance_method = st.selectbox(
        "Class-imbalance method",
        IMBALANCE_METHODS,
        index=0,
        key="imbalance_method",
    )
        with st.popover("ℹ️ Class-imbalance methods"):
            st.markdown(
                """
                **Class weighting**  
                Gives greater importance to the rare class during model training without changing the data.

                **SMOTE**  
                Creates synthetic minority-class examples to increase representation of the rare class in the training data.

                **Under-sampling**  
                Reduces the number of majority-class examples used for training, creating a more balanced training set.
                """
            )
        
    with c3:
        test_fraction = st.select_slider(
            "Test fraction",
            options=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
            value=0.20,
            key="test_fraction",
        )

    if model_name == "XGBoost":
        st.caption(
            "The XGBoost settings match the notebook case study: 500 estimators, depth 5, "
            "learning rate 0.03, with class weighting selected by default."
        )
    elif model_name == "Gradient Boost":
        st.caption(
            "The Gradient Boost settings match the notebook case study: 300 estimators, depth 3, "
            "learning rate 0.05, with class balancing applied via sample weighting by default. "
            "This configuration achieves the strongest documented result in this workbench "
            "(recall ≈48% vs. XGBoost's ≈25%, at a comparable precision)."
    )

    run_model = st.button("Run model", type="primary")

    config_signature = (
        model_name,
        imbalance_method,
        test_fraction,
        tuple(X.columns),
        len(X),
        int(y.sum()),
    )

    if run_model:
        with st.spinner("Fitting model and scoring the untouched test set..."):
            try:
                result = fit_and_score_model(
                    X,
                    y,
                    model_name,
                    imbalance_method,
                    test_fraction,
                )
                st.session_state["rare_event_result"] = result
                st.session_state["rare_event_signature"] = config_signature
            except Exception as error:
                st.exception(error)

    result = st.session_state.get("rare_event_result")
    stored_signature = st.session_state.get("rare_event_signature")

    if result is None:
        st.info("Choose the model settings and click **Run model**.")
    else:
        if stored_signature != config_signature:
            st.warning(
                "The controls have changed since the displayed model was fitted. "
                "Click **Run model** to update the results."
            )

        # Initialise the Model Workbench threshold at the value that maximises
        # F1 for the currently fitted model. Reset it whenever the fitted model
        # configuration changes, but preserve the user's manual slider choice
        # during ordinary Streamlit reruns.
        workbench_thresholds = threshold_table(
            result["y_test"],
            result["y_prob"],
        )
        best_workbench_row = workbench_thresholds.loc[
            workbench_thresholds["f1"].idxmax()
        ]
        best_workbench_threshold = float(
            round(best_workbench_row["threshold"], 2)
        )

        threshold_state_key = "model_workbench_threshold"
        threshold_signature_key = "model_workbench_threshold_signature"

        if st.session_state.get(threshold_signature_key) != stored_signature:
            st.session_state[threshold_state_key] = best_workbench_threshold
            st.session_state[threshold_signature_key] = stored_signature

        decision_col, threshold_note_col = st.columns(2)

        with decision_col:
            threshold = st.slider(
                "Classification threshold",
                min_value=0.01,
                max_value=0.99,
                step=0.01,
                key=threshold_state_key,
                help=(
                    "The slider starts at the threshold that maximises F1 for the "
                    "current fitted model. Moving it changes the classification "
                    "decision without retraining the model."
                ),
            )
            st.caption(
                f"The default threshold ({best_workbench_threshold:.2f}) is selected by "
                "sweeping candidate thresholds from 0.01 to 0.99 and choosing the value "
                "that maximises F1. Move the slider to inspect another operating point; "
                "the full precision–recall trade-off can be explored in the Threshold "
                "Tuning tab."
            )

        with threshold_note_col:
            st.markdown(
                "**How to read the threshold**"
            )
            st.caption(
                "A higher threshold usually produces fewer alerts and fewer false positives, "
                "but can miss more true rare events. A lower threshold usually increases recall "
                "at the cost of a larger investigation workload."
            )

        current = metrics_at_threshold(
            result["y_test"],
            result["y_prob"],
            threshold,
        )

        a, b, c, d = st.columns(4)
        a.metric("Precision", f"{current['precision']:.3f}")
        b.metric("Recall", f"{current['recall']:.3f}")
        c.metric("F1 score", f"{current['f1']:.3f}")
        d.metric("Alert rate", f"{100 * current['alert_rate']:.2f}%")

        st.caption(
            f"Test set: {result['test_rows']:,} cases, including "
            f"{result['test_positive']:,} rare events. At this threshold, "
            f"{current['alerts']:,} cases would be sent for review."
        )

        left, right = st.columns(2)
        with left:
            fig = make_confusion_plot(current)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with right:
            fig = make_pr_plot(result["y_test"], result["y_prob"], threshold)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        left, right = st.columns(2)
        with left:
            fig = make_roc_plot(result["y_test"], result["y_prob"])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with right:
            confidence = st.session_state.get("probability_ci", 95.0)
            fig = make_probability_separation_plot(
                result["y_test"],
                result["y_prob"],
                confidence,
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.select_slider(
                "Probability mean confidence interval",
                options=[80.0,90.0, 95.0, 99.0, 99.9, 99.99,99.999],
                value=confidence,
                key="probability_ci",
                help=(
                    "Controls the uncertainty interval around the mean predicted probability "
                    "for each class. It does not change the classifier or its predictions."
                ),
            )
            st.caption(
                "The points show the mean predicted probability for suspect and non-suspect cases; "
                "the error bars show uncertainty around each mean. Increasing the confidence level "
                "widens the interval but does not change the model predictions."
            )

        if result["feature_importance"] is not None:
            with st.expander("Feature importance"):
                fig, importance_table = make_feature_importance_plot(
                    result["feature_names"],
                    result["feature_importance"],
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.dataframe(
                    importance_table,
                    use_container_width=True,
                    hide_index=True,
                )


with tabs[3]:
    st.subheader(" Threshold Tuning")
    st.markdown(
        "The classification threshold is an operational decision, not just a model parameter. "
        "Increasing it generally reduces false positives and investigation workload, but also "
        "allows more true rare events to go undetected."

        " F1 combines precision and recall, rewarding models that detect rare events while limiting false positives."
    )

    result = st.session_state.get("rare_event_result")

    if result is None:
        precomputed_path = DATA_DIR / "thresholds.csv"
        if source_type == "Built-in fraud case study" and precomputed_path.exists():
            raw_thresholds = load_csv(precomputed_path)
            rename_map = {
                "f-score": "f1",
                "alert": "alert_rate",
            }
            table = raw_thresholds.rename(columns=rename_map)
            table = table[["threshold", "f1", "precision", "recall", "alert_rate"]]
            
            st.info(
                "Showing the precomputed Gradient Boost + class-weighting threshold sweep from the "
                "notebook (the strongest documented result). Run a different model in Model "
                "Workbench to generate a new sweep live."
    
            )
        else:
            st.info("Run a model in **Model Workbench** to explore its threshold trade-offs.")
            table = None
    else:
        table = threshold_table(result["y_test"], result["y_prob"])

    if table is not None:
        best_row = table.loc[table["f1"].idxmax()]
        threshold_choice = st.slider(
            "Inspect threshold",
            min_value=0.01,
            max_value=0.99,
            value=float(round(best_row["threshold"], 2)),
            step=0.01,
            key="threshold_tuning_slider",
        )

        nearest_index = (table["threshold"] - threshold_choice).abs().idxmin()
        selected = table.loc[nearest_index]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{selected['precision']:.3f}")
        m2.metric("Recall", f"{selected['recall']:.3f}")
        m3.metric("F1 score", f"{selected['f1']:.3f}")
        m4.metric("Alert rate", f"{100 * selected['alert_rate']:.2f}%")

        st.markdown(
            f"""
            <div style="font-size: 20px; font-weight: 600;">
            Maximum F1 = {best_row['f1']:.3f} at threshold {best_row['threshold']:.2f}.
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        fig = make_threshold_plot(table, threshold_choice)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        with st.expander("Threshold table"):
            display_table = table.copy()
            display_table["alert_rate"] *= 100
            display_table = display_table.rename(
                columns={"alert_rate": "alert_rate_percent"}
            )
            st.dataframe(display_table, use_container_width=True, hide_index=True)


with tabs[4]:
    st.subheader("Interpretation")

    st.markdown(
        f"""
The case study begins with a rare-event prevalence of only **{100 * summary['rate']:.2f}%**.
A classifier that predicts the majority class every time can therefore appear highly accurate
while being useless for detection.

The useful question is operational:

> **How many genuine rare events can we identify for the amount of investigation workload we create?**

Class weighting, SMOTE and under-sampling alter how the model learns from the minority class.
The probability threshold then determines how those model scores are converted into alerts.
A higher threshold normally improves precision and reduces the alert rate, but at the cost of
lower recall.
        """
    )

    result = st.session_state.get("rare_event_result")
    if result is not None:
        table = threshold_table(result["y_test"], result["y_prob"])
        best = table.loc[table["f1"].idxmax()]
        best_metrics = metrics_at_threshold(
            result["y_test"],
            result["y_prob"],
            float(best["threshold"]),
        )

        st.markdown(
            f"""
For the currently fitted model, the threshold that maximises F1 is approximately
**{best['threshold']:.2f}**, producing **{best['precision']:.1%} precision**,
**{best['recall']:.1%} recall**, **F1 = {best['f1']:.3f}**, and an alert rate of
**{best['alert_rate']:.2%}** on the held-out test set.

That corresponds to **{best_metrics['alerts']:,} alerts** from
**{result['test_rows']:,} test cases**.
            """
        )
    elif source_type == "Built-in fraud case study":
        st.markdown(
            """
            In the original notebook, **XGBoost with class weighting** achieved the strongest balance.
At a threshold of approximately **0.65**, it produced roughly **20.0% precision**, **25.3% recall**,
**F1 ≈ 0.224**, and an alert rate of approximately **1.90%**. This illustrates why threshold selection
matters: the aim is not merely to rank fraud risk, but to create a review queue that is both
useful and operationally manageable.
            """
        )

    st.caption(
        "This application is an analytical demonstration. Model performance depends on the "
        "data-generating process, feature quality, class prevalence and the relative costs of "
        "false positives and false negatives."
    )
