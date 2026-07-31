# Imports

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import math


# Analyze missingness

def missing_summary(df, vars=None):
    """ 
    Summarize missingess rate and count for each variable in the df
    If 'vars' is None, summarizes all variables in the df
    """

    # Define which cols to use
    cols = vars if vars is not None else df.columns

    summary = pd.DataFrame({
        "Count missing": df[cols].isna().sum(),
        "Percent missing": (df[cols].isna().mean()*100).round(2),
        "Number remaining": df[cols].notna().sum()
    })
    
    return summary.sort_values("Percent missing", ascending=False)


def attrition_table(df, flag_col, labels=None):
    total = len(df)
    eligible = df[flag_col].sum()
    flag = df[flag_col].isna().sum()  

    ineligible = total - eligible - flag

    return pd.DataFrame({
        "Stage": ["Total sample", "Eligible", "Excluded"],
        "N": [total, eligible, ineligible],
        "Pct of total": [100.0, round(100*eligible/total, 1), round(100*ineligible/total, 1)]
    })


# Demonstrate who is included/not in our study
def summarize_cohort(df, flag_col, compare_cols):
    # summarize the subjects being used in the analysis and compare them to
    # those excluded

    included = df[df[flag_col] == True]
    excluded = df[df[flag_col] == False]
    rows = []
    for col in compare_cols:
        if df[col].dtype in ["float64", "int64", "bool"]:
            rows.append({
                "Variable": col,
                "Included mean": round(included[col].mean(), 2),
                "Excluded mean": round(excluded[col].mean(), 2),
                "Included n": included[col].notna().sum(),
                "Excluded n": excluded[col].notna().sum(),
            })
        else:
            # categorical 
            inc_pct = included[col].value_counts(normalize=True) * 100
            exc_pct = excluded[col].value_counts(normalize=True) * 100
            for cat in df[col].dropna().unique():
                rows.append({
                    "Variable": f"{col} = {cat}",
                    "Included mean": round(inc_pct.get(cat, 0), 1),
                    "Excluded mean": round(exc_pct.get(cat, 0), 1),
                    "Included n": included[col].eq(cat).sum(),
                    "Excluded n": excluded[col].eq(cat).sum(),
                })
    return pd.DataFrame(rows)



""" 
EDA 
- Define functions for examining and plotting for later use
- The functions are called later in each model section
- Descriptive statistics, distribution plots, outlier detection, relationship visualization
"""

# Descriptive statistics
def summarize_cont_columns(df, cols):
    # Returns a table of the continuous variables, their mean,
    # standard deviation, median, IQR
    # Age, BMI, HbA1c, hsCRP, jetlag

    desc = df[cols].describe().round(2).T

    return desc


def summarize_cat_columns(df, cols):
    # Returns a table of the categorical variables, their categories,
    # the count, and percentage
    # Sex, race, smoking, PA?

    rows = []
    for col in cols:
        counts = df[col].value_counts()
        pct = df[col].value_counts(normalize=True) * 100
        rows.append(pd.DataFrame({'Variable': col, "Category": counts.index, "Count":counts.values, "pct": pct.values.round(1)}))

    return pd.concat(rows, ignore_index=True)


def style_axes(ax):
    """Consistent styling for all plots"""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(axis="y", alpha=0.25)

    # Don't need to repeat X-axis label, same as title
    ax.set_xlabel("")
    ax.tick_params(labelsize=8)


"""
Distribution plots

"""

def plot_dist(df, column, ax=None):
    # For each continuous variable, plot the distribution

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    #n_unique = df[column].nunique(dropna=True)
    #bins = min(30, n_unique) if n_unique <= 30 else 30

    sns.histplot(
        data=df,
        x=column,
        bins="auto",
        ax=ax,
        edgecolor="white",
        linewidth=0.5
    )
    ax.set_title(
        column.replace("_", " ").title(),
        fontsize=10,
        fontweight="bold"
    )
    lower = df[column].quantile(0.005)
    upper = df[column].quantile(0.995)
    ax.set_xlim(lower, upper)
    ax.set_ylabel("Count")

    style_axes(ax)


def plot_counts(df, column, ax=None):
    # plot the counts for each categorical variable
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    sns.countplot(
        data=df,
        x=column,
        ax=ax
    )
    ax.set_title(
        column.replace("_", " ").title(),
        fontsize=10,
        fontweight="bold"
    )
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)

    style_axes(ax)

def plot_grid(df, columns, plot_func, figsize_per_plot=(3, 2), title="Distribution Plot"):
    """
    Plot a list of variables in a nearly square grid
    """
    n = len(columns)

    # Try to make a square grid
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_plot[0] * ncols,
                 figsize_per_plot[1] * nrows)
    )

    axes = axes.flatten()
    for ax, col in zip(axes, columns):
        plot_func(df, col, ax=ax)

    for ax in axes[n:]:
        fig.delaxes(ax)
        
    fig.suptitle(
        title,
        fontsize=20,
        fontweight="bold"
    )
    plt.tight_layout(rect=[0,0,1,.97])
    plt.show()

"""
Outlier Detection

"""


def identify_outs(df, column, ax=None):
    # Use boxplots and IQR

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 2))

    sns.boxplot(
        data=df,
        x=column,
        ax=ax
    )
    ax.set_title(
        column.replace("_", " ").title(),
        fontsize=10,
        fontweight="bold"
    )
    ax.set_xlabel("")
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outs = df[(df[column] <lower) | (df[column] > upper)]
    return outs

def get_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[column] < lower) | (df[column] > upper)]


"""
Explore relationships

"""

def plot_correlation_heatmap(df, variables):
    """
    Plot a correlation heatmap for selected continuous variables 
    Works for continuous variables and categorical encoded as numeric
    """
    method = 'pearson'

    corr = df[variables].corr(method=method)
    plt.figure(figsize=(8,6))
    ax = sns.heatmap(
        corr,
        annot=True,
        fmt=".2f", 
        cmap="coolwarm",
        square=True,
        annot_kws={"size": 6},
        cbar_kws={
            "shrink": 0.6,  
            "aspect": 20     
        }
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)

    plt.xticks(fontsize=8, rotation=45, ha="right")
    plt.yticks(fontsize=8, rotation=0)
    plt.tick_params(
        axis="both",
        which="both",
        length=4
    )

    plt.title(f"{method.capitalize()} Correlation Matrix")
    plt.tight_layout()
    plt.show()


def plot_pairplot(df, variables, hue=None):
    """
    Generates a scatter plot matrix (pairplot) with KDE diagonals
    """

    sns.pairplot(
        df[variables + ([hue] if hue else [])],
        diag_kind="kde",
        corner=True,
        hue=hue
    )

    plt.show()


"""
Explore depression scores

"""

def plot_depression_by_category(df, category, ax, outcome="phq9_score", max_categories=10):
    """
    Plot the distribution of the depression scores across levels of a
    categorical variable. If 'category' is numeric with many unique values
    then it is binned into groups.
    """

    plot = df[[category, outcome]].copy()

    # count unique categories, create bins if many unique values
    unique = plot[category].nunique()
    is_numeric = pd.api.types.is_numeric_dtype(plot[category])
    if is_numeric and unique > max_categories:
        plot[category] = pd.qcut(plot[category], q=max_categories, duplicates="drop")
        plot[category] = plot[category].apply(
                lambda iv: f"{int(iv.left)}-{int(iv.right)}"
            )
    
    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=plot,
        x=category,
        y=outcome,
        ax=ax
    )

    ax.set_title(f"{outcome} vs {category}")

    n_boxes = plot[category].nunique()
    fontsize = 10 if n_boxes <= 6 else 8
    ax.tick_params(axis='x', labelsize=fontsize)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
        label.set_rotation_mode('anchor')


def plot_scatter(df, x, y, ax):
    """
    Plots a scatter plot with regression line and 95% CI
    """
    sns.regplot(
        data=df,
        x=x,
        y=y,
        ax=ax,
        scatter_kws={"alpha": 0.4},
        line_kws={"color": "red"}
    )

    ax.set_title(f"{y} vs {x}")



"""
Column definitions
- By type
- By category to plot by

"""

# Continuous
cont_columns = [
    "weekday_reported_sleep_hours",
    "weekend_reported_sleep_hours",
    "phq9_score",
    "hba1c",
    "hs_crp",
    "log_hs_crp",
    "weekday_sleep_time_minutes",
    "weekday_wake_time_minutes",
    "weekend_sleep_time_minutes",
    "weekend_wake_time_minutes",
    "weekday_sleep_midpoint_minutes",
    "weekend_sleep_midpoint_minutes",
    "weekday_sleep_midpoint",
    "weekend_sleep_midpoint",
    "social_jetlag_hours",
    "weighted_average_sleep_duration",
    "age",
    "income_poverty_ratio",
    "bmi",
]

# Categorical
nominal_columns = [
    "sex",
    "race_ethnicity",
    "smoking_status",
]

# Categorical (ordinal)
ordinal_columns = [
    "phq9_little_interest",
    "phq9_feeling_down",
    "phq9_sleep_problems",
    "phq9_feeling_tired",
    "phq9_appetite_problems",
    "phq9_feeling_bad_about_self",
    "phq9_trouble_concentrating",
    "phq9_moving_or_speaking_slowly",
    "phq9_thoughts_better_off_dead",
    "current_smoking_frequency",
    "phq9_items_answered",
]

cat_columns = nominal_columns + ordinal_columns

# binary
binary_cols = [
    "phq9_complete",
    "depression_indicator",
    "hs_crp_gt_10",
    "vigorous_recreation",
    "moderate_recreation",
    "any_recreational_activity",
    "ever_smoked_100_cigarettes",
]

# Eligibility
eligibility_flag_columns = [
    "eligible_primary_hscrp_model",
    "eligible_hba1c_sensitivity_model",
    "eligible_hscrp_le_10_sensitivity_model",
]

interaction_columns = [
    "sjl_x_log_hs_crp",
    "sjl_x_hba1c",
]


# Define groups of variables to plot by
phq_columns = [
    "phq9_score",
    "phq9_little_interest",
    "phq9_feeling_down",
    "phq9_sleep_problems",
    "phq9_feeling_tired",
    "phq9_appetite_problems",
    "phq9_feeling_bad_about_self",
    "phq9_trouble_concentrating",
    "phq9_moving_or_speaking_slowly",
    "phq9_thoughts_better_off_dead",
]

sleep_columns = [
    "weekday_sleep_midpoint",
    "weekend_sleep_midpoint",
    "social_jetlag_hours",
    "weekday_reported_sleep_hours",
    "weekend_reported_sleep_hours",
    "weighted_average_sleep_duration",
]

biomarker_columns = [
    "hba1c",
    "hs_crp",
    "log_hs_crp",
]

demographic_columns = [
    "age",
    "income_poverty_ratio",
    "bmi",
]

analysis_columns = [
    # outcome
    "phq9_score",

    # Sleep
    "weekday_reported_sleep_hours",
    "weekend_reported_sleep_hours",
    "weekday_sleep_time_minutes",
    "weekday_wake_time_minutes",
    "weekend_sleep_time_minutes",
    "weekend_wake_time_minutes",

    # Metabolic 
    "hba1c",
    "hs_crp",
    "log_hs_crp",
    "bmi",

    # Demographic
    "age",
    "income_poverty_ratio",

    # Lifestyle
    "current_smoking_frequency",
    "moderate_recreation",
    "any_recreational_activity",
    "ever_smoked_100_cigarettes"
]

sleep_depression_columns = [
    "phq9_score",
    "weighted_average_sleep_duration",
    "weekday_reported_sleep_hours",
    "weekend_reported_sleep_hours",
    "social_jetlag_hours"
]

lifestyle_columns = [
    "age",
    "income_poverty_ratio",
    "current_smoking_frequency",
    "moderate_recreation",
    "any_recreational_activity"
]

metabolic_columns = [
    "phq9_score",
    "hba1c",
    "log_hs_crp",
    "bmi"
]
