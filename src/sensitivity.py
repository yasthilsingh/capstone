"""
Sensitivity-analysis helpers for the SIADS 699 Capstone project.

The primary analysis relates *social jetlag* (exposure) to *depression*
(PHQ-9) and tests whether an inflammatory biomarker modifies that relationship.
This module provides the pieces used by ``04_sensitivity``:

- Derived analysis variants the cleaned dataset does not export (alternative
  social-jetlag definitions and binary PHQ-9 indicators).
- Survey-weighted (design-based) GLM fitting that mirrors the primary model in
  ``03_primary_model`` exactly, via :class:`SvyModelSpec` / :func:`fit_svy_terms`
  / :func:`run_svy_grid_terms`, so every specification is directly comparable.
- Helpers to read 03's saved coefficients and compare each specification to
  them (:func:`primary_references_from_table`, :func:`compare_to_primary_terms`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Derived-variable construction
# ---------------------------------------------------------------------
#
# These helpers build the sensitivity-specific analysis variants that the
# cleaned dataset from ``01_data_prep`` does not export: the alternative
# social-jetlag definitions (signed / duration-based), the prorated PHQ-9
# score, and the binary PHQ-9 indicators at multiple cut-points. They operate
# on the *raw* NHANES variable codes (e.g. ``DPQ010``, ``SLQ300``, ``SLD012``).

# PHQ-9 scored items (DPQ100 is a functional-difficulty item, not scored).
PHQ9_ITEMS: List[str] = [
    "DPQ010",
    "DPQ020",
    "DPQ030",
    "DPQ040",
    "DPQ050",
    "DPQ060",
    "DPQ070",
    "DPQ080",
    "DPQ090",
]

# Sleep-timing variables (character "HH:MM" in NHANES).
WEEKDAY_SLEEP_TIME = "SLQ300"
WEEKDAY_WAKE_TIME = "SLQ310"
WEEKEND_SLEEP_TIME = "SLQ320"
WEEKEND_WAKE_TIME = "SLQ330"

# Sleep-duration variables (numeric hours).
WEEKDAY_SLEEP_HOURS = "SLD012"
WEEKEND_SLEEP_HOURS = "SLD013"

# Refused / Don't know codes used across the depression screener.
_MISSING_CODES = {7, 9, 77, 99, 777, 999}


def _clean_phq_item(series: pd.Series) -> pd.Series:
    """Coerce a PHQ item to numeric and set refused/don't-know codes to NaN."""
    cleaned = pd.to_numeric(series, errors="coerce")
    return cleaned.mask(cleaned.isin(_MISSING_CODES))


def phq9_score(
    df: pd.DataFrame,
    items: List[str] | None = None,
    missing_rule: str = "complete",
    max_missing: int = 2,
) -> pd.Series:
    """
    Compute the continuous PHQ-9 depression score (range 0-27).

    Parameters
    ----------
    df:
        DataFrame containing the raw ``DPQ0x0`` items.
    items:
        PHQ-9 item columns. Defaults to :data:`PHQ9_ITEMS`.
    missing_rule:
        - ``"complete"``: require all items present, otherwise NaN.
        - ``"prorated"``: allow up to ``max_missing`` missing items and
          prorate the score as ``mean(available) * n_items``.
    max_missing:
        Maximum number of missing items tolerated when ``missing_rule`` is
        ``"prorated"``.

    Returns
    -------
    pd.Series
        The PHQ-9 total score aligned to ``df.index``.
    """
    items = items or PHQ9_ITEMS
    missing = set(items) - set(df.columns)
    if missing:
        raise KeyError(f"Missing PHQ-9 item columns: {sorted(missing)}")

    cleaned = df[items].apply(_clean_phq_item)
    n_items = len(items)
    n_present = cleaned.notna().sum(axis=1)

    if missing_rule == "complete":
        total = cleaned.sum(axis=1, min_count=n_items)
    elif missing_rule == "prorated":
        item_mean = cleaned.mean(axis=1)
        total = item_mean * n_items
        total = total.mask(n_present < (n_items - max_missing))
    else:
        raise ValueError("missing_rule must be 'complete' or 'prorated'")

    return total.rename("phq9_score")


def phq9_binary(
    df: pd.DataFrame,
    cutoff: int = 10,
    items: List[str] | None = None,
    missing_rule: str = "complete",
    max_missing: int = 2,
    score: pd.Series | None = None,
) -> pd.Series:
    """
    Binary depression indicator: 1 if PHQ-9 >= ``cutoff`` else 0 (NaN preserved).

    A pre-computed ``score`` may be passed to avoid recomputation.
    """
    if score is None:
        score = phq9_score(
            df,
            items=items,
            missing_rule=missing_rule,
            max_missing=max_missing,
        )
    binary = (score >= cutoff).astype("float")
    binary = binary.mask(score.isna())
    return binary.rename("phq9_positive")


def _to_time_string(value) -> str | float:
    """Normalize a raw NHANES time cell to an ``"HH:MM"`` string or NaN."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if value is None:
        return np.nan
    if isinstance(value, float) and np.isnan(value):
        return np.nan
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return np.nan
    return text


def parse_clock_hours(series: pd.Series) -> pd.Series:
    """
    Parse an NHANES ``"HH:MM"`` clock column into hours since midnight (float).

    Returns values in ``[0, 24)``. Unparseable entries become NaN.
    """
    normalized = series.map(_to_time_string)
    parsed = pd.to_datetime(normalized, format="%H:%M", errors="coerce")
    hours = parsed.dt.hour + parsed.dt.minute / 60.0
    return hours.rename(series.name)


def _sleep_midpoint(sleep_hours: pd.Series, wake_hours: pd.Series) -> pd.Series:
    """
    Compute mid-sleep clock time (hours) from sleep-onset and wake clock times,
    handling the common case where sleep onset is before midnight.
    """
    # Shift wake time forward by 24h when it is "earlier" than onset (crossed midnight).
    adjusted_wake = wake_hours.where(wake_hours >= sleep_hours, wake_hours + 24.0)
    midpoint = (sleep_hours + adjusted_wake) / 2.0
    return midpoint.mod(24.0)


def social_jetlag(
    df: pd.DataFrame,
    method: str = "midpoint",
) -> pd.Series:
    """
    Derive social jetlag from weekday vs. weekend sleep patterns.

    Methods
    -------
    - ``"midpoint"``     : absolute difference in mid-sleep clock time between
                           weekend (free) and weekday (work) days. Baseline.
    - ``"signed_midpoint"``: signed (weekend - weekday) mid-sleep difference.
    - ``"duration"``     : absolute difference in reported sleep hours
                           (``SLD013`` - ``SLD012``).
    - ``"signed_duration"``: signed sleep-hours difference (weekend - weekday).

    Returns
    -------
    pd.Series
        Social jetlag in hours aligned to ``df.index``.
    """
    if method in {"midpoint", "signed_midpoint"}:
        required = [
            WEEKDAY_SLEEP_TIME,
            WEEKDAY_WAKE_TIME,
            WEEKEND_SLEEP_TIME,
            WEEKEND_WAKE_TIME,
        ]
        missing = set(required) - set(df.columns)
        if missing:
            raise KeyError(f"Missing sleep-timing columns: {sorted(missing)}")

        weekday_mid = _sleep_midpoint(
            parse_clock_hours(df[WEEKDAY_SLEEP_TIME]),
            parse_clock_hours(df[WEEKDAY_WAKE_TIME]),
        )
        weekend_mid = _sleep_midpoint(
            parse_clock_hours(df[WEEKEND_SLEEP_TIME]),
            parse_clock_hours(df[WEEKEND_WAKE_TIME]),
        )
        diff = weekend_mid - weekday_mid
        # Wrap to the shortest signed distance on the 24h clock: (-12, 12].
        diff = ((diff + 12.0) % 24.0) - 12.0
        result = diff if method == "signed_midpoint" else diff.abs()

    elif method in {"duration", "signed_duration"}:
        required = [WEEKDAY_SLEEP_HOURS, WEEKEND_SLEEP_HOURS]
        missing = set(required) - set(df.columns)
        if missing:
            raise KeyError(f"Missing sleep-duration columns: {sorted(missing)}")

        weekday_hours = pd.to_numeric(df[WEEKDAY_SLEEP_HOURS], errors="coerce")
        weekend_hours = pd.to_numeric(df[WEEKEND_SLEEP_HOURS], errors="coerce")
        diff = weekend_hours - weekday_hours
        result = diff if method == "signed_duration" else diff.abs()

    else:
        raise ValueError(
            "method must be one of "
            "{'midpoint', 'signed_midpoint', 'duration', 'signed_duration'}"
        )

    return result.rename("social_jetlag")


# ---------------------------------------------------------------------
# Specification description
# ---------------------------------------------------------------------

@dataclass
class ModelSpec:
    """
    Lightweight descriptor naming the exposure and interacting biomarker.

    Used only to locate the exposure-main, biomarker-main, and interaction
    terms in 03's saved coefficient table (see
    :func:`primary_references_from_table`); no model is fitted from it.
    """

    label: str
    outcome: str
    exposure: str
    interaction: Optional[str] = None


# ---------------------------------------------------------------------
# Term roles and coefficient-name matching
# ---------------------------------------------------------------------

# The three coefficients reported per interaction spec: the exposure (social
# jetlag) main effect, the biomarker main effect, and their product term.
DEFAULT_ROLES = ("exposure_main", "biomarker_main", "interaction")


def _match_main_effect(var: str, param_names: Sequence[str]) -> Optional[str]:
    """Find the main-effect coefficient for ``var`` (a term without ``:``)."""
    for name in param_names:
        if ":" in name:
            continue
        if name == var or name.startswith(f"{var}["):
            return name
    for name in param_names:  # substring fallback (handles C()/Q() decoration)
        if ":" not in name and var in name:
            return name
    return None


def _match_interaction(a: str, b: str, param_names: Sequence[str]) -> Optional[str]:
    """Find the ``a`` x ``b`` product coefficient (a term containing ``:``)."""
    for name in param_names:
        if ":" in name and a in name and b in name:
            return name
    return None


def _role_term(role: str, spec: ModelSpec, param_names: Sequence[str]) -> Optional[str]:
    """Resolve a role label to the model coefficient name for ``spec``."""
    if role in {"exposure_main", "sjl_main"}:
        return _match_main_effect(spec.exposure, param_names)
    if role in {"biomarker_main", "interaction_var_main"}:
        if not spec.interaction:
            return None
        return _match_main_effect(spec.interaction, param_names)
    if role == "interaction":
        if not spec.interaction:
            return None
        return _match_interaction(spec.exposure, spec.interaction, param_names)
    raise ValueError(f"Unknown role: {role}")


# ---------------------------------------------------------------------
# Survey-weighted (design-based) fitting - mirrors 03_primary_model
# ---------------------------------------------------------------------
#
# These helpers reproduce the primary model's estimation strategy exactly so
# every sensitivity spec is directly comparable to 03. The model is a
# design-based survey GLM (``svy`` package) using NHANES MEC weights, strata,
# and PSUs, with the same categorical coding and reference levels as 03. Only
# the varied element (exposure, biomarker, outcome, covariates, or subset)
# changes between specs; everything else is held identical to the primary model.

# NHANES 2017-March 2020 MEC survey-design columns.
SVY_STRATUM = "SDMVSTRA"
SVY_PSU = "SDMVPSU"
SVY_WEIGHT = "WTMECPRP"

# Categorical reference levels, matching 03_primary_model exactly.
SVY_CAT_REFS: dict = {
    "sex": "Female",
    "race_ethnicity": "Non-Hispanic White",
    "smoking_status": "Never",
    "work_schedule": "Traditional daytime",
}

# Primary adjustment set (03): sleep duration, age, sex, race/ethnicity,
# income-to-poverty ratio, recreational physical activity, and smoking status.
SVY_PRIMARY_COVARIATES: tuple = (
    "weighted_average_sleep_duration",
    "age",
    "sex",
    "race_ethnicity",
    "income_poverty_ratio",
    "any_recreational_activity",
    "smoking_status",
)

_DEFAULT_LINKS = {
    "gaussian": "identity",
    "binomial": "logit",
    "poisson": "log",
    "gamma": "inverse",
}


@dataclass
class SvyModelSpec:
    """
    A survey-weighted (design-based) specification mirroring 03_primary_model.

    Parameters
    ----------
    label:
        Human-readable name for this specification.
    outcome:
        Outcome column name.
    exposure:
        Primary exposure column name (e.g. ``social_jetlag_hours``).
    biomarker:
        Optional interacting biomarker column. When set, the model includes the
        biomarker main effect and its ``exposure x biomarker`` cross term.
    covariates:
        Adjustment covariate column names. Defaults to the 03 primary set.
    categorical_refs:
        Mapping of categorical covariate -> reference level, matching 03.
    family:
        ``svy`` GLM family (e.g. ``"gaussian"`` or ``"binomial"``).
    link:
        Optional link; when ``None`` the family default is used.
    subset:
        Optional pandas ``query`` string to restrict the analysis sample (e.g.
        an eligibility flag). Rows are physically filtered before fitting, so the
        survey design sees only the analysis sample - reproducing 03's use of a
        pre-filtered complete-case file.
    group:
        Sensitivity-theme grouping label.
    """

    label: str
    outcome: str
    exposure: str
    biomarker: Optional[str] = None
    covariates: Sequence[str] = SVY_PRIMARY_COVARIATES
    categorical_refs: Mapping[str, str] = field(
        default_factory=lambda: dict(SVY_CAT_REFS)
    )
    family: str = "gaussian"
    link: Optional[str] = None
    subset: Optional[str] = None
    group: str = "primary"


def _svy_link_for(family: str, link: Optional[str]) -> str:
    if link is not None:
        return link
    return _DEFAULT_LINKS.get(family, "identity")


def build_svy_predictors(spec: SvyModelSpec) -> list:
    """
    Build the ``svy`` predictor list for a spec, matching 03's coding: the
    exposure, the biomarker main effect and cross term (if any), then covariates
    with categoricals wrapped in ``svy.Cat(..., ref=...)`` at the 03 reference.
    """
    import svy

    predictors: list = [spec.exposure]
    if spec.biomarker:
        predictors.append(spec.biomarker)
        predictors.append(svy.Cross(spec.exposure, spec.biomarker))
    refs = spec.categorical_refs or {}
    for cov in spec.covariates:
        if cov in refs:
            predictors.append(svy.Cat(cov, ref=refs[cov]))
        else:
            predictors.append(cov)
    return predictors


def _svy_model_columns(spec: SvyModelSpec) -> List[str]:
    cols = [spec.outcome, spec.exposure]
    if spec.biomarker:
        cols.append(spec.biomarker)
    cols.extend(spec.covariates)
    cols.extend([SVY_STRATUM, SVY_PSU, SVY_WEIGHT])
    return list(dict.fromkeys(cols))


def prepare_svy_frame(df: pd.DataFrame, spec: SvyModelSpec) -> pd.DataFrame:
    """
    Subset rows (via ``spec.subset``) and columns for a survey spec, then drop
    rows missing any modeled or design variable (complete-case within the spec).
    """
    frame = df.query(spec.subset) if spec.subset else df
    cols = [c for c in _svy_model_columns(spec) if c in frame.columns]
    frame = frame.loc[:, cols].copy()
    return frame.dropna(subset=cols)


def _svy_role_term(role: str, spec: SvyModelSpec, terms: Sequence[str]) -> Optional[str]:
    """Resolve a role label to the fitted ``svy`` coefficient term name."""
    if role in {"exposure_main", "sjl_main"}:
        return spec.exposure if spec.exposure in terms else None
    if role in {"biomarker_main", "interaction_var_main"}:
        if not spec.biomarker:
            return None
        return spec.biomarker if spec.biomarker in terms else None
    if role == "interaction":
        if not spec.biomarker:
            return None
        for term in terms:
            if ":" in term and spec.exposure in term and spec.biomarker in term:
                return term
        return None
    raise ValueError(f"Unknown role: {role}")


def fit_svy_terms(
    df: pd.DataFrame,
    spec: SvyModelSpec,
    roles: Sequence[str] = DEFAULT_ROLES,
) -> List[dict]:
    """
    Fit one survey-weighted spec **once** and return one tidy row per requested
    term (exposure main, biomarker main, interaction).

    Rows carry ``reported`` (role) and ``biomarker`` fields plus the estimate,
    CI, and p-value, so survey results slot straight into the existing comparison
    and plotting code. ``weighted`` is always ``True``. A failed fit yields one
    error row per role, keeping grid runs resilient.
    """
    import polars as pl
    import svy

    meta = {
        "label": spec.label,
        "group": spec.group,
        "outcome": spec.outcome,
        "exposure": spec.exposure,
        "biomarker": spec.biomarker,
        "family": spec.family,
        "weighted": True,
    }

    def _blank(role: str) -> dict:
        return {
            **meta, "reported": role, "n": 0, "term": None,
            "estimate": np.nan, "std_err": np.nan, "ci_low": np.nan,
            "ci_high": np.nan, "p_value": np.nan, "converged": False,
            "odds_ratio": np.nan, "error": None,
        }

    try:
        frame = prepare_svy_frame(df, spec)
        n = int(len(frame))
        if n == 0:
            raise ValueError("No complete cases for this specification.")

        design = svy.Design(stratum=SVY_STRATUM, psu=SVY_PSU, wgt=SVY_WEIGHT)
        sample = svy.Sample(
            data=pl.from_pandas(frame, include_index=False),
            design=design,
        )
        fitted = sample.glm.fit(
            y=spec.outcome,
            x=build_svy_predictors(spec),
            family=spec.family,
            link=_svy_link_for(spec.family, spec.link),
            drop_nulls=True,
            alpha=0.05,
        )
        coefs = {c.term: c for c in fitted.fitted.coefs}
        terms = list(coefs)

        rows = []
        for role in roles:
            row = _blank(role)
            row["n"] = n
            term = _svy_role_term(role, spec, terms)
            if term is None:
                row["error"] = f"Term for role '{role}' not found in {terms}"
                rows.append(row)
                continue
            coef = coefs[term]
            row.update(
                term=term,
                estimate=float(coef.est),
                std_err=float(coef.se),
                ci_low=float(coef.lci),
                ci_high=float(coef.uci),
                p_value=float(coef.wald.p_value),
                converged=True,
            )
            if spec.family == "binomial":
                row["odds_ratio"] = float(np.exp(coef.est))
            rows.append(row)
        return rows

    except Exception as exc:  # noqa: BLE001 - grid resilience by design
        message = f"{type(exc).__name__}: {exc}"
        return [{**_blank(role), "error": message} for role in roles]


def run_svy_grid_terms(
    df: pd.DataFrame,
    specs: Sequence[SvyModelSpec],
    roles: Sequence[str] = DEFAULT_ROLES,
) -> pd.DataFrame:
    """
    Fit a list of survey-weighted specs and return one tidy row per (spec, term),
    ready to compare against 03's per-role primary references.
    """
    rows = []
    for spec in specs:
        rows.extend(fit_svy_terms(df, spec, roles))
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["group", "label", "reported"]).reset_index(drop=True)
    return result


def reverse_direction_svy(
    df: pd.DataFrame,
    depression_col: str,
    target_col: str,
    covariates: Sequence[str] = SVY_PRIMARY_COVARIATES,
    categorical_refs: Optional[Mapping[str, str]] = None,
    family: str = "gaussian",
    link: Optional[str] = None,
    subset: Optional[str] = None,
    label: Optional[str] = None,
) -> dict:
    """
    Survey-weighted reverse-direction model: depression is the exposure and a
    biomarker / sleep measure is the outcome, adjusting for the same covariates
    and coding as 03 (dropping the target from the covariate set if present).
    """
    refs = dict(categorical_refs) if categorical_refs is not None else dict(SVY_CAT_REFS)
    covs = [c for c in covariates if c != target_col and c != depression_col]
    spec = SvyModelSpec(
        label=label or f"reverse: {depression_col} -> {target_col}",
        outcome=target_col,
        exposure=depression_col,
        biomarker=None,
        covariates=covs,
        categorical_refs=refs,
        family=family,
        link=link,
        subset=subset,
        group="reverse_direction",
    )
    row = fit_svy_terms(df, spec, roles=["exposure_main"])[0]
    row["reported"] = "reverse_depression_effect"
    row["biomarker"] = None
    return row


# ---------------------------------------------------------------------
# Comparison to the primary estimate
# ---------------------------------------------------------------------

def primary_references_from_table(
    table: pd.DataFrame,
    spec: ModelSpec,
    roles: Sequence[str] = DEFAULT_ROLES,
) -> dict:
    """
    Extract ``{role: {estimate, ci_low, ci_high}}`` from a tidy 03 result table
    (columns ``term, estimate, ci_low, ci_high``) by matching each role's term.
    """
    terms = table["term"].astype(str).tolist()
    refs = {}
    for role in roles:
        term = _role_term(role, spec, terms)
        if term is None:
            refs[role] = {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan}
            continue
        hit = table.loc[table["term"].astype(str) == term].iloc[0]
        refs[role] = {
            "estimate": float(hit["estimate"]),
            "ci_low": float(hit["ci_low"]),
            "ci_high": float(hit["ci_high"]),
        }
    return refs


def compare_to_primary_terms(
    terms_df: pd.DataFrame,
    primary_by_role: dict,
) -> pd.DataFrame:
    """
    Annotate a multi-term sensitivity result (from
    :func:`run_svy_grid_terms`) against per-role primary references.

    Each row's ``delta`` / ``pct_change`` / ``sign_flip`` / ``consistent`` are
    computed relative to the primary estimate of the **same** ``reported`` role,
    so main effects compare to main effects and interactions to interactions.
    ``significant`` is defined as ``p_value < 0.05``.
    """
    out = terms_df.copy()
    ref_est = out["reported"].map(lambda r: primary_by_role.get(r, {}).get("estimate", np.nan))
    ref_low = out["reported"].map(lambda r: primary_by_role.get(r, {}).get("ci_low", np.nan))
    ref_high = out["reported"].map(lambda r: primary_by_role.get(r, {}).get("ci_high", np.nan))

    out["primary_estimate"] = ref_est.astype(float)
    out["delta"] = out["estimate"] - out["primary_estimate"]

    with np.errstate(divide="ignore", invalid="ignore"):
        out["pct_change"] = np.where(
            out["primary_estimate"].abs() > 0,
            (out["estimate"] - out["primary_estimate"]) / out["primary_estimate"].abs() * 100.0,
            np.nan,
        )

    out["sign_flip"] = np.sign(out["estimate"]) != np.sign(out["primary_estimate"])
    out["significant"] = out["p_value"].astype(float) < 0.05
    within = (out["estimate"] >= ref_low.astype(float)) & (out["estimate"] <= ref_high.astype(float))
    out["consistent"] = out["significant"] & ~out["sign_flip"] & within

    return out



# ---------------------------------------------------------------------
# Simple imputation for complete-case vs. imputed sensitivity
# ---------------------------------------------------------------------

def simple_impute(
    df: pd.DataFrame,
    columns: Sequence[str],
    strategy: str = "median",
) -> pd.DataFrame:
    """
    Return a copy of ``df`` with ``columns`` imputed by a simple strategy and a
    companion ``<col>_was_missing`` indicator for each imputed column.

    ``strategy`` may be ``"median"`` or ``"mean"``. This is intentionally simple;
    for multiple imputation (MICE) use ``sklearn.impute.IterativeImputer``.
    """
    if strategy not in {"median", "mean"}:
        raise ValueError("strategy must be 'median' or 'mean'")

    out = df.copy()
    for col in columns:
        numeric = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_was_missing"] = numeric.isna().astype("int")
        fill = numeric.median() if strategy == "median" else numeric.mean()
        out[col] = numeric.fillna(fill)
    return out
