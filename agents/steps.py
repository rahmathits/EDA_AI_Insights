from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import llm_helper
from agents.base import BaseAgent

MAX_SAMPLE_ROWS = 500  # cap payload size sent to the browser


def _numeric_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _categorical_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(exclude=[np.number]).columns.tolist()


# ---------------------------------------------------------------- Step 1 ---
class PreprocessingAgent(BaseAgent):
    """Step 1: Data Loading & Preprocessing."""
    name = "preprocessing"

    def run(self, df: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        actions = []
        before_shape = df.shape

        df.columns = [str(c).strip() for c in df.columns]
        actions.append("trimmed whitespace from column names")

        obj_cols = df.select_dtypes(include=["object"]).columns
        for c in obj_cols:
            df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})
        if len(obj_cols):
            actions.append(f"trimmed whitespace / normalized blanks in {len(obj_cols)} text column(s)")

        converted = []
        for c in obj_cols:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().sum() >= 0.9 * df[c].notna().sum() and df[c].notna().sum() > 0:
                df[c] = coerced
                converted.append(c)
        if converted:
            actions.append(f"auto-converted numeric-looking text columns: {converted}")

        dup_count = int(df.duplicated().sum())
        if dup_count:
            df.drop_duplicates(inplace=True)
            actions.append(f"dropped {dup_count} duplicate row(s)")

        return {
            "actions": actions,
            "shape_before": before_shape,
            "shape_after": df.shape,
            "duplicate_rows_removed": dup_count,
            "_cleaned_df": df,  # stripped out by orchestrator before JSON serialization
        }


# ---------------------------------------------------------------- Step 2 ---
class StructureAgent(BaseAgent):
    """Step 2: Understanding Data Structure."""
    name = "structure"

    def run(self, df: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        dtype_counts = df.dtypes.astype(str).value_counts().to_dict()
        return {
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "columns": [
                {"name": c, "dtype": str(df[c].dtype), "unique_values": int(df[c].nunique())}
                for c in df.columns
            ],
            "dtype_counts": dtype_counts,
            "memory_bytes": int(df.memory_usage(deep=True).sum()),
            "sample_rows": df.head(5).replace({np.nan: None}).to_dict(orient="records"),
        }


# ---------------------------------------------------------------- Step 3 ---
class MissingValuesAgent(BaseAgent):
    """Step 3: Detecting Missing Values."""
    name = "missing_values"

    def run(self, df: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        n = len(df)
        per_col = []
        for c in df.columns:
            missing = int(df[c].isna().sum())
            per_col.append({
                "column": c,
                "missing_count": missing,
                "missing_pct": round(100 * missing / n, 2) if n else 0.0,
            })
        per_col.sort(key=lambda r: r["missing_pct"], reverse=True)

        rows_with_missing = int(df.isna().any(axis=1).sum())
        return {
            "per_column": per_col,
            "total_missing_cells": int(df.isna().sum().sum()),
            "rows_with_any_missing": rows_with_missing,
            "rows_with_missing_pct": round(100 * rows_with_missing / n, 2) if n else 0.0,
        }


# ---------------------------------------------------------------- Step 4 ---
class OutlierAgent(BaseAgent):
    """Step 4: Identifying Outliers (IQR + Z-score)."""
    name = "outliers"

    def run(self, df: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        results = []
        for c in _numeric_cols(df):
            series = df[c].dropna()
            if series.empty:
                continue
            q1, q3 = np.percentile(series, [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            iqr_outliers = series[(series < lower) | (series > upper)]

            z = np.abs(stats.zscore(series)) if series.std(ddof=0) > 0 else np.zeros(len(series))
            z_outliers = series[z > 3]

            results.append({
                "column": c,
                "min": float(series.min()), "q1": float(q1), "median": float(series.median()),
                "q3": float(q3), "max": float(series.max()),
                "iqr_bounds": [float(lower), float(upper)],
                "iqr_outlier_count": int(len(iqr_outliers)),
                "zscore_outlier_count": int(len(z_outliers)),
                "outlier_sample": [float(v) for v in iqr_outliers.head(20).tolist()],
            })
        return {"columns": results}


# ---------------------------------------------------------------- Step 5 ---
class CorrelationAgent(BaseAgent):
    """Step 5: Finding Patterns & Correlations."""
    name = "correlation"

    def run(self, df: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        num_cols = _numeric_cols(df)
        if len(num_cols) < 2:
            self.warn(context.get("session_id", "-"), self.name, "fewer than 2 numeric columns; correlation skipped")
            return {"matrix": [], "columns": num_cols, "top_pairs": []}

        corr = df[num_cols].corr(method="pearson")
        matrix = [
            {"x": a, "y": b, "value": None if pd.isna(corr.loc[a, b]) else round(float(corr.loc[a, b]), 4)}
            for a in num_cols for b in num_cols
        ]
        pairs = []
        for i, a in enumerate(num_cols):
            for b in num_cols[i + 1:]:
                v = corr.loc[a, b]
                if pd.notna(v):
                    pairs.append({"x": a, "y": b, "value": round(float(v), 4)})
        pairs.sort(key=lambda p: abs(p["value"]), reverse=True)

        return {"matrix": matrix, "columns": num_cols, "top_pairs": pairs[:15]}


# ---------------------------------------------------------------- Step 6 ---
class DistributionAgent(BaseAgent):
    """Step 6: Visualizing Distributions (histogram/density payloads)."""
    name = "distributions"

    def run(self, df: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        numeric = {}
        for c in _numeric_cols(df):
            series = df[c].dropna()
            if series.empty:
                continue
            counts, edges = np.histogram(series, bins=min(30, max(5, int(np.sqrt(len(series))))))
            numeric[c] = {
                "bins": [{"x0": float(edges[i]), "x1": float(edges[i + 1]), "count": int(counts[i])}
                         for i in range(len(counts))],
                "mean": float(series.mean()), "std": float(series.std()),
                "skewness": float(stats.skew(series)), "kurtosis": float(stats.kurtosis(series)),
            }

        categorical = {}
        for c in _categorical_cols(df):
            vc = df[c].value_counts().head(10)
            categorical[c] = [{"category": str(k), "count": int(v)} for k, v in vc.items()]

        return {"numeric": numeric, "categorical": categorical}


# ---------------------------------------------------------------- Step 7 ---
class AssumptionsAgent(BaseAgent):
    """Step 7: Checking Statistical Assumptions (normality, variance homogeneity)."""
    name = "assumptions"

    def run(self, df: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        session_id = context.get("session_id", "-")
        normality = []
        for c in _numeric_cols(df):
            series = df[c].dropna()
            if len(series) < 3:
                self.warn(session_id, self.name, f"'{c}' has <3 non-null values; normality test skipped")
                continue
            sample = series.sample(5000, random_state=42) if len(series) > 5000 else series
            try:
                stat, p = stats.shapiro(sample)
                normality.append({
                    "column": c, "test": "shapiro-wilk",
                    "statistic": float(stat), "p_value": float(p),
                    "likely_normal": bool(p > 0.05),
                })
            except Exception as exc:  # noqa: BLE001
                self.warn(session_id, self.name, f"shapiro-wilk failed for '{c}': {exc}")

        num_cols = _numeric_cols(df)
        variance_homogeneity = None
        if len(num_cols) >= 2:
            samples = [df[c].dropna() for c in num_cols if df[c].dropna().shape[0] > 1]
            if len(samples) >= 2:
                try:
                    stat, p = stats.levene(*samples)
                    variance_homogeneity = {
                        "test": "levene", "statistic": float(stat), "p_value": float(p),
                        "equal_variance_likely": bool(p > 0.05),
                    }
                except Exception as exc:  # noqa: BLE001
                    self.warn(session_id, self.name, f"levene test failed: {exc}")

        return {"normality": normality, "variance_homogeneity": variance_homogeneity}


# ---------------------------------------------------------------- Step 8 ---
class DimensionalityAgent(BaseAgent):
    """Step 8: Dimensionality Reduction (PCA)."""
    name = "dimensionality"

    def run(self, df: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        num_cols = _numeric_cols(df)
        session_id = context.get("session_id", "-")
        if len(num_cols) < 2:
            self.warn(session_id, self.name, "need >=2 numeric columns for PCA; skipped")
            return {"available": False, "reason": "fewer than 2 numeric columns"}

        clean = df[num_cols].dropna()
        if len(clean) < 3:
            self.warn(session_id, self.name, "fewer than 3 complete numeric rows for PCA; skipped")
            return {"available": False, "reason": "not enough complete numeric rows"}

        scaled = StandardScaler().fit_transform(clean)
        n_components = min(3, len(num_cols), len(clean))
        pca = PCA(n_components=n_components, random_state=42)
        projected = pca.fit_transform(scaled)

        sample_idx = np.random.RandomState(42).choice(
            len(projected), size=min(MAX_SAMPLE_ROWS, len(projected)), replace=False
        )
        points = [
            {"pc1": float(projected[i, 0]),
             "pc2": float(projected[i, 1]) if n_components > 1 else 0.0,
             "pc3": float(projected[i, 2]) if n_components > 2 else 0.0}
            for i in sample_idx
        ]
        loadings = [
            {"column": col, **{f"pc{j+1}": float(pca.components_[j][k]) for j in range(n_components)}}
            for k, col in enumerate(num_cols)
        ]

        return {
            "available": True,
            "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
            "n_components": n_components,
            "points": points,
            "loadings": loadings,
        }


# ---------------------------------------------------------------- Step 9 ---
class MultivariateAgent(BaseAgent):
    """Step 9: Multivariate Analysis (top correlated pair + clustering)."""
    name = "multivariate"

    def run(self, df: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        session_id = context.get("session_id", "-")
        num_cols = _numeric_cols(df)
        result: Dict[str, Any] = {"scatter": None, "clusters": None, "parallel_coordinates": None}

        corr_data = context.get("correlation", {}).get("data", {})
        top_pairs = corr_data.get("top_pairs", []) if corr_data else []
        if top_pairs:
            x, y = top_pairs[0]["x"], top_pairs[0]["y"]
            sub = df[[x, y]].dropna()
            sample = sub.sample(min(MAX_SAMPLE_ROWS, len(sub)), random_state=42) if len(sub) else sub
            result["scatter"] = {
                "x_col": x, "y_col": y,
                "points": [{"x": float(r[x]), "y": float(r[y])} for _, r in sample.iterrows()],
            }

        if len(num_cols) >= 2:
            clean = df[num_cols].dropna()
            if len(clean) >= 10:
                try:
                    scaled = StandardScaler().fit_transform(clean)
                    k = min(4, max(2, len(clean) // 20))
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = km.fit_predict(scaled)
                    sample_idx = np.random.RandomState(42).choice(
                        len(clean), size=min(MAX_SAMPLE_ROWS, len(clean)), replace=False
                    )
                    result["clusters"] = {
                        "k": k,
                        "points": [
                            {**{c: float(clean.iloc[i][c]) for c in num_cols[:2]}, "cluster": int(labels[i])}
                            for i in sample_idx
                        ],
                        "columns_used": num_cols[:2],
                    }
                except Exception as exc:  # noqa: BLE001
                    self.warn(session_id, self.name, f"KMeans clustering failed: {exc}")

        if 3 <= len(num_cols) <= 8:
            clean = df[num_cols].dropna()
            sample = clean.sample(min(150, len(clean)), random_state=42) if len(clean) else clean
            result["parallel_coordinates"] = {
                "columns": num_cols,
                "rows": sample.to_dict(orient="records"),
            }

        return result


# --------------------------------------------------------------- Step 10 ---
class SummaryAgent(BaseAgent):
    """Step 10: Summary & Insights -- synthesizes findings from all prior agents."""
    name = "summary"

    def run(self, df: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        insights: List[str] = []

        structure = context.get("structure", {}).get("data") or {}
        if structure:
            insights.append(f"Dataset has {structure.get('n_rows')} rows and {structure.get('n_cols')} columns.")

        missing = context.get("missing_values", {}).get("data") or {}
        if missing:
            worst = [r for r in missing.get("per_column", []) if r["missing_pct"] > 0][:3]
            if worst:
                cols = ", ".join(f"{r['column']} ({r['missing_pct']}%)" for r in worst)
                insights.append(f"Columns with the most missing data: {cols}.")
            else:
                insights.append("No missing values detected.")

        outliers = context.get("outliers", {}).get("data") or {}
        if outliers:
            flagged = [c for c in outliers.get("columns", []) if c["iqr_outlier_count"] > 0]
            if flagged:
                flagged.sort(key=lambda c: c["iqr_outlier_count"], reverse=True)
                top = flagged[0]
                insights.append(
                    f"'{top['column']}' has the most IQR outliers ({top['iqr_outlier_count']} values)."
                )

        corr = context.get("correlation", {}).get("data") or {}
        if corr and corr.get("top_pairs"):
            p = corr["top_pairs"][0]
            insights.append(f"Strongest correlation: {p['x']} & {p['y']} (r = {p['value']}).")

        assumptions = context.get("assumptions", {}).get("data") or {}
        if assumptions:
            non_normal = [n["column"] for n in assumptions.get("normality", []) if not n["likely_normal"]]
            if non_normal:
                insights.append(f"Columns likely non-normal (Shapiro-Wilk p<=0.05): {', '.join(non_normal[:5])}.")

        dim = context.get("dimensionality", {}).get("data") or {}
        if dim and dim.get("available"):
            ev = dim["explained_variance_ratio"]
            insights.append(
                f"First {len(ev)} principal component(s) explain {round(100*sum(ev),1)}% of variance."
            )

        return {
            "insights": insights,
            "n_agents_run": len([k for k in context if k != "session_id"]),
        }

    async def annotate(self, data, session_id, context=None, api_key=None, business_context=None, **kwargs):
        """
        Overridden: instead of interpreting its own JSON in isolation, the
        Summary agent synthesizes the rule-based insights plus every other
        step's LLM commentary into a structured 5-section business-case
        report: results & recommendations, industry/business/policy
        implications, limitations, alternative explanations, and key
        learnings/methodology. Returns a dict with those 5 keys, or None.
        """
        if not self.use_llm or not llm_helper.is_available(api_key) or not data:
            return None
        commentary_map = {}
        for key, result in (context or {}).items():
            if key == "session_id":
                continue
            step_data = result.get("data") if isinstance(result, dict) else None
            note = step_data.get("ai_commentary") if isinstance(step_data, dict) else None
            if note:
                commentary_map[key] = note
        try:
            return await llm_helper.get_business_case_report(
                data.get("insights", []), commentary_map, business_context, session_id, api_key=api_key
            )
        except Exception as exc:  # noqa: BLE001
            self.warn(session_id, self.name, f"business case report failed: {exc}")
            return None


PIPELINE: List[type[BaseAgent]] = [
    PreprocessingAgent, StructureAgent, MissingValuesAgent, OutlierAgent,
    CorrelationAgent, DistributionAgent, AssumptionsAgent, DimensionalityAgent,
    MultivariateAgent, SummaryAgent,
]
