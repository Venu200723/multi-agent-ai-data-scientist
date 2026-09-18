"""Streamlit UI for the multi-agent DS pipeline — demo-ready MVP."""

from __future__ import annotations

import traceback
from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from .orchestrator import Orchestrator


PREFERRED_TARGETS = (
    "churn",
    "target",
    "label",
    "y",
    "default",
    "converted",
    "outcome",
    "salary",
    "price",
    "revenue",
)


def _default_target_index(columns: list[str]) -> int:
    lower_map = {c.lower(): i for i, c in enumerate(columns)}
    for name in PREFERRED_TARGETS:
        if name in lower_map:
            return lower_map[name]
    return max(0, len(columns) - 1)


def _metric_cols(metrics: dict) -> None:
    keys = list(metrics.keys())
    cols = st.columns(max(len(keys), 1))
    for col, key in zip(cols, keys):
        val = metrics[key]
        col.metric(key.replace("_", " ").title(), f"{val:.4f}" if isinstance(val, float) else val)


def main() -> None:
    st.set_page_config(
        page_title="Multi-Agent AI Data Scientist",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🤖 Multi-Agent Autonomous AI Data Scientist")
    st.caption(
        "Upload a CSV → Cleaning → EDA → ML → Explainability → Markdown/HTML report. "
        "No paid API required for the core demo."
    )

    with st.sidebar:
        st.header("Controls")
        st.markdown(
            """
**Pipeline agents**
1. CleaningAgent  
2. EDAAgent  
3. MLAgent (RandomForest)  
4. ExplainAgent (permutation + offline narrative)  
5. ReportAgent  
            """
        )
        st.divider()
        st.markdown("Optional: set `OPENAI_API_KEY` for LLM narratives (offline fallback always works).")
        show_history = st.checkbox("Show agent action log", value=True)

    default = Path(__file__).resolve().parents[1] / "data" / "sample.csv"
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    use_sample = st.checkbox("Use bundled sample.csv (telecom churn)", value=uploaded is None)

    df: pd.DataFrame | None = None
    src_name = ""
    try:
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            src_name = uploaded.name
        elif use_sample and default.exists():
            df = pd.read_csv(default)
            src_name = "data/sample.csv"
        else:
            st.info("Upload a CSV or enable the bundled sample dataset to begin.")
            return
    except Exception as exc:
        st.error(f"Failed to read CSV: {exc}")
        return

    if df is None or df.empty:
        st.warning("Loaded dataframe is empty.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

    with st.expander("Preview data (first 20 rows)", expanded=True):
        st.dataframe(df.head(20), use_container_width=True)

    target = st.selectbox(
        "Target column",
        options=list(df.columns),
        index=_default_target_index(list(df.columns)),
        help="Column the model should predict (classification or regression auto-detected).",
    )

    run = st.button("▶ Run pipeline", type="primary", use_container_width=False)

    if not run:
        st.markdown(
            """
### Interview tip
Click **Run pipeline** with the sample dataset and target `churn`.  
Expect holdout metrics, feature-importance chart, narrative, and downloadable report in ~10–30s.
            """
        )
        return

    status = st.status("Agents at work…", expanded=True)
    result = None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "input.csv"
            df.to_csv(csv_path, index=False)
            artifacts = Path(tmp) / "artifacts"
            report = Path(tmp) / "report.md"
            orch = Orchestrator(artifacts_dir=artifacts, report_path=report)

            status.write("① CleaningAgent…")
            status.write("② EDAAgent…")
            status.write("③ MLAgent…")
            status.write("④ ExplainAgent…")
            status.write("⑤ ReportAgent…")

            result = orch.run(csv_path, target=target, report_path=report)

            # Persist report bytes before temp dir vanishes
            md_bytes = Path(result["report"]["markdown_path"]).read_bytes()
            html_bytes = Path(result["report"]["html_path"]).read_bytes()
            figure_blobs: list[tuple[str, bytes]] = []
            for fig in result["eda"].get("figures", []) + result["explain"].get("figures", []):
                p = Path(fig)
                if p.exists():
                    figure_blobs.append((p.name, p.read_bytes()))

            # Keep serializable / displayable pieces in session
            display = {
                "target": result["target"],
                "task": result["ml"]["task"],
                "metrics": result["ml"]["metrics"],
                "n_train": result["ml"]["n_train"],
                "n_test": result["ml"]["n_test"],
                "narrative": result["explain"]["narrative"],
                "importances": result["explain"]["importances"],
                "eda_profile": result["eda"].get("profile", {}),
                "top_correlations": result["eda"].get("top_correlations", []),
                "cleaning_actions": result["cleaning"].get("actions", []),
                "history": result.get("history", []),
                "md_bytes": md_bytes,
                "html_bytes": html_bytes,
                "figures": figure_blobs,
            }
            st.session_state["last_result"] = display
        status.update(label="Pipeline complete ✅", state="complete")
    except Exception as exc:
        status.update(label="Pipeline failed", state="error")
        st.error(f"**Pipeline error:** {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        return

    display = st.session_state.get("last_result")
    if not display:
        return

    st.success(
        f"Done — **{display['task']}** on `{display['target']}` "
        f"(train={display['n_train']}, test={display['n_test']})"
    )

    st.subheader("📊 Holdout metrics")
    _metric_cols(display["metrics"])

    left, right = st.columns(2)
    with left:
        st.subheader("🔍 Explanation")
        st.write(display["narrative"])
        if display["importances"]:
            st.subheader("Feature importances")
            st.dataframe(pd.DataFrame(display["importances"]), use_container_width=True, hide_index=True)
    with right:
        st.subheader("📈 Charts")
        if display["figures"]:
            for name, blob in display["figures"]:
                st.caption(name)
                st.image(blob, use_container_width=True)
        else:
            st.info("No figures generated.")

    st.subheader("📄 Report")
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download Markdown report",
        data=display["md_bytes"],
        file_name="pipeline_report.md",
        mime="text/markdown",
    )
    d2.download_button(
        "Download HTML report",
        data=display["html_bytes"],
        file_name="pipeline_report.html",
        mime="text/html",
    )
    with st.expander("Report preview (markdown)"):
        st.markdown(display["md_bytes"].decode("utf-8"))

    if show_history:
        with st.expander("Agent action log"):
            for line in display.get("history", []):
                st.write(f"- {line}")


if __name__ == "__main__":
    main()
