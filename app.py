from __future__ import annotations
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    ListFlowable,
    ListItem,
)

import html
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ai_client import check_worker_health, generate_game_concept, get_worker_url
from src.app_data import (
    get_cluster_games,
    get_cluster_pain_points,
    latest_snapshot_name,
    load_app_data,
    split_pipe,
)
from src.live_pipeline import PIPELINE_STAGES, pipeline_snapshot_summary, run_stage, summarize_stage
from src.market_analysis import calculate_market_metrics, get_developer_presence, get_top_games


# =============================================================================
# APP CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="GameScout AI",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# VISUAL SYSTEM
# =============================================================================

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #080a13 0%, #101426 55%, #090b14 100%);
        }
        [data-testid="stSidebar"] {
            background: #0b0e19;
            border-right: 1px solid rgba(148, 163, 184, 0.12);
        }
        .block-container {
            max-width: 1250px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }
        h1, h2, h3 { letter-spacing: -0.02em; }
        .gs-badge {
            display: inline-block;
            border: 1px solid rgba(168,85,247,.35);
            background: rgba(124,58,237,.12);
            color: #c4b5fd;
            border-radius: 999px;
            padding: .32rem .7rem;
            font-size: .78rem;
            margin-bottom: .7rem;
        }
        .gs-card {
            border: 1px solid rgba(148,163,184,.14);
            background: rgba(18,22,38,.74);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin: .35rem 0 .75rem 0;
        }
        .gs-hero {
            border: 1px solid rgba(139,92,246,.28);
            background: radial-gradient(circle at top right, rgba(124,58,237,.19), rgba(18,22,38,.78) 55%);
            border-radius: 24px;
            padding: 2.2rem 2.3rem;
            margin: 1rem 0 1.4rem 0;
        }
        .gs-hero-title {
            font-size: clamp(2.4rem, 6vw, 5rem);
            line-height: .98;
            font-weight: 900;
            letter-spacing: -.055em;
            color: #f8fafc;
            margin: .4rem 0 1rem 0;
        }
        .gs-hero-copy {
            max-width: 780px;
            color: #aeb5c7;
            font-size: 1.12rem;
            line-height: 1.65;
        }
        .gs-muted { color: #98a2b7; }
        .gs-title { font-size: 1.05rem; font-weight: 700; color: #f8fafc; }
        .gs-score { font-size: 2rem; font-weight: 800; color: #c4b5fd; }
        .gs-warning {
            border-left: 3px solid #f59e0b;
            padding: .5rem .8rem;
            background: rgba(245,158,11,.08);
            border-radius: 6px;
        }
        .gs-success {
            border-left: 3px solid #34d399;
            padding: .5rem .8rem;
            background: rgba(52,211,153,.08);
            border-radius: 6px;
        }
        .pipeline-mini {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: .55rem;
            margin-top: 1.2rem;
        }
        .pipeline-mini > div {
            border: 1px solid rgba(148,163,184,.12);
            border-radius: 12px;
            padding: .65rem .8rem;
            background: rgba(10,13,24,.55);
            color: #cbd5e1;
            font-size: .88rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(148,163,184,.14);
            background: rgba(18,22,38,.65);
            border-radius: 14px;
            padding: .8rem;
            min-height: 118px;
        }
        div[data-testid="stMetricValue"] {
            white-space: normal;
            overflow: visible;
            text-overflow: clip;
            line-height: 1.05;
        }
        div.stButton > button {
            border-radius: 11px;
            font-weight: 700;
            min-height: 3rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SESSION STATE
# =============================================================================

if "analysis_complete" not in st.session_state:
    st.session_state["analysis_complete"] = False

if "analysis_mode" not in st.session_state:
    st.session_state["analysis_mode"] = None

if "analysis_duration" not in st.session_state:
    st.session_state["analysis_duration"] = None


# =============================================================================
# LIVE ANALYSIS EXPERIENCE
# =============================================================================


def render_landing_page() -> None:
    """Landing page shown before the market analysis has been launched."""
    st.markdown('<span class="gs-badge">AI-powered mobile game intelligence</span>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="gs-hero">
            <div class="gs-hero-title">Discover.<br>Analyze.<br>Create.</div>
            <div class="gs-hero-copy">
                GameScout AI starts from the current US App Store charts, builds market segments with NLP and
                unsupervised machine learning, ranks product opportunities, mines public player reviews, and
                turns the resulting evidence into original game concepts with generative AI.
            </div>
            <div class="pipeline-mini">
                <div>01 · App Store scraping</div>
                <div>02 · Metadata enrichment</div>
                <div>03 · TF-IDF + SVD</div>
                <div>04 · KMeans segments</div>
                <div>05 · Review pain points</div>
                <div>06 · Workers AI concepts</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Run the analysis")
    st.write(
        "The default mode executes the real Python pipeline in front of you. "
        "The cached snapshot remains available only as a fallback if an external App Store endpoint is unavailable."
    )

    include_reviews = True
    

    primary, fallback = st.columns([1.25, 0.75])
    with primary:

        review_limit = st.radio(
            "Maximum reviews per game",
            ["⚡ 20 (Fast) [Recommended]", "⚖️ 50 (Balanced)", "🔬 Unlimited (Slow)"],
            horizontal=True,
            help="Higher limits improve pain-point analysis but increase execution time."
        )

        if review_limit == "⚡ 20 (Fast) [Recommended]":
            max_reviews = 20
            eta = "~2 min"
        elif review_limit == "⚖️ 50 (Balanced)":
            max_reviews = 50
            eta = "~5 min"
        else:
            max_reviews = None
            eta = ">10 min"

        st.caption(f"⏱ Estimated duration: {eta}")    
        
        # Two launch options aligned on the same row
        live_col, cached_col = st.columns([1.7, 1])

        with live_col:
            start_live = st.button(
                "🚀 Start Live Analysis",
                type="primary",
                width="stretch",
            )

        with cached_col:
            use_cached = st.button(
                "Use Cached Snapshot",
                width="stretch",
            )

        if start_live:
            run_live_analysis(max_reviews)

        if use_cached:
            st.session_state["analysis_complete"] = True
            st.session_state["analysis_mode"] = "Cached snapshot"
            st.session_state["analysis_duration"] = 0.0
            st.rerun()

        st.caption(
            "Live mode requires internet access. Cached mode uses the exact CSV outputs committed with the project for a reproducible evaluator experience."
        )


def run_live_analysis(max_reviews=None, include_reviews: bool = True) -> None:
    """Execute the real project scripts and expose their progress inside Streamlit."""
    selected_stages = PIPELINE_STAGES if include_reviews else PIPELINE_STAGES[:4]
    started_at = time.perf_counter()
    progress = st.progress(0, text="Preparing live analysis…")

    st.markdown("### Live pipeline")
    st.caption("Each step below is running the actual project script — no precomputed animation.")

    for index, stage in enumerate(selected_stages, start=1):
        progress.progress(
            (index - 1) / len(selected_stages),
            text=f"Step {index}/{len(selected_stages)} · {stage.title}",
        )

        with st.status(
            f"{index}. {stage.title}",
            expanded=True,
        ) as status:
            st.write(stage.description)
            log_placeholder = st.empty()
            recent_lines: list[str] = []

            try:
                
                for line in run_stage(stage,max_reviews=max_reviews,):
                    if line.strip():
                        recent_lines.append(line)
                        recent_lines = recent_lines[-10:]
                        log_placeholder.code("\n".join(recent_lines), language="text")
            except Exception as error:
                status.update(label=f"{stage.title} failed", state="error", expanded=True)
                st.error(f"Live stage failed: {error}")
                st.info(
                    "Your previously cached snapshot has not been deleted. You can use it to continue the product demo, "
                    "or fix the external-data issue and run the live analysis again."
                )
                if st.button("Continue with cached snapshot", key=f"fallback_{stage.key}"):
                    st.session_state["analysis_complete"] = True
                    st.session_state["analysis_mode"] = "Cached fallback after live error"
                    st.session_state["analysis_duration"] = time.perf_counter() - started_at
                    st.rerun()
                st.stop()

            facts = summarize_stage(stage.key)
            fact_columns = st.columns(max(1, min(len(facts), 4)))
            for column, (label, value) in zip(fact_columns, facts.items()):
                column.metric(label, value)

            status.update(label=f"✓ {stage.title}", state="complete", expanded=False)

    elapsed = time.perf_counter() - started_at
    progress.progress(1.0, text="Analysis complete")
    st.success(f"End-to-end market analysis completed in {elapsed:.1f} seconds.")

    # Ensure Streamlit reloads the newly generated CSV files on the next rerun.
    get_data.clear()
    st.session_state["analysis_complete"] = True
    st.session_state["analysis_mode"] = "Live analysis"
    st.session_state["analysis_duration"] = elapsed
    st.session_state.pop("last_concept", None)
    st.session_state.pop("last_concept_segment", None)

    time.sleep(0.5)
    st.rerun()


# =============================================================================
# DATA
# =============================================================================

@st.cache_data(show_spinner=False)
def get_data() -> dict[str, pd.DataFrame]:
    return load_app_data()


# Stop here until the evaluator launches an analysis.
if not st.session_state["analysis_complete"]:
    render_landing_page()
    st.stop()


try:
    data = get_data()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.info("The app expects the generated CSV files in data/processed/. Run the live analysis again.")
    if st.button("Back to analysis launcher"):
        st.session_state["analysis_complete"] = False
        st.rerun()
    st.stop()


# =============================================================================
# HELPERS
# =============================================================================

def build_concept_pdf(
    concept: dict,
    segment_name: str,
    opportunity_score: float,
) -> bytes:
    """Build a downloadable PDF for one generated game concept."""

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    title_style.alignment = TA_CENTER

    story = []

    story.append(
        Paragraph(
            str(concept.get("title", "Untitled Game Concept")),
            title_style,
        )
    )

    story.append(
        Paragraph(
            str(concept.get("one_line_pitch", "")),
            styles["Italic"],
        )
    )

    story.append(Spacer(1, 18))

    story.append(
        Paragraph(
            f"<b>Source market segment:</b> {segment_name}",
            styles["BodyText"],
        )
    )

    story.append(
        Paragraph(
            f"<b>Opportunity Score:</b> {opportunity_score:.1f}/100",
            styles["BodyText"],
        )
    )

    story.append(Spacer(1, 14))

    sections = [
        ("Genre", concept.get("genre")),
        ("Target audience", concept.get("target_audience")),
        ("Originality score", f"{concept.get('originality_score', '—')}/100"),
        ("Core mechanic", concept.get("core_mechanic")),
        ("Unique twist", concept.get("unique_twist")),
        ("Market rationale", concept.get("market_rationale")),
        ("Main risk", concept.get("main_risk")),
    ]

    for heading, content in sections:
        if content:
            story.append(
                Paragraph(
                    heading,
                    styles["Heading2"],
                )
            )

            story.append(
                Paragraph(
                    str(content),
                    styles["BodyText"],
                )
            )

            story.append(Spacer(1, 10))

    list_sections = [
        ("Core loop", concept.get("core_loop", [])),
        (
            "Pain points addressed",
            concept.get("pain_points_addressed", []),
        ),
        (
            "Retention ideas",
            concept.get("retention_ideas", []),
        ),
        (
            "Monetization",
            concept.get("monetization", []),
        ),
    ]

    for heading, items in list_sections:
        if isinstance(items, list) and items:
            story.append(
                Paragraph(
                    heading,
                    styles["Heading2"],
                )
            )

            story.append(
                ListFlowable(
                    [
                        ListItem(
                            Paragraph(
                                str(item),
                                styles["BodyText"],
                            )
                        )
                        for item in items
                    ],
                    bulletType="bullet",
                )
            )

            story.append(Spacer(1, 10))

    document.build(story)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes

def render_page_navigation(
    current_page: str,
) -> None:
    pages = [
        "Analysis Summary",
        "Market Overview",
        "Game Segments",
        "Opportunities",
        "Concept Generator",
    ]

    current_index = pages.index(current_page)

    previous_page = (
        pages[current_index - 1]
        if current_index > 0
        else None
    )

    next_page = (
        pages[current_index + 1]
        if current_index < len(pages) - 1
        else None
    )

    st.markdown("---")

    previous_col, spacer_col, next_col = st.columns(
        [1, 2, 1]
    )

    with previous_col:
        if previous_page is not None:
            if st.button(
                "← Previous",
                width="stretch",
                key=f"previous_{current_page}",
            ):
                st.session_state["page"] = previous_page
                st.rerun()

    with next_col:
        if next_page is not None:
            if st.button(
                "Next →",
                type="primary",
                width="stretch",
                key=f"next_{current_page}",
            ):
                st.session_state["page"] = next_page
                st.rerun()

def format_number(value: object) -> str:
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return "—"


def format_float(value: object, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def opportunity_by_cluster(cluster_id: int) -> pd.Series | None:
    rows = data["opportunities"][data["opportunities"]["cluster_id"] == cluster_id]
    if rows.empty:
        return None
    return rows.iloc[0]


def pain_point_label(keywords: object) -> str:
    terms = split_pipe(keywords)
    if not terms:
        return "Player complaint"
    return " / ".join(term.title() for term in terms[:3])


def build_ai_payload(
    opportunity: pd.Series,
    style: str,
    previous_titles: list[str] | None = None,
    ) -> dict:
    cluster_id = int(opportunity["cluster_id"])
    pain_points = get_cluster_pain_points(data, cluster_id)

    ai_pain_points = []
    for item in pain_points:
        examples = item.get("example_reviews", [])
        ai_pain_points.append(
            {
                "share": item["share"],
                "keywords": item["keywords"],
                "example_reviews": " | ".join(str(x) for x in examples),
            }
        )

    return {
        "cluster_name": str(opportunity["cluster_name"]),
        "opportunity_score": float(opportunity["adjusted_opportunity_score"]),
        "average_rating": float(opportunity["average_rating"]),
        "game_count": int(opportunity["game_count"]),
        "developer_count": int(opportunity["developer_count"]),
        "average_chart_position": float(opportunity["average_chart_position"]),
        "representative_games": split_pipe(opportunity["representative_games"]),
        "keywords": split_pipe(opportunity["keywords"]),
        "signal_summary": str(opportunity.get("signal_summary", "")),
        "pain_points": ai_pain_points,
        "concept_style": style,
        "previous_titles": previous_titles or [],
    }


def render_concept(concept: dict) -> None:
    st.markdown("---")
    st.markdown('<span class="gs-badge">AI-generated concept</span>', unsafe_allow_html=True)
    st.title(str(concept.get("title", "Untitled concept")))
    st.caption(str(concept.get("one_line_pitch", "")))

    col1, col2, col3 = st.columns(3)
    col1.metric("Genre", str(concept.get("genre", "—")))
    col2.metric("Originality", f"{concept.get('originality_score', '—')}/100")
    col3.metric("Target audience", str(concept.get("target_audience", "—")))

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Core mechanic")
        st.write(concept.get("core_mechanic", "—"))

        st.subheader("Core loop")
        loop = concept.get("core_loop", [])
        if isinstance(loop, list) and loop:
            for index, step in enumerate(loop, start=1):
                st.write(f"**{index}.** {step}")
        else:
            st.write("—")

        st.subheader("Unique twist")
        st.write(concept.get("unique_twist", "—"))

    with right:
        st.subheader("Market rationale")
        st.write(concept.get("market_rationale", "—"))

        st.subheader("Main risk")
        st.write(concept.get("main_risk", "—"))

        addressed = concept.get("pain_points_addressed", [])
        if isinstance(addressed, list) and addressed:
            st.subheader("Pain points addressed")
            for item in addressed:
                st.write(f"✓ {item}")

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.subheader("Retention ideas")
        items = concept.get("retention_ideas", [])
        if isinstance(items, list):
            for item in items:
                st.write(f"• {item}")

    with bottom_right:
        st.subheader("Monetization")
        items = concept.get("monetization", [])
        if isinstance(items, list):
            for item in items:
                st.write(f"• {item}")


# =============================================================================
# SIDEBAR / NAVIGATION
# =============================================================================

with st.sidebar:
    st.title("🎮 GameScout AI")
    st.caption("App Store market intelligence → game concepts")

    pages = [
    "Analysis Summary",
    "Market Overview",
    "Game Segments",
    "Opportunities",
    "Concept Generator",
    ]

    if "page" not in st.session_state:
        st.session_state["page"] = "Analysis Summary"

    page = st.radio(
        "Navigate",
        pages,
        index=pages.index(
            st.session_state["page"]
        ),
        label_visibility="collapsed",
    )

    st.session_state["page"] = page

    st.divider()
    st.caption(f"Mode: {st.session_state.get('analysis_mode', '—')}")
    st.caption(f"Snapshot: {latest_snapshot_name()}")
    st.caption("US App Store · Games")

    if st.button("↻ Start a new analysis", width="stretch"):
        st.session_state["analysis_complete"] = False
        st.session_state["analysis_mode"] = None
        st.session_state["analysis_duration"] = None
        st.session_state.pop("last_concept", None)
        st.session_state.pop("last_concept_segment", None)
        st.rerun()


# =============================================================================
# PAGE 0 — ANALYSIS SUMMARY
# =============================================================================

if page == "Analysis Summary":
    st.markdown('<span class="gs-badge">End-to-end pipeline</span>', unsafe_allow_html=True)
    st.title("Analysis Complete")
    st.write(
        "The market snapshot below is the output of the complete GameScout pipeline: live App Store collection, "
        "metadata enrichment, NLP segmentation, opportunity scoring and public-review analysis."
    )

    summary = pipeline_snapshot_summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Games analyzed", summary["Games"])
    c2.metric(
    "Market segments",
    data["opportunities"]["cluster_id"].nunique(),)
    c3.metric("Reviews collected", summary["Reviews"])
    c4.metric("Pain-point groups", summary["Pain points"])

    duration = st.session_state.get("analysis_duration")
    if duration and duration > 0:
        st.success(f"{st.session_state.get('analysis_mode')} completed in {duration:.1f} seconds.")
    else:
        st.info("Cached snapshot loaded. Run a new live analysis from the sidebar to replay the complete pipeline.")

    st.subheader("Pipeline")
    labels = [
        ("1", "App Store", "Top Free + Top Paid"),
        ("2", "Metadata", "Descriptions · ratings · genres"),
        ("3", "NLP", "TF-IDF → SVD"),
        ("4", "Segments", "KMeans"),
        ("5", "Signals", "Opportunity score + reviews"),
        ("6", "AI", "Concept generation"),
    ]
    cols = st.columns(3)
    for index, (number, title, copy) in enumerate(labels):
        with cols[index % 3]:
            st.markdown(
                f"""
                <div class="gs-card">
                    <div class="gs-muted">STEP {number}</div>
                    <div class="gs-title">{html.escape(title)}</div>
                    <div class="gs-muted">{html.escape(copy)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.info("Next: inspect the market signals, then use Concept Generator to turn one opportunity into a new game idea.")

    render_page_navigation("Analysis Summary")

# =============================================================================
# PAGE 1 — MARKET OVERVIEW
# =============================================================================

elif page == "Market Overview":
    st.markdown('<span class="gs-badge">Current market snapshot</span>', unsafe_allow_html=True)
    st.title("Market Overview")
    st.write("A current snapshot of the US App Store Games rankings, enriched with public metadata.")

    games = data["enriched_games"]
    metrics = calculate_market_metrics(games)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unique games", format_number(metrics["unique_games"]))
    c2.metric("Top free", format_number(metrics["free_games"]))
    c3.metric("Top paid", format_number(metrics["paid_games"]))
    c4.metric("Developers", format_number(metrics["developers"]))

    st.subheader("Current rankings")
    left, right = st.columns([1.05, 0.95])

    with left:
        developer_presence = get_developer_presence(games, number_of_developers=10)
        chart = px.bar(
            developer_presence.sort_values("ranked_games"),
            x="ranked_games",
            y="developer",
            orientation="h",
            labels={"ranked_games": "Ranked games", "developer": ""},
        )
        chart.update_layout(
            height=420,
            margin=dict(l=0, r=10, t=25, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#aeb5c7"),
            xaxis=dict(gridcolor="rgba(148,163,184,.12)"),
        )
        chart.update_traces(marker_color="#8b5cf6")
        st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})

    with right:
        chart_type = st.segmented_control(
            "Ranking",
            options=["top_free", "top_paid"],
            format_func=lambda x: "Top Free" if x == "top_free" else "Top Paid",
            default="top_free",
        )
        top_games = get_top_games(games, chart_type or "top_free", number_of_games=10)
        display = top_games[["chart_position", "title", "developer"]].copy()
        display.columns = ["#", "Game", "Developer"]
        st.dataframe(display, hide_index=True, width="stretch", height=385)

    render_page_navigation("Market Overview")


# =============================================================================
# PAGE 2 — GAME SEGMENTS
# =============================================================================

elif page == "Game Segments":
    st.markdown('<span class="gs-badge">Unsupervised ML</span>', unsafe_allow_html=True)
    st.title("Game Segments")
    st.write(
        "Descriptions are represented with TF-IDF, reduced with SVD, then grouped with KMeans. "
        "The segments are exploratory: franchise clusters and broad groups are intentionally kept visible."
    )

    opportunities = data["opportunities"].sort_values("cluster_id")
    cluster_options = {
        f"Cluster {int(row.cluster_id)} · {row.cluster_name}": int(row.cluster_id)
        for row in opportunities.itertuples()
    }

    selection = st.selectbox("Choose a segment", list(cluster_options.keys()))
    cluster_id = cluster_options[selection]
    opportunity = opportunity_by_cluster(cluster_id)
    cluster_games = get_cluster_games(data, cluster_id)

    if opportunity is None:
        st.warning("No opportunity metadata is available for this cluster.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Games", int(opportunity["game_count"]))
    c2.metric("Developers", int(opportunity["developer_count"]))
    c3.metric("Average rating", format_float(opportunity["average_rating"], 2))
    c4.metric("Semantic coherence", format_float(opportunity["semantic_coherence"], 3))

    st.subheader(str(opportunity["cluster_name"]))
    st.caption(str(opportunity["segment_type"]))

    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown("**Representative games**")
        for game in split_pipe(opportunity["representative_games"]):
            st.write(f"• {game}")

        st.markdown("**Dominant keywords**")
        st.write(" · ".join(split_pipe(opportunity["keywords"])[:10]))

    with right:
        st.markdown("**Cluster diagnostics**")
        st.write(f"Average chart position: **{format_float(opportunity['average_chart_position'], 1)}**")
        st.write(f"Top developer: **{opportunity['top_developer']}**")
        st.write(f"Top developer share: **{float(opportunity['top_developer_share']):.0%}**")
        warning = opportunity.get("warning")
        if isinstance(warning, str) and warning.strip():
            st.warning(warning)

    st.subheader("Games in this segment")
    columns = [
        col
        for col in ["title", "developer", "chart_type", "chart_position", "average_rating", "rating_count"]
        if col in cluster_games.columns
    ]
    display = cluster_games[columns].sort_values("chart_position").copy()
    st.dataframe(display, hide_index=True, width="stretch", height=340)

    render_page_navigation("Game Segments")

# =============================================================================
# PAGE 3 — OPPORTUNITIES
# =============================================================================

elif page == "Opportunities":
    st.markdown('<span class="gs-badge">Decision-support layer</span>', unsafe_allow_html=True)
    st.title("Opportunity Ranking")
    st.write(
        "The score combines current chart strength, player validation, scarcity, developer diversity, "
        "semantic coherence and confidence. It is a transparent heuristic — not a revenue forecast."
    )

    opportunities = data["opportunities"].sort_values("adjusted_opportunity_score", ascending=False)

    type_filter = st.multiselect(
        "Segment types",
        sorted(opportunities["segment_type"].dropna().unique().tolist()),
        default=sorted(opportunities["segment_type"].dropna().unique().tolist()),
    )
    filtered = opportunities[opportunities["segment_type"].isin(type_filter)].copy()

    if filtered.empty:
        st.info("No segments match the selected filters.")
    else:
        top = filtered.iloc[0]
        st.markdown(
            f"""
            <div class="gs-card">
                <div class="gs-muted">Top current signal</div>
                <div class="gs-title">{html.escape(str(top['cluster_name']))}</div>
                <div class="gs-score">{float(top['adjusted_opportunity_score']):.1f}/100</div>
                <div class="gs-muted">{html.escape(str(top['signal_summary']))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        display_columns = [
            "opportunity_rank",
            "cluster_name",
            "adjusted_opportunity_score",
            "segment_type",
            "average_rating",
            "game_count",
            "developer_count",
            "confidence_level",
        ]
        table = filtered[display_columns].copy()
        table.columns = ["Rank", "Segment", "Score", "Type", "Avg rating", "Games", "Developers", "Confidence"]
        st.dataframe(table, hide_index=True, width="stretch", height=390)

        st.subheader("Inspect score breakdown")
        segment = st.selectbox("Segment", filtered["cluster_name"].tolist())
        row = filtered[filtered["cluster_name"] == segment].iloc[0]

        score_data = pd.DataFrame(
            {
                "Component": [
                    "Market strength",
                    "Player validation",
                    "Competition gap",
                    "Developer diversity",
                    "Semantic coherence",
                    "Confidence",
                ],
                "Score": [
                    row["market_strength_score"],
                    row["player_validation_score"],
                    row["competition_gap_score"],
                    row["developer_diversity_score"],
                    row["semantic_coherence_score"],
                    row["confidence_score"],
                ],
                "Maximum": [25, 20, 15, 15, 15, 10],
            }
        )
        score_data["Percent"] = 100 * score_data["Score"] / score_data["Maximum"]
        chart = px.bar(score_data, x="Percent", y="Component", orientation="h", text="Score")
        chart.update_layout(
            height=340,
            xaxis_range=[0, 100],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#aeb5c7"),
            xaxis_title="Share of component maximum (%)",
            yaxis_title="",
            margin=dict(l=0, r=10, t=10, b=0),
        )
        chart.update_traces(marker_color="#8b5cf6")
        st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})

        if isinstance(row.get("warning"), str) and row["warning"].strip():
            st.warning(row["warning"])
        st.info(str(row["signal_summary"]))

    render_page_navigation("Opportunities")


# =============================================================================
# PAGE 4 — CONCEPT GENERATOR
# =============================================================================

elif page == "Concept Generator":
    st.markdown('<span class="gs-badge">Generative AI</span>', unsafe_allow_html=True)
    st.title("Game Concept Generator")
    st.write(
        "Choose a market segment. GameScout sends only the selected market evidence and available review signals "
        "to a Cloudflare Workers AI backend, then returns one original mobile game concept."
    )

    opportunities = data["opportunities"].copy()
    generator_candidates = opportunities[
        opportunities["segment_type"] != "Mixed / low-confidence cluster"
    ].sort_values("adjusted_opportunity_score", ascending=False)

    selected_name = st.selectbox("Market segment", generator_candidates["cluster_name"].tolist())
    opportunity = generator_candidates[generator_candidates["cluster_name"] == selected_name].iloc[0]
    cluster_id = int(opportunity["cluster_id"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Opportunity score", f"{float(opportunity['adjusted_opportunity_score']):.1f}/100")
    c2.metric("Avg rating", format_float(opportunity["average_rating"], 2))
    c3.metric("Ranked games", int(opportunity["game_count"]))
    c4.metric("Developers", int(opportunity["developer_count"]))

    left, right = st.columns(2)
    with left:
        st.subheader("Market evidence")
        st.write(str(opportunity["signal_summary"]))
        st.markdown("**Representative games**")
        for game in split_pipe(opportunity["representative_games"]):
            st.write(f"• {game}")
        st.markdown("**Keywords**")
        st.write(" · ".join(split_pipe(opportunity["keywords"])[:10]))

    with right:
        st.subheader("Player review evidence")
        pain_points = get_cluster_pain_points(data, cluster_id)
        if not pain_points:
            st.info("Insufficient public review coverage for this segment. The AI will use market signals only.")
        else:
            for item in pain_points:
                st.markdown(
                    f"**{pain_point_label(item['keywords'])}** · {float(item['share']):.0%} of collected negative reviews"
                )
                for example in item.get("example_reviews", [])[:1]:
                    st.caption(f'“{example}”')

    style_label = st.radio(
        "Concept direction",
        ["Safe", "Differentiated", "Bold"],
        horizontal=True,
        index=1,
        help="Safe stays closest to validated mechanics. Bold prioritizes novelty.",
    )
    style = style_label.lower()

    worker_url = get_worker_url()
    with st.expander("AI service status", expanded=False):
        st.code(worker_url)
        if st.button("Check AI service", key="health_check"):
            with st.spinner("Checking Cloudflare Worker..."):
                healthy, message = check_worker_health(worker_url)
            if healthy:
                st.success(f"Online · {message}")
            else:
                st.error(f"Unavailable · {message}")

    generate_clicked = st.button("✨ Generate Game Concept", type="primary", width="stretch")

    if generate_clicked:
        history_key = f"title_history_{selected_name}"

        if history_key not in st.session_state:
            st.session_state[history_key] = []

        previous_titles = st.session_state[history_key][-5:]

        payload = build_ai_payload(
            opportunity,
            style,
            previous_titles=previous_titles,
        )
        generated_at = time.perf_counter()
        with st.spinner("Generating a concept from market evidence…"):
            try:
                concept = generate_game_concept(payload, worker_url=worker_url)
            except RuntimeError as error:
                st.error(str(error))
                st.info(
                    "During local development run `npx wrangler dev`. For the evaluator, deploy the Worker once with "
                    "`npx wrangler deploy` so no API key is required."
                )
            else:
                st.session_state["last_concept"] = concept
                st.session_state["last_concept_segment"] = selected_name
                new_title = str(concept.get("title", "")).strip()

                if (
                    new_title
                    and new_title not in st.session_state[history_key]
                ):
                    st.session_state[history_key].append(new_title)
                st.session_state["last_generation_time"] = time.perf_counter() - generated_at
                st.session_state.pop(
                    f"concept_pdf_{selected_name}",
                    None,
                )

    if ("last_concept" in st.session_state and st.session_state.get("last_concept_segment") == selected_name):
        concept = st.session_state["last_concept"]

        render_concept(concept)

        if st.session_state.get("last_generation_time") is not None:
            st.caption(
                f"Generated in "
                f"{float(st.session_state['last_generation_time']):.1f} seconds "
                f"via Cloudflare Workers AI."
            )

        pdf_key = f"concept_pdf_{selected_name}"

        if pdf_key not in st.session_state:
            st.session_state[pdf_key] = build_concept_pdf(
                concept=concept,
                segment_name=selected_name,
                opportunity_score=float(
                    opportunity["adjusted_opportunity_score"]
                ),
            )

        pdf_bytes = st.session_state[pdf_key]

        safe_title = (
            str(concept.get("title", "game_concept"))
            .lower()
            .replace(" ", "_")
            .replace(":", "")
            .replace("/", "_")
        )

        st.download_button(
            label="📄 Download concept as PDF",
            data=pdf_bytes,
            file_name=f"{safe_title}.pdf",
            mime="application/pdf",
            width="stretch",
        )

    st.caption(
        "The generated concept is creative output, grounded by the displayed evidence but not a prediction of commercial success."
    )

    render_page_navigation("Concept Generator")