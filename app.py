
import streamlit as st

st.set_page_config(
    page_title="AI Shift Planner",
    page_icon="📅",
    layout="wide"
)

st.title("📅 AI Shift Planner")

st.subheader("AI-powered employee shift scheduling system")

st.write(
    """
    Welcome to AI Shift Planner!

    The system helps managers prepare weekly employee schedules,
    identify scheduling conflicts and make informed decisions.
    """
)

st.divider()

st.header("AI Agent")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🔍 Conflict Analysis")
    st.write(
        "Identifies missing information, contradictions "
        "and potential scheduling conflicts."
    )

with col2:
    st.subheader("📊 Planning Assistant")
    st.write(
        "Prepares the scheduling task and coordinates "
        "the scheduling algorithm."
    )

with col3:
    st.subheader("💡 Decision Support")
    st.write(
        "Explains scheduling problems and suggests "
        "alternative solutions to the manager."
    )

st.divider()

st.info(
    "Initial application version. Scheduling, database "
    "and AI integration will be added in the next stages."
)
