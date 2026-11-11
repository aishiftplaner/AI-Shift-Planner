
import streamlit as st
from supabase import create_client

st.set_page_config(
    page_title="AI Shift Planner",
    page_icon="📅",
    layout="wide"
)

st.title("📅 AI Shift Planner")
st.subheader("Inteligentní plánování směn")

st.write(
    "Systém pro analýzu dostupnosti zaměstnanců "
    "a přípravu týdenního rozvrhu směn."
)

st.divider()

st.header("Kontrola připojení k databázi")

try:
    supabase = create_client(
        st.secrets["SUPABASE_URL"].strip(),
        st.secrets["SUPABASE_KEY"].strip()
    )

    response = (
        supabase
        .table("employees")
        .select("employee_id")
        .execute()
    )

    employees = response.data or []

    st.success("Připojení k databázi Supabase je úspěšné!")

    st.metric(
        label="Počet zaměstnanců v databázi",
        value=len(employees)
    )

except Exception as e:
    st.error("Připojení k databázi se nezdařilo.")
    with st.expander("Technické podrobnosti"):
        st.code(str(e))

st.divider()

st.header("Hlavní funkce systému")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Conflict Analysis")
    st.write("Analýza konfliktů a chyb ve vstupních údajích.")

with col2:
    st.subheader("2. Planning Assistant")
    st.write("Příprava návrhu týdenního rozvrhu směn.")

with col3:
    st.subheader("3. Decision Support")
    st.write("Vysvětlení problémů a návrhy alternativních řešení.")
