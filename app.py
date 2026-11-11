
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
st.header("Kontrola databáze Supabase")

tables = [
    "employees",
    "branches",
    "shift_requirements",
    "weekly_preferences",
    "company_rules",
    "schedule_runs",
    "validation_issues",
    "schedule_assignments",
    "manager_approvals"
]

try:
    supabase = create_client(
        st.secrets["SUPABASE_URL"].strip(),
        st.secrets["SUPABASE_KEY"].strip()
    )

    results = []
    all_successful = True

    for table in tables:
        try:
            response = (
                supabase
                .table(table)
                .select("*")
                .limit(1000)
                .execute()
            )

            results.append({
                "Tabulka": table,
                "Status": "OK",
                "Počet záznamů": len(response.data or [])
            })

        except Exception as e:
            all_successful = False
            results.append({
                "Tabulka": table,
                "Status": "CHYBA",
                "Počet záznamů": str(e)
            })

    if all_successful:
        st.success("Všech 9 tabulek je dostupných!")
    else:
        st.warning("Některé tabulky nejsou dostupné.")

    st.dataframe(results, use_container_width=True)

except Exception as e:
    st.error("Připojení k databázi se nezdařilo.")
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

