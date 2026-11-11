
import streamlit as st
from supabase import create_client

st.set_page_config(
    page_title="AI Shift Planner",
    page_icon="📅",
    layout="wide"
)

# Připojení k databázi
@st.cache_resource
def connect_supabase():
    return create_client(
        st.secrets["SUPABASE_URL"].strip(),
        st.secrets["SUPABASE_KEY"].strip()
    )

@st.cache_data(ttl=30)
def load_data(table, columns="*"):
    db = connect_supabase()
    response = (
        db.table(table)
        .select(columns)
        .execute()
    )
    return response.data or []

# Hlavička aplikace
st.title("📅 AI Shift Planner")
st.subheader("Manager Dashboard")
st.caption("Systém pro správu a plánování pracovních směn.")

st.divider()

try:
    employees = load_data(
        "employees",
        "employee_id, contract_type, work_days_per_week, "
        "days_off_per_week, employment_status"
    )

    branches = load_data(
        "branches",
        "branch_id, branch_name, shift_type, is_active"
    )

    shifts = load_data(
        "shift_requirements",
        "branch_id, shift_type, start_time, end_time, "
        "required_employees, break_minutes, is_active"
    )

    rules = load_data(
        "company_rules",
        "rule_key, rule_category, rule_value, "
        "rule_description, is_active"
    )

    # Souhrnné ukazatele
    st.header("Přehled společnosti")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Zaměstnanci", len(employees))
    col2.metric("Pobočky", len(branches))
    col3.metric("Typy směn", len(shifts))
    col4.metric("Pravidla", len(rules))

    st.divider()

    # Jednotlivé sekce
    tab1, tab2, tab3, tab4 = st.tabs([
        "Zaměstnanci",
        "Pobočky",
        "Požadavky na směny",
        "Pravidla společnosti"
    ])

    with tab1:
        st.header("Seznam zaměstnanců")

        if employees:
            st.dataframe(
                employees,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Zatím nejsou k dispozici žádní zaměstnanci.")

    with tab2:
        st.header("Seznam poboček")

        if branches:
            st.dataframe(
                branches,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Zatím nejsou k dispozici žádné pobočky.")

    with tab3:
        st.header("Požadavky na směny")

        if shifts:
            st.dataframe(
                shifts,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Zatím nejsou nastaveny požadavky na směny.")

    with tab4:
        st.header("Pravidla společnosti")

        if rules:
            st.dataframe(
                rules,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Zatím nejsou nastavena pravidla.")

    st.divider()
    st.caption("AI Shift Planner | Testovací verze")

except Exception as e:
    st.error("Nepodařilo se načíst data z databáze.")
    with st.expander("Technické podrobnosti"):
        st.code(str(e))


