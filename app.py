
import streamlit as st
import pandas as pd
from supabase import create_client

# ==========================================
# 1. PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="AI Shift Planner",
    page_icon="📅",
    layout="wide"
)

st.title("📅 AI Shift Planner")
st.subheader("Manager Dashboard")

# ==========================================
# 2. SUPABASE CONNECTION
# ==========================================

@st.cache_resource
def init_supabase():
    url = st.secrets["SUPABASE_URL"]
    secret_key = st.secrets["SUPABASE_SECRET_KEY"]

    return create_client(url, secret_key)


try:
    supabase = init_supabase()
    st.success("Supabase connection successful!")

except Exception as e:
    st.error("Could not connect to Supabase.")
    st.exception(e)
    st.stop()


# ==========================================
# 3. DATABASE FUNCTIONS
# ==========================================

def load_table(table_name):
    response = (
        supabase
        .table(table_name)
        .select("*")
        .execute()
    )

    return pd.DataFrame(response.data)


# ==========================================
# 4. LOAD DATABASE TABLES
# ==========================================

try:
    employees = load_table("employees")
    branches = load_table("branches")
    shift_requirements = load_table("shift_requirements")
    company_rules = load_table("company_rules")

except Exception as e:
    st.error("Could not load database tables.")
    st.exception(e)
    st.stop()


# ==========================================
# 5. DASHBOARD SUMMARY
# ==========================================

st.divider()
st.header("Database Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Employees", len(employees))
col2.metric("Branches", len(branches))
col3.metric("Shift Requirements", len(shift_requirements))
col4.metric("Company Rules", len(company_rules))


# ==========================================
# 6. DATABASE TABLES
# ==========================================

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "Employees",
    "Branches",
    "Shift Requirements",
    "Company Rules"
])

with tab1:
    st.subheader("Employees")
    st.dataframe(employees, use_container_width=True)

with tab2:
    st.subheader("Branches")
    st.dataframe(branches, use_container_width=True)

with tab3:
    st.subheader("Shift Requirements")
    st.dataframe(shift_requirements, use_container_width=True)

with tab4:
    st.subheader("Company Rules")
    st.dataframe(company_rules, use_container_width=True)


# ==========================================
# 7. CONNECTION STATUS
# ==========================================

st.divider()
st.caption("AI Shift Planner | Database connection active")



