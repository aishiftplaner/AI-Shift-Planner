
import streamlit as st
import pandas as pd

from supabase import create_client
from datetime import date, datetime, timedelta, timezone


# ==========================================
# 1. PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="AI Shift Planner",
    page_icon="📅",
    layout="wide"
)


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
# 4. EMPLOYEE AVAILABILITY FORM
# ==========================================

def employee_form():

    st.title("📅 Employee Availability Form")
    st.write(
        "Please submit your availability "
        "and shift preferences for the upcoming week."
    )

    try:
        employees = load_table("employees")

        if employees.empty:
            st.error("No employees found in the database.")
            return

        active_employees = employees[
            employees["employment_status"] == "Active"
        ].copy()

        active_employees = active_employees.sort_values(
            "employee_id"
        )

        employee_ids = active_employees["employee_id"].tolist()

        employee_id = st.selectbox(
            "Your Employee ID",
            employee_ids
        )

        employee = active_employees[
            active_employees["employee_id"] == employee_id
        ].iloc[0]

        required_days_off = int(employee["days_off_per_week"])
        work_days = int(employee["work_days_per_week"])

        st.info(
            f"Your contract: {work_days} working days "
            f"and {required_days_off} days off per week."
        )

        st.divider()

        today = date.today()
        monday = today - timedelta(days=today.weekday())

        week_start = st.date_input(
            "Week starting on (Monday)",
            value=monday
        )

        weekdays = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]

        required_days = st.multiselect(
            f"Choose your days off (exactly {required_days_off})",
            weekdays
        )

        st.caption(
            f"Selected: {len(required_days)} of "
            f"{required_days_off} required days off."
        )

        preferred_shift = st.selectbox(
            "Preferred shift",
            ["Morning", "Evening", "Flexible"]
        )

        long_shift_consent = st.selectbox(
            "Are you willing to work a Long Shift (09:00–21:00)?",
            ["Unknown", "Yes", "No"]
        )

        comment = st.text_area(
            "Additional comment (optional)",
            placeholder="Write any additional information here..."
        )

        submitted = st.button(
            "Submit availability",
            type="primary"
        )

        if submitted:

            if week_start.weekday() != 0:
                st.error(
                    "The selected week must start on Monday."
                )

            elif len(required_days) != required_days_off:
                st.error(
                    f"Please select exactly "
                    f"{required_days_off} days off."
                )

            else:

                payload = {
                    "employee_id": employee_id,
                    "week_start_date": week_start.isoformat(),
                    "required_days_off": required_days,
                    "preferred_shift": preferred_shift,
                    "long_shift_consent": long_shift_consent,
                    "submission_status": "Received",
                    "comment": comment,
                    "submitted_at": datetime.now(
                        timezone.utc
                    ).isoformat()
                }

                try:
                    response = (
                        supabase
                        .table("weekly_preferences")
                        .upsert(
                            payload,
                            on_conflict="employee_id,week_start_date"
                        )
                        .execute()
                    )

                    st.success(
                        "Your availability has been submitted successfully!"
                    )

                except Exception as e:
                    st.error("Could not save your submission.")
                    st.exception(e)

    except Exception as e:
        st.error("Could not load employee data.")
        st.exception(e)


# ==========================================
# 5. SELECT APPLICATION PAGE
# ==========================================

page = st.query_params.get("page", "dashboard")

if page == "employee":
    employee_form()
    st.stop()


# ==========================================
# 6. MANAGER DASHBOARD
# ==========================================

st.title("📅 AI Shift Planner")
st.subheader("Manager Dashboard")

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
# 7. DASHBOARD SUMMARY
# ==========================================

st.divider()
st.header("Database Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Employees", len(employees))
col2.metric("Branches", len(branches))
col3.metric("Shift Requirements", len(shift_requirements))
col4.metric("Company Rules", len(company_rules))


# ==========================================
# 8. DATABASE TABLES
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
# 9. FOOTER
# ==========================================

st.divider()
st.caption(
    "AI Shift Planner | Manager Dashboard"
)




