
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


def get_monday():
    today = date.today()
    return today - timedelta(days=today.weekday())


# ==========================================
# 4. EMPLOYEE FORM
# ==========================================

def employee_form():

    st.title("📅 Employee Availability Form")

    st.write(
        "Please submit your availability "
        "for the upcoming working week."
    )

    try:
        employees = load_table("employees")

        if employees.empty:
            st.warning("No employees found.")
            return

        active_employees = employees[
            employees["employment_status"] == "Active"
        ].copy()

        active_employees = active_employees.sort_values(
            "employee_id"
        )

        employee_id = st.selectbox(
            "Your Employee ID",
            active_employees["employee_id"].tolist()
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

        week_start = st.date_input(
            "Week starting on (Monday)",
            value=get_monday()
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
            f"Choose exactly {required_days_off} days off",
            weekdays
        )

        st.caption(
            f"Selected: {len(required_days)} "
            f"of {required_days_off} days."
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
            "Additional comment (optional)"
        )

        if st.button("Submit availability", type="primary"):

            if week_start.weekday() != 0:
                st.error("The selected week must start on Monday.")

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
                    supabase.table(
                        "weekly_preferences"
                    ).upsert(
                        payload,
                        on_conflict="employee_id,week_start_date"
                    ).execute()

                    st.success(
                        "Your availability was submitted successfully!"
                    )

                except Exception as e:
                    st.error("Could not save your submission.")
                    st.exception(e)

    except Exception as e:
        st.error("Could not load employee data.")
        st.exception(e)


# ==========================================
# 5. EMPLOYEE PAGE ROUTING
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
# 7. WEEK SELECTION
# ==========================================

st.divider()
st.header("Weekly Schedule Management")

selected_week = st.date_input(
    "Select week (Monday)",
    value=get_monday(),
    key="manager_week"
)

if selected_week.weekday() != 0:
    st.warning("Please select a Monday.")
    st.stop()


# ==========================================
# 8. LOAD WEEKLY PREFERENCES
# ==========================================

try:
    preferences = load_table("weekly_preferences")

except Exception as e:
    st.error("Could not load weekly preferences.")
    st.exception(e)
    st.stop()


if not preferences.empty:

    preferences["week_start_date"] = pd.to_datetime(
        preferences["week_start_date"]
    ).dt.date

    week_preferences = preferences[
        preferences["week_start_date"] == selected_week
    ].copy()

else:
    week_preferences = pd.DataFrame()


# ==========================================
# 9. MERGE EMPLOYEES AND SUBMISSIONS
# ==========================================

active_employees = employees[
    employees["employment_status"] == "Active"
].copy()

if not week_preferences.empty:

    week_preferences = week_preferences.drop_duplicates(
        subset=["employee_id"],
        keep="last"
    )

    weekly_view = active_employees.merge(
        week_preferences[
            [
                "employee_id",
                "required_days_off",
                "preferred_shift",
                "long_shift_consent",
                "submission_status",
                "comment",
                "submitted_at"
            ]
        ],
        on="employee_id",
        how="left"
    )

else:
    weekly_view = active_employees.copy()

    for column in [
        "required_days_off",
        "preferred_shift",
        "long_shift_consent",
        "submission_status",
        "comment",
        "submitted_at"
    ]:
        weekly_view[column] = None


weekly_view["submission_status"] = (
    weekly_view["submission_status"].fillna("Missing")
)

weekly_view["required_days_off"] = weekly_view[
    "required_days_off"
].apply(
    lambda x: ", ".join(x) if isinstance(x, list) else "—"
)

for column in [
    "preferred_shift",
    "long_shift_consent",
    "comment",
    "submitted_at"
]:
    weekly_view[column] = weekly_view[column].fillna("—")


# ==========================================
# 10. WEEKLY SUBMISSION SUMMARY
# ==========================================

total_employees = len(active_employees)

received_count = int(
    (weekly_view["submission_status"] == "Received").sum()
)

missing_count = total_employees - received_count

st.divider()
st.header("Weekly Submission Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Active Employees", total_employees)
col2.metric("Received Submissions", received_count)
col3.metric("Missing Submissions", missing_count)


# ==========================================
# 11. DASHBOARD TABS
# ==========================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Employees",
    "Branches",
    "Shift Requirements",
    "Company Rules",
    "Weekly Preferences"
])


with tab1:
    st.subheader("Employees")
    st.dataframe(
        employees,
        use_container_width=True,
        hide_index=True
    )


with tab2:
    st.subheader("Branches")
    st.dataframe(
        branches,
        use_container_width=True,
        hide_index=True
    )


with tab3:
    st.subheader("Shift Requirements")
    st.dataframe(
        shift_requirements,
        use_container_width=True,
        hide_index=True
    )


with tab4:
    st.subheader("Company Rules")
    st.dataframe(
        company_rules,
        use_container_width=True,
        hide_index=True
    )


with tab5:
    st.subheader("Weekly Employee Preferences")

    st.write(
        f"Week starting: {selected_week}"
    )

    st.dataframe(
        weekly_view[
            [
                "employee_id",
                "work_days_per_week",
                "days_off_per_week",
                "required_days_off",
                "preferred_shift",
                "long_shift_consent",
                "submission_status",
                "comment"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Employees with Missing Submissions")

    missing_employees = weekly_view[
        weekly_view["submission_status"] == "Missing"
    ]

    if missing_employees.empty:
        st.success("All employees have submitted their preferences.")

    else:
        st.dataframe(
            missing_employees[
                ["employee_id", "work_days_per_week", "days_off_per_week"]
            ],
            use_container_width=True,
            hide_index=True
        )


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.caption(
    "AI Shift Planner | Manager Dashboard"
)


