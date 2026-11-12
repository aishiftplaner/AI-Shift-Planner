
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
# 4. VALIDATION ENGINE
# ==========================================

def validate_week_data(active_employees, week_preferences):

    issues = []

    weekdays = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    valid_shifts = ["Morning", "Evening", "Flexible"]
    valid_consent = ["Yes", "No", "Unknown"]

    submissions = {}

    if not week_preferences.empty:
        for _, row in week_preferences.iterrows():
            submissions[row["employee_id"]] = row

    for _, employee in active_employees.iterrows():

        employee_id = employee["employee_id"]

        work_days = int(employee["work_days_per_week"])
        days_off = int(employee["days_off_per_week"])

        # Check employee contract

        if work_days + days_off != 7:

            issues.append({
                "employee_id": employee_id,
                "issue_type": "INVALID_CONTRACT",
                "severity": "Critical",
                "description": (
                    "Working days and days off do not add up to 7."
                ),
                "recommended_action": (
                    "Check the employee contract data."
                )
            })

        # Check submission existence

        if employee_id not in submissions:

            issues.append({
                "employee_id": employee_id,
                "issue_type": "MISSING_SUBMISSION",
                "severity": "Critical",
                "description": (
                    "Employee has not submitted weekly preferences."
                ),
                "recommended_action": (
                    "Request the employee to complete the form."
                )
            })

            continue

        submission = submissions[employee_id]

        # Check submission status

        if submission.get("submission_status") != "Received":

            issues.append({
                "employee_id": employee_id,
                "issue_type": "INVALID_SUBMISSION_STATUS",
                "severity": "Critical",
                "description": (
                    "Submission is not marked as Received."
                ),
                "recommended_action": (
                    "Check and correct the submission."
                )
            })

            continue

        # Check days off

        selected_days = submission.get("required_days_off")

        if not isinstance(selected_days, list):

            issues.append({
                "employee_id": employee_id,
                "issue_type": "INVALID_DAYS_OFF",
                "severity": "Critical",
                "description": (
                    "Days off are missing or have an invalid format."
                ),
                "recommended_action": (
                    "Ask the employee to submit days off again."
                )
            })

        else:

            if len(selected_days) != days_off:

                issues.append({
                    "employee_id": employee_id,
                    "issue_type": "WRONG_DAYS_OFF_COUNT",
                    "severity": "Critical",
                    "description": (
                        f"Contract requires {days_off} days off, "
                        f"but employee selected {len(selected_days)}."
                    ),
                    "recommended_action": (
                        "Ask the employee to select the correct number of days."
                    )
                })

            if len(selected_days) != len(set(selected_days)):

                issues.append({
                    "employee_id": employee_id,
                    "issue_type": "DUPLICATE_DAYS",
                    "severity": "Critical",
                    "description": (
                        "The same day was selected more than once."
                    ),
                    "recommended_action": (
                        "Correct the selected days off."
                    )
                })

            invalid_days = [
                day for day in selected_days
                if day not in weekdays
            ]

            if invalid_days:

                issues.append({
                    "employee_id": employee_id,
                    "issue_type": "INVALID_WEEKDAY",
                    "severity": "Critical",
                    "description": (
                        f"Invalid weekday values: {invalid_days}"
                    ),
                    "recommended_action": (
                        "Use valid weekday names."
                    )
                })

        # Check preferred shift

        preferred_shift = submission.get("preferred_shift")

        if preferred_shift not in valid_shifts:

            issues.append({
                "employee_id": employee_id,
                "issue_type": "INVALID_SHIFT_PREFERENCE",
                "severity": "Critical",
                "description": (
                    "Preferred shift is missing or invalid."
                ),
                "recommended_action": (
                    "Ask the employee to select a valid shift preference."
                )
            })

        # Check Long Shift consent

        consent = submission.get("long_shift_consent")

        if consent not in valid_consent:

            issues.append({
                "employee_id": employee_id,
                "issue_type": "INVALID_LONG_SHIFT_CONSENT",
                "severity": "Critical",
                "description": (
                    "Long Shift consent is missing or invalid."
                ),
                "recommended_action": (
                    "Ask the employee to clarify their Long Shift consent."
                )
            )

        elif consent == "Unknown":

            issues.append({
                "employee_id": employee_id,
                "issue_type": "UNKNOWN_LONG_SHIFT_CONSENT",
                "severity": "Critical",
                "description": (
                    "Long Shift consent is Unknown. "
                    "Scheduling is blocked until clarified."
                ),
                "recommended_action": (
                    "Ask the employee to select Yes or No."
                )
            })

    return issues


# ==========================================
# 5. EMPLOYEE AVAILABILITY FORM
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
# 6. EMPLOYEE PAGE ROUTING
# ==========================================

page = st.query_params.get("page", "dashboard")

if page == "employee":
    employee_form()
    st.stop()


# ==========================================
# 7. MANAGER DASHBOARD
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
# 8. WEEK SELECTION
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
# 9. LOAD WEEKLY PREFERENCES
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
# 10. PREPARE EMPLOYEE DATA
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
# 11. WEEKLY SUBMISSION SUMMARY
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
# 12. DASHBOARD TABS
# ==========================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Employees",
    "Branches",
    "Shift Requirements",
    "Company Rules",
    "Weekly Preferences",
    "Validation Engine"
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
# 13. VALIDATION ENGINE INTERFACE
# ==========================================

with tab6:

    st.subheader("Weekly Data Validation")

    st.write(
        "The Validation Engine checks employee submissions "
        "before the scheduling algorithm can start."
    )

    st.write(f"Week starting: {selected_week}")

    if st.button("Run Validation", type="primary"):

        validation_issues = validate_week_data(
            active_employees,
            week_preferences
        )

        st.session_state["validation_result"] = {
            "week": selected_week.isoformat(),
            "issues": validation_issues
        }


    result = st.session_state.get("validation_result")

    if result and result["week"] == selected_week.isoformat():

        issues = result["issues"]

        critical_count = sum(
            1 for issue in issues
            if issue["severity"] == "Critical"
        )

        warning_count = sum(
            1 for issue in issues
            if issue["severity"] == "Warning"
        )

        st.divider()

        col1, col2 = st.columns(2)

        col1.metric("Critical Issues", critical_count)
        col2.metric("Warnings", warning_count)

        if critical_count > 0:

            st.error(
                "BLOCKED — Scheduling cannot start. "
                "Critical issues must be resolved."
            )

        elif warning_count > 0:

            st.warning(
                "Review the warnings before proceeding."
            )

        else:

            st.success(
                "READY — All employee submissions passed validation."
            )

        if issues:

            st.subheader("Validation Issues")

            st.dataframe(
                pd.DataFrame(issues),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.success(
                "No validation issues found."
            )

    else:

        st.info(
            "Select a week and click Run Validation "
            "to check the submissions."
        )


# ==========================================
# 14. FOOTER
# ==========================================

st.divider()

st.caption(
    "AI Shift Planner | Manager Dashboard"
)



