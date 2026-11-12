import streamlit as st
import pandas as pd
import numpy as np

from supabase import create_client
from datetime import date, datetime, timedelta, timezone, time
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

# ==========================================
# 1. PAGE SETTINGS
# ==========================================
st.set_page_config(page_title="AI Shift Planner", page_icon="📅", layout="wide")

# ==========================================
# 2. SUPABASE CONNECTION
# ==========================================
@st.cache_resource
def init_supabase():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_SECRET_KEY"]
    )

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
    response = supabase.table(table_name).select("*").execute()
    return pd.DataFrame(response.data or [])

def get_monday():
    today = date.today()
    return today - timedelta(days=today.weekday())

def save_validation_issues(issues, selected_week):
    week_label = selected_week.isoformat()
    current_issue_keys = set()
    records = []

    for issue in issues:
        employee_id = issue.get("employee_id")
        issue_type = issue["issue_type"]
        current_issue_keys.add((
            str(employee_id) if employee_id is not None else None,
            str(issue_type)
        ))
        records.append({
            "run_id": None,
            "employee_id": employee_id,
            "week_start_date": week_label,
            "work_date": None,
            "issue_type": issue_type,
            "severity": issue["severity"],
            "description": f"Week starting {week_label}. {issue['description']}",
            "recommended_action": issue.get("recommended_action"),
            "resolution_status": "Open"
        })

    if records:
        supabase.table("validation_issues").upsert(
            records,
            on_conflict="week_start_date,employee_id,issue_type"
        ).execute()

    existing = (
        supabase.table("validation_issues")
        .select("issue_id,employee_id,issue_type")
        .eq("week_start_date", week_label)
        .eq("resolution_status", "Open")
        .execute()
    ).data or []

    resolved_count = 0
    for old in existing:
        old_key = (
            str(old["employee_id"]) if old.get("employee_id") is not None else None,
            str(old["issue_type"])
        )
        if old_key not in current_issue_keys:
            supabase.table("validation_issues").update(
                {"resolution_status": "Resolved"}
            ).eq("issue_id", old["issue_id"]).eq(
                "resolution_status", "Open"
            ).execute()
            resolved_count += 1

    return {"saved_count": len(records), "resolved_count": resolved_count}

# ==========================================
# 4. VALIDATION ENGINE
# ==========================================
def validate_week_data(active_employees, week_preferences):
    issues = []
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"]
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

        if work_days + days_off != 7:
            issues.append({
                "employee_id": employee_id,
                "issue_type": "INVALID_CONTRACT",
                "severity": "Critical",
                "description": "Working days and days off do not add up to 7.",
                "recommended_action": "Check the employee contract data."
            })

        if employee_id not in submissions:
            issues.append({
                "employee_id": employee_id,
                "issue_type": "MISSING_SUBMISSION",
                "severity": "Critical",
                "description": "Employee has not submitted weekly preferences.",
                "recommended_action": "Request the employee to complete the form."
            })
            continue

        submission = submissions[employee_id]
        if submission.get("submission_status") != "Received":
            issues.append({
                "employee_id": employee_id,
                "issue_type": "INVALID_SUBMISSION_STATUS",
                "severity": "Critical",
                "description": "Submission is not marked as Received.",
                "recommended_action": "Check and correct the submission."
            })
            continue

        selected_days = submission.get("required_days_off")
        if not isinstance(selected_days, list):
            issues.append({
                "employee_id": employee_id,
                "issue_type": "INVALID_DAYS_OFF",
                "severity": "Critical",
                "description": "Days off are missing or have an invalid format.",
                "recommended_action": "Ask the employee to submit days off again."
            })
        else:
            if len(selected_days) != days_off:
                issues.append({
                    "employee_id": employee_id,
                    "issue_type": "WRONG_DAYS_OFF_COUNT",
                    "severity": "Critical",
                    "description": f"Contract requires {days_off} days off, but employee selected {len(selected_days)}.",
                    "recommended_action": "Ask the employee to select the correct number of days."
                })
            if len(selected_days) != len(set(map(str, selected_days))):
                issues.append({
                    "employee_id": employee_id,
                    "issue_type": "DUPLICATE_DAYS",
                    "severity": "Critical",
                    "description": "The same day was selected more than once.",
                    "recommended_action": "Correct the selected days off."
                })
            invalid_days = [d for d in selected_days if d not in weekdays]
            if invalid_days:
                issues.append({
                    "employee_id": employee_id,
                    "issue_type": "INVALID_WEEKDAY",
                    "severity": "Critical",
                    "description": f"Invalid weekday values: {invalid_days}",
                    "recommended_action": "Use valid weekday names."
                })

        if submission.get("preferred_shift") not in valid_shifts:
            issues.append({
                "employee_id": employee_id,
                "issue_type": "INVALID_SHIFT_PREFERENCE",
                "severity": "Critical",
                "description": "Preferred shift is missing or invalid.",
                "recommended_action": "Ask the employee to select a valid shift preference."
            })

        consent = submission.get("long_shift_consent")
        if consent not in valid_consent:
            issues.append({
                "employee_id": employee_id,
                "issue_type": "INVALID_LONG_SHIFT_CONSENT",
                "severity": "Critical",
                "description": "Long Shift consent is missing or invalid.",
                "recommended_action": "Ask the employee to clarify their Long Shift consent."
            })
        elif consent == "Unknown":
            issues.append({
                "employee_id": employee_id,
                "issue_type": "UNKNOWN_LONG_SHIFT_CONSENT",
                "severity": "Critical",
                "description": "Long Shift consent is Unknown. Scheduling is blocked until clarified.",
                "recommended_action": "Ask the employee to select Yes or No."
            })
    return issues

# ==========================================
# 5. EMPLOYEE AVAILABILITY FORM
# ==========================================
def employee_form():
    st.title("📅 Employee Availability Form")
    st.write("Please submit your availability for the upcoming working week.")
    try:
        employees = load_table("employees")
        if employees.empty:
            st.warning("No employees found.")
            return
        active = employees[employees["employment_status"] == "Active"].copy()
        if active.empty:
            st.warning("No active employees found.")
            return
        active = active.sort_values("employee_id")
        employee_id = st.selectbox("Your Employee ID", active["employee_id"].tolist())
        employee = active[active["employee_id"] == employee_id].iloc[0]
        days_off_count = int(employee["days_off_per_week"])
        work_days = int(employee["work_days_per_week"])
        st.info(f"Your contract: {work_days} working days and {days_off_count} days off per week.")
        st.divider()
        week_start = st.date_input("Week starting on (Monday)", value=get_monday())
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        required_days = st.multiselect(f"Choose exactly {days_off_count} days off", weekdays)
        st.caption(f"Selected: {len(required_days)} of {days_off_count} days.")
        preferred_shift = st.selectbox("Preferred shift", ["Morning", "Evening", "Flexible"])
        consent = st.selectbox("Are you willing to work a Long Shift (09:00–21:00)?",
                               ["Unknown", "Yes", "No"])
        comment = st.text_area("Additional comment (optional)")

        if st.button("Submit availability", type="primary"):
            if week_start.weekday() != 0:
                st.error("The selected week must start on Monday.")
            elif len(required_days) != days_off_count:
                st.error(f"Please select exactly {days_off_count} days off.")
            else:
                payload = {
                    "employee_id": employee_id,
                    "week_start_date": week_start.isoformat(),
                    "required_days_off": required_days,
                    "preferred_shift": preferred_shift,
                    "long_shift_consent": consent,
                    "submission_status": "Received",
                    "comment": comment,
                    "submitted_at": datetime.now(timezone.utc).isoformat()
                }
                try:
                    supabase.table("weekly_preferences").upsert(
                        payload, on_conflict="employee_id,week_start_date"
                    ).execute()
                    st.success("Your availability was submitted successfully!")
                except Exception as e:
                    st.error("Could not save your submission.")
                    st.exception(e)
    except Exception as e:
        st.error("Could not load employee data.")
        st.exception(e)

# ==========================================
# 6. SCHEDULING ALGORITHM
# ==========================================
SHIFT_ORDER = ["Morning", "Evening", "Long"]
PREFERENCE_PENALTY = 10.0
MIN_DAILY_REST_HOURS = 11.0
MIN_WEEKLY_REST_HOURS = 35.0

def time_to_minutes(value):
    """Convert a database time value such as 06:45:00 to minutes after midnight."""
    if value is None:
        raise ValueError("A shift requirement is missing a start or end time.")
    if isinstance(value, time):
        return value.hour * 60 + value.minute
    parts = str(value).split(":")
    return int(parts[0]) * 60 + int(parts[1])

def prepare_shift_requirements(requirements, branches):
    if requirements.empty:
        raise ValueError("No active shift requirements were found.")
    req = requirements.copy()
    if "is_active" in req.columns:
        req = req[req["is_active"] == True].copy()
    if req.empty:
        raise ValueError("No active shift requirements were found.")

    needed = {"branch_id", "shift_type", "start_time", "end_time",
              "required_employees", "break_minutes"}
    missing = needed - set(req.columns)
    if missing:
        raise ValueError(f"shift_requirements is missing columns: {sorted(missing)}")

    req["required_employees"] = pd.to_numeric(req["required_employees"], errors="coerce")
    req["break_minutes"] = pd.to_numeric(req["break_minutes"], errors="coerce")
    if req[["required_employees", "break_minutes"]].isna().any().any():
        raise ValueError("Staffing or break values contain empty/non-numeric values.")
    if (req["required_employees"] < 1).any():
        raise ValueError("Every active shift requirement must require at least one employee.")
    if not set(req["shift_type"]).issubset(set(SHIFT_ORDER)):
        raise ValueError("A shift requirement contains an unsupported shift_type.")

    branch_ids = set(branches["branch_id"].astype(str))
    if not set(req["branch_id"].astype(str)).issubset(branch_ids):
        raise ValueError("A shift requirement refers to a branch that was not loaded.")
    return req

def build_solver_constraints(employees, preferences, demand, shift_times):
    n_emp = len(employees)
    n_days = 7
    n_shifts = len(SHIFT_ORDER)
    n_vars = n_emp * n_days * n_shifts

    def idx(e, d, s):
        return (e * n_days + d) * n_shifts + s

    rows, lows, highs = [], [], []

    def add_constraint(coefficients, lower, upper):
        rows.append(coefficients)
        lows.append(lower)
        highs.append(upper)

    pref_by_id = {r["employee_id"]: r for _, r in preferences.iterrows()}

    # Exact contracted working days for every employee.
    for e, (_, emp) in enumerate(employees.iterrows()):
        coefficients = {}
        for d in range(7):
            for s in range(n_shifts):
                coefficients[idx(e, d, s)] = 1
        add_constraint(coefficients, int(emp["work_days_per_week"]),
                       int(emp["work_days_per_week"]))

    # Exactly the required number of employees on each shift each day.
    for d in range(7):
        for s, shift in enumerate(SHIFT_ORDER):
            coefficients = {idx(e, d, s): 1 for e in range(n_emp)}
            add_constraint(coefficients, demand[shift], demand[shift])

    # At most one shift per employee per day.
    for e in range(n_emp):
        for d in range(7):
            coefficients = {idx(e, d, s): 1 for s in range(n_shifts)}
            add_constraint(coefficients, 0, 1)

    # Exact mandatory days off.
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"]
    for e, (_, emp) in enumerate(employees.iterrows()):
        pref = pref_by_id[emp["employee_id"]]
        days_off = pref["required_days_off"]
        for d, day_name in enumerate(weekdays):
            if day_name in days_off:
                coefficients = {idx(e, d, s): 1 for s in range(n_shifts)}
                add_constraint(coefficients, 0, 0)

    # Long shifts require explicit Yes consent; fair cap of one Long per week.
    for e, (_, emp) in enumerate(employees.iterrows()):
        pref = pref_by_id[emp["employee_id"]]
        long_var_indices = [idx(e, d, 2) for d in range(7)]
        if pref["long_shift_consent"] != "Yes":
            for var in long_var_indices:
                add_constraint({var: 1}, 0, 0)
        else:
            add_constraint({var: 1 for var in long_var_indices}, 0, 1)

    # Minimum 11 hours between any two consecutive shifts, including week wrap.
    for e in range(n_emp):
        for d1 in range(7):
            for s1, sh1 in enumerate(SHIFT_ORDER):
                if sh1 not in shift_times:
                    continue
                start1, end1 = shift_times[sh1]
                for delta in range(1, 7):
                    d2 = (d1 + delta) % 7
                    for s2, sh2 in enumerate(SHIFT_ORDER):
                        if sh2 not in shift_times:
                            continue
                        start2, end2 = shift_times[sh2]
                        rest_hours = (delta * 24 + start2 / 60 - end1 / 60)
                        if rest_hours < MIN_DAILY_REST_HOURS:
                            add_constraint({
                                idx(e, d1, s1): 1,
                                idx(e, d2, s2): 1
                            }, 0, 1)

    matrix = lil_matrix((len(rows), n_vars), dtype=float)
    for r, coefficients in enumerate(rows):
        for c, value in coefficients.items():
            matrix[r, c] = value

    lower_bounds = np.zeros(n_vars)
    upper_bounds = np.ones(n_vars)
    integrality = np.ones(n_vars, dtype=int)

    # Preference deviations are soft constraints; Flexible has no penalty.
    objective = np.zeros(n_vars)
    for e, (_, emp) in enumerate(employees.iterrows()):
        pref = pref_by_id[emp["employee_id"]]["preferred_shift"]
        for d in range(7):
            for s, shift in enumerate(SHIFT_ORDER):
                if shift == "Long":
                    objective[idx(e, d, s)] = 0.25
                elif pref in ("Morning", "Evening") and pref != shift:
                    objective[idx(e, d, s)] = PREFERENCE_PENALTY

    return (
        objective, integrality, Bounds(lower_bounds, upper_bounds),
        LinearConstraint(matrix.tocsr(), np.array(lows), np.array(highs)),
        idx
    )

def calculate_weekly_rest(assignments, employees):
    """Check each employee has at least one cyclic rest interval of 35 hours."""
    problems = []
    for employee_id in employees["employee_id"]:
        rows = assignments[assignments["employee_id"] == employee_id]
        if rows.empty:
            problems.append(f"{employee_id}: no shifts assigned")
            continue
        intervals = []
        for _, row in rows.iterrows():
            day_offset = (row["work_date"] - assignments["work_date"].min()).days
            start = time_to_minutes(row["start_time"]) / 60 + day_offset * 24
            end = time_to_minutes(row["end_time"]) / 60 + day_offset * 24
            if end <= start:
                end += 24
            intervals.append((start, end))
        intervals.sort()
        gaps = []
        for i in range(len(intervals) - 1):
            gaps.append(intervals[i + 1][0] - intervals[i][1])
        gaps.append(intervals[0][0] + 168 - intervals[-1][1])
        if max(gaps) < MIN_WEEKLY_REST_HOURS:
            problems.append(
                f"{employee_id}: longest weekly rest is {max(gaps):.1f} hours"
            )
    return problems

def run_scheduling_algorithm(active_employees, week_preferences,
                             shift_requirements, branches, selected_week):
    employees = active_employees.sort_values("employee_id").reset_index(drop=True).copy()
    preferences = week_preferences.copy()
    req = prepare_shift_requirements(shift_requirements, branches)

    if len(preferences) != len(employees):
        raise ValueError("The number of weekly submissions does not match active employees.")

    # Total staffing demand by shift type.
    demand = {shift: 0 for shift in SHIFT_ORDER}
    for shift in SHIFT_ORDER:
        demand[shift] = int(req.loc[req["shift_type"] == shift, "required_employees"].sum())

    total_demand = sum(demand.values())
    total_contract_days = int(employees["work_days_per_week"].sum())
    if total_demand * 7 != total_contract_days:
        raise ValueError(
            f"Capacity mismatch: shift requirements need {total_demand * 7} "
            f"employee-days, but contracts provide {total_contract_days}."
        )

    # A shift type must have consistent times/break duration across branches.
    shift_times = {}
    shift_breaks = {}
    for shift in SHIFT_ORDER:
        subset = req[req["shift_type"] == shift]
        if subset.empty:
            continue
        times = set((str(r["start_time"]), str(r["end_time"])) for _, r in subset.iterrows())
        if len(times) != 1:
            raise ValueError(f"{shift} shifts have inconsistent start/end times between branches.")
        first = subset.iloc[0]
        shift_times[shift] = (
            time_to_minutes(first["start_time"]),
            time_to_minutes(first["end_time"])
        )
        breaks = set(int(x) for x in subset["break_minutes"])
        if len(breaks) != 1:
            raise ValueError(f"{shift} shifts have inconsistent break durations between branches.")
        shift_breaks[shift] = int(first["break_minutes"])

    for shift in SHIFT_ORDER:
        if demand[shift] > 0 and shift not in shift_times:
            raise ValueError(f"Missing time definition for {shift}.")

    # SciPy MILP optimization.
    objective, integrality, bounds, constraints, idx = build_solver_constraints(
        employees, preferences, demand, shift_times
    )
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"time_limit": 90}
    )

    if not result.success or result.x is None:
        return {
            "feasible": False,
            "message": result.message or "The optimization solver did not find a feasible schedule.",
            "assignments": pd.DataFrame(),
            "assigned_positions": 0,
            "unassigned_positions": total_demand * 7
        }

    # Convert the mathematical solution into employee/day/shift assignments.
    planned = []
    for e, (_, emp) in enumerate(employees.iterrows()):
        for d in range(7):
            for s, shift in enumerate(SHIFT_ORDER):
                if result.x[idx(e, d, s)] > 0.5:
                    planned.append({
                        "employee_id": emp["employee_id"],
                        "work_date": selected_week + timedelta(days=d),
                        "shift_type": shift
                    })

    planned_df = pd.DataFrame(planned)
    if len(planned_df) != total_demand * 7:
        return {
            "feasible": False,
            "message": "The solver result did not match the required number of weekly positions.",
            "assignments": pd.DataFrame(),
            "assigned_positions": len(planned_df),
            "unassigned_positions": total_demand * 7 - len(planned_df)
        }

    # Assign employees to actual branches, respecting each branch's shift requirement.
    branch_rows = []
    for d in range(7):
        work_date = selected_week + timedelta(days=d)
        for shift in SHIFT_ORDER:
            employees_for_shift = sorted(
                planned_df[
                    (planned_df["work_date"] == work_date) &
                    (planned_df["shift_type"] == shift)
                ]["employee_id"].tolist()
            )
            branch_demand = req[req["shift_type"] == shift].sort_values("branch_id")
            position = 0
            for _, requirement in branch_demand.iterrows():
                for _ in range(int(requirement["required_employees"])):
                    if position >= len(employees_for_shift):
                        raise ValueError(f"Not enough employees for {shift} on {work_date}.")
                    branch_rows.append({
                        "employee_id": employees_for_shift[position],
                        "branch_id": requirement["branch_id"],
                        "work_date": work_date,
                        "shift_type": shift,
                        "start_time": str(requirement["start_time"]),
                        "end_time": str(requirement["end_time"]),
                        "break_minutes": int(requirement["break_minutes"])
                    })
                    position += 1
            if position != len(employees_for_shift):
                raise ValueError(f"Unassigned {shift} employees remain on {work_date}.")

    assignments = pd.DataFrame(branch_rows)

    # Independent post-solve checks before any schedule is saved as Feasible.
    if assignments.duplicated(["employee_id", "work_date"]).any():
        raise ValueError("Post-check failed: an employee has more than one shift on a day.")

    for employee_id, group in assignments.groupby("employee_id"):
        emp = employees[employees["employee_id"] == employee_id].iloc[0]
        pref = preferences[preferences["employee_id"] == employee_id].iloc[0]
        if len(group) != int(emp["work_days_per_week"]):
            raise ValueError(f"Post-check failed: incorrect number of workdays for {employee_id}.")
        if (group["shift_type"] == "Long").any() and pref["long_shift_consent"] != "Yes":
            raise ValueError(f"Post-check failed: Long Shift without consent for {employee_id}.")

    weekly_rest_problems = calculate_weekly_rest(assignments, employees)
    if weekly_rest_problems:
        return {
            "feasible": False,
            "message": "The solver found a staffing allocation, but the 35-hour weekly-rest check failed: "
                       + "; ".join(weekly_rest_problems[:8]),
            "assignments": pd.DataFrame(),
            "assigned_positions": len(assignments),
            "unassigned_positions": 0
        }

    return {
        "feasible": True,
        "message": "Optimal schedule found. All implemented hard constraints passed.",
        "assignments": assignments,
        "assigned_positions": len(assignments),
        "unassigned_positions": 0
    }

def create_schedule_run(selected_week):
    response = supabase.table("schedule_runs").insert({
        "week_start_date": selected_week.isoformat(),
        "run_status": "Running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "manager_approval_status": "Not_Reviewed",
        "assigned_positions": 0,
        "unassigned_positions": 0
    }).execute()
    if not response.data:
        raise RuntimeError("Supabase did not return the created schedule run.")
    return response.data[0]["run_id"]

def update_schedule_run(run_id, status, assigned, unassigned):
    payload = {
        "run_status": status,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "assigned_positions": int(assigned),
        "unassigned_positions": int(unassigned)
    }
    supabase.table("schedule_runs").update(payload).eq("run_id", run_id).execute()

def persist_schedule(run_id, assignments):
    records = []
    for _, row in assignments.iterrows():
        records.append({
            "run_id": int(run_id),
            "employee_id": row["employee_id"],
            "branch_id": row["branch_id"],
            "work_date": row["work_date"].isoformat(),
            "shift_type": row["shift_type"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "break_minutes": int(row["break_minutes"])
        })
    if records:
        # Insert, not upsert: every run is immutable history.
        supabase.table("schedule_assignments").insert(records).execute()

def approve_or_reject_run(run_id, decision, comment):
    supabase.table("manager_approvals").insert({
        "run_id": int(run_id),
        "approval_status": decision,
        "manager_comment": comment or None,
        "decided_at": datetime.now(timezone.utc).isoformat()
    }).execute()
    supabase.table("schedule_runs").update({
        "manager_approval_status": decision
    }).eq("run_id", int(run_id)).execute()

# ==========================================
# 7. PAGE ROUTING
# ==========================================
page = st.query_params.get("page", "dashboard")
if page == "employee":
    employee_form()
    st.stop()

# ==========================================
# 8. MANAGER DASHBOARD
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

st.divider()
st.header("Weekly Schedule Management")
selected_week = st.date_input("Select week (Monday)", value=get_monday(), key="manager_week")
if selected_week.weekday() != 0:
    st.warning("Please select a Monday.")
    st.stop()

try:
    preferences = load_table("weekly_preferences")
except Exception as e:
    st.error("Could not load weekly preferences.")
    st.exception(e)
    st.stop()

if not preferences.empty:
    preferences["week_start_date"] = pd.to_datetime(preferences["week_start_date"]).dt.date
    week_preferences = preferences[preferences["week_start_date"] == selected_week].copy()
else:
    week_preferences = pd.DataFrame()

active_employees = employees[employees["employment_status"] == "Active"].copy()

if not week_preferences.empty:
    week_preferences = week_preferences.drop_duplicates(subset=["employee_id"], keep="last")
    weekly_view = active_employees.merge(
        week_preferences[["employee_id", "required_days_off", "preferred_shift",
                           "long_shift_consent", "submission_status", "comment", "submitted_at"]],
        on="employee_id", how="left"
    )
else:
    weekly_view = active_employees.copy()
    for column in ["required_days_off", "preferred_shift", "long_shift_consent",
                   "submission_status", "comment", "submitted_at"]:
        weekly_view[column] = None

weekly_view["submission_status"] = weekly_view["submission_status"].fillna("Missing")
weekly_view["required_days_off"] = weekly_view["required_days_off"].apply(
    lambda x: ", ".join(map(str, x)) if isinstance(x, list) else "—"
)
for column in ["preferred_shift", "long_shift_consent", "comment", "submitted_at"]:
    weekly_view[column] = weekly_view[column].fillna("—")

total_employees = len(active_employees)
received_count = int((weekly_view["submission_status"] == "Received").sum())
missing_count = total_employees - received_count

st.divider()
st.header("Weekly Submission Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Active Employees", total_employees)
c2.metric("Received Submissions", received_count)
c3.metric("Missing Submissions", missing_count)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Employees", "Branches", "Shift Requirements", "Company Rules",
    "Weekly Preferences", "Validation Engine", "Scheduling"
])

with tab1:
    st.subheader("Employees")
    st.dataframe(employees, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Branches")
    st.dataframe(branches, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Shift Requirements")
    st.dataframe(shift_requirements, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Company Rules")
    st.dataframe(company_rules, use_container_width=True, hide_index=True)

with tab5:
    st.subheader("Weekly Employee Preferences")
    st.write(f"Week starting: {selected_week}")
    st.dataframe(
        weekly_view[["employee_id", "work_days_per_week", "days_off_per_week",
                     "required_days_off", "preferred_shift", "long_shift_consent",
                     "submission_status", "comment"]],
        use_container_width=True, hide_index=True
    )
    st.divider()
    st.subheader("Employees with Missing Submissions")
    missing_employees = weekly_view[weekly_view["submission_status"] == "Missing"]
    if missing_employees.empty:
        st.success("All employees have submitted their preferences.")
    else:
        st.dataframe(
            missing_employees[["employee_id", "work_days_per_week", "days_off_per_week"]],
            use_container_width=True, hide_index=True
        )

with tab6:
    st.subheader("Weekly Data Validation")
    st.write("The Validation Engine checks employee submissions before scheduling.")
    st.write(f"Week starting: {selected_week}")

    if st.button("Run Validation", type="primary"):
        validation_issues = validate_week_data(active_employees, week_preferences)
        st.session_state["validation_result"] = {
            "week": selected_week.isoformat(), "issues": validation_issues
        }
        try:
            sync_result = save_validation_issues(validation_issues, selected_week)
            st.session_state["validation_save_status"] = {
                "week": selected_week.isoformat(),
                "saved_count": sync_result["saved_count"],
                "resolved_count": sync_result["resolved_count"],
                "error": None
            }
        except Exception as e:
            st.session_state["validation_save_status"] = {
                "week": selected_week.isoformat(),
                "saved_count": 0, "resolved_count": 0, "error": str(e)
            }

    result = st.session_state.get("validation_result")
    if result and result["week"] == selected_week.isoformat():
        issues = result["issues"]
        critical_count = sum(1 for x in issues if x["severity"] == "Critical")
        warning_count = sum(1 for x in issues if x["severity"] == "Warning")
        st.divider()
        a, b = st.columns(2)
        a.metric("Critical Issues", critical_count)
        b.metric("Warnings", warning_count)

        if critical_count:
            st.error("BLOCKED — Scheduling cannot start. Critical issues must be resolved.")
        elif warning_count:
            st.warning("Review the warnings before proceeding.")
        else:
            st.success("READY — All employee submissions passed validation.")

        save_status = st.session_state.get("validation_save_status")
        if save_status and save_status["week"] == selected_week.isoformat():
            if save_status["error"]:
                st.error("Validation completed, but synchronizing issues in Supabase failed.")
                st.code(save_status["error"])
            else:
                st.success(
                    f"Current issues saved/updated: {save_status['saved_count']}. "
                    f"Old issues marked Resolved: {save_status['resolved_count']}."
                )
        if issues:
            st.subheader("Validation Issues")
            st.dataframe(pd.DataFrame(issues), use_container_width=True, hide_index=True)
        else:
            st.success("No validation issues found.")

with tab7:
    st.subheader("Scheduling Algorithm")
    st.write(
        "The mathematical optimizer allocates employees to shifts. "
        "Branch positions are assigned after the shift allocation."
    )
    st.info(
        "Hard rules: exact contract workdays and requested days off, "
        "maximum one shift per day, explicit Long Shift consent, "
        "minimum 11-hour daily rest and a 35-hour weekly-rest check. "
        "Shift preferences are treated as soft preferences."
    )

    validation = st.session_state.get("validation_result")
    validation_ready = bool(
        validation
        and validation.get("week") == selected_week.isoformat()
        and len(validation.get("issues", [])) == 0
    )

    if validation_ready:
        st.success("Validation status: READY")
    else:
        st.warning("Scheduling is locked. Run Validation and obtain READY for this exact week.")

    if st.button("Generate Weekly Schedule", type="primary", disabled=not validation_ready):
        run_id = None
        try:
            run_id = create_schedule_run(selected_week)
            with st.spinner("The optimization algorithm is generating the schedule..."):
                outcome = run_scheduling_algorithm(
                    active_employees, week_preferences, shift_requirements,
                    branches, selected_week
                )
            if outcome["feasible"]:
                persist_schedule(run_id, outcome["assignments"])
                update_schedule_run(
                    run_id, "Feasible", outcome["assigned_positions"],
                    outcome["unassigned_positions"]
                )
                st.session_state["last_schedule_run_id"] = int(run_id)
                st.success(f"Schedule generated successfully. Run ID: {run_id}")
            else:
                update_schedule_run(
                    run_id, "Infeasible", outcome["assigned_positions"],
                    outcome["unassigned_positions"]
                )
                st.session_state["last_schedule_run_id"] = int(run_id)
                st.error("No valid schedule was produced.")
            st.session_state["last_schedule_message"] = outcome["message"]
        except Exception as e:
            if run_id is not None:
                try:
                    update_schedule_run(run_id, "Failed", 0, 0)
                except Exception:
                    pass
            st.error("Scheduling run failed. No schedule was approved or published.")
            st.exception(e)

    if st.session_state.get("last_schedule_message"):
        st.caption(st.session_state["last_schedule_message"])

    # Load and display run history for the selected week.
    try:
        runs_response = (
            supabase.table("schedule_runs").select("*")
            .eq("week_start_date", selected_week.isoformat())
            .order("run_id", desc=True).execute()
        )
        runs = pd.DataFrame(runs_response.data or [])
    except Exception as e:
        runs = pd.DataFrame()
        st.error("Could not load schedule run history.")
        st.exception(e)

    if not runs.empty:
        st.divider()
        st.subheader("Schedule Run History")
        st.dataframe(runs, use_container_width=True, hide_index=True)

        feasible_runs = runs[runs["run_status"] == "Feasible"]
        if not feasible_runs.empty:
            selected_run_id = st.selectbox(
                "View generated schedule (Run ID)",
                feasible_runs["run_id"].astype(int).tolist(),
                key="view_schedule_run"
            )
            try:
                assignment_response = (
                    supabase.table("schedule_assignments").select("*")
                    .eq("run_id", int(selected_run_id))
                    .order("work_date").execute()
                )
                schedule_df = pd.DataFrame(assignment_response.data or [])
                if not schedule_df.empty:
                    display_df = schedule_df.copy()
                    display_df["work_date"] = pd.to_datetime(display_df["work_date"]).dt.date
                    st.subheader("Generated Weekly Schedule")
                    st.dataframe(
                        display_df[["work_date", "branch_id", "employee_id",
                                    "shift_type", "start_time", "end_time", "break_minutes"]],
                        use_container_width=True, hide_index=True
                    )

                    selected_run = feasible_runs[
                        feasible_runs["run_id"].astype(int) == int(selected_run_id)
                    ].iloc[0]
                    if selected_run["manager_approval_status"] == "Not_Reviewed":
                        st.divider()
                        st.subheader("Manager Approval")
                        manager_comment = st.text_area(
                            "Manager comment (optional)", key=f"approval_comment_{selected_run_id}"
                        )
                        approve_col, reject_col = st.columns(2)
                        with approve_col:
                            if st.button("Approve Schedule", type="primary",
                                         key=f"approve_{selected_run_id}"):
                                approve_or_reject_run(
                                    selected_run_id, "Approved", manager_comment
                                )
                                st.success("Schedule approved. The decision was saved.")
                                st.rerun()
                        with reject_col:
                            if st.button("Reject Schedule", key=f"reject_{selected_run_id}"):
                                approve_or_reject_run(
                                    selected_run_id, "Rejected", manager_comment
                                )
                                st.warning("Schedule rejected. The decision was saved.")
                                st.rerun()
                    else:
                        st.info(
                            f"Manager decision: {selected_run['manager_approval_status']}"
                        )
            except Exception as e:
                st.error("Could not load the selected schedule.")
                st.exception(e)

# ==========================================
# 9. FOOTER
# ==========================================
st.divider()
st.caption("AI Shift Planner | Manager Dashboard")
