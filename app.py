
import streamlit as st
import pandas as pd
import numpy as np
import json
import unicodedata

from supabase import create_client
from google import genai

from datetime import date, datetime, timedelta, timezone, time
from zoneinfo import ZoneInfo
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix


# ==========================================
# 1. PAGE SETTINGS
# ==========================================
st.set_page_config(
    page_title="AI Shift Planner",
    page_icon="📅",
    layout="wide"
)


def apply_light_theme():
    """Apply a consistent, high-contrast light interface."""
    st.markdown(
        """
        <style>
        :root {
            color-scheme: light;
            --text-color: #1F2937;
            --background-color: #F5F7FB;
            --secondary-background-color: #FFFFFF;
            --primary-color: #2563EB;
        }
        html, body, [data-testid="stAppViewContainer"] {
            background: #F5F7FB !important;
            color: #1F2937 !important;
        }
        [data-testid="stHeader"] {
            background: rgba(245,247,251,0.96) !important;
        }
        .stApp, .stApp p, .stApp label, .stApp h1, .stApp h2,
        .stApp h3, .stApp h4, .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stMetricLabel"],
        .stApp [data-testid="stMetricValue"] {
            color: #1F2937 !important;
        }
        [data-testid="stMetric"] {
            background: #FFFFFF !important;
            border: 1px solid #DCE3ED !important;
            padding: 16px !important;
            border-radius: 12px !important;
            box-shadow: 0 1px 3px rgba(15,23,42,0.04);
        }
        [data-testid="stTabs"] button {
            color: #334155 !important;
            font-weight: 600 !important;
            padding-top: 12px !important;
            padding-bottom: 12px !important;
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: #1D4ED8 !important;
        }
        [data-testid="stExpander"] {
            background: #FFFFFF !important;
            border: 1px solid #DCE3ED !important;
            border-radius: 12px !important;
        }
        [data-testid="stDataFrame"] {
            background: #FFFFFF !important;
            border: 1px solid #DCE3ED !important;
            border-radius: 10px !important;
            overflow: hidden !important;
        }
        /* Keep status messages readable on the light interface. */
        [data-testid="stAlert"] {
            border-radius: 10px !important;
        }
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] div,
        [data-testid="stAlert"] span {
            color: #1F2937 !important;
        }
        /* Streamlit's dataframe uses an embedded grid; the app theme is
           also fixed to light in .streamlit/config.toml. */
        [data-testid="stDataFrame"] canvas {
            background: #FFFFFF !important;
        }
        [data-testid="stDataFrame"] [role="grid"] {
            background: #FFFFFF !important;
            color: #1F2937 !important;
        }
        input, textarea, [data-baseweb="select"] > div {
            background: #FFFFFF !important;
            color: #1F2937 !important;
        }
        [data-testid="stButton"] button {
            border-radius: 9px !important;
            font-weight: 600 !important;
        }
        [data-testid="stButton"] button[kind="primary"] {
            background: #2563EB !important;
            color: #FFFFFF !important;
            border-color: #2563EB !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


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
    st.error("Nepodařilo se připojit k databázi Supabase.")
    st.exception(e)
    st.stop()


# ==========================================
# 2.1. GEMINI AI CONNECTION
# ==========================================
GEMINI_MODEL = "gemini-3.5-flash-lite"


@st.cache_resource
def init_gemini():
    api_key = st.secrets["GEMINI_API_KEY"]
    return genai.Client(api_key=api_key)


def test_gemini_connection():
    client = init_gemini()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=(
            "Reply with exactly: "
            "AI Shift Planner connection successful."
        )
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.text


# ==========================================
# 2.2. AI CONFLICT ANALYSIS
# ==========================================
def generate_ai_conflict_analysis(issues, selected_week):
    """
    Gemini explains validation findings and proposes manager actions.

    Only anonymized issue information is sent to Gemini.
    Employee IDs, names and personal comments are excluded.
    Gemini does not modify the database or schedule.
    """

    client = init_gemini()

    grouped_issues = {}

    for issue in issues:
        issue_type = str(issue.get("issue_type", "UNKNOWN"))
        severity = str(issue.get("severity", "Unknown"))
        description = str(issue.get("description", "No description"))
        action = str(
            issue.get("recommended_action", "No action specified")
        )

        key = (issue_type, severity, description, action)

        if key not in grouped_issues:
            grouped_issues[key] = {
                "issue_type": issue_type,
                "severity": severity,
                "description": description,
                "recommended_action": action,
                "count": 0
            }

        grouped_issues[key]["count"] += 1

    safe_issues = list(grouped_issues.values())

    severity_counts = {
        "Critical": sum(
            item["count"]
            for item in safe_issues
            if item["severity"] == "Critical"
        ),
        "Warning": sum(
            item["count"]
            for item in safe_issues
            if item["severity"] == "Warning"
        ),
        "Other": sum(
            item["count"]
            for item in safe_issues
            if item["severity"] not in ("Critical", "Warning")
        )
    }

    payload = {
        "week_start_date": selected_week.isoformat(),
        "total_issues": len(issues),
        "severity_counts": severity_counts,
        "grouped_issues": safe_issues
    }

    prompt = f"""
Jsi modul AI analýzy konfliktů v systému plánování směn zaměstnanců. Odpovídej výhradně česky.

Analyze the following validation results for the manager.

VALIDATION DATA:
{json.dumps(payload, ensure_ascii=False, indent=2)}

YOUR TASKS:

1. Give a short overall assessment of the validation results.
2. Explain each distinct issue type in understandable language.
3. Explain how the reported issues affect scheduling readiness.
4. Group repeated issues and mention their counts.
5. Suggest concrete actions for the manager.
6. Prioritize Critical issues before Warnings.
7. If there are no issues, explain that the submitted data passed
   the current validation checks.

IMPORTANT RULES:

- Use only facts contained in the validation data.
- Do not invent employees, shifts or staffing shortages.
- Clearly distinguish confirmed findings from possible consequences.
- Never recommend violating contract workdays, mandatory days off,
  Long Shift consent, daily rest or weekly rest requirements.
- Do not create employee assignments or a weekly schedule.
- Do not change database records or validation results.
- The manager makes the final decision.
- Do not include or infer employee identities.
- Write in clear, professional English.
- Use headings and bullet points.
- Keep the report concise and practical.

Return only the manager-facing analysis.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty analysis.")

    return response.text


# ==========================================
# 2.3. AI PLANNING ASSISTANT
# ==========================================
def generate_ai_planning_assistant(
    active_employees,
    week_preferences,
    shift_requirements,
    outcome,
    selected_week,
    failure_type="solver_infeasible"
):
    """
    Gemini analyzes an unsuccessful scheduling attempt.

    Only aggregated and anonymized information is sent.
    Gemini proposes scenarios but never changes data or constraints.
    """

    client = init_gemini()

    employees = active_employees.copy()
    preferences = week_preferences.copy()
    requirements = shift_requirements.copy()

    # --------------------------------------
    # Employee contract summary
    # --------------------------------------
    contract_distribution = {}

    if not employees.empty:
        for contract_days, group in employees.groupby(
            "work_days_per_week"
        ):
            contract_distribution[str(contract_days)] = int(len(group))

    total_employees = int(len(employees))

    total_contract_days = 0

    if not employees.empty:
        total_contract_days = int(
            pd.to_numeric(
                employees["work_days_per_week"],
                errors="coerce"
            ).fillna(0).sum()
        )

    # --------------------------------------
    # Preference summary
    # --------------------------------------
    preference_counts = {
        "Morning": 0,
        "Evening": 0,
        "Flexible": 0,
        "Unknown": 0
    }

    consent_counts = {
        "Yes": 0,
        "No": 0,
        "Unknown": 0
    }

    days_off_distribution = {}

    if not preferences.empty:

        for value, count in preferences[
            "preferred_shift"
        ].fillna("Unknown").value_counts().items():

            key = str(value)

            if key in preference_counts:
                preference_counts[key] += int(count)
            else:
                preference_counts["Unknown"] += int(count)

        for value, count in preferences[
            "long_shift_consent"
        ].fillna("Unknown").value_counts().items():

            key = str(value)

            if key in consent_counts:
                consent_counts[key] += int(count)
            else:
                consent_counts["Unknown"] += int(count)

        for days in preferences["required_days_off"]:
            if isinstance(days, list):
                for day in days:
                    day_name = str(day)
                    days_off_distribution[day_name] = (
                        days_off_distribution.get(day_name, 0) + 1
                    )

    # --------------------------------------
    # Shift demand summary
    # --------------------------------------
    shift_demand = {
        "Morning": 0,
        "Evening": 0,
        "Long": 0
    }

    daily_positions = 0
    requirement_summary = []

    if not requirements.empty:

        req = requirements.copy()

        if "is_active" in req.columns:
            req = req[req["is_active"] == True].copy()

        if not req.empty:

            for shift in SHIFT_ORDER:
                values = pd.to_numeric(
                    req.loc[
                        req["shift_type"] == shift,
                        "required_employees"
                    ],
                    errors="coerce"
                ).fillna(0)

                shift_demand[shift] = int(values.sum())

            daily_positions = int(
                sum(shift_demand.values())
            )

            for _, row in req.iterrows():
                requirement_summary.append({
                    "shift_type": str(row["shift_type"]),
                    "required_employees_per_day": int(
                        row["required_employees"]
                    ),
                    "start_time": str(row["start_time"]),
                    "end_time": str(row["end_time"])
                })

    weekly_positions = daily_positions * 7

    # --------------------------------------
    # Failure classification
    # --------------------------------------
    failure_messages = {
        "solver_infeasible": (
            "The mathematical optimizer did not find a feasible solution."
        ),
        "weekly_rest_check_failed": (
            "The optimizer produced an allocation, but the independent "
            "35-hour weekly-rest check did not pass."
        ),
        "execution_error": (
            "The scheduling process stopped because of an execution error. "
            "The exact technical error is not shared with Gemini."
        )
    }

    failure_description = failure_messages.get(
        failure_type,
        "The scheduling process did not produce an approved schedule."
    )

    safe_outcome = {
        "feasible": bool(outcome.get("feasible", False)),
        "assigned_positions": int(
            outcome.get("assigned_positions", 0)
        ),
        "unassigned_positions": int(
            outcome.get("unassigned_positions", 0)
        ),
        "failure_category": failure_type,
        "failure_description": failure_description
    }

    # --------------------------------------
    # Safe anonymized payload
    # --------------------------------------
    payload = {
        "week_start_date": selected_week.isoformat(),
        "employees": {
            "total_active_employees": total_employees,
            "contract_workdays_distribution": contract_distribution,
            "total_contract_workdays_per_week": total_contract_days
        },
        "preferences": {
            "preferred_shift_counts": preference_counts,
            "long_shift_consent_counts": consent_counts,
            "days_off_frequency_by_weekday": days_off_distribution
        },
        "shift_requirements": {
            "employees_required_per_shift_type_per_day": shift_demand,
            "total_positions_per_day": daily_positions,
            "total_positions_per_week": weekly_positions,
            "requirements": requirement_summary
        },
        "scheduling_result": safe_outcome
    }

    prompt = f"""
You are the AI Planning Assistant of an employee shift-planning system.

Your task is to help a manager understand why a scheduling attempt
was unsuccessful and identify safe options for further investigation.

SCHEDULING DATA:
{json.dumps(payload, ensure_ascii=False, indent=2)}

YOUR TASKS:

1. Explain the scheduling result in understandable language.
2. Identify possible causes supported by the supplied data.
3. Compare total weekly contract workdays with total weekly shift positions.
4. Review the aggregated Long Shift consent information.
5. Review the aggregated shift preferences and requested days off.
6. Suggest up to three practical scenarios for the manager.
7. For every scenario, explain:
   - what the manager could investigate or request;
   - which information or input may need to be updated;
   - what must be checked before another scheduling attempt.
8. Distinguish confirmed facts from possible explanations.
9. If the supplied data is insufficient to identify the cause,
   say so clearly and recommend what should be checked next.

IMPORTANT SYSTEM RULES:

- Do not invent facts, employees, staffing shortages or availability.
- Do not identify individual employees.
- Do not request or infer personal information.
- Do not recommend changing or ignoring hard constraints.
- Never suggest violating contract workdays or mandatory days off.
- Never suggest assigning a Long Shift without explicit Yes consent.
- Never suggest violating the 11-hour daily rest or 35-hour weekly rest rules.
- Employee shift preferences are soft preferences and may be reviewed
  only through a voluntary, updated employee submission.
- Mandatory days off are not soft preferences. They must not be silently
  changed by the manager or by the AI.
- Do not directly change employee data, preferences, shift requirements,
  database records or schedules.
- Do not generate a final employee-to-shift assignment.
- Do not claim that a proposed scenario is feasible before the
  mathematical optimizer has checked it.
- If capacity appears insufficient, recommend that the manager verify
  staffing requirements and available staffing. Do not assume that
  hiring or changing contracts is automatically necessary.
- The manager makes the final decision.
- Write in professional, clear English.
- Use headings and bullet points.
- Keep recommendations practical and concise.

End with a short section titled "Next Manager Decision".
Return only the manager-facing report.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty planning report.")

    return response.text




# ==========================================
# 2.4. AI DECISION SUPPORT
# ==========================================
def calculate_decision_metrics(
    run_id,
    assignments,
    active_employees,
    week_preferences,
    shift_requirements
):
    """
    Calculate transparent, deterministic metrics for one saved schedule.
    Only aggregated metrics are passed to Gemini; employee identifiers
    are used locally for matching and are never included in the AI payload.
    """

    schedule = assignments.copy()
    employees = active_employees.copy()
    preferences = week_preferences.copy()
    requirements = shift_requirements.copy()

    if schedule.empty:
        raise ValueError(f"Schedule run {run_id} contains no assignments.")

    required_columns = {
        "employee_id", "work_date", "shift_type",
        "start_time", "end_time", "break_minutes"
    }
    missing_columns = required_columns - set(schedule.columns)
    if missing_columns:
        raise ValueError(
            "Schedule assignments are missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    schedule["work_date"] = pd.to_datetime(
        schedule["work_date"], errors="coerce"
    ).dt.date

    if schedule["work_date"].isna().any():
        raise ValueError(
            f"Schedule run {run_id} contains invalid work dates."
        )

    schedule["employee_id"] = schedule["employee_id"].astype(str)
    employees["employee_id"] = employees["employee_id"].astype(str)
    preferences["employee_id"] = preferences["employee_id"].astype(str)

    if "is_active" in requirements.columns:
        requirements = requirements[
            requirements["is_active"] == True
        ].copy()

    expected_positions = 0
    if not requirements.empty:
        expected_positions = int(
            pd.to_numeric(
                requirements["required_employees"], errors="coerce"
            ).fillna(0).sum() * 7
        )

    assigned_positions = int(len(schedule))
    coverage_pct = (
        round(assigned_positions / expected_positions * 100, 1)
        if expected_positions else None
    )

    # Calculate net working hours from shift duration minus break time.
    def minutes_from_value(value):
        if isinstance(value, time):
            return value.hour * 60 + value.minute
        parts = str(value).split(":")
        return int(parts[0]) * 60 + int(parts[1])

    schedule["net_hours"] = schedule.apply(
        lambda row: (
            (
                (minutes_from_value(row["end_time"])
                 - minutes_from_value(row["start_time"])) % 1440
            ) - float(row["break_minutes"] or 0)
        ) / 60,
        axis=1
    )

    hours_by_shift = {}
    shift_counts = {}
    for shift in SHIFT_ORDER:
        subset = schedule[schedule["shift_type"] == shift]
        shift_counts[shift] = int(len(subset))
        hours_by_shift[shift] = round(
            float(subset["net_hours"].sum()), 1
        )

    total_net_hours = round(float(schedule["net_hours"].sum()), 1)

    # Match each employee's actual assignments with contract and preferences.
    pref_by_employee = {
        row["employee_id"]: row
        for _, row in preferences.iterrows()
    }
    employee_by_id = {
        row["employee_id"]: row
        for _, row in employees.iterrows()
    }

    contract_checked = 0
    contract_compliant = 0
    days_off_checked = 0
    days_off_compliant = 0
    long_consent_violations = 0
    multiple_long_violations = 0
    duplicate_day_violations = int(
        schedule.duplicated(["employee_id", "work_date"]).sum()
    )

    for employee_id, employee in employee_by_id.items():
        employee_rows = schedule[
            schedule["employee_id"] == employee_id
        ]
        worked_days = set(employee_rows["work_date"].tolist())

        contract_checked += 1
        expected_workdays = int(employee["work_days_per_week"])
        if len(worked_days) == expected_workdays:
            contract_compliant += 1

        pref = pref_by_employee.get(employee_id)
        if pref is not None:
            required_days_off = pref.get("required_days_off")
            if isinstance(required_days_off, list):
                days_off_checked += 1
                weekdays = {
                    normalize_weekday(day_name)
                    for day_name in required_days_off
                    if normalize_weekday(day_name) is not None
                }
                actual_worked_weekdays = {
                    day.strftime("%A") for day in worked_days
                }
                if not (weekdays & actual_worked_weekdays):
                    days_off_compliant += 1

            long_count = int(
                (employee_rows["shift_type"] == "Long").sum()
            )
            if long_count and pref.get("long_shift_consent") != "Yes":
                long_consent_violations += long_count
            if long_count > 1:
                multiple_long_violations += 1

    # Preference match is measured only where a clear Morning/Evening
    # preference exists. Flexible and Long assignments are excluded.
    preference_eligible = 0
    preference_matches = 0
    for _, row in schedule.iterrows():
        pref = pref_by_employee.get(row["employee_id"])
        if pref is None:
            continue
        preferred = pref.get("preferred_shift")
        actual = row["shift_type"]
        if preferred in ("Morning", "Evening") and actual in (
            "Morning", "Evening"
        ):
            preference_eligible += 1
            if actual == preferred:
                preference_matches += 1

    preference_match_pct = (
        round(preference_matches / preference_eligible * 100, 1)
        if preference_eligible else None
    )

    return {
        "variant": f"Běh {int(run_id)}",
        "run_id": int(run_id),
        "assigned_positions": assigned_positions,
        "expected_positions": expected_positions,
        "coverage_percent": coverage_pct,
        "shift_counts": shift_counts,
        "net_hours_by_shift": hours_by_shift,
        "total_net_hours": total_net_hours,
        "contract_compliance": {
            "employees_meeting_exact_workdays": contract_compliant,
            "employees_checked": contract_checked
        },
        "requested_days_off_compliance": {
            "employees_respecting_requested_days_off": days_off_compliant,
            "employees_checked": days_off_checked
        },
        "preference_match": {
            "matched_assignments": preference_matches,
            "eligible_assignments": preference_eligible,
            "match_percent": preference_match_pct
        },
        "constraint_checks": {
            "long_without_explicit_consent": long_consent_violations,
            "employees_with_more_than_one_long_shift": multiple_long_violations,
            "duplicate_employee_day_assignments": duplicate_day_violations
        }
    }




def compare_schedule_assignments(run_a_id, assignments_a, run_b_id, assignments_b):
    """
    Compare two saved schedules at the actual assignment level.
    Employee identifiers are used only locally and are never returned.
    """

    required = {
        "employee_id", "work_date", "branch_id", "shift_type",
        "start_time", "end_time", "break_minutes"
    }

    for run_id, frame in (
        (run_a_id, assignments_a),
        (run_b_id, assignments_b)
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"Run {run_id} is missing columns required for assignment "
                "comparison: " + ", ".join(sorted(missing))
            )
        if frame.empty:
            raise ValueError(
                f"Run {run_id} contains no assignments to compare."
            )

    def prepare(frame):
        result = frame.copy()
        result["employee_id"] = result["employee_id"].astype(str)
        result["work_date"] = pd.to_datetime(
            result["work_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")

        if result["work_date"].isna().any():
            raise ValueError(
                "A schedule contains an invalid work date."
            )

        for column in (
            "branch_id", "shift_type", "start_time",
            "end_time", "break_minutes"
        ):
            result[column] = result[column].fillna("").astype(str)

        return result

    a = prepare(assignments_a)
    b = prepare(assignments_b)

    slot_columns = [
        "work_date", "branch_id", "shift_type",
        "start_time", "end_time", "break_minutes"
    ]

    def grouped_assignments(frame):
        grouped = {}
        for slot_key, group in frame.groupby(
            slot_columns, dropna=False, sort=False
        ):
            if not isinstance(slot_key, tuple):
                slot_key = (slot_key,)
            grouped[slot_key] = set(group["employee_id"].tolist())
        return grouped

    slots_a = grouped_assignments(a)
    slots_b = grouped_assignments(b)

    keys_a = set(slots_a)
    keys_b = set(slots_b)
    common_keys = keys_a & keys_b

    changed_slots = 0
    changed_employee_entries = 0

    for key in common_keys:
        employees_a = slots_a[key]
        employees_b = slots_b[key]
        if employees_a != employees_b:
            changed_slots += 1
            changed_employee_entries += len(
                employees_a.symmetric_difference(employees_b)
            )

    # Count how many employees have a different set of assigned shifts.
    def employee_signatures(frame):
        signatures = {}
        for _, row in frame.iterrows():
            employee_id = row["employee_id"]
            slot = tuple(row[column] for column in slot_columns)
            signatures.setdefault(employee_id, set()).add(slot)
        return signatures

    signatures_a = employee_signatures(a)
    signatures_b = employee_signatures(b)
    all_employee_ids = set(signatures_a) | set(signatures_b)
    employees_with_changed_schedules = sum(
        signatures_a.get(employee_id, set())
        != signatures_b.get(employee_id, set())
        for employee_id in all_employee_ids
    )

    # Compare staffing counts for each date / branch / shift combination.
    bucket_columns = ["work_date", "branch_id", "shift_type"]
    counts_a = a.groupby(bucket_columns).size().to_dict()
    counts_b = b.groupby(bucket_columns).size().to_dict()
    all_buckets = set(counts_a) | set(counts_b)
    buckets_with_different_counts = sum(
        counts_a.get(bucket, 0) != counts_b.get(bucket, 0)
        for bucket in all_buckets
    )

    return {
        "comparison": f"Běh {int(run_a_id)} vs. běh {int(run_b_id)}",
        "run_a_id": int(run_a_id),
        "run_b_id": int(run_b_id),
        "total_assignments_run_a": int(len(a)),
        "total_assignments_run_b": int(len(b)),
        "common_shift_slots": int(len(common_keys)),
        "slots_with_different_employee_assignments": int(changed_slots),
        "employee_to_slot_entries_changed": int(
            changed_employee_entries
        ),
        "employees_with_changed_weekly_schedules": int(
            employees_with_changed_schedules
        ),
        "date_branch_shift_buckets_with_different_staffing": int(
            buckets_with_different_counts
        ),
        "slot_groups_only_in_run_a": int(len(keys_a - keys_b)),
        "slot_groups_only_in_run_b": int(len(keys_b - keys_a))
    }


def generate_ai_decision_support(metrics, selected_week, assignment_comparisons=None):
    """
    Gemini compares aggregated metrics from up to five feasible schedule runs.
    It advises the manager but never selects, edits or approves a schedule.
    """

    client = init_gemini()

    safe_payload = {
        "week_start_date": selected_week.isoformat(),
        "number_of_variants": len(metrics),
        "comparison_scope_note": (
            "Modul podpory rozhodování samostatně nepřepočítává denní ani "
            "týdenní odpočinek. Tyto kontroly provádí plánovací algoritmus "
            "a následná validace po výpočtu."
        ),
        "variants": metrics,
        "actual_assignment_comparisons": assignment_comparisons or []
    }

    prompt = f"""
Jsi modul podpory rozhodování s umělou inteligencí v systému plánování směn.

Porovnej uložené proveditelné varianty rozvrhu podle souhrnných ukazatelů
 a anonymizovaných výsledků porovnání skutečného přiřazení směn uvedených níže.

DATA PRO POROVNÁNÍ ROZVRHŮ:
{json.dumps(safe_payload, ensure_ascii=False, indent=2)}

ÚKOLY:
1. Porovnej varianty pomocí stejných ukazatelů a vysvětli podstatné rozdíly.
2. Zhodnoť pokrytí směn, čistý počet pracovních hodin, rozdělení směn,
   dodržení smluvního počtu pracovních dnů a požadovaných dnů volna.
3. Zhodnoť procento splnění preferencí a uveď počet způsobilých přiřazení,
   který byl použit jako základ výpočtu.
4. Upozorni na zaznamenané směny Long bez souhlasu, opakované směny Long
   nebo duplicitní přiřazení zaměstnance v jednom dni.
5. Urči nejsilnější a nejslabší stránky každé varianty.
6. Na základě dodaných údajů doporuč, která varianta se jeví jako nejlepší,
   a vysvětli související kompromisy.
7. Pokud jsou ukazatele shodné nebo neúplné, otevřeně to uveď a nevytvářej
   umělé pořadí variant.
8. Zhodnoť skutečné rozdíly v přiřazení mezi každou dvojicí běhů:
   změněná směnová místa, změněná přiřazení zaměstnanců ke směnám,
   zaměstnance se změněným týdenním rozvrhem, rozdíly v počtu pracovníků
   podle data, pobočky a směny a skupiny směn přítomné pouze v jedné variantě.
9. Zakonči zprávu krátkou částí s názvem „Další rozhodnutí manažera“.

DŮLEŽITÁ PRAVIDLA:
- Používej pouze dodané ukazatele. Nevymýšlej si fakta ani příčiny.
- Označení běhu jako proveditelného samo o sobě nedokazuje splnění všech
  možných právních nebo provozních požadavků.
- Netvrď, že byla kontrolována metrika, která v datech chybí.
- Neodvozuj totožnost zaměstnanců ani nepopisuj jednotlivé zaměstnance.
- Údaje o rozdílech v přiřazení jsou pouze souhrnné počty. Vysvětli,
  co měří, ale nevymýšlej, kteří zaměstnanci se změnili ani proč.
- Nedoporučuj porušení smluv, požadovaných dnů volna, souhlasu,
  denního odpočinku ani týdenního odpočinku.
- Neměň, nevytvářej, neschvaluj ani nezveřejňuj žádný rozvrh.
- Netvrď, že Gemini je matematický optimalizátor.
- Konečné rozhodnutí vždy provádí manažer.
- Jasně odděluj naměřené výsledky od doporučení.
- Celý výstup napiš výhradně česky, spisovným a profesionálním jazykem.
  Používej české nadpisy a stručné odrážky. Nepoužívej angličtinu.

Vrať pouze srovnávací zprávu pro manažera v češtině.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    if not response.text:
        raise RuntimeError("Gemini nevrátil žádnou zprávu podpory rozhodování.")

    return response.text


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
            str(old["employee_id"])
            if old.get("employee_id") is not None else None,
            str(old["issue_type"])
        )

        if old_key not in current_issue_keys:
            supabase.table("validation_issues").update(
                {"resolution_status": "Resolved"}
            ).eq(
                "issue_id", old["issue_id"]
            ).eq(
                "resolution_status", "Open"
            ).execute()

            resolved_count += 1

    return {
        "saved_count": len(records),
        "resolved_count": resolved_count
    }


# ==========================================
# WEEKDAY NORMALIZATION
# ==========================================
WEEKDAY_ALIASES = {
    "monday": "Monday", "pondeli": "Monday",
    "tuesday": "Tuesday", "utery": "Tuesday",
    "wednesday": "Wednesday", "streda": "Wednesday",
    "thursday": "Thursday", "ctvrtek": "Thursday",
    "friday": "Friday", "patek": "Friday",
    "saturday": "Saturday", "sobota": "Saturday",
    "sunday": "Sunday", "nedele": "Sunday",
}

def normalize_weekday(value):
    """Return a canonical English weekday for stored English/Czech values."""
    if value is None:
        return None
    normalized = unicodedata.normalize("NFKD", str(value).strip().casefold())
    normalized = "".join(
        character for character in normalized
        if not unicodedata.combining(character)
    )
    return WEEKDAY_ALIASES.get(normalized)


# ==========================================
# 4. VALIDATION ENGINE
# ==========================================
def validate_week_data(active_employees, week_preferences):
    issues = []

    # Database values are stored in a canonical language-independent format.
    weekdays = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
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
                    "description": (
                        f"Contract requires {days_off} days off, "
                        f"but employee selected {len(selected_days)}."
                    ),
                    "recommended_action": (
                        "Ask the employee to select the correct number of days."
                    )
                })

            if len(selected_days) != len(set(map(str, selected_days))):
                issues.append({
                    "employee_id": employee_id,
                    "issue_type": "DUPLICATE_DAYS",
                    "severity": "Critical",
                    "description": "The same day was selected more than once.",
                    "recommended_action": "Correct the selected days off."
                })

            normalized_days = [normalize_weekday(day) for day in selected_days]
            invalid_days = [
                day for day, normalized in zip(selected_days, normalized_days)
                if normalized not in weekdays
            ]

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
                "recommended_action": (
                    "Ask the employee to select a valid shift preference."
                )
            })

        consent = submission.get("long_shift_consent")

        if consent not in valid_consent:
            issues.append({
                "employee_id": employee_id,
                "issue_type": "INVALID_LONG_SHIFT_CONSENT",
                "severity": "Critical",
                "description": "Long Shift consent is missing or invalid.",
                "recommended_action": (
                    "Ask the employee to clarify their Long Shift consent."
                )
            })

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
    apply_light_theme()
    # The application interface is intentionally Czech-only.
    cz = True

    st.title("📅 Formulář dostupnosti zaměstnance" if cz else "📅 Employee Availability Form")

    st.write(
        "Vyplňte prosím svou dostupnost pro nadcházející pracovní týden."
        if cz else
        "Please submit your availability for the upcoming working week."
    )

    try:
        employees = load_table("employees")

        if employees.empty:
            st.warning("Nebyli nalezeni žádní zaměstnanci." if cz else "No employees found.")
            return

        active = employees[
            employees["employment_status"] == "Active"
        ].copy()

        if active.empty:
            st.warning("Nebyli nalezeni žádní aktivní zaměstnanci." if cz else "No active employees found.")
            return

        active = active.sort_values("employee_id")

        employee_id = st.selectbox(
            ("Vaše ID zaměstnance" if cz else "Your Employee ID"),
            active["employee_id"].tolist()
        )

        employee = active[
            active["employee_id"] == employee_id
        ].iloc[0]

        days_off_count = int(employee["days_off_per_week"])
        work_days = int(employee["work_days_per_week"])

        st.info(
            (
                f"Vaše smlouva: {work_days} pracovních dnů "
                f"a {days_off_count} dnů volna týdně."
            ) if cz else (
                f"Your contract: {work_days} working days "
                f"and {days_off_count} days off per week."
            )
        )

        st.divider()

        week_start = st.date_input(
            ("Týden začínající v pondělí" if cz else "Week starting on (Monday)"),
            value=get_monday()
        )

        prague_tz = ZoneInfo("Europe/Prague")
        now_local = datetime.now(prague_tz)
        current_monday = now_local.date() - timedelta(days=now_local.weekday())
        submission_deadline = datetime.combine(
            week_start - timedelta(days=3),
            time(12, 0),
            tzinfo=prague_tz
        )
        form_locked = (
            week_start < current_monday
            or now_local >= submission_deadline
        )

        if form_locked:
            st.warning(
                "Termín odevzdání uplynul. Změny pro tento týden jsou uzamčeny."
                if cz else
                "The submission deadline has passed. Changes for this week are locked."
            )
        else:
            st.info(
                (
                    f"Termín odevzdání: pátek 12:00 ({submission_deadline:%d.%m.%Y %H:%M})."
                    if cz else
                    f"Submission deadline: Friday 12:00 ({submission_deadline:%d %b %Y, %H:%M})."
                )
            )

        weekdays = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        ]
        weekday_labels = {
            "Monday": "Pondělí", "Tuesday": "Úterý",
            "Wednesday": "Středa", "Thursday": "Čtvrtek",
            "Friday": "Pátek", "Saturday": "Sobota",
            "Sunday": "Neděle"
        }

        required_days = st.multiselect(
            (f"Vyberte přesně {days_off_count} dnů volna" if cz else f"Choose exactly {days_off_count} days off"),
            weekdays,
            format_func=lambda day: weekday_labels[day] if cz else day,
            disabled=form_locked
        )

        st.caption(
            (f"Vybráno: {len(required_days)} z {days_off_count} dnů." if cz else f"Selected: {len(required_days)} of {days_off_count} days.")
        )

        preferred_shift = st.selectbox(
            ("Preferovaná směna" if cz else "Preferred shift"),
            ["Morning", "Evening", "Flexible"],
            format_func=lambda value: (
                {"Morning": "Ranní", "Evening": "Večerní", "Flexible": "Flexibilní"}[value]
                if cz else value
            ),
            disabled=form_locked
        )

        consent = st.selectbox(
            ("Souhlasíte s dlouhou směnou (09:00–21:00)?" if cz else "Are you willing to work a Long Shift (09:00–21:00)?"),
            ["Unknown", "Yes", "No"],
            format_func=lambda value: (
                {"Unknown": "Nevyplněno", "Yes": "Ano", "No": "Ne"}[value]
                if cz else value
            ),
            disabled=form_locked
        )

        comment = st.text_area("Další komentář (nepovinné)" if cz else "Additional comment (optional)", disabled=form_locked)

        if st.button("Odeslat dostupnost" if cz else "Submit availability", type="primary", disabled=form_locked):

            if week_start.weekday() != 0:
                st.error("Vybraný týden musí začínat v pondělí." if cz else "The selected week must start on Monday.")

            elif len(required_days) != days_off_count:
                st.error(
                    (f"Vyberte přesně {days_off_count} dnů volna." if cz else f"Please select exactly {days_off_count} days off.")
                )

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
                        payload,
                        on_conflict="employee_id,week_start_date"
                    ).execute()

                    st.success(
                        "Vaše dostupnost byla úspěšně odeslána!" if cz else "Your availability was submitted successfully!"
                    )

                except Exception as e:
                    st.error("Odeslání se nepodařilo uložit." if cz else "Could not save your submission.")
                    st.exception(e)

    except Exception as e:
        st.error("Nepodařilo se načíst údaje zaměstnance." if cz else "Could not load employee data.")
        st.exception(e)


# ==========================================
# 6. SCHEDULING ALGORITHM
# ==========================================
SHIFT_ORDER = ["Morning", "Evening", "Long"]
PREFERENCE_PENALTY = 10.0
MIN_DAILY_REST_HOURS = 11.0
MIN_WEEKLY_REST_HOURS = 35.0


def time_to_minutes(value):
    """Convert a database time value to minutes after midnight."""

    if value is None:
        raise ValueError(
            "A shift requirement is missing a start or end time."
        )

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

    needed = {
        "branch_id", "shift_type", "start_time", "end_time",
        "required_employees", "break_minutes"
    }

    missing = needed - set(req.columns)

    if missing:
        raise ValueError(
            f"shift_requirements is missing columns: {sorted(missing)}"
        )

    req["required_employees"] = pd.to_numeric(
        req["required_employees"], errors="coerce"
    )

    req["break_minutes"] = pd.to_numeric(
        req["break_minutes"], errors="coerce"
    )

    if req[["required_employees", "break_minutes"]].isna().any().any():
        raise ValueError(
            "Staffing or break values contain empty/non-numeric values."
        )

    if (req["required_employees"] < 1).any():
        raise ValueError(
            "Every active shift requirement must require at least one employee."
        )

    if not set(req["shift_type"]).issubset(set(SHIFT_ORDER)):
        raise ValueError(
            "A shift requirement contains an unsupported shift_type."
        )

    branch_ids = set(branches["branch_id"].astype(str))

    if not set(req["branch_id"].astype(str)).issubset(branch_ids):
        raise ValueError(
            "A shift requirement refers to a branch that was not loaded."
        )

    return req


def build_solver_constraints(
    employees, preferences, demand, shift_times
):
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

    pref_by_id = {
        r["employee_id"]: r for _, r in preferences.iterrows()
    }

    # Exact contracted working days.
    for e, (_, emp) in enumerate(employees.iterrows()):
        coefficients = {}

        for d in range(7):
            for s in range(n_shifts):
                coefficients[idx(e, d, s)] = 1

        add_constraint(
            coefficients,
            int(emp["work_days_per_week"]),
            int(emp["work_days_per_week"])
        )

    # Exact staffing demand for each shift and day.
    for d in range(7):
        for s, shift in enumerate(SHIFT_ORDER):
            coefficients = {
                idx(e, d, s): 1 for e in range(n_emp)
            }

            add_constraint(
                coefficients,
                demand[shift],
                demand[shift]
            )

    # Maximum one shift per employee per day.
    for e in range(n_emp):
        for d in range(7):
            coefficients = {
                idx(e, d, s): 1 for s in range(n_shifts)
            }

            add_constraint(coefficients, 0, 1)

    # Exact mandatory days off.
    weekdays = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]

    for e, (_, emp) in enumerate(employees.iterrows()):
        pref = pref_by_id[emp["employee_id"]]
        days_off = {
            normalize_weekday(day_name)
            for day_name in pref["required_days_off"]
        }

        for d, day_name in enumerate(weekdays):
            if day_name in days_off:
                coefficients = {
                    idx(e, d, s): 1 for s in range(n_shifts)
                }

                add_constraint(coefficients, 0, 0)

    # Long shifts require explicit consent.
    # Maximum one Long shift per employee per week.
    for e, (_, emp) in enumerate(employees.iterrows()):
        pref = pref_by_id[emp["employee_id"]]

        long_var_indices = [
            idx(e, d, 2) for d in range(7)
        ]

        if pref["long_shift_consent"] != "Yes":
            for var in long_var_indices:
                add_constraint({var: 1}, 0, 0)
        else:
            add_constraint(
                {var: 1 for var in long_var_indices},
                0, 1
            )

    # Minimum 11 hours between shifts, including week wrap.
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

                        rest_hours = (
                            delta * 24
                            + start2 / 60
                            - end1 / 60
                        )

                        if rest_hours < MIN_DAILY_REST_HOURS:
                            add_constraint({
                                idx(e, d1, s1): 1,
                                idx(e, d2, s2): 1
                            }, 0, 1)

    matrix = lil_matrix(
        (len(rows), n_vars),
        dtype=float
    )

    for r, coefficients in enumerate(rows):
        for c, value in coefficients.items():
            matrix[r, c] = value

    lower_bounds = np.zeros(n_vars)
    upper_bounds = np.ones(n_vars)
    integrality = np.ones(n_vars, dtype=int)

    # Preference deviations are soft constraints.
    objective = np.zeros(n_vars)

    for e, (_, emp) in enumerate(employees.iterrows()):
        pref = pref_by_id[emp["employee_id"]]["preferred_shift"]

        for d in range(7):
            for s, shift in enumerate(SHIFT_ORDER):

                if shift == "Long":
                    objective[idx(e, d, s)] = 0.25

                elif (
                    pref in ("Morning", "Evening")
                    and pref != shift
                ):
                    objective[idx(e, d, s)] = PREFERENCE_PENALTY

    return (
        objective,
        integrality,
        Bounds(lower_bounds, upper_bounds),
        LinearConstraint(
            matrix.tocsr(),
            np.array(lows),
            np.array(highs)
        ),
        idx
    )


def calculate_weekly_rest(assignments, employees):
    """Check each employee has at least one cyclic rest interval of 35 hours."""

    problems = []

    for employee_id in employees["employee_id"]:
        rows = assignments[
            assignments["employee_id"] == employee_id
        ]

        if rows.empty:
            problems.append(
                f"{employee_id}: no shifts assigned"
            )
            continue

        intervals = []

        for _, row in rows.iterrows():
            day_offset = (
                row["work_date"] - assignments["work_date"].min()
            ).days

            start = (
                time_to_minutes(row["start_time"]) / 60
                + day_offset * 24
            )

            end = (
                time_to_minutes(row["end_time"]) / 60
                + day_offset * 24
            )

            if end <= start:
                end += 24

            intervals.append((start, end))

        intervals.sort()

        gaps = []

        for i in range(len(intervals) - 1):
            gaps.append(
                intervals[i + 1][0] - intervals[i][1]
            )

        gaps.append(
            intervals[0][0] + 168 - intervals[-1][1]
        )

        if max(gaps) < MIN_WEEKLY_REST_HOURS:
            problems.append(
                f"{employee_id}: longest weekly rest is "
                f"{max(gaps):.1f} hours"
            )

    return problems


def find_long_shift_consent_violations(assignments, preferences):
    """Return assignments that violate the explicit Long Shift consent rule.

    This is a deterministic check over generated assignments. It does not
    modify data or call Gemini.
    """
    required_assignment_columns = {"employee_id", "shift_type"}
    required_preference_columns = {"employee_id", "long_shift_consent"}

    missing_assignments = required_assignment_columns - set(assignments.columns)
    missing_preferences = required_preference_columns - set(preferences.columns)

    if missing_assignments:
        raise ValueError(
            "Assignments are missing columns: "
            + ", ".join(sorted(missing_assignments))
        )
    if missing_preferences:
        raise ValueError(
            "Preferences are missing columns: "
            + ", ".join(sorted(missing_preferences))
        )

    consent_by_employee = (
        preferences.drop_duplicates("employee_id", keep="last")
        .set_index("employee_id")["long_shift_consent"]
        .to_dict()
    )

    violations = []
    long_assignments = assignments[
        assignments["shift_type"].astype(str).str.strip().eq("Long")
    ]

    for _, assignment in long_assignments.iterrows():
        employee_id = assignment["employee_id"]
        consent = consent_by_employee.get(employee_id)

        if consent != "Yes":
            violations.append({
                "employee_id": employee_id,
                "long_shift_consent": consent,
                "work_date": assignment.get("work_date"),
                "branch_id": assignment.get("branch_id"),
                "shift_type": "Long",
                "reason": (
                    "Long Shift requires explicit Yes consent; "
                    f"received {consent!r}."
                )
            })

    return violations


def run_long_shift_consent_self_test():
    """Run isolated positive/negative tests without touching Supabase."""
    test_assignments = pd.DataFrame([
        {"employee_id": "TEST_E01", "shift_type": "Long", "work_date": "2026-11-16"}
    ])

    no_consent = pd.DataFrame([
        {"employee_id": "TEST_E01", "long_shift_consent": "No"}
    ])
    yes_consent = pd.DataFrame([
        {"employee_id": "TEST_E01", "long_shift_consent": "Yes"}
    ])

    no_result = find_long_shift_consent_violations(
        test_assignments, no_consent
    )
    yes_result = find_long_shift_consent_violations(
        test_assignments, yes_consent
    )

    return {
        "no_consent_case_detected": len(no_result) == 1,
        "yes_consent_case_allowed": len(yes_result) == 0,
        "no_consent_details": no_result,
        "yes_consent_details": yes_result
    }


def run_scheduling_algorithm(
    active_employees,
    week_preferences,
    shift_requirements,
    branches,
    selected_week
):
    employees = (
        active_employees
        .sort_values("employee_id")
        .reset_index(drop=True)
        .copy()
    )

    preferences = week_preferences.copy()

    req = prepare_shift_requirements(
        shift_requirements, branches
    )

    if len(preferences) != len(employees):
        raise ValueError(
            "The number of weekly submissions does not match active employees."
        )

    # Total staffing demand by shift type.
    demand = {
        shift: 0 for shift in SHIFT_ORDER
    }

    for shift in SHIFT_ORDER:
        demand[shift] = int(
            req.loc[
                req["shift_type"] == shift,
                "required_employees"
            ].sum()
        )

    total_demand = sum(demand.values())

    total_contract_days = int(
        employees["work_days_per_week"].sum()
    )

    if total_demand * 7 != total_contract_days:
        raise ValueError(
            f"Capacity mismatch: shift requirements need "
            f"{total_demand * 7} employee-days, but contracts provide "
            f"{total_contract_days}."
        )

    # Consistent shift times and break duration.
    shift_times = {}
    shift_breaks = {}

    for shift in SHIFT_ORDER:
        subset = req[
            req["shift_type"] == shift
        ]

        if subset.empty:
            continue

        times = set(
            (str(r["start_time"]), str(r["end_time"]))
            for _, r in subset.iterrows()
        )

        if len(times) != 1:
            raise ValueError(
                f"{shift} shifts have inconsistent start/end times between branches."
            )

        first = subset.iloc[0]

        shift_times[shift] = (
            time_to_minutes(first["start_time"]),
            time_to_minutes(first["end_time"])
        )

        breaks = set(
            int(x) for x in subset["break_minutes"]
        )

        if len(breaks) != 1:
            raise ValueError(
                f"{shift} shifts have inconsistent break durations between branches."
            )

        shift_breaks[shift] = int(first["break_minutes"])

    for shift in SHIFT_ORDER:
        if demand[shift] > 0 and shift not in shift_times:
            raise ValueError(
                f"Missing time definition for {shift}."
            )

    # SciPy MILP optimization.
    objective, integrality, bounds, constraints, idx = (
        build_solver_constraints(
            employees,
            preferences,
            demand,
            shift_times
        )
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
            "failure_type": "solver_infeasible",
            "message": (
                result.message
                or "The optimization solver did not find a feasible schedule."
            ),
            "assignments": pd.DataFrame(),
            "assigned_positions": 0,
            "unassigned_positions": total_demand * 7
        }

    # Convert solution into employee/day/shift assignments.
    planned = []

    for e, (_, emp) in enumerate(employees.iterrows()):
        for d in range(7):
            for s, shift in enumerate(SHIFT_ORDER):

                if result.x[idx(e, d, s)] > 0.5:
                    planned.append({
                        "employee_id": emp["employee_id"],
                        "work_date": (
                            selected_week + timedelta(days=d)
                        ),
                        "shift_type": shift
                    })

    planned_df = pd.DataFrame(planned)

    if len(planned_df) != total_demand * 7:
        return {
            "feasible": False,
            "failure_type": "solver_infeasible",
            "message": (
                "The solver result did not match the required "
                "number of weekly positions."
            ),
            "assignments": pd.DataFrame(),
            "assigned_positions": len(planned_df),
            "unassigned_positions": (
                total_demand * 7 - len(planned_df)
            )
        }

    # Assign employees to actual branches.
    branch_rows = []

    for d in range(7):
        work_date = selected_week + timedelta(days=d)

        for shift in SHIFT_ORDER:
            employees_for_shift = sorted(
                planned_df[
                    (planned_df["work_date"] == work_date)
                    & (planned_df["shift_type"] == shift)
                ]["employee_id"].tolist()
            )

            branch_demand = (
                req[req["shift_type"] == shift]
                .sort_values("branch_id")
            )

            position = 0

            for _, requirement in branch_demand.iterrows():
                for _ in range(
                    int(requirement["required_employees"])
                ):

                    if position >= len(employees_for_shift):
                        raise ValueError(
                            f"Not enough employees for {shift} on {work_date}."
                        )

                    branch_rows.append({
                        "employee_id": employees_for_shift[position],
                        "branch_id": requirement["branch_id"],
                        "work_date": work_date,
                        "shift_type": shift,
                        "start_time": str(requirement["start_time"]),
                        "end_time": str(requirement["end_time"]),
                        "break_minutes": int(
                            requirement["break_minutes"]
                        )
                    })

                    position += 1

            if position != len(employees_for_shift):
                raise ValueError(
                    f"Unassigned {shift} employees remain on {work_date}."
                )

    assignments = pd.DataFrame(branch_rows)

    # Independent post-solve checks.
    if assignments.duplicated(
        ["employee_id", "work_date"]
    ).any():
        raise ValueError(
            "Post-check failed: an employee has more than one shift on a day."
        )

    for employee_id, group in assignments.groupby("employee_id"):
        emp = employees[
            employees["employee_id"] == employee_id
        ].iloc[0]

        pref = preferences[
            preferences["employee_id"] == employee_id
        ].iloc[0]

        if len(group) != int(emp["work_days_per_week"]):
            raise ValueError(
                f"Post-check failed: incorrect number of workdays for {employee_id}."
            )

    long_consent_violations = find_long_shift_consent_violations(
        assignments, preferences
    )
    if long_consent_violations:
        first_violation = long_consent_violations[0]
        raise ValueError(
            "Post-check failed: Long Shift without explicit Yes consent "
            f"for {first_violation['employee_id']}."
        )

    weekly_rest_problems = calculate_weekly_rest(
        assignments, employees
    )

    if weekly_rest_problems:
        return {
            "feasible": False,
            "failure_type": "weekly_rest_check_failed",
            "message": (
                "The solver found a staffing allocation, but "
                "the 35-hour weekly-rest check failed: "
                + "; ".join(weekly_rest_problems[:8])
            ),
            "assignments": pd.DataFrame(),
            "assigned_positions": len(assignments),
            "unassigned_positions": 0
        }

    return {
        "feasible": True,
        "failure_type": None,
        "message": (
            "Optimal schedule found. All implemented hard constraints passed."
        ),
        "assignments": assignments,
        "assigned_positions": len(assignments),
        "unassigned_positions": 0
    }


# ==========================================
# 7. SCHEDULE RUN AND APPROVAL FUNCTIONS
# ==========================================
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
        raise RuntimeError(
            "Supabase did not return the created schedule run."
        )

    return response.data[0]["run_id"]


def update_schedule_run(run_id, status, assigned, unassigned):
    payload = {
        "run_status": status,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "assigned_positions": int(assigned),
        "unassigned_positions": int(unassigned)
    }

    supabase.table("schedule_runs").update(
        payload
    ).eq("run_id", run_id).execute()


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
        # Every schedule run is saved as immutable history.
        supabase.table("schedule_assignments").insert(
            records
        ).execute()


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
# 8. PAGE ROUTING
# ==========================================
page = st.query_params.get("page", "dashboard")

if page == "employee":
    employee_form()
    st.stop()


# ==========================================
# 9. MANAGER DASHBOARD
# ==========================================
st.title("📅 Plánovač směn s AI")
st.subheader("Manažerský přehled")

apply_light_theme()

# The manager interface is intentionally Czech-only.
is_czech = True

tab_labels = ["Přehled", "Zaměstnanci a dostupnost", "Správa směn", "AI přehledy a reporty"]

# ==========================================
# 9.1. GEMINI API CONNECTION TEST
# ==========================================
with st.expander("Test připojení Gemini AI"):
    st.write(
        "Ověřte, zda je rozhraní Gemini API připojeno."
    )

    if st.button("Otestovat připojení Gemini"):
        try:
            with st.spinner("Připojování ke Gemini..."):
                answer = test_gemini_connection()

            st.success(
                "Gemini API je úspěšně připojeno!"
            )

            st.write("Odpověď Gemini:")
            st.write(answer)

        except Exception as e:
            st.error("Připojení ke Gemini API se nezdařilo.")
            st.error(str(e))

# ==========================================
# 9.2. LOAD DATABASE TABLES
# ==========================================
try:
    employees = load_table("employees")
    branches = load_table("branches")
    shift_requirements = load_table("shift_requirements")
    company_rules = load_table("company_rules")

except Exception as e:
    st.error("Nepodařilo se načíst tabulky z databáze.")
    st.exception(e)
    st.stop()


st.divider()
st.header("Týdenní plánování směn" if is_czech else "Weekly Schedule Management")

selected_week = st.date_input(
    "Vyberte týden (pondělí)" if is_czech else "Select week (Monday)",
    value=get_monday(),
    key="manager_week"
)

if selected_week.weekday() != 0:
    st.warning("Vyberte prosím pondělí.")
    st.stop()


# ==========================================
# 9.3. LOAD WEEKLY PREFERENCES
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

weekly_view["required_days_off"] = (
    weekly_view["required_days_off"].apply(
        lambda x: ", ".join(map(str, x))
        if isinstance(x, list) else "—"
    )
)

for column in [
    "preferred_shift",
    "long_shift_consent",
    "comment",
    "submitted_at"
]:
    weekly_view[column] = weekly_view[column].fillna("—")


total_employees = len(active_employees)

received_count = int(
    (weekly_view["submission_status"] == "Received").sum()
)

missing_count = total_employees - received_count

dashboard_tab, employees_tab, schedule_tab, ai_tab = st.tabs(tab_labels)

with dashboard_tab:
    st.subheader("Přehled" if is_czech else "Weekly Overview")
    st.write(
        (f"Týden začínající: {selected_week}" if is_czech
         else f"Týden začínající: {selected_week}")
    )

    current_validation = st.session_state.get("validation_result")
    validation_matches_week = bool(
        current_validation
        and current_validation.get("week") == selected_week.isoformat()
    )
    validation_issue_count = (
        len(current_validation.get("issues", []))
        if validation_matches_week else None
    )
    status_col1, status_col2 = st.columns(2)
    if validation_matches_week and validation_issue_count == 0:
        status_col1.success("Připraveno k plánování" if is_czech else "Ready for scheduling")
    elif validation_matches_week:
        status_col1.warning(
            f"Počet problémů: {validation_issue_count}" if is_czech
            else f"Počet problémů při kontrole: {validation_issue_count}"
        )
    else:
        status_col1.info("Kontrola zatím nebyla spuštěna" if is_czech else "Validation not run yet")
    status_col2.metric(
        "Dokončení žádostí" if is_czech else "Submission completion",
        f"{received_count}/{total_employees}"
    )
    if total_employees:
        st.progress(received_count / total_employees)

    st.markdown(
        '<a href="?page=employee" target="_blank" '
        'style="display:inline-block;padding:10px 16px;border-radius:9px;'
        'background:#2563EB;color:white !important;text-decoration:none;font-weight:600;">'
        + ("Otevřít formulář dostupnosti zaměstnance" if is_czech else "Open Employee Availability Form")
        + '</a>',
        unsafe_allow_html=True
    )

    st.divider()
    st.header("Přehled odevzdání za týden" if is_czech else "Weekly Submission Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Aktivní zaměstnanci" if is_czech else "Active Employees", total_employees)
    c2.metric("Odevzdáno" if is_czech else "Received Submissions", received_count)
    c3.metric("Chybějící odevzdání" if is_czech else "Missing Submissions", missing_count)

    with st.expander(
        "Nastavení provozu" if is_czech else "Operational Settings",
        expanded=False
    ):
        st.caption(
            "Pobočky, požadavky na směny a firemní pravidla."
            if is_czech else
            "Branches, shift staffing requirements and company rules."
        )

        st.subheader("Pobočky" if is_czech else "Branches")

        st.dataframe(
            branches,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Požadavky na směny" if is_czech else "Shift Requirements")

        st.dataframe(
            shift_requirements,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Firemní pravidla" if is_czech else "Company Rules")

        st.dataframe(
            company_rules,
            use_container_width=True,
            hide_index=True
        )

with employees_tab:
    st.subheader("Zaměstnanci a dostupnost" if is_czech else "Employees & Availability")

    st.subheader("Zaměstnanci" if is_czech else "Employees")

    st.dataframe(
        employees,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Týdenní preference zaměstnanců" if is_czech else "Weekly Employee Preferences")

    st.write(f"Týden začínající: {selected_week}")

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

    st.subheader("Zaměstnanci bez odevzdání" if is_czech else "Employees with Missing Submissions")

    missing_employees = weekly_view[
        weekly_view["submission_status"] == "Missing"
    ]

    if missing_employees.empty:
        st.success(
            "Všichni zaměstnanci odevzdali své preference."
        )

    else:
        st.dataframe(
            missing_employees[
                [
                    "employee_id",
                    "work_days_per_week",
                    "days_off_per_week"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

with schedule_tab:
    st.subheader("Správa směn" if is_czech else "Schedule Management")

    st.subheader("Kontrola týdenních údajů" if is_czech else "Weekly Data Validation")

    st.write(
        "Kontrolní modul ověřuje odevzdané údaje zaměstnanců před vytvořením rozvrhu."
    )

    st.write(f"Týden začínající: {selected_week}")

    if st.button("Spustit kontrolu" if is_czech else "Run Validation", type="primary"):

        validation_issues = validate_week_data(
            active_employees,
            week_preferences
        )

        st.session_state["validation_result"] = {
            "week": selected_week.isoformat(),
            "issues": validation_issues
        }

        st.session_state.pop("ai_conflict_analysis", None)

        try:
            sync_result = save_validation_issues(
                validation_issues,
                selected_week
            )

            st.session_state["validation_save_status"] = {
                "week": selected_week.isoformat(),
                "saved_count": sync_result["saved_count"],
                "resolved_count": sync_result["resolved_count"],
                "error": None
            }

        except Exception as e:
            st.session_state["validation_save_status"] = {
                "week": selected_week.isoformat(),
                "saved_count": 0,
                "resolved_count": 0,
                "error": str(e)
            }

    result = st.session_state.get("validation_result")

    if result and result["week"] == selected_week.isoformat():

        issues = result["issues"]

        critical_count = sum(
            1 for x in issues
            if x["severity"] == "Critical"
        )

        warning_count = sum(
            1 for x in issues
            if x["severity"] == "Warning"
        )

        st.divider()

        a, b = st.columns(2)

        a.metric("Kritické problémy", critical_count)
        b.metric("Varování", warning_count)

        if critical_count:
            st.error(
                "BLOKOVÁNO — Plánování nelze spustit. "
                "Nejprve je nutné vyřešit kritické problémy."
            )

        elif warning_count:
            st.warning(
                "Před pokračováním zkontrolujte varování."
            )

        else:
            st.success(
                "PŘIPRAVENO — Všechna odevzdání zaměstnanců prošla kontrolou."
            )

        save_status = st.session_state.get(
            "validation_save_status"
        )

        if (
            save_status
            and save_status["week"] == selected_week.isoformat()
        ):

            if save_status["error"]:
                st.error(
                    "Kontrola byla dokončena, ale synchronizace problémů "
                    "v databázi Supabase se nezdařila."
                )

                st.code(save_status["error"])

            else:
                st.success(
                    f"Aktuálně uložené nebo aktualizované problémy: "
                    f"{save_status['saved_count']}. "
                    f"Starší problémy označené jako vyřešené: "
                    f"{save_status['resolved_count']}."
                )

        if issues:
            st.subheader("Zjištěné problémy" if is_czech else "Validation Issues")

            issues_df = pd.DataFrame(issues).copy()
            issue_column_names = {
                "employee_id": "ID zaměstnance",
                "issue_type": "Typ problému",
                "severity": "Závažnost",
                "description": "Popis",
                "recommended_action": "Doporučený postup",
                "resolution_status": "Stav řešení",
            }
            issues_df = issues_df.rename(columns=issue_column_names)
            if "Závažnost" in issues_df.columns:
                issues_df["Závažnost"] = issues_df["Závažnost"].replace({
                    "Critical": "Kritická", "Warning": "Varování"
                })
            if "Stav řešení" in issues_df.columns:
                issues_df["Stav řešení"] = issues_df["Stav řešení"].replace({
                    "Open": "Otevřeno", "Resolved": "Vyřešeno"
                })
            st.dataframe(
                issues_df,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.success("Nebyly nalezeny žádné problémy." if is_czech else "No validation issues found.")

    st.subheader("Algoritmus plánování" if is_czech else "Scheduling Algorithm")

    st.write(
        "Matematický optimalizátor přiřazuje zaměstnance ke směnám. "
        "Přiřazení zaměstnanců na jednotlivé pobočky následuje po rozdělení směn."
    )

    st.info(
        "Pevná pravidla: přesný počet pracovních dnů podle smlouvy a požadované volné dny, "
        "nejvýše jedna směna denně, výslovný souhlas s dlouhou směnou, "
        "minimálně 11 hodin denního odpočinku a kontrola 35hodinového týdenního odpočinku. "
        "Preference směn jsou zohledněny jako měkčí podmínky."
    )

    validation = st.session_state.get("validation_result")

    validation_ready = bool(
        validation
        and validation.get("week") == selected_week.isoformat()
        and len(validation.get("issues", [])) == 0
    )

    if validation_ready:
        st.success("Stav kontroly: PŘIPRAVENO")

    else:
        st.warning(
            "Plánování je uzamčeno. Spusťte kontrolu a získejte stav PŘIPRAVENO "
            "pro tento konkrétní týden."
        )

    # --------------------------------------
    # Generate weekly schedule
    # --------------------------------------
    if st.button(
        "Vygenerovat týdenní rozvrh",
        type="primary",
        disabled=not validation_ready
    ):

        run_id = None

        # Clear previous Planning Assistant results for this week.
        st.session_state.pop("ai_planning_assistant", None)

        try:
            run_id = create_schedule_run(selected_week)

            with st.spinner(
                "Optimalizační algoritmus právě vytváří rozvrh..."
            ):

                outcome = run_scheduling_algorithm(
                    active_employees,
                    week_preferences,
                    shift_requirements,
                    branches,
                    selected_week
                )

            if outcome["feasible"]:

                persist_schedule(
                    run_id,
                    outcome["assignments"]
                )

                update_schedule_run(
                    run_id,
                    "Feasible",
                    outcome["assigned_positions"],
                    outcome["unassigned_positions"]
                )

                st.session_state["last_schedule_run_id"] = int(run_id)

                st.session_state["last_schedule_attempt"] = {
                    "week": selected_week.isoformat(),
                    "status": "Feasible",
                    "failure_type": None,
                    "outcome": outcome
                }

                st.success(
                    f"Rozvrh byl úspěšně vytvořen. ID běhu: {run_id}"
                )

            else:
                update_schedule_run(
                    run_id,
                    "Infeasible",
                    outcome["assigned_positions"],
                    outcome["unassigned_positions"]
                )

                st.session_state["last_schedule_run_id"] = int(run_id)

                failure_type = outcome.get(
                    "failure_type",
                    "solver_infeasible"
                )

                st.session_state["last_schedule_attempt"] = {
                    "week": selected_week.isoformat(),
                    "status": "Infeasible",
                    "failure_type": failure_type,
                    "outcome": outcome
                }

                st.error("Nepodařilo se vytvořit platný rozvrh.")

            st.session_state["last_schedule_message"] = (
                outcome["message"]
            )

        except Exception as e:

            if run_id is not None:
                try:
                    update_schedule_run(
                        run_id, "Failed", 0, 0
                    )
                except Exception:
                    pass

            safe_failure_outcome = {
                "feasible": False,
                "message": "Spuštění plánování se nezdařilo.",
                "assigned_positions": 0,
                "unassigned_positions": 0
            }

            st.session_state["last_schedule_attempt"] = {
                "week": selected_week.isoformat(),
                "status": "Failed",
                "failure_type": "execution_error",
                "outcome": safe_failure_outcome
            }

            st.session_state["last_schedule_message"] = (
                "Spuštění plánování se nezdařilo. "
                "Před dalším pokusem zkontrolujte technickou chybu."
            )

            st.error(
                "Běh plánování selhal. "
                "Žádný rozvrh nebyl schválen ani zveřejněn."
            )

            st.exception(e)

    if st.session_state.get("last_schedule_message"):
        st.caption(
            st.session_state["last_schedule_message"]
        )

    # --------------------------------------
    # Load run history for selected week
    # --------------------------------------
    try:
        runs_response = (
            supabase.table("schedule_runs")
            .select("*")
            .eq("week_start_date", selected_week.isoformat())
            .order("run_id", desc=True)
            .execute()
        )

        runs = pd.DataFrame(
            runs_response.data or []
        )

    except Exception as e:
        runs = pd.DataFrame()

        st.error(
            "Nepodařilo se načíst historii plánování."
        )

        st.exception(e)

    if not runs.empty:

        st.divider()
        st.subheader("Historie plánování" if is_czech else "Schedule Run History")

        st.dataframe(
            runs,
            use_container_width=True,
            hide_index=True
        )

        feasible_runs = runs[
            runs["run_status"] == "Feasible"
        ]

        if not feasible_runs.empty:

            selected_run_id = st.selectbox(
                "Zobrazit vytvořený rozvrh (ID běhu)",
                feasible_runs["run_id"].astype(int).tolist(),
                key="view_schedule_run"
            )

            try:
                assignment_response = (
                    supabase.table("schedule_assignments")
                    .select("*")
                    .eq("run_id", int(selected_run_id))
                    .order("work_date")
                    .execute()
                )

                schedule_df = pd.DataFrame(
                    assignment_response.data or []
                )

                if not schedule_df.empty:

                    display_df = schedule_df.copy()

                    display_df["work_date"] = pd.to_datetime(
                        display_df["work_date"]
                    ).dt.date

                    st.subheader("Vygenerovaný týdenní rozvrh" if is_czech else "Generated Weekly Schedule")

                    st.dataframe(
                        display_df[
                            [
                                "work_date",
                                "branch_id",
                                "employee_id",
                                "shift_type",
                                "start_time",
                                "end_time",
                                "break_minutes"
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True
                    )

                    selected_run = feasible_runs[
                        feasible_runs["run_id"].astype(int)
                        == int(selected_run_id)
                    ].iloc[0]

                    if (
                        selected_run["manager_approval_status"]
                        == "Not_Reviewed"
                    ):

                        st.divider()
                        st.subheader("Schválení manažerem" if is_czech else "Manager Approval")

                        manager_comment = st.text_area(
                            "Komentář manažera (nepovinný)",
                            key=f"approval_comment_{selected_run_id}"
                        )

                        approve_col, reject_col = st.columns(2)

                        with approve_col:

                            if st.button(
                                "Schválit rozvrh",
                                type="primary",
                                key=f"approve_{selected_run_id}"
                            ):

                                try:
                                    approve_or_reject_run(
                                        selected_run_id,
                                        "Approved",
                                        manager_comment
                                    )

                                    st.success(
                                        "Rozvrh byl schválen. "
                                        "Rozhodnutí bylo uloženo."
                                    )

                                    st.rerun()

                                except Exception as e:
                                    st.error(
                                        "Nepodařilo se uložit schválení manažera."
                                    )
                                    st.exception(e)

                        with reject_col:

                            if st.button(
                                "Zamítnout rozvrh",
                                key=f"reject_{selected_run_id}"
                            ):

                                try:
                                    approve_or_reject_run(
                                        selected_run_id,
                                        "Rejected",
                                        manager_comment
                                    )

                                    st.warning(
                                        "Rozvrh byl zamítnut. "
                                        "Rozhodnutí bylo uloženo."
                                    )

                                    st.rerun()

                                except Exception as e:
                                    st.error(
                                        "Nepodařilo se uložit zamítnutí manažera."
                                    )
                                    st.exception(e)

                    else:
                        st.info(
                            "Rozhodnutí manažera: "
                            f"{selected_run['manager_approval_status']}"
                        )

            except Exception as e:
                st.error(
                    "Nepodařilo se načíst vybraný rozvrh."
                )

                st.exception(e)

with ai_tab:
    st.subheader("AI přehledy a reporty" if is_czech else "AI Insights & Reports")

    # ----------------------------------
    # AI CONFLICT ANALYSIS
    # ----------------------------------
    st.divider()
    st.subheader("🤖 AI analýza konfliktů" if is_czech else "🤖 AI Conflict Analysis")

    st.write(
        "Gemini analyzuje výsledky kontroly, vysvětluje jejich dopad "
        "a navrhuje manažerovi další kroky."
    )

    st.caption(
        "Do Gemini se odesílají pouze anonymizované informace o kontrole. "
        "Umělá inteligence nemění rozvrh ani databázi."
    )

    if st.button(
        "Spustit AI analýzu konfliktů" if is_czech else "Run AI Conflict Analysis",
        type="primary",
        key="run_ai_conflict_analysis"
    ):

        try:
            with st.spinner(
                "Gemini analyzuje výsledky kontroly..."
            ):

                ai_report = generate_ai_conflict_analysis(
                    issues,
                    selected_week
                )

            st.session_state["ai_conflict_analysis"] = {
                "week": selected_week.isoformat(),
                "report": ai_report,
                "error": None
            }

        except Exception as e:
            st.session_state["ai_conflict_analysis"] = {
                "week": selected_week.isoformat(),
                "report": None,
                "error": str(e)
            }

    ai_result = st.session_state.get(
        "ai_conflict_analysis"
    )

    if (
        ai_result
        and ai_result.get("week") == selected_week.isoformat()
    ):

        if ai_result.get("error"):
            st.error("Analýza konfliktů pomocí AI se nezdařila.")
            st.code(ai_result["error"])

        elif ai_result.get("report"):
            st.markdown("### Zpráva AI pro manažera")
            st.markdown(ai_result["report"])

    st.subheader("🤖 AI asistent plánování" if is_czech else "🤖 AI Planning Assistant")

    st.write(
        "Pokud matematický optimalizátor nedokáže vytvořit platný rozvrh, "
        "Gemini může analyzovat souhrnné údaje o plánování a navrhnout "
        "alternativní postupy pro manažera."
    )

    st.caption(
        "Umělá inteligence pouze navrhuje scénáře. Nemění "
        "preference zaměstnanců, smlouvy, požadavky na směny ani záznamy v databázi."
    )

    last_attempt = st.session_state.get(
        "last_schedule_attempt"
    )

    planning_available = bool(
        last_attempt
        and last_attempt.get("week") == selected_week.isoformat()
        and last_attempt.get("status") in ("Infeasible", "Failed")
    )

    if planning_available:

        st.warning(
            "Poslední pokus o vytvoření rozvrhu pro tento týden nebyl úspěšný. "
            "Můžete požádat AI asistenta plánování o analýzu situace."
        )

        if st.button(
            "Spustit AI asistenta plánování" if is_czech else "Run AI Planning Assistant",
            type="primary",
            key="run_ai_planning_assistant"
        ):

            try:
                with st.spinner(
                    "Gemini analyzuje situaci při plánování..."
                ):

                    planning_report = generate_ai_planning_assistant(
                        active_employees=active_employees,
                        week_preferences=week_preferences,
                        shift_requirements=shift_requirements,
                        outcome=last_attempt["outcome"],
                        selected_week=selected_week,
                        failure_type=last_attempt["failure_type"]
                    )

                st.session_state["ai_planning_assistant"] = {
                    "week": selected_week.isoformat(),
                    "report": planning_report,
                    "error": None
                }

            except Exception as e:
                st.session_state["ai_planning_assistant"] = {
                    "week": selected_week.isoformat(),
                    "report": None,
                    "error": str(e)
                }

        planning_result = st.session_state.get(
            "ai_planning_assistant"
        )

        if (
            planning_result
            and planning_result.get("week") == selected_week.isoformat()
        ):

            if planning_result.get("error"):
                st.error("AI asistent plánování selhal.")
                st.code(planning_result["error"])

            elif planning_result.get("report"):
                st.markdown("### Doporučení AI pro plánování")
                st.markdown(planning_result["report"])

    else:
        st.info(
            "AI asistent plánování bude dostupný po neúspěšném "
            "pokusu o vytvoření rozvrhu pro vybraný týden."
        )

    # --------------------------------------
    # AI DECISION SUPPORT
    # --------------------------------------
    st.divider()
    st.subheader("🤖 AI podpora rozhodování")

    st.write(
        "Porovnejte nedávné úspěšné varianty rozvrhu pomocí přehledných "
        "provozních ukazatelů a získejte doporučení s podporou AI."
    )

    st.caption(
        "Gemini obdrží souhrnné ukazatele a anonymizované informace o přiřazení "
        "pouze počty rozdílů. ID zaměstnanců se do Gemini neodesílají. "
        "AI žádný rozvrh neupravuje, nevybírá, neschvaluje ani nezveřejňuje."
    )

    if st.button(
        "Spustit AI podporu rozhodování",
        type="primary",
        key="run_ai_decision_support"
    ):
        st.session_state.pop("ai_decision_support", None)

        try:
            recent_runs_response = (
                supabase.table("schedule_runs")
                .select("run_id,run_status,manager_approval_status,completed_at")
                .eq("week_start_date", selected_week.isoformat())
                .eq("run_status", "Feasible")
                .order("run_id", desc=True)
                .limit(5)
                .execute()
            )

            recent_runs = recent_runs_response.data or []

            if not recent_runs:
                st.session_state["ai_decision_support"] = {
                    "week": selected_week.isoformat(),
                    "metrics": [],
                    "report": None,
                    "error": (
                        "Pro vybraný týden nebyly nalezeny žádné úspěšné varianty rozvrhu "
                        "pro vybraný týden."
                    )
                }
            else:
                comparison_metrics = []
                assignments_by_run = {}

                with st.spinner(
                    "Počítají se ukazatele a skutečné rozdíly v přiřazení směn..."
                ):
                    for run in recent_runs:
                        run_id = int(run["run_id"])

                        assignments_response = (
                            supabase.table("schedule_assignments")
                            .select("*")
                            .eq("run_id", run_id)
                            .execute()
                        )

                        assignments_df = pd.DataFrame(
                            assignments_response.data or []
                        )
                        assignments_by_run[run_id] = assignments_df

                        metrics = calculate_decision_metrics(
                            run_id=run_id,
                            assignments=assignments_df,
                            active_employees=active_employees,
                            week_preferences=week_preferences,
                            shift_requirements=shift_requirements
                        )

                        metrics["manager_approval_status"] = (
                            run.get("manager_approval_status")
                            or "Nezaznamenáno"
                        )
                        comparison_metrics.append(metrics)

                assignment_comparisons = []
                run_ids = list(assignments_by_run.keys())
                for index_a in range(len(run_ids)):
                    for index_b in range(index_a + 1, len(run_ids)):
                        run_a_id = run_ids[index_a]
                        run_b_id = run_ids[index_b]
                        assignment_comparisons.append(
                            compare_schedule_assignments(
                                run_a_id,
                                assignments_by_run[run_a_id],
                                run_b_id,
                                assignments_by_run[run_b_id]
                            )
                        )

                with st.spinner(
                    "Gemini porovnává ukazatele rozvrhů a "
                    "anonymizovaných rozdílů v přiřazení..."
                ):
                    decision_report = generate_ai_decision_support(
                        comparison_metrics,
                        selected_week,
                        assignment_comparisons
                    )

                st.session_state["ai_decision_support"] = {
                    "week": selected_week.isoformat(),
                    "metrics": comparison_metrics,
                    "assignment_comparisons": assignment_comparisons,
                    "report": decision_report,
                    "error": None
                }

        except Exception as e:
            st.session_state["ai_decision_support"] = {
                "week": selected_week.isoformat(),
                "metrics": [],
                "report": None,
                "error": str(e)
            }

    decision_result = st.session_state.get("ai_decision_support")

    if (
        decision_result
        and decision_result.get("week") == selected_week.isoformat()
    ):
        if decision_result.get("error"):
            st.error("AI podpora rozhodování nedokázala dokončit porovnání.")
            st.error(decision_result["error"])

        else:
            decision_metrics = decision_result.get("metrics", [])

            if decision_metrics:
                st.markdown("### Ukazatele variant rozvrhu")

                summary_rows = []
                for item in decision_metrics:
                    preference = item["preference_match"]
                    contracts = item["contract_compliance"]
                    days_off = item["requested_days_off_compliance"]
                    checks = item["constraint_checks"]

                    summary_rows.append({
                        "Varianta": item["variant"],
                        "Schválení manažerem": {
                            "Approved": "Schváleno",
                            "Rejected": "Zamítnuto",
                            "Not_Reviewed": "Nezkontrolováno",
                            "Not recorded": "Nezaznamenáno"
                        }.get(item["manager_approval_status"], item["manager_approval_status"]),
                        "Obsazená místa": (
                            f'{item["assigned_positions"]}/'
                            f'{item["expected_positions"]}'
                        ),
                        "Pokrytí směn (%)": item["coverage_percent"],
                        "Splnění preferencí (%)": preference["match_percent"],
                        "Soulad s počtem pracovních dnů podle smlouvy": (
                            f'{contracts["employees_meeting_exact_workdays"]}/'
                            f'{contracts["employees_checked"]}'
                        ),
                        "Dodržení požadovaných dnů volna": (
                            f'{days_off["employees_respecting_requested_days_off"]}/'
                            f'{days_off["employees_checked"]}'
                        ),
                        "Čistý počet hodin": item["total_net_hours"],
                        "Směny Long bez souhlasu": (
                            checks["long_without_explicit_consent"]
                        ),
                        "Zaměstnanci s více směnami Long": (
                            checks["employees_with_more_than_one_long_shift"]
                        ),
                        "Duplicitní směna zaměstnance v jednom dni": (
                            checks["duplicate_employee_day_assignments"]
                        )
                    })

                st.dataframe(
                    pd.DataFrame(summary_rows),
                    use_container_width=True,
                    hide_index=True
                )

                assignment_comparisons = decision_result.get(
                    "assignment_comparisons", []
                )
                if assignment_comparisons:
                    st.markdown("### Rozdíly ve skutečném přiřazení směn")
                    st.caption(
                        "Párové porovnání uložených přiřazení podle data, "
                        "pobočky, směny a zaměstnance. Identifikátory zaměstnanců "
                        "se nezobrazují ani se neodesílají do Gemini."
                    )

                    assignment_rows = []
                    for comparison in assignment_comparisons:
                        assignment_rows.append({
                            "Porovnávané běhy": comparison["comparison"],
                            "Počet přiřazení (A / B)": (
                                f'{comparison["total_assignments_run_a"]} / '
                                f'{comparison["total_assignments_run_b"]}'
                            ),
                            "Směnová místa s odlišnými zaměstnanci": (
                                comparison[
                                    "slots_with_different_employee_assignments"
                                ]
                            ),
                            "Změněná přiřazení zaměstnanců na směnová místa": (
                                comparison[
                                    "employee_to_slot_entries_changed"
                                ]
                            ),
                            "Zaměstnanci se změněným týdenním rozvrhem": (
                                comparison[
                                    "employees_with_changed_weekly_schedules"
                                ]
                            ),
                            "Kombinace data, pobočky a směny s odlišným obsazením": (
                                comparison[
                                    "date_branch_shift_buckets_with_different_staffing"
                                ]
                            ),
                            "Skupiny směn pouze ve variantě A": (
                                comparison["slot_groups_only_in_run_a"]
                            ),
                            "Skupiny směn pouze ve variantě B": (
                                comparison["slot_groups_only_in_run_b"]
                            )
                        })

                    st.dataframe(
                        pd.DataFrame(assignment_rows),
                        use_container_width=True,
                        hide_index=True
                    )
                elif len(decision_metrics) == 1:
                    st.info(
                        "Je k dispozici pouze jedna úspěšná varianta. "
                        "Porovnání skutečného přiřazení směn vyžaduje alespoň "
                        "dvě varianty."
                    )

                for item in decision_metrics:
                    with st.expander(
                        f'{item["variant"]} — rozdělení směn a počet hodin'
                    ):
                        detail_rows = []
                        for shift in SHIFT_ORDER:
                            detail_rows.append({
                                "Směna": shift,
                                "Počet směn": item["shift_counts"][shift],
                                "Čistý počet hodin": item["net_hours_by_shift"][shift]
                            })
                        st.dataframe(
                            pd.DataFrame(detail_rows),
                            use_container_width=True,
                            hide_index=True
                        )

                if decision_result.get("report"):
                    st.markdown("### Doporučení AI pro manažera")
                    st.markdown(decision_result["report"])

                if len(decision_metrics) == 1:
                    st.info(
                        "Je k dispozici pouze jedna úspěšná varianta. "
                        "Zpráva ji vyhodnotí, ale nelze provést "
                        "smysluplné porovnání různých možností."
                    )
            else:
                st.info(
                    "Nejsou k dispozici žádné ukazatele rozvrhů k porovnání."
                )

# ==========================================
# 10. FOOTER
# ==========================================
st.divider()

st.caption(
    "Plánovač směn s AI | Manažerský přehled"
)
