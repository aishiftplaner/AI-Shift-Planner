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
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    for e, (_, emp) in enumerate(employees.iterrows()):
        pref = pref_by_id[emp["employee_id"]]
        days_off = pref["required_days_off"]

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

        if (
            (group["shift_type"] == "Long").any()
            and pref["long_shift_consent"] != "Yes"
        ):
            raise ValueError(
                f"Post-check failed: Long Shift without consent for {employee_id}."
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
st.title("📅 AI Shift Planner")
st.subheader("Manager Dashboard")


# ==========================================
# 9.1. GEMINI API CONNECTION TEST
# ==========================================
with st.expander("Gemini AI Connection Test"):
    st.write(
        "Check whether the Gemini API is connected."
    )

    if st.button("Test Gemini Connection"):
        try:
            with st.spinner("Connecting to Gemini..."):
                answer = test_gemini_connection()

            st.success(
                "Gemini API connected successfully!"
            )

            st.write("Gemini response:")
            st.write(answer)

        except Exception as e:
            st.error("Gemini API connection failed.")
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
    st.error("Could not load database tables.")
    st.exception(e)
    st.stop()


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


# ==========================================
# 9.4. WEEKLY SUBMISSION OVERVIEW
# ==========================================
st.divider()
st.header("Weekly Submission Overview")

c1, c2, c3 = st.columns(3)

c1.metric("Active Employees", total_employees)
c2.metric("Received Submissions", received_count)
c3.metric("Missing Submissions", missing_count)


# ==========================================
# 9.5. DASHBOARD TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Employees",
    "Branches",
    "Shift Requirements",
    "Company Rules",
    "Weekly Preferences",
    "Validation Engine",
    "Scheduling"
])


# ==========================================
# TAB 1. EMPLOYEES
# ==========================================
with tab1:
    st.subheader("Employees")

    st.dataframe(
        employees,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# TAB 2. BRANCHES
# ==========================================
with tab2:
    st.subheader("Branches")

    st.dataframe(
        branches,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# TAB 3. SHIFT REQUIREMENTS
# ==========================================
with tab3:
    st.subheader("Shift Requirements")

    st.dataframe(
        shift_requirements,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# TAB 4. COMPANY RULES
# ==========================================
with tab4:
    st.subheader("Company Rules")

    st.dataframe(
        company_rules,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# TAB 5. WEEKLY PREFERENCES
# ==========================================
with tab5:
    st.subheader("Weekly Employee Preferences")

    st.write(f"Week starting: {selected_week}")

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
        st.success(
            "All employees have submitted their preferences."
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


# ==========================================
# TAB 6. VALIDATION ENGINE
# ==========================================
with tab6:
    st.subheader("Weekly Data Validation")

    st.write(
        "The Validation Engine checks employee submissions before scheduling."
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

        a.metric("Critical Issues", critical_count)
        b.metric("Warnings", warning_count)

        if critical_count:
            st.error(
                "BLOCKED — Scheduling cannot start. "
                "Critical issues must be resolved."
            )

        elif warning_count:
            st.warning(
                "Review the warnings before proceeding."
            )

        else:
            st.success(
                "READY — All employee submissions passed validation."
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
                    "Validation completed, but synchronizing issues "
                    "in Supabase failed."
                )

                st.code(save_status["error"])

            else:
                st.success(
                    f"Current issues saved/updated: "
                    f"{save_status['saved_count']}. "
                    f"Old issues marked Resolved: "
                    f"{save_status['resolved_count']}."
                )

        if issues:
            st.subheader("Validation Issues")

            st.dataframe(
                pd.DataFrame(issues),
                use_container_width=True,
                hide_index=True
            )

        else:
            st.success("No validation issues found.")

        # ----------------------------------
        # AI CONFLICT ANALYSIS
        # ----------------------------------
        st.divider()
        st.subheader("🤖 AI Conflict Analysis")

        st.write(
            "Gemini analyzes validation findings, explains their impact "
            "and suggests actions for the manager."
        )

        st.caption(
            "Only anonymized validation information is sent to Gemini. "
            "The AI does not modify the schedule or database."
        )

        if st.button(
            "Run AI Conflict Analysis",
            type="primary",
            key="run_ai_conflict_analysis"
        ):

            try:
                with st.spinner(
                    "Gemini is analyzing the validation results..."
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
                st.error("AI Conflict Analysis failed.")
                st.code(ai_result["error"])

            elif ai_result.get("report"):
                st.markdown("### AI Manager Report")
                st.markdown(ai_result["report"])


# ==========================================
# TAB 7. SCHEDULING
# ==========================================
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
        st.warning(
            "Scheduling is locked. Run Validation and obtain READY "
            "for this exact week."
        )

    # --------------------------------------
    # Generate weekly schedule
    # --------------------------------------
    if st.button(
        "Generate Weekly Schedule",
        type="primary",
        disabled=not validation_ready
    ):

        run_id = None

        # Clear previous Planning Assistant results for this week.
        st.session_state.pop("ai_planning_assistant", None)

        try:
            run_id = create_schedule_run(selected_week)

            with st.spinner(
                "The optimization algorithm is generating the schedule..."
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
                    f"Schedule generated successfully. Run ID: {run_id}"
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

                st.error("No valid schedule was produced.")

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
                "message": "Scheduling execution failed.",
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
                "Scheduling execution failed. "
                "Check the technical error before trying again."
            )

            st.error(
                "Scheduling run failed. "
                "No schedule was approved or published."
            )

            st.exception(e)

    if st.session_state.get("last_schedule_message"):
        st.caption(
            st.session_state["last_schedule_message"]
        )

    # --------------------------------------
    # AI PLANNING ASSISTANT
    # --------------------------------------
    st.divider()
    st.subheader("🤖 AI Planning Assistant")

    st.write(
        "If the mathematical optimizer cannot produce a valid schedule, "
        "Gemini can analyze aggregated planning data and suggest "
        "alternative actions for the manager."
    )

    st.caption(
        "The AI only proposes scenarios. It does not change employee "
        "preferences, contracts, shift requirements or database records."
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
            "The latest scheduling attempt for this week was unsuccessful. "
            "You can ask the AI Planning Assistant to analyze the situation."
        )

        if st.button(
            "Run AI Planning Assistant",
            type="primary",
            key="run_ai_planning_assistant"
        ):

            try:
                with st.spinner(
                    "Gemini is analyzing the planning situation..."
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
                st.error("AI Planning Assistant failed.")
                st.code(planning_result["error"])

            elif planning_result.get("report"):
                st.markdown("### AI Planning Recommendations")
                st.markdown(planning_result["report"])

    else:
        st.info(
            "AI Planning Assistant becomes available after an unsuccessful "
            "scheduling attempt for the selected week."
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
            "Could not load schedule run history."
        )

        st.exception(e)

    if not runs.empty:

        st.divider()
        st.subheader("Schedule Run History")

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
                "View generated schedule (Run ID)",
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

                    st.subheader("Generated Weekly Schedule")

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
                        st.subheader("Manager Approval")

                        manager_comment = st.text_area(
                            "Manager comment (optional)",
                            key=f"approval_comment_{selected_run_id}"
                        )

                        approve_col, reject_col = st.columns(2)

                        with approve_col:

                            if st.button(
                                "Approve Schedule",
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
                                        "Schedule approved. "
                                        "The decision was saved."
                                    )

                                    st.rerun()

                                except Exception as e:
                                    st.error(
                                        "Could not save manager approval."
                                    )
                                    st.exception(e)

                        with reject_col:

                            if st.button(
                                "Reject Schedule",
                                key=f"reject_{selected_run_id}"
                            ):

                                try:
                                    approve_or_reject_run(
                                        selected_run_id,
                                        "Rejected",
                                        manager_comment
                                    )

                                    st.warning(
                                        "Schedule rejected. "
                                        "The decision was saved."
                                    )

                                    st.rerun()

                                except Exception as e:
                                    st.error(
                                        "Could not save manager rejection."
                                    )
                                    st.exception(e)

                    else:
                        st.info(
                            "Manager decision: "
                            f"{selected_run['manager_approval_status']}"
                        )

            except Exception as e:
                st.error(
                    "Could not load the selected schedule."
                )

                st.exception(e)




# ==========================================
# 9.6. AI DECISION SUPPORT
# ==========================================
st.divider()
st.header("🤖 AI Decision Support")

st.write(
    "Gemini compares successful schedule runs and helps the manager "
    "review their operational quality before making an approval decision."
)

st.caption(
    "The comparison uses anonymized schedule metrics. Gemini cannot "
    "change schedules, database records or approval status."
)

try:
    decision_runs_response = (
        supabase.table("schedule_runs")
        .select("run_id,run_status,manager_approval_status,completed_at")
        .eq("week_start_date", selected_week.isoformat())
        .eq("run_status", "Feasible")
        .order("run_id", desc=True)
        .limit(5)
        .execute()
    )

    decision_runs = decision_runs_response.data or []

except Exception as e:
    decision_runs = []
    st.error("Could not load successful runs for decision support.")
    st.exception(e)

if not decision_runs:
    st.info(
        "AI Decision Support becomes available after at least one "
        "successful schedule has been generated for this week."
    )

else:
    st.write(
        f"Successful schedule candidates available: {len(decision_runs)} "
        f"(the five most recent runs at most)."
    )

    decision_summaries = []
    summary_errors = []

    for run in decision_runs:
        try:
            assignment_response = (
                supabase.table("schedule_assignments")
                .select("*")
                .eq("run_id", int(run["run_id"]))
                .execute()
            )

            run_assignments = pd.DataFrame(
                assignment_response.data or []
            )

            summary = build_schedule_quality_summary(
                run_id=run["run_id"],
                assignments=run_assignments,
                active_employees=active_employees,
                week_preferences=week_preferences,
                shift_requirements=shift_requirements
            )

            summary["manager_approval_status"] = (
                run.get("manager_approval_status", "Unknown")
            )
            decision_summaries.append(summary)

        except Exception as e:
            summary_errors.append(
                f"Run {run['run_id']}: {str(e)}"
            )

    if summary_errors:
        with st.expander("Candidate data issues"):
            for message in summary_errors:
                st.warning(message)

    if decision_summaries:
        st.subheader("Deterministic Candidate Metrics")

        metrics_df = pd.DataFrame(decision_summaries)

        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button(
            "Run AI Decision Support",
            type="primary",
            key="run_ai_decision_support"
        ):
            try:
                with st.spinner(
                    "Gemini is comparing the successful schedule candidates..."
                ):
                    decision_report = generate_ai_decision_support(
                        decision_summaries,
                        selected_week
                    )

                st.session_state["ai_decision_support"] = {
                    "week": selected_week.isoformat(),
                    "report": decision_report,
                    "error": None
                }

            except Exception as e:
                st.session_state["ai_decision_support"] = {
                    "week": selected_week.isoformat(),
                    "report": None,
                    "error": str(e)
                }

        decision_result = st.session_state.get(
            "ai_decision_support"
        )

        if (
            decision_result
            and decision_result.get("week") == selected_week.isoformat()
        ):
            if decision_result.get("error"):
                st.error("AI Decision Support failed.")
                st.code(decision_result["error"])

            elif decision_result.get("report"):
                st.subheader("AI Manager Decision Report")
                st.markdown(decision_result["report"])

    else:
        st.warning(
            "No schedule candidate could be summarized. "
            "Review the candidate data issues above."
        )


# ==========================================
# 10. FOOTER
# ==========================================
st.divider()

st.caption(
    "AI Shift Planner | Manager Dashboard"
)
