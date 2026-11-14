
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
    # AI DECISION SUPPORT
    # --------------------------------------
    st.divider()
    st.subheader("🤖 AI Decision Support")

    st.write(
        "Compare recent successful schedule variants using transparent "
        "operational metrics and receive an AI-supported recommendation."
    )

    st.caption(
        "Gemini receives aggregated metrics and anonymized assignment "
        "difference counts only. Employee IDs are not sent to Gemini. "
        "It does not modify, select, approve or publish any schedule."
    )

    if st.button(
        "Run AI Decision Support",
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
                        "No successful schedule variants were found "
                        "for the selected week."
                    )
                }
            else:
                comparison_metrics = []
                assignments_by_run = {}

                with st.spinner(
                    "Calculating metrics and actual assignment differences..."
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
                            or "Not recorded"
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
                    "Gemini is comparing schedule metrics and "
                    "anonymized assignment differences..."
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
            st.error("AI Decision Support could not complete the comparison.")
            st.error(decision_result["error"])

        else:
            decision_metrics = decision_result.get("metrics", [])

            if decision_metrics:
                st.markdown("### Schedule Variant Metrics")

                summary_rows = []
                for item in decision_metrics:
                    preference = item["preference_match"]
                    contracts = item["contract_compliance"]
                    days_off = item["requested_days_off_compliance"]
                    checks = item["constraint_checks"]

                    summary_rows.append({
                        "Variant": item["variant"],
                        "Approval": item["manager_approval_status"],
                        "Positions": (
                            f'{item["assigned_positions"]}/'
                            f'{item["expected_positions"]}'
                        ),
                        "Coverage (%)": item["coverage_percent"],
                        "Preference match (%)": preference["match_percent"],
                        "Contract days compliant": (
                            f'{contracts["employees_meeting_exact_workdays"]}/'
                            f'{contracts["employees_checked"]}'
                        ),
                        "Requested days off respected": (
                            f'{days_off["employees_respecting_requested_days_off"]}/'
                            f'{days_off["employees_checked"]}'
                        ),
                        "Net hours": item["total_net_hours"],
                        "Long consent violations": (
                            checks["long_without_explicit_consent"]
                        ),
                        "Multiple Long violations": (
                            checks["employees_with_more_than_one_long_shift"]
                        ),
                        "Duplicate employee/day": (
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
                    st.markdown("### Actual Assignment Differences")
                    st.caption(
                        "Pairwise comparison of saved assignments by date, "
                        "branch, shift and employee. Employee identifiers "
                        "are not displayed or sent to Gemini."
                    )

                    assignment_rows = []
                    for comparison in assignment_comparisons:
                        assignment_rows.append({
                            "Compared runs": comparison["comparison"],
                            "Assignments (A / B)": (
                                f'{comparison["total_assignments_run_a"]} / '
                                f'{comparison["total_assignments_run_b"]}'
                            ),
                            "Shift slots with different employees": (
                                comparison[
                                    "slots_with_different_employee_assignments"
                                ]
                            ),
                            "Changed employee-to-slot entries": (
                                comparison[
                                    "employee_to_slot_entries_changed"
                                ]
                            ),
                            "Employees with changed weekly schedules": (
                                comparison[
                                    "employees_with_changed_weekly_schedules"
                                ]
                            ),
                            "Date/branch/shift groups with different staffing": (
                                comparison[
                                    "date_branch_shift_buckets_with_different_staffing"
                                ]
                            ),
                            "Slot groups only in A": (
                                comparison["slot_groups_only_in_run_a"]
                            ),
                            "Slot groups only in B": (
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
                        "Only one successful variant is available. "
                        "Actual assignment differences require at least "
                        "two variants."
                    )

                for item in decision_metrics:
                    with st.expander(
                        f'{item["variant"]} — shift distribution and hours'
                    ):
                        detail_rows = []
                        for shift in SHIFT_ORDER:
                            detail_rows.append({
                                "Shift": shift,
                                "Assignments": item["shift_counts"][shift],
                                "Net hours": item["net_hours_by_shift"][shift]
                            })
                        st.dataframe(
                            pd.DataFrame(detail_rows),
                            use_container_width=True,
                            hide_index=True
                        )

                if decision_result.get("report"):
                    st.markdown("### AI Manager Recommendation")
                    st.markdown(decision_result["report"])

                if len(decision_metrics) == 1:
                    st.info(
                        "Only one successful variant is available. "
                        "The report evaluates it, but cannot make a "
                        "meaningful comparison between alternatives."
                    )
            else:
                st.info(
                    "No schedule metrics are available for comparison."
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
# 10. FOOTER
# ==========================================
st.divider()

st.caption(
    "AI Shift Planner | Manager Dashboard"
)
