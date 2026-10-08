# Change drills

A panel can ask you to change the code live. Time yourself at 5 minutes per drill.

After each change:

1. Run `python -m pytest -q` and the app.
2. Commit the change.
3. Run `git checkout .` (or `git revert`) to reset for the next drill.

Hints are below each drill. Try the drill first.

## SQL and KPIs (src/kpi.py)

1. **Add a per-station defect-rate column to the OEE query.**
    - Hint: `ROUND(defects * 1.0 / units, 4) AS defect_rate` in the final SELECT.
2. **Make the bottleneck the station with the lowest OEE instead of the lowest throughput.**
    - Hint: change `"units_per_run_hour"` to `"oee"` in `find_bottleneck`. Explain why throughput is still the better definition of a bottleneck.
3. **Add a shift filter to `oee_by_station`.**
    - Hint: add `AND shift = COALESCE(?, shift)` in the WHERE clause, and a second param `(line_id, shift)`.
4. **Show downtime by line and shift together.**
    - Hint: `GROUP BY line_id, shift`.
5. **Return the top 3 worst days by downtime.**
    - Hint: `GROUP BY date ORDER BY SUM(downtime_minutes) DESC LIMIT 3`.

## Model (src/model.py)

6. **Add gradient boosting as a fourth candidate.**
    - Hint: import `GradientBoostingClassifier` and add it to `CANDIDATES`. It has no `class_weight` parameter, so explain the trade-off.
7. **Remove an engineered feature and compare the results.**
    - Hint: delete `"power_w"` from `NUMERIC`. Expect F1 to drop, which proves the feature matters.
8. **Change the cost assumption.**
    - Hint: call `train_and_save(cost_missed=100000, cost_false_alarm=1000)`. Expect a lower threshold, higher recall and more false alarms. Explain why.
9. **Use mean imputation instead of median.**
    - Hint: `SimpleImputer(strategy="mean")`. Say why median is safer with outliers.
10. **Report accuracy too, and explain why it misleads here.**
    - Hint: add `accuracy_score` to `evaluate`.

## App (app.py)

11. **Add a fourth KPI tile for average quality.**
    - Hint: `st.columns(4)`, then `c4.metric("Quality", f"{oee['quality'].mean():.1%}")`.
12. **Block prediction when tool wear is above 300.**
    - Hint: use an `if` before `predict_one` and `st.stop()`.
13. **Let the user set the alert threshold with a slider.**
    - Hint: pass a slider value instead of `saved["threshold"]`, and compare the probability in the app.

## Git

14. Make a branch, commit, merge back, and delete the branch.
15. Undo your last commit while keeping the changes (`git reset --soft HEAD~1`), then revert a pushed commit (`git revert <hash>`).
