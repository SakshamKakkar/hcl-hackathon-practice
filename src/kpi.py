"""
kpi.py
Manufacturing KPIs computed in SQL: OEE, bottleneck station, downtime by shift.

OEE (Overall Equipment Effectiveness) = Availability x Performance x Quality
  Availability = run time / planned time               (lost to breakdowns, changeovers)
  Performance  = (ideal cycle time x units) / run time (lost to running slower than ideal)
  Quality      = good units / total units              (lost to defects / scrap)
World-class OEE is often quoted as ~85%.
"""

import pandas as pd                     # results come back as DataFrames
from src.db import run_query            # our helper that runs SQL on factory.db

# SQL is kept in constants so it is easy to read, test and change during the demo.
OEE_BY_STATION_SQL = """
WITH totals AS (                                              -- CTE: first sum raw minutes/units per line+station
    SELECT line_id,
           station,
           SUM(planned_minutes)                    AS planned_min,   -- total planned production time
           SUM(planned_minutes - downtime_minutes) AS run_min,       -- time the station actually ran
           SUM(ideal_cycle_sec * units_produced) / 60.0 AS ideal_min,-- time the output SHOULD have taken at ideal speed
           SUM(units_produced)                     AS units,
           SUM(defective_units)                    AS defects
    FROM production_log
    WHERE line_id = COALESCE(?, line_id)                      -- optional filter: pass None to include every line
    GROUP BY line_id, station
)
SELECT line_id,
       station,
       units,
       ROUND(run_min / planned_min, 4)            AS availability,
       ROUND(ideal_min / run_min, 4)              AS performance,
       ROUND((units - defects) * 1.0 / units, 4)  AS quality,      -- *1.0 forces decimal (not integer) division
       ROUND((run_min / planned_min) * (ideal_min / run_min) * ((units - defects) * 1.0 / units), 4) AS oee,
       ROUND(units * 60.0 / run_min, 2)           AS units_per_run_hour   -- throughput while running
FROM totals
ORDER BY line_id, station;
"""
# Note: we sum first and divide later. Averaging per-shift percentages would weight a short shift
# the same as a long one, which is wrong.

DOWNTIME_BY_SHIFT_SQL = """
SELECT shift,
       ROUND(AVG(downtime_minutes), 1) AS avg_downtime_min,          -- average minutes lost per station-shift
       ROUND(SUM(downtime_minutes) / 60.0, 1) AS total_downtime_hr,
       RANK() OVER (ORDER BY AVG(downtime_minutes) DESC) AS worst_rank   -- window function: 1 = worst shift
FROM production_log
GROUP BY shift
ORDER BY worst_rank;
"""


def oee_by_station(line_id: str | None = None) -> pd.DataFrame:
    """OEE components per station; line_id=None means all lines."""
    return run_query(OEE_BY_STATION_SQL, params=(line_id,))     # the "?" in the SQL is filled with line_id


def find_bottleneck(oee_df: pd.DataFrame) -> pd.DataFrame:
    """The bottleneck is the station with the lowest throughput on each line: it caps the whole line's output."""
    idx = oee_df.groupby("line_id")["units_per_run_hour"].idxmin()   # row index of the slowest station per line
    return oee_df.loc[idx, ["line_id", "station", "units_per_run_hour"]].reset_index(drop=True)


def downtime_by_shift() -> pd.DataFrame:
    """Downtime aggregated per shift, ranked worst first."""
    return run_query(DOWNTIME_BY_SHIFT_SQL)


def downtime_saving(reduction_pct: float, cost_per_min: float, line_id: str | None = None) -> dict:
    """What-if: if downtime falls by reduction_pct %, how many minutes and rupees do we save per month?"""
    df = run_query(
        "SELECT SUM(downtime_minutes) AS dt FROM production_log WHERE line_id = COALESCE(?, line_id)",
        params=(line_id,),
    )
    total_dt = float(df["dt"].iloc[0])                          # total downtime minutes in the data (one month)
    saved_min = total_dt * reduction_pct / 100                  # minutes recovered by the improvement
    return {
        "monthly_downtime_min": round(total_dt, 1),
        "minutes_saved": round(saved_min, 1),
        "rupees_saved": round(saved_min * cost_per_min, 0),     # cost_per_min is an assumption the user enters
    }
