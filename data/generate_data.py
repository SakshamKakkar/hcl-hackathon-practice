"""
generate_data.py
Creates two FAKE (synthetic) manufacturing datasets so we can practise without internet:
  1. sensor_readings.csv  -> one row per machine sensor reading, with a 0/1 "failure" label
  2. production_log.csv   -> one row per station per shift, used for OEE / throughput / bottleneck KPIs
The failure rules are inspired by the public AI4I 2020 Predictive Maintenance dataset.
Run:  python data/generate_data.py
"""

import numpy as np                      # numpy: fast maths on arrays + random numbers
import pandas as pd                     # pandas: tables (DataFrames) and CSV saving
from pathlib import Path                # Path: OS-independent way to build file paths

RNG = np.random.default_rng(42)         # random generator with a fixed seed -> same data every run (reproducible)
OUT_DIR = Path(__file__).parent         # folder this script lives in (the data/ folder)


def make_sensor_readings(n_rows: int = 5000) -> pd.DataFrame:
    """Return a DataFrame of n_rows fake sensor readings with a failure label."""
    machine_type = RNG.choice(["L", "M", "H"], size=n_rows, p=[0.5, 0.3, 0.2])   # L/M/H = low/medium/high quality machine variants
    machine_id = RNG.integers(1, 21, size=n_rows)                                # 20 machines, ids 1..20
    line_id = np.where(machine_id <= 10, "LINE_A", "LINE_B")                     # machines 1-10 on line A, 11-20 on line B
    air_temp = RNG.normal(300, 2, size=n_rows)                                   # air temperature in Kelvin, mean 300, std 2
    process_temp = air_temp + 10 + RNG.normal(0, 1, size=n_rows)                 # process runs ~10 K hotter than air
    rpm = RNG.normal(1540, 180, size=n_rows).clip(1150, 2900)                    # spindle speed in revolutions per minute, clipped to a sane range
    torque = (40 - (rpm - 1540) * 0.03 + RNG.normal(0, 8, size=n_rows)).clip(5, 80)  # torque (Nm) drops as rpm rises (physics: power ~ constant)
    wear_rate = np.select([machine_type == "L", machine_type == "M"], [5, 3], default=2)  # cheaper machines wear tools faster
    tool_wear = RNG.integers(0, 240, size=n_rows) * wear_rate / 5                # minutes of tool use, scaled by machine type

    power_w = torque * rpm * 2 * np.pi / 60                                      # mechanical power in watts = torque x angular speed
    temp_diff = process_temp - air_temp                                          # heat-dissipation margin

    # ---- failure rules (each is a realistic failure mode) ----
    tool_wear_fail = tool_wear > 225                                             # worn-out tool
    heat_fail = (temp_diff < 8.6) & (rpm < 1380)                                 # poor cooling at low speed
    power_fail = (power_w < 3500) | (power_w > 9000)                             # power out of the safe band
    overstrain_fail = tool_wear * torque > np.select([machine_type == "L", machine_type == "M"], [11000, 12000], default=13000)  # wear x load too high
    random_fail = RNG.random(n_rows) < 0.002                                     # 0.2% unexplained failures (real life is noisy)
    failure = (tool_wear_fail & (RNG.random(n_rows) < 0.3)) | heat_fail | power_fail | overstrain_fail | random_fail  # any mode -> failure

    timestamps = pd.Timestamp("2026-09-01") + pd.to_timedelta(np.sort(RNG.integers(0, 30 * 24 * 60, size=n_rows)), unit="m")  # spread over 30 days, sorted in time

    df = pd.DataFrame({                                                          # assemble all columns into one table
        "reading_id": np.arange(1, n_rows + 1),
        "timestamp": timestamps,
        "machine_id": machine_id,
        "line_id": line_id,
        "machine_type": machine_type,
        "air_temp_k": air_temp.round(1),
        "process_temp_k": process_temp.round(1),
        "rpm": rpm.round(0).astype(int),
        "torque_nm": torque.round(1),
        "tool_wear_min": tool_wear.round(0).astype(int),
        "failure": failure.astype(int),                                          # bool -> 0/1 so models and SQL handle it easily
    })
    missing_idx = RNG.choice(n_rows, size=int(0.01 * n_rows), replace=False)     # pick 1% of rows ...
    df.loc[missing_idx, "torque_nm"] = np.nan                                    # ... and blank their torque: real data has gaps
    return df


def make_production_log() -> pd.DataFrame:
    """Return one row per (date, shift, line, station) with planned time, downtime, output and defects."""
    stations = {"S1_CUTTING": 30, "S2_WELDING": 42, "S3_PAINTING": 36, "S4_ASSEMBLY": 33}  # station -> ideal cycle time in seconds (welding is slowest)
    rows = []                                                                    # we append dicts here, then build a DataFrame once (fast)
    for day in pd.date_range("2026-09-01", "2026-09-30"):                         # every day in September
        for shift in ["A", "B", "C"]:                                            # three 8-hour shifts
            for line in ["LINE_A", "LINE_B"]:                                    # two production lines
                for station, ideal_cycle in stations.items():                    # four stations on each line
                    planned = 480                                                # 8 hours x 60 = planned production minutes
                    downtime = max(0, RNG.normal(35 if shift == "C" else 20, 10))   # night shift C has more downtime on average
                    run_min = planned - downtime                                 # minutes the station actually ran
                    speed = RNG.uniform(0.80, 0.97)                              # running slower than ideal (performance loss)
                    units = int(run_min * 60 / ideal_cycle * speed)              # units = run seconds / cycle seconds x speed
                    defect_rate = RNG.uniform(0.01, 0.04) + (0.02 if station == "S3_PAINTING" else 0)  # painting has extra defects
                    rows.append({
                        "date": day.date(), "shift": shift, "line_id": line, "station": station,
                        "planned_minutes": planned, "downtime_minutes": round(downtime, 1),
                        "ideal_cycle_sec": ideal_cycle, "units_produced": units,
                        "defective_units": int(units * defect_rate),
                    })
    return pd.DataFrame(rows)                                                    # list of dicts -> table


if __name__ == "__main__":                                                       # only runs when executed directly, not when imported
    make_sensor_readings().to_csv(OUT_DIR / "sensor_readings.csv", index=False)  # index=False: don't write pandas' row numbers
    make_production_log().to_csv(OUT_DIR / "production_log.csv", index=False)
    print("Wrote sensor_readings.csv and production_log.csv to", OUT_DIR)
