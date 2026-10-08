"""
db.py
Stage the raw CSV files into a SQLite database, the same "raw file -> structured table" pattern
used in real data pipelines (and in my HDFC FinOps project).
"""

import sqlite3                          # built-in Python module: SQLite is a whole SQL database in one file, no server needed
from pathlib import Path                # file paths that work on Windows, Mac and Linux
import pandas as pd                     # read CSVs and write tables into the database

ROOT = Path(__file__).resolve().parent.parent      # project root folder (src/ -> its parent)
DATA_DIR = ROOT / "data"                           # where the CSV files live
DB_PATH = ROOT / "factory.db"                      # the SQLite database file we create


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open (or create) the SQLite database file and return a connection object."""
    return sqlite3.connect(db_path)                # creates the file if it doesn't exist yet


def load_csvs_to_db(db_path: Path = DB_PATH) -> dict:
    """Load both CSVs into tables. Returns {table_name: row_count} so we can sanity-check the load."""
    tables = {                                     # table name in SQL -> CSV file on disk
        "sensor_readings": DATA_DIR / "sensor_readings.csv",
        "production_log": DATA_DIR / "production_log.csv",
    }
    counts = {}                                    # will hold row counts per table
    with get_connection(db_path) as conn:          # "with" closes/commits the connection automatically
        for name, csv_path in tables.items():      # loop over each table we want to create
            df = pd.read_csv(csv_path)             # read the CSV into a DataFrame
            df.to_sql(name, conn, if_exists="replace", index=False)   # write it as a SQL table; replace = re-runnable
            counts[name] = len(df)                 # remember how many rows we loaded
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sensor_machine ON sensor_readings(machine_id)")  # index speeds up WHERE/GROUP BY on machine_id
    return counts


def run_query(sql: str, params: tuple = (), db_path: Path = DB_PATH) -> pd.DataFrame:
    """Run any SELECT query and return the result as a DataFrame."""
    with get_connection(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)   # params use "?" placeholders -> safe from SQL injection


if __name__ == "__main__":
    print(load_csvs_to_db())                       # e.g. {'sensor_readings': 5000, 'production_log': 720}
