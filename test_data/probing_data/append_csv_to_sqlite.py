import sqlite3
import pandas as pd
import argparse

def append_csv_to_sqlite(csv_file, table_name, db_file):
    """Appends a CSV file to a given SQLite table with matching column names."""
    
    # Load CSV into a DataFrame
    df = pd.read_csv(csv_file)

    # Connect to SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Get existing table column names
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing_columns = {row[1] for row in cursor.fetchall()}  # Column names from DB

    # Filter DataFrame to only include matching columns
    df = df[[col for col in df.columns if col in existing_columns]]

    if df.empty:
        print("No matching columns found. Nothing to insert.")
    else:
        # Append DataFrame to the table
        df.to_sql(table_name, conn, if_exists="append", index=False)
        print(f"Successfully appended {len(df)} rows to {table_name}.")

    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Append a CSV file to an SQLite table with matching columns.")
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument("table_name", help="SQLite table name")
    parser.add_argument("db_file", help="Path to the SQLite database file")

    args = parser.parse_args()
    append_csv_to_sqlite(args.csv_file, args.table_name, args.db_file)