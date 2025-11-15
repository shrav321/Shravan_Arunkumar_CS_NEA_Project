import sqlite3

DB_PATH = "options.db"


def get_connection():
    return sqlite3.connect(DB_PATH)
def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # CONTRACT table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS CONTRACT (
        contract_id TEXT PRIMARY KEY,
        ticker      TEXT NOT NULL,
        expiry      TEXT NOT NULL,
        strike      REAL NOT NULL,
        type        TEXT NOT NULL
    );
    """)

    # TRADE table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS TRADE (
        trade_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        contract_id TEXT NOT NULL,
        quantity    INTEGER NOT NULL,
        price       REAL NOT NULL,
        timestamp   TEXT NOT NULL,
        side        TEXT NOT NULL,
        FOREIGN KEY (contract_id) REFERENCES CONTRACT(contract_id)
    );
    """)