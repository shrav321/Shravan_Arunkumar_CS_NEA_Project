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
      # CASH table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS CASH (
        balance REAL NOT NULL
    );
    """)

    # Ensure exactly one CASH row
    cur.execute("SELECT COUNT(*) FROM CASH;")
    row_count = cur.fetchone()[0]
    if row_count == 0:
        cur.execute("INSERT INTO CASH (balance) VALUES (?);", (0.0,))

    # PRICE table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS PRICE (
        ticker TEXT NOT NULL,
        date   TEXT NOT NULL,
        close  REAL NOT NULL,
        PRIMARY KEY (ticker, date)
    );
    """)

    conn.commit()
    conn.close()

def list_tables():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cur.fetchall()]
    conn.close()
    return tables

def insert_trade(contract_id, quantity, price, timestamp, side):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO TRADE (contract_id, quantity, price, timestamp, side)
        VALUES (?, ?, ?, ?, ?);
    """, (contract_id, quantity, price, timestamp, side))
    conn.commit()
    conn.close()