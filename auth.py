import re
import sqlite3
from contextlib import closing
from typing import Optional

DB_PATH = "users.db"
MIN_PASSWORD_LENGTH = 8
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ------------------------------------------------------------------ #
# Database
# ------------------------------------------------------------------ #

def _connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
                email         TEXT NOT NULL UNIQUE COLLATE NOCASE,
                full_name     TEXT NOT NULL,
                password      TEXT NOT NULL,
                created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def register_user(
    username: str, email: str, full_name: str, password: str, db_path: str = DB_PATH
) -> tuple[bool, str]:
    """Create a new user. Returns (success, message)."""
    username, email, full_name = username.strip(), email.strip(), full_name.strip()

    if not USERNAME_PATTERN.match(username):
        return False, "Username must be 3-32 characters (letters, digits, _ . -)."
    if not EMAIL_PATTERN.match(email):
        return False, "Please enter a valid email address."
    if not full_name:
        return False, "Full name is required."
    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters."

    try:
        with closing(_connect(db_path)) as conn, conn:
            conn.execute(
                "INSERT INTO users (username, email, full_name, password) "
                "VALUES (?, ?, ?, ?)",
                (username, email, full_name, password),
            )
    except sqlite3.IntegrityError:
        return False, "Username or email is already registered."

    return True, "Registration successful. Please log in."


def authenticate(username: str, password: str, db_path: str = DB_PATH) -> Optional[dict]:
    """Return the user's details if the credentials are valid, else None."""
    with closing(_connect(db_path)) as conn:
        row = conn.execute(
            "SELECT id, username, email, full_name, password "
            "FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()

    if row is None:
        return None
    if password != row["password"]:
        return None

    return {
        "id": row["id"],
        "username": row["username"],
        "email": row["email"],
        "full_name": row["full_name"],
    }
