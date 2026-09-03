import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "hr_company.db")

def init_db():
    """Initializes the HR Company SQLite database with sample employee records."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed existing database at {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create Employees Table
    cursor.execute("""
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        job_title TEXT NOT NULL,
        department TEXT NOT NULL,
        joining_date TEXT NOT NULL,
        manager_name TEXT NOT NULL,
        manager_email TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'ACTIVE'
    )
    """)

    # Create Resignation Records Table for audit logging
    cursor.execute("""
    CREATE TABLE resignation_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT NOT NULL,
        employee_email TEXT NOT NULL,
        submission_date TEXT NOT NULL,
        requested_lwd TEXT,
        calculated_lwd TEXT NOT NULL,
        applicable_clause TEXT NOT NULL,
        notice_days INTEGER NOT NULL,
        hr_status TEXT NOT NULL,
        hr_reviewer_notes TEXT,
        confirmed_lwd TEXT NOT NULL,
        logged_at TEXT NOT NULL,
        FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
    )
    """)

    sample_employees = [
        (
            "EMP-1001",
            "Jane Doe",
            "jane.doe@company.com",
            "Senior Software Engineer",
            "Core Platform Engineering",
            "2024-03-01",  # ~2.5 years tenure
            "Marcus Vance",
            "marcus.vance@company.com",
            "ACTIVE"
        ),
        (
            "EMP-1002",
            "Alex Rivera",
            "alex.rivera@company.com",
            "Junior UI/UX Designer",
            "Product Design",
            "2026-05-15",  # ~3.5 months tenure (Probation)
            "Clara Oswald",
            "clara.oswald@company.com",
            "ACTIVE"
        ),
        (
            "EMP-1003",
            "Priya Sharma",
            "priya.sharma@company.com",
            "Product Manager",
            "Growth & Engagement",
            "2025-06-01",  # ~1.25 years tenure (Standard 6m - 2y)
            "David Miller",
            "david.miller@company.com",
            "ACTIVE"
        ),
        (
            "EMP-1004",
            "Marcus Vance",
            "marcus.vance@company.com",
            "Engineering Director",
            "Engineering Leadership",
            "2023-01-10",  # Executive / Director
            "Elena Rostova (VP Engineering)",
            "elena.rostova@company.com",
            "ACTIVE"
        ),
        (
            "EMP-1005",
            "Emily Watson",
            "emily.watson@company.com",
            "Lead Data Scientist",
            "Applied AI & Machine Learning",
            "2024-01-15",  # ~2.6 years tenure
            "Marcus Vance",
            "marcus.vance@company.com",
            "ACTIVE"
        )
    ]

    cursor.executemany("""
    INSERT INTO employees (
        employee_id, name, email, job_title, department, joining_date, manager_name, manager_email, status
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, sample_employees)

    conn.commit()
    conn.close()
    print(f"Successfully initialized '{DB_PATH}' with {len(sample_employees)} sample employees.")

if __name__ == "__main__":
    init_db()
