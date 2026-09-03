# 🤖 Autonomous AI HR Resignation Assistant

An end-to-end intelligent HR agent workflow that processes employee resignation emails, queries corporate database records, performs Policy RAG against company separation handbooks, enforces Human-in-the-Loop (HITL) HR Manager approval, and automates official letter dispatches.

---

## 🏗️ 5-Step Agentic Architecture

```
[ Step 1: Email Ingestion ]
      │  • Extracts sender, resignation intent, dates, and sentiment.
      ▼
[ Step 2: Employee DB Lookup ]
      │  • Queries SQLite HR DB for join date, title, dept, manager.
      │  • Computes exact tenure (Probation / Standard / Tenured).
      ▼
[ Step 3: Policy RAG & LWD Engine ]
      │  • Retrieves applicable clauses from Employee Handbook.
      │  • Calculates official Last Working Day (LWD = Submission + Notice Days).
      ▼
[ Step 4: Human-in-the-Loop Review Gate ]
      │  • Displays Slack/Email style HR Manager decision card.
      │  • Options: [1] Approve Policy LWD, [2] Override/Early Exit, [3] Reject/Hold.
      ▼
[ Step 5: Post-Approval Dispatch & Audit ]
         • Formal Employee Resignation Acceptance Letter.
         • Line Manager Handover & KT Briefing.
         • SQLite Audit Log update.
```

---

## 🚀 Quickstart & Usage

### 1. Initialize the HR Database
```bash
uv run python setup_hr_db.py
```

### 2. Run Automated Multi-Scenario Demo Suite
Runs real-world test cases (e.g., 2.5-year tenured engineer, probationary designer, etc.):
```bash
uv run python resignation-agent.py
```

### 3. Run Interactive CLI Mode
Allows custom email pasting or selecting pre-built employee scenarios with live Human-in-the-Loop approval:
```bash
uv run python resignation-agent.py --interactive
```

---

## 📁 Project Structure
- [`resignation-agent.py`](file:///c:/Users/ATHER/Desktop/Agentic_AI%20new/resignation-agent.py): Core agentic pipeline, LangGraph/Gemini AI tools, and HITL gate.
- [`company_policy_handbook.md`](file:///c:/Users/ATHER/Desktop/Agentic_AI%20new/company_policy_handbook.md): Company separation policy clauses, notice period matrices, and handover rules.
- [`setup_hr_db.py`](file:///c:/Users/ATHER/Desktop/Agentic_AI%20new/setup_hr_db.py): SQLite database schema & mock employee data loader.
- [`hr_company.db`](file:///c:/Users/ATHER/Desktop/Agentic_AI%20new/hr_company.db): Active SQLite database (`employees` and `resignation_logs` tables).
