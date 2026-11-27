"""Local Excel storage for the original CoSec app (no SQL server required)."""
import os
import re
import json
import difflib
import tempfile
import shutil
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from filelock import FileLock
from openpyxl import Workbook, load_workbook

ROOT = Path(__file__).resolve().parent
FILE = Path(os.getenv("COSEC_EXCEL_PATH", str(ROOT / "CoSec_Task_Management_Sheet.xlsx"))).expanduser().resolve()
SHEETS = {
    "Companies": ["company_id", "company_name", "added_at"],
    "Categories": ["category_id", "category_name", "category_desc"],
    "Tasks": ["task_id", "company", "category", "date_received", "raw_message", "created_by", "created_at", "fee_amount"],
    "Task_Counter": ["counter_date", "last_no"],
    "Audit_Log": ["task_id", "entry_no", "action_type", "old_value", "new_value", "audited_by", "audited_at"],
}

def _safe(value):
    # Avoid Excel interpreting untrusted message text as a formula or DDE link.
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value

def _rows(ws):
    rows = list(ws.values)
    if not rows:
        return []
    return [dict(zip(rows[0], row)) for row in rows[1:] if any(v is not None for v in row)]

def _save(wb):
    FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(suffix=".xlsx", dir=FILE.parent)
    os.close(fd)
    try:
        wb.save(temp)
        os.replace(temp, FILE)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def check_connection():
    """Validate workbook headers, ignoring completely empty trailing columns."""
    with FileLock(str(FILE) + ".lock", timeout=15):
        if not FILE.exists():
            raise FileNotFoundError(
                f"Workbook not found: {FILE}. "
                "Restore the workbook or set COSEC_EXCEL_PATH."
            )

        wb = load_workbook(FILE)
        try:
            for name, headers in SHEETS.items():
                if name not in wb.sheetnames:
                    raise ValueError(f"Missing Excel sheet {name}.")

                ws = wb[name]

                # Check only the columns defined in SHEETS
                actual = [
                    ws.cell(1, i).value
                    for i in range(1, len(headers) + 1)
                ]

                if actual != headers:
                    raise ValueError(
                        f"Unexpected columns in {name}; "
                        f"found {actual}; expected {headers}."
                    )

                # Reject extra columns if they contain actual data
                for col in range(len(headers) + 1, ws.max_column + 1):
                    for row in range(1, ws.max_row + 1):
                        if ws.cell(row, col).value is not None:
                            raise ValueError(
                                f"Unexpected data in {name}, "
                                f"row {row}, column {col}."
                            )
        finally:
            wb.close()

    return True


@contextmanager
def _book(write=False):
    check_connection()
    # This lock serializes writes and avoids simultaneous reads of an incomplete file.
    with FileLock(str(FILE) + ".lock", timeout=15):
        wb = load_workbook(FILE)
        try:
            yield wb
            if write:
                _save(wb)
        finally:
            wb.close()

def get_categories():
    with _book() as wb:
        return [r["category_name"] for r in _rows(wb["Categories"])]

def get_companies():
    with _book() as wb:
        return sorted([r["company_name"] for r in _rows(wb["Companies"])], key=str.lower)

# ---------- company name matching ----------
def _norm(name):
    n = name.lower()
    n = re.sub(r"[.,]", "", n)
    n = re.sub(r"\b(sdn|bhd|berhad|pte|ltd|limited)\b", "", n)
    return re.sub(r"\s+", " ", n).strip()


def company_base(name):
    """Normalise punctuation and an optional SDN BHD suffix."""
    name = str(name or "").casefold()
    name = re.sub(r"[^a-z0-9]+", " ", name).strip()
    name = re.sub(r"\s+sdn bhd$", "", name)
    return " ".join(name.split())


def resolve_company(name, known_companies):
    """
    Return the registered name only when the match is unique.
    Never guess between different legal entities.
    """
    if not name:
        return None

    key = company_base(name)
    matches = [
        company for company in known_companies
        if company_base(company) == key
    ]

    if len(matches) == 1:
        return matches[0]

    return None

def match_company(name, threshold=0.85):
    if not name:
        return name
    best, score = None, 0.0
    for existing in get_companies():
        s = difflib.SequenceMatcher(None, _norm(name), _norm(existing)).ratio()
        if s > score:
            best, score = existing, s
    return best if best and score >= threshold else name

def _date(value):
    if value is None or str(value).strip().lower() in ("", "none", "null"):
        return None

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    return date.fromisoformat(str(value).strip()[:10])

def _timestamp(value):
    if value is None or str(value).strip().lower() in ("", "none", "null"):
        return None

    if isinstance(value, datetime):
        return value

    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    
    return datetime.fromisoformat(str(value))

def list_tasks(company=None):
    with _book() as wb:
        records = []
        for r in _rows(wb["Tasks"]):
            task_id = r.get("task_id")
            if task_id is None or not str(task_id).strip():
                continue
            name = r.get("company")
            if company and name != company:
                continue
            records.append({"task_id":task_id, "company":name,
                            "date_received":_date(r.get("date_received")),
                            "category":r.get("category"), "created_by":r.get("created_by"),
                            "created_at":_timestamp(r.get("created_at")),
                            "fee_amount":r.get("fee_amount")})
        return sorted(records, key=lambda r: (r["date_received"] or date.min, str(r["task_id"] or "")), reverse=True)


def get_task(task_id):
    with _book() as wb:
        for r in _rows(wb["Tasks"]):
            if r["task_id"] == task_id:
                return {"task_id":task_id, "company":r.get("company"),
                        "category":r.get("category"), "date_received":_date(r.get("date_received")),
                        "raw_message":r.get("raw_message"), "created_by":r.get("created_by"),
                        "created_at":_timestamp(r.get("created_at")), "fee_amount":r.get("fee_amount")}
    raise ValueError(f"Task {task_id} was not found")

def get_audit(task_id):
    with _book() as wb:
        result = []
        for r in _rows(wb["Audit_Log"]):
            if r["task_id"] == task_id:
                result.append({"entry_no":int(r["entry_no"]),
                               "action_type":r["action_type"],
                               "old_value":json.loads(r["old_value"]) if r["old_value"] else None,
                               "new_value":json.loads(r["new_value"]), "audited_by":r["audited_by"],
                               "audited_at":_timestamp(r["audited_at"])})
        return sorted(result, key=lambda r:r["entry_no"], reverse=True)

def _to_date(value, field):
    if value in (None, ""):
        raise ValueError(f"'{field}' is required.")
    try:
        return _date(value)
    except (ValueError, TypeError):
        raise ValueError(f"'{field}' must be in YYYY-MM-DD format (got '{value}').")

def _optional_fee(value):
    """A missing fee is unknown (None), not zero. Amounts use two decimal places."""
    from decimal import Decimal, InvalidOperation
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ValueError("Fee must be a valid non-negative amount or blank.")
    cleaned = str(value).strip().replace(",", "")
    cleaned = re.sub(r"^(?:RM|MYR)\s*", "", cleaned, flags=re.I)
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        raise ValueError(f"Invalid fee '{value}'; enter an amount or leave it blank.")
    if not number.is_finite() or number < 0 or number.as_tuple().exponent < -2:
        raise ValueError("Fee must be non-negative with at most two decimal places, or blank.")
    return float(number)


# ---------- duplicate detection ----------
def _dup_key(company, category, received):
    # Punctuation and whitespace differences in a registered company should not
    # hide a duplicate. Categories compare case-insensitively.
    return (
        company_base(company),
        " ".join(str(category or "").casefold().split()),
        _to_date(received, "date_received").isoformat(),
    )


def _source_key(value):
    # This is intentionally conservative: semantically similar but differently
    # worded instructions are *possible* duplicates, not confirmed duplicates.
    return " ".join(str(value or "").casefold().split())


def _find_duplicates_in_workbook(wb, tasks, raw_message):
    """Return potential duplicates in Excel and within the incoming batch.

    The match is per company + category + received date. Matching source text
    further identifies an exact repeated submission. Does not modify workbook.
    """
    existing = {}
    for row in _rows(wb["Tasks"]):
        if not row.get("task_id") or not row.get("company") or not row.get("category") or not row.get("date_received"):
            continue
        key = _dup_key(row["company"], row["category"], row["date_received"])
        existing.setdefault(key, []).append(row)

    seen_batch = {}
    matches = []
    for index, task in enumerate(tasks, 1):
        name, cat, received = task.get("company"), task.get("category"), task.get("date_received")
        if not name or not cat or not received:
            continue
        key = _dup_key(name, cat, received)
        for old in existing.get(key, []):
            exact = _source_key(old.get("raw_message")) == _source_key(raw_message)
            matches.append({
                "row": index,
                "company": str(name), "category": str(cat),
                "date_received": key[2], "task_id": str(old["task_id"]),
                "kind": "exact" if exact else "possible",
                "source": "existing",
            })
        if key in seen_batch:
            matches.append({
                "row": index, "company": str(name), "category": str(cat),
                "date_received": key[2], "task_id": None,
                "kind": "exact", "source": "batch",
                "first_row": seen_batch[key],
            })
        else:
            seen_batch[key] = index
    return matches


def find_duplicate_tasks(tasks, raw_message):
    """Read-only preview for the confirmation screen."""
    with _book() as wb:
        return _find_duplicates_in_workbook(wb, tasks, raw_message)


def insert_tasks(tasks, raw_message, user, approve_new_companies=False, approve_new_categories=False, allow_duplicates=False):
    """Save all tasks atomically; validation failure leaves workbook unchanged."""
    if not str(user).strip():
        raise ValueError("Uploader name is required.")
    with _book(write=True) as wb:
        cats = {str(r["category_name"]).casefold():r for r in _rows(wb["Categories"])}
        company_rows = [
            r for r in _rows(wb["Companies"])
            if r.get("company_name")
        ]

        existing_companies = {
            str(r["company_name"]).strip().casefold(): r
            for r in company_rows
        }
        counters = {str(r["counter_date"])[:10]:int(r["last_no"]) for r in _rows(wb["Task_Counter"])}
        new_companies, created = {}, []
        staged = []
        new_categories = {}
        # The final duplicate check happens inside the same write lock as
        # insertion, so simultaneous users cannot bypass the preview check.
        duplicates = _find_duplicates_in_workbook(wb, tasks, raw_message)
        if duplicates and not allow_duplicates:
            summary = "; ".join(
                f"row {d['row']} ({d['company']} / {d['category']}, {d['date_received']})"
                + (f" matches task {d['task_id']}" if d['task_id'] else f" repeats row {d['first_row']}")
                for d in duplicates[:5]
            )
            raise ValueError("Possible duplicate task(s) detected: " + summary
                             + ". Review existing records or explicitly confirm 'Save duplicates anyway'.")
        # These approvals are explicit confirmation-screen choices, not AI decisions.
        for t in tasks:
            name = str(t.get("company") or "").strip()
            category = str(t.get("category") or "").strip()
            if not name:
                raise ValueError("Company is required.")
            if not category:
                raise ValueError(f"Category is required (company: {name}).")
            cat = cats.get(category.casefold()) or new_categories.get(category.casefold())
            if not cat:
                if not approve_new_categories:
                    raise ValueError(f"New category '{category}' needs explicit approval before saving.")
                ids = [int(r["category_id"]) for r in list(cats.values())+list(new_categories.values())]
                cat = {"category_id": max(ids, default=0)+1, "category_name": category}
                new_categories[category.casefold()] = cat
            received = _to_date(t.get("date_received"), "date_received")
            fee_amount = _optional_fee(t.get("fee_amount"))
            key = received.isoformat()
            next_no = counters.get(key,0)+1
            if next_no > 999:
                raise ValueError(f"More than 999 tasks for {key}: task ID limit reached.")
            counters[key] = next_no
            # Match abbreviated company names to registered names.
            matched = resolve_company(
                name,
                [r["company_name"] for r in company_rows]
            )

            if matched:
                name = matched

            company_key = name.casefold()

            comp = (
                existing_companies.get(company_key)
                or new_companies.get(company_key)
            )
            if comp is None:
                if not approve_new_companies:
                    raise ValueError(f"New company '{name}' needs explicit approval before saving.")
                next_company_id = len(existing_companies)+len(new_companies)+1
                # Use max ID to preserve uniqueness if users removed spreadsheet rows.
                ids = [int(r["company_id"]) for r in list(existing_companies.values())+list(new_companies.values())]
                next_company_id = max(ids,default=0)+1
                comp = {"company_id":next_company_id,"company_name":name}
                new_companies[company_key] = comp
            tid = f"T{received:%y%m%d}{next_no:03d}"
            created.append(tid)
            staged.append((tid,comp,cat,received,fee_amount))
        now = datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).replace(tzinfo=None)
        for comp in new_companies.values():
            wb["Companies"].append([comp["company_id"], _safe(comp["company_name"]), now])
        for cat in new_categories.values():
            wb["Categories"].append([cat["category_id"], _safe(cat["category_name"]), None])
        for tid,comp,cat,received,fee_amount in staged:
            task_ws = wb["Tasks"]
            # Preformatted workbook templates can contain 1,000 empty task rows.
            # Excel append() would place the new record after those empty rows.
            target_row = next(
                (i for i in range(2, task_ws.max_row + 1)
                 if all(task_ws.cell(i, c).value is None for c in range(1, 9))),
                task_ws.max_row + 1,
            )
            values = [tid, _safe(comp["company_name"]), _safe(cat["category_name"]), received,
                      _safe(raw_message), _safe(user.strip()), now, fee_amount]
            for col, value in enumerate(values, 1):
                task_ws.cell(target_row, col).value = value
            task_ws.cell(target_row, 8).number_format = '"RM "#,##0.00' 
            snapshot = {"company":comp["company_name"], "category":cat["category_name"],
                        "date_received":received.isoformat(), "fee_amount":fee_amount}
            wb["Audit_Log"].append([tid,1,"CREATE",None,
                                    json.dumps(snapshot,ensure_ascii=False),_safe(user.strip()),now])
        # Rewrite counter sheet in memory; no disk change until all stages succeed.
        ws=wb["Task_Counter"]
        if ws.max_row > 1:
            ws.delete_rows(2, ws.max_row-1)
        for d,n in sorted(counters.items()):
            ws.append([d,n])
        return created


def update_task(task_id, changes, user, note="", expected=None):
    """Update a task and append a detailed audit entry, in one locked workbook save.

    Only existing companies/categories are selectable. expected holds the values
    seen by the editor and prevents silently overwriting someone else's changes.
    """
    if not str(user).strip():
        raise ValueError("Your name is required to update a task.")
    note = str(note or "").strip()
    if len(note) > 2000:
        raise ValueError("Update note must be 2000 characters or fewer.")
    permitted = {"company", "category", "date_received", "fee_amount"}
    if set(changes) - permitted:
        raise ValueError("Unsupported update fields: " + ", ".join(sorted(set(changes) - permitted)))
    with _book(write=True) as wb:
        ws = wb["Tasks"]
        companies = {str(r["company_name"]).strip().casefold(): r for r in _rows(wb["Companies"])}
        categories = {str(r["category_name"]).strip().casefold(): r for r in _rows(wb["Categories"])}
        found_row = None
        for row_num in range(2, ws.max_row + 1):
            if str(ws.cell(row_num, 1).value) == str(task_id):
                found_row = row_num
                break
        if found_row is None:
            raise ValueError(f"Task {task_id} was not found.")
        row = found_row
        original = {
            "company": ws.cell(row, 2).value,
            "category": ws.cell(row, 3).value,
            "date_received": _date(ws.cell(row, 4).value).isoformat(),
            "fee_amount": ws.cell(row, 8).value,
        }
        if expected is not None:
            old_expected = dict(expected)
            if "date_received" in old_expected and old_expected["date_received"] is not None:
                old_expected["date_received"] = _to_date(old_expected["date_received"], "date_received").isoformat()
            for key in permitted:
                if key in old_expected and old_expected[key] != original[key]:
                    raise ValueError("This task changed since you opened it. Refresh and review before saving.")
        updated = original.copy()
        if "company" in changes:
            value = str(changes["company"] or "").strip()
            match = companies.get(value.casefold())
            if not match:
                raise ValueError("Choose an existing company from the list.")
            updated["company"] = match["company_name"]
        if "category" in changes:
            value = str(changes["category"] or "").strip()
            match = categories.get(value.casefold())
            if not match:
                raise ValueError("Choose an existing category from the list.")
            updated["category"] = match["category_name"]
        if "date_received" in changes:
            updated["date_received"] = _to_date(changes["date_received"], "date_received").isoformat()
        if "fee_amount" in changes:
            updated["fee_amount"] = _optional_fee(changes["fee_amount"])
        changed_keys = [key for key in ("company", "category", "date_received", "fee_amount") if updated[key] != original[key]]
        if not changed_keys and not note:
            raise ValueError("Change a field or enter an update note before saving.")
        if "company" in changed_keys:
            ws.cell(row, 2, _safe(updated["company"]))
        if "category" in changed_keys:
            ws.cell(row, 3, _safe(updated["category"]))
        if "date_received" in changed_keys:
            ws.cell(row, 4, _to_date(updated["date_received"], "date_received"))
        if "fee_amount" in changed_keys:
            ws.cell(row, 8).value = updated["fee_amount"]
        # Append a full before/after snapshot so the history is understandable.
        old_snapshot = dict(original)
        new_snapshot = dict(updated)
        if note:
            new_snapshot["update_note"] = note
        entries = [int(r["entry_no"]) for r in _rows(wb["Audit_Log"]) if r["task_id"] == task_id]
        entry_no = max(entries, default=0) + 1
        now = datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).replace(tzinfo=None)
        wb["Audit_Log"].append([
            task_id, entry_no, "UPDATE",
            json.dumps(old_snapshot, ensure_ascii=False),
            json.dumps(new_snapshot, ensure_ascii=False),
            _safe(user.strip()), now,
        ])
        return entry_no
