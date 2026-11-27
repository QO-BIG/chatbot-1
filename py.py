
# from openpyxl import load_workbook
# from pathlib import Path
# import shutil
# from datetime import datetime

# # Use the workbook path displayed in your Streamlit sidebar.
# path = Path(r"C:\Users\Owner\Downloads\chatbot-main\chatbot-main\CoSec_Task_Management_Sheet.xlsx")

# if not path.exists():
#     raise FileNotFoundError(f"Workbook not found: {path}")

# # Create a uniquely named backup before modifying anything.
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# backup = path.with_name(
#     f"{path.stem}_backup_{timestamp}.xlsx"
# )

# shutil.copy2(path, backup)
# print("Backup created:", backup)

# wb = load_workbook(path)
# ws = wb["Tasks"]

# headers = [cell.value for cell in ws[1]]
# print("BEFORE:", headers)

# if "status" in headers:
#     col = headers.index("status") + 1
#     ws.delete_cols(col)
#     print("Removed status column.")
# else:
#     print("No status column found.")

# expected = [
#     "task_id",
#     "company_id",
#     "category_id",
#     "date_received",
#     "raw_message",
#     "created_by",
#     "created_at",
#     "fee_amount",
# ]

# after = [cell.value for cell in ws[1]]
# print("AFTER:", after)

# if after != expected:
#     print("WARNING: Headers still do not match the expected structure.")
#     print("No changes saved. Inspect the printed headers.")
# else:
#     wb.save(path)
#     print("SUCCESS: Workbook saved with correct columns.")

# wb.close()



# from openpyxl import load_workbook
# from pathlib import Path
# from datetime import datetime
# import shutil

# path = Path(
#     r"C:\Users\Owner\Downloads\chatbot-main\chatbot-main\CoSec_Task_Management_Sheet.xlsx"
# )

# expected = [
#     "task_id", "company_id", "category_id",
#     "date_received", "raw_message", "created_by",
#     "created_at", "fee_amount"
# ]

# # Back up the workbook first
# backup = path.with_name(
#     f"{path.stem}_backup_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
# )
# shutil.copy2(path, backup)

# wb = load_workbook(path)
# ws = wb["Tasks"]

# headers = [ws.cell(1, i).value for i in range(1, 9)]

# if headers != expected:
#     print("First eight columns don't match. No changes saved.")

# else:
#     # Check that extra column I contains only old status values
#     allowed = {
#         None, "", "Pending", "In Progress",
#         "On Hold", "Completed", "Cancelled"
#     }

#     unexpected = [
#         (row, ws.cell(row, 9).value)
#         for row in range(1, ws.max_row + 1)
#         if ws.cell(row, 9).value not in allowed
#     ]

#     if unexpected:
#         print("Unexpected data found in column I.")
#         print("Examples:", unexpected[:10])
#         print("No changes saved.")
#     else:
#         ws.delete_cols(9)
#         wb.save(path)
#         print("Status column I removed successfully.")

# wb.close()
# print("Backup:", backup)

"""Compact blank task rows safely. Does not change other worksheets.

Run: python repair_tasks_sheet.py
Optional: python repair_tasks_sheet.py --file path/to/CoSec_Task_Management_Sheet.xlsx

Stops if Audit_Log has CREATE entries without matching Tasks rows; see report.
Never creates reconstructed tasks silently.
"""
import argparse
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from filelock import FileLock
from openpyxl import load_workbook

EXPECTED = ['task_id','company','category','date_received','raw_message','created_by','created_at','fee_amount']
OLD = ['task_id','company_id','category_id','date_received','raw_message','created_by','created_at','fee_amount']


def repair(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Workbook not found: {path}')
    with FileLock(str(path)+'.lock', timeout=15):
        wb = load_workbook(path)
        try:
            if 'Tasks' not in wb.sheetnames or 'Audit_Log' not in wb.sheetnames:
                raise ValueError('Workbook must contain Tasks and Audit_Log')
            ws = wb['Tasks']
            headers = [ws.cell(1,i).value for i in range(1,9)]
            if headers not in (EXPECTED, OLD):
                raise ValueError(f'Unknown Tasks headers: {headers}. No changes made.')
            # Exactly identify the real rows. Copy cell values and number formats;
            # rewrite only the Tasks sheet, not any other worksheets.
            task_rows = []
            for row in range(2, ws.max_row+1):
                vals = [ws.cell(row,col).value for col in range(1,9)]
                if vals[0] not in (None,''):
                    task_rows.append((vals, [ws.cell(row,col).number_format for col in range(1,9)]))
                elif any(v is not None for v in vals):
                    raise ValueError(f'Row {row} has data but no task_id; aborting for manual review')
            existing = [str(x[0][0]) for x in task_rows]
            if len(existing)!=len(set(existing)):
                raise ValueError('Duplicate Task IDs found; no changes made.')
            audit_ids = {str(r[0]) for r in wb['Audit_Log'].iter_rows(min_row=2, values_only=True)
                         if r[0] is not None and str(r[3]).upper()=='CREATE'}
            missing = sorted(audit_ids-set(existing))
            first_old = next((i for i in range(2,ws.max_row+1) if ws.cell(i,1).value not in (None,'')),None)
            blanks = ws.max_row-1-len(task_rows)
            print(f'File: {path}')
            print(f'Existing Tasks records: {len(task_rows)}; blank rows: {blanks}; first task Excel row: {first_old}')
            print(f'Audit CREATE entries with no corresponding task: {len(missing)}')
            if missing:
                print('Missing task IDs:', ', '.join(missing[:30]))
                print('No data changed. These need separate investigation; audit snapshots may allow partial recovery.')
                return False
            if not blanks:
                print('Tasks are already contiguous. No changes needed.')
                return True
            stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
            backup=path.with_name(f'{path.stem}_backup_before_task_repair_{stamp}{path.suffix}')
            shutil.copy2(path,backup)
            # Rebuild values in the same worksheet, preserving the other sheets.
            ws.delete_rows(2, ws.max_row-1)
            for idx,(vals,number_formats) in enumerate(task_rows,2):
                for col,value in enumerate(vals,1):
                    cell=ws.cell(idx,col)
                    cell.value=value
                    cell.number_format=number_formats[col-1]
            ws.freeze_panes = 'A2'
            ws.auto_filter.ref=f'A1:H{max(2,len(task_rows)+1)}'
            fd,temp=tempfile.mkstemp(dir=path.parent, suffix='.xlsx');os.close(fd)
            try:
                wb.save(temp)
                os.replace(temp,path)
            finally:
                if os.path.exists(temp):os.unlink(temp)
            print(f'REPAIRED: task rows now start at Excel row 2. Backup saved: {backup}')
            return True
        finally:
            wb.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--file',default=os.getenv('COSEC_EXCEL_PATH','CoSec_Task_Management_Sheet.xlsx'))
    args=p.parse_args()
    repair(args.file)
