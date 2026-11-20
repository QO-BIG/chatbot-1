
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



from openpyxl import load_workbook
from pathlib import Path
from datetime import datetime
import shutil

path = Path(
    r"C:\Users\Owner\Downloads\chatbot-main\chatbot-main\CoSec_Task_Management_Sheet.xlsx"
)

expected = [
    "task_id", "company_id", "category_id",
    "date_received", "raw_message", "created_by",
    "created_at", "fee_amount"
]

# Back up the workbook first
backup = path.with_name(
    f"{path.stem}_backup_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
)
shutil.copy2(path, backup)

wb = load_workbook(path)
ws = wb["Tasks"]

headers = [ws.cell(1, i).value for i in range(1, 9)]

if headers != expected:
    print("First eight columns don't match. No changes saved.")

else:
    # Check that extra column I contains only old status values
    allowed = {
        None, "", "Pending", "In Progress",
        "On Hold", "Completed", "Cancelled"
    }

    unexpected = [
        (row, ws.cell(row, 9).value)
        for row in range(1, ws.max_row + 1)
        if ws.cell(row, 9).value not in allowed
    ]

    if unexpected:
        print("Unexpected data found in column I.")
        print("Examples:", unexpected[:10])
        print("No changes saved.")
    else:
        ws.delete_cols(9)
        wb.save(path)
        print("Status column I removed successfully.")

wb.close()
print("Backup:", backup)
