"""Generate sample_data/employee_handbook.pdf (fpdf2, no external assets)."""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

SECTIONS: list[tuple[str, str]] = [
    (
        "Welcome to Acme Corp",
        "This handbook summarises the policies that apply to all Acme Corp employees. "
        "It is reviewed annually by the People team. Where local law conflicts with "
        "this handbook, local law prevails.",
    ),
    (
        "Working Hours and Remote Work",
        "Standard working hours are 9:00 to 17:30, Monday to Friday, with core "
        "collaboration hours between 10:00 and 15:00. Employees may work remotely "
        "up to three days per week with manager approval. Fully remote arrangements "
        "require sign-off from the department head and HR.",
    ),
    (
        "Vacation Policy",
        "Full-time employees accrue 20 paid vacation days per calendar year, "
        "increasing by one day for each completed year of service up to a maximum "
        "of 25 days. Up to 5 unused days may be carried over into the first quarter "
        "of the following year. Vacation requests must be submitted in the HR portal "
        "at least two weeks in advance.",
    ),
    (
        "Sick Leave",
        "Employees receive 10 paid sick days per year. Absences longer than three "
        "consecutive days require a doctor's note. Unused sick days do not carry "
        "over and are not paid out.",
    ),
    (
        "Expense Policy",
        "Business travel must be pre-approved by your manager. Meal expenses are "
        "reimbursed up to 50 USD per person per day with itemised receipts. Flights "
        "and hotels must be booked through the corporate travel tool using the "
        "company card. Expense reports are due within 30 days of the trip.",
    ),
    (
        "Code of Conduct",
        "We treat colleagues, customers and partners with respect. Harassment, "
        "discrimination and retaliation are not tolerated. Concerns can be raised "
        "confidentially with HR or through the anonymous ethics hotline.",
    ),
]


def generate(path: Path) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_font("helvetica", "B", 20)
    pdf.cell(0, 12, "Acme Corp Employee Handbook", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 11)
    pdf.cell(0, 8, "Version 2.3 - internal use only", new_x="LMARGIN", new_y="NEXT")
    for index, (title, body) in enumerate(SECTIONS):
        if index and index % 2 == 0:  # spread sections over pages
            pdf.add_page()
        pdf.ln(6)
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 11)
        pdf.multi_cell(0, 6, body)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))
    print(f"wrote {path}")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parent.parent / "sample_data" / "employee_handbook.pdf")
