"""
CourseMate — sample data generator.

Optional: creates a small set of synthetic course PDFs (syllabus, grading
policy, assignment guide, etc.) so you can try the app immediately without
your own documents. Real usage: drop your own course PDFs into data/ instead
and run ingest.py.

Usage:
    python generate_sample_data.py
"""

import os
from fpdf import FPDF
from fpdf.enums import XPos, YPos

import config

COURSE_NAME = "CS 301: Data Structures and Algorithms"


def clean(text: str) -> str:
    """Replace smart/special characters with plain ASCII equivalents (fpdf core fonts are Latin-1 only)."""
    return (
        text.replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2026", "...")
    )


def make_pdf(filename: str, title: str, sections: dict):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, clean(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(4)

    for heading, body in sections.items():
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, clean(heading), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, clean(body.strip()))
        pdf.ln(3)

    path = os.path.join(str(config.DATA_DIR), filename)
    pdf.output(path)
    print(f"  Created {path}")


make_pdf(
    "Syllabus.pdf",
    f"{COURSE_NAME} - Course Syllabus",
    {
        "Course Description": (
            "This course covers fundamental data structures (arrays, linked lists, "
            "trees, graphs, hash tables) and algorithms (sorting, searching, "
            "dynamic programming, graph traversal). Students will implement and "
            "analyze these structures in Python."
        ),
        "Schedule": (
            "Lectures: Monday and Wednesday, 10:00-11:15 AM, Room ENG-204.\n"
            "Lab section: Friday, 2:00-3:30 PM, Computer Lab 3.\n"
            "Office hours: Tuesday 1-3 PM, or by appointment."
        ),
        "Required Materials": (
            "Textbook: 'Introduction to Algorithms' (CLRS), 4th edition. "
            "A laptop capable of running Python 3.10+ is required for labs."
        ),
        "Prerequisites": (
            "CS 201 (Intro to Programming) and CS 210 (Discrete Mathematics), "
            "both with a grade of C or better."
        ),
    },
)

make_pdf(
    "Grading_Policy.pdf",
    f"{COURSE_NAME} - Grading Policy",
    {
        "Grade Breakdown": (
            "Homework assignments: 25%.\n"
            "Two midterm exams: 30% (15% each).\n"
            "Final exam: 30%.\n"
            "Lab participation: 15%."
        ),
        "Letter Grade Scale": (
            "A: 93-100, A-: 90-92, B+: 87-89, B: 83-86, B-: 80-82, "
            "C+: 77-79, C: 73-76, C-: 70-72, D: 60-69, F: below 60."
        ),
        "Late Work Policy": (
            "Assignments lose 10% per day late, up to 3 days. After 3 days, "
            "late work receives a zero unless a documented extension was "
            "approved in advance by the instructor."
        ),
        "Regrade Requests": (
            "Submit regrade requests within 7 days of a grade being posted, "
            "via email with a specific explanation of the disputed points. "
            "Requests submitted after 7 days will not be considered."
        ),
    },
)

make_pdf(
    "Assignment_Guidelines.pdf",
    f"{COURSE_NAME} - Assignment Submission Guidelines",
    {
        "Submission Format": (
            "All code must be submitted via the course GitHub Classroom link. "
            "Include a README.md describing your approach and how to run "
            "your code. Submissions without a README lose 5 points."
        ),
        "Academic Integrity": (
            "You may discuss approaches with classmates, but all code you "
            "submit must be written by you. Copying code from classmates, "
            "the internet, or AI tools without attribution is a violation "
            "of the academic integrity policy and will be reported."
        ),
        "Collaboration Policy": (
            "Pair programming is allowed only on assignments explicitly "
            "marked 'Group OK' on the assignment sheet. All other "
            "assignments are individual work."
        ),
        "Extensions": (
            "Email the instructor at least 24 hours before the deadline "
            "to request an extension. Extensions are granted for documented "
            "medical or family emergencies, generally up to 72 hours."
        ),
    },
)

make_pdf(
    "Exam_Information.pdf",
    f"{COURSE_NAME} - Exam Information",
    {
        "Midterm 1": (
            "Date: Week 6, in class. Covers arrays, linked lists, stacks, "
            "queues, and Big-O analysis. Closed book, one handwritten note "
            "card (both sides) allowed."
        ),
        "Midterm 2": (
            "Date: Week 11, in class. Covers trees, heaps, hash tables, and "
            "basic graph algorithms. Same note card policy as Midterm 1."
        ),
        "Final Exam": (
            "Cumulative, covers the entire course. Scheduled during finals "
            "week per the university exam calendar. Two note cards allowed."
        ),
        "Makeup Exam Policy": (
            "Makeup exams are only offered for documented emergencies "
            "communicated to the instructor before the exam date whenever "
            "possible. Unexcused absences from an exam result in a zero."
        ),
    },
)

make_pdf(
    "Office_Hours_and_Contact.pdf",
    f"{COURSE_NAME} - Office Hours and Contact Info",
    {
        "Instructor Contact": (
            "Email: instructor@university.edu (best for course questions; "
            "responses within 24-48 hours on weekdays).\n"
            "Office: ENG-310. Office hours: Tuesday 1-3 PM."
        ),
        "Teaching Assistants": (
            "TA office hours are posted on the course website and rotate "
            "weekly. TAs hold sessions in the Computer Lab 3 study area, "
            "Monday and Thursday evenings, 6-8 PM."
        ),
        "Discussion Forum": (
            "Use the course forum (linked on the course website) for "
            "general questions so other students can benefit from the "
            "answer. Do not post code solutions publicly before the "
            "deadline."
        ),
        "Emergency Contact": (
            "For urgent, time-sensitive issues (e.g., illness before an "
            "exam), email the instructor directly with 'URGENT' in the "
            "subject line."
        ),
    },
)

print("\nSample course PDFs created in data/. Now run: python ingest.py")
