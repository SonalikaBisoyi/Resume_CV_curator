"""
PDF Export — generates nicely formatted Resume and Cover Letter PDFs
using reportlab. Called from app.py after the agent finishes.
"""

import re
import tempfile
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# ─── COLOUR PALETTE ───────────────────────────────────────────────────────────
ACCENT   = colors.HexColor("#7c6af7")   # purple accent
DARK     = colors.HexColor("#1a1a2e")   # near-black for headings
BODY     = colors.HexColor("#2d2d3a")   # body text
MUTED    = colors.HexColor("#666680")   # subtext / rules


# ─── SHARED STYLES ────────────────────────────────────────────────────────────

def _base_styles():
    s = getSampleStyleSheet()

    def add(name, **kwargs):
        if name not in s:
            s.add(ParagraphStyle(name, **kwargs))

    add("DocTitle",    fontName="Helvetica-Bold",    fontSize=20,   textColor=DARK,   spaceAfter=4,   alignment=TA_CENTER)
    add("ContactLine", fontName="Helvetica",          fontSize=9,    textColor=MUTED,  spaceAfter=14,  alignment=TA_CENTER)
    add("SectionHead", fontName="Helvetica-Bold",    fontSize=10,   textColor=ACCENT, spaceBefore=12, spaceAfter=3, letterSpacing=1.2)
    add("JobTitle",    fontName="Helvetica-Bold",    fontSize=10,   textColor=DARK,   spaceBefore=6,  spaceAfter=1)
    add("JobMeta",     fontName="Helvetica-Oblique", fontSize=9,    textColor=MUTED,  spaceAfter=3)
    add("Bullet",      fontName="Helvetica",          fontSize=9.5,  textColor=BODY,   leftIndent=14,  spaceAfter=2, bulletIndent=4)
    add("BodyText2",   fontName="Helvetica",          fontSize=9.5,  textColor=BODY,   spaceAfter=4,   leading=14)
    add("CLBody",      fontName="Helvetica",          fontSize=10.5, textColor=BODY,   spaceAfter=10,  leading=16)
    add("CLDate",      fontName="Helvetica",          fontSize=10,   textColor=MUTED,  spaceAfter=16)

    return s


# ─── RESUME PDF ───────────────────────────────────────────────────────────────

def _rule(width=6.5*inch):
    return HRFlowable(width=width, thickness=0.5, color=ACCENT, spaceAfter=4, spaceBefore=2)


def _parse_resume(text: str, styles) -> list:
    """
    Heuristic parser: turns plain-text resume into reportlab flowables.
    Handles most common resume formats.
    """
    story = []
    lines = [l.rstrip() for l in text.splitlines()]

    # First non-blank line → name
    name_added = False
    contact_lines = []
    body_started = False
    section_keywords = {"experience", "education", "skills", "projects",
                        "certifications", "summary", "objective", "awards",
                        "publications", "languages", "interests", "work"}

    i = 0
    while i < len(lines):
        line = lines[i]

        # skip blanks after contact block
        if not line.strip():
            i += 1
            continue

        stripped = line.strip()
        low = stripped.lower()

        # ── Name (first meaningful line) ──────────────────────────────────────
        if not name_added:
            story.append(Paragraph(stripped, styles["DocTitle"]))
            name_added = True
            i += 1
            # collect contact lines (email/phone/url on the next 1-3 lines)
            while i < len(lines) and lines[i].strip() and \
                  not any(kw in lines[i].lower() for kw in section_keywords):
                contact_lines.append(lines[i].strip())
                i += 1
            if contact_lines:
                story.append(Paragraph(" · ".join(contact_lines), styles["ContactLine"]))
            story.append(_rule())
            continue

        # ── Section heading ───────────────────────────────────────────────────
        is_section = (
            (stripped.isupper() and len(stripped) < 40) or
            (any(kw == low.rstrip(":") for kw in section_keywords)) or
            (stripped.endswith(":") and len(stripped) < 35 and stripped[:-1].lower() in section_keywords)
        )
        if is_section:
            story.append(Paragraph(stripped.upper().rstrip(":"), styles["SectionHead"]))
            story.append(_rule())
            i += 1
            continue

        # ── Bullet points ─────────────────────────────────────────────────────
        if stripped.startswith(("•", "-", "–", "*", "·")):
            bullet_text = stripped.lstrip("•-–*· ").strip()
            story.append(Paragraph(f"• {bullet_text}", styles["Bullet"]))
            i += 1
            continue

        # ── Lines that look like "Job | Company | Date" ───────────────────────
        if "|" in stripped or ("  " in stripped and re.search(r'\d{4}', stripped)):
            parts = [p.strip() for p in re.split(r'\s{2,}|\|', stripped)]
            if len(parts) >= 2:
                story.append(Paragraph(parts[0], styles["JobTitle"]))
                story.append(Paragraph("  ·  ".join(parts[1:]), styles["JobMeta"]))
            else:
                story.append(Paragraph(stripped, styles["JobTitle"]))
            i += 1
            continue

        # ── Everything else → body text ───────────────────────────────────────
        story.append(Paragraph(stripped, styles["BodyText2"]))
        i += 1

    return story


def generate_resume_pdf(resume_text: str, output_path: str = None) -> str:
    """Generate a formatted resume PDF. Returns the file path."""
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix="tailored_resume_")
        output_path = tmp.name
        tmp.close()

    styles = _base_styles()

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch,
        topMargin=0.65*inch,
        bottomMargin=0.65*inch,
    )

    story = _parse_resume(resume_text, styles)
    doc.build(story)
    return output_path


# ─── COVER LETTER PDF ─────────────────────────────────────────────────────────

def generate_cover_letter_pdf(cover_letter_text: str, output_path: str = None) -> str:
    """Generate a formatted cover letter PDF. Returns the file path."""
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix="cover_letter_")
        output_path = tmp.name
        tmp.close()

    styles = _base_styles()

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=1.0*inch,
        rightMargin=1.0*inch,
        topMargin=0.9*inch,
        bottomMargin=0.9*inch,
    )

    story = []

    # Accent bar at top
    story.append(_rule(width=6.0*inch))
    story.append(Spacer(1, 6))

    # Parse paragraphs
    paragraphs = [p.strip() for p in cover_letter_text.split("\n\n") if p.strip()]

    for idx, para in enumerate(paragraphs):
        # Lines within a paragraph
        inner_lines = [l.strip() for l in para.splitlines() if l.strip()]

        # Detect salutation (Dear ...) or closing (Sincerely, etc.)
        first = inner_lines[0] if inner_lines else ""
        low_first = first.lower()

        if low_first.startswith("dear") or low_first.startswith("to whom"):
            story.append(Spacer(1, 8))
            story.append(Paragraph(first, styles["JobTitle"]))
            story.append(Spacer(1, 8))
            continue

        closing_words = ("sincerely", "regards", "best", "thank you", "yours")
        if any(low_first.startswith(c) for c in closing_words):
            story.append(Spacer(1, 16))
            for l in inner_lines:
                story.append(Paragraph(l, styles["CLBody"]))
            continue

        # Normal paragraph
        full_para = " ".join(inner_lines)
        story.append(Paragraph(full_para, styles["CLBody"]))

    story.append(Spacer(1, 10))
    story.append(_rule(width=6.0*inch))

    doc.build(story)
    return output_path