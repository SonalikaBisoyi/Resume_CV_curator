"""
Gradio Web UI for Resume & Cover Letter Tailoring Agent
Run: python app.py
"""

import gradio as gr
import json
import os
from agent import run_agent
from pdf_export import generate_resume_pdf, generate_cover_letter_pdf

# ─── PDF EXTRACTION ───────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except ImportError:
        return "❌ pymupdf not installed. Run: pip install pymupdf"
    except Exception as e:
        return f"❌ Could not read PDF: {e}"


def handle_pdf_upload(pdf_file):
    if pdf_file is None:
        return gr.update(), "_Upload a PDF to auto-fill the text below, or paste directly._"
    text = extract_text_from_pdf(pdf_file)
    if text.startswith("❌"):
        return gr.update(), text
    return gr.update(value=text), f"✅ PDF extracted — {len(text.split())} words loaded. Review below before running."


# ─── UI THEME ─────────────────────────────────────────────────────────────────

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --border: #1e1e2e;
    --accent: #7c6af7;
    --accent2: #f76a8a;
    --text: #e8e8f0;
    --muted: #666680;
}

* { box-sizing: border-box; }
body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Syne', sans-serif !important;
}
.gradio-container { max-width: 1200px !important; }
h1, h2, h3, label, .label-wrap span {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}
#header {
    text-align: center;
    padding: 48px 24px 32px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 32px;
    background: linear-gradient(135deg, #0a0a0f 0%, #12101e 100%);
}
#header h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #7c6af7, #f76a8a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
    letter-spacing: -1px;
}
#header p { color: var(--muted); font-size: 1.05rem; }
textarea, input[type="text"], input[type="password"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    border-radius: 8px !important;
}
textarea:focus, input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,106,247,0.2) !important;
}
button.primary {
    background: linear-gradient(135deg, #7c6af7, #f76a8a) !important;
    border: none !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 14px 32px !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}
button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(124,106,247,0.3) !important;
}
.tab-nav button { font-family: 'Syne', sans-serif !important; font-weight: 600 !important; }
.tab-nav button.selected {
    border-bottom-color: var(--accent) !important;
    color: var(--accent) !important;
}
"""

# ─── PROCESSING FUNCTION ──────────────────────────────────────────────────────

def process(hf_token, resume, job_description, progress=gr.Progress()):
    no_files = (None, None)

    if not hf_token.strip():
        return "❌ Please enter your HuggingFace API token.", "", "", "", "", *no_files
    if not resume.strip():
        return "❌ Please paste your resume or upload a PDF.", "", "", "", "", *no_files
    if not job_description.strip():
        return "❌ Please paste the job description.", "", "", "", "", *no_files

    try:
        progress(0.1, desc="🔍 Analyzing job description...")
        results = run_agent(resume, job_description, hf_token.strip())
        progress(0.9, desc="📄 Generating PDFs...")

        # ── Format analysis text ──────────────────────────────────────────────
        jd = results.get("jd_analysis", {})
        jd_summary = f"""**Role:** {jd.get('job_title','N/A')} @ {jd.get('company','N/A')}
**Experience:** {jd.get('years_experience','N/A')} · **Tone:** {jd.get('tone','N/A')}

**Required Skills:**
{chr(10).join(f"• {s}" for s in jd.get('required_skills',[]))}

**Preferred Skills:**
{chr(10).join(f"• {s}" for s in jd.get('preferred_skills',[]))}

**Keywords:** {', '.join(jd.get('keywords',[]))}"""

        gaps = results.get("gap_analysis", {})
        gap_md = f"""**✅ Matching Skills ({len(gaps.get('matching_skills',[]))}):** {', '.join(gaps.get('matching_skills',[])) or 'None found'}

**⚠️ Missing Skills ({len(gaps.get('missing_skills',[]))}):** {', '.join(gaps.get('missing_skills',[])) or 'None — great match!'}

**💡 Strongest Selling Points:**
{chr(10).join(f"• {s}" for s in gaps.get('strongest_selling_points',[]))}

**📌 Suggested Emphasis:**
{chr(10).join(f"• {s}" for s in gaps.get('suggested_emphasis',[]))}"""

        review = results.get("review", {})
        review_md = f"""### Quality Scores
🟣 **Resume:** {review.get('resume_score','N/A')}/100 · **Cover Letter:** {review.get('cover_letter_score','N/A')}/100 · **ATS:** {str(review.get('ats_keyword_coverage','N/A')).upper()}

---
**Resume Strengths:** {' · '.join(f"✅ {s}" for s in review.get('resume_strengths',[]))}

**Resume Improvements:** {' · '.join(f"⚠️ {s}" for s in review.get('resume_improvements',[]))}

**Cover Letter Strengths:** {' · '.join(f"✅ {s}" for s in review.get('cover_letter_strengths',[]))}

**Cover Letter Improvements:** {' · '.join(f"⚠️ {s}" for s in review.get('cover_letter_improvements',[]))}

---
**Overall:** {review.get('overall_recommendation','')}"""

        tailored_resume = results.get("tailored_resume", "")
        cover_letter    = results.get("cover_letter", "")

        # ── Generate PDFs ─────────────────────────────────────────────────────
        resume_pdf_path = generate_resume_pdf(tailored_resume)
        cl_pdf_path     = generate_cover_letter_pdf(cover_letter)

        progress(1.0, desc="✅ Complete!")

        status = (f"✅ Done! Resume: {review.get('resume_score')}/100 · "
                  f"Cover Letter: {review.get('cover_letter_score')}/100 · "
                  f"PDFs ready to download ⬇️")

        return (
            status,
            jd_summary + "\n\n---\n\n" + gap_md,
            tailored_resume,
            cover_letter,
            review_md,
            resume_pdf_path,
            cl_pdf_path,
        )

    except Exception as e:
        msg = str(e)
        if "401" in msg or "Unauthorized" in msg:
            msg = "❌ Invalid HuggingFace token."
        elif "rate" in msg.lower():
            msg = "❌ Rate limit hit — wait a moment and retry."
        else:
            msg = f"❌ Error: {msg}"
        return msg, "", "", "", "", None, None


# ─── SAMPLE DATA ──────────────────────────────────────────────────────────────

SAMPLE_RESUME = """Jane Smith
jane@email.com | linkedin.com/in/janesmith | San Francisco, CA

EXPERIENCE

Software Engineer | TechCorp | 2021-Present
• Built and maintained REST APIs using Python/FastAPI serving 100k+ daily active users
• Led migration of monolithic app to microservices, cutting deployment time by 60%
• Mentored 2 junior engineers; conducted weekly code reviews

Junior Developer | StartupXYZ | 2019-2021
• Developed React dashboard components used by 500+ enterprise clients
• Wrote comprehensive unit and integration tests (80% coverage with pytest)

SKILLS
Python, FastAPI, React, PostgreSQL, Docker, Redis, AWS (EC2, S3), Git, CI/CD

EDUCATION
B.S. Computer Science | State University | 2019 | GPA 3.7"""

SAMPLE_JD = """Senior Software Engineer - Backend
Acme Corp | San Francisco, CA | Full-time

Requirements:
• 4+ years Python backend development
• Microservices and distributed systems experience
• Strong PostgreSQL and Redis knowledge
• Docker + Kubernetes in production
• CI/CD experience (GitHub Actions, CircleCI)

Nice to Have: Kafka, AWS/GCP, fintech experience
Compensation: $160k-$200k + equity + benefits"""


# ─── GRADIO UI ────────────────────────────────────────────────────────────────

with gr.Blocks() as demo:

    gr.HTML("""
    <div id="header">
        <h1>⚡ Resume Tailor Agent</h1>
        <p>Powered by Qwen-72B on HuggingFace · 5-step agentic pipeline · Export as PDF</p>
    </div>
    """)

    with gr.Row():
        # ── LEFT COLUMN: inputs ───────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 🔑 Setup")
            hf_token = gr.Textbox(
                label="HuggingFace API Token", type="password",
                placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxx",
                info="Free token at huggingface.co/settings/tokens"
            )

            gr.Markdown("### 📄 Your Resume")
            pdf_upload = gr.File(
                label="📎 Upload Resume PDF (optional)",
                file_types=[".pdf"], type="filepath"
            )
            pdf_status = gr.Markdown(
                value="_Upload a PDF to auto-fill, or paste below._"
            )
            resume_input = gr.Textbox(
                label="Resume Text",
                lines=12,
                placeholder="Paste your resume here...",
                value=SAMPLE_RESUME
            )
            pdf_upload.change(
                fn=handle_pdf_upload,
                inputs=[pdf_upload],
                outputs=[resume_input, pdf_status]
            )

            gr.Markdown("### 💼 Job Description")
            jd_input = gr.Textbox(
                label="Paste the job description",
                lines=12,
                placeholder="Paste the full job posting...",
                value=SAMPLE_JD
            )

            run_btn = gr.Button("🚀 Run Agent (5 Steps)", variant="primary", size="lg")
            status_output = gr.Markdown()

        # ── RIGHT COLUMN: results ─────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Results")

            with gr.Tabs():
                with gr.Tab("🔍 JD Analysis & Gaps"):
                    analysis_output = gr.Markdown()

                with gr.Tab("📝 Tailored Resume"):
                    resume_output = gr.Textbox(label="Tailored Resume", lines=20)

                with gr.Tab("💌 Cover Letter"):
                    cover_output = gr.Textbox(label="Cover Letter", lines=20)

                with gr.Tab("⭐ Quality Review"):
                    review_output = gr.Markdown()

            # ── PDF download section ──────────────────────────────────────────
            gr.Markdown("### ⬇️ Download as PDF")
            with gr.Row():
                resume_pdf_out = gr.File(
                    label="📄 Tailored Resume PDF",
                    interactive=False
                )
                cl_pdf_out = gr.File(
                    label="💌 Cover Letter PDF",
                    interactive=False
                )

    gr.HTML("""
    <div style="text-align:center;padding:24px;color:#444466;font-size:.85rem;
                border-top:1px solid #1e1e2e;margin-top:32px;">
        Pipeline: JD Analysis → Gap Analysis → Resume Tailoring → Cover Letter → Quality Review
        <br>Model: Qwen/Qwen2.5-72B-Instruct · PDF export via ReportLab
    </div>
    """)

    run_btn.click(
        fn=process,
        inputs=[hf_token, resume_input, jd_input],
        outputs=[
            status_output,
            analysis_output,
            resume_output,
            cover_output,
            review_output,
            resume_pdf_out,
            cl_pdf_out,
        ]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Base(),
        css=custom_css
    )
