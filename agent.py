"""
Resume & Cover Letter Tailoring Agent
Uses Qwen (via HuggingFace Inference API) as the free LLM backbone.
Agentic loop: Analyze JD → Extract Skills → Tailor Resume → Write Cover Letter → Review
"""

import os
import json
import re
from huggingface_hub import InferenceClient

# ─── CONFIG ───────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"   # change to any HF chat model you like


# ─── HELPER: single chat call ─────────────────────────────────────────────────

def chat(client: InferenceClient, system: str, user: str, max_tokens: int = 1000, temperature: float = 0.3) -> str:
    """Call the HF chat-completions endpoint (works with ALL modern instruct models)."""
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


# ─── AGENT TOOLS ──────────────────────────────────────────────────────────────

def analyze_job_description(client: InferenceClient, jd: str) -> dict:
    """Tool 1: Extract structured info from the job description."""
    system = "You are a job analysis expert. Always respond with valid JSON only — no markdown, no extra text."
    user = f"""Analyze this job description and return ONLY a JSON object with this exact structure:
{{
  "job_title": "...",
  "company": "...",
  "required_skills": ["skill1", "skill2"],
  "preferred_skills": ["skill1", "skill2"],
  "key_responsibilities": ["resp1", "resp2"],
  "keywords": ["kw1", "kw2"],
  "tone": "formal/startup/technical/creative",
  "years_experience": "X years or entry-level"
}}

Job Description:
{jd}"""

    raw = chat(client, system, user, max_tokens=800, temperature=0.2)

    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return {
        "job_title": "Position", "company": "Company",
        "required_skills": [], "preferred_skills": [],
        "key_responsibilities": [], "keywords": [],
        "tone": "formal", "years_experience": "Not specified"
    }


def gap_analysis(client: InferenceClient, resume: str, jd_analysis: dict) -> dict:
    """Tool 2: Find gaps and strengths between resume and JD."""
    system = "You are a career coach. Always respond with valid JSON only — no markdown, no extra text."
    user = f"""Compare this resume against the job requirements and return ONLY a JSON object:
{{
  "matching_skills": ["skills found in both"],
  "missing_skills": ["required skills not in resume"],
  "transferable_experiences": ["resume experiences that map to JD needs"],
  "strongest_selling_points": ["top 3 reasons this candidate fits"],
  "suggested_emphasis": ["what to highlight more strongly"]
}}

RESUME:
{resume}

JOB REQUIREMENTS:
- Title: {jd_analysis.get('job_title')}
- Required Skills: {', '.join(jd_analysis.get('required_skills', []))}
- Preferred Skills: {', '.join(jd_analysis.get('preferred_skills', []))}
- Key Responsibilities: {', '.join(jd_analysis.get('key_responsibilities', []))}"""

    raw = chat(client, system, user, max_tokens=700, temperature=0.3)

    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return {
        "matching_skills": [], "missing_skills": [],
        "transferable_experiences": [], "strongest_selling_points": [],
        "suggested_emphasis": []
    }


def tailor_resume(client: InferenceClient, resume: str, jd_analysis: dict, gap_data: dict) -> str:
    """Tool 3: Rewrite and tailor the resume for the specific JD."""
    keywords_str   = ', '.join(jd_analysis.get('keywords', []))
    strengths_str  = '\n'.join(f"- {s}" for s in gap_data.get('strongest_selling_points', []))
    emphasis_str   = '\n'.join(f"- {s}" for s in gap_data.get('suggested_emphasis', []))

    system = "You are an expert resume writer. Return only the tailored resume text — no commentary."
    user = f"""Tailor this resume for the target job.

ORIGINAL RESUME:
{resume}

TARGET JOB: {jd_analysis.get('job_title')} at {jd_analysis.get('company')}
KEYWORDS TO INCLUDE: {keywords_str}
STRONGEST SELLING POINTS:
{strengths_str}
AREAS TO EMPHASIZE:
{emphasis_str}

Rules:
1. Keep ALL facts accurate — do not invent experience
2. Reorder bullets to put most relevant experience first
3. Naturally weave in keywords from the JD
4. Quantify achievements where possible
5. Match tone: {jd_analysis.get('tone', 'professional')}

Write the complete tailored resume:"""

    return chat(client, system, user, max_tokens=1500, temperature=0.4)


def write_cover_letter(client: InferenceClient, resume: str, jd: str, jd_analysis: dict, gap_data: dict) -> str:
    """Tool 4: Write a compelling, personalized cover letter."""
    selling_points = '\n'.join(f"- {s}" for s in gap_data.get('strongest_selling_points', []))

    system = "You are an expert cover letter writer. Return only the cover letter text — no commentary."
    user = f"""Write a compelling, personalized cover letter.

CANDIDATE RESUME:
{resume}

JOB DESCRIPTION:
{jd}

KEY SELLING POINTS:
{selling_points}

Requirements:
- Address hiring team at {jd_analysis.get('company', 'the company')}
- Opening: hook with a specific reason you want THIS role
- Middle: connect 2-3 real experiences to their exact needs
- Closing: confident call to action
- Tone: {jd_analysis.get('tone', 'professional')} but warm and human
- 3-4 paragraphs, NOT generic
- Do NOT open with "I am writing to express my interest"

Write the complete cover letter:"""

    return chat(client, system, user, max_tokens=1000, temperature=0.6)


def quality_review(client: InferenceClient, tailored_resume: str, cover_letter: str, jd_analysis: dict) -> dict:
    """Tool 5: Self-critique and score the outputs."""
    system = "You are a senior hiring manager. Always respond with valid JSON only — no markdown, no extra text."
    user = f"""Review this tailored resume and cover letter for the role below and return ONLY a JSON object:
{{
  "resume_score": 85,
  "cover_letter_score": 88,
  "resume_strengths": ["strength1", "strength2"],
  "resume_improvements": ["improvement1"],
  "cover_letter_strengths": ["strength1"],
  "cover_letter_improvements": ["improvement1"],
  "ats_keyword_coverage": "high/medium/low",
  "overall_recommendation": "brief summary"
}}

TARGET ROLE: {jd_analysis.get('job_title')} at {jd_analysis.get('company')}
REQUIRED SKILLS: {', '.join(jd_analysis.get('required_skills', []))}

TAILORED RESUME (first 600 chars):
{tailored_resume[:600]}...

COVER LETTER (first 400 chars):
{cover_letter[:400]}..."""

    raw = chat(client, system, user, max_tokens=600, temperature=0.2)

    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return {
        "resume_score": 75, "cover_letter_score": 75,
        "resume_strengths": ["Well-structured"],
        "resume_improvements": ["Add more keywords"],
        "cover_letter_strengths": ["Personalized"],
        "cover_letter_improvements": ["Stronger opening"],
        "ats_keyword_coverage": "medium",
        "overall_recommendation": "Good foundation, minor tweaks needed"
    }


# ─── MAIN AGENTIC LOOP ────────────────────────────────────────────────────────

def run_agent(resume: str, job_description: str, hf_token: str = HF_TOKEN) -> dict:
    """5-step agentic pipeline."""
    print("\n🤖 Resume Tailoring Agent Starting...\n" + "=" * 60)

    if not hf_token:
        raise ValueError("HuggingFace token required. Set HF_TOKEN env variable.")

    client = InferenceClient(token=hf_token)

    results = {"steps": [], "jd_analysis": None, "gap_analysis": None,
               "tailored_resume": None, "cover_letter": None, "review": None}

    # Step 1
    print("📋 Step 1/5: Analyzing Job Description...")
    jd_analysis = analyze_job_description(client, job_description)
    results["jd_analysis"] = jd_analysis
    results["steps"].append({"step": 1, "name": "JD Analysis", "status": "complete",
                              "output": f"Found {len(jd_analysis.get('required_skills', []))} required skills"})
    print(f"   ✅ Role: {jd_analysis.get('job_title')} at {jd_analysis.get('company')}")

    # Step 2
    print("\n🔍 Step 2/5: Running Gap Analysis...")
    gaps = gap_analysis(client, resume, jd_analysis)
    results["gap_analysis"] = gaps
    results["steps"].append({"step": 2, "name": "Gap Analysis", "status": "complete",
                              "output": f"Found {len(gaps.get('matching_skills', []))} matching skills"})
    print(f"   ✅ Matching: {len(gaps.get('matching_skills', []))} | Missing: {len(gaps.get('missing_skills', []))}")

    # Step 3
    print("\n✍️  Step 3/5: Tailoring Resume...")
    tailored_resume = tailor_resume(client, resume, jd_analysis, gaps)
    results["tailored_resume"] = tailored_resume
    results["steps"].append({"step": 3, "name": "Resume Tailoring", "status": "complete",
                              "output": f"{len(tailored_resume.split())} words"})
    print(f"   ✅ Resume tailored ({len(tailored_resume.split())} words)")

    # Step 4
    print("\n💌 Step 4/5: Writing Cover Letter...")
    cover_letter = write_cover_letter(client, resume, job_description, jd_analysis, gaps)
    results["cover_letter"] = cover_letter
    results["steps"].append({"step": 4, "name": "Cover Letter", "status": "complete",
                              "output": f"{len(cover_letter.split())} words"})
    print(f"   ✅ Cover letter written ({len(cover_letter.split())} words)")

    # Step 5
    print("\n🎯 Step 5/5: Quality Review...")
    review = quality_review(client, tailored_resume, cover_letter, jd_analysis)
    results["review"] = review
    results["steps"].append({"step": 5, "name": "Quality Review", "status": "complete",
                              "output": f"Resume: {review.get('resume_score')}/100 | CL: {review.get('cover_letter_score')}/100"})
    print(f"   ✅ Resume: {review.get('resume_score')}/100 | Cover Letter: {review.get('cover_letter_score')}/100")

    print("\n" + "=" * 60 + "\n🎉 Agent Complete!\n")
    return results


# ─── CLI ENTRY POINT ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("❌ Set HF_TOKEN environment variable first.")
    else:
        sample_resume = """Jane Smith | jane@email.com
Software Engineer at TechCorp (2021-Present)
- Built REST APIs with Python/FastAPI, 100k daily users
- Led microservices migration, cut deploy time 60%
Skills: Python, FastAPI, React, PostgreSQL, Docker, Redis, AWS"""

        sample_jd = """Senior Backend Engineer - Acme Corp
Requirements: 4+ years Python, microservices, PostgreSQL, Redis, Docker/K8s, CI/CD"""

        results = run_agent(sample_resume, sample_jd, token)
        print(results["tailored_resume"][:300])