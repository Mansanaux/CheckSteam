import os
import time
import streamlit as st
import google.generativeai as genai

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Atoinfinity", layout="wide")
st.title("🚀 Atoinfinity: Capability Intelligence")

# ----------------------------
# API Key (secrets/env only)
# ----------------------------
api_key = os.getenv("Gemini_API") or st.secrets.get("Gemini_API", None)
if not api_key:
    st.error("Missing **Gemini_API**. Set it in Streamlit Cloud → App → Settings → Secrets.")
    st.stop()

try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Failed to configure Gemini API: {e}")
    st.stop()

# ----------------------------
# Simple config (easy to change)
# ----------------------------
DEFAULT_MODEL = "gemini-1.5-flash"  # most reliable baseline
COOLDOWN_SECONDS = 10
MAX_SKILLS_CHARS = 6000  # prevents huge prompts / token blowups

# ----------------------------
# Sidebar inputs
# ----------------------------
with st.sidebar:
    st.header("Setup")
    role = st.text_input("Target Role", placeholder="e.g., HR / Cloud Engineer / DevOps")
    company = st.text_input("Target Company (optional)", placeholder="e.g., TCS")
    style = st.selectbox("Output Style", ["Concise", "Detailed"], index=0)

skills = st.text_area("Current Skills & Experience", height=220)

# ----------------------------
# Cooldown (prevents rate-limit spam)
# ----------------------------
if "last_call" not in st.session_state:
    st.session_state.last_call = 0.0

def enforce_cooldown():
    now = time.time()
    elapsed = now - st.session_state.last_call
    if elapsed < COOLDOWN_SECONDS:
        st.warning(f"Please wait {int(COOLDOWN_SECONDS - elapsed)}s and try again.")
        st.stop()
    st.session_state.last_call = now

# ----------------------------
# Session-only cache (no cross-user mixing)
# ----------------------------
# key: (role, company, style, skills_trimmed)
if "cache" not in st.session_state:
    st.session_state.cache = {}

def build_prompt(role: str, company: str, style: str, skills_text: str) -> str:
    detail = "Keep it short and bullet-heavy." if style == "Concise" else "Add short examples where useful."
    company_line = f"Company: {company}" if company.strip() else "Company: (not specified)"

    return f"""
You are a senior career strategist.

Role: {role}
{company_line}

Profile:
{skills_text}

Output (markdown with ## headings and bullets):
1) Readiness score /100 + 3 reasons
2) Strengths (5 bullets)
3) Gaps: Must-have vs Nice-to-have
4) 30-60-90 plan (weekly actions)
5) 3 portfolio projects (outcomes + stack)
6) Certifications (only if essential)
7) Interview prep checklist + 1-minute pitch

Rules: Avoid generic advice. Be specific. {detail}
""".strip()

def call_gemini(prompt: str, style: str) -> str:
    temperature = 0.6 if style == "Concise" else 0.75

    model = genai.GenerativeModel(
        model_name=DEFAULT_MODEL,
        generation_config={
            "temperature": temperature,
            "top_p": 0.9,
            "max_output_tokens": 1400,
        },
    )

    resp = model.generate_content(prompt)

    text = getattr(resp, "text", None)
    if not text or not text.strip():
        raise RuntimeError("Gemini returned an empty response.")
    return text

# ----------------------------
# Generate
# ----------------------------
if st.button("Generate Strategy", type="primary"):
    if not role.strip() or not skills.strip():
        st.warning("Enter Target Role and your Skills.")
        st.stop()

    enforce_cooldown()

    skills_trimmed = skills.strip()[:MAX_SKILLS_CHARS]
    cache_key = (role.strip(), company.strip(), style, skills_trimmed)

    # Session-only cache: same user only
    if cache_key in st.session_state.cache:
        st.info("Showing cached result (this session).")
        output = st.session_state.cache[cache_key]
        st.markdown(output)
        st.stop()

    prompt = build_prompt(role.strip(), company.strip(), style, skills_trimmed)

    try:
        with st.spinner("Generating..."):
            output = call_gemini(prompt, style)

        st.session_state.cache[cache_key] = output

        st.success("Done")
        st.markdown(output)

        st.download_button(
            "Download as Markdown",
            data=output.encode("utf-8"),
            file_name="atoinfinity_output.md",
            mime="text/markdown",
        )

    except Exception as e:
        msg = str(e).lower()
        if "429" in msg or "quota" in msg or "rate" in msg:
            st.error("Rate limit / quota reached. Try again later (or enable billing / use a different project).")
        else:
            st.error(f"Error: {e}")
