import os
import time
import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="Atoinfinity", layout="wide")
st.title("🚀 Atoinfinity: Capability Intelligence")

# --- Key: secrets/env only ---
api_key = os.getenv("Gemini_API") or st.secrets.get("Gemini_API", None)
if not api_key:
    st.error("Missing Gemini_API. Add it in Streamlit Cloud → App → Settings → Secrets.")
    st.stop()

genai.configure(api_key=api_key)

# --- Simple input UI ---
with st.sidebar:
    st.header("Setup")
    role = st.text_input("Target Role", placeholder="e.g., HR / Cloud Engineer / DevOps")
    company = st.text_input("Target Company (optional)", placeholder="e.g., TCS")
    style = st.selectbox("Output", ["Concise", "Detailed"], index=0)

skills = st.text_area("Current Skills & Experience", height=220)

# --- Cooldown to avoid rate-limit ---
COOLDOWN = 10  # seconds
if "last_call" not in st.session_state:
    st.session_state.last_call = 0.0

def enforce_cooldown():
    now = time.time()
    if now - st.session_state.last_call < COOLDOWN:
        wait = int(COOLDOWN - (now - st.session_state.last_call))
        st.warning(f"Please wait {wait}s and try again.")
        st.stop()
    st.session_state.last_call = now

# --- Cache to avoid repeated calls for same input ---
@st.cache_data(ttl=3600)
def call_gemini(prompt: str):
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",  # most reliable for free-tier style usage
        generation_config={"temperature": 0.6, "top_p": 0.9, "max_output_tokens": 1400},
    )
    resp = model.generate_content(prompt)
    return getattr(resp, "text", "")

def build_prompt(role: str, company: str, style: str, skills: str) -> str:
    detail = "Keep it short and bullet-heavy." if style == "Concise" else "Add short examples where useful."
    company_line = f"Company: {company}" if company.strip() else "Company: (not specified)"
    return f"""
You are a senior career strategist.

Role: {role}
{company_line}

Profile:
{skills}

Output:
1) Readiness score /100 + 3 reasons
2) Strengths (5 bullets)
3) Gaps: Must-have vs Nice-to-have
4) 30-60-90 plan (weekly actions)
5) 3 portfolio projects (outcomes + stack)
6) Certifications (only if essential)
7) Interview prep checklist + 1-minute pitch

Rules: Use markdown headings (##) and bullets. {detail}
""".strip()

if st.button("Generate Strategy", type="primary"):
    if not role.strip() or not skills.strip():
        st.warning("Enter Target Role and your Skills.")
        st.stop()

    enforce_cooldown()
    prompt = build_prompt(role, company, style, skills)

    try:
        with st.spinner("Generating..."):
            output = call_gemini(prompt)

        if not output:
            st.error("No response received. Try again.")
            st.stop()

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
            st.error("Rate limit / quota reached. Try again in a minute, or later today.")
        else:
            st.error(f"Error: {e}")
