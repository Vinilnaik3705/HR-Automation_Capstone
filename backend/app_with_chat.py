import os
import re
import io
import sys
import json
import toml
import shutil
import tempfile
import traceback
import uuid
from typing import List, Optional, Dict, Any, Tuple
import streamlit as st
import pandas as pd
from datetime import datetime
from tempfile import NamedTemporaryFile

import psycopg2
from psycopg2.extras import RealDictCursor
from interview_scheduler import show_interview_scheduling_section
from feedback_agent import show_feedback_section, create_feedback_tables

try:
    import fitz  
except Exception:
    fitz = None

try:
    from docx import Document
except Exception:
    Document = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import faiss 
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

OpenAIClient = None
try:
    from openai import OpenAI as OpenAI_v2  
    OpenAIClient = "openai_v2"
except Exception:
    try:
        import openai  
        OpenAIClient = "openai_v1"
    except Exception:
        OpenAIClient = None

ADMIN_ROLES = {"Admin", "HR Manager", "Tech Lead", "Team Manager", "Student"}

def is_admin() -> bool:
    return bool(st.session_state.get("user")) and str(st.session_state.user.get("role", "")).lower() in ADMIN_ROLES

def require_admin(feature_name: str) -> bool:
    if not is_admin():
        st.error(f"⛔ You don't have permission to access **{feature_name}**.")
        return False
    return True

@st.cache_resource
def get_db_connection():
    try:
        if os.path.exists("secrets.toml"):
            secrets = toml.load("secrets.toml")
            db_config = secrets.get('database', {})
        else:
            db_config = {}
        
        conn = psycopg2.connect(
            host=db_config.get('host', os.getenv('DB_HOST', 'localhost')),
            database=db_config.get('name', os.getenv('DB_NAME', 'resume_analyzer')),
            user=db_config.get('user', os.getenv('DB_USER', 'postgres')),
            password=db_config.get('password', os.getenv('DB_PASSWORD', 'password')),
            port=db_config.get('port', os.getenv('DB_PORT', '5432'))
        )
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

def log_user_action(action: str, description: str = ""):
    conn = get_db_connection()
    if conn and 'user' in st.session_state:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO app_logs (user_id, action, description, ip_address, user_agent)
                    VALUES (%s, %s, %s, %s, %s)
                """, (st.session_state.user['id'], action, description, "N/A", "Streamlit App"))
            conn.commit()
        except Exception as e:
            print(f"Logging error: {e}")

def save_upload_session(total_files: int, skills_text: str, session_id: str):
    conn = get_db_connection()
    if conn and 'user' in st.session_state:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO upload_sessions (user_id, session_id, total_files, skills_text, status)
                    VALUES (%s, %s, %s, %s, 'processing')
                """, (st.session_state.user['id'], session_id, total_files, skills_text))
            conn.commit()
        except Exception as e:
            print(f"Error saving upload session: {e}")

def update_upload_session(session_id: str, processed: int, failed: int, status: str):
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE upload_sessions 
                    SET processed_files = %s, failed_files = %s, status = %s, end_time = CURRENT_TIMESTAMP
                    WHERE session_id = %s
                """, (processed, failed, status, session_id))
            conn.commit()
        except Exception as e:
            print(f"Error updating upload session: {e}")

def save_resume_data(file_data: Dict, session_id: str):
    conn = get_db_connection()
    if not (conn and 'user' in st.session_state):
        return False

    user_id = st.session_state.user['id']
    filename = file_data['File']
    file_size = len(file_data.get('raw_text', ''))
    file_type = os.path.splitext(filename)[1][1:]

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id FROM resume_files
                WHERE user_id = %s AND filename = %s
                ORDER BY upload_date DESC
                LIMIT 1
            """, (user_id, filename))
            row = cur.fetchone()

            if row:
                file_id = row[0]

                cur.execute("""
                    UPDATE resume_files
                    SET file_size = %s,
                        file_type = %s,
                        session_id = %s,
                        processed = TRUE,
                        upload_date = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (file_size, file_type, session_id, file_id))

                cur.execute("""
                    UPDATE resume_data
                    SET candidate_name = %s,
                        candidate_email = %s,
                        candidate_phone = %s,
                        skills = %s,
                        extracted_text = %s,
                        raw_parsed_data = %s
                    WHERE resume_file_id = %s AND user_id = %s
                """, (
                    file_data['Name'],
                    file_data['Email'],
                    file_data['Mobile'],
                    file_data['Skills'],
                    file_data.get('raw_text', ''),
                    json.dumps(file_data),
                    file_id,
                    user_id
                ))

            else:
                cur.execute("""
                    INSERT INTO resume_files (user_id, filename, file_size, file_type, session_id, processed)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (user_id, filename, file_size, file_type, session_id, True))
                file_id = cur.fetchone()[0]

                cur.execute("""
                    INSERT INTO resume_data (resume_file_id, user_id, candidate_name, candidate_email, 
                                             candidate_phone, skills, extracted_text, raw_parsed_data)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    file_id, user_id,
                    file_data['Name'], file_data['Email'], file_data['Mobile'],
                    file_data['Skills'], file_data.get('raw_text', ''),
                    json.dumps(file_data)
                ))

        conn.commit()
        return True

    except Exception as e:
        print(f"Error saving (upserting) resume data: {e}")
        conn.rollback()
        return False

def save_matching_session(jd_text: str, req_skills: str, use_faiss: bool, skill_weight: float, 
                         shortlist_count: int, results: List[Dict]):
    conn = get_db_connection()
    if conn and 'user' in st.session_state:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO matching_sessions (user_id, jd_text, required_skills, use_faiss, 
                                                 skill_weight, shortlist_count, results)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (st.session_state.user['id'], jd_text, req_skills, use_faiss, 
                      skill_weight, shortlist_count, json.dumps(results)))
            conn.commit()
            log_user_action("job_matching", f"Matched {len(results)} resumes")
            return True
        except Exception as e:
            print(f"Error saving matching session: {e}")
            return False

def save_chat_history(prompt: str, response: str, model: str, context: str = ""):
    conn = get_db_connection()
    if conn and 'user' in st.session_state:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO chat_history (user_id, user_prompt, ai_response, model_used, context_used)
                    VALUES (%s, %s, %s, %s, %s)
                """, (st.session_state.user['id'], prompt, response, model, context))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving chat history: {e}")
            return False

def get_user_history():
    conn = get_db_connection()
    if conn and 'user' in st.session_state:
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT session_id, total_files, processed_files, failed_files, 
                           start_time, status 
                    FROM upload_sessions 
                    WHERE user_id = %s 
                    ORDER BY start_time DESC 
                    LIMIT 10
                """, (st.session_state.user['id'],))
                upload_sessions = cur.fetchall()
                
                cur.execute("""
                    SELECT id, jd_text, shortlist_count, created_at 
                    FROM matching_sessions 
                    WHERE user_id = %s 
                    ORDER BY created_at DESC 
                    LIMIT 10
                """, (st.session_state.user['id'],))
                matching_sessions = cur.fetchall()
                
                cur.execute("""
                    SELECT COUNT(*) as total_resumes 
                    FROM resume_files 
                    WHERE user_id = %s
                """, (st.session_state.user['id'],))
                resume_count = cur.fetchone()['total_resumes']
                
            return {
                'upload_sessions': upload_sessions,
                'matching_sessions': matching_sessions,
                'resume_count': resume_count
            }
        except Exception as e:
            print(f"Error getting user history: {e}")
            return None

def get_user_resumes():
    conn = get_db_connection()
    if conn and 'user' in st.session_state:
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT rd.candidate_name, rd.candidate_email, rd.candidate_phone, 
                           rd.skills, rf.filename, rf.upload_date
                    FROM resume_data rd
                    JOIN resume_files rf ON rd.resume_file_id = rf.id
                    WHERE rd.user_id = %s
                    ORDER BY rf.upload_date DESC
                """, (st.session_state.user['id'],))
                return cur.fetchall()
        except Exception as e:
            print(f"Error getting user resumes: {e}")
            return []

def setup_authentication():
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'show_register' not in st.session_state:
        st.session_state.show_register = False

def register_user(username: str, password: str, email: str):
    conn = None
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM users WHERE username = %s", (username,))
                if cur.fetchone():
                    st.error("Username already exists. Please choose a different username.")
                    return False
                
                cur.execute("SELECT id FROM users WHERE email = %s", (email,))
                if cur.fetchone():
                    st.error("Email already exists. Please use a different email.")
                    return False
                
                cur.execute("""
                    INSERT INTO users (username, password_hash, email, role)
                    VALUES (%s, %s, %s, 'user')
                """, (username, password, email))
                
                conn.commit()
                st.success("🎉 Registration successful! You can now login with your credentials.")
                log_user_action("user_registration", f"New user registered: {username}")
                return True
        else:
            st.error("Cannot connect to database")
            return False
    except Exception as e:
        st.error(f"Registration error: {str(e)}")
        return False

def login_user(username: str, password: str):
    conn = None
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, username, email, role FROM users 
                    WHERE username = %s AND password_hash = %s AND is_active = TRUE
                """, (username, password))
                user = cur.fetchone()
                
                if user:
                    st.session_state.user = user
                    st.session_state.authenticated = True
                    log_user_action("login", f"User {username} logged in")
                    return True
                else:
                    st.error("Invalid username or password")
                    return False
        else:
            st.error("Cannot connect to database")
            return False
    except Exception as e:
        st.error(f"Login error: {str(e)}")
        return False

def logout_user():
    if 'user' in st.session_state:
        log_user_action("logout", f"User {st.session_state.user['username']} logged out")
    st.session_state.user = None
    st.session_state.authenticated = False
    st.session_state.raw_texts = {}
    st.session_state.results_rows = []
    st.session_state.scored_results = None
    st.session_state.show_register = False

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(
    r"(\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}"
    r"|\+\d{1,3}\s?\(\d{1,4}\)\s?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}"
    r"|\b91[-.\s]?\d{5}[-.\s]?\d{5}\b"
    r"|\b\+91[-.\s]?\d{5}[-.\s]?\d{5}\b"
    r"|\b0\d{5}[-.\s]?\d{5}\b"
    r"|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"
    r"|\b\(\d{3}\)[-.\s]?\d{3}[-.\s]?\d{4}\b"
    r"|\b\d{10}\b"
    r"|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"
    r"|\b\d{5}[-.\s]\d{5}\b"
    r"|\b(?:6|7|8|9)\d{9}\b"
    r"|\b\d{4}[-.\s]?\d{3}[-.\s]?\d{3}\b)"
)

MAJOR_SECTION_HINTS = (
    "education", "experience", "work experience", "employment", "skills", "projects",
    "certification", "certifications", "awards", "publications", "summary", "objective",
    "profile", "interests", "languages"
)
HEADER_STOP_WORDS = {
    "degree", "certificate", "degree/certificate", "year", "institute", "cgpa", "gpa",
    "highlights", "responsibilities", "role", "company", "organization", "university",
    "college", "board", "class", "standard", "state", "country", "city", "address",
    "contact", "linkedin", "github", "email", "phone", "mobile", "website", "portfolio"
}

def norm_text(s: str) -> str:
    s = re.sub(r"[•·∙●◦➤➥▪■◆▶►]+", " ", s)
    s = re.sub(r"[,_–—\-–]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _clean_line_for_name(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _looks_like_section_header(line: str) -> bool:
    low = line.lower()
    if any(k in low for k in HEADER_STOP_WORDS):
        return True
    if len(line.split()) <= 4 and not re.search(r"\d|@|http", line):
        if any(h in low for h in MAJOR_SECTION_HINTS):
            return True
    return False

def _guess_name_from_email(email: str) -> Optional[str]:
    local = email.split("@")[0]
    local = re.sub(r"\d+", "", local)
    parts = [p for p in re.split(r"[._-]+", local) if len(p) > 0]
    parts = [p.capitalize() for p in parts[:4]]
    name = " ".join(parts).strip()
    return name or None

def extract_name(text: str) -> Optional[str]:
    lines = [_clean_line_for_name(l) for l in text.splitlines() if l.strip()]
    if not lines:
        return None

    contact_idx = None
    first_section_idx = None
    for i, l in enumerate(lines[:100]):
        if contact_idx is None and (EMAIL_RE.search(l) or PHONE_RE.search(l)):
            contact_idx = i
        low = l.lower()
        if first_section_idx is None and any(h in low for h in MAJOR_SECTION_HINTS):
            first_section_idx = i
        if contact_idx is not None and first_section_idx is not None:
            break

    stop = min([x for x in [contact_idx, first_section_idx, 30] if x is not None] or [30])
    stop = max(1, min(stop, len(lines)))

    candidates = []
    for idx, line in enumerate(lines[:stop]):
        if _looks_like_section_header(line):
            continue
        if "@" in line or "http" in line or "www." in line:
            continue
        if re.search(r"\d", line):
            continue

        tokens = re.findall(r"[A-Za-z][A-Za-z'’\-]*\.?", line)
        if not tokens:
            continue
        if not (1 <= len(tokens) <= 5):
            continue

        initials = [t for t in tokens if len(re.sub(r"[.'’\-]", "", t)) == 1]
        if len(initials) > 1 or (len(initials) == 1 and tokens.index(initials[0]) != 0):
            continue

        cap_tokens = sum(1 for t in tokens if t[0].isupper() or t.isupper())
        score = cap_tokens + (2 if len(tokens) in (2, 3) else 0) + (1 if idx <= 5 else 0)
        if line.isupper():
            score += 1

        candidates.append((score, idx, " ".join(t.strip(" .") for t in tokens)))

    if candidates:
        candidates.sort(key=lambda x: (-x[0], x[1]))
        return candidates[0][2]

    if contact_idx and contact_idx > 0:
        for j in range(max(0, contact_idx - 3), contact_idx):
            l = lines[j]
            if _looks_like_section_header(l) or "@" in l or re.search(r"\d", l):
                continue
            tokens = re.findall(r"[A-Za-z][A-Za-z'’\-]*\.?", l)
            if 1 <= len(tokens) <= 5:
                return " ".join(t.strip(" .") for t in tokens)

    m = EMAIL_RE.search("\n".join(lines[:100]))
    if m:
        g = _guess_name_from_email(m.group(0))
        if g:
            return g

    return None

def extract_contact_number(text: str) -> Optional[str]:
    all_matches = []
    for match in re.findall(PHONE_RE, text):
        if isinstance(match, tuple):
            match = max(match, key=len)
        cleaned = re.sub(r"[^\d\+]", "", match)
        if 8 <= len(cleaned.replace("+", "")) <= 15 and match not in all_matches:
            all_matches.append(match)
    if not all_matches:
        return None

    def phone_score(phone: str) -> int:
        score = 0
        if "+" in phone:
            score += 2
        if len(re.sub(r"\D", "", phone)) == 10:
            score += 1
        if re.search(r"[6-9]\d{9}", phone):
            score += 1
        return score

    all_matches.sort(key=phone_score, reverse=True)
    return all_matches[0]

def extract_email(text: str) -> Optional[str]:
    matches = EMAIL_RE.findall(text)
    if matches:
        for email in matches:
            if not any(placeholder in email.lower() for placeholder in ["example.com", "test.com", "placeholder"]):
                return email
    return None

def extract_skills(text: str, skills_list: List[str]) -> List[str]:
    skills = []
    text_lower = text.lower()
    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill.lower()))
        if re.search(pattern, text_lower):
            skills.append(skill)
    return sorted(list(set(skills)))

def extract_text_from_pdf_path(pdf_path: str) -> str:
    if fitz is None:
        return "__ERROR__: PyMuPDF (fitz) not installed"
    try:
        text = ""
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text
    except Exception as e:
        return f"__ERROR__: PDF read failed: {e}"

def extract_text_from_docx_path(docx_path: str) -> str:
    if Document is None:
        return "__ERROR__: python-docx not installed"
    try:
        doc = Document(docx_path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return text
    except Exception as e:
        return f"__ERROR__: DOCX read failed: {e}"

def extract_text_from_uploaded(uploaded_file) -> Tuple[str, str]:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name
    try:
        if suffix == ".pdf":
            text = extract_text_from_pdf_path(temp_path)
        elif suffix == ".docx":
            text = extract_text_from_docx_path(temp_path)
        else:
            text = "__ERROR__: Unsupported file type"
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass
    return text, uploaded_file.name

_embedder_cache = {"model": None}

def get_embedder():
    if _embedder_cache["model"] is not None:
        return _embedder_cache["model"]
    model = None
    if SentenceTransformer is not None:
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            model = None
    _embedder_cache["model"] = model
    return model

def embed_texts(texts: List[str]):
    model = get_embedder()
    if model is None:
        return None
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    import numpy as _np
    return _np.asarray(embs, dtype="float32")

def tfidf_cosine(jd_text: str, candidate_texts: List[str]):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  
        from sklearn.metrics.pairwise import cosine_similarity      
    except Exception:
        return None
    vec = TfidfVectorizer(stop_words="english", max_features=50000)
    X = vec.fit_transform([jd_text] + candidate_texts)
    sims = cosine_similarity(X[0:1], X[1:]).flatten()
    import numpy as _np
    return _np.asarray(sims, dtype="float32")

def jaccard_similarity(jd_text: str, candidate_texts: List[str]):
    def toks(s: str):
        return set(re.findall(r"[a-z]+", s.lower()))
    jd_set = toks(jd_text)
    res = []
    for t in candidate_texts:
        cs = toks(t)
        if not jd_set or not cs:
            res.append(0.0)
        else:
            res.append(len(jd_set & cs) / len(jd_set | cs))
    import numpy as _np
    return _np.asarray(res, dtype="float32")

def compute_text_similarities(jd_text: str, candidate_texts: List[str], use_faiss: bool):
    import numpy as _np
    embs = embed_texts(candidate_texts)
    if embs is not None:
        q = embed_texts([jd_text])
        if q is None:
            sims = _np.zeros(len(candidate_texts), dtype="float32")
        else:
            qv = q[0].astype("float32")
            if use_faiss and faiss is not None:
                index = faiss.IndexFlatIP(embs.shape[1])
                index.add(embs)
                sims = (embs @ qv).astype("float32")
            else:
                sims = (embs @ qv).astype("float32")
        sims = _np.clip((sims + 1.0) / 2.0, 0.0, 1.0)
        return sims

    sims = tfidf_cosine(jd_text, candidate_texts)
    if sims is not None:
        return _np.clip(sims, 0.0, 1.0)

    return jaccard_similarity(jd_text, candidate_texts)

def process_text_to_row(text: str, filename: str, skills_list: List[str]) -> Dict[str, Any]:
    if text.startswith("__ERROR__"):
        return {
            "File": os.path.basename(filename),
            "Name": "",
            "Email": "",
            "Mobile": "",
            "Skills": "",
            "Error": text.replace("__ERROR__: ", "")
        }
    name = extract_name(text) or ""
    email = extract_email(text) or ""
    phone = extract_contact_number(text) or ""
    skills = extract_skills(text, skills_list)
    return {
        "File": os.path.basename(filename),
        "Name": name,
        "Email": email,
        "Mobile": phone,
        "Skills": ", ".join(skills),
        "Error": ""
    }

def normalize_skill(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def compute_skill_match(required: List[str], candidate_skills_csv: str) -> float:
    if not required:
        return 0.0
    cand = [normalize_skill(x) for x in candidate_skills_csv.split(",") if x.strip()]
    req = [normalize_skill(x) for x in required if x.strip()]
    if not req:
        return 0.0
    inter = set(cand) & set(req)
    return len(inter) / len(set(req))

CHAT_SYSTEM_PROMPT = (
    "You are a precise resume analysis assistant. You will receive a user request and a CONTEXT "
    "containing snippets of resume text. Answer strictly using only the provided context. "
    "If the answer is not in the context, say 'Not found in provided resumes'. "
    "When listing people or rows, include File/Name/Email/Phone when available. "
    "If it helps, also return a compact JSON object at the end inside a fenced code block ```json ... ``` "
)

def load_api_key_from_secrets(secret_path: str = "secrets.toml") -> Optional[str]:
    if os.path.exists(secret_path):
        try:
            data = toml.load(secret_path)
            key = data.get("OPENAI_API_KEY") or data.get("openai_api_key")
            if key:
                return key
        except Exception:
            pass
    return os.getenv("OPENAI_API_KEY")

def get_openai_client(api_key: Optional[str] = None):
    key = api_key or load_api_key_from_secrets()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found in secrets.toml or environment")

    if OpenAIClient == "openai_v2":
        client = OpenAI_v2(api_key=key)
        return {"type": "v2", "client": client}
    elif OpenAIClient == "openai_v1":
        import openai as _openai
        _openai.api_key = key
        return {"type": "v1", "client": _openai}
    else:
        raise RuntimeError("No supported OpenAI client installed. Install 'openai' or new OpenAI SDK.")

def ask_openai(prompt: str, context: str, model: str = "gpt-4o-mini", api_key: Optional[str] = None) -> str:
    cl = get_openai_client(api_key=api_key)
    try:
        if cl["type"] == "v2":
            client = cl["client"]
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                    {"role": "user", "content": f"REQUEST:\n{prompt}\n\nCONTEXT:\n{context}"},
                ],
                temperature=0.2,
                max_tokens=1200,
            )
            return resp.choices[0].message.content or ""
        else:
            _openai = cl["client"]
            resp = _openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                    {"role": "user", "content": f"REQUEST:\n{prompt}\n\nCONTEXT:\n{context}"},
                ],
                temperature=0.2,
                max_tokens=1200,
            )
            return resp.choices[0].message["content"] or ""
    except Exception as e:
        raise

def initialize_session_state():
    if 'raw_texts' not in st.session_state:
        st.session_state.raw_texts = {}
    if 'results_rows' not in st.session_state:
        st.session_state.results_rows = []
    if 'scored_results' not in st.session_state:
        st.session_state.scored_results = None
    if 'jd_text' not in st.session_state:
        st.session_state.jd_text = ""
    if 'skills_list' not in st.session_state:
        st.session_state.skills_list = DEFAULT_SKILLS.copy()
    if 'shortlist_count' not in st.session_state:
        st.session_state.shortlist_count = 5
    if 'req_skills_list' not in st.session_state:
        st.session_state.req_skills_list = []

DEFAULT_SKILLS = [
    'Python','Data Analysis','Machine Learning','Communication','Project Management','Deep Learning','SQL',
    'Tableau','Excel','Java','JavaScript','React','Node.js','AWS','Azure','Docker','Kubernetes','Git','Linux',
    'Statistics','R','TensorFlow','PyTorch','Pandas','NumPy','Matplotlib','Seaborn','Scikit-learn','Spark',
    'HTML','CSS','MongoDB','MySQL','PostgreSQL','NoSQL'
]

def main():
    st.set_page_config(
        page_title="Resume Parser & Analyzer",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    setup_authentication()
    
    if not st.session_state.authenticated:
        show_login_section()
    else:
        show_main_application()

def show_login_section():
    st.title("🔐 Resume Analyzer")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login to Your Account")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username", key="login_username")
            password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")
            submit_login = st.form_submit_button("Login", type="primary", use_container_width=True)
            
            if submit_login:
                if username and password:
                    if login_user(username, password):
                        st.success(f"Welcome {username}!")
                        st.rerun()
                else:
                    st.error("Please enter both username and password")
        
        st.markdown("---")
        st.info("**Don't have an account?** Switch to the **Register** tab to create one!")
    
    with tab2:
        st.subheader("Create New Account")
        
        with st.form("register_form"):
            new_username = st.text_input("Choose Username", placeholder="Enter a unique username", key="reg_username")
            new_email = st.text_input("Email Address", placeholder="Enter your email", key="reg_email")
            new_password = st.text_input("Choose Password", type="password", placeholder="Create a password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password", key="reg_confirm")
            
            submit_register = st.form_submit_button("Create Account", type="primary", use_container_width=True)
            
            if submit_register:
                if not all([new_username, new_email, new_password, confirm_password]):
                    st.error("Please fill in all fields")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                elif len(new_password) < 4:
                    st.error("Password should be at least 4 characters long")
                else:
                    if register_user(new_username, new_password, new_email):
                        st.rerun()
        
        st.markdown("---")
        st.info("**Already have an account?** Switch to the **Login** tab to sign in!")

def show_main_application():
    st.title(f"📄 Resume Parser & Analyzer")
    st.write(f"Welcome **{st.session_state.user['username']}**!")
    
    with st.sidebar:
        st.write(f"Logged in as: **{st.session_state.user['username']}**")
        if st.button("Logout", use_container_width=True):
            logout_user()
            st.rerun()
        
        history = get_user_history()
        if history:
            st.metric("Total Resumes", history['resume_count'])
    
    initialize_session_state()
    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose a module",
        ["Upload Resumes", "Parse & View", "Job Match", "AI Chat", "Interview Scheduling", "Feedback Collection", "History & Analytics"]
    )
    
    if app_mode == "Upload Resumes":
        show_upload_section()
    elif app_mode == "Parse & View":
        show_parse_section()
    elif app_mode == "Job Match":
        show_match_section()
    elif app_mode == "AI Chat":
        show_chat_section()
    elif app_mode == "Interview Scheduling": 
        show_interview_scheduling_section()
    elif app_mode == "Feedback Collection":
        show_feedback_section()
    elif app_mode == "History & Analytics":
        show_history_section()

def show_upload_section():
    st.header("📤 Upload Resumes")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose resume files (PDF/DOCX)",
            type=['pdf', 'docx'],
            accept_multiple_files=True,
            help="Upload multiple PDF or DOCX files"
        )
    
    with col2:
        custom_skills = st.text_area(
            "Custom Skills (comma-separated)",
            value=", ".join(DEFAULT_SKILLS),
            help="Add or modify skills to look for in resumes",
            height=150
        )
    
    if uploaded_files:
        st.session_state.skills_list = [s.strip() for s in custom_skills.split(",") if s.strip()]
        
        if st.button("Process Uploaded Files", type="primary", use_container_width=True):
            with st.spinner("Processing resumes..."):
                session_id = str(uuid.uuid4())
                save_upload_session(len(uploaded_files), custom_skills, session_id)
                
                success_count = 0
                fail_count = 0
                processed_rows = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uf in enumerate(uploaded_files):
                    status_text.text(f"Processing {uf.name}... ({i+1}/{len(uploaded_files)})")
                    try:
                        text, fname = extract_text_from_uploaded(uf)
                        row = process_text_to_row(text, fname, st.session_state.skills_list)
                        
                        row['raw_text'] = text
                        
                        if save_resume_data(row, session_id):
                            success_count += 1
                            processed_rows.append(row)
                        else:
                            fail_count += 1
                            
                    except Exception as e:
                        fail_count += 1
                        processed_rows.append({
                            "File": uf.name,
                            "Name": "",
                            "Email": "",
                            "Mobile": "",
                            "Skills": "",
                            "Error": f"Processing failed: {e}"
                        })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                update_upload_session(session_id, success_count, fail_count, 'completed')
                
                st.session_state.results_rows = processed_rows
                for row in processed_rows:
                    if not row.get('Error') and 'raw_text' in row:
                        st.session_state.raw_texts[row['File']] = row['raw_text']
                
                status_text.text("Processing complete!")
                st.success(f"✅ Processed {success_count} files successfully, {fail_count} failed")
                log_user_action("upload_resumes", f"Uploaded {success_count} resumes")
    
    st.subheader("Current Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        history = get_user_history()
        total_resumes = history['resume_count'] if history else 0
        st.metric("Total Resumes in DB", total_resumes)
    with col2:
        successful_parses = len([r for r in st.session_state.results_rows if not r.get("Error")])
        st.metric("Current Session Success", successful_parses)
    with col3:
        errors = len([r for r in st.session_state.results_rows if r.get("Error")])
        st.metric("Current Session Errors", errors)

def show_parse_section():
    st.header("🔍 Parsed Resume Data")
    
    db_resumes = get_user_resumes()
    
    if not db_resumes and not st.session_state.results_rows:
        st.warning("No resumes processed yet. Please upload files first.")
        return
    
    if db_resumes:
        st.subheader("All Resumes from Database")
        db_df = pd.DataFrame(db_resumes)
        st.dataframe(db_df, use_container_width=True)
        
        st.subheader("Database Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Candidates", len(db_resumes))
        with col2:
            emails = len([r for r in db_resumes if r['candidate_email']])
            st.metric("With Email", emails)
        with col3:
            phones = len([r for r in db_resumes if r['candidate_phone']])
            st.metric("With Phone", phones)
        with col4:
            names = len([r for r in db_resumes if r['candidate_name']])
            st.metric("With Name", names)
    
    if st.session_state.results_rows:
        st.subheader("Current Session Results")
        df = pd.DataFrame(st.session_state.results_rows)
        df_view = df.drop(columns=['Error', 'raw_text'], errors='ignore')
        st.dataframe(df_view, use_container_width=True)
    
    if db_resumes:
        if st.button("Export All Data to CSV", use_container_width=True):
            csv = pd.DataFrame(db_resumes).to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="all_resume_data.csv",
                mime="text/csv",
                use_container_width=True
            )

def show_match_section():
    st.header("🎯 Job Description Matching")
    
    if not st.session_state.raw_texts:
        st.warning("No resumes processed yet. Please upload files first.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Job Description")
        jd_text = st.text_area(
            "Paste the job description here",
            height=200,
            value=st.session_state.get('jd_text', ''),
            help="Enter the job description to match against resumes",
            placeholder="Enter the job description including required skills, experience, and qualifications..."
        )
        st.session_state.jd_text = jd_text
    
    with col2:
        st.subheader("Matching Settings")
        use_faiss = st.checkbox("Use FAISS for faster matching", value=False)
        w_skill = st.slider("Skill Match Weight", 0.0, 1.0, 0.55, 0.05)
        
        max_candidates = len(st.session_state.raw_texts)
        shortlist_count = st.number_input(
            "Number of candidates to shortlist",
            min_value=1,
            max_value=max_candidates,
            value=min(5, max_candidates),
            help=f"Select how many top candidates to shortlist (max: {max_candidates})"
        )
        st.session_state.shortlist_count = shortlist_count
        
        req_skills = st.text_area(
            "Required Skills (comma-separated)",
            value=", ".join(st.session_state.skills_list),
            help="Skills that are required for this job",
            height=100
        )
    
    if st.button("Run Matching", type="primary", use_container_width=True) and jd_text.strip():
        with st.spinner("Computing matches..."):
            results = run_matching(jd_text, req_skills, use_faiss, w_skill)
            st.session_state.scored_results = results
            save_matching_session(jd_text, req_skills, use_faiss, w_skill, shortlist_count, results)
            
    if st.session_state.scored_results:
        show_match_results()

def run_matching(jd_text: str, req_skills_text: str, use_faiss: bool, w_skill: float):
    req_skills = [s.strip() for s in req_skills_text.split(",") if s.strip()]
    st.session_state.req_skills_list = req_skills
    
    file_order = list(st.session_state.raw_texts.keys())
    cand_texts = [st.session_state.raw_texts.get(f, "") for f in file_order]
    
    try:
        sims = compute_text_similarities(jd_text, cand_texts, use_faiss=use_faiss)
    except Exception as e:
        st.error(f"Error computing similarities: {e}")
        sims = None
    
    import numpy as _np
    if sims is None:
        sims = _np.zeros(len(cand_texts), dtype="float32")
    
    rows_df = []
    for i, fname in enumerate(file_order):
        base_row = next((r for r in st.session_state.results_rows if r["File"] == fname), None)
        skills_csv = base_row["Skills"] if base_row else ""
        skill_score = compute_skill_match(req_skills, skills_csv)
        text_sim = float(sims[i])
        match_score = round(( (1.0 - w_skill) * text_sim) + (w_skill * skill_score), 4)
        
        rows_df.append({
            "File": fname,
            "Name": base_row.get("Name", "") if base_row else "",
            "Email": base_row.get("Email", "") if base_row else "",
            "Mobile": base_row.get("Mobile", "") if base_row else "",
            "Skills": skills_csv,
            "TextSim": round(float(text_sim), 4),
            "SkillMatch": round(float(skill_score), 4),
            "MatchScore": match_score
        })
    
    rows_sorted = sorted(rows_df, key=lambda x: (x["MatchScore"], x["TextSim"], x["SkillMatch"]), reverse=True)
    return rows_sorted

def show_match_results():
    st.subheader("Matching Results")
    
    shortlist_count = st.session_state.shortlist_count
    shortlisted = st.session_state.scored_results[:shortlist_count]
    df = pd.DataFrame(shortlisted)
    
    st.write(f"## Top {shortlist_count} Shortlisted Candidates:")
    st.dataframe(df, use_container_width=True)
    
    st.subheader("Shortlist Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_match = df['MatchScore'].mean()
        st.metric("Average Match Score", f"{avg_match:.2%}")
    with col2:
        avg_skill = df['SkillMatch'].mean()
        st.metric("Average Skill Match", f"{avg_skill:.2%}")
    with col3:
        avg_text = df['TextSim'].mean()
        st.metric("Average Text Similarity", f"{avg_text:.2%}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Match Score Distribution")
        st.bar_chart(df.set_index('Name')['MatchScore'])

    st.subheader("Candidate Details")
    for i, candidate in enumerate(shortlisted):
        with st.expander(f"{i+1}. {candidate['Name']} - Score: {candidate['MatchScore']:.2%}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Email:** {candidate['Email']}")
                st.write(f"**Phone:** {candidate['Mobile']}")
                st.write(f"**File:** {candidate['File']}")
            with col2:
                st.write(f"**Text Similarity:** {candidate['TextSim']:.2%}")
                st.write(f"**Skill Match:** {candidate['SkillMatch']:.2%}")
                st.write(f"**Overall Score:** {candidate['MatchScore']:.2%}")
            
            st.write(f"**Skills:** {candidate['Skills']}")
    
    st.subheader("Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Shortlisted Candidates", use_container_width=True):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Shortlist CSV",
                data=csv,
                file_name=f"shortlisted_candidates_{shortlist_count}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        if st.button("Export All Results", use_container_width=True):
            all_df = pd.DataFrame(st.session_state.scored_results)
            csv = all_df.to_csv(index=False)
            st.download_button(
                label="Download All Results CSV",
                data=csv,
                file_name="all_matching_results.csv",
                mime="text/csv",
                use_container_width=True
            )

def show_history_section():
    if st.button("Initialize Feedback Tables", type="secondary"):
        create_feedback_tables()
        st.success("Feedback tables initialized!")
        
def show_chat_section():
    st.header("🤖 AI Resume Assistant")
    
    if not st.session_state.raw_texts:
        st.warning("No resumes processed yet. Please upload files first.")
        return
    
    st.info("Ask questions about the uploaded resumes. The AI will analyze the content and provide insights.")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        prompt = st.text_area(
            "Your question",
            placeholder="e.g., Who has the most experience in machine learning? List candidates with Python skills.",
            height=100
        )
    
    with col2:
        shortlist_only = st.checkbox("Use shortlisted only", value=True)
        model_choice = st.selectbox("Model", ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"])
    
    with col3:
        max_chars = st.number_input("Max context chars", min_value=1000, max_value=100000, value=30000)
    
    if st.button("Ask AI", type="primary", use_container_width=True) and prompt.strip():
        with st.spinner("Analyzing resumes..."):
            try:
                context = build_context_from_state(max_chars, shortlist_only)
                answer = ask_openai(prompt, context, model=model_choice)
                
                save_chat_history(prompt, answer, model_choice, context[:500])
                
                st.subheader("AI Response")
                st.write(answer)
                
            except Exception as e:
                st.error(f"Error calling OpenAI: {e}")

def build_context_from_state(max_chars: int, shortlist_only: bool = False) -> str:
    raw_map = st.session_state.raw_texts
    if not raw_map:
        return ""
    
    ordered_files = list(raw_map.keys())
    if shortlist_only and st.session_state.scored_results:
        shortlist_count = st.session_state.shortlist_count
        shortlisted_files = [r["File"] for r in st.session_state.scored_results[:shortlist_count]]
        ordered_files = [f for f in ordered_files if f in shortlisted_files]
    
    chunks = []
    used = 0
    for f in ordered_files:
        text = raw_map.get(f, "")
        if not text:
            continue
        header = f"\n\n===== FILE: {f} =====\n"
        take = max_chars - used - len(header)
        if take <= 0:
            break
        snippet = text[:take]
        chunks.append(header + snippet)
        used += len(header) + len(snippet)
    return "".join(chunks)

def show_history_section():
    st.header("📊 Your Activity History")
    
    history = get_user_history()
    if not history:
        st.info("No history found")
        return
    
    st.subheader("Overall Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Resumes", history['resume_count'])
    with col2:
        total_uploads = len(history['upload_sessions'])
        st.metric("Upload Sessions", total_uploads)
    with col3:
        total_matches = len(history['matching_sessions'])
        st.metric("Matching Sessions", total_matches)
    with col4:
        successful_uploads = sum(1 for s in history['upload_sessions'] if s['status'] == 'completed')
        st.metric("Successful Uploads", successful_uploads)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Recent Upload Sessions")
        if history['upload_sessions']:
            upload_df = pd.DataFrame(history['upload_sessions'])
            if 'start_time' in upload_df.columns:
                upload_df['start_time'] = pd.to_datetime(upload_df['start_time']).dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(upload_df, use_container_width=True)
        else:
            st.info("No upload sessions found")
    
    with col2:
        st.subheader("Recent Matching Sessions")
        if history['matching_sessions']:
            match_df = pd.DataFrame(history['matching_sessions'])
            if 'created_at' in match_df.columns:
                match_df['created_at'] = pd.to_datetime(match_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(match_df, use_container_width=True)
        else:
            st.info("No matching sessions found")

if __name__ == "__main__":
    main()