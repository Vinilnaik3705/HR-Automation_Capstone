import os
import re
import io
import sys
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import streamlit as st
from docx import Document
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
from tempfile import NamedTemporaryFile


try:
    import faiss
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


# Regex / Normalization / Heuristics

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


# Extraction: Name / Email / Phone / Skills

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


# File text extraction

def extract_text_from_pdf_path(pdf_path: str) -> str:
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
    try:
        doc = Document(docx_path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return text
    except Exception as e:
        return f"__ERROR__: DOCX read failed: {e}"

def extract_text_from_uploaded(uploaded_file) -> Tuple[str, str]:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
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

def extract_text_from_path(file_path: str) -> Tuple[str, str]:
    suffix = os.path.splitext(file_path)[1].lower()
    if suffix == ".pdf":
        return extract_text_from_pdf_path(file_path), file_path
    elif suffix == ".docx":
        return extract_text_from_docx_path(file_path), file_path
    else:
        return "__ERROR__: Unsupported file type", file_path


# Embeddings / Similarity

def get_embedder():
    if "embedder" in st.session_state:
        return st.session_state["embedder"]
    model = None
    if SentenceTransformer is not None:
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            st.warning(f"Embedding model load failed ({e}). Falling back to TF-IDF.")
            model = None
    else:
        st.info("sentence-transformers not installed; will use TF-IDF or Jaccard fallback.")
    st.session_state["embedder"] = model
    return model

def embed_texts(texts: List[str]) -> Optional[np.ndarray]:
    model = get_embedder()
    if model is None:
        return None
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embs, dtype="float32")

def tfidf_cosine(jd_text: str, candidate_texts: List[str]) -> Optional[np.ndarray]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return None
    vec = TfidfVectorizer(stop_words="english", max_features=50000)
    X = vec.fit_transform([jd_text] + candidate_texts)
    sims = cosine_similarity(X[0:1], X[1:]).flatten()
    return sims.astype("float32")

def jaccard_similarity(jd_text: str, candidate_texts: List[str]) -> np.ndarray:
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
    return np.asarray(res, dtype="float32")

def compute_text_similarities(jd_text: str, candidate_texts: List[str], use_faiss: bool) -> np.ndarray:

    embs = embed_texts(candidate_texts)
    if embs is not None:
        q = embed_texts([jd_text])
        if q is None:
            sims = np.zeros(len(candidate_texts), dtype="float32")
        else:
            qv = q[0].astype("float32")
            if use_faiss and faiss is not None:
                index = faiss.IndexFlatIP(embs.shape[1])
                index.add(embs)
                D, I = index.search(np.expand_dims(qv, 0), len(candidate_texts))
                # D is already in order of added items; we need to reorder to original order
                # Since we did a single query for all items, use pure dot product instead to preserve order
                # For stability, fall back to dot product here:
                sims = (embs @ qv).astype("float32")
            else:
                sims = (embs @ qv).astype("float32")  # cosine similarity with normalized embeddings
        # Normalize to [0,1]
        sims = np.clip((sims + 1.0) / 2.0, 0.0, 1.0)
        return sims

    # 2) Fallback: TF-IDF cosine
    sims = tfidf_cosine(jd_text, candidate_texts)
    if sims is not None:
        return np.clip(sims, 0.0, 1.0)

    # 3) Fallback: word Jaccard
    return jaccard_similarity(jd_text, candidate_texts)


# Resume processing (row creation)

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


# Skill matching / Scoring

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


# Streamlit UI

def main():
    st.set_page_config(page_title="Resume Parser + Matching (Bulk)", page_icon="📄", layout="wide")

    st.title("📄 Resume Parser")
    st.caption("Upload many resumes (PDF/DOCX) or point to a local folder. Extract contact info and skills, compute profile match scores with cosine similarity (Sentence-Transformers, FAISS optional), and auto-generate shortlists.")

    if "raw_texts" not in st.session_state:
        st.session_state["raw_texts"] = {}

    with st.expander("Configure global skills to detect", expanded=False):
        default_skills = [
            'Python','Data Analysis','Machine Learning','Communication','Project Management','Deep Learning','SQL',
            'Tableau','Excel','Java','JavaScript','React','Node.js','AWS','Azure','Docker','Kubernetes','Git','Linux',
            'Statistics','R','TensorFlow','PyTorch','Pandas','NumPy','Matplotlib','Seaborn','Scikit-learn','Spark',
            'HTML','CSS','MongoDB','MySQL','PostgreSQL','NoSQL'
        ]
        skills_text = st.text_area("Comma-separated list of skills", value=", ".join(default_skills), height=100)
        user_skills = [s.strip() for s in skills_text.split(",") if s.strip()]

    tab_upload, tab_folder, tab_match = st.tabs(["Upload multiple files", "Process local folder (advanced)", "Match & Shortlist"])

    if "results_df" not in st.session_state:
        st.session_state["results_df"] = None

    # 1) Upload mode
    with tab_upload:
        uploaded_files = st.file_uploader(
            "Upload multiple resumes (PDF/DOCX)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            help="Drag and drop many files here"
        )
        if uploaded_files:
            if st.button("Process uploaded resumes", type="primary"):
                rows: List[Dict[str, Any]] = []
                total = len(uploaded_files)
                progress_bar = st.progress(0)
                status = st.empty()

                raw_map = st.session_state["raw_texts"]
                raw_map.clear()

                for i, uf in enumerate(uploaded_files):
                    text, fname = extract_text_from_uploaded(uf)
                    raw_map[os.path.basename(fname)] = "" if text.startswith("__ERROR__") else text
                    row = process_text_to_row(text, fname, user_skills)
                    rows.append(row)

                    status.write(f"Processed {i+1} of {total}: {row['File']}")
                    progress_bar.progress((i + 1) / total)

                st.session_state["results_df"] = pd.DataFrame(rows)
                st.success(f"Completed processing {total} resumes.")
                st.dataframe(st.session_state["results_df"], use_container_width=True)

    # 2) Folder mode
    with tab_folder:
        st.warning("Folder mode reads files from your local machine where Streamlit is running. Use only when running locally.")
        folder_path = st.text_input("Folder path (e.g., C:\\\\resumes or /Users/me/resumes)")
        recurse = st.checkbox("Include subfolders", value=True)
        exts = st.multiselect("Allowed extensions", [".pdf", ".docx"], default=[".pdf", ".docx"])

        if st.button("Scan and process folder", type="primary", disabled=not folder_path):
            file_paths: List[str] = []
            if os.path.isdir(folder_path):
                if recurse:
                    for root, _, files in os.walk(folder_path):
                        for f in files:
                            if os.path.splitext(f)[1].lower() in set(exts):
                                file_paths.append(os.path.join(root, f))
                else:
                    for f in os.listdir(folder_path):
                        p = os.path.join(folder_path, f)
                        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in set(exts):
                            file_paths.append(p)
            else:
                st.error("Folder path not found.")
                file_paths = []

            if file_paths:
                rows: List[Dict[str, Any]] = []
                total = len(file_paths)
                progress_bar = st.progress(0)
                status = st.empty()
                st.info(f"Found {total} files to process.")

                raw_map = st.session_state["raw_texts"]
                raw_map.clear()

                for i, path in enumerate(file_paths):
                    text, fname = extract_text_from_path(path)
                    raw_map[os.path.basename(fname)] = "" if text.startswith("__ERROR__") else text
                    row = process_text_to_row(text, fname, user_skills)
                    rows.append(row)

                    status.write(f"Processed {i+1} of {total}: {os.path.basename(fname)}")
                    progress_bar.progress((i + 1) / total)

                st.session_state["results_df"] = pd.DataFrame(rows)
                st.success(f"Completed processing {total} resumes.")
                st.dataframe(st.session_state["results_df"], use_container_width=True)
            else:
                st.warning("No files found for the given folder and extensions.")

    # 3) Matching & Shortlisting
    with tab_match:
        st.subheader("Profile Matching & Auto-Shortlisting")
        results_df = st.session_state.get("results_df")
        if results_df is None or results_df.empty:
            st.info("Please process some resumes first (Upload or Folder tab).")
        else:
            colA, colB = st.columns([2, 1])
            with colA:
                jd_text = st.text_area(
                    "Paste Job Description / Ideal Profile",
                    height=180,
                    placeholder="Paste responsibilities, required skills, and qualifications here..."
                )
                req_skills_text = st.text_input(
                    "Required skills for scoring (comma-separated, overrides globals if provided)",
                    value=""
                )
            with colB:
                use_faiss = st.checkbox("Use FAISS (if installed)", value=False, help="For large batches; falls back automatically if unavailable.")
                w_skill = st.slider("Skill weight", 0.0, 1.0, 0.55, 0.05)
                w_text = 1.0 - w_skill
                st.write(f"Text similarity weight: {w_text:.2f}")

                strategy = st.radio("Shortlist strategy", ["Top-K", "Threshold"], horizontal=True)
                top_k = st.number_input("Top-K", min_value=1, max_value=5000, value=min(20, len(results_df)))
                threshold = st.slider("Score threshold", 0.0, 1.0, 0.65, 0.01)

            if st.button("Compute match scores", type="primary", disabled=(not jd_text.strip())):
                file_order = results_df["File"].tolist()
                raw_map = st.session_state["raw_texts"]
                cand_texts = [raw_map.get(f, "") for f in file_order]
                text_sims = compute_text_similarities(jd_text, cand_texts, use_faiss=use_faiss)
                if req_skills_text.strip():
                    req_skills = [s.strip() for s in req_skills_text.split(",") if s.strip()]
                else:
                    req_skills = [s.strip() for s in skills_text.split(",") if s.strip()]

                skill_scores = results_df["Skills"].apply(lambda s: compute_skill_match(req_skills, s)).values.astype("float32")

                # Final weighted score
                final_scores = (w_text * text_sims) + (w_skill * skill_scores)

                scored_df = results_df.copy()
                scored_df["TextSim"] = np.round(text_sims, 4)
                scored_df["SkillMatch"] = np.round(skill_scores, 4)
                scored_df["MatchScore"] = np.round(final_scores, 4)
                scored_df.sort_values(["MatchScore", "TextSim", "SkillMatch"], ascending=False, inplace=True, ignore_index=True)

                st.session_state["scored_df"] = scored_df

                st.success("Match scores computed.")
                st.dataframe(scored_df, use_container_width=True)

                # Dashboard: bar chart top 20
                try:
                    import plotly.express as px  # type: ignore
                    topN = scored_df.head(min(20, len(scored_df)))
                    fig = px.bar(topN, x="File", y="MatchScore", color="SkillMatch", title="Top Candidates by MatchScore", height=420)
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.bar_chart(scored_df.set_index("File")["MatchScore"].head(min(20, len(scored_df))))

            # Shortlist generation + downloads
            scored_df = st.session_state.get("scored_df")
            if scored_df is not None and not scored_df.empty:
                if strategy == "Top-K":
                    shortlist_df = scored_df.head(int(top_k)).copy()
                else:
                    shortlist_df = scored_df[scored_df["MatchScore"] >= float(threshold)].copy()

                st.markdown("### Shortlist")
                st.dataframe(shortlist_df, use_container_width=True)

                # Downloads
                buf_all = io.BytesIO()
                with pd.ExcelWriter(buf_all, engine="openpyxl") as writer:
                    scored_df.to_excel(writer, index=False, sheet_name="AllCandidates")
                buf_all.seek(0)

                buf_short = io.BytesIO()
                with pd.ExcelWriter(buf_short, engine="openpyxl") as writer:
                    shortlist_df.to_excel(writer, index=False, sheet_name="Shortlist")
                buf_short.seek(0)

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download All (Scored) Excel",
                        data=buf_all.getvalue(),
                        file_name=f"resumes_scored_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                with col2:
                    st.download_button(
                        label="✅ Download Shortlist Excel",
                        data=buf_short.getvalue(),
                        file_name=f"shortlist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )


    # Download raw parsing (without scores) if any

    results_df = st.session_state.get("results_df")
    if results_df is not None and not results_df.empty:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            results_df.to_excel(writer, index=False, sheet_name="Resumes")
        buffer.seek(0)

        st.download_button(
            label="📥 Download Parsed (No Scores) Excel",
            data=buffer.getvalue(),
            file_name=f"resumes_parsed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

if __name__ == "__main__":

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        get_script_run_ctx = None

    ctx = None
    if get_script_run_ctx:
        try:
            ctx = get_script_run_ctx()
        except Exception:
            ctx = None

    if ctx is not None:
        main()
    else:

        try:
            from streamlit.web import cli as stcli  # type: ignore
            import os as _os
            sys.argv = ["streamlit", "run", _os.path.abspath(__file__)]
            sys.exit(stcli.main())
        except Exception:

            import subprocess, os as _os
            subprocess.run([sys.executable, "-m", "streamlit", "run", _os.path.abspath(__file__)])