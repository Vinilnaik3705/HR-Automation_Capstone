import streamlit as st
import re
import fitz
from docx import Document
import tempfile
import os
from typing import List, Optional

st.set_page_config(
    page_title="Perfect Resume Parser",
    page_icon="📄",
    layout="wide"
)


def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""


def extract_text_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""


def extract_name(text):
    lines = text.split('\n')
    for i, line in enumerate(lines[:10]):
        line = line.strip()
        words = line.split()
        if 2 <= len(words) <= 4:
            name_candidate = True
            for word in words:
                if (not word[0].isupper() or
                        len(word) < 2 or
                        len(word) > 15 or
                        any(char.isdigit() for char in word)):
                    name_candidate = False
                    break
            if name_candidate and not any(term in line.lower() for term in
                                          ['email', 'phone', '@', 'http', 'linkedin', 'github']):
                return line
    return None


def extract_contact_number(text):
    patterns = [
        r"\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
        r"\+\d{1,3}\s?\(\d{1,4}\)\s?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
        r"\b91[-.\s]?\d{5}[-.\s]?\d{5}\b",
        r"\b\+91[-.\s]?\d{5}[-.\s]?\d{5}\b",
        r"\b0\d{5}[-.\s]?\d{5}\b",
        r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
        r"\b\(\d{3}\)[-.\s]?\d{3}[-.\s]?\d{4}\b",
        r"\b\d{10}\b",
        r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b",
        r"\b\d{5}[-.\s]\d{5}\b",
        r"\b(?:6|7|8|9)\d{9}\b",
        r"\b\d{4}[-.\s]?\d{3}[-.\s]?\d{3}\b",
    ]
    all_matches = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            cleaned = re.sub(r'[^\d\+]', '', match)
            if 8 <= len(cleaned.replace('+', '')) <= 15:
                if match not in all_matches:
                    all_matches.append(match)
    if all_matches:
        def phone_score(phone):
            score = 0
            if '+' in phone:
                score += 2
            if len(re.sub(r'\D', '', phone)) == 10:
                score += 1
            if re.search(r'[6-9]\d{9}', phone):
                score += 1
            return score

        all_matches.sort(key=phone_score, reverse=True)
        return all_matches[0]
    return None


def extract_email(text):
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    matches = re.findall(pattern, text)
    if matches:
        for email in matches:
            if not any(placeholder in email.lower() for placeholder in
                       ['example.com', 'test.com', 'placeholder']):
                return email
    return None


def extract_skills(text, skills_list):
    skills = []
    text_lower = text.lower()
    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill.lower()))
        if re.search(pattern, text_lower):
            skills.append(skill)
    return skills


# Text normalization (kills OCR/punctuation noise)
def norm_text(s: str) -> str:
    s = re.sub(r"[•·∙●◦➤➥▪■◆▶►]+", " ", s)  # bullets → space
    s = re.sub(r"[,_–—\-–]+", " ", s)  # dashes/commas → space
    s = re.sub(r"\s+", " ", s).strip()  # collapse spaces
    return s


# Core ontology labels we'll return
DEGREE_LABELS = [
    "BTECH", "BE", "BSC", "BA", "BCOM", "BCA",
    "MTECH", "ME", "MSC", "MA", "MBA", "MCA",
    "PHD", "DIPLOMA", "PGD", "MS", "BPHARM", "MPHARM", "MBBS", "MD", "BBA", "LLB", "LLM"
]

# Order matters: longer/full forms before short forms to avoid early matches
DEGREE_PATTERNS = [
    (r"\b(bachelor(?:'s)?\s+of\s+technology)\b", "BTECH"),
    (r"\b(master(?:'s)?\s+of\s+technology)\b", "MTECH"),
    (r"\b(bachelor(?:'s)?\s+of\s+engineering)\b", "BE"),
    (r"\b(master(?:'s)?\s+of\s+engineering)\b", "ME"),
    (r"\b(bachelor(?:'s)?\s+of\s+science)\b", "BSC"),
    (r"\b(master(?:'s)?\s+of\s+science)\b", "MSC"),
    (r"\b(bachelor(?:'s)?\s+of\s+arts)\b", "BA"),
    (r"\b(master(?:'s)?\s+of\s+arts)\b", "MA"),
    (r"\b(bachelor(?:'s)?\s+of\s+commerce)\b", "BCOM"),
    (r"\b(master(?:'s)?\s+of\s+business\s+administration)\b", "MBA"),
    (r"\b(master(?:'s)?\s+of\s+computer\s+applications?)\b", "MCA"),
    (r"\b(post\s*graduate\s*diploma)\b", "PGD"),
    (r"\b(diploma)\b", "DIPLOMA"),
    (r"\b(doctor\s+of\s+philosophy|ph\.?\s*d)\b", "PHD"),
    (r"\b(bachelor(?:'s)?\s+of\s+computer\s+applications?)\b", "BCA"),
    (r"\b(bachelor(?:'s)?\s+of\s+business\s+administration)\b", "BBA"),
    (r"\b(bachelor(?:'s)?\s+of\s+pharmacy)\b", "BPHARM"),
    (r"\b(master(?:'s)?\s+of\s+pharmacy)\b", "MPHARM"),
    (r"\b(bachelor(?:'s)?\s+of\s+medicine\b.*?\bsurgery)\b|\bmbbs\b", "MBBS"),
    (r"\bdoctor\b.*\bmedicine\b|\bmd\b", "MD"),
    (r"\b(bachelor(?:'s)?\s+of\s+laws)\b|\bll\.?b\b", "LLB"),
    (r"\b(master(?:'s)?\s+of\s+laws)\b|\bll\.?m\b", "LLM"),
    # Abbrev (put after long forms)
    (r"\bm\.?\s?tech\b", "MTECH"),
    (r"\bb\.?\s?tech\b", "BTECH"),
    (r"\bm\.?\s?e\b", "ME"),
    (r"\bb\.?\s?e\b", "BE"),
    (r"\bm\.?\s?sc\b", "MSC"),
    (r"\bb\.?\s?sc\b", "BSC"),
    (r"\bm\.?\s?a\b", "MA"),
    (r"\bb\.?\s?a\b", "BA"),
    (r"\bmba\b", "MBA"),
    (r"\bmca\b", "MCA"),
    (r"\bbca\b", "BCA"),
    (r"\bbba\b", "BBA"),
    (r"\bpg\s*diploma\b|\bpgdm\b", "PGD"),
    (r"\bdiploma\b", "DIPLOMA"),
    (r"\bms\b", "MS"),  # some resumes use US-style MS
]

# Common majors/branches
MAJOR_RE = re.compile(
    r"\b(?:in|of)\s+(?P<major>"
    r"(computer\s*science(?:\s*&\s*engineering)?|cse|information\s*technology|it|"
    r"electronics(?:\s*(?:and|&)\s*communication)?|ece|eee|mechanical|civil|electrical|"
    r"ai|artificial\s*intelligence|machine\s*learning|data\s*science|statistics|mathematics|"
    r"biotechnology|chemical|physics|chemistry|mechatronics|instrumentation|"
    r"cyber\s*security|software\s*engineering)"
    r")\b",
    re.I
)


def extract_degree_and_major(block: str):
    """
    block: one education line/group. returns (degree_norm, major_norm, span)
    """
    line = norm_text(block)
    line_low = line.lower()

    degree_norm = None
    matched_span = None

    # pass 1: long/full forms first
    for pat, norm in DEGREE_PATTERNS:
        m = re.search(pat, line_low, flags=re.I)
        if m:
            degree_norm = norm
            matched_span = m.span()
            break

    # pass 2: if nothing matched, allow very loose patterns like 'B Tech'
    if not degree_norm:
        m = re.search(r"\b(b\s*\.?\s*tech|m\s*\.?\s*tech|b\s*\.?\s*e|m\s*\.?\s*e)\b", line_low)
        if m:
            raw = re.sub(r"\s|\.", "", m.group(0)).upper()
            degree_norm = {"BTECH": "BTECH", "MTECH": "MTECH", "BE": "BE", "ME": "ME"}.get(
                {"BTECH": "BTECH", "MTECH": "MTECH", "BE": "BE", "ME": "ME"}.get(raw, raw), raw
            )
            matched_span = m.span()

    # major (look around match; if not, on whole line)
    major = None
    search_text = line if not matched_span else line[max(0, matched_span[0] - 40): matched_span[1] + 60]
    mm = MAJOR_RE.search(search_text)
    if not mm:
        mm = MAJOR_RE.search(line)
    if mm:
        major = mm.group("major")
        # normalize short forms
        major = (major
                 .replace("&", "and")
                 .replace("cse", "Computer Science and Engineering")
                 .replace("ece", "Electronics and Communication")
                 .replace("eee", "Electrical and Electronics")
                 .strip().title())

        # Special-case IT
        if major.lower() in {"it"}:
            major = "Information Technology"

    return degree_norm, major


def extract_degree_from_parentheses(block: str):
    line = norm_text(block)
    # e.g., "B.Tech (Computer Science and Engineering)"
    m = re.search(r"\(([^)]+)\)", line)
    degree, major = extract_degree_and_major(line)
    if not major and m:
        # try major inside parentheses
        mm = MAJOR_RE.search(m.group(1))
        if mm:
            major = mm.group("major").replace("&", "and").title()
            if major.lower() == "It": major = "Information Technology"
    return degree, major


YEAR_RE = re.compile(r"\b(19|20)\d{2}\b(?:\s*[-–]\s*(19|20)\d{2}\b)?")


def parse_education_entry(block: str):
    deg, maj = extract_degree_from_parentheses(block)
    # give small bonus if year exists (typical edu line)
    has_year = bool(YEAR_RE.search(block))
    score = (1.0 if deg else 0.0) + (0.5 if maj else 0.0) + (0.2 if has_year else 0.0)
    return {"degree": deg, "major": maj, "score": score, "raw": block}


def extract_degrees(edu_section_text: str):
    # split into candidate entries (bullets/blank-line/year boundaries)
    chunks = []
    buf = []
    for ln in edu_section_text.splitlines():
        if re.match(r"^\s*[•\-\u2022]\s+", ln) or YEAR_RE.search(ln):
            if buf:
                chunks.append("\n".join(buf).strip())
                buf = []
        if ln.strip():
            buf.append(ln)
    if buf: chunks.append("\n".join(buf).strip())

    # rank entries by score; keep those with a degree label
    parsed = [parse_education_entry(c) for c in chunks]
    parsed.sort(key=lambda x: x["score"], reverse=True)
    return [p for p in parsed if p["degree"]]


def extract_education(text):
    # First try to find the education section
    lines = text.split('\n')
    education_section = []
    in_education_section = False

    for line in lines:
        if re.search(r"(?i)^education|academic|qualification", line):
            in_education_section = True
            continue
        if in_education_section:
            # Stop at next major section
            if re.search(r"(?i)^experience|work history|skills|projects|certification", line):
                break
            education_section.append(line)

    # If we found an education section, use the advanced parser
    if education_section:
        edu_text = "\n".join(education_section)
        entries = extract_degrees(edu_text)

        # Format the results for display
        education = []
        for entry in entries:
            degree_str = entry["degree"]
            if entry["major"]:
                degree_str += f" in {entry['major']}"
            education.append(degree_str)

        return education

    # Fallback: use the old method if no education section found
    education = []
    degree_patterns = [
        r"(?i)\bB\.?\s*Tech\.?(?:\s+[A-Za-z\s]{1,20})?(?=\s|$|,|\.)",
        r"(?i)\bM\.?\s*Tech\.?(?:\s+[A-Za-z\s]{1,20})?(?=\s|$|,|\.)",
        r"(?i)\bB\.?\s*E\.?(?:\s+[A-Za-z\s]{1,20})?(?=\s|$|,|\.)",
        r"(?i)\bB\.?\s*Sc\.?(?:\s+[A-Za-z\s]{1,20})?(?=\s|$|,|\.)",
        r"(?i)\bM\.?\s*Sc\.?(?:\s+[A-Za-z\s]{1,20})?(?=\s|$|,|\.)",
        r"(?i)\bPh\.?\s*D\.?(?:\s+[A-Za-z\s]{1,20})?(?=\s|$|,|\.)",
        r"(?i)\bBachelor\s+of\s+[A-Za-z]+(?:\s+[A-Za-z]+)?(?=\s|$|,|\.)",
        r"(?i)\bMaster\s+of\s+[A-Za-z]+(?:\s+[A-Za-z]+)?(?=\s|$|,|\.)",
        r"(?i)\bDoctorate\s+in\s+[A-Za-z]+(?:\s+[A-Za-z]+)?(?=\s|$|,|\.)",
        r"(?i)\bClass\s+XII(?:\s+[A-Za-z]+)?(?=\s|$|,|\.)",
        r"(?i)\bClass\s+X(?:\s+[A-Za-z]+)?(?=\s|$|,|\.)",
        r"(?i)\bHigh\s+School(?:\s+[A-Za-z]+)?(?=\s|$|,|\.)",
        r"(?i)\bSenior\s+Secondary(?:\s+[A-Za-z]+)?(?=\s|$|,|\.)",
        r"(?i)\bDiploma(?:\s+[A-Za-z\s]{1,20})?(?=\s|$|,|\.)",
    ]

    for pattern in degree_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            match = match.strip()
            if match and len(match) > 3 and match not in education:
                education.append(match)

    return education


def extract_college(text):
    colleges = []
    college_patterns = [
        r"(?i)Indian Institute of Technology\s+[A-Za-z\s\(\)]+",
        r"(?i)National Institute of Technology\s+[A-Za-z\s\(\)]+",
        r"(?i)Indian Institute of Science\s+[A-Za-z\s\(\)]+",
        r"(?i)Birla Institute of Technology and Science\s+[A-Za-z\s\(\)]+",
        r"(?i)IIT\s+[A-Za-z\s\(\)]+",
        r"(?i)NIT\s+[A-Za-z\s\(\)]+",
        r"(?i)BITS\s+[A-Za-z\s\(\)]+",
        r"(?i)IISc\s+[A-Za-z\s\(\)]+",
        r"(?i)[A-Z][a-z]+\s+University(?:\s+of\s+[A-Za-z\s]+)?",
        r"(?i)[A-Z][a-z]+\s+College(?:\s+of\s+[A-Za-z\s]+)?",
        r"(?i)[A-Z][a-z]+\s+Institute of Technology(?:\s+[A-Za-z\s]+)?",
        r"(?i)[A-Z][a-z]+\s+Institute of Science(?:\s+[A-Za-z\s]+)?",
    ]
    lines = text.split('\n')
    for i, line in enumerate(lines):
        line = line.strip()
        for pattern in college_patterns:
            match = re.search(pattern, line)
            if match:
                college = match.group(0).strip()
                if college not in colleges and len(college) > 10:
                    colleges.append(college)
        if i < len(lines) - 2:
            two_lines = line + " " + lines[i + 1].strip()
            for pattern in college_patterns:
                match = re.search(pattern, two_lines)
                if match:
                    college = match.group(0).strip()
                    if college not in colleges and len(college) > 10:
                        colleges.append(college)
            three_lines = two_lines + " " + lines[i + 2].strip()
            for pattern in college_patterns:
                match = re.search(pattern, three_lines)
                if match:
                    college = match.group(0).strip()
                    if college not in colleges and len(college) > 10:
                        colleges.append(college)
    if colleges:
        unique_colleges = list(set(colleges))
        unique_colleges.sort(key=len, reverse=True)
        best_college = unique_colleges[0]
        best_college = re.sub(r'\s+', ' ', best_college)
        best_college = re.sub(r'[^\w\s\(\)\-]', '', best_college)
        return best_college
    return None


def extract_experience(text):
    experience = []
    patterns = [
        r"(?i)(\d+\+?\s*years?[\s\w]*experience)",
        r"(?i)(\d+\s*[-–]\s*\d+\s*years?)",
        r"(?i)(\d+\s*\+\s*years?)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            if match.strip() not in experience:
                experience.append(match.strip())
    return experience


def main():
    st.title("🎯 Resume Screening Agent")
    st.markdown("Upload a resume (PDF or DOCX) for precise information extraction")
    predefined_skills = [
        'Python', 'Data Analysis', 'Machine Learning', 'Communication',
        'Project Management', 'Deep Learning', 'SQL', 'Tableau', 'Excel',
        'Java', 'JavaScript', 'React', 'Node.js', 'AWS', 'Azure', 'Docker',
        'Kubernetes', 'Git', 'Linux', 'Statistics', 'R', 'TensorFlow', 'PyTorch',
        'Pandas', 'NumPy', 'Matplotlib', 'Seaborn', 'Scikit-learn', 'Spark',
        'HTML', 'CSS', 'MongoDB', 'MySQL', 'PostgreSQL', 'NoSQL'
    ]
    uploaded_file = st.file_uploader(
        "Choose a resume file",
        type=['pdf', 'docx'],
        help="Upload PDF or DOCX format"
    )
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        try:
            if uploaded_file.name.lower().endswith('.pdf'):
                text = extract_text_from_pdf(tmp_path)
            else:
                text = extract_text_from_docx(tmp_path)
            if text:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.subheader("📊 Extracted Information")
                    name = extract_name(text)
                    phone = extract_contact_number(text)
                    email = extract_email(text)
                    education = extract_education(text)
                    college = extract_college(text)
                    skills = extract_skills(text, predefined_skills)
                    experience = extract_experience(text)
                    if name:
                        st.success(f"**👤 Name:** {name}")
                    else:
                        st.warning("Name not found")
                    if phone:
                        st.info(f"**📞 Phone:** {phone}")
                    else:
                        st.warning("Phone not found")
                    if email:
                        st.info(f"**📧 Email:** {email}")
                    else:
                        st.warning("Email not found")
                    if experience:
                        st.success("**💼 Experience:**")
                        for exp in experience:
                            st.write(f"• {exp}")
                    if education:
                        st.success("**🎓 Education Degrees:**")
                        for edu in education:
                            st.write(f"• {edu}")
                    else:
                        st.warning("Education not found")
                    if college:
                        st.info(f"**🏫 College/University:** {college}")
                    else:
                        st.warning("College not found")
                    if skills:
                        st.success("**🛠️ Skills:**")
                        cols = st.columns(3)
                        for i, skill in enumerate(skills):
                            cols[i % 3].write(f"• {skill}")
                    else:
                        st.warning("No skills found")
                with col2:
                    st.subheader("📝 Raw Text Preview")
                    st.text_area(
                        "Extracted Text (first 1000 characters)",
                        text[:1000] + "..." if len(text) > 1000 else text,
                        height=400,
                        label_visibility="collapsed"
                    )
                    st.info(f"**Text Statistics:** {len(text)} characters, {len(text.split())} words")
                    st.download_button(
                        label="📥 Download Full Extracted Text",
                        data=text,
                        file_name="extracted_resume_text.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            else:
                st.error("Failed to extract text from the file")
        except Exception as e:
            st.error(f"Error processing file: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    else:
        st.info("👆 Please upload a resume file to get started")


if __name__ == "__main__":
    main()