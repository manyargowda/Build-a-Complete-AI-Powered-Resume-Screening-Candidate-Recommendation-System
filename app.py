from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)


# ==========================================================
# SKILLS DATABASE
# ==========================================================

SKILLS = [
    "python", "java", "c", "c++", "sql", "mysql",
    "mongodb", "postgresql", "html", "css", "javascript",
    "react", "angular", "node.js", "flask", "django",
    "machine learning", "deep learning",
    "artificial intelligence", "data science",
    "data analysis", "data engineering",
    "data visualization", "pandas", "numpy",
    "scikit-learn", "tensorflow", "pytorch",
    "power bi", "tableau", "excel",
    "spark", "apache spark", "hadoop", "etl",
    "aws", "azure", "google cloud",
    "cloud computing", "docker", "kubernetes",
    "git", "github", "linux", "rest api",
    "api", "statistics", "communication",
    "problem solving"
]


# ==========================================================
# JOB TITLES
# ==========================================================

JOB_TITLES = [
    "data engineer",
    "data analyst",
    "software engineer",
    "software developer",
    "python developer",
    "java developer",
    "web developer",
    "full stack developer",
    "frontend developer",
    "backend developer",
    "machine learning engineer",
    "machine learning developer",
    "data scientist",
    "ai engineer",
    "artificial intelligence engineer",
    "cloud engineer",
    "devops engineer",
    "database administrator",
    "database developer"
]


# ==========================================================
# EDUCATION KEYWORDS
# ==========================================================

EDUCATION_KEYWORDS = [
    "b.e", "be", "btech", "b.tech",
    "bachelor", "bachelors", "engineering",
    "computer science", "information science",
    "information technology", "electronics",
    "master", "m.tech", "mtech", "mca",
    "bca", "mba", "degree", "graduation"
]


# ==========================================================
# EXPERIENCE PATTERNS
# ==========================================================

EXPERIENCE_PATTERNS = [
    r"(\d+)\+?\s*years?\s*(?:of)?\s*experience",
    r"(\d+)\+?\s*yrs?\s*(?:of)?\s*experience",
    r"(\d+)\+?\s*years?",
    r"(\d+)\+?\s*months?\s*(?:of)?\s*experience"
]


# ==========================================================
# PDF TEXT EXTRACTION
# ==========================================================

def extract_text_from_pdf(pdf_file):

    text = ""

    try:

        reader = PdfReader(pdf_file)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"

    except Exception as e:

        print("PDF extraction error:", e)

    return text


# ==========================================================
# CLEAN TEXT
# ==========================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z0-9+#.\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==========================================================
# FIND SKILLS
# ==========================================================

def find_skills(text):

    text = clean_text(text)

    found_skills = []

    for skill in SKILLS:

        pattern = (
            r"(?<!\w)"
            + re.escape(skill.lower())
            + r"(?!\w)"
        )

        if re.search(pattern, text):

            found_skills.append(skill)

    return sorted(set(found_skills))


# ==========================================================
# FIND JOB TITLES
# ==========================================================

def find_job_titles(text):

    text = clean_text(text)

    found_titles = []

    for title in JOB_TITLES:

        if title in text:

            found_titles.append(title)

    return sorted(set(found_titles))


# ==========================================================
# TF-IDF SIMILARITY
# ==========================================================

def calculate_text_similarity(
    job_description,
    resume_text
):

    job_description = clean_text(
        job_description
    )

    resume_text = clean_text(
        resume_text
    )

    if not job_description or not resume_text:

        return 0

    try:

        documents = [
            job_description,
            resume_text
        ]

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        matrix = vectorizer.fit_transform(
            documents
        )

        similarity = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]

        return round(
            similarity * 100,
            2
        )

    except Exception as e:

        print("Similarity error:", e)

        return 0


# ==========================================================
# SKILL SCORE
# ==========================================================

def calculate_skill_score(
    job_skills,
    resume_skills
):

    if not job_skills:

        return 100

    matched = set(job_skills).intersection(
        set(resume_skills)
    )

    score = (
        len(matched) /
        len(job_skills)
    ) * 100

    return round(
        score,
        2
    )


# ==========================================================
# EXPERIENCE EXTRACTION
# ==========================================================

def extract_experience(text):

    text = clean_text(text)

    years = []

    for pattern in EXPERIENCE_PATTERNS:

        matches = re.findall(
            pattern,
            text
        )

        for match in matches:

            try:

                value = float(match)

                if value <= 50:

                    years.append(value)

            except:

                pass

    if years:

        return max(years)

    return 0


# ==========================================================
# EXPERIENCE SCORE
# ==========================================================

def calculate_experience_score(
    required,
    candidate
):

    if required <= 0:

        return 100

    if candidate >= required:

        return 100

    if candidate <= 0:

        return 0

    return round(
        (candidate / required) * 100,
        2
    )


# ==========================================================
# EDUCATION SCORE
# ==========================================================

def calculate_education_score(
    job_description,
    resume_text
):

    job_text = clean_text(
        job_description
    )

    resume_text = clean_text(
        resume_text
    )

    required = []

    for keyword in EDUCATION_KEYWORDS:

        if keyword in job_text:

            required.append(
                keyword
            )

    if not required:

        return 100

    matched = 0

    for keyword in required:

        if keyword in resume_text:

            matched += 1

    return round(
        (matched / len(required)) * 100,
        2
    )


# ==========================================================
# JOB TITLE SCORE
# ==========================================================

def calculate_job_title_score(
    job_description,
    resume_text
):

    job_titles = find_job_titles(
        job_description
    )

    resume_titles = find_job_titles(
        resume_text
    )

    if not job_titles:

        return 100

    for title in job_titles:

        if title in resume_titles:

            return 100

    for job_title in job_titles:

        for resume_title in resume_titles:

            job_words = set(
                job_title.split()
            )

            resume_words = set(
                resume_title.split()
            )

            overlap = job_words.intersection(
                resume_words
            )

            if overlap:

                return 50

    return 0


# ==========================================================
# FINAL SCORE
# ==========================================================

def calculate_final_score(
    skill_score,
    text_score,
    experience_score,
    education_score,
    title_score
):

    score = (

        skill_score * 0.35

        +

        text_score * 0.30

        +

        experience_score * 0.15

        +

        education_score * 0.10

        +

        title_score * 0.10

    )

    return round(
        score,
        2
    )


# ==========================================================
# RECOMMENDATION
# ==========================================================

def get_recommendation(score):

    if score >= 80:

        return (
            "Highly Recommended",
            "success"
        )

    elif score >= 65:

        return (
            "Recommended",
            "success"
        )

    elif score >= 50:

        return (
            "Needs Review",
            "warning"
        )

    else:

        return (
            "Not Recommended",
            "danger"
        )


# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================================
# MULTIPLE RESUME SCREENING
# ==========================================================

@app.route(
    "/screen",
    methods=["POST"]
)
def screen():

    # ------------------------------------------------------
    # JOB DESCRIPTION
    # ------------------------------------------------------

    job_description = request.form.get(
        "job_description",
        ""
    ).strip()


    # ------------------------------------------------------
    # MULTIPLE RESUMES
    # ------------------------------------------------------

    resumes = request.files.getlist(
        "resumes"
    )


    # ------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------

    if not job_description:

        return render_template(
            "index.html",
            error="Please enter a job description."
        )


    if not resumes:

        return render_template(
            "index.html",
            error="Please upload at least one resume."
        )


    # ------------------------------------------------------
    # JOB ANALYSIS
    # ------------------------------------------------------

    job_skills = find_skills(
        job_description
    )

    required_experience = extract_experience(
        job_description
    )


    candidates = []


    # ======================================================
    # PROCESS EACH RESUME
    # ======================================================

    for resume in resumes:

        if not resume.filename:

            continue


        if not resume.filename.lower().endswith(
            ".pdf"
        ):

            continue


        # --------------------------------------------------
        # EXTRACT TEXT
        # --------------------------------------------------

        resume_text = extract_text_from_pdf(
            resume
        )


        if not resume_text.strip():

            continue


        # --------------------------------------------------
        # SKILLS
        # --------------------------------------------------

        resume_skills = find_skills(
            resume_text
        )


        matched_skills = sorted(
            list(
                set(job_skills)
                &
                set(resume_skills)
            )
        )


        missing_skills = sorted(
            list(
                set(job_skills)
                -
                set(resume_skills)
            )
        )


        # --------------------------------------------------
        # SKILL SCORE
        # --------------------------------------------------

        skill_score = calculate_skill_score(
            job_skills,
            resume_skills
        )


        # --------------------------------------------------
        # TEXT SCORE
        # --------------------------------------------------

        text_score = calculate_text_similarity(
            job_description,
            resume_text
        )


        # --------------------------------------------------
        # EXPERIENCE
        # --------------------------------------------------

        candidate_experience = extract_experience(
            resume_text
        )


        experience_score = calculate_experience_score(
            required_experience,
            candidate_experience
        )


        # --------------------------------------------------
        # EDUCATION
        # --------------------------------------------------

        education_score = calculate_education_score(
            job_description,
            resume_text
        )


        # --------------------------------------------------
        # JOB TITLE
        # --------------------------------------------------

        title_score = calculate_job_title_score(
            job_description,
            resume_text
        )


        # --------------------------------------------------
        # FINAL SCORE
        # --------------------------------------------------

        final_score = calculate_final_score(

            skill_score,

            text_score,

            experience_score,

            education_score,

            title_score

        )


        # --------------------------------------------------
        # RECOMMENDATION
        # --------------------------------------------------

        recommendation, result_class = (
            get_recommendation(
                final_score
            )
        )


        # --------------------------------------------------
        # STORE CANDIDATE
        # --------------------------------------------------

        candidates.append({

            "filename": resume.filename,

            "score": final_score,

            "skill_score": skill_score,

            "text_score": text_score,

            "experience_score": experience_score,

            "education_score": education_score,

            "title_score": title_score,

            "required_experience":
                required_experience,

            "candidate_experience":
                candidate_experience,

            "matched_skills":
                matched_skills,

            "missing_skills":
                missing_skills,

            "recommendation":
                recommendation,

            "result_class":
                result_class

        })


    # ======================================================
    # SORT CANDIDATES
    # ======================================================

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )


    # ======================================================
    # ADD RANK
    # ======================================================

    for index, candidate in enumerate(
        candidates,
        start=1
    ):

        candidate["rank"] = index


    # ======================================================
    # SUMMARY
    # ======================================================

    total_candidates = len(
        candidates
    )

    highly_recommended = sum(

        1

        for candidate in candidates

        if candidate["score"] >= 80

    )

    recommended = sum(

        1

        for candidate in candidates

        if 65 <= candidate["score"] < 80

    )

    needs_review = sum(

        1

        for candidate in candidates

        if 50 <= candidate["score"] < 65

    )

    not_recommended = sum(

        1

        for candidate in candidates

        if candidate["score"] < 50

    )


    # ======================================================
    # RETURN DASHBOARD
    # ======================================================

    return render_template(

        "index.html",

        job_description=job_description,

        job_skills=job_skills,

        candidates=candidates,

        total_candidates=total_candidates,

        highly_recommended=highly_recommended,

        recommended=recommended,

        needs_review=needs_review,

        not_recommended=not_recommended

    )


# ==========================================================
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )