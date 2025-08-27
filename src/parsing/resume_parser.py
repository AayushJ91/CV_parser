import re
from pathlib import Path
import spacy
from src.parsing.utils import clean_text
from src.nlp.skills import load_skills, compile_patterns, detect_skills
# Load spaCy model (make sure en_core_web_sm is installed)
nlp = spacy.load("en_core_web_sm")

# Example skill list (replace with your skills.csv logic if available)
# Skill_label = ["Python","Java", "SQL", "C++", "C", "C#", "JavaScript", "TypeScript", 
#                "HTML" "CSS", "SQL", "PostgreSQL", "MySQL", "SQLite", "MongoDB", "Redis", "Django"
#                "Flask", "FastAPI", "Spring", "Node.js", "Express", "React", "Angular", 
#                "Vue", "Next.js", "Tailwind CSS", "Bootstrap", "jQuery", "REST API", "GraphQL", "Git", 
#                "Linux", "Docker", "Kubernetes", "CI/CD", "AWS", "Azure", "GCP", "NumPy", "Pandas", "scikit-learn", "TensorFlow", 
#                "PyTorch", "Keras", "NLTK", "spaCy", "Transformers", "OpenCV", "Power BI","Tableau", "Excel", "Apache Spark"
#                "Hadoop", "Airflow", "Kafka", "Elasticsearch", "MLflow", "SQL Server", "NoSQL"]

# SKILLS = load_skills("data/skill.csv")
# pattern = compile_patterns(SKILLS)

def extract_email(text: str):
    """Extract first email address from text"""
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else None

def extract_phone(text: str):
    """Extract phone number (basic pattern)"""
    match = re.search(r"\+?\d[\d\s-]{8,12}\d", text)
    return match.group(0) if match else None

def extract_name(text: str):
    lines = text.splitlines()
    # Only check the first few lines to avoid picking up job roles
    for line in lines[:3]:  # adjust to match your resume format
        doc = nlp(line)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Only return the PERSON entity if the line doesn't contain typical job words
                if not any(job_word in line.lower() for job_word in ["role:", "position:", "engineer", "developer"]):
                    return ent.text
    return None


def extract_education(text: str):
    """Check for presence of education keywords"""
    edu_keywords = ["B.Tech", "BE", "M.Tech", "ME", "M.Sc", "MBA", "PhD", "University", "College"]
    education = [word for word in edu_keywords if word.lower() in text.lower()]
    return education

def extract_skills(text: str):
    """
    Extract skills dynamically by looking for 'Skills' section in resume text.
    Returns a list of skills found.
    """
    skills = []

    # Case-insensitive search for "skills" section
    match = re.search(r"(Skills|skills|technical skills|key skills|skills & expertise)(.*?)(experience|education|projects|certifications|$)", 
                      text, re.IGNORECASE | re.DOTALL)
    print(match)
    if match:
        # Group(2) contains text between "skills" and next section
        skills_text = match.group(2)

        # Split by common delimiters
        parts = re.split(r"[\n,;•-]", skills_text)

        # Clean up extra spaces and filter out junk
        skills = [p.strip() for p in parts if len(p.strip()) > 1]

    return skills


# Wrapper for parsing .txt file
def parse_resume(txt_file: str) -> dict:
    """Parse resume from a .txt file"""
    text = Path(txt_file).read_text(encoding="utf-8")
    text = text.strip()
    text = clean_text(text)
    return {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "education": extract_education(text),
        "skills": extract_skills(text)
    }
