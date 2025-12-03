import re
from pathlib import Path
import spacy
from parsing.utils import clean_text
from nlp.skills import load_skills, compile_patterns, detect_skills
from transformers import pipeline

# Load zero-shot classifier once

nlp = spacy.load("en_core_web_trf")

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
    doc = nlp(text)
    """Extract first email address from text"""
    emails = [token.text for token in doc if token.like_email]
    return emails


def extract_name(text: str):
    lines = text.splitlines()
    # Only check the first few lines to avoid picking up job roles
    for line in lines[:3]:  # adjust to match your resume format
        doc = nlp(line)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Only return the PERSON entity if the line doesn't contain typical job words
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


def extract_sections(text):
    # Define possible fields (case-insensitive)
    fields = [ "Education", "Skills",
        "Experience", "Projects", "Certifications", "Hobbies", "Sector"
    ]
    
    # Build a regex pattern like: (Education|Skills|Experience) -
    pattern = r"(?i)(" + "|".join(fields) + r")\s*\s*"
    
    # Split text into sections based on field names
    parts = re.split(pattern, text)
    extracted = {}
    
    # The split will give ['', 'Field1', 'content1', 'Field2', 'content2', ...]
    for i in range(1, len(parts), 2):
        field = parts[i].strip().capitalize()
        content = parts[i+1].strip()
        # Stop content at the next field keyword if it's concatenated
        next_field_match = re.search(pattern, content)
        if next_field_match:
            content = content[:next_field_match.start()].strip()
        extracted[field] = content

    return extracted


# Wrapper for parsing .txt file
def parse_resume(txt_file: str) -> dict:
    """Parse resume from a .txt file"""
    text = Path(txt_file).read_text(encoding="utf-8", errors='ignore')
    text = text.strip()
    text = clean_text(text)
    return {
        "name": extract_name(text),
        "email": extract_email(text),
        "education": extract_education(text),
        "skills": extract_skills(text)
    }



def identify_sector(cv_text: str) -> str:
    """
    Identify the most likely job sector of a CV using keyword matching.
    """

    # Define sector keywords — expand this list as needed
    sector_keywords = {
        "Software Developer": [
            "python", "java", "c++", "software development", "backend", "frontend",
            "full stack", "api", "django", "flask", "spring", "git", "docker"
        ],
        "Data Scientist": [
            "machine learning", "data analysis", "pandas", "numpy", "tensorflow",
            "pytorch", "scikit-learn", "data visualization", "statistics", "eda"
        ],
        "Product Manager": [
            "roadmap", "stakeholders", "agile", "scrum", "product strategy",
            "user stories", "requirements gathering", "market research"
        ],
        "UI/UX Designer": [
            "figma", "adobe xd", "wireframes", "user experience", "user interface",
            "prototyping", "design thinking", "usability testing"
        ],
        "DevOps Engineer": [
            "aws", "azure", "jenkins", "kubernetes", "ci/cd", "terraform",
            "infrastructure", "linux", "cloud deployment"
        ],
        "Business Analyst": [
            "business requirements", "kpis", "data modeling", "sql", "tableau",
            "power bi", "process improvement", "gap analysis"
        ]
    }

    # Convert text to lowercase
    text = cv_text.lower()

    # Count keyword matches per sector
    scores = {}
    for sector, keywords in sector_keywords.items():
        count = sum(1 for kw in keywords if re.search(rf"\b{kw}\b", text))
        scores[sector] = count

    # Pick the best matching sector
    best_sector = max(scores, key=scores.get)
    if scores[best_sector] == 0:
        return "Uncategorized"

    return best_sector