from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import datetime
from fastapi import File, UploadFile
import PyPDF2
import docx


# ---------- Initialize App ----------
app = FastAPI()

# ---------- Request Models ----------
class ResumeInput(BaseModel):
    text: str

class JobDescInput(BaseModel):
    text: str

class MatchInput(BaseModel):
    resume_text: str
    job_text: str

# ---------- Dummy Endpoints ----------
@app.get("/")
def home():
    return {"message": "ML API is running 🚀"}

@app.post("/parse-resume")
def parse_resume(resume: ResumeInput):
    return {
        "parsed_data": {
            "name": "John Doe",
            "experience": "5 years",
            "previous_company": "XYZ Corp"
        }
    }

@app.post("/parse-jobdesc")
def parse_jobdesc(job: JobDescInput):
    return {
        "parsed_data": {
            "role": "Software Engineer",
            "experience_required": "3+ years",
            "skills_required": ["Python", "FastAPI", "Machine Learning"]
        }
    }

# ---------- Synonym Dictionary ----------
synonym_dict = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "cv": "computer vision",
    "nlp": "natural language processing",
    "js": "javascript",
    "py": "python",
    "db": "database"
}

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove special characters
    for abbr, full in synonym_dict.items():
        text = re.sub(rf"\b{abbr}\b", full, text)  # replace abbreviation with full form
    return text

# ---------- Log Writer ----------
def write_log(content: str):
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(content + "\n")

# ---------- Matching Endpoint ----------
@app.post("/match")
def match_texts(input_data: MatchInput):
    # Preprocess text
    resume_text = preprocess_text(input_data.resume_text)
    job_text = preprocess_text(input_data.job_text)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_text])

    # Cosine Similarity
    similarity_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    # ---------- Debug Logging ----------
    log_content = (
        f"\n--- Matching Debug Info ---\n"
        f"Timestamp: {datetime.datetime.now()}\n"
        f"Original Resume: {input_data.resume_text}\n"
        f"Preprocessed Resume: {resume_text}\n"
        f"Original Job: {input_data.job_text}\n"
        f"Preprocessed Job: {job_text}\n"
        f"Similarity Score: {similarity_score:.3f}\n"
        f"---------------------------\n"
    )

    print(log_content)           # show in Git Bash
    write_log(log_content)       # save in log.txt

    return {
        "match_score": round(float(similarity_score), 3),
        "match_percentage": f"{round(similarity_score * 100, 1)}%"
    }
@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload a PDF or DOCX resume, extract text, and return it.
    """
    content = ""
    
    if file.filename.endswith(".pdf"):
        # Reset pointer before reading
        file.file.seek(0)
        pdf_reader = PyPDF2.PdfReader(file.file)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                content += text + "\n"
    
    elif file.filename.endswith(".docx"):
        # Reset pointer before reading
        file.file.seek(0)
        doc = docx.Document(file.file)
        for para in doc.paragraphs:
            content += para.text + "\n"
    
    else:
        return {"error": "Only PDF and DOCX files are supported"}
    
    return {
        "filename": file.filename,
        "extracted_text": content.strip()
    }
