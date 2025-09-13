from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from typing import Dict, List, Optional
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from datetime import datetime
import os
import requests
from dotenv import load_dotenv


load_dotenv()

app = FastAPI(title="AI Evaluation Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000", 
        "http://localhost:8001",  # Auth service
        "http://localhost:8002"   # Resume service
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


security = HTTPBearer()
genai.configure(api_key=os.getenv("GENAI_API_KEY"))

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

evaluation_storage = {}

class CandidateEvaluationRequest(BaseModel):
    job_requirements: Dict
    candidate_profile: Dict
    resume_content: str
    
class EvaluationScores(BaseModel):
    technical_skills: int
    experience: int
    cultural_fit: int
    education: int
    overall_score: int
    confidence: int
    
    
class AIEvaluator:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-1.5-pro")
        self.model_weights = {
            "technical_skills": 0.4,
            "experience": 0.3,
            "cultural_fit": 0.2,
            "education": 0.1
        }
        
    def calculate_semantic_similarity(self, job_text: str, resume_text: str) -> float:
        """Calculate semantic similarity between job and resume"""
        try:
            embeddings = sentence_model.encode([job_text, resume_text])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity * 100)  # Convert to percentage
        except Exception as e:
            print(f"Semantic similarity error: {e}")
            return 50.0  # Default score
    
    def evaluate_technical_skills(self, required_skills: List[str], candidate_skills: List[str]) -> int:
        """Evaluate technical skills match"""
        if not required_skills:
            return 50
        
        matched_skills = set(skill.lower() for skill in required_skills).intersection(
            set(skill.lower() for skill in candidate_skills)
        )
        skill_match_ratio = len(matched_skills) / len(required_skills)
        
        # Bonus for additional relevant skills
        bonus = min(len(candidate_skills) - len(matched_skills), 3) * 5
        
        return min(int(skill_match_ratio * 80 + bonus), 100)
    
    def evaluate_experience(self, required_years: int, candidate_years: int) -> int:
        """Evaluate experience level"""
        if candidate_years >= required_years:
            return 90 + min((candidate_years - required_years) * 2, 10)
        else:
            # Penalty for less experience, but not too harsh
            shortage = required_years - candidate_years
            penalty = min(shortage * 15, 60)
            return max(90 - penalty, 20)
    
    async def generate_ai_explanation(self, scores: Dict, job_title: str, candidate_name: str) -> str:
        """Generate explanation using Gemini API"""
        prompt = f"""
        As an AI recruitment assistant, provide a clear explanation for why candidate {candidate_name} 
        received the following scores for the {job_title} position:
        
        Technical Skills: {scores['technical_skills']}/100
        Experience: {scores['experience']}/100
        Cultural Fit: {scores['cultural_fit']}/100
        Overall Score: {scores['overall_score']}/100
        
        Provide specific reasoning for each score and overall recommendation.
        Keep it professional, constructive, and under 200 words.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Gemini API error: {str(e)}")
            return f"Strong candidate with {scores['overall_score']}% match. Technical skills align well with requirements, and experience level is appropriate for the role."
    
    async def evaluate_candidate(self, request: CandidateEvaluationRequest) -> Dict:
        """Main evaluation function"""
        job_req = request.job_requirements
        candidate = request.candidate_profile
        
        # Calculate individual scores
        technical_score = self.evaluate_technical_skills(
            job_req.get("required_skills", []),
            candidate.get("skills", [])
        )
        
        experience_score = self.evaluate_experience(
            job_req.get("required_experience", 0),
            candidate.get("experience_years", 0)
        )
        
        # Semantic similarity for cultural fit
        job_description = job_req.get("description", "")
        cultural_fit_score = int(self.calculate_semantic_similarity(
            job_description, request.resume_content[:1000]  # Limit text length
        ))
        
        # Education score
        education_score = 75 if candidate.get("education", {}).get("degree") else 50
        
        # Calculate weighted overall score
        overall_score = int(
            technical_score * self.model_weights["technical_skills"] +
            experience_score * self.model_weights["experience"] +
            cultural_fit_score * self.model_weights["cultural_fit"] +
            education_score * self.model_weights["education"]
        )
        
        # Generate confidence score
        confidence = min(85 + (overall_score - 70) // 5, 95) if overall_score >= 60 else 60
        
        scores = {
            "technical_skills": technical_score,
            "experience": experience_score,
            "cultural_fit": cultural_fit_score,
            "education": education_score,
            "overall_score": overall_score,
            "confidence": confidence
        }
        
        # Generate AI explanation
        explanation = await self.generate_ai_explanation(
            scores,
            job_req.get("title", "Software Developer"),
            candidate.get("name", "Candidate")
        )
        
        evaluation_id = f"eval_{hash(str(scores))}_{int(datetime.now().timestamp())}"
        
        return {
            "evaluation_id": evaluation_id,
            "scores": scores,
            "explanation": explanation,
            "model_used": "gemini-1.5-pro + semantic-similarity",
            "timestamp": datetime.now().isoformat()
        }

# Verify JWT token with auth service
async def verify_token(token: str = Depends(security)):
    try:
        response = requests.get(
            "http://localhost:8001/profile",
            headers={"Authorization": f"Bearer {token.credentials}"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=401, detail="Invalid token")
    except:
        raise HTTPException(status_code=401, detail="Token verification failed")

evaluator = AIEvaluator()

@app.post("/api/v1/ai-evaluation/match")
async def evaluate_candidate_match(
    request: CandidateEvaluationRequest,
    current_user: dict = Depends(verify_token)
):
    try:
        result = await evaluator.evaluate_candidate(request)
        
        # Store result
        evaluation_storage[result["evaluation_id"]] = {
            **result,
            "user_id": current_user["user"]["id"],
            "created_at": datetime.now().isoformat()
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/api/v1/ai-evaluation/explain/{evaluation_id}")
async def get_evaluation_explanation(
    evaluation_id: str,
    current_user: dict = Depends(verify_token)
):
    if evaluation_id not in evaluation_storage:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    stored_eval = evaluation_storage[evaluation_id]
    
    return {
        "evaluation_id": evaluation_id,
        "explanation": stored_eval["explanation"],
        "scores": stored_eval["scores"],
        "confidence_factors": [
            "Technical skills alignment with job requirements",
            "Experience level matches expected qualifications", 
            "Resume content indicates cultural fit",
            "Educational background supports role requirements"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "ai-evaluation-gemini",
        "port": 8004,
        "model": "gemini-1.5-pro"
    }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
    