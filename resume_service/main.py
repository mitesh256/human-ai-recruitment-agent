# from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
# from fastapi.security import HTTPBearer
# import PyPDF2
# import io
# import re
# from typing import List, Dict, Optional
# import requests
# import os
# from datetime import datetime

# app = FastAPI(title="Resume Processing Service", version="1.0.0")
# security = HTTPBearer()

# class ResumeProcessor:
#     def __init__(self):
#         self.skills_taxonomy = {
#             "programming": ["python", "java", "javascript", "c++", "sql", "fastapi", "react", "node", "php"],
#             "databases": ["postgresql", "mysql", "mongodb", "redis", "sqlite"],
#             "tools": ["docker", "kubernetes", "git", "jenkins", "aws", "azure"],
#             "frameworks": ["django", "flask", "express", "spring", "angular", "vue"],
#             "cloud": ["aws", "azure", "gcp", "heroku", "vercel"]
#         }
    
#     def extract_text_from_pdf(self, file_content: bytes) -> str:
#         """Extract text from PDF file"""
#         try:
#             pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
#             text = ""
#             for page in pdf_reader.pages:
#                 text += page.extract_text()
#             return text
#         except Exception as e:
#             raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")
    
#     def extract_skills(self, text: str) -> List[str]:
#         """Extract skills using pattern matching"""
#         text_lower = text.lower()
#         extracted_skills = []
        
#         for category, skills in self.skills_taxonomy.items():
#             for skill in skills:
#                 if skill in text_lower:
#                     extracted_skills.append(skill)
        
#         return list(set(extracted_skills))
    
#     def extract_experience(self, text: str) -> int:
#         """Extract years of experience using regex patterns"""
#         patterns = [
#             r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
#             r'experience[:\s]*(\d+)\+?\s*years?',
#             r'(\d+)\+?\s*yrs?\s+exp'
#         ]
        
#         for pattern in patterns:
#             match = re.search(pattern, text.lower())
#             if match:
#                 return int(match.group(1))
#         return 0
    
#     def extract_education(self, text: str) -> Dict:
#         """Extract education information"""
#         education_patterns = {
#             "degree": r'(bachelor|master|phd|doctorate|b\.?s\.?|m\.?s\.?|b\.?a\.?|m\.?a\.?)[\s\.]*(computer science|engineering|business|arts|science)?',
#             "university": r'(university|college|institute|school)'
#         }
        
#         education = {"degree": "", "field": "", "institution": ""}
        
#         degree_match = re.search(education_patterns["degree"], text.lower())
#         if degree_match:
#             education["degree"] = degree_match.group(1)
#             education["field"] = degree_match.group(2) or ""
        
#         return education
    
#     def calculate_quality_score(self, parsed_content: Dict) -> int:
#         """Calculate resume quality score (0-100)"""
#         score = 0
        
#         # Skills presence (40 points)
#         skills_count = len(parsed_content.get("skills", []))
#         score += min(skills_count * 8, 40)
        
#         # Experience (30 points)  
#         experience = parsed_content.get("experience_years", 0)
#         score += min(experience * 5, 30)
        
#         # Education (20 points)
#         if parsed_content.get("education", {}).get("degree"):
#             score += 20
        
#         # Contact information (10 points)
#         text = parsed_content.get("raw_text", "").lower()
#         if "@" in text:  # Email present
#             score += 5
#         if re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', text):  # Phone present
#             score += 5
        
#         return min(score, 100)

# # Verify JWT token with auth service
# async def verify_token(token: str = Depends(security)):
#     try:
#         # Call auth service to verify token
#         response = requests.get(
#             "http://localhost:8001/profile",
#             headers={"Authorization": f"Bearer {token.credentials}"}
#         )
#         if response.status_code == 200:
#             return response.json()
#         else:
#             raise HTTPException(status_code=401, detail="Invalid token")
#     except:
#         raise HTTPException(status_code=401, detail="Token verification failed")

# processor = ResumeProcessor()

# @app.post("/api/v1/resume/upload")
# async def upload_resume(
#     file: UploadFile = File(...), 
#     current_user: dict = Depends(verify_token)
# ):
#     # Validate file type
#     if not file.filename.lower().endswith('.pdf'):
#         raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
#     # Read and process file
#     content = await file.read()
#     raw_text = processor.extract_text_from_pdf(content)
    
#     # Extract information
#     skills = processor.extract_skills(raw_text)
#     experience_years = processor.extract_experience(raw_text)
#     education = processor.extract_education(raw_text)
    
#     parsed_content = {
#         "raw_text": raw_text,
#         "skills": skills,
#         "experience_years": experience_years,
#         "education": education
#     }
    
#     quality_score = processor.calculate_quality_score(parsed_content)
    
#     # Generate response
#     return {
#         "message": "Resume processed successfully",
#         "resume_id": f"resume_{hash(file.filename)}_{int(datetime.now().timestamp())}",
#         "user_id": current_user["user"]["id"],
#         "data": {
#             "filename": file.filename,
#             "quality_score": quality_score,
#             "skills": skills,
#             "experience_years": experience_years,
#             "education": education,
#             "processed_at": datetime.now().isoformat()
#         }
#     }

# @app.get("/api/v1/resume/{resume_id}/analysis")
# async def get_resume_analysis(resume_id: str, current_user: dict = Depends(verify_token)):
#     # In production, fetch from database
#     return {
#         "resume_id": resume_id,
#         "ai_score": 87,
#         "breakdown": {
#             "technical_skills": 35,
#             "experience_match": 28, 
#             "education_relevance": 18,
#             "presentation_quality": 6
#         },
#         "recommendations": [
#             "Add more cloud technologies (AWS, Azure)",
#             "Include specific project outcomes",
#             "Add leadership experience examples"
#         ]
#     }

# @app.get("/health")
# async def health_check():
#     return {"status": "healthy", "service": "resume-processing", "port": 8002}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8002)


# from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
# from fastapi.security import HTTPBearer
# import PyPDF2
# import io
# import re
# from typing import List, Dict, Optional
# import requests
# import os
# from datetime import datetime
# import json


# app = FastAPI(title="Resume Processing Service", version="1.0.0")
# security = HTTPBearer()

# # In-memory storage (use database in production)
# resume_storage = {}


# class ResumeProcessor:
#     def __init__(self):
#         self.skills_taxonomy = {
#             "programming": ["python", "java", "javascript", "c++", "sql", "fastapi", "react", "node", "php"],
#             "databases": ["postgresql", "mysql", "mongodb", "redis", "sqlite"],
#             "tools": ["docker", "kubernetes", "git", "jenkins", "aws", "azure"],
#             "frameworks": ["django", "flask", "express", "spring", "angular", "vue"],
#             "cloud": ["aws", "azure", "gcp", "heroku", "vercel"]
#         }
        
#         # AI scoring weights
#         self.ai_weights = {
#             "technical_skills": 0.4,
#             "experience_match": 0.3,
#             "education_relevance": 0.2,
#             "presentation_quality": 0.1
#         }
    
#     def extract_text_from_pdf(self, file_content: bytes) -> str:
#         """Extract text from PDF file"""
#         try:
#             pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
#             text = ""
#             for page in pdf_reader.pages:
#                 text += page.extract_text()
#             return text
#         except Exception as e:
#             raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")
    
#     def extract_skills(self, text: str) -> List[str]:
#         """Extract skills using pattern matching"""
#         text_lower = text.lower()
#         extracted_skills = []
        
#         for category, skills in self.skills_taxonomy.items():
#             for skill in skills:
#                 if skill in text_lower:
#                     extracted_skills.append(skill)
        
#         return list(set(extracted_skills))
    
#     def extract_experience(self, text: str) -> int:
#         """Extract years of experience using regex patterns"""
#         patterns = [
#             r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
#             r'experience[:\s]*(\d+)\+?\s*years?',
#             r'(\d+)\+?\s*yrs?\s+exp'
#         ]
        
#         for pattern in patterns:
#             match = re.search(pattern, text.lower())
#             if match:
#                 return int(match.group(1))
#         return 0
    
#     def extract_education(self, text: str) -> Dict:
#         """Extract education information"""
#         education_patterns = {
#             "degree": r'(bachelor|master|phd|doctorate|b\.?s\.?|m\.?s\.?|b\.?a\.?|m\.?a\.?)[\s\.]*(computer science|engineering|business|arts|science)?',
#             "university": r'(university|college|institute|school)'
#         }
        
#         education = {"degree": "", "field": "", "institution": ""}
        
#         degree_match = re.search(education_patterns["degree"], text.lower())
#         if degree_match:
#             education["degree"] = degree_match.group(1)
#             education["field"] = degree_match.group(2) or ""
        
#         return education
    
#     def calculate_quality_score(self, parsed_content: Dict) -> int:
#         """Calculate resume quality score (0-100)"""
#         score = 0
        
#         # Skills presence (40 points)
#         skills_count = len(parsed_content.get("skills", []))
#         score += min(skills_count * 8, 40)
        
#         # Experience (30 points)  
#         experience = parsed_content.get("experience_years", 0)
#         score += min(experience * 5, 30)
        
#         # Education (20 points)
#         if parsed_content.get("education", {}).get("degree"):
#             score += 20
        
#         # Contact information (10 points)
#         text = parsed_content.get("raw_text", "").lower()
#         if "@" in text:  # Email present
#             score += 5
#         if re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', text):  # Phone present
#             score += 5
        
#         return min(score, 100)
    
#     def calculate_ai_score(self, parsed_content: Dict) -> Dict:
#         """Calculate AI score with detailed breakdown based on resume content"""
        
#         # Technical Skills Score (0-40)
#         skills = parsed_content.get("skills", [])
#         technical_skills_score = min(len(skills) * 5, 40)
        
#         # Experience Match Score (0-35)
#         experience_years = parsed_content.get("experience_years", 0)
#         experience_match_score = min(experience_years * 3.5, 35)
        
#         # Education Relevance Score (0-20)
#         education = parsed_content.get("education", {})
#         education_score = 0
#         if education.get("degree"):
#             if any(field in education.get("field", "").lower() for field in ["computer", "engineering", "science"]):
#                 education_score = 20
#             else:
#                 education_score = 15
        
#         # Presentation Quality Score (0-5)
#         text = parsed_content.get("raw_text", "")
#         presentation_score = 0
#         if len(text) > 500:  # Sufficient content
#             presentation_score += 2
#         if "@" in text.lower():  # Contact info present
#             presentation_score += 1.5
#         if re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', text):  # Phone present
#             presentation_score += 1.5
        
#         # Calculate total AI score
#         total_ai_score = technical_skills_score + experience_match_score + education_score + presentation_score
        
#         # Generate recommendations based on scores
#         recommendations = []
#         if technical_skills_score < 25:
#             recommendations.append("Add more technical skills (programming languages, frameworks)")
#         if experience_match_score < 20:
#             recommendations.append("Include more detailed work experience and achievements")
#         if education_score < 15:
#             recommendations.append("Highlight relevant educational background")
#         if presentation_score < 3:
#             recommendations.append("Improve resume formatting and include complete contact information")
        
#         # Add skill-specific recommendations
#         if "aws" not in [s.lower() for s in skills] and "azure" not in [s.lower() for s in skills]:
#             recommendations.append("Consider adding cloud technologies (AWS, Azure)")
        
#         return {
#             "ai_score": int(total_ai_score),
#             "breakdown": {
#                 "technical_skills": int(technical_skills_score),
#                 "experience_match": int(experience_match_score),
#                 "education_relevance": int(education_score),
#                 "presentation_quality": int(presentation_score)
#             },
#             "recommendations": recommendations
#         }


# # Verify JWT token with auth service
# async def verify_token(token: str = Depends(security)):
#     try:
#         # Call auth service to verify token
#         response = requests.get(
#             "http://localhost:8001/profile",
#             headers={"Authorization": f"Bearer {token.credentials}"}
#         )
#         if response.status_code == 200:
#             return response.json()
#         else:
#             raise HTTPException(status_code=401, detail="Invalid token")
#     except:
#         raise HTTPException(status_code=401, detail="Token verification failed")


# processor = ResumeProcessor()


# @app.post("/api/v1/resume/upload")
# async def upload_resume(
#     file: UploadFile = File(...), 
#     current_user: dict = Depends(verify_token)
# ):
#     # Validate file type
#     if not file.filename.lower().endswith('.pdf'):
#         raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
#     # Read and process file
#     content = await file.read()
#     raw_text = processor.extract_text_from_pdf(content)
    
#     # Extract information
#     skills = processor.extract_skills(raw_text)
#     experience_years = processor.extract_experience(raw_text)
#     education = processor.extract_education(raw_text)
    
#     parsed_content = {
#         "raw_text": raw_text,
#         "skills": skills,
#         "experience_years": experience_years,
#         "education": education
#     }
    
#     quality_score = processor.calculate_quality_score(parsed_content)
#     ai_analysis = processor.calculate_ai_score(parsed_content)
    
#     # Generate resume ID
#     resume_id = f"resume_{hash(file.filename)}_{int(datetime.now().timestamp())}"
    
#     # Store resume data for later retrieval (use database in production)
#     resume_storage[resume_id] = {
#         "parsed_content": parsed_content,
#         "quality_score": quality_score,
#         "ai_analysis": ai_analysis,
#         "user_id": current_user["user"]["id"],
#         "filename": file.filename,
#         "processed_at": datetime.now().isoformat()
#     }
    
#     # Generate response
#     return {
#         "message": "Resume processed successfully",
#         "resume_id": resume_id,
#         "user_id": current_user["user"]["id"],
#         "data": {
#             "filename": file.filename,
#             "quality_score": quality_score,
#             "ai_score": ai_analysis["ai_score"],
#             "skills": skills,
#             "experience_years": experience_years,
#             "education": education,
#             "processed_at": datetime.now().isoformat()
#         }
#     }


# @app.get("/api/v1/resume/{resume_id}/analysis")
# async def get_resume_analysis(resume_id: str, current_user: dict = Depends(verify_token)):
#     # Fetch from storage (in production, fetch from database)
#     if resume_id not in resume_storage:
#         raise HTTPException(status_code=404, detail="Resume not found")
    
#     stored_data = resume_storage[resume_id]
    
#     # Verify user owns this resume
#     if stored_data["user_id"] != current_user["user"]["id"]:
#         raise HTTPException(status_code=403, detail="Access denied")
    
#     return {
#         "resume_id": resume_id,
#         "ai_score": stored_data["ai_analysis"]["ai_score"],
#         "breakdown": stored_data["ai_analysis"]["breakdown"],
#         "recommendations": stored_data["ai_analysis"]["recommendations"],
#         "quality_score": stored_data["quality_score"]
#     }


# @app.get("/health")
# async def health_check():
#     return {"status": "healthy", "service": "resume-processing", "port": 8002}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8002)



from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import HTTPBearer
import PyPDF2
import io
import re
from typing import List, Dict, Optional
import requests
import os
from datetime import datetime
import json
import google.generativeai as genai
from pydantic import BaseModel

app = FastAPI(title="Resume Processing Service", version="1.0.0")
security = HTTPBearer()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# In-memory storage (use database in production)
resume_storage = {}

# Pydantic models for structured Gemini responses
class SkillAssessment(BaseModel):
    skill: str
    proficiency_level: str  # "Beginner", "Intermediate", "Advanced", "Expert"
    years_experience: int
    relevance_score: int  # 1-10

class ResumeAnalysis(BaseModel):
    technical_skills: List[SkillAssessment]
    soft_skills: List[str]
    experience_years: int
    education_level: str
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    overall_score: int  # 1-100
    confidence_level: int  # 1-100

class ResumeProcessor:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        self.skills_taxonomy = {
            "programming": ["python", "java", "javascript", "c++", "sql", "fastapi", "react", "node", "php"],
            "databases": ["postgresql", "mysql", "mongodb", "redis", "sqlite"],
            "tools": ["docker", "kubernetes", "git", "jenkins", "aws", "azure"],
            "frameworks": ["django", "flask", "express", "spring", "angular", "vue"],
            "cloud": ["aws", "azure", "gcp", "heroku", "vercel"]
        }
    
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills using pattern matching (fallback method)"""
        text_lower = text.lower()
        extracted_skills = []
        
        for category, skills in self.skills_taxonomy.items():
            for skill in skills:
                if skill in text_lower:
                    extracted_skills.append(skill)
        
        return list(set(extracted_skills))
    
    def extract_experience(self, text: str) -> int:
        """Extract years of experience using regex patterns (fallback method)"""
        patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'experience[:\s]*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*yrs?\s+exp'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))
        return 0
    
    def extract_education(self, text: str) -> Dict:
        """Extract education information (fallback method)"""
        education_patterns = {
            "degree": r'(bachelor|master|phd|doctorate|b\.?s\.?|m\.?s\.?|b\.?a\.?|m\.?a\.?)[\s\.]*(computer science|engineering|business|arts|science)?',
            "university": r'(university|college|institute|school)'
        }
        
        education = {"degree": "", "field": "", "institution": ""}
        
        degree_match = re.search(education_patterns["degree"], text.lower())
        if degree_match:
            education["degree"] = degree_match.group(1)
            education["field"] = degree_match.group(2) or ""
        
        return education
    
    async def analyze_resume_with_gemini(self, resume_text: str, job_title: str = "Software Developer") -> ResumeAnalysis:
        """Analyze resume using Gemini API with structured output"""
        
        prompt = f"""
        You are an expert AI recruiter analyzing a resume for a {job_title} position. 
        Analyze the following resume text and provide a comprehensive evaluation.

        Resume Text:
        {resume_text}

        Please analyze this resume and provide:
        1. Technical skills with proficiency levels and experience years
        2. Soft skills identified
        3. Total years of experience
        4. Education level
        5. Key strengths
        6. Areas for improvement
        7. Specific recommendations for career growth
        8. Overall score (1-100) based on industry standards
        9. Confidence level in your assessment (1-100)

        Focus on:
        - Technical skill relevance and depth
        - Experience progression and achievements
        - Education background alignment
        - Communication and leadership indicators
        - Overall career trajectory

        Be constructive and provide actionable feedback.
        """

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=ResumeAnalysis
                )
            )
            
            # Parse the JSON response
            analysis_data = json.loads(response.text)
            return ResumeAnalysis(**analysis_data)
            
        except Exception as e:
            print(f"Gemini API error: {str(e)}")
            # Fallback to basic analysis
            return self._fallback_analysis(resume_text)
    
    def _fallback_analysis(self, resume_text: str) -> ResumeAnalysis:
        """Fallback analysis if Gemini API fails"""
        skills = self.extract_skills(resume_text)
        experience_years = self.extract_experience(resume_text)
        
        # Create basic skill assessments
        skill_assessments = []
        for skill in skills[:10]:  # Limit to top 10 skills
            skill_assessments.append(SkillAssessment(
                skill=skill,
                proficiency_level="Intermediate",
                years_experience=min(experience_years, 5),
                relevance_score=7
            ))
        
        return ResumeAnalysis(
            technical_skills=skill_assessments,
            soft_skills=["Communication", "Problem Solving", "Teamwork"],
            experience_years=experience_years,
            education_level="Bachelor's Degree" if len(resume_text) > 1000 else "Not specified",
            strengths=["Technical Skills", "Experience"],
            weaknesses=["Could improve leadership experience"],
            recommendations=[
                "Consider adding more project details",
                "Highlight achievements with metrics",
                "Add relevant certifications"
            ],
            overall_score=min(60 + len(skills) * 3 + experience_years * 2, 100),
            confidence_level=75
        )
    
    def calculate_quality_score(self, parsed_content: Dict) -> int:
        """Calculate basic quality score (legacy method)"""
        score = 0
        
        # Skills presence (40 points)
        skills_count = len(parsed_content.get("skills", []))
        score += min(skills_count * 8, 40)
        
        # Experience (30 points)  
        experience = parsed_content.get("experience_years", 0)
        score += min(experience * 5, 30)
        
        # Education (20 points)
        if parsed_content.get("education", {}).get("degree"):
            score += 20
        
        # Contact information (10 points)
        text = parsed_content.get("raw_text", "").lower()
        if "@" in text:  # Email present
            score += 5
        if re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', text):  # Phone present
            score += 5
        
        return min(score, 100)
    
    async def generate_personalized_feedback(self, analysis: ResumeAnalysis, job_requirements: Dict = None) -> Dict:
        """Generate personalized feedback using Gemini"""
        
        job_context = ""
        if job_requirements:
            job_context = f"""
            Job Requirements:
            - Required Skills: {job_requirements.get('required_skills', [])}
            - Experience Level: {job_requirements.get('experience_level', 'Not specified')}
            - Education: {job_requirements.get('education', 'Not specified')}
            """
        
        feedback_prompt = f"""
        Based on the resume analysis, provide personalized career advice and improvement suggestions.
        
        Analysis Summary:
        - Overall Score: {analysis.overall_score}/100
        - Experience: {analysis.experience_years} years
        - Key Strengths: {', '.join(analysis.strengths)}
        - Areas for Improvement: {', '.join(analysis.weaknesses)}
        
        {job_context}
        
        Provide:
        1. 3-5 specific, actionable improvement recommendations
        2. Suggested next steps for career advancement
        3. Skills to prioritize for development
        4. Interview preparation tips based on their background
        
        Make it personal, constructive, and motivating.
        """
        
        try:
            response = self.model.generate_content(feedback_prompt)
            return {
                "personalized_feedback": response.text,
                "action_items": analysis.recommendations,
                "skill_development_priority": [skill.skill for skill in analysis.technical_skills[:5]],
                "estimated_improvement_timeline": "2-3 months with focused effort"
            }
        except Exception as e:
            return {
                "personalized_feedback": "Focus on strengthening your technical skills and gaining more hands-on experience with relevant technologies.",
                "action_items": analysis.recommendations,
                "skill_development_priority": ["Python", "JavaScript", "SQL"],
                "estimated_improvement_timeline": "2-3 months with focused effort"
            }

# Verify JWT token with auth service
async def verify_token(token: str = Depends(security)):
    try:
        # Call auth service to verify token
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

processor = ResumeProcessor()

@app.post("/api/v1/resume/upload")
async def upload_resume(
    file: UploadFile = File(...), 
    job_title: Optional[str] = "Software Developer",
    current_user: dict = Depends(verify_token)
):
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Read and process file
    content = await file.read()
    raw_text = processor.extract_text_from_pdf(content)
    
    # Get comprehensive analysis from Gemini
    try:
        gemini_analysis = await processor.analyze_resume_with_gemini(raw_text, job_title)
        
        # Generate personalized feedback
        personalized_feedback = await processor.generate_personalized_feedback(gemini_analysis)
        
        # Extract basic info for backward compatibility
        skills = [skill.skill for skill in gemini_analysis.technical_skills]
        experience_years = gemini_analysis.experience_years
        
    except Exception as e:
        print(f"Gemini analysis failed, using fallback: {str(e)}")
        # Fallback to original processing
        skills = processor.extract_skills(raw_text)
        experience_years = processor.extract_experience(raw_text)
        gemini_analysis = processor._fallback_analysis(raw_text)
        personalized_feedback = await processor.generate_personalized_feedback(gemini_analysis)
    
    education = processor.extract_education(raw_text)
    
    parsed_content = {
        "raw_text": raw_text,
        "skills": skills,
        "experience_years": experience_years,
        "education": education
    }
    
    quality_score = processor.calculate_quality_score(parsed_content)
    
    # Generate resume ID
    resume_id = f"resume_{hash(file.filename)}_{int(datetime.now().timestamp())}"
    
    # Store comprehensive resume data
    resume_storage[resume_id] = {
        "parsed_content": parsed_content,
        "quality_score": quality_score,
        "gemini_analysis": gemini_analysis.dict(),
        "personalized_feedback": personalized_feedback,
        "user_id": current_user["user"]["id"],
        "filename": file.filename,
        "job_title": job_title,
        "processed_at": datetime.now().isoformat()
    }
    
    # Generate enhanced response
    return {
        "message": "Resume processed successfully with AI analysis",
        "resume_id": resume_id,
        "user_id": current_user["user"]["id"],
        "data": {
            "filename": file.filename,
            "quality_score": quality_score,
            "ai_score": gemini_analysis.overall_score,
            "confidence_level": gemini_analysis.confidence_level,
            "skills": skills,
            "technical_skills_detailed": [skill.dict() for skill in gemini_analysis.technical_skills],
            "soft_skills": gemini_analysis.soft_skills,
            "experience_years": experience_years,
            "education": education,
            "strengths": gemini_analysis.strengths,
            "improvement_areas": gemini_analysis.weaknesses,
            "recommendations": gemini_analysis.recommendations,
            "personalized_feedback": personalized_feedback["personalized_feedback"],
            "processed_at": datetime.now().isoformat()
        }
    }

@app.get("/api/v1/resume/{resume_id}/analysis")
async def get_resume_analysis(resume_id: str, current_user: dict = Depends(verify_token)):
    # Fetch from storage
    if resume_id not in resume_storage:
        raise HTTPException(status_code=404, detail="Resume not found")
    
    stored_data = resume_storage[resume_id]
    
    # Verify user owns this resume
    if stored_data["user_id"] != current_user["user"]["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        "resume_id": resume_id,
        "ai_score": stored_data["gemini_analysis"]["overall_score"],
        "confidence_level": stored_data["gemini_analysis"]["confidence_level"],
        "detailed_breakdown": {
            "technical_skills": stored_data["gemini_analysis"]["technical_skills"],
            "soft_skills": stored_data["gemini_analysis"]["soft_skills"],
            "strengths": stored_data["gemini_analysis"]["strengths"],
            "weaknesses": stored_data["gemini_analysis"]["weaknesses"],
            "recommendations": stored_data["gemini_analysis"]["recommendations"]
        },
        "personalized_feedback": stored_data["personalized_feedback"],
        "quality_score": stored_data["quality_score"],
        "processed_at": stored_data["processed_at"]
    }

@app.post("/api/v1/resume/{resume_id}/feedback")
async def generate_job_specific_feedback(
    resume_id: str,
    job_requirements: Dict,
    current_user: dict = Depends(verify_token)
):
    """Generate job-specific feedback for a resume"""
    if resume_id not in resume_storage:
        raise HTTPException(status_code=404, detail="Resume not found")
    
    stored_data = resume_storage[resume_id]
    if stored_data["user_id"] != current_user["user"]["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Generate job-specific feedback
    analysis = ResumeAnalysis(**stored_data["gemini_analysis"])
    job_specific_feedback = await processor.generate_personalized_feedback(analysis, job_requirements)
    
    return {
        "resume_id": resume_id,
        "job_specific_analysis": job_specific_feedback,
        "match_score": min(analysis.overall_score + 10, 100) if any(
            skill.skill.lower() in [req.lower() for req in job_requirements.get('required_skills', [])]
            for skill in analysis.technical_skills
        ) else max(analysis.overall_score - 15, 0),
        "generated_at": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "resume-processing-gemini", "port": 8002}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
