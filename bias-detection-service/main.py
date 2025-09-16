from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import mysql.connector
from mysql.connector import Error
import json
import re
from datetime import datetime, timedelta
import requests
import os
from collections import Counter
import statistics
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Bias Detection Service",
    description="AI-powered bias detection for fair recruitment practices",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8001",  # Auth service
        "http://localhost:8002",  # Resume service
        "http://localhost:8004",  # AI evaluation service
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# Database connection
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DATABASE"),
            port=int(os.getenv("MYSQL_PORT", "3306"))
        )
        return connection
    except Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection failed: {str(e)}"
        )

# Verify JWT token with auth service
async def verify_token(token: str = Depends(security)):
    try:
        response = requests.get(
            "http://localhost:8001/profile",
            headers={"Authorization": f"Bearer {token.credentials}"},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=401, detail="Invalid token")
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=401, detail="Token verification failed")

# Pydantic Models
class BiasAnalysisRequest(BaseModel):
    candidates: List[Dict]
    job_id: str
    organization_id: str

class CandidateData(BaseModel):
    id: str
    name: str
    email: str
    skills: List[str]
    experience_years: int
    education: Optional[Dict] = {}
    location: Optional[str] = None
    resume_text: Optional[str] = ""

class BiasAnalysisResponse(BaseModel):
    job_id: str
    analysis_timestamp: str
    overall_bias_detected: bool
    bias_score: float
    total_candidates_analyzed: int
    detailed_analysis: Dict
    priority_recommendations: List[str]
    confidence_level: int

class BiasDetector:
    def __init__(self):
        self.fairness_thresholds = {
            "gender": {
                "min_female_ratio": 0.25,
                "max_male_ratio": 0.85,
                "target_balance": 0.4
            },
            "university_diversity": {
                "min_institutions": 5,
                "target_institutions": 8
            },
            "geographic_diversity": {
                "min_locations": 3,
                "target_locations": 5
            },
            "experience_balance": {
                "junior_ratio": {"min": 0.15, "max": 0.4},
                "senior_ratio": {"min": 0.2, "max": 0.6}
            },
            "age_diversity": {
                "min_age_range": 10,
                "target_age_range": 20
            }
        }
        
        # Gender indicators for basic gender detection
        self.male_indicators = [
            'he', 'his', 'him', 'mr', 'john', 'michael', 'david', 'robert', 'james',
            'william', 'richard', 'charles', 'joseph', 'thomas', 'christopher', 'daniel',
            'paul', 'mark', 'donald', 'george', 'kenneth', 'steven', 'edward', 'brian'
        ]
        
        self.female_indicators = [
            'she', 'her', 'hers', 'ms', 'mrs', 'mary', 'patricia', 'jennifer', 'linda',
            'elizabeth', 'barbara', 'susan', 'jessica', 'sarah', 'karen', 'nancy',
            'lisa', 'betty', 'helen', 'sandra', 'donna', 'carol', 'ruth', 'sharon'
        ]

    def detect_gender_bias(self, candidates: List[Dict]) -> Dict:
        """Detect gender bias in candidate selection"""
        if not candidates:
            return {
                "bias_detected": False,
                "stats": {"male": 0, "female": 0, "unknown": 0},
                "ratios": {"female_ratio": 0, "male_ratio": 0},
                "recommendations": []
            }

        gender_stats = {"male": 0, "female": 0, "unknown": 0}
        
        for candidate in candidates:
            name = candidate.get("name", "").lower()
            resume_text = candidate.get("resume_text", "").lower()
            email = candidate.get("email", "").lower()
            
            # Analyze name, resume text, and email for gender indicators
            full_text = f"{name} {resume_text} {email}"
            
            male_score = sum(1 for indicator in self.male_indicators if indicator in full_text)
            female_score = sum(1 for indicator in self.female_indicators if indicator in full_text)
            
            if male_score > female_score:
                gender_stats["male"] += 1
            elif female_score > male_score:
                gender_stats["female"] += 1
            else:
                gender_stats["unknown"] += 1
        
        total = len(candidates)
        female_ratio = gender_stats["female"] / total
        male_ratio = gender_stats["male"] / total
        
        # Check for bias
        thresholds = self.fairness_thresholds["gender"]
        bias_detected = (
            female_ratio < thresholds["min_female_ratio"] or
            male_ratio > thresholds["max_male_ratio"]
        )
        
        return {
            "bias_detected": bias_detected,
            "stats": gender_stats,
            "ratios": {
                "female_ratio": round(female_ratio, 3),
                "male_ratio": round(male_ratio, 3),
                "unknown_ratio": round(gender_stats["unknown"] / total, 3)
            },
            "severity": self._calculate_severity(female_ratio, thresholds["target_balance"]),
            "recommendations": self._generate_gender_recommendations(bias_detected, female_ratio, male_ratio)
        }

    def detect_university_bias(self, candidates: List[Dict]) -> Dict:
        """Detect university diversity bias"""
        if not candidates:
            return {
                "bias_detected": False,
                "unique_institutions": 0,
                "recommendations": []
            }

        institutions = set()
        education_levels = Counter()
        
        for candidate in candidates:
            education = candidate.get("education", {})
            institution = education.get("institution", "").strip().lower()
            degree_level = education.get("degree", "").strip().lower()
            
            if institution:
                institutions.add(institution)
            if degree_level:
                education_levels[degree_level] += 1
        
        unique_institutions = len(institutions)
        threshold = self.fairness_thresholds["university_diversity"]
        bias_detected = unique_institutions < threshold["min_institutions"]
        
        # Check for elite university bias
        elite_universities = {
            'harvard', 'mit', 'stanford', 'caltech', 'princeton', 'yale',
            'columbia', 'uchicago', 'upenn', 'dartmouth', 'cornell', 'brown'
        }
        
        elite_count = len([inst for inst in institutions if any(elite in inst for elite in elite_universities)])
        elite_ratio = elite_count / max(unique_institutions, 1)
        
        return {
            "bias_detected": bias_detected,
            "unique_institutions": unique_institutions,
            "institution_list": list(institutions)[:10],  # Limit for privacy
            "education_levels": dict(education_levels),
            "elite_university_ratio": round(elite_ratio, 3),
            "elite_bias_detected": elite_ratio > 0.6,
            "recommendations": self._generate_university_recommendations(bias_detected, unique_institutions, elite_ratio)
        }

    def detect_experience_bias(self, candidates: List[Dict]) -> Dict:
        """Detect experience level bias"""
        if not candidates:
            return {
                "bias_detected": False,
                "stats": {"junior": 0, "mid": 0, "senior": 0},
                "recommendations": []
            }

        experience_levels = {"junior": 0, "mid": 0, "senior": 0}
        experience_years = []
        
        for candidate in candidates:
            years = candidate.get("experience_years", 0)
            experience_years.append(years)
            
            if years <= 2:
                experience_levels["junior"] += 1
            elif years <= 7:
                experience_levels["mid"] += 1
            else:
                experience_levels["senior"] += 1
        
        total = len(candidates)
        if total == 0:
            return {"bias_detected": False, "stats": experience_levels, "recommendations": []}
        
        junior_ratio = experience_levels["junior"] / total
        senior_ratio = experience_levels["senior"] / total
        
        thresholds = self.fairness_thresholds["experience_balance"]
        bias_detected = (
            junior_ratio < thresholds["junior_ratio"]["min"] or
            senior_ratio > thresholds["senior_ratio"]["max"]
        )
        
        # Calculate experience statistics
        avg_experience = statistics.mean(experience_years) if experience_years else 0
        experience_std = statistics.stdev(experience_years) if len(experience_years) > 1 else 0
        
        return {
            "bias_detected": bias_detected,
            "stats": experience_levels,
            "ratios": {
                "junior_ratio": round(junior_ratio, 3),
                "mid_ratio": round(experience_levels["mid"] / total, 3),
                "senior_ratio": round(senior_ratio, 3)
            },
            "statistics": {
                "average_experience": round(avg_experience, 2),
                "experience_std_dev": round(experience_std, 2),
                "min_experience": min(experience_years) if experience_years else 0,
                "max_experience": max(experience_years) if experience_years else 0
            },
            "recommendations": self._generate_experience_recommendations(bias_detected, experience_levels, total)
        }

    def detect_location_bias(self, candidates: List[Dict]) -> Dict:
        """Detect geographic diversity bias"""
        if not candidates:
            return {
                "bias_detected": False,
                "unique_locations": 0,
                "recommendations": []
            }

        locations = set()
        location_counter = Counter()
        
        for candidate in candidates:
            location = candidate.get("location", "").strip().lower()
            if location and location not in ['', 'n/a', 'not specified']:
                # Extract city/state from location string
                location_clean = re.sub(r'[,.]', '', location).strip()
                locations.add(location_clean)
                location_counter[location_clean] += 1
        
        unique_locations = len(locations)
        threshold = self.fairness_thresholds["geographic_diversity"]
        bias_detected = unique_locations < threshold["min_locations"]
        
        # Check for concentration in specific areas
        if location_counter:
            most_common_location = location_counter.most_common(1)[0]
            concentration_ratio = most_common_location[1] / len(candidates)
        else:
            concentration_ratio = 0
        
        return {
            "bias_detected": bias_detected,
            "unique_locations": unique_locations,
            "location_distribution": dict(location_counter.most_common(5)),
            "concentration_ratio": round(concentration_ratio, 3),
            "high_concentration_detected": concentration_ratio > 0.5,
            "recommendations": self._generate_location_recommendations(bias_detected, unique_locations, concentration_ratio)
        }

    def detect_skills_bias(self, candidates: List[Dict]) -> Dict:
        """Detect bias in technical skills diversity"""
        if not candidates:
            return {
                "bias_detected": False,
                "unique_skills": 0,
                "recommendations": []
            }

        all_skills = []
        skill_counter = Counter()
        
        for candidate in candidates:
            skills = candidate.get("skills", [])
            if isinstance(skills, list):
                all_skills.extend([skill.lower().strip() for skill in skills])
            elif isinstance(skills, str):
                # Handle comma-separated skills string
                skills_list = [skill.lower().strip() for skill in skills.split(',')]
                all_skills.extend(skills_list)
        
        skill_counter = Counter(all_skills)
        unique_skills = len(skill_counter)
        
        # Check for over-reliance on specific skills
        if skill_counter:
            most_common_skills = skill_counter.most_common(3)
            top_skill_ratio = most_common_skills[0][1] / len(candidates) if most_common_skills else 0
        else:
            top_skill_ratio = 0
            most_common_skills = []
        
        bias_detected = unique_skills < 10 or top_skill_ratio > 0.8
        
        return {
            "bias_detected": bias_detected,
            "unique_skills": unique_skills,
            "total_skill_mentions": len(all_skills),
            "most_common_skills": dict(most_common_skills),
            "top_skill_dominance": round(top_skill_ratio, 3),
            "recommendations": self._generate_skills_recommendations(bias_detected, unique_skills, top_skill_ratio)
        }

    def _calculate_severity(self, actual_ratio: float, target_ratio: float) -> str:
        """Calculate bias severity level"""
        difference = abs(actual_ratio - target_ratio)
        if difference < 0.1:
            return "low"
        elif difference < 0.25:
            return "medium"
        else:
            return "high"

    def _generate_gender_recommendations(self, bias_detected: bool, female_ratio: float, male_ratio: float) -> List[str]:
        """Generate gender bias recommendations"""
        if not bias_detected:
            return ["Gender balance looks reasonable. Continue monitoring to maintain fairness."]
        
        recommendations = []
        if female_ratio < 0.25:
            recommendations.extend([
                "Review job descriptions for potentially exclusionary language",
                "Expand recruitment to women-focused professional networks and communities",
                "Consider implementing structured interviews to reduce unconscious bias",
                "Review sourcing channels for gender diversity",
                "Partner with organizations that support women in your industry"
            ])
        
        if male_ratio > 0.85:
            recommendations.extend([
                "Examine screening criteria for potential bias against diverse candidates",
                "Implement blind resume screening for initial reviews",
                "Ensure diverse interview panels",
                "Review company culture messaging for inclusivity"
            ])
        
        return recommendations[:5]  # Limit to top 5

    def _generate_university_recommendations(self, bias_detected: bool, count: int, elite_ratio: float) -> List[str]:
        """Generate university bias recommendations"""
        recommendations = []
        
        if bias_detected or count < 8:
            recommendations.extend([
                f"Only {count} unique institutions represented - expand recruitment beyond traditional sources",
                "Consider candidates from state schools, community colleges, and non-traditional backgrounds",
                "Review job requirements to focus on skills rather than specific degrees",
                "Partner with diverse educational institutions for recruitment"
            ])
        
        if elite_ratio > 0.6:
            recommendations.extend([
                "High concentration of candidates from elite universities detected",
                "Expand outreach to include candidates from diverse educational backgrounds",
                "Focus on skills and experience over university prestige",
                "Consider candidates with non-traditional educational paths"
            ])
        
        return recommendations[:5]

    def _generate_experience_recommendations(self, bias_detected: bool, levels: Dict, total: int) -> List[str]:
        """Generate experience bias recommendations"""
        if not bias_detected:
            return ["Experience level distribution looks balanced across junior, mid, and senior levels."]
        
        recommendations = []
        junior_ratio = levels["junior"] / total
        senior_ratio = levels["senior"] / total
        
        if junior_ratio < 0.15:
            recommendations.extend([
                "Consider entry-level candidates for growth opportunities and fresh perspectives",
                "Review if senior experience is truly required for all positions",
                "Create mentorship programs to support junior talent development"
            ])
        
        if senior_ratio > 0.6:
            recommendations.extend([
                "High concentration of senior candidates - consider cost-effectiveness",
                "Evaluate if all positions require senior-level experience",
                "Balance team composition with different experience levels"
            ])
        
        return recommendations[:5]

    def _generate_location_recommendations(self, bias_detected: bool, count: int, concentration: float) -> List[str]:
        """Generate location bias recommendations"""
        recommendations = []
        
        if bias_detected or count < 5:
            recommendations.extend([
                "Limited geographic diversity detected",
                "Expand recruitment to different metropolitan areas and regions",
                "Consider remote work options to access broader talent pools",
                "Review job posting distribution across different geographic markets"
            ])
        
        if concentration > 0.5:
            recommendations.extend([
                "High concentration of candidates from single location detected",
                "Diversify recruitment sources geographically",
                "Consider relocation assistance to attract remote talent"
            ])
        
        return recommendations[:4]

    def _generate_skills_recommendations(self, bias_detected: bool, unique_skills: int, top_skill_ratio: float) -> List[str]:
        """Generate skills bias recommendations"""
        recommendations = []
        
        if unique_skills < 10:
            recommendations.extend([
                "Limited skills diversity detected in candidate pool",
                "Expand recruitment to include candidates with complementary skill sets",
                "Review job requirements to avoid over-specification"
            ])
        
        if top_skill_ratio > 0.8:
            recommendations.extend([
                "Over-reliance on single skill set detected",
                "Consider candidates with transferable skills and growth potential",
                "Diversify technical requirements to build well-rounded teams"
            ])
        
        return recommendations[:4]

    def analyze_bias(self, request: BiasAnalysisRequest) -> Dict:
        """Main bias analysis function"""
        candidates = request.candidates
        
        if not candidates:
            return {
                "job_id": request.job_id,
                "analysis_timestamp": datetime.now().isoformat(),
                "overall_bias_detected": False,
                "bias_score": 0,
                "total_candidates_analyzed": 0,
                "detailed_analysis": {},
                "priority_recommendations": ["No candidates to analyze"],
                "confidence_level": 100
            }
        
        # Perform different bias analyses
        gender_analysis = self.detect_gender_bias(candidates)
        university_analysis = self.detect_university_bias(candidates)
        experience_analysis = self.detect_experience_bias(candidates)
        location_analysis = self.detect_location_bias(candidates)
        skills_analysis = self.detect_skills_bias(candidates)
        
        # Calculate overall bias score
        bias_factors = [
            gender_analysis["bias_detected"],
            university_analysis["bias_detected"],
            experience_analysis["bias_detected"],
            location_analysis["bias_detected"],
            skills_analysis["bias_detected"]
        ]
        
        bias_score = sum(bias_factors) / len(bias_factors) * 100
        overall_bias_detected = any(bias_factors)
        
        # Calculate confidence level based on data quality
        confidence = self._calculate_confidence_level(candidates, gender_analysis, university_analysis)
        
        return {
            "job_id": request.job_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "overall_bias_detected": overall_bias_detected,
            "bias_score": round(bias_score, 1),
            "total_candidates_analyzed": len(candidates),
            "detailed_analysis": {
                "gender": gender_analysis,
                "university": university_analysis,
                "experience": experience_analysis,
                "location": location_analysis,
                "skills": skills_analysis
            },
            "priority_recommendations": self._get_priority_recommendations(
                gender_analysis, university_analysis, experience_analysis, 
                location_analysis, skills_analysis
            ),
            "confidence_level": confidence
        }

    def _calculate_confidence_level(self, candidates: List[Dict], gender_analysis: Dict, university_analysis: Dict) -> int:
        """Calculate confidence level in bias analysis"""
        base_confidence = 70
        
        # Increase confidence with more candidates
        if len(candidates) > 20:
            base_confidence += 15
        elif len(candidates) > 10:
            base_confidence += 10
        elif len(candidates) < 5:
            base_confidence -= 20
        
        # Increase confidence if we have good data quality
        unknown_gender_ratio = gender_analysis["ratios"].get("unknown_ratio", 1)
        if unknown_gender_ratio < 0.3:
            base_confidence += 10
        elif unknown_gender_ratio > 0.7:
            base_confidence -= 15
        
        # Adjust based on data completeness
        complete_profiles = sum(1 for candidate in candidates 
                              if candidate.get("name") and candidate.get("skills"))
        completion_ratio = complete_profiles / len(candidates)
        
        if completion_ratio > 0.8:
            base_confidence += 10
        elif completion_ratio < 0.5:
            base_confidence -= 15
        
        return max(min(base_confidence, 95), 30)

    def _get_priority_recommendations(self, gender: Dict, university: Dict, 
                                    experience: Dict, location: Dict, skills: Dict) -> List[str]:
        """Get top priority recommendations across all bias types"""
        all_recommendations = []
        
        # Prioritize by severity and impact
        if gender["bias_detected"]:
            all_recommendations.extend(gender["recommendations"][:2])
        
        if university["bias_detected"]:
            all_recommendations.extend(university["recommendations"][:1])
        
        if experience["bias_detected"]:
            all_recommendations.extend(experience["recommendations"][:1])
        
        if location["bias_detected"]:
            all_recommendations.extend(location["recommendations"][:1])
        
        if skills["bias_detected"]:
            all_recommendations.extend(skills["recommendations"][:1])
        
        # If no bias detected, provide general recommendations
        if not any([gender["bias_detected"], university["bias_detected"], 
                   experience["bias_detected"], location["bias_detected"], skills["bias_detected"]]):
            all_recommendations = [
                "Great job! No significant bias detected in current candidate pool",
                "Continue monitoring diversity metrics to maintain fairness",
                "Consider implementing regular bias audits for ongoing improvement"
            ]
        
        return all_recommendations[:8]  # Limit to top 8

# Initialize bias detector
detector = BiasDetector()

# API Endpoints
@app.get("/")
async def root():
    return {
        "service": "Bias Detection Service",
        "status": "healthy",
        "version": "1.0.0",
        "features": ["Gender Bias", "University Bias", "Experience Bias", "Location Bias", "Skills Bias"],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/bias/detect")
async def detect_bias(
    request: BiasAnalysisRequest,
    current_user: dict = Depends(verify_token)
):
    """Analyze candidate pool for various types of bias"""
    try:
        # Verify user has permission (recruiters and admins only)
        if current_user["user"]["role"] not in ["recruiter", "admin"]:
            raise HTTPException(
                status_code=403, 
                detail="Only recruiters and administrators can perform bias analysis"
            )
        
        # Perform bias analysis
        analysis = detector.analyze_bias(request)
        
        # Store analysis in database for tracking
        connection = get_db_connection()
        cursor = connection.cursor()
        
        try:
            bias_analysis_id = f"bias_{request.job_id}_{int(datetime.now().timestamp())}"
            
            cursor.execute("""
                INSERT INTO bias_analyses 
                (id, job_id, organization_id, analyzed_by, total_candidates, 
                 overall_bias_detected, bias_score, confidence_level, 
                 detailed_analysis, recommendations, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                bias_analysis_id,
                request.job_id,
                request.organization_id,
                current_user["user"]["id"],
                analysis["total_candidates_analyzed"],
                analysis["overall_bias_detected"],
                analysis["bias_score"],
                analysis["confidence_level"],
                json.dumps(analysis["detailed_analysis"]),
                json.dumps(analysis["priority_recommendations"]),
                datetime.now()
            ))
            
            connection.commit()
            
        except mysql.connector.Error as e:
            # If table doesn't exist, continue without storing (graceful degradation)
            print(f"Database storage failed (table may not exist): {str(e)}")
            
        finally:
            cursor.close()
            connection.close()
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bias analysis failed: {str(e)}")

@app.get("/api/v1/bias/trends/{job_id}")
async def get_bias_trends(
    job_id: str, 
    days: int = 30,
    current_user: dict = Depends(verify_token)
):
    """Get bias trends for a specific job over time"""
    if current_user["user"]["role"] not in ["recruiter", "admin"]:
        raise HTTPException(status_code=403, detail="Access forbidden")
    
    # This would fetch historical data from database in production
    # For now, return mock trend data
    return {
        "job_id": job_id,
        "period_days": days,
        "trend_data": {
            "gender_balance_trend": [0.25, 0.28, 0.32, 0.35, 0.38],
            "university_diversity_trend": [5, 6, 7, 8, 9],
            "experience_diversity_trend": [0.6, 0.65, 0.7, 0.75, 0.8],
            "location_diversity_trend": [3, 4, 4, 5, 6],
            "overall_fairness_score": [72, 75, 78, 82, 85]
        },
        "improvement_rate": "+18% improvement in diversity metrics over the selected period",
        "analysis_timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/bias/metrics/summary")
async def get_bias_metrics_summary(current_user: dict = Depends(verify_token)):
    """Get overall bias metrics summary across all jobs"""
    if current_user["user"]["role"] not in ["recruiter", "admin"]:
        raise HTTPException(status_code=403, detail="Access forbidden")
    
    # Mock data - in production this would aggregate from database
    return {
        "summary": {
            "total_analyses_conducted": 156,
            "bias_detected_percentage": 23.1,
            "most_common_bias_type": "University bias (elite school preference)",
            "average_diversity_score": 78.4,
            "improvement_trend": "+12% over last quarter"
        },
        "bias_breakdown": {
            "gender_bias": {"detected": 18, "percentage": 11.5},
            "university_bias": {"detected": 31, "percentage": 19.9},
            "experience_bias": {"detected": 24, "percentage": 15.4},
            "location_bias": {"detected": 15, "percentage": 9.6},
            "skills_bias": {"detected": 12, "percentage": 7.7}
        },
        "recommendations_impact": {
            "implemented": 89,
            "pending": 23,
            "success_rate": 0.79
        },
        "generated_at": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        db_status = "connected"
        cursor.close()
        connection.close()
    except:
        db_status = "disconnected"
    
    return {
        "status": "healthy",
        "service": "bias-detection",
        "port": 8005,
        "database": db_status,
        "features": ["Gender Bias Detection", "University Bias Detection", "Experience Bias Detection", "Location Bias Detection", "Skills Bias Detection"],
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
