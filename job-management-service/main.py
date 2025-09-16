from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import mysql.connector
from mysql.connector import Error
import requests
import os
import uuid
from datetime import datetime, timedelta
from enum import Enum
import json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Job Management Service",
    description="Handles job postings, applications, and job matching",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8001",  # Auth service
        "http://localhost:8002",  # Resume service
        "http://localhost:8004"   # AI evaluation service
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
            status_code=500,
            detail="Database connection failed"
        )

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class JobType(str, Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERNSHIP = "internship"
    FREELANCE = "freelance"

class JobStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"
    FILLED = "filled"

class ApplicationStatus(str, Enum):
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    SHORTLISTED = "shortlisted"
    INTERVIEW_SCHEDULED = "interview_scheduled"
    REJECTED = "rejected"
    OFFERED = "offered"
    HIRED = "hired"

class JobCreate(BaseModel):
    title: str = Field(..., min_length=3, max_length=200)
    description: str = Field(..., min_length=50)
    requirements: Dict[str, Any] = Field(default_factory=dict)
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    experience_level: str = Field(default="mid")  # junior, mid, senior
    education_level: Optional[str] = None
    job_type: JobType = Field(default=JobType.FULL_TIME)
    location: Optional[str] = None
    is_remote: bool = Field(default=False)
    salary_min: Optional[int] = Field(None, ge=0)
    salary_max: Optional[int] = Field(None, ge=0)
    benefits: List[str] = Field(default_factory=list)
    company_name: str = Field(..., min_length=2)
    department: Optional[str] = None
    deadline: Optional[datetime] = None

class JobUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    requirements: Optional[Dict[str, Any]] = None
    required_skills: Optional[List[str]] = None
    preferred_skills: Optional[List[str]] = None
    experience_level: Optional[str] = None
    education_level: Optional[str] = None
    job_type: Optional[JobType] = None
    location: Optional[str] = None
    is_remote: Optional[bool] = None
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    benefits: Optional[List[str]] = None
    department: Optional[str] = None
    deadline: Optional[datetime] = None
    status: Optional[JobStatus] = None

class JobResponse(BaseModel):
    id: str
    title: str
    description: str
    requirements: Dict[str, Any]
    required_skills: List[str]
    preferred_skills: List[str]
    experience_level: str
    education_level: Optional[str]
    job_type: str
    location: Optional[str]
    is_remote: bool
    salary_min: Optional[int]
    salary_max: Optional[int]
    benefits: List[str]
    company_name: str
    department: Optional[str]
    status: str
    posted_by: str
    deadline: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    applications_count: int

class ApplicationCreate(BaseModel):
    job_id: str
    cover_letter: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None

class ApplicationResponse(BaseModel):
    id: str
    job_id: str
    job_title: str
    company_name: str
    candidate_id: str
    candidate_name: str
    candidate_email: str
    resume_id: Optional[str]
    cover_letter: Optional[str]
    status: str
    ai_score: Optional[int]
    applied_at: datetime
    updated_at: datetime

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def verify_token(token: str = Depends(security)):
    """Verify JWT token with auth service"""
    try:
        response = requests.get(
            "http://localhost:8001/profile",
            headers={"Authorization": f"Bearer {token.credentials}"},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=401, detail="Invalid token")
    except:
        raise HTTPException(status_code=401, detail="Token verification failed")

def get_job_by_id(job_id: str, connection):
    """Get job details by ID"""
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT j.*, u.email as recruiter_email,
                   COUNT(a.id) as applications_count
            FROM jobs j
            JOIN users u ON j.posted_by = u.id
            LEFT JOIN applications a ON j.id = a.job_id
            WHERE j.id = %s
            GROUP BY j.id
        """, (job_id,))
        return cursor.fetchone()
    finally:
        cursor.close()

# =============================================================================
# DATABASE SETUP
# =============================================================================

def create_tables():
    """Create necessary tables if they don't exist"""
    connection = get_db_connection()
    cursor = connection.cursor()
    
    try:
        # Jobs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
                title VARCHAR(255) NOT NULL,
                description TEXT NOT NULL,
                requirements JSON,
                required_skills JSON,
                preferred_skills JSON,
                experience_level ENUM('junior', 'mid', 'senior') DEFAULT 'mid',
                education_level VARCHAR(100),
                job_type ENUM('full_time', 'part_time', 'contract', 'internship', 'freelance') DEFAULT 'full_time',
                location VARCHAR(255),
                is_remote BOOLEAN DEFAULT FALSE,
                salary_min INT,
                salary_max INT,
                benefits JSON,
                company_name VARCHAR(255) NOT NULL,
                department VARCHAR(255),
                status ENUM('draft', 'active', 'paused', 'closed', 'filled') DEFAULT 'draft',
                posted_by CHAR(36) NOT NULL,
                deadline DATETIME,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (posted_by) REFERENCES users(id) ON DELETE CASCADE,
                INDEX idx_jobs_status (status),
                INDEX idx_jobs_posted_by (posted_by),
                INDEX idx_jobs_location (location),
                INDEX idx_jobs_created (created_at)
            )
        """)
        
        # Update applications table if needed
        cursor.execute("""
            ALTER TABLE applications 
            ADD COLUMN IF NOT EXISTS cover_letter TEXT,
            ADD COLUMN IF NOT EXISTS additional_info JSON,
            ADD COLUMN IF NOT EXISTS ai_score INT,
            ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        """)
        
        connection.commit()
        print("✅ Database tables created/updated successfully")
        
    except Error as e:
        print(f"❌ Database setup error: {e}")
    finally:
        cursor.close()
        connection.close()

# Initialize database on startup
create_tables()

# =============================================================================
# JOB MANAGEMENT ENDPOINTS
# =============================================================================

@app.post("/api/v1/jobs", response_model=JobResponse)
async def create_job(job: JobCreate, current_user: dict = Depends(verify_token)):
    """Create a new job posting (recruiters/admins only)"""
    if current_user["user"]["role"] not in ["recruiter", "admin"]:
        raise HTTPException(status_code=403, detail="Only recruiters can create jobs")
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        job_id = str(uuid.uuid4())
        
        cursor.execute("""
            INSERT INTO jobs (
                id, title, description, requirements, required_skills, preferred_skills,
                experience_level, education_level, job_type, location, is_remote,
                salary_min, salary_max, benefits, company_name, department,
                posted_by, deadline, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            job_id, job.title, job.description, json.dumps(job.requirements),
            json.dumps(job.required_skills), json.dumps(job.preferred_skills),
            job.experience_level, job.education_level, job.job_type.value,
            job.location, job.is_remote, job.salary_min, job.salary_max,
            json.dumps(job.benefits), job.company_name, job.department,
            current_user["user"]["id"], job.deadline, "active"
        ))
        
        connection.commit()
        
        # Return created job
        created_job = get_job_by_id(job_id, connection)
        return JobResponse(
            id=created_job["id"],
            title=created_job["title"],
            description=created_job["description"],
            requirements=json.loads(created_job["requirements"] or "{}"),
            required_skills=json.loads(created_job["required_skills"] or "[]"),
            preferred_skills=json.loads(created_job["preferred_skills"] or "[]"),
            experience_level=created_job["experience_level"],
            education_level=created_job["education_level"],
            job_type=created_job["job_type"],
            location=created_job["location"],
            is_remote=created_job["is_remote"],
            salary_min=created_job["salary_min"],
            salary_max=created_job["salary_max"],
            benefits=json.loads(created_job["benefits"] or "[]"),
            company_name=created_job["company_name"],
            department=created_job["department"],
            status=created_job["status"],
            posted_by=created_job["posted_by"],
            deadline=created_job["deadline"],
            created_at=created_job["created_at"],
            updated_at=created_job["updated_at"],
            applications_count=created_job["applications_count"]
        )
        
    finally:
        cursor.close()
        connection.close()

@app.get("/api/v1/jobs")
async def list_jobs(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    status: Optional[str] = Query(None),
    location: Optional[str] = Query(None),
    job_type: Optional[str] = Query(None),
    remote_only: bool = Query(False),
    search: Optional[str] = Query(None),
    current_user: dict = Depends(verify_token)
):
    """List all jobs with filtering and search"""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Build query with filters
        where_conditions = ["j.status != 'draft'"]  # Hide draft jobs from public
        params = []
        
        if current_user["user"]["role"] in ["recruiter", "admin"]:
            where_conditions = []  # Recruiters can see all jobs including drafts
        
        if status:
            where_conditions.append("j.status = %s")
            params.append(status)
        
        if location:
            where_conditions.append("j.location LIKE %s")
            params.append(f"%{location}%")
        
        if job_type:
            where_conditions.append("j.job_type = %s")
            params.append(job_type)
        
        if remote_only:
            where_conditions.append("j.is_remote = TRUE")
        
        if search:
            where_conditions.append("(j.title LIKE %s OR j.description LIKE %s OR j.company_name LIKE %s)")
            params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])
        
        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        query = f"""
            SELECT j.*, u.email as recruiter_email,
                    COUNT(a.id) as applications_count
            FROM jobs j
            JOIN users u ON j.posted_by = u.id
            LEFT JOIN applications a ON j.id = a.job_id
            {where_clause}
            GROUP BY j.id
            ORDER BY j.created_at DESC
            LIMIT %s OFFSET %s
        """
        
        params.extend([limit, skip])
        cursor.execute(query, params)
        jobs = cursor.fetchall()
        
        # Count total jobs
        count_query = f"SELECT COUNT(DISTINCT j.id) as total FROM jobs j JOIN users u ON j.posted_by = u.id {where_clause}"
        cursor.execute(count_query, params[:-2])  # Exclude limit and offset
        total = cursor.fetchone()["total"]
        
        # Format response
        job_list = []
        for job in jobs:
            job_list.append(JobResponse(
                id=job["id"],
                title=job["title"],
                description=job["description"][:200] + "..." if len(job["description"]) > 200 else job["description"],
                requirements=json.loads(job["requirements"] or "{}"),
                required_skills=json.loads(job["required_skills"] or "[]"),
                preferred_skills=json.loads(job["preferred_skills"] or "[]"),
                experience_level=job["experience_level"],
                education_level=job["education_level"],
                job_type=job["job_type"],
                location=job["location"],
                is_remote=job["is_remote"],
                salary_min=job["salary_min"],
                salary_max=job["salary_max"],
                benefits=json.loads(job["benefits"] or "[]"),
                company_name=job["company_name"],
                department=job["department"],
                status=job["status"],
                posted_by=job["posted_by"],
                deadline=job["deadline"],
                created_at=job["created_at"],
                updated_at=job["updated_at"],
                applications_count=job["applications_count"]
            ))
        
        return {
            "jobs": job_list,
            "total": total,
            "skip": skip,
            "limit": limit
        }
        
    finally:
        cursor.close()
        connection.close()

@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, current_user: dict = Depends(verify_token)):
    """Get job details by ID"""
    connection = get_db_connection()
    
    try:
        job = get_job_by_id(job_id, connection)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Check if user can view this job
        if job["status"] == "draft" and current_user["user"]["role"] not in ["recruiter", "admin"]:
            if job["posted_by"] != current_user["user"]["id"]:
                raise HTTPException(status_code=404, detail="Job not found")
        
        return JobResponse(
            id=job["id"],
            title=job["title"],
            description=job["description"],
            requirements=json.loads(job["requirements"] or "{}"),
            required_skills=json.loads(job["required_skills"] or "[]"),
            preferred_skills=json.loads(job["preferred_skills"] or "[]"),
            experience_level=job["experience_level"],
            education_level=job["education_level"],
            job_type=job["job_type"],
            location=job["location"],
            is_remote=job["is_remote"],
            salary_min=job["salary_min"],
            salary_max=job["salary_max"],
            benefits=json.loads(job["benefits"] or "[]"),
            company_name=job["company_name"],
            department=job["department"],
            status=job["status"],
            posted_by=job["posted_by"],
            deadline=job["deadline"],
            created_at=job["created_at"],
            updated_at=job["updated_at"],
            applications_count=job["applications_count"]
        )
        
    finally:
        connection.close()

@app.put("/api/v1/jobs/{job_id}", response_model=JobResponse)
async def update_job(
    job_id: str, 
    job_update: JobUpdate, 
    current_user: dict = Depends(verify_token)
):
    """Update job details (only by job creator or admin)"""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Check if job exists and user has permission
        existing_job = get_job_by_id(job_id, connection)
        if not existing_job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if (current_user["user"]["role"] != "admin" and 
            existing_job["posted_by"] != current_user["user"]["id"]):
            raise HTTPException(status_code=403, detail="Not authorized to update this job")
        
        # Build update query dynamically
        update_fields = []
        params = []
        
        for field, value in job_update.dict(exclude_unset=True).items():
            if value is not None:
                if field in ["requirements", "required_skills", "preferred_skills", "benefits"]:
                    update_fields.append(f"{field} = %s")
                    params.append(json.dumps(value))
                elif field == "job_type":
                    update_fields.append(f"{field} = %s")
                    params.append(value.value if hasattr(value, 'value') else value)
                else:
                    update_fields.append(f"{field} = %s")
                    params.append(value)
        
        if not update_fields:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        query = f"UPDATE jobs SET {', '.join(update_fields)} WHERE id = %s"
        params.append(job_id)
        
        cursor.execute(query, params)
        connection.commit()
        
        # Return updated job
        updated_job = get_job_by_id(job_id, connection)
        return JobResponse(
            id=updated_job["id"],
            title=updated_job["title"],
            description=updated_job["description"],
            requirements=json.loads(updated_job["requirements"] or "{}"),
            required_skills=json.loads(updated_job["required_skills"] or "[]"),
            preferred_skills=json.loads(updated_job["preferred_skills"] or "[]"),
            experience_level=updated_job["experience_level"],
            education_level=updated_job["education_level"],
            job_type=updated_job["job_type"],
            location=updated_job["location"],
            is_remote=updated_job["is_remote"],
            salary_min=updated_job["salary_min"],
            salary_max=updated_job["salary_max"],
            benefits=json.loads(updated_job["benefits"] or "[]"),
            company_name=updated_job["company_name"],
            department=updated_job["department"],
            status=updated_job["status"],
            posted_by=updated_job["posted_by"],
            deadline=updated_job["deadline"],
            created_at=updated_job["created_at"],
            updated_at=updated_job["updated_at"],
            applications_count=updated_job["applications_count"]
        )
        
    finally:
        cursor.close()
        connection.close()

@app.delete("/api/v1/jobs/{job_id}")
async def delete_job(job_id: str, current_user: dict = Depends(verify_token)):
    """Delete a job (only by creator or admin)"""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Check if job exists and user has permission
        existing_job = get_job_by_id(job_id, connection)
        if not existing_job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if (current_user["user"]["role"] != "admin" and 
            existing_job["posted_by"] != current_user["user"]["id"]):
            raise HTTPException(status_code=403, detail="Not authorized to delete this job")
        
        # Check if job has applications
        cursor.execute("SELECT COUNT(*) as count FROM applications WHERE job_id = %s", (job_id,))
        app_count = cursor.fetchone()["count"]
        
        if app_count > 0:
            # Don't actually delete, just mark as closed
            cursor.execute("UPDATE jobs SET status = 'closed' WHERE id = %s", (job_id,))
            connection.commit()
            return {"message": f"Job closed due to {app_count} existing applications"}
        else:
            # Safe to delete
            cursor.execute("DELETE FROM jobs WHERE id = %s", (job_id,))
            connection.commit()
            return {"message": "Job deleted successfully"}
        
    finally:
        cursor.close()
        connection.close()

# =============================================================================
# APPLICATION MANAGEMENT ENDPOINTS
# =============================================================================

@app.post("/api/v1/jobs/{job_id}/apply")
async def apply_to_job(
    job_id: str, 
    application: ApplicationCreate, 
    current_user: dict = Depends(verify_token)
):
    """Apply to a job (candidates only)"""
    if current_user["user"]["role"] != "candidate":
        raise HTTPException(status_code=403, detail="Only candidates can apply to jobs")
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Check if job exists and is active
        cursor.execute("SELECT * FROM jobs WHERE id = %s", (job_id,))
        job = cursor.fetchone()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job["status"] not in ["active"]:
            raise HTTPException(status_code=400, detail="Job is not accepting applications")
        
        # Check if already applied
        cursor.execute("""
            SELECT id FROM applications 
            WHERE job_id = %s AND candidate_id = %s
        """, (job_id, current_user["user"]["id"]))
        
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Already applied to this job")
        
        # Get candidate's latest resume
        cursor.execute("""
            SELECT id FROM resumes 
            WHERE candidate_id = %s 
            ORDER BY upload_date DESC 
            LIMIT 1
        """, (current_user["user"]["id"],))
        
        resume_result = cursor.fetchone()
        resume_id = resume_result["id"] if resume_result else None
        
        # Create application
        application_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO applications (
                id, job_id, candidate_id, resume_id, 
                cover_letter, additional_info, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            application_id, job_id, current_user["user"]["id"], resume_id,
            application.cover_letter, json.dumps(application.additional_info or {}), 
            "submitted"
        ))
        
        connection.commit()
        
        # Trigger AI evaluation in background (call AI evaluation service)
        try:
            # Get candidate profile for AI evaluation
            cursor.execute("SELECT * FROM candidate_profile WHERE user_id = %s", (current_user["user"]["id"],))
            profile = cursor.fetchone()
            
            if profile and resume_id:
                # Get resume content
                cursor.execute("SELECT content_text FROM resumes WHERE id = %s", (resume_id,))
                resume_data = cursor.fetchone()
                
                # Call AI evaluation service
                ai_payload = {
                    "job_requirements": {
                        "title": job["title"],
                        "required_skills": json.loads(job["required_skills"] or "[]"),
                        "required_experience": 3,  # Default, should be from job
                        "description": job["description"]
                    },
                    "candidate_profile": {
                        "name": profile.get("full_name", "Candidate"),
                        "skills": profile.get("skills", "").split(", ") if profile.get("skills") else [],
                        "experience_years": profile.get("experience_years", 0)
                    },
                    "resume_content": resume_data["content_text"] if resume_data else ""
                }
                
                ai_response = requests.post(
                    "http://localhost:8004/api/v1/ai-evaluation/match",
                    json=ai_payload,
                    headers={"Authorization": f"Bearer {current_user['user']['id']}"},
                    timeout=10
                )
                
                if ai_response.status_code == 200:
                    ai_result = ai_response.json()
                    ai_score = ai_result["scores"]["overall_score"]
                    
                    # Update application with AI score
                    cursor.execute("""
                        UPDATE applications SET ai_score = %s WHERE id = %s
                    """, (ai_score, application_id))
                    connection.commit()
        
        except Exception as e:
            print(f"AI evaluation failed: {e}")
            # Continue without AI score
            pass
        
        return {
            "message": "Application submitted successfully",
            "application_id": application_id,
            "job_title": job["title"],
            "company": job["company_name"],
            "applied_at": datetime.now().isoformat()
        }
        
    finally:
        cursor.close()
        connection.close()

@app.get("/api/v1/applications")
async def get_my_applications(current_user: dict = Depends(verify_token)):
    """Get current user's applications (candidates) or job applications (recruiters)"""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        if current_user["user"]["role"] == "candidate":
            # Get candidate's own applications
            cursor.execute("""
                SELECT a.*, j.title as job_title, j.company_name,
                        j.location, j.job_type, j.salary_min, j.salary_max
                FROM applications a
                JOIN jobs j ON a.job_id = j.id
                WHERE a.candidate_id = %s
                ORDER BY a.created_at DESC
            """, (current_user["user"]["id"],))
            
            applications = cursor.fetchall()
            
            return {
                "applications": [
                    {
                        "id": app["id"],
                        "job_id": app["job_id"],
                        "job_title": app["job_title"],
                        "company_name": app["company_name"],
                        "location": app["location"],
                        "job_type": app["job_type"],
                        "salary_range": f"${app['salary_min']:,} - ${app['salary_max']:,}" if app["salary_min"] and app["salary_max"] else None,
                        "status": app["status"],
                        "ai_score": app.get("ai_score"),
                        "applied_at": app["created_at"],
                        "updated_at": app.get("updated_at", app["created_at"])
                    }
                    for app in applications
                ],
                "total": len(applications)
            }
            
        elif current_user["user"]["role"] in ["recruiter", "admin"]:
            # Get applications for recruiter's jobs
            cursor.execute("""
                SELECT a.*, j.title as job_title, j.company_name,
                        u.email as candidate_email, 
                        cp.full_name as candidate_name
                FROM applications a
                JOIN jobs j ON a.job_id = j.id
                JOIN users u ON a.candidate_id = u.id
                LEFT JOIN candidate_profile cp ON a.candidate_id = cp.user_id
                WHERE j.posted_by = %s
                ORDER BY a.created_at DESC
            """, (current_user["user"]["id"],))
            
            applications = cursor.fetchall()
            
            return {
                "applications": [
                    ApplicationResponse(
                        id=app["id"],
                        job_id=app["job_id"],
                        job_title=app["job_title"],
                        company_name=app["company_name"],
                        candidate_id=app["candidate_id"],
                        candidate_name=app.get("candidate_name") or app["candidate_email"],
                        candidate_email=app["candidate_email"],
                        resume_id=app.get("resume_id"),
                        cover_letter=app.get("cover_letter"),
                        status=app["status"],
                        ai_score=app.get("ai_score"),
                        applied_at=app["created_at"],
                        updated_at=app.get("updated_at", app["created_at"])
                    )
                    for app in applications
                ],
                "total": len(applications)
            }
        
        else:
            raise HTTPException(status_code=403, detail="Access denied")
            
    finally:
        cursor.close()
        connection.close()

@app.put("/api/v1/applications/{application_id}/status")
async def update_application_status(
    application_id: str, 
    status: ApplicationStatus,
    current_user: dict = Depends(verify_token)
):
    """Update application status (recruiters only)"""
    if current_user["user"]["role"] not in ["recruiter", "admin"]:
        raise HTTPException(status_code=403, detail="Only recruiters can update application status")
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Check if application exists and recruiter owns the job
        cursor.execute("""
            SELECT a.*, j.posted_by
            FROM applications a
            JOIN jobs j ON a.job_id = j.id
            WHERE a.id = %s
        """, (application_id,))
        
        application = cursor.fetchone()
        if not application:
            raise HTTPException(status_code=404, detail="Application not found")
        
        if (current_user["user"]["role"] != "admin" and 
            application["posted_by"] != current_user["user"]["id"]):
            raise HTTPException(status_code=403, detail="Not authorized to update this application")
        
        # Update status
        cursor.execute("""
            UPDATE applications SET status = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (status.value, application_id))
        
        connection.commit()
        
        return {
            "message": f"Application status updated to {status.value}",
            "application_id": application_id,
            "new_status": status.value
        }
        
    finally:
        cursor.close()
        connection.close()

@app.get("/api/v1/jobs/{job_id}/applications")
async def get_job_applications(
    job_id: str, 
    current_user: dict = Depends(verify_token)
):
    """Get all applications for a specific job (recruiters only)"""
    if current_user["user"]["role"] not in ["recruiter", "admin"]:
        raise HTTPException(status_code=403, detail="Only recruiters can view job applications")
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Check if job exists and user has permission
        job = get_job_by_id(job_id, connection)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if (current_user["user"]["role"] != "admin" and 
            job["posted_by"] != current_user["user"]["id"]):
            raise HTTPException(status_code=403, detail="Not authorized to view these applications")
        
        # Get applications
        cursor.execute("""
            SELECT a.*, u.email as candidate_email,
                    cp.full_name as candidate_name, cp.phone, cp.location,
                    cp.experience_years, cp.skills, cp.linkedin_url
            FROM applications a
            JOIN users u ON a.candidate_id = u.id
            LEFT JOIN candidate_profile cp ON a.candidate_id = cp.user_id
            WHERE a.job_id = %s
            ORDER BY a.ai_score DESC, a.created_at DESC
        """, (job_id,))
        
        applications = cursor.fetchall()
        
        return {
            "job_title": job["title"],
            "company_name": job["company_name"],
            "applications": [
                {
                    "id": app["id"],
                    "candidate_id": app["candidate_id"],
                    "candidate_name": app.get("candidate_name") or app["candidate_email"],
                    "candidate_email": app["candidate_email"],
                    "phone": app.get("phone"),
                    "location": app.get("location"),
                    "experience_years": app.get("experience_years", 0),
                    "skills": app.get("skills", "").split(", ") if app.get("skills") else [],
                    "linkedin_url": app.get("linkedin_url"),
                    "resume_id": app.get("resume_id"),
                    "cover_letter": app.get("cover_letter"),
                    "status": app["status"],
                    "ai_score": app.get("ai_score"),
                    "applied_at": app["created_at"],
                    "updated_at": app.get("updated_at")
                }
                for app in applications
            ],
            "total": len(applications)
        }
        
    finally:
        cursor.close()
        connection.close()

# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
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
        "service": "job-management",
        "port": 8003,
        "database": db_status,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/jobs/stats")
async def get_job_stats(current_user: dict = Depends(verify_token)):
    """Get job statistics for dashboard"""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        if current_user["user"]["role"] == "candidate":
            # Candidate statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_applications,
                    SUM(CASE WHEN status = 'submitted' THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN status = 'shortlisted' THEN 1 ELSE 0 END) as shortlisted,
                    SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected,
                    AVG(ai_score) as avg_ai_score
                FROM applications 
                WHERE candidate_id = %s
            """, (current_user["user"]["id"],))
            
            stats = cursor.fetchone()
            return {
                "total_applications": stats["total_applications"] or 0,
                "pending": stats["pending"] or 0,
                "shortlisted": stats["shortlisted"] or 0,
                "rejected": stats["rejected"] or 0,
                "average_ai_score": round(stats["avg_ai_score"] or 0, 1)
            }
            
        elif current_user["user"]["role"] in ["recruiter", "admin"]:
            # Recruiter statistics
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT j.id) as total_jobs,
                    SUM(CASE WHEN j.status = 'active' THEN 1 ELSE 0 END) as active_jobs,
                    COUNT(a.id) as total_applications,
                    AVG(a.ai_score) as avg_candidate_score
                FROM jobs j
                LEFT JOIN applications a ON j.id = a.job_id
                WHERE j.posted_by = %s
            """, (current_user["user"]["id"],))
            
            stats = cursor.fetchone()
            return {
                "total_jobs": stats["total_jobs"] or 0,
                "active_jobs": stats["active_jobs"] or 0,
                "total_applications": stats["total_applications"] or 0,
                "average_candidate_score": round(stats["avg_candidate_score"] or 0, 1)
            }
        
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
