from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import mysql.connector
from mysql.connector import Error
from typing import Any, Dict, Optional, List
import os
import uuid
from pathlib import Path 
import PyPDF2
import json
import re  
import aiofiles
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Human AI Recruitment Agent",
    description="An AI-powered recruitment agent to streamline hiring processes.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ✅ FIXED: Consistent naming
UPLOAD_DIR = "uploads/resumes"
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

secret_key = os.getenv("JWT_SECRET_KEY")
algorithm = os.getenv("JWT_ALGORITHM", "HS256")
access_token_expire_minutes = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))

# Security
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

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
            detail="Error connecting to the database"
        )

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class UserRegistration(BaseModel):
    email: EmailStr
    password: str
    role: Optional[str] = "candidate"
    
class UserLogin(BaseModel):
    email: EmailStr
    password: str
    
class UserResponse(BaseModel):
    id: str
    email: str
    role: str
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str

class CandidateProfileCreate(BaseModel):
    full_name: str
    phone: Optional[str] = None
    location: Optional[str] = None
    experience_years: Optional[int] = 0
    linkedin_url: Optional[str] = None

class CandidateProfileResponse(BaseModel):
    id: str
    user_id: str
    full_name: str
    phone: Optional[str] = None
    location: Optional[str] = None
    experience_years: Optional[int] = 0
    skills: Optional[str]
    resume_url: Optional[str]
    linkedin_url: Optional[str]
    profile_completed: bool
    ai_score: int
    created_at: datetime

class ResumeUploadResponse(BaseModel):
    id: str
    filename: str
    file_size: int
    upload_date: datetime
    parsed_data: Dict[str, Any]

class CandidateListItem(BaseModel):
    id: str
    email: str
    full_name: Optional[str]
    ai_score: Optional[int]
    profile_completed: Optional[bool]
    created_at: datetime

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        from datetime import datetime, timedelta, timezone
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=algorithm)
    return encoded_jwt

def get_user_by_email(email: str):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        return user
    finally:
        cursor.close()
        connection.close()

def authenticate_user(email: str, password: str):
    user = get_user_by_email(email)
    if not user:
        return False
    if not verify_password(password, user['password_hash']):
        return False
    return user

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = get_user_by_email(email)
    if user is None:
        raise credentials_exception
    return user

async def get_profile_internal(user_id: str):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM candidate_profile WHERE user_id = %s", (user_id,))
        row = cursor.fetchone()
        return row
    finally:
        cursor.close()
        connection.close()

# Resume processing helper functions
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"
    
def basic_resume_parser(text: str) -> Dict[str, Any]:
    skills = []
    experience_years = 0
    education = []
    
    common_skills = [
        # Programming skills
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'swift',
        # Web development skills
        'html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask',
        # Database
        'mysql', 'postgresql', 'mongodb', 'sqlite', 'oracle',
        # Data science
        'numpy', 'pandas', 'scikit-learn', 'tensorflow', 'keras', 'pytorch',
        # Mobile development
        'flutter', 'react-native', 'swift', 'kotlin',
        # Other tools
        'docker', 'kubernetes', 'git', 'jenkins', 'aws', 'azure', 'google cloud'
    ]
    
    text_lower = text.lower()
    
    for skill in common_skills:
        if skill in text_lower:
            skills.append(skill.title())
    
    experience_patterns = [
        r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
        r'experience[:\s]*(\d+)\+?\s*years?',
        r'(\d+)\+?\s*yrs?\s+exp',
        r'(\d+)\+?\s*years?\s+(?:in|of|working)',
        r'over\s+(\d+)\s+years?',
        r'more than\s+(\d+)\s+years?'
    ]
    
    for pattern in experience_patterns:
        match = re.search(pattern, text_lower)
        if match:
            experience_years = max(experience_years, int(match.group(1)))
    
    education_keywords = [
        'bachelor', 'master', 'phd', 'degree', 'diploma', 'certification'
    ]
    for keyword in education_keywords:
        if keyword in text_lower:
            education.append(keyword.title())

    return {
        "skills": list(set(skills)),
        "experience_years": experience_years,
        "education": list(set(education)),
        "raw_text_length": len(text),
        "skills_count": len(set(skills))
    }

def calculate_profile_score(parsed_data: Dict[str, Any]) -> int:
    score = 0
    
    # Skills scoring (40 points max)
    skills_count = len(parsed_data.get("skills", []))
    if skills_count > 0:
        skills_score = min(skills_count * 8, 40)
        score += skills_score
        
    # Experience scoring (40 points max)
    experience = parsed_data.get("experience_years", 0)
    if experience > 0:
        if experience <= 2:
            exp_score = experience * 10
        elif experience <= 5:
            exp_score = 20 + (experience - 2) * 5
        elif experience <= 10:
            exp_score = 30 + (experience - 5) * 5
        else:
            exp_score = 40
        score += exp_score

    # Education scoring (15 points max)
    education_count = len(parsed_data.get("education", []))
    if education_count > 0:
        score += min(education_count * 7, 15)
        
    # Resume completeness scoring (10 points max)
    text_length = parsed_data.get("raw_text_length", 0)
    if text_length > 500:
        score += 10
    elif text_length > 200:
        score += 5
    
    return min(score, 100)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Human AI Recruitment Agent",
        "status": "healthy",
        "time": datetime.utcnow().isoformat(),
        "database": "MYSQL",
        "features": ["Authentication", "Resume Processing", "AI Scoring"]
    }
    
        
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token_expires = timedelta(minutes=access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user['email']},
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/login")
async def login_user(credentials: UserLogin):
    user = authenticate_user(credentials.email, credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token_expires = timedelta(minutes=access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user['email']},
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


# Add this to your auth_service/main.py after the existing login endpoint
@app.post("/register")
async def register_user(user: UserRegistration):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Check if email already exists
        cursor.execute("SELECT * FROM users WHERE email = %s", (user.email,))
        if cursor.fetchone():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        user_id = str(uuid.uuid4())
        hashed_password = get_password_hash(user.password)
        
        cursor.execute(
            "INSERT INTO users (id, email, password_hash, role) VALUES (%s, %s, %s, %s)",
            (user_id, user.email, hashed_password, user.role)
        )
        connection.commit()
        
        # Generate access token immediately
        access_token_expires = timedelta(minutes=access_token_expire_minutes)
        access_token = create_access_token(
            data={"sub": user.email},
            expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": user_id,
                "email": user.email,
                "role": user.role
            },
            "message": "Registration successful"
        }
        
    except mysql.connector.Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )
    finally:
        cursor.close()
        connection.close()




@app.get("/profile")
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    return {
        "user": {
            "id": current_user['id'],
            "email": current_user['email'],
            "role": current_user['role'],
            "created_at": current_user['created_at']
        },
        "message": "Profile retrieved successfully"
    }

# =============================================================================
# ✅ FIXED: PROPER CANDIDATE PROFILE ENDPOINTS
# =============================================================================

@app.get("/candidate/profile", response_model=CandidateProfileResponse)
async def get_own_profile(current_user: dict = Depends(get_current_user)):
    """Get current candidate's own profile"""
    if current_user["role"] != 'candidate':
        raise HTTPException(status_code=403, detail="Only candidates can access this endpoint")
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        # Get profile with latest resume info
        cursor.execute("""
            SELECT 
                cp.*,
                r.filename as latest_resume_filename,
                r.ai_analysis,
                r.upload_date as latest_resume_date
            FROM candidate_profile cp
            LEFT JOIN resumes r ON cp.user_id = r.candidate_id
            WHERE cp.user_id = %s
            ORDER BY r.upload_date DESC
            LIMIT 1
        """, (current_user["id"],))
        
        profile_data = cursor.fetchone()
        
        if not profile_data:
            # Create a basic profile if none exists
            profile_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO candidate_profile 
                (id, user_id, full_name, ai_score, profile_completed, created_at, updated_at)
                VALUES (%s, %s, %s, 0, FALSE, NOW(), NOW())
            """, (profile_id, current_user["id"], current_user["email"]))
            connection.commit()
            
            # Fetch the newly created profile
            cursor.execute("""
                SELECT 
                    cp.*,
                    r.filename as latest_resume_filename,
                    r.ai_analysis,
                    r.upload_date as latest_resume_date
                FROM candidate_profile cp
                LEFT JOIN resumes r ON cp.user_id = r.candidate_id
                WHERE cp.user_id = %s
                ORDER BY r.upload_date DESC
                LIMIT 1
            """, (current_user["id"],))
            profile_data = cursor.fetchone()
        
        return CandidateProfileResponse(
            id=profile_data['id'],
            user_id=profile_data['user_id'],
            full_name=profile_data['full_name'] or current_user["email"],
            phone=profile_data.get('phone'),
            location=profile_data.get('location'),
            experience_years=profile_data.get('experience_years', 0),
            skills=profile_data.get('skills'),
            resume_url=profile_data.get('resume_url'),
            linkedin_url=profile_data.get('linkedin_url'),
            profile_completed=bool(profile_data.get('profile_completed', False)),
            ai_score=profile_data.get('ai_score', 0),
            created_at=profile_data['created_at']
        )
    finally:
        cursor.close()
        connection.close()


@app.post("/candidate/profile", response_model=CandidateProfileResponse)
async def create_or_update_profile(
    profile: CandidateProfileCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create or update current candidate's profile (POST - with request body)"""
    if current_user["role"] != 'candidate':
        raise HTTPException(status_code=403, detail="Only candidates can access this endpoint")
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        cursor.execute("SELECT id FROM candidate_profile WHERE user_id=%s", (current_user["id"],))
        exists = cursor.fetchone()
        
        if exists:
            # Update existing profile
            cursor.execute("""
                UPDATE candidate_profile SET 
                full_name=%s, phone=%s, location=%s, experience_years=%s, 
                linkedin_url=%s, updated_at=NOW()
                WHERE user_id=%s
            """, (
                profile.full_name, profile.phone, profile.location,
                profile.experience_years, profile.linkedin_url, current_user["id"]
            ))
        else:
            # Create new profile
            profile_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO candidate_profile (
                    id, user_id, full_name, phone, location, 
                    experience_years, linkedin_url, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            """, (
                profile_id, current_user["id"], profile.full_name, 
                profile.phone, profile.location, profile.experience_years, profile.linkedin_url
            ))
        
        connection.commit()
    finally:
        cursor.close()
        connection.close()
    
    # Return updated profile
    profile_data = await get_profile_internal(current_user["id"])
    return CandidateProfileResponse(
        id=profile_data['id'],
        user_id=profile_data['user_id'],
        full_name=profile_data['full_name'],
        phone=profile_data.get('phone'),
        location=profile_data.get('location'),
        experience_years=profile_data.get('experience_years', 0),
        skills=profile_data.get('skills'),
        resume_url=profile_data.get('resume_url'),
        linkedin_url=profile_data.get('linkedin_url'),
        profile_completed=bool(profile_data.get('profile_completed')),
        ai_score=profile_data.get('ai_score', 0),
        created_at=profile_data['created_at']
    )

@app.get("/candidate/{candidate_id}/profile", response_model=CandidateProfileResponse)
async def get_candidate_profile(candidate_id: str, current_user: dict = Depends(get_current_user)):
    """Get any candidate's profile (recruiters/admins only)"""
    if current_user["role"] not in ("recruiter", "admin"):
        raise HTTPException(status_code=403, detail="Access forbidden")

    profile_data = await get_profile_internal(candidate_id)
    if not profile_data:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    return CandidateProfileResponse(
        id=profile_data['id'],
        user_id=profile_data['user_id'],
        full_name=profile_data['full_name'],
        phone=profile_data.get('phone'),
        location=profile_data.get('location'),
        experience_years=profile_data.get('experience_years', 0),
        skills=profile_data.get('skills'),
        resume_url=profile_data.get('resume_url'),
        linkedin_url=profile_data.get('linkedin_url'),
        profile_completed=bool(profile_data.get('profile_completed')),
        ai_score=profile_data.get('ai_score', 0),
        created_at=profile_data['created_at']
    )

# ✅ NEW: Candidates listing endpoint for recruiters
# @app.get("/candidates")
# async def list_candidates(current_user: dict = Depends(get_current_user)):
#     """List all candidates (recruiters/admins only)"""
#     if current_user["role"] not in ["recruiter", "admin"]:
#         raise HTTPException(status_code=403, detail="Access forbidden")
    
#     connection = get_db_connection()
#     cursor = connection.cursor(dictionary=True)
    
#     try:
#         cursor.execute("""
#             SELECT u.id, u.email, u.created_at, cp.full_name, cp.ai_score, cp.profile_completed
#             FROM users u
#             LEFT JOIN candidate_profile cp ON u.id = cp.user_id
#             WHERE u.role = 'candidate'
#             ORDER BY cp.ai_score DESC, u.created_at DESC
#         """)
#         candidates = cursor.fetchall()
        
#         # Convert to response format
#         candidate_list = []
#         for candidate in candidates:
#             candidate_list.append(CandidateListItem(
#                 id=candidate['id'],
#                 email=candidate['email'],
#                 full_name=candidate.get('full_name'),
#                 ai_score=candidate.get('ai_score'),
#                 profile_completed=bool(candidate.get('profile_completed', False)),
#                 created_at=candidate['created_at']
#             ))
        
#         return {"candidates": candidate_list, "total": len(candidate_list)}
#     finally:
#         cursor.close()
#         connection.close()


@app.get("/candidates")
async def list_candidates(current_user: dict = Depends(get_current_user)):
    """List all candidates with full details for recruiters"""
    if current_user["role"] not in ["recruiter", "admin"]:
        raise HTTPException(status_code=403, detail="Access forbidden")
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        cursor.execute("""
            SELECT 
                u.id, u.email, u.created_at as user_created_at,
                cp.full_name, cp.phone, cp.location, cp.experience_years,
                cp.skills, cp.resume_url, cp.linkedin_url, 
                cp.ai_score, cp.profile_completed,
                cp.created_at as profile_created_at,
                r.filename as resume_filename, r.file_size, r.upload_date
            FROM users u
            LEFT JOIN candidate_profile cp ON u.id = cp.user_id
            LEFT JOIN resumes r ON u.id = r.candidate_id
            WHERE u.role = 'candidate'
            ORDER BY cp.ai_score DESC, u.created_at DESC
        """)
        candidates = cursor.fetchall()
        
        # Convert to response format
        candidate_list = []
        for candidate in candidates:
            candidate_data = {
                "id": candidate['id'],
                "email": candidate['email'],
                "full_name": candidate.get('full_name'),
                "phone": candidate.get('phone'),
                "location": candidate.get('location'),
                "experience_years": candidate.get('experience_years', 0),
                "skills": candidate.get('skills', '').split(', ') if candidate.get('skills') else [],
                "resume_url": candidate.get('resume_url'),
                "resume_filename": candidate.get('resume_filename'),
                "file_size": candidate.get('file_size'),
                "linkedin_url": candidate.get('linkedin_url'),
                "ai_score": candidate.get('ai_score', 0),
                "profile_completed": bool(candidate.get('profile_completed', False)),
                "user_created_at": candidate['user_created_at'],
                "profile_created_at": candidate.get('profile_created_at'),
                "resume_upload_date": candidate.get('upload_date')
            }
            candidate_list.append(candidate_data)
        
        return {"candidates": candidate_list, "total": len(candidate_list)}
    finally:
        cursor.close()
        connection.close()

# Add endpoint to download/view resume
@app.get("/candidate/{candidate_id}/resume")
async def get_candidate_resume(candidate_id: str, current_user: dict = Depends(get_current_user)):
    """Get candidate's resume file for recruiters"""
    if current_user["role"] not in ["recruiter", "admin"]:
        raise HTTPException(status_code=403, detail="Access forbidden")
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        cursor.execute("""
            SELECT r.filename, r.file_path, r.content_text, r.ai_analysis
            FROM resumes r
            JOIN users u ON r.candidate_id = u.id
            WHERE u.id = %s AND u.role = 'candidate'
            ORDER BY r.upload_date DESC
            LIMIT 1
        """, (candidate_id,))
        
        resume = cursor.fetchone()
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        # Return resume info (file path can be used by frontend to download)
        return {
            "filename": resume['filename'],
            "file_path": resume['file_path'],
            "download_url": f"/uploads/resumes/{resume['filename']}",
            "ai_analysis": json.loads(resume['ai_analysis']) if resume['ai_analysis'] else None,
            "content_preview": resume['content_text'][:500] if resume['content_text'] else None
        }
    finally:
        cursor.close()
        connection.close()



# =============================================================================
# RESUME UPLOAD (UNCHANGED - ALREADY WORKING)
# =============================================================================

@app.post("/candidate/resume/upload", response_model=ResumeUploadResponse)
async def upload_resume(
    file: UploadFile = File(...), 
    current_user: dict = Depends(get_current_user)
):
    if current_user['role'] != 'candidate':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Only candidates can upload resumes."
        )
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Only PDF files are supported."
        )
    
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="File size exceeds 10MB limit."
        )
    
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    connection = None
    cursor = None
    
    try:
        # Save file to disk
        async with aiofiles.open(file_path, 'wb') as out_file:
            await out_file.write(content)
            
        # Process the resume
        extracted_text = extract_text_from_pdf(file_path)
        parsed_data = basic_resume_parser(extracted_text)
        ai_score = calculate_profile_score(parsed_data)
        
        # Save resume data to database
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        resume_id = str(uuid.uuid4())
        
        cursor.execute("""
            INSERT INTO resumes (
                id, candidate_id, filename, file_path, file_size, 
                content_text, parsed_skills, parsed_experience, 
                parsed_education, ai_analysis
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            resume_id, 
            current_user['id'], 
            filename, 
            file_path, 
            len(content),
            extracted_text[:5000], 
            json.dumps(parsed_data.get("skills", [])),
            json.dumps({"years": parsed_data.get("experience_years", 0)}),
            json.dumps(parsed_data.get("education", [])),
            json.dumps({"score": ai_score, "parsed_data": parsed_data})
        ))
        
        cursor.execute(
            "SELECT id FROM candidate_profile WHERE user_id = %s", 
            (current_user['id'],)
        )
        profile_exists = cursor.fetchone()
        
        if profile_exists:
            cursor.execute("""
                UPDATE candidate_profile 
                SET skills = %s, ai_score = %s, resume_url = %s, profile_completed = TRUE
                WHERE user_id = %s
            """, (
                ", ".join(parsed_data.get("skills", [])), 
                ai_score, 
                f"/uploads/resumes/{filename}", 
                current_user['id']
            ))
        else:
            profile_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO candidate_profile 
                (id, user_id, full_name, skills, ai_score, resume_url, profile_completed)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                profile_id, 
                current_user['id'], 
                "Candidate", 
                ", ".join(parsed_data.get("skills", [])), 
                ai_score, 
                f"/uploads/resumes/{filename}", 
                True
            ))
        
        connection.commit()
        
        return ResumeUploadResponse(
            id=resume_id,
            filename=filename,
            file_size=len(content),
            upload_date=datetime.utcnow(),
            parsed_data={
                **parsed_data,
                "ai_score": ai_score,
                "analysis_summary": f"Found {len(parsed_data.get('skills', []))} skills, {parsed_data.get('experience_years', 0)} years experience, {len(parsed_data.get('education', []))} education entries."
            }
        )
        
    except Exception as e:
        # Cleanup: Remove file if database operation failed
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500, 
            detail=f"Resume upload failed: {str(e)}"
        )
    finally:
        # Always cleanup database connections
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.get("/candidate/resumes")
async def get_candidate_resumes(current_user: dict = Depends(get_current_user)):
    """Get all resumes for current candidate"""
    if current_user["role"] != 'candidate':
        raise HTTPException(status_code=403, detail="Only candidates can access this endpoint")
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT 
                id, filename, file_size, upload_date,
                ai_analysis, content_text
            FROM resumes 
            WHERE candidate_id = %s 
            ORDER BY upload_date DESC
        """, (current_user["id"],))
        
        resumes = cursor.fetchall()
        
        # Parse AI analysis JSON
        for resume in resumes:
            if resume['ai_analysis']:
                try:
                    resume['ai_analysis'] = json.loads(resume['ai_analysis'])
                except:
                    resume['ai_analysis'] = None
        
        return {"resumes": resumes, "total": len(resumes)}
    finally:
        cursor.close()
        connection.close()



# =============================================================================
# OTHER ENDPOINTS (UNCHANGED)
# =============================================================================

@app.get("/health")
async def health_check():
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        db_status = "connected"
    except:
        db_status = "disconnected"
    finally:
        cursor.close()
        connection.close()
    return {
        "service": "Human AI Recruitment Agent",
        "status": "healthy",
        "time": datetime.utcnow().isoformat(),
        "database": db_status
    }

@app.get("/stats")
async def get_system_stats(current_user: dict = Depends(get_current_user)):
    """Get system usage statistics (admin/recruiter only)"""
    if current_user['role'] not in ['admin', 'recruiter']:
        raise HTTPException(status_code=403, detail="Access forbidden")
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        cursor.execute("SELECT role, COUNT(*) as count FROM users GROUP BY role")
        user_stats = cursor.fetchall()
        
        cursor.execute("SELECT COUNT(*) as total_resumes FROM resumes")
        resume_stats = cursor.fetchone()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_profiles, 
                SUM(profile_completed) as completed_profiles, 
                AVG(ai_score) as avg_ai_score
            FROM candidate_profile
        """)
        profile_stats = cursor.fetchone()
        
        return {
            "users": {role['role']: role['count'] for role in user_stats},
            "resumes_uploaded": resume_stats['total_resumes'] or 0,
            "profiles": {
                "total": profile_stats['total_profiles'] or 0,
                "completed": profile_stats['completed_profiles'] or 0,
                "average_ai_score": round(profile_stats['avg_ai_score'] or 0, 2)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
