# from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, BackgroundTasks
# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware 
# from pydantic import BaseModel, EmailStr
# from passlib.context import CryptContext
# from jose import JWTError, jwt
# from datetime import datetime, timedelta, timezone
# import mysql.connector
# from mysql.connector import Error
# from typing import Any, Dict, Optional, List
# import os
# import uuid
# from pathlib import Path 
# import PyPDF2
# import json
# import re  
# import aiofiles
# import smtplib
# import secrets
# import hashlib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

# from dotenv import load_dotenv
# import io
# import logging

# # Load environment variables
# load_dotenv()

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # FastAPI App Configuration
# app = FastAPI(
#     title="Human AI Recruitment Agent - Auth Service",
#     description="Complete authentication and user management service with password reset functionality",
#     version="2.0.0",
#     docs_url="/docs",
#     redoc_url="/redoc"
# )

# # CORS Configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:3000",
#         "http://127.0.0.1:3000",
#         "https://localhost:3000",
#         os.getenv("FRONTEND_URL", "http://localhost:3000")
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # File Upload Configuration
# UPLOAD_DIR = "uploads/resumes"
# Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
# app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# # Security Configuration
# secret_key = os.getenv("JWT_SECRET_KEY")
# algorithm = os.getenv("JWT_ALGORITHM", "HS256")
# access_token_expire_minutes = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))

# # Email Configuration
# SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
# SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
# SMTP_USERNAME = os.getenv("SMTP_USERNAME")
# SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
# FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# # Security instances
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# # =============================================================================
# # DATABASE CONNECTION
# # =============================================================================

# def get_db_connection():
#     """Get database connection with improved error handling"""
#     try:
#         connection = mysql.connector.connect(
#             host=os.getenv("MYSQL_HOST"),
#             user=os.getenv("MYSQL_USER"),
#             password=os.getenv("MYSQL_PASSWORD"),
#             database=os.getenv("MYSQL_DATABASE"),
#             port=int(os.getenv("MYSQL_PORT", "3306")),
#             autocommit=False,
#             pool_name="auth_pool",
#             pool_size=5
#         )
#         return connection
#     except Error as e:
#         logger.error(f"Database connection error: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Database connection failed"
#         )

# # =============================================================================
# # PYDANTIC MODELS
# # =============================================================================

# class UserRegistration(BaseModel):
#     email: EmailStr
#     password: str
#     role: Optional[str] = "candidate"
#     full_name: Optional[str] = None
    
# class UserLogin(BaseModel):
#     email: EmailStr
#     password: str
    
# class UserResponse(BaseModel):
#     id: str
#     email: str
#     role: str
#     full_name: Optional[str] = None
#     created_at: datetime
#     is_verified: bool = False

# class Token(BaseModel):
#     access_token: str
#     token_type: str
#     expires_in: int
#     user: UserResponse

# class CandidateProfileCreate(BaseModel):
#     full_name: str
#     phone: Optional[str] = None
#     location: Optional[str] = None
#     experience_years: Optional[int] = 0
#     linkedin_url: Optional[str] = None

# class CandidateProfileResponse(BaseModel):
#     id: str
#     user_id: str
#     full_name: str
#     phone: Optional[str] = None
#     location: Optional[str] = None
#     experience_years: Optional[int] = 0
#     skills: Optional[str]
#     resume_url: Optional[str]
#     linkedin_url: Optional[str]
#     profile_completed: bool
#     ai_score: int
#     created_at: datetime

# class ResumeUploadResponse(BaseModel):
#     id: str
#     filename: str
#     file_size: int
#     upload_date: datetime
#     parsed_data: Dict[str, Any]

# class CandidateListItem(BaseModel):
#     id: str
#     email: str
#     full_name: Optional[str]
#     ai_score: Optional[int]
#     profile_completed: Optional[bool]
#     created_at: datetime

# # Password Reset Models
# class ForgotPasswordRequest(BaseModel):
#     email: EmailStr

# class ResetPasswordRequest(BaseModel):
#     token: str
#     new_password: str

# class ChangePasswordRequest(BaseModel):
#     current_password: str
#     new_password: str

# class MessageResponse(BaseModel):
#     message: str
#     success: bool

# class EmailVerificationRequest(BaseModel):
#     email: EmailStr

# # =============================================================================
# # HELPER FUNCTIONS
# # =============================================================================

# def verify_password(plain_password: str, hashed_password: str) -> bool:
#     """Verify a password against its hash"""
#     return pwd_context.verify(plain_password, hashed_password)

# def get_password_hash(password: str) -> str:
#     """Hash a password"""
#     return pwd_context.hash(password)

# def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
#     """Create JWT access token"""
#     to_encode = data.copy()
#     if expires_delta:
#         expire = datetime.now(timezone.utc) + expires_delta
#     else:
#         expire = datetime.now(timezone.utc) + timedelta(minutes=access_token_expire_minutes)
    
#     to_encode.update({"exp": expire})
#     encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=algorithm)
#     return encoded_jwt

# def get_user_by_email(email: str):
#     """Get user by email address"""
#     connection = get_db_connection()
#     cursor = connection.cursor(dictionary=True)
    
#     try:
#         cursor.execute("""
#             SELECT u.*, cp.full_name 
#             FROM users u 
#             LEFT JOIN candidate_profile cp ON u.id = cp.user_id 
#             WHERE u.email = %s
#         """, (email,))
#         user = cursor.fetchone()
#         return user
#     except Error as e:
#         logger.error(f"Error fetching user: {e}")
#         return None
#     finally:
#         cursor.close()
#         connection.close()

# def get_user_by_id(user_id: str):
#     """Get user by ID"""
#     connection = get_db_connection()
#     cursor = connection.cursor(dictionary=True)
    
#     try:
#         cursor.execute("""
#             SELECT u.*, cp.full_name 
#             FROM users u 
#             LEFT JOIN candidate_profile cp ON u.id = cp.user_id 
#             WHERE u.id = %s
#         """, (user_id,))
#         user = cursor.fetchone()
#         return user
#     except Error as e:
#         logger.error(f"Error fetching user by ID: {e}")
#         return None
#     finally:
#         cursor.close()
#         connection.close()

# def authenticate_user(email: str, password: str):
#     """Authenticate user with email and password"""
#     user = get_user_by_email(email)
#     if not user:
#         return False
#     if not verify_password(password, user['password_hash']):
#         return False
#     return user

# async def get_current_user(token: str = Depends(oauth2_scheme)):
#     """Get current authenticated user"""
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
    
#     try:
#         payload = jwt.decode(token, secret_key, algorithms=[algorithm])
#         email: str = payload.get("sub")
#         if email is None:
#             raise credentials_exception
#     except JWTError:
#         raise credentials_exception

#     user = get_user_by_email(email)
#     if user is None:
#         raise credentials_exception
#     return user

# def validate_password_strength(password: str) -> tuple[bool, str]:
#     """Validate password strength"""
#     if len(password) < 8:
#         return False, "Password must be at least 8 characters long"
    
#     if not re.search(r"[A-Z]", password):
#         return False, "Password must contain at least one uppercase letter"
    
#     if not re.search(r"[a-z]", password):
#         return False, "Password must contain at least one lowercase letter"
    
#     if not re.search(r"\d", password):
#         return False, "Password must contain at least one number"
    
#     if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
#         return False, "Password must contain at least one special character"
    
#     return True, "Password is strong"

# def generate_reset_token() -> str:
#     """Generate secure reset token"""
#     return secrets.token_urlsafe(32)

# def generate_verification_token() -> str:
#     """Generate email verification token"""
#     return secrets.token_urlsafe(32)

# async def send_password_reset_email(email: str, reset_token: str, user_name: str = None):
#     """Send password reset email"""
#     if not SMTP_USERNAME or not SMTP_PASSWORD:
#         logger.warning("Email credentials not configured")
#         return False
        
#     reset_url = f"{FRONTEND_URL}/reset-password?token={reset_token}"
    
#     msg = MimeMultipart('alternative')
#     msg['Subject'] = "Password Reset Request - Human AI Recruitment Agent"
#     msg['From'] = SMTP_USERNAME
#     msg['To'] = email
    
#     # HTML content
#     html = f"""
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <style>
#             body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
#             .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
#             .header {{ background: #2563eb; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }}
#             .content {{ padding: 30px; background: #f8fafc; border-radius: 0 0 8px 8px; }}
#             .button {{ 
#                 display: inline-block; 
#                 padding: 12px 30px; 
#                 background: #2563eb; 
#                 color: white !important; 
#                 text-decoration: none; 
#                 border-radius: 5px;
#                 margin: 20px 0;
#                 font-weight: bold;
#             }}
#             .footer {{ padding: 20px; text-align: center; color: #666; font-size: 12px; }}
#             .warning {{ background: #fef3cd; border: 1px solid #fecaca; padding: 10px; border-radius: 4px; margin: 15px 0; }}
#         </style>
#     </head>
#     <body>
#         <div class="container">
#             <div class="header">
#                 <h1>🔐 Password Reset Request</h1>
#             </div>
#             <div class="content">
#                 <h2>Hello{f", {user_name}" if user_name else ""}!</h2>
#                 <p>We received a request to reset your password for your Human AI Recruitment Agent account.</p>
                
#                 <div class="warning">
#                     <strong>⚠️ Security Notice:</strong> If you didn't request this password reset, please ignore this email or contact our support team immediately.
#                 </div>
                
#                 <p>Click the button below to reset your password:</p>
#                 <div style="text-align: center;">
#                     <a href="{reset_url}" class="button">Reset My Password</a>
#                 </div>
                
#                 <p>Or copy and paste this link into your browser:</p>
#                 <p style="word-break: break-all; color: #2563eb; background: #e5e7eb; padding: 10px; border-radius: 4px;">{reset_url}</p>
                
#                 <div class="warning">
#                     <strong>⏰ Important:</strong> This link will expire in <strong>1 hour</strong> for security reasons.
#                 </div>
                
#                 <p>If you're having trouble with the link, please contact our support team.</p>
#             </div>
#             <div class="footer">
#                 <p>© 2025 Human AI Recruitment Agent. All rights reserved.</p>
#                 <p>This is an automated email, please do not reply directly to this message.</p>
#             </div>
#         </div>
#     </body>
#     </html>
#     """
    
#     # Plain text fallback
#     text = f"""
#     Password Reset Request - Human AI Recruitment Agent
    
#     Hello{f", {user_name}" if user_name else ""}!
    
#     We received a request to reset your password for your Human AI Recruitment Agent account.
    
#     Please click the following link to reset your password:
#     {reset_url}
    
#     IMPORTANT: This link will expire in 1 hour for security reasons.
    
#     If you didn't request this password reset, please ignore this email or contact support.
    
#     © 2025 Human AI Recruitment Agent. All rights reserved.
#     """
    
#     msg.attach(MimeText(text, 'plain'))
#     msg.attach(MimeText(html, 'html'))
    
#     try:
#         server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
#         server.starttls()
#         server.login(SMTP_USERNAME, SMTP_PASSWORD)
#         server.send_message(msg)
#         server.quit()
#         logger.info(f"Password reset email sent to {email}")
#         return True
#     except Exception as e:
#         logger.error(f"Email sending failed: {e}")
#         return False

# async def send_verification_email(email: str, verification_token: str, user_name: str = None):
#     """Send email verification email"""
#     if not SMTP_USERNAME or not SMTP_PASSWORD:
#         logger.warning("Email credentials not configured")
#         return False
        
#     verification_url = f"{FRONTEND_URL}/verify-email?token={verification_token}"
    
#     msg = MimeMultipart('alternative')
#     msg['Subject'] = "Email Verification - Human AI Recruitment Agent"
#     msg['From'] = SMTP_USERNAME
#     msg['To'] = email
    
#     # HTML content
#     html = f"""
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <style>
#             body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
#             .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
#             .header {{ background: #16a34a; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }}
#             .content {{ padding: 30px; background: #f8fafc; border-radius: 0 0 8px 8px; }}
#             .button {{ 
#                 display: inline-block; 
#                 padding: 12px 30px; 
#                 background: #16a34a; 
#                 color: white !important; 
#                 text-decoration: none; 
#                 border-radius: 5px;
#                 margin: 20px 0;
#                 font-weight: bold;
#             }}
#             .footer {{ padding: 20px; text-align: center; color: #666; font-size: 12px; }}
#         </style>
#     </head>
#     <body>
#         <div class="container">
#             <div class="header">
#                 <h1>✅ Welcome! Please Verify Your Email</h1>
#             </div>
#             <div class="content">
#                 <h2>Hello{f", {user_name}" if user_name else ""}!</h2>
#                 <p>Thank you for registering with Human AI Recruitment Agent!</p>
#                 <p>Please verify your email address to complete your registration and access all features.</p>
                
#                 <div style="text-align: center;">
#                     <a href="{verification_url}" class="button">Verify My Email</a>
#                 </div>
                
#                 <p>Or copy and paste this link into your browser:</p>
#                 <p style="word-break: break-all; color: #16a34a; background: #e5e7eb; padding: 10px; border-radius: 4px;">{verification_url}</p>
                
#                 <p>This verification link will expire in 24 hours for security reasons.</p>
#             </div>
#             <div class="footer">
#                 <p>© 2025 Human AI Recruitment Agent. All rights reserved.</p>
#                 <p>This is an automated email, please do not reply directly to this message.</p>
#             </div>
#         </div>
#     </body>
#     </html>
#     """
    
#     msg.attach(MimeText(html, 'html'))
    
#     try:
#         server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
#         server.starttls()
#         server.login(SMTP_USERNAME, SMTP_PASSWORD)
#         server.send_message(msg)
#         server.quit()
#         logger.info(f"Verification email sent to {email}")
#         return True
#     except Exception as e:
#         logger.error(f"Email sending failed: {e}")
#         return False

# # Resume processing helper functions
# def extract_text_from_pdf(file_path: str) -> str:
#     """Extract text from a PDF file."""
#     try:
#         with open(file_path, "rb") as file:
#             pdf_reader = PyPDF2.PdfReader(file)
#             text = ""
#             for page in pdf_reader.pages:
#                 text += page.extract_text()
#             return text
#     except Exception as e:
#         logger.error(f"Error extracting text from PDF: {e}")
#         return f"Error extracting text from PDF: {str(e)}"
    
# def basic_resume_parser(text: str) -> Dict[str, Any]:
#     """Parse resume text and extract key information"""
#     skills = []
#     experience_years = 0
#     education = []
    
#     common_skills = [
#         # Programming skills
#         'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'swift',
#         # Web development skills
#         'html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'fastapi',
#         # Database
#         'mysql', 'postgresql', 'mongodb', 'sqlite', 'oracle', 'redis',
#         # Data science
#         'numpy', 'pandas', 'scikit-learn', 'tensorflow', 'keras', 'pytorch',
#         # Mobile development
#         'flutter', 'react-native', 'swift', 'kotlin',
#         # Other tools
#         'docker', 'kubernetes', 'git', 'jenkins', 'aws', 'azure', 'google cloud'
#     ]
    
#     text_lower = text.lower()
    
#     for skill in common_skills:
#         if skill in text_lower:
#             skills.append(skill.title())
    
#     experience_patterns = [
#         r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
#         r'experience[:\s]*(\d+)\+?\s*years?',
#         r'(\d+)\+?\s*yrs?\s+exp',
#         r'(\d+)\+?\s*years?\s+(?:in|of|working)',
#         r'over\s+(\d+)\s+years?',
#         r'more than\s+(\d+)\s+years?'
#     ]
    
#     for pattern in experience_patterns:
#         match = re.search(pattern, text_lower)
#         if match:
#             experience_years = max(experience_years, int(match.group(1)))
    
#     education_keywords = [
#         'bachelor', 'master', 'phd', 'degree', 'diploma', 'certification'
#     ]
#     for keyword in education_keywords:
#         if keyword in text_lower:
#             education.append(keyword.title())

#     return {
#         "skills": list(set(skills)),
#         "experience_years": experience_years,
#         "education": list(set(education)),
#         "raw_text_length": len(text),
#         "skills_count": len(set(skills))
#     }

# def calculate_profile_score(parsed_data: Dict[str, Any]) -> int:
#     """Calculate profile quality score (0-100)"""
#     score = 0
    
#     # Skills scoring (40 points max)
#     skills_count = len(parsed_data.get("skills", []))
#     if skills_count > 0:
#         skills_score = min(skills_count * 8, 40)
#         score += skills_score
        
#     # Experience scoring (40 points max)
#     experience = parsed_data.get("experience_years", 0)
#     if experience > 0:
#         if experience <= 2:
#             exp_score = experience * 10
#         elif experience <= 5:
#             exp_score = 20 + (experience - 2) * 5
#         elif experience <= 10:
#             exp_score = 30 + (experience - 5) * 5
#         else:
#             exp_score = 40
#         score += exp_score

#     # Education scoring (15 points max)
#     education_count = len(parsed_data.get("education", []))
#     if education_count > 0:
#         score += min(education_count * 7, 15)
        
#     # Resume completeness scoring (10 points max)
#     text_length = parsed_data.get("raw_text_length", 0)
#     if text_length > 500:
#         score += 10
#     elif text_length > 200:
#         score += 5
    
#     return min(score, 100)

# async def get_profile_internal(user_id: str):
#     """Internal function to get candidate profile"""
#     connection = get_db_connection()
#     cursor = connection.cursor(dictionary=True)
#     try:
#         cursor.execute("SELECT * FROM candidate_profile WHERE user_id = %s", (user_id,))
#         row = cursor.fetchone()
#         return row
#     finally:
#         cursor.close()
#         connection.close()

# # =============================================================================
# # API ENDPOINTS
# # =============================================================================

# @app.get("/")
# async def root():
#     """Health check and service information endpoint"""
#     return {
#         "service": "Human AI Recruitment Agent - Auth Service",
#         "status": "healthy",
#         "time": datetime.utcnow().isoformat(),
#         "database": "MySQL",
#         "version": "2.0.0",
#         "features": [
#             "Authentication & Authorization",
#             "Password Reset",
#             "Email Verification", 
#             "Resume Processing",
#             "AI Scoring",
#             "Profile Management"
#         ]
#     }

# @app.get("/health")
# async def health_check():
#     """Detailed health check endpoint"""
#     connection = get_db_connection()
#     try:
#         cursor = connection.cursor()
#         cursor.execute("SELECT 1")
#         cursor.fetchone()
#         db_status = "connected"
        
#         # Check if required tables exist
#         cursor.execute("SHOW TABLES LIKE 'users'")
#         users_table = cursor.fetchone()
        
#         cursor.execute("SHOW TABLES LIKE 'password_reset_tokens'")
#         reset_table = cursor.fetchone()
        
#         tables_status = {
#             "users_table": "exists" if users_table else "missing",
#             "password_reset_tokens_table": "exists" if reset_table else "missing"
#         }
        
#     except Exception as e:
#         db_status = f"error: {str(e)}"
#         tables_status = {"error": str(e)}
#     finally:
#         cursor.close()
#         connection.close()
    
#     return {
#         "service": "Human AI Recruitment Agent - Auth Service",
#         "status": "healthy",
#         "time": datetime.utcnow().isoformat(),
#         "database": db_status,
#         "tables": tables_status,
#         "email_configured": bool(SMTP_USERNAME and SMTP_PASSWORD)
#     }

# # =============================================================================
# # AUTHENTICATION ENDPOINTS
# # =============================================================================

# @app.post("/register", response_model=Token)
# async def register_user(user: UserRegistration, background_tasks: BackgroundTasks):
#     """Register a new user with email verification"""
    
#     # Validate password strength
#     is_strong, message = validate_password_strength(user.password)
#     if not is_strong:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail=message
#         )
    
#     connection = get_db_connection()
#     cursor = connection.cursor(dictionary=True)
    
#     try:
#         # Check if email already exists
#         cursor.execute("SELECT * FROM users WHERE email = %s", (user.email,))
#         if cursor.fetchone():
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Email already registered"
#             )
        
#         # Create new user
#         user_id = str(uuid.uuid4())
#         hashed_password = get_password_hash(user.password)
#         verification_token = generate_verification_token()
        
#         cursor.execute("""
#             INSERT INTO users (id, email, password_hash, role, is_verified, email_verification_token) 
#             VALUES (%s, %s, %s, %s, %s, %s)
#         """, (user_id, user.email, hashed_password, user.role, False, verification_token))
        
#         # Create profile for candidates
#         if user.role == "candidate" and user.full_name:
#             profile_id = str(uuid.uuid4())
#             cursor.execute("""
#                 INSERT INTO candidate_profile 
#                 (id, user_id, full_name, ai_score, profile_completed, created_at, updated_at)
#                 VALUES (%s, %s, %s, 0, FALSE, NOW(), NOW())
#             """, (profile_id, user_id, user.full_name))
        
#         connection.commit()
        
#         # Send verification email in background
#         background_tasks.add_task(
#             send_verification_email,
#             user.email,
#             verification_token,
#             user.full_name or user.email
#         )
        
#         # Generate access token
#         access_token_expires = timedelta(minutes=access_token_expire_minutes)
#         access_token = create_access_token(
#             data={"sub": user.email},
#             expires_delta=access_token_expires
#         )
        
#         user_response = UserResponse(
#             id=user_id,
#             email=user.email,
#             role=user.role,
#             full_name=user.full_name,
#             created_at=datetime.utcnow(),
#             is_verified=False
#         )
        
#         return Token(
#             access_token=access_token,
#             token_type="bearer",
#             expires_in=access_token_expire_minutes * 60,
#             user=user_response
#         )
        
#     except mysql.connector.Error as e:
#         connection.rollback()
#         logger.error(f"Registration error: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Registration failed: {str(e)}"
#         )
#     finally:
#         cursor.close()
#         connection.close()

# @app.post("/login", response_model=Token)
# async def login_user(credentials: UserLogin):
#     """User login endpoint"""
#     user = authenticate_user(credentials.email, credentials.password)
#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect email or password",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
    
#     # Update last login time
#     connection = get_db_connection()
#     cursor = connection.cursor()
#     try:
#         cursor.execute(
#             "UPDATE users SET last_login = NOW(), updated_at = NOW() WHERE id = %s",
#             (user['id'],)
#         )
#         connection.commit()
#     except Exception as e:
#         logger.error(f"Error updating last login: {e}")
#     finally:
#         cursor.close()
#         connection.close()
        
#     access_token_expires = timedelta(minutes=access_token_expire_minutes)
#     access_token = create_access_token(
#         data={"sub": user['email']},
#         expires_delta=access_token_expires
#     )
    
#     user_response = UserResponse(
#         id=user['id'],
#         email=user['email'],
#         role=user['role'],
#         full_name=user.get('full_name'),
#         created_at=user['created_at'],
#         is_verified=user.get('is_verified', False)
#     )
    
#     return Token(
#         access_token=access_token,
#         token_type="bearer",
#         expires_in=access_token_expire_minutes * 60,
#         user=user_response
#     )

# @app.post("/token", response_model=Token)
# async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
#     """OAuth2 compatible token endpoint"""
#     user = authenticate_user(form_data.username, form_data.password)
#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect email or password",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
        
#     access_token_expires = timedelta(minutes=access_token_expire_minutes)
#     access_token = create_access_token(
#         data={"sub": user['email']},
#         expires_delta=access_token_expires
#     )
    
#     user_response = UserResponse(
#         id=user['id'],
#         email=user['email'],
#         role=user['role'],
#         full_name=user.get('full_name'),
#         created_at=user['created_at'],
#         is_verified=user.get('is_verified', False)
#     )
    
#     return Token(
#         access_token=access_token,
#         token_type="bearer",
#         expires_in=access_token_expire_minutes * 60,
#         user=user_response
#     )

# @app.get("/profile", response_model=UserResponse)
# async def get_user_profile(current_user: dict = Depends(get_current_user)):
#     """Get current user profile"""
#     return UserResponse(
#         id=current_user['id'],
#         email=current_user['email'],
#         role=current_user['role'],
#         full_name=current_user.get('full_name'),
#         created_at=current_user['created_at'],
#         is_verified=current_user.get('is_verified', False)
#     )

# # =============================================================================
# # PASSWORD RESET ENDPOINTS
# # =============================================================================

# @app.post("/forgot-password", response_model=MessageResponse)
# async def forgot_password(request: ForgotPasswordRequest, background_tasks: BackgroundTasks):
#     """Send password reset email"""
#     connection = get_db_connection()
#     cursor = connection.cursor(dictionary=True)
    
#     try:
#         # Check if user exists
#         cursor.execute("SELECT id, email, full_name FROM users WHERE email = %s", (request.email,))
#         user = cursor.fetchone()
        
#         # Always return success for security (don't reveal if email exists)
#         response_message = MessageResponse(
#             message="If an account with that email exists, a password reset link has been sent.",
#             success=True
#         )
        
#         if not user:
#             return response_message
        
#         # Generate reset token
#         reset_token = generate_reset_token()
#         token_hash = hashlib.sha256(reset_token.encode()).hexdigest()
        
#         # Clean up old tokens for this user
#         cursor.execute("DELETE FROM password_reset_tokens WHERE user_id = %s", (user["id"],))
        
#         # Store reset token (expires in 1 hour)
#         token_id = str(uuid.uuid4())
#         expires_at = datetime.utcnow() + timedelta(hours=1)
        
#         cursor.execute("""
#             INSERT INTO password_reset_tokens 
#             (id, user_id, token_hash, expires_at) 
#             VALUES (%s, %s, %s, %s)
#         """, (token_id, user["id"], token_hash, expires_at))
        
#         connection.commit()
        
#         # Send email in background
#         background_tasks.add_task(
#             send_password_reset_email, 
#             request.email, 
#             reset_token,
#             user.get("full_name") or user.get("email")
#         )
        
#         return response_message
        
#     except Exception as e:
#         logger.error(f"Forgot password error: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to process password reset request"
#         )
#     finally:
#         cursor.close()
#         connection.close()

# @app.post("/reset-password", response_model=MessageResponse)
# async def reset_password(request: ResetPasswordRequest):
#     """Reset password using token"""
#     # Validate password strength
#     is_strong, message = validate_password_strength(request.new_password)
#     if not is_strong:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail=message
#         )
    
#     connection = get_db_connection()
#     cursor = connection.cursor(dictionary=True)
    
#     try:
#         # Hash the provided token
#         token_hash = hashlib.sha256(request.token.encode()).hexdigest()
        
#         # Find valid reset token
#         cursor.execute("""
#             SELECT prt.*, u.email 
#             FROM password_reset_tokens prt
#             JOIN users u ON prt.user_id = u.id
#             WHERE prt.token_hash = %s AND prt.used = FALSE AND prt.expires_at > NOW()
#         """, (token_hash,))
        
#         token_data = cursor.fetchone()
        
#         if not token_data:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Invalid or expired reset token"
#             )
        
#         # Hash new password
#         new_password_hash = get_password_hash(request.new_password)
        
#         # Update user password
#         cursor.execute("""
#             UPDATE users 
#             SET password_hash = %s, updated_at = NOW() 
#             WHERE id = %s
#         """, (new_password_hash, token_data["user_id"]))
        
#         # Mark token as used
#         cursor.execute("""
#             UPDATE password_reset_tokens 
#             SET used = TRUE, used_at = NOW() 
#             WHERE id = %s
#         """, (token_data["id"],))
        
#         connection.commit()
        
#         return MessageResponse(
#             message="Password has been successfully reset. You can now login with your new password.",
#             success=True
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Reset password error: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to reset password"
#         )
#     finally:
#         cursor.close()
#         connection.close()

# @app.post("/change-password", response_model=MessageResponse)
# async def change_password(request: ChangePasswordRequest, current_user: dict = Depends(get_current_user)):
#     """Change password for authenticated user"""
#     # Validate new password strength
#     is_strong, message = validate_password_strength(request.new_password)
#     if not is_strong:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail=message
#         )
    
#     # Verify current password
#     if not verify_password(request.current_password, current_user["password_hash"]):
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Current password is incorrect"
#         )
    
#     # Check if new password is different from current
#     if verify_password(request.new_password, current_user["password_hash"]):
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="New password must be different from current password"
#         )
    
#     connection = get_db_connection()
#     cursor = connection.cursor()
    
#     try:
#         # Hash new password
#         new_password_hash = get_password_hash(request.new_password)
        
#         # Update password in database
#         cursor.execute("""
#             UPDATE users 
#             SET password_hash = %s, updated_at = NOW() 
#             WHERE id = %s
#         """, (new_password_hash, current_user["id"]))
        
#         connection.commit()
        
#         return MessageResponse(
#             message="Password changed successfully",
#             success=True
#         )
        
#     except Exception as e:
#         logger.error(f"Change password error: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to change password"
#         )
#     finally:
#         cursor.close()
#         connection.close()

# @app.get("/verify-reset-token")
# async def verify_reset_token(token: str):
#     """Verify if a reset token is valid (for frontend validation)"""
#     connection = get_db_connection()
#     cursor = connection.cursor(dictionary=True)
    
#     try:
#         token_hash = hashlib.sha256(token.encode()).hexdigest()
        
#         cursor.execute("""
#             SELECT prt.expires_at, u.email
#             FROM password_reset_tokens prt
#             JOIN users u ON prt.user_id = u.id
#             WHERE prt.token_hash = %s AND prt.used = FALSE
#         """, (token_hash,))
        
#         token_data = cursor.fetchone()
        
#         if not token_data:
#             return {"valid": False, "message": "Invalid token"}
        
#         # Check if token has expired
#         if datetime.utcnow() > token_data["expires_at"]:
#             return {"valid": False, "message": "Token has expired"}
        
#         return {
#             "valid": True, 
#             "message": "Token is valid",
#             "email": token_data["email"]  # Can show email on reset form
#         }
        
#     except Exception as e:
#         logger.error(f"Verify token error: {e}")
#         return {"valid": False, "message": "Error verifying token"}
#     finally:
#         cursor.close()
#         connection.close()

# # =============================================================================
# # EMAIL VERIFICATION ENDPOINTS
# # =============================================================================

# @app.post("/verify-email", response_model=MessageResponse)
# async def verify_email(token: str):
#     """Verify email address using token"""
#     connection = get_db_connection()
#     cursor = connection.cursor(dictionary=True)
    
#     try:
#         # Find user with verification token
#         cursor.execute("""
#             SELECT id, email, is_verified 
#             FROM users 
#             WHERE email_verification_token = %s
#         """, (token,))
        
#         user = cursor.fetchone()
        
#         if not user:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Invalid verification token"
#             )
        
#         if user['is_verified']:
#             return MessageResponse(
#                 message="Email is already verified",
#                 success=True
#             )
        
#         # Update user as verified
#         cursor.execute("""
#             UPDATE users 
#             SET is_verified = TRUE, email_verification_token = NULL, updated_at = NOW() 
#             WHERE id = %s
#         """, (user["id"],))
        
#         connection.commit()
        
#         return MessageResponse(
#             message="Email verified successfully",
#             success=True
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Email verification error: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to verify email"
#         )
#     finally:
#         cursor.close()
#         connection.close()

# @app.post("/resend-verification", response_model=MessageResponse)
# async def resend_verification_email(
#     request: EmailVerificationRequest, 
#     background_tasks: BackgroundTasks
# ):
#     """Resend email verification"""
#     connection = get_db_connection()
#     cursor = connection.cursor(dictionary=True)
    
#     try:
#         # Check if user exists and is not verified
#         cursor.execute("""
#             SELECT id, email, full_name, is_verified 
#             FROM users 
#             WHERE email = %s
#         """, (request.email,))
        
#         user = cursor.fetchone()
        
#         if not user:
#             return MessageResponse(
#                 message="If the email exists, a verification link has been sent.",
#                 success=True
#             )
        
#         if user['is_verified']:
#             return MessageResponse(
#                 message="Email is already verified",
#                 success=True
#             )
        
#         # Generate new verification token
#         verification_token = generate_verification_token()
        
#         cursor.execute("""
#             UPDATE users 
#             SET email_verification_token = %s, updated_at = NOW() 
#             WHERE id = %s
#         """, (verification_token, user["id"]))
        
#         connection.commit()
        
#         # Send verification email in background
#         background_tasks.add_task(
#             send_verification_email,
#             request.email,
#             verification_token,
#             user.get("full_name") or request.email
#         )
        
#         return MessageResponse(
#             message="Verification email sent successfully",
#             success=True
#         )
        
#     except Exception as e:
#         logger.error(f"Resend verification error: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to resend verification email"
#         )
#     finally:
#         cursor.close()
#         connection.close()

# # =============================================================================
# # CANDIDATE PROFILE ENDPOINTS
# # =============================================================================

# @app.get("/candidate/profile", response_model=CandidateProfileResponse)
# async def get_own_profile(current_user: dict = Depends(get_current_user)):
#     """Get current candidate's own profile"""
#     if current_user["role"] != 'candidate':
#         raise HTTPException(status_code=403, detail="Only candidates can access this endpoint")
    
#     connection = get_db_connection()
#     cursor = connection.cursor(dictionary=True)
#     try:
#         # Get profile with latest resume info
#         cursor.execute("""
#             SELECT 
#                 cp.*,
#                 r.filename as latest_resume_filename,
#                 r.ai_analysis,
#                 r.upload_date as latest_resume_date
#             FROM candidate_profile cp
#             LEFT JOIN resumes r ON cp.user_id = r.candidate_id
#             WHERE cp.user_id = %s
#             ORDER BY r.upload_date DESC
#             LIMIT 1
#         """, (current_user["id"],))
        
#         profile_data = cursor.fetchone()
        
#         if not profile_data:
#             # Create a basic profile if none exists
#             profile_id = str(uuid.uuid4())
#             cursor.execute("""
#                 INSERT INTO candidate_profile 
#                 (id, user_id, full_name, ai_score, profile_completed, created_at, updated_at)
#                 VALUES (%s, %s, %s, 0, FALSE, NOW(), NOW())
#             """, (profile_id, current_user["id"], current_user.get("full_name") or current_user["email"]))
#             connection.commit()
            
#             # Fetch the newly created profile
#             cursor.execute("""
#                 SELECT 
#                     cp.*,
#                     r.filename as latest_resume_filename,
#                     r.ai_analysis,
#                     r.upload_date as latest_resume_date
#                 FROM candidate_profile cp
#                 LEFT JOIN resumes r ON cp.user_id = r.candidate_id
#                 WHERE cp.user_id = %s
#                 ORDER BY r.upload_date DESC
#                 LIMIT 1
#             """, (current_user["id"],))
#             profile_data = cursor.fetchone()
        
#         return CandidateProfileResponse(
#             id=profile_data['id'],
#             user_id=profile_data['user_id'],
#             full_name=profile_data['full_name'] or current_user.get("full_name") or current_user["email"],
#             phone=profile_data.get('phone'),
#             location=profile_data.get('location'),
#             experience_years=profile_data.get('experience_years', 0),
#             skills=profile_data.get('skills'),
#             resume_url=profile_data.get('resume_url'),
#             linkedin_url=profile_data.get('linkedin_url'),
#             profile_completed=bool(profile_data.get('profile_completed', False)),
#             ai_score=profile_data.get('ai_score', 0),
#             created_at=profile_data['created_at']
#         )
#     finally:
#         cursor.close()
#         connection.close()

# @app.post("/candidate/profile", response_model=CandidateProfileResponse)
# async def create_or_update_profile(
#     profile: CandidateProfileCreate,
#     current_user: dict = Depends(get_current_user)
# ):
#     """Create or update current candidate's profile"""
#     if current_user["role"] != 'candidate':
#         raise HTTPException(status_code=403, detail="Only candidates can access this endpoint")
    
#     connection = get_db_connection()
#     cursor = connection.cursor(dictionary=True)
    
#     try:
#         cursor.execute("SELECT id FROM candidate_profile WHERE user_id=%s", (current_user["id"],))
#         exists = cursor.fetchone()
        
#         if exists:
#             # Update existing profile
#             cursor.execute("""
#                 UPDATE candidate_profile SET 
#                 full_name=%s, phone=%s, location=%s, experience_years=%s, 
#                 linkedin_url=%s, updated_at=NOW()
#                 WHERE user_id=%s
#             """, (
#                 profile.full_name, profile.phone, profile.location,
#                 profile.experience_years, profile.linkedin_url, current_user["id"]
#             ))
#         else:
#             # Create new profile
#             profile_id = str(uuid.uuid4())
#             cursor.execute("""
#                 INSERT INTO candidate_profile (
#                     id, user_id, full_name, phone, location, 
#                     experience_years, linkedin_url, created_at, updated_at
#                 ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
#             """, (
#                 profile_id, current_user["id"], profile.full_name, 
#                 profile.phone, profile.location, profile.experience_years, profile.linkedin_url
#             ))
        
#         connection.commit()
#     finally:
#         cursor.close()
#         connection.close()
    
#     # Return updated profile
#     profile_data = await get_profile_internal(current_user["id"])
#     return CandidateProfileResponse(
#         id=profile_data['id'],
#         user_id=profile_data['user_id'],
#         full_name=profile_data['full_name'],
#         phone=profile_data.get('phone'),
#         location=profile_data.get('location'),
#         experience_years=profile_data.get('experience_years', 0),
#         skills=profile_data.get('skills'),
#         resume_url=profile_data.get('resume_url'),
#         linkedin_url=profile_data.get('linkedin_url'),
#         profile_completed=bool(profile_data.get('profile_completed')),
#         ai_score=profile_data.get('ai_score', 0),
#         created_at=profile_data['created_at']
#     )

# # =============================================================================
# # RESUME UPLOAD ENDPOINTS
# # =============================================================================

# @app.post("/candidate/resume/upload", response_model=ResumeUploadResponse)
# async def upload_resume(
#     file: UploadFile = File(...), 
#     current_user: dict = Depends(get_current_user)
# ):
#     """Upload and process candidate resume"""
#     if current_user['role'] != 'candidate':
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN, 
#             detail="Only candidates can upload resumes."
#         )
    
#     if not file.filename.lower().endswith('.pdf'):
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST, 
#             detail="Only PDF files are supported."
#         )
    
#     content = await file.read()
#     if len(content) > 10 * 1024 * 1024:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST, 
#             detail="File size exceeds 10MB limit."
#         )
    
#     file_id = str(uuid.uuid4())
#     filename = f"{file_id}_{file.filename}"
#     file_path = os.path.join(UPLOAD_DIR, filename)
    
#     connection = None
#     cursor = None
    
#     try:
#         # Save file to disk
#         async with aiofiles.open(file_path, 'wb') as out_file:
#             await out_file.write(content)
            
#         # Process the resume
#         extracted_text = extract_text_from_pdf(file_path)
#         parsed_data = basic_resume_parser(extracted_text)
#         ai_score = calculate_profile_score(parsed_data)
        
#         # Save resume data to database
#         connection = get_db_connection()
#         cursor = connection.cursor(dictionary=True)
#         resume_id = str(uuid.uuid4())
        
#         cursor.execute("""
#             INSERT INTO resumes (
#                 id, candidate_id, filename, file_path, file_size, 
#                 content_text, parsed_skills, parsed_experience, 
#                 parsed_education, ai_analysis
#             ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#         """, (
#             resume_id, 
#             current_user['id'], 
#             filename, 
#             file_path, 
#             len(content),
#             extracted_text[:5000], 
#             json.dumps(parsed_data.get("skills", [])),
#             json.dumps({"years": parsed_data.get("experience_years", 0)}),
#             json.dumps(parsed_data.get("education", [])),
#             json.dumps({"score": ai_score, "parsed_data": parsed_data})
#         ))
        
#         cursor.execute(
#             "SELECT id FROM candidate_profile WHERE user_id = %s", 
#             (current_user['id'],)
#         )
#         profile_exists = cursor.fetchone()
        
#         if profile_exists:
#             cursor.execute("""
#                 UPDATE candidate_profile 
#                 SET skills = %s, ai_score = %s, resume_url = %s, profile_completed = TRUE, updated_at = NOW()
#                 WHERE user_id = %s
#             """, (
#                 ", ".join(parsed_data.get("skills", [])), 
#                 ai_score, 
#                 f"/uploads/resumes/{filename}", 
#                 current_user['id']
#             ))
#         else:
#             profile_id = str(uuid.uuid4())
#             cursor.execute("""
#                 INSERT INTO candidate_profile 
#                 (id, user_id, full_name, skills, ai_score, resume_url, profile_completed, created_at, updated_at)
#                 VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
#             """, (
#                 profile_id, 
#                 current_user['id'], 
#                 current_user.get('full_name') or "Candidate", 
#                 ", ".join(parsed_data.get("skills", [])), 
#                 ai_score, 
#                 f"/uploads/resumes/{filename}", 
#                 True
#             ))
        
#         connection.commit()
        
#         return ResumeUploadResponse(
#             id=resume_id,
#             filename=filename,
#             file_size=len(content),
#             upload_date=datetime.utcnow(),
#             parsed_data={
#                 **parsed_data,
#                 "ai_score": ai_score,
#                 "analysis_summary": f"Found {len(parsed_data.get('skills', []))} skills, {parsed_data.get('experience_years', 0)} years experience, {len(parsed_data.get('education', []))} education entries."
#             }
#         )
        
#     except Exception as e:
#         # Cleanup: Remove file if database operation failed
#         if os.path.exists(file_path):
#             os.remove(file_path)
#         logger.error(f"Resume upload error: {e}")
#         raise HTTPException(
#             status_code=500, 
#             detail=f"Resume upload failed: {str(e)}"
#         )
#     finally:
#         # Always cleanup database connections
#         if cursor:
#             cursor.close()
#         if connection:
#             connection.close()

# @app.get("/candidate/resumes")
# async def get_candidate_resumes(current_user: dict = Depends(get_current_user)):
#     """Get all resumes for current candidate"""
#     if current_user["role"] != 'candidate':
#         raise HTTPException(status_code=403, detail="Only candidates can access this endpoint")
    
#     connection = get_db_connection()
#     cursor = connection.cursor(dictionary=True)
#     try:
#         cursor.execute("""
#             SELECT 
#                 id, filename, file_size, upload_date,
#                 ai_analysis, content_text
#             FROM resumes 
#             WHERE candidate_id = %s 
#             ORDER BY upload_date DESC
#         """, (current_user["id"],))
        
#         resumes = cursor.fetchall()
        
#         # Parse AI analysis JSON
#         for resume in resumes:
#             if resume['ai_analysis']:
#                 try:
#                     resume['ai_analysis'] = json.loads(resume['ai_analysis'])
#                 except:
#                     resume['ai_analysis'] = None
        
#         return {"resumes": resumes, "total": len(resumes)}
#     finally:
#         cursor.close()
#         connection.close()

# # =============================================================================
# # RECRUITER ENDPOINTS
# # =============================================================================

# @app.get("/candidates")
# async def list_candidates(current_user: dict = Depends(get_current_user)):
#     """List all candidates with full details for recruiters"""
#     if current_user["role"] not in ["recruiter", "admin"]:
#         raise HTTPException(status_code=403, detail="Access forbidden")
    
#     connection = get_db_connection()
#     cursor = connection.cursor(dictionary=True)
    
#     try:
#         cursor.execute("""
#             SELECT 
#                 u.id, u.email, u.created_at as user_created_at, u.is_verified,
#                 cp.full_name, cp.phone, cp.location, cp.experience_years,
#                 cp.skills, cp.resume_url, cp.linkedin_url, 
#                 cp.ai_score, cp.profile_completed,
#                 cp.created_at as profile_created_at,
#                 r.filename as resume_filename, r.file_size, r.upload_date
#             FROM users u
#             LEFT JOIN candidate_profile cp ON u.id = cp.user_id
#             LEFT JOIN resumes r ON u.id = r.candidate_id
#             WHERE u.role = 'candidate'
#             ORDER BY cp.ai_score DESC, u.created_at DESC
#         """)
#         candidates = cursor.fetchall()
        
#         # Convert to response format
#         candidate_list = []
#         for candidate in candidates:
#             candidate_data = {
#                 "id": candidate['id'],
#                 "email": candidate['email'],
#                 "full_name": candidate.get('full_name'),
#                 "phone": candidate.get('phone'),
#                 "location": candidate.get('location'),
#                 "experience_years": candidate.get('experience_years', 0),
#                 "skills": candidate.get('skills', '').split(', ') if candidate.get('skills') else [],
#                 "resume_url": candidate.get('resume_url'),
#                 "resume_filename": candidate.get('resume_filename'),
#                 "file_size": candidate.get('file_size'),
#                 "linkedin_url": candidate.get('linkedin_url'),
#                 "ai_score": candidate.get('ai_score', 0),
#                 "profile_completed": bool(candidate.get('profile_completed', False)),
#                 "is_verified": bool(candidate.get('is_verified', False)),
#                 "user_created_at": candidate['user_created_at'],
#                 "profile_created_at": candidate.get('profile_created_at'),
#                 "resume_upload_date": candidate.get('upload_date')
#             }
#             candidate_list.append(candidate_data)
        
#         return {"candidates": candidate_list, "total": len(candidate_list)}
#     finally:
#         cursor.close()
#         connection.close()

# @app.get("/candidate/{candidate_id}/profile", response_model=CandidateProfileResponse)
# async def get_candidate_profile(candidate_id: str, current_user: dict = Depends(get_current_user)):
#     """Get any candidate's profile (recruiters/admins only)"""
#     if current_user["role"] not in ("recruiter", "admin"):
#         raise HTTPException(status_code=403, detail="Access forbidden")

#     profile_data = await get_profile_internal(candidate_id)
#     if not profile_data:
#         raise HTTPException(status_code=404, detail="Profile not found")
    
#     return CandidateProfileResponse(
#         id=profile_data['id'],
#         user_id=profile_data['user_id'],
#         full_name=profile_data['full_name'],
#         phone=profile_data.get('phone'),
#         location=profile_data.get('location'),
#         experience_years=profile_data.get('experience_years', 0),
#         skills=profile_data.get('skills'),
#         resume_url=profile_data.get('resume_url'),
#         linkedin_url=profile_data.get('linkedin_url'),
#         profile_completed=bool(profile_data.get('profile_completed')),
#         ai_score=profile_data.get('ai_score', 0),
#         created_at=profile_data['created_at']
#     )

# @app.get("/candidate/{candidate_id}/resume")
# async def get_candidate_resume(candidate_id: str, current_user: dict = Depends(get_current_user)):
#     """Get candidate's resume file for recruiters"""
#     if current_user["role"] not in ["recruiter", "admin"]:
#         raise HTTPException(status_code=403, detail="Access forbidden")
    
#     connection = get_db_connection()
#     cursor = connection.cursor(dictionary=True)
    
#     try:
#         cursor.execute("""
#             SELECT r.filename, r.file_path, r.content_text, r.ai_analysis
#             FROM resumes r
#             JOIN users u ON r.candidate_id = u.id
#             WHERE u.id = %s AND u.role = 'candidate'
#             ORDER BY r.upload_date DESC
#             LIMIT 1
#         """, (candidate_id,))
        
#         resume = cursor.fetchone()
#         if not resume:
#             raise HTTPException(status_code=404, detail="Resume not found")
        
#         # Return resume info (file path can be used by frontend to download)
#         return {
#             "filename": resume['filename'],
#             "file_path": resume['file_path'],
#             "download_url": f"/uploads/resumes/{resume['filename']}",
#             "ai_analysis": json.loads(resume['ai_analysis']) if resume['ai_analysis'] else None,
#             "content_preview": resume['content_text'][:500] if resume['content_text'] else None
#         }
#     finally:
#         cursor.close()
#         connection.close()

# # =============================================================================
# # ADMIN ENDPOINTS
# # =============================================================================

# @app.get("/stats")
# async def get_system_stats(current_user: dict = Depends(get_current_user)):
#     """Get system usage statistics (admin/recruiter only)"""
#     if current_user['role'] not in ['admin', 'recruiter']:
#         raise HTTPException(status_code=403, detail="Access forbidden")
    
#     connection = get_db_connection()
#     cursor = connection.cursor(dictionary=True)
    
#     try:
#         cursor.execute("SELECT role, COUNT(*) as count FROM users GROUP BY role")
#         user_stats = cursor.fetchall()
        
#         cursor.execute("SELECT COUNT(*) as total_resumes FROM resumes")
#         resume_stats = cursor.fetchone()
        
#         cursor.execute("""
#             SELECT 
#                 COUNT(*) as total_profiles, 
#                 SUM(profile_completed) as completed_profiles, 
#                 AVG(ai_score) as avg_ai_score
#             FROM candidate_profile
#         """)
#         profile_stats = cursor.fetchone()
        
#         cursor.execute("""
#             SELECT 
#                 COUNT(*) as verified_users
#             FROM users 
#             WHERE is_verified = TRUE
#         """)
#         verification_stats = cursor.fetchone()
        
#         return {
#             "users": {role['role']: role['count'] for role in user_stats},
#             "resumes_uploaded": resume_stats['total_resumes'] or 0,
#             "verified_users": verification_stats['verified_users'] or 0,
#             "profiles": {
#                 "total": profile_stats['total_profiles'] or 0,
#                 "completed": profile_stats['completed_profiles'] or 0,
#                 "average_ai_score": round(profile_stats['avg_ai_score'] or 0, 2)
#             },
#             "timestamp": datetime.utcnow().isoformat()
#         }
#     finally:
#         cursor.close()
#         connection.close()

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)





from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
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
    title="Human AI Recruitment Agent - Auth Service",
    description="Authentication and user management service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://localhost:3000",
        "http://localhost:8002",  # Resume service
        "http://localhost:8004",  # AI evaluation service
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fixed: Consistent naming and proper directory creation
UPLOAD_DIR = "uploads/resumes"
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Fixed: Better environment variable handling with defaults
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))

# Database configuration with better error handling
DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "recruitment_db"),
    "port": int(os.getenv("MYSQL_PORT", "3306"))
}

# Security
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Fixed: Improved database connection with better error handling
def get_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to connect to database"
            )
    except Error as e:
        print(f"Database connection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection failed"
        )

# =============================================================================
# PYDANTIC MODELS - FIXED
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
    skills: Optional[str] = None
    resume_url: Optional[str] = None
    linkedin_url: Optional[str] = None
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
# HELPER FUNCTIONS - FIXED
# =============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

# Fixed: Proper timezone handling
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Fixed: Better error handling in database queries
def get_user_by_email(email: str):
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        return user
    except Exception as e:
        print(f"Error fetching user: {e}")
        return None
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

def authenticate_user(email: str, password: str):
    user = get_user_by_email(email)
    if not user:
        return False
    if not verify_password(password, user['password_hash']):
        return False
    return user

# Fixed: Better JWT token validation
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError as e:
        print(f"JWT Error: {e}")
        raise credentials_exception

    user = get_user_by_email(email)
    if user is None:
        raise credentials_exception
    return user

# Fixed: Better error handling in profile fetching
async def get_profile_internal(user_id: str):
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM candidate_profile WHERE user_id = %s", (user_id,))
        row = cursor.fetchone()
        return row
    except Exception as e:
        print(f"Error fetching profile: {e}")
        return None
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

# Resume processing helper functions - FIXED
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
        print(f"PDF extraction error: {e}")
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
            exp_score = 30 + (experience - 5) * 2
        else:
            exp_score = 40
        score += exp_score

    # Education scoring (15 points max)
    education_count = len(parsed_data.get("education", []))
    if education_count > 0:
        score += min(education_count * 7, 15)
        
    # Resume completeness scoring (5 points max)
    text_length = parsed_data.get("raw_text_length", 0)
    if text_length > 500:
        score += 5
    elif text_length > 200:
        score += 3
    
    return min(score, 100)

# =============================================================================
# API ENDPOINTS - FIXED
# =============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Human AI Recruitment Agent - Auth Service",
        "status": "healthy",
        "time": datetime.now(timezone.utc).isoformat(),
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
        
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
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
        
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user['email']},
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

# Fixed: Better error handling in registration
@app.post("/register")
async def register_user(user: UserRegistration):
    connection = None
    cursor = None
    
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Check if email already exists
        cursor.execute("SELECT * FROM users WHERE email = %s", (user.email,))
        if cursor.fetchone():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Validate password strength
        if len(user.password) < 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 6 characters long"
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
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
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
        print(f"Database error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed due to database error"
        )
    except Exception as e:
        print(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

@app.get("/profile")
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    return {
        "user": {
            "id": current_user['id'],
            "email": current_user['email'],
            "role": current_user['role'],
            "created_at": current_user['created_at'].isoformat() if current_user['created_at'] else None
        },
        "message": "Profile retrieved successfully"
    }

# Fixed: Better candidate profile management
@app.get("/candidate/profile", response_model=CandidateProfileResponse)
async def get_own_profile(current_user: dict = Depends(get_current_user)):
    """Get current candidate's own profile"""
    if current_user["role"] != 'candidate':
        raise HTTPException(status_code=403, detail="Only candidates can access this endpoint")
    
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
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
            """, (profile_id, current_user["id"], current_user.get("email", "New User")))
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
            full_name=profile_data['full_name'] or current_user.get("email", "User"),
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
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

@app.post("/candidate/profile", response_model=CandidateProfileResponse)
async def create_or_update_profile(
    profile: CandidateProfileCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create or update current candidate's profile"""
    if current_user["role"] != 'candidate':
        raise HTTPException(status_code=403, detail="Only candidates can access this endpoint")
    
    connection = None
    cursor = None
    
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        cursor.execute("SELECT id FROM candidate_profile WHERE user_id=%s", (current_user["id"],))
        exists = cursor.fetchone()
        
        if exists:
            # Update existing profile
            cursor.execute("""
                UPDATE candidate_profile SET 
                full_name=%s, phone=%s, location=%s, experience_years=%s, 
                linkedin_url=%s, profile_completed=TRUE, updated_at=NOW()
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
                    experience_years, linkedin_url, profile_completed, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE, NOW(), NOW())
            """, (
                profile_id, current_user["id"], profile.full_name, 
                profile.phone, profile.location, profile.experience_years, profile.linkedin_url
            ))
        
        connection.commit()
        
        # Return updated profile
        updated_profile = await get_profile_internal(current_user["id"])
        if not updated_profile:
            raise HTTPException(status_code=500, detail="Failed to retrieve updated profile")
            
        return CandidateProfileResponse(
            id=updated_profile['id'],
            user_id=updated_profile['user_id'],
            full_name=updated_profile['full_name'],
            phone=updated_profile.get('phone'),
            location=updated_profile.get('location'),
            experience_years=updated_profile.get('experience_years', 0),
            skills=updated_profile.get('skills'),
            resume_url=updated_profile.get('resume_url'),
            linkedin_url=updated_profile.get('linkedin_url'),
            profile_completed=bool(updated_profile.get('profile_completed')),
            ai_score=updated_profile.get('ai_score', 0),
            created_at=updated_profile['created_at']
        )
    
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

# Fixed: Better candidates listing
@app.get("/candidates")
async def list_candidates(current_user: dict = Depends(get_current_user)):
    """List all candidates with full details for recruiters"""
    if current_user["role"] not in ["recruiter", "admin"]:
        raise HTTPException(status_code=403, detail="Access forbidden")
    
    connection = None
    cursor = None
    
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
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
            LEFT JOIN (
                SELECT candidate_id, filename, file_size, upload_date,
                       ROW_NUMBER() OVER (PARTITION BY candidate_id ORDER BY upload_date DESC) as rn
                FROM resumes
            ) r ON u.id = r.candidate_id AND r.rn = 1
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
                "user_created_at": candidate['user_created_at'].isoformat() if candidate['user_created_at'] else None,
                "profile_created_at": candidate.get('profile_created_at').isoformat() if candidate.get('profile_created_at') else None,
                "resume_upload_date": candidate.get('upload_date').isoformat() if candidate.get('upload_date') else None
            }
            candidate_list.append(candidate_data)
        
        return {"candidates": candidate_list, "total": len(candidate_list)}
    
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

# Resume upload endpoint - FIXED
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
    if len(content) > 10 * 1024 * 1024:  # 10MB limit
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
        
        # Update or create candidate profile
        cursor.execute(
            "SELECT id FROM candidate_profile WHERE user_id = %s", 
            (current_user['id'],)
        )
        profile_exists = cursor.fetchone()
        
        if profile_exists:
            cursor.execute("""
                UPDATE candidate_profile 
                SET skills = %s, ai_score = %s, resume_url = %s, profile_completed = TRUE, updated_at = NOW()
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
                (id, user_id, full_name, skills, ai_score, resume_url, profile_completed, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            """, (
                profile_id, 
                current_user['id'], 
                current_user.get('email', 'Candidate'), 
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
            upload_date=datetime.now(timezone.utc),
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
        if connection and connection.is_connected():
            connection.close()

# Health check endpoint - FIXED
@app.get("/health")
async def health_check():
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
        "service": "Human AI Recruitment Agent - Auth Service",
        "status": "healthy",
        "time": datetime.now(timezone.utc).isoformat(),
        "database": db_status,
        "port": 8001
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
