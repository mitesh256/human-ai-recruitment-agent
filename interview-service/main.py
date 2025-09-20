from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from typing import List, Dict, Optional, Any, Union
import mysql.connector
from mysql.connector import Error
import requests
import os
import uuid
import json
import asyncio
from datetime import datetime, timedelta
from enum import Enum
import google.generativeai as genai
import websockets
import aiofiles
from dotenv import load_dotenv
import re
import base64
import calendar
from collections import defaultdict

load_dotenv()

app = FastAPI(
    title="Interview Management Service",
    description="Handles interview scheduling, video interviews, and AI-powered analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8001",  # Auth service
        "http://localhost:8002",  # Resume service
        "http://localhost:8003",  # Job service
        "http://localhost:8004",  # AI evaluation service
        "http://localhost:8006",  # Communication service
        "http://localhost:8007"   # Analytics service
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# Configure Gemini for AI analysis
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

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

class InterviewType(str, Enum):
    PHONE = "phone"
    VIDEO = "video"
    IN_PERSON = "in_person"
    AI_ASSESSMENT = "ai_assessment"
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    PANEL = "panel"
    GROUP = "group"

class InterviewStatus(str, Enum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    RESCHEDULED = "rescheduled"
    NO_SHOW = "no_show"

class InterviewRound(str, Enum):
    SCREENING = "screening"
    FIRST_ROUND = "first_round"
    SECOND_ROUND = "second_round"
    FINAL_ROUND = "final_round"
    ADDITIONAL = "additional"

class InterviewFeedback(BaseModel):
    technical_skills: Optional[int] = Field(None, ge=1, le=10, description="Rating 1-10")
    communication_skills: Optional[int] = Field(None, ge=1, le=10)
    problem_solving: Optional[int] = Field(None, ge=1, le=10)
    cultural_fit: Optional[int] = Field(None, ge=1, le=10)
    overall_rating: Optional[int] = Field(None, ge=1, le=10)
    strengths: Optional[str] = None
    weaknesses: Optional[str] = None
    comments: Optional[str] = None
    recommendation: Optional[str] = Field(None, description="hire, reject, maybe")
    follow_up_required: bool = False

class InterviewScheduleRequest(BaseModel):
    job_id: str = Field(..., description="Job ID")
    candidate_id: str = Field(..., description="Candidate ID")
    interviewer_ids: List[str] = Field(..., description="List of interviewer user IDs")
    interview_type: InterviewType
    interview_round: InterviewRound = InterviewRound.SCREENING
    scheduled_datetime: datetime = Field(..., description="Interview date and time")
    duration_minutes: int = Field(default=60, ge=15, le=300)
    location: Optional[str] = None
    video_meeting_url: Optional[str] = None
    instructions: Optional[str] = None
    preparation_materials: Optional[List[str]] = []
    send_notifications: bool = True

class InterviewRescheduleRequest(BaseModel):
    interview_id: str
    new_datetime: datetime
    reason: str = Field(..., min_length=10, max_length=500)
    send_notifications: bool = True

class InterviewFeedbackRequest(BaseModel):
    interview_id: str
    interviewer_id: str
    feedback: InterviewFeedback
    private_notes: Optional[str] = None

class VideoInterviewConfig(BaseModel):
    enable_recording: bool = True
    enable_screen_share: bool = True
    enable_chat: bool = True
    max_participants: int = Field(default=10, ge=2, le=50)
    waiting_room: bool = True
    auto_admit_users: bool = False

class AIInterviewAnalysis(BaseModel):
    transcript: str
    video_analysis: Optional[Dict[str, Any]] = None
    audio_analysis: Optional[Dict[str, Any]] = None
    engagement_score: Optional[float] = Field(None, ge=0, le=100)
    confidence_level: Optional[float] = Field(None, ge=0, le=100)
    communication_clarity: Optional[float] = Field(None, ge=0, le=100)
    technical_competency: Optional[float] = Field(None, ge=0, le=100)
    key_insights: List[str] = []
    red_flags: List[str] = []
    recommendations: List[str] = []

class InterviewAvailabilityRequest(BaseModel):
    interviewer_ids: List[str]
    start_date: datetime
    end_date: datetime
    duration_minutes: int = 60
    timezone: str = "UTC"

# =============================================================================
# DATABASE MODELS
# =============================================================================

def create_tables():
    """Create necessary tables if they don't exist"""
    connection = get_db_connection()
    cursor = connection.cursor()
    
    try:
        # Interviews table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interviews (
                id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
                job_id CHAR(36) NOT NULL,
                candidate_id CHAR(36) NOT NULL,
                interview_type ENUM('phone', 'video', 'in_person', 'ai_assessment', 'technical', 'behavioral', 'panel', 'group') NOT NULL,
                interview_round ENUM('screening', 'first_round', 'second_round', 'final_round', 'additional') NOT NULL,
                status ENUM('scheduled', 'in_progress', 'completed', 'cancelled', 'rescheduled', 'no_show') DEFAULT 'scheduled',
                scheduled_datetime DATETIME NOT NULL,
                duration_minutes INT DEFAULT 60,
                location VARCHAR(500),
                video_meeting_url TEXT,
                video_meeting_id VARCHAR(100),
                instructions TEXT,
                preparation_materials JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                created_by CHAR(36) NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
                FOREIGN KEY (candidate_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE CASCADE,
                INDEX idx_interviews_job (job_id),
                INDEX idx_interviews_candidate (candidate_id),
                INDEX idx_interviews_datetime (scheduled_datetime),
                INDEX idx_interviews_status (status)
            )
        """)
        
        # Interview participants (interviewers)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interview_participants (
                id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
                interview_id CHAR(36) NOT NULL,
                user_id CHAR(36) NOT NULL,
                role ENUM('interviewer', 'candidate', 'observer') NOT NULL,
                status ENUM('invited', 'confirmed', 'declined', 'attended', 'no_show') DEFAULT 'invited',
                joined_at DATETIME NULL,
                left_at DATETIME NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (interview_id) REFERENCES interviews(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE KEY unique_interview_participant (interview_id, user_id),
                INDEX idx_participants_interview (interview_id),
                INDEX idx_participants_user (user_id)
            )
        """)
        
        # Interview feedback
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interview_feedback (
                id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
                interview_id CHAR(36) NOT NULL,
                interviewer_id CHAR(36) NOT NULL,
                technical_skills INT CHECK (technical_skills BETWEEN 1 AND 10),
                communication_skills INT CHECK (communication_skills BETWEEN 1 AND 10),
                problem_solving INT CHECK (problem_solving BETWEEN 1 AND 10),
                cultural_fit INT CHECK (cultural_fit BETWEEN 1 AND 10),
                overall_rating INT CHECK (overall_rating BETWEEN 1 AND 10),
                strengths TEXT,
                weaknesses TEXT,
                comments TEXT,
                recommendation ENUM('hire', 'reject', 'maybe'),
                follow_up_required BOOLEAN DEFAULT FALSE,
                private_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (interview_id) REFERENCES interviews(id) ON DELETE CASCADE,
                FOREIGN KEY (interviewer_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE KEY unique_interview_feedback (interview_id, interviewer_id),
                INDEX idx_feedback_interview (interview_id),
                INDEX idx_feedback_interviewer (interviewer_id)
            )
        """)
        
        # Interview recordings and analysis
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interview_recordings (
                id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
                interview_id CHAR(36) NOT NULL,
                recording_url VARCHAR(1000),
                recording_type ENUM('video', 'audio', 'screen_share') DEFAULT 'video',
                file_size BIGINT,
                duration_seconds INT,
                transcript TEXT,
                ai_analysis JSON,
                processing_status ENUM('pending', 'processing', 'completed', 'failed') DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP NULL,
                FOREIGN KEY (interview_id) REFERENCES interviews(id) ON DELETE CASCADE,
                INDEX idx_recordings_interview (interview_id),
                INDEX idx_recordings_status (processing_status)
            )
        """)
        
        # Interview availability (for scheduling)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interviewer_availability (
                id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
                user_id CHAR(36) NOT NULL,
                day_of_week INT NOT NULL CHECK (day_of_week BETWEEN 0 AND 6),
                start_time TIME NOT NULL,
                end_time TIME NOT NULL,
                timezone VARCHAR(50) DEFAULT 'UTC',
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                INDEX idx_availability_user (user_id),
                INDEX idx_availability_day (day_of_week)
            )
        """)
        
        # Interview questions and answers
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interview_questions (
                id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
                interview_id CHAR(36) NOT NULL,
                question_text TEXT NOT NULL,
                question_type ENUM('technical', 'behavioral', 'situational', 'cultural', 'other') DEFAULT 'other',
                answer_text TEXT,
                rating INT CHECK (rating BETWEEN 1 AND 10),
                notes TEXT,
                duration_seconds INT,
                asked_by CHAR(36),
                asked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (interview_id) REFERENCES interviews(id) ON DELETE CASCADE,
                FOREIGN KEY (asked_by) REFERENCES users(id) ON DELETE SET NULL,
                INDEX idx_questions_interview (interview_id),
                INDEX idx_questions_type (question_type)
            )
        """)
        
        connection.commit()
        print("✅ Interview service tables created successfully")
        
    except Error as e:
        print(f"❌ Database setup error: {e}")
    finally:
        cursor.close()
        connection.close()

# Initialize database
create_tables()

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

async def send_notification(user_id: str, notification_type: str, title: str, message: str, data: Dict = None):
    """Send notification via communication service"""
    try:
        response = requests.post(
            "http://localhost:8006/api/v1/communication/notify",
            json={
                "user_id": user_id,
                "type": notification_type,
                "title": title,
                "message": message,
                "data": data or {},
                "channels": ["websocket", "email"]
            },
            timeout=10
        )
        return response.status_code == 200
    except:
        return False

def generate_video_meeting_url(interview_id: str) -> str:
    """Generate video meeting URL (placeholder - integrate with actual video service)"""
    # In production, integrate with services like:
    # - Zoom API
    # - Google Meet API
    # - Microsoft Teams API
    # - Custom WebRTC solution
    return f"https://video-platform.example.com/room/{interview_id}"

def get_available_time_slots(interviewer_ids: List[str], start_date: datetime, end_date: datetime, duration_minutes: int = 60) -> List[Dict]:
    """Get available time slots for interviewers"""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Get interviewer availability
        availability_query = """
            SELECT user_id, day_of_week, start_time, end_time, timezone
            FROM interviewer_availability 
            WHERE user_id IN ({}) AND is_active = TRUE
        """.format(','.join(['%s'] * len(interviewer_ids)))
        
        cursor.execute(availability_query, interviewer_ids)
        availability_data = cursor.fetchall()
        
        # Get existing interviews in date range
        existing_interviews_query = """
            SELECT ip.user_id, i.scheduled_datetime, i.duration_minutes
            FROM interview_participants ip
            JOIN interviews i ON ip.interview_id = i.id
            WHERE ip.user_id IN ({}) 
                AND i.scheduled_datetime BETWEEN %s AND %s
                AND i.status IN ('scheduled', 'in_progress')
        """.format(','.join(['%s'] * len(interviewer_ids)))
        
        cursor.execute(existing_interviews_query, interviewer_ids + [start_date, end_date])
        existing_interviews = cursor.fetchall()
        
        # Calculate available slots (simplified logic)
        available_slots = []
        current_date = start_date.date()
        
        while current_date <= end_date.date():
            day_of_week = current_date.weekday()  # 0=Monday, 6=Sunday
            
            # Find availability for this day
            day_availability = [
                av for av in availability_data 
                if av['day_of_week'] == day_of_week
            ]
            
            if day_availability:
                # Generate time slots (simplified - every hour from 9 AM to 5 PM)
                for hour in range(9, 17):
                    slot_datetime = datetime.combine(current_date, datetime.min.time().replace(hour=hour))
                    
                    # Check if all interviewers are available
                    all_available = True
                    for interviewer_id in interviewer_ids:
                        # Check if interviewer has conflicting interview
                        conflict = any(
                            ei['user_id'] == interviewer_id and
                            abs((ei['scheduled_datetime'] - slot_datetime).total_seconds()) < (duration_minutes * 60)
                            for ei in existing_interviews
                        )
                        if conflict:
                            all_available = False
                            break
                    
                    if all_available:
                        available_slots.append({
                            "datetime": slot_datetime.isoformat(),
                            "available_interviewers": interviewer_ids,
                            "duration_minutes": duration_minutes
                        })
            
            current_date += timedelta(days=1)
        
        return available_slots[:20]  # Limit to 20 slots
        
    finally:
        cursor.close()
        connection.close()

# =============================================================================
# AI INTERVIEW ANALYSIS ENGINE
# =============================================================================

class AIInterviewAnalyzer:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-pro')
    
    async def analyze_interview_transcript(self, transcript: str, job_context: Dict, candidate_context: Dict) -> AIInterviewAnalysis:
        """Analyze interview transcript using AI"""
        try:
            prompt = f"""
            Analyze this interview transcript and provide detailed insights:
            
            Job Context: {json.dumps(job_context)}
            Candidate Context: {json.dumps(candidate_context)}
            
            Transcript:
            {transcript}
            
            Please analyze the following aspects and provide scores (0-100):
            1. Engagement Score - How engaged was the candidate?
            2. Confidence Level - How confident did the candidate appear?
            3. Communication Clarity - How clearly did they communicate?
            4. Technical Competency - Based on technical questions/answers
            
            Also provide:
            - Key insights (3-5 points)
            - Red flags (if any)
            - Recommendations for next steps
            
            Format your response as JSON with the structure matching AIInterviewAnalysis model.
            """
            
            response = self.model.generate_content(prompt)
            analysis_text = response.text.strip()
            
            # Parse AI response (simplified - in production use more robust parsing)
            try:
                # Extract scores and insights from AI response
                engagement_score = self._extract_score(analysis_text, "engagement")
                confidence_level = self._extract_score(analysis_text, "confidence")
                communication_clarity = self._extract_score(analysis_text, "communication")
                technical_competency = self._extract_score(analysis_text, "technical")
                
                key_insights = self._extract_list_items(analysis_text, "insights")
                red_flags = self._extract_list_items(analysis_text, "red flags")
                recommendations = self._extract_list_items(analysis_text, "recommendations")
                
                return AIInterviewAnalysis(
                    transcript=transcript,
                    engagement_score=engagement_score,
                    confidence_level=confidence_level,
                    communication_clarity=communication_clarity,
                    technical_competency=technical_competency,
                    key_insights=key_insights,
                    red_flags=red_flags,
                    recommendations=recommendations
                )
                
            except Exception as parse_error:
                print(f"Error parsing AI response: {parse_error}")
                # Return basic analysis if parsing fails
                return AIInterviewAnalysis(
                    transcript=transcript,
                    engagement_score=75.0,
                    confidence_level=70.0,
                    communication_clarity=80.0,
                    technical_competency=65.0,
                    key_insights=["AI analysis completed", "Review transcript for detailed insights"],
                    red_flags=[],
                    recommendations=["Conduct follow-up interview", "Review technical skills"]
                )
                
        except Exception as e:
            print(f"AI analysis error: {e}")
            # Return default analysis if AI fails
            return AIInterviewAnalysis(
                transcript=transcript,
                engagement_score=70.0,
                confidence_level=70.0,
                communication_clarity=70.0,
                technical_competency=70.0,
                key_insights=["AI analysis temporarily unavailable"],
                red_flags=[],
                recommendations=["Manual review recommended"]
            )
    
    def _extract_score(self, text: str, metric: str) -> float:
        """Extract score from AI response text"""
        try:
            # Look for patterns like "Engagement Score: 85" or "engagement: 85/100"
            pattern = rf"{metric}.*?(\d+)"
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return float(matches[0])
            return 70.0  # Default score
        except:
            return 70.0
    
    def _extract_list_items(self, text: str, section: str) -> List[str]:
        """Extract list items from AI response text"""
        try:
            # Simple extraction - look for bullet points or numbered lists
            lines = text.split('\n')
            items = []
            in_section = False
            
            for line in lines:
                if section.lower() in line.lower():
                    in_section = True
                    continue
                    
                if in_section:
                    if line.strip().startswith(('-', '•', '*')) or re.match(r'^\d+\.', line.strip()):
                        items.append(line.strip().lstrip('-•*0123456789. '))
                    elif line.strip() == '' and items:
                        break
                        
            return items[:5]  # Limit to 5 items
        except:
            return []

ai_analyzer = AIInterviewAnalyzer()

# =============================================================================
# WEBSOCKET CONNECTION MANAGER FOR LIVE INTERVIEWS
# =============================================================================

class InterviewConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}  # interview_id -> {user_id -> websocket}
        self.interview_sessions: Dict[str, Dict] = {}  # interview_id -> session_data
    
    async def connect(self, websocket: WebSocket, interview_id: str, user_id: str):
        await websocket.accept()
        
        if interview_id not in self.active_connections:
            self.active_connections[interview_id] = {}
            self.interview_sessions[interview_id] = {
                "participants": {},
                "started_at": datetime.now(),
                "status": "active"
            }
        
        self.active_connections[interview_id][user_id] = websocket
        self.interview_sessions[interview_id]["participants"][user_id] = {
            "joined_at": datetime.now(),
            "status": "connected"
        }
        
        # Notify other participants
        await self.broadcast_to_interview(interview_id, {
            "type": "participant_joined",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }, exclude_user=user_id)
    
    def disconnect(self, interview_id: str, user_id: str):
        if interview_id in self.active_connections:
            if user_id in self.active_connections[interview_id]:
                del self.active_connections[interview_id][user_id]
            
            if interview_id in self.interview_sessions:
                if user_id in self.interview_sessions[interview_id]["participants"]:
                    self.interview_sessions[interview_id]["participants"][user_id]["status"] = "disconnected"
                    self.interview_sessions[interview_id]["participants"][user_id]["left_at"] = datetime.now()
            
            # Clean up empty interview sessions
            if not self.active_connections[interview_id]:
                del self.active_connections[interview_id]
                if interview_id in self.interview_sessions:
                    self.interview_sessions[interview_id]["status"] = "ended"
    
    async def send_personal_message(self, interview_id: str, user_id: str, message: Dict):
        if (interview_id in self.active_connections and 
            user_id in self.active_connections[interview_id]):
            websocket = self.active_connections[interview_id][user_id]
            try:
                await websocket.send_text(json.dumps(message))
                return True
            except:
                self.disconnect(interview_id, user_id)
        return False
    
    async def broadcast_to_interview(self, interview_id: str, message: Dict, exclude_user: str = None):
        if interview_id in self.active_connections:
            disconnected_users = []
            
            for user_id, websocket in self.active_connections[interview_id].items():
                if exclude_user and user_id == exclude_user:
                    continue
                    
                try:
                    await websocket.send_text(json.dumps(message))
                except:
                    disconnected_users.append(user_id)
            
            # Clean up disconnected users
            for user_id in disconnected_users:
                self.disconnect(interview_id, user_id)

interview_manager = InterviewConnectionManager()

# =============================================================================
# MAIN ENDPOINTS
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
        "service": "interview-management",
        "port": 8008,
        "database": db_status,
        "active_interviews": len(interview_manager.active_connections),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/interviews/schedule")
async def schedule_interview(
    request: InterviewScheduleRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(verify_token)
):
    """Schedule a new interview"""
    if current_user["user"]["role"] not in ["admin", "recruiter"]:
        raise HTTPException(status_code=403, detail="Access forbidden")
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Verify job exists and user has access
        cursor.execute("""
            SELECT id, title, company_id FROM jobs 
            WHERE id = %s AND (created_by = %s OR %s IN ('admin'))
        """, (request.job_id, current_user["user"]["id"], current_user["user"]["role"]))
        
        job = cursor.fetchone()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found or access denied")
        
        # Verify candidate exists
        cursor.execute("SELECT id, email FROM users WHERE id = %s", (request.candidate_id,))
        candidate = cursor.fetchone()
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        # Verify interviewers exist
        interviewer_placeholders = ','.join(['%s'] * len(request.interviewer_ids))
        cursor.execute(f"""
            SELECT id, email, full_name FROM users 
            WHERE id IN ({interviewer_placeholders}) AND role IN ('recruiter', 'admin', 'interviewer')
        """, request.interviewer_ids)
        
        interviewers = cursor.fetchall()
        if len(interviewers) != len(request.interviewer_ids):
            raise HTTPException(status_code=400, detail="Some interviewers not found or invalid")
        
        # Create interview
        interview_id = str(uuid.uuid4())
        video_meeting_url = None
        video_meeting_id = None
        
        if request.interview_type in [InterviewType.VIDEO, InterviewType.PANEL]:
            video_meeting_url = generate_video_meeting_url(interview_id)
            video_meeting_id = interview_id
        
        cursor.execute("""
            INSERT INTO interviews 
            (id, job_id, candidate_id, interview_type, interview_round, scheduled_datetime, 
             duration_minutes, location, video_meeting_url, video_meeting_id, instructions, 
             preparation_materials, created_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            interview_id, request.job_id, request.candidate_id, request.interview_type.value,
            request.interview_round.value, request.scheduled_datetime, request.duration_minutes,
            request.location, video_meeting_url, video_meeting_id, request.instructions,
            json.dumps(request.preparation_materials or []), current_user["user"]["id"]
        ))
        
        # Add interviewers as participants
        for interviewer in interviewers:
            cursor.execute("""
                INSERT INTO interview_participants (id, interview_id, user_id, role, status)
                VALUES (%s, %s, %s, 'interviewer', 'invited')
            """, (str(uuid.uuid4()), interview_id, interviewer["id"]))
        
        # Add candidate as participant
        cursor.execute("""
            INSERT INTO interview_participants (id, interview_id, user_id, role, status)
            VALUES (%s, %s, %s, 'candidate', 'invited')
        """, (str(uuid.uuid4()), interview_id, request.candidate_id))
        
        connection.commit()
        
        # Send notifications if requested
        if request.send_notifications:
            background_tasks.add_task(
                send_interview_notifications,
                interview_id,
                candidate,
                interviewers,
                job,
                request
            )
        
        return {
            "interview_id": interview_id,
            "message": "Interview scheduled successfully",
            "scheduled_datetime": request.scheduled_datetime.isoformat(),
            "video_meeting_url": video_meeting_url,
            "participants_count": len(interviewers) + 1
        }
        
    except Exception as e:
        connection.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to schedule interview: {str(e)}")
    finally:
        cursor.close()
        connection.close()

async def send_interview_notifications(interview_id: str, candidate: Dict, interviewers: List[Dict], job: Dict, request: InterviewScheduleRequest):
    """Send interview notifications to all participants"""
    
    # Notify candidate
    candidate_message = f"""
    Your interview for {job['title']} has been scheduled.
    
    Interview Details:
    - Date & Time: {request.scheduled_datetime.strftime('%Y-%m-%d at %I:%M %p')}
    - Type: {request.interview_type.value.replace('_', ' ').title()}
    - Duration: {request.duration_minutes} minutes
    - Location: {request.location or 'Video call'}
    
    {f'Meeting URL: {request.video_meeting_url}' if request.video_meeting_url else ''}
    {f'Instructions: {request.instructions}' if request.instructions else ''}
    """
    
    await send_notification(
        candidate["id"],
        "interview_scheduled",
        "Interview Scheduled",
        candidate_message.strip(),
        {
            "interview_id": interview_id,
            "job_title": job["title"],
            "scheduled_datetime": request.scheduled_datetime.isoformat()
        }
    )
    
    # Notify interviewers
    for interviewer in interviewers:
        interviewer_message = f"""
        You have been assigned to conduct an interview.
        
        Candidate: Interview scheduled
        Job: {job['title']}
        Date & Time: {request.scheduled_datetime.strftime('%Y-%m-%d at %I:%M %p')}
        Type: {request.interview_type.value.replace('_', ' ').title()}
        Duration: {request.duration_minutes} minutes
        
        {f'Meeting URL: {request.video_meeting_url}' if request.video_meeting_url else ''}
        {f'Instructions: {request.instructions}' if request.instructions else ''}
        """
        
        await send_notification(
            interviewer["id"],
            "interview_assigned",
            "Interview Assignment",
            interviewer_message.strip(),
            {
                "interview_id": interview_id,
                "job_title": job["title"],
                "scheduled_datetime": request.scheduled_datetime.isoformat()
            }
        )

@app.get("/api/v1/interviews")
async def list_interviews(
    status: Optional[InterviewStatus] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    job_id: Optional[str] = None,
    limit: int = 20,
    current_user: dict = Depends(verify_token)
):
    """List interviews based on user role and filters"""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Build query based on user role
        base_query = """
            SELECT 
                i.*,
                j.title as job_title,
                j.company_id,
                c.email as candidate_email,
                c.full_name as candidate_name,
                GROUP_CONCAT(
                    CONCAT(int_user.full_name, ' (', ip.role, ')')
                    SEPARATOR ', '
                ) as participants
            FROM interviews i
            JOIN jobs j ON i.job_id = j.id
            JOIN users c ON i.candidate_id = c.id
            LEFT JOIN interview_participants ip ON i.id = ip.interview_id
            LEFT JOIN users int_user ON ip.user_id = int_user.id
        """
        
        conditions = []
        params = []
        
        # Role-based access control
        if current_user["user"]["role"] == "candidate":
            conditions.append("i.candidate_id = %s")
            params.append(current_user["user"]["id"])
        elif current_user["user"]["role"] == "recruiter":
            conditions.append("""
                (i.created_by = %s OR 
                 EXISTS (SELECT 1 FROM interview_participants ip2 
                        WHERE ip2.interview_id = i.id AND ip2.user_id = %s))
            """)
            params.extend([current_user["user"]["id"], current_user["user"]["id"]])
        # Admin can see all interviews
        
        # Apply filters
        if status:
            conditions.append("i.status = %s")
            params.append(status.value)
        
        if date_from:
            conditions.append("i.scheduled_datetime >= %s")
            params.append(date_from)
        
        if date_to:
            conditions.append("i.scheduled_datetime <= %s")
            params.append(date_to)
        
        if job_id:
            conditions.append("i.job_id = %s")
            params.append(job_id)
        
        # Combine query
        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)
        
        base_query += """
            GROUP BY i.id
            ORDER BY i.scheduled_datetime DESC
            LIMIT %s
        """
        params.append(limit)
        
        cursor.execute(base_query, params)
        interviews = cursor.fetchall()
        
        # Parse JSON fields
        for interview in interviews:
            if interview['preparation_materials']:
                interview['preparation_materials'] = json.loads(interview['preparation_materials'])
        
        return {
            "interviews": interviews,
            "total": len(interviews)
        }
        
    finally:
        cursor.close()
        connection.close()

@app.get("/api/v1/interviews/{interview_id}")
async def get_interview(
    interview_id: str,
    current_user: dict = Depends(verify_token)
):
    """Get interview details"""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Get interview with access control
        cursor.execute("""
            SELECT 
                i.*,
                j.title as job_title,
                j.description as job_description,
                j.company_id,
                c.email as candidate_email,
                c.full_name as candidate_name,
                c.phone as candidate_phone
            FROM interviews i
            JOIN jobs j ON i.job_id = j.id
            JOIN users c ON i.candidate_id = c.id
            WHERE i.id = %s
        """, (interview_id,))
        
        interview = cursor.fetchone()
        if not interview:
            raise HTTPException(status_code=404, detail="Interview not found")
        
        # Check access permissions
        user_role = current_user["user"]["role"]
        user_id = current_user["user"]["id"]
        
        has_access = False
        if user_role == "admin":
            has_access = True
        elif user_role == "candidate" and interview["candidate_id"] == user_id:
            has_access = True
        elif user_role in ["recruiter", "interviewer"]:
            # Check if user is interviewer or creator
            cursor.execute("""
                SELECT 1 FROM interview_participants 
                WHERE interview_id = %s AND user_id = %s
                UNION
                SELECT 1 FROM interviews WHERE id = %s AND created_by = %s
            """, (interview_id, user_id, interview_id, user_id))
            has_access = cursor.fetchone() is not None
        
        if not has_access:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get participants
        cursor.execute("""
            SELECT 
                ip.*,
                u.full_name,
                u.email,
                u.role as user_role
            FROM interview_participants ip
            JOIN users u ON ip.user_id = u.id
            WHERE ip.interview_id = %s
            ORDER BY ip.role, u.full_name
        """, (interview_id,))
        
        participants = cursor.fetchall()
        
        # Get feedback (if user has access)
        feedback = []
        if user_role in ["admin", "recruiter"] or interview["candidate_id"] == user_id:
            cursor.execute("""
                SELECT 
                    if_.*,
                    u.full_name as interviewer_name
                FROM interview_feedback if_
                JOIN users u ON if_.interviewer_id = u.id
                WHERE if_.interview_id = %s
            """, (interview_id,))
            feedback = cursor.fetchall()
        
        # Parse JSON fields
        if interview['preparation_materials']:
            interview['preparation_materials'] = json.loads(interview['preparation_materials'])
        
        return {
            "interview": interview,
            "participants": participants,
            "feedback": feedback,
            "can_provide_feedback": user_role in ["admin", "recruiter", "interviewer"] and any(
                p['user_id'] == user_id and p['role'] == 'interviewer' 
                for p in participants
            )
        }
        
    finally:
        cursor.close()
        connection.close()

@app.put("/api/v1/interviews/{interview_id}/reschedule")
async def reschedule_interview(
    interview_id: str,
    request: InterviewRescheduleRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(verify_token)
):
    """Reschedule an existing interview"""
    if current_user["user"]["role"] not in ["admin", "recruiter"]:
        raise HTTPException(status_code=403, detail="Access forbidden")
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Get current interview
        cursor.execute("""
            SELECT i.*, j.title as job_title
            FROM interviews i
            JOIN jobs j ON i.job_id = j.id
            WHERE i.id = %s AND (i.created_by = %s OR %s = 'admin')
        """, (interview_id, current_user["user"]["id"], current_user["user"]["role"]))
        
        interview = cursor.fetchone()
        if not interview:
            raise HTTPException(status_code=404, detail="Interview not found or access denied")
        
        if interview["status"] not in ["scheduled", "rescheduled"]:
            raise HTTPException(status_code=400, detail="Interview cannot be rescheduled")
        
        # Update interview
        old_datetime = interview["scheduled_datetime"]
        cursor.execute("""
            UPDATE interviews 
            SET scheduled_datetime = %s, status = 'rescheduled', updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (request.new_datetime, interview_id))
        
        # Log reschedule reason (could create a separate table for this)
        # For now, we'll just update
        
        connection.commit()
        
        # Send notifications
        if request.send_notifications:
            background_tasks.add_task(
                send_reschedule_notifications,
                interview_id,
                old_datetime,
                request.new_datetime,
                request.reason,
                interview
            )
        
        return {
            "message": "Interview rescheduled successfully",
            "old_datetime": old_datetime.isoformat(),
            "new_datetime": request.new_datetime.isoformat(),
            "reason": request.reason
        }
        
    except Exception as e:
        connection.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to reschedule interview: {str(e)}")
    finally:
        cursor.close()
        connection.close()

async def send_reschedule_notifications(interview_id: str, old_datetime: datetime, new_datetime: datetime, reason: str, interview: Dict):
    """Send reschedule notifications to all participants"""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Get all participants
        cursor.execute("""
            SELECT ip.user_id, u.full_name, u.email
            FROM interview_participants ip
            JOIN users u ON ip.user_id = u.id
            WHERE ip.interview_id = %s
        """, (interview_id,))
        
        participants = cursor.fetchall()
        
        for participant in participants:
            message = f"""
            Interview Rescheduled
            
            Job: {interview['job_title']}
            
            Original Time: {old_datetime.strftime('%Y-%m-%d at %I:%M %p')}
            New Time: {new_datetime.strftime('%Y-%m-%d at %I:%M %p')}
            
            Reason: {reason}
            
            Please update your calendar accordingly.
            """
            
            await send_notification(
                participant["user_id"],
                "interview_rescheduled",
                "Interview Rescheduled",
                message.strip(),
                {
                    "interview_id": interview_id,
                    "old_datetime": old_datetime.isoformat(),
                    "new_datetime": new_datetime.isoformat()
                }
            )
    
    finally:
        cursor.close()
        connection.close()

@app.post("/api/v1/interviews/{interview_id}/feedback")
async def submit_interview_feedback(
    interview_id: str,
    request: InterviewFeedbackRequest,
    current_user: dict = Depends(verify_token)
):
    """Submit interview feedback"""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Verify interview exists and user is an interviewer
        cursor.execute("""
            SELECT ip.id
            FROM interview_participants ip
            JOIN interviews i ON ip.interview_id = i.id
            WHERE ip.interview_id = %s 
                AND ip.user_id = %s 
                AND ip.role = 'interviewer'
                AND i.status IN ('completed', 'in_progress')
        """, (interview_id, current_user["user"]["id"]))
        
        participant = cursor.fetchone()
        if not participant:
            raise HTTPException(status_code=403, detail="Not authorized to provide feedback for this interview")
        
        # Insert or update feedback
        feedback_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO interview_feedback 
            (id, interview_id, interviewer_id, technical_skills, communication_skills, 
             problem_solving, cultural_fit, overall_rating, strengths, weaknesses, 
             comments, recommendation, follow_up_required, private_notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                technical_skills = VALUES(technical_skills),
                communication_skills = VALUES(communication_skills),
                problem_solving = VALUES(problem_solving),
                cultural_fit = VALUES(cultural_fit),
                overall_rating = VALUES(overall_rating),
                strengths = VALUES(strengths),
                weaknesses = VALUES(weaknesses),
                comments = VALUES(comments),
                recommendation = VALUES(recommendation),
                follow_up_required = VALUES(follow_up_required),
                private_notes = VALUES(private_notes),
                updated_at = CURRENT_TIMESTAMP
        """, (
            feedback_id, interview_id, current_user["user"]["id"],
            request.feedback.technical_skills, request.feedback.communication_skills,
            request.feedback.problem_solving, request.feedback.cultural_fit,
            request.feedback.overall_rating, request.feedback.strengths,
            request.feedback.weaknesses, request.feedback.comments,
            request.feedback.recommendation, request.feedback.follow_up_required,
            request.private_notes
        ))
        
        connection.commit()
        
        return {
            "message": "Feedback submitted successfully",
            "feedback_id": feedback_id
        }
        
    except Exception as e:
        connection.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")
    finally:
        cursor.close()
        connection.close()

@app.get("/api/v1/interviews/availability")
async def get_interviewer_availability(
    request: InterviewAvailabilityRequest = Depends(),
    current_user: dict = Depends(verify_token)
):
    """Get available time slots for interviewers"""
    if current_user["user"]["role"] not in ["admin", "recruiter"]:
        raise HTTPException(status_code=403, detail="Access forbidden")
    
    try:
        available_slots = get_available_time_slots(
            request.interviewer_ids,
            request.start_date,
            request.end_date,
            request.duration_minutes
        )
        
        return {
            "available_slots": available_slots,
            "total_slots": len(available_slots),
            "date_range": {
                "start": request.start_date.isoformat(),
                "end": request.end_date.isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get availability: {str(e)}")

@app.post("/api/v1/interviews/{interview_id}/start")
async def start_interview(
    interview_id: str,
    current_user: dict = Depends(verify_token)
):
    """Start an interview session"""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Verify user is participant
        cursor.execute("""
            SELECT i.*, ip.role
            FROM interviews i
            JOIN interview_participants ip ON i.id = ip.interview_id
            WHERE i.id = %s AND ip.user_id = %s
        """, (interview_id, current_user["user"]["id"]))
        
        result = cursor.fetchone()
        if not result:
            raise HTTPException(status_code=403, detail="Not authorized to start this interview")
        
        # Update interview status
        cursor.execute("""
            UPDATE interviews 
            SET status = 'in_progress', updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND status = 'scheduled'
        """, (interview_id,))
        
        # Update participant status
        cursor.execute("""
            UPDATE interview_participants 
            SET status = 'attended', joined_at = CURRENT_TIMESTAMP
            WHERE interview_id = %s AND user_id = %s
        """, (interview_id, current_user["user"]["id"]))
        
        connection.commit()
        
        return {
            "message": "Interview started successfully",
            "interview_id": interview_id,
            "video_meeting_url": result.get("video_meeting_url"),
            "status": "in_progress"
        }
        
    except Exception as e:
        connection.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to start interview: {str(e)}")
    finally:
        cursor.close()
        connection.close()

@app.post("/api/v1/interviews/{interview_id}/complete")
async def complete_interview(
    interview_id: str,
    transcript: Optional[str] = None,
    current_user: dict = Depends(verify_token)
):
    """Complete an interview and trigger AI analysis"""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Verify user is participant and interview is in progress
        cursor.execute("""
            SELECT i.*, ip.role, j.title, j.requirements
            FROM interviews i
            JOIN interview_participants ip ON i.id = ip.interview_id
            JOIN jobs j ON i.job_id = j.id
            WHERE i.id = %s AND ip.user_id = %s AND i.status = 'in_progress'
        """, (interview_id, current_user["user"]["id"]))
        
        result = cursor.fetchone()
        if not result:
            raise HTTPException(status_code=403, detail="Not authorized to complete this interview")
        
        # Update interview status
        cursor.execute("""
            UPDATE interviews 
            SET status = 'completed', updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (interview_id,))
        
        # If transcript provided, store and analyze
        if transcript and len(transcript.strip()) > 50:
            # Store transcript
            recording_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO interview_recordings 
                (id, interview_id, transcript, processing_status)
                VALUES (%s, %s, %s, 'completed')
            """, (recording_id, interview_id, transcript))
            
            # Trigger AI analysis (background task)
            job_context = {
                "title": result["title"],
                "requirements": result.get("requirements", "")
            }
            
            # Get candidate info for context
            cursor.execute("""
                SELECT full_name, email FROM users WHERE id = %s
            """, (result["candidate_id"],))
            candidate = cursor.fetchone()
            
            candidate_context = {
                "name": candidate["full_name"] if candidate else "Unknown",
                "email": candidate["email"] if candidate else ""
            }
            
            # Perform AI analysis
            try:
                analysis = await ai_analyzer.analyze_interview_transcript(
                    transcript, job_context, candidate_context
                )
                
                # Store analysis
                cursor.execute("""
                    UPDATE interview_recordings 
                    SET ai_analysis = %s, processed_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (json.dumps(analysis.dict()), recording_id))
                
            except Exception as analysis_error:
                print(f"AI analysis failed: {analysis_error}")
        
        connection.commit()
        
        return {
            "message": "Interview completed successfully",
            "interview_id": interview_id,
            "status": "completed",
            "transcript_processed": transcript is not None
        }
        
    except Exception as e:
        connection.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to complete interview: {str(e)}")
    finally:
        cursor.close()
        connection.close()

# =============================================================================
# WEBSOCKET ENDPOINTS FOR LIVE INTERVIEWS
# =============================================================================

@app.websocket("/ws/interview/{interview_id}/{user_id}")
async def interview_websocket(websocket: WebSocket, interview_id: str, user_id: str):
    """WebSocket endpoint for live interview sessions"""
    try:
        # Verify user has access to this interview
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT 1 FROM interview_participants 
            WHERE interview_id = %s AND user_id = %s
        """, (interview_id, user_id))
        
        if not cursor.fetchone():
            await websocket.close(code=4003, reason="Access denied")
            return
        
        cursor.close()
        connection.close()
        
        await interview_manager.connect(websocket, interview_id, user_id)
        
        # Send welcome message
        await interview_manager.send_personal_message(interview_id, user_id, {
            "type": "connected",
            "message": "Connected to interview session",
            "timestamp": datetime.now().isoformat()
        })
        
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "chat":
                    # Broadcast chat message to all participants
                    await interview_manager.broadcast_to_interview(interview_id, {
                        "type": "chat",
                        "user_id": user_id,
                        "message": message.get("message", ""),
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif message.get("type") == "screen_share":
                    # Handle screen sharing events
                    await interview_manager.broadcast_to_interview(interview_id, {
                        "type": "screen_share",
                        "user_id": user_id,
                        "action": message.get("action", "start"),  # start/stop
                        "timestamp": datetime.now().isoformat()
                    }, exclude_user=user_id)
                
                elif message.get("type") == "recording":
                    # Handle recording events
                    await interview_manager.broadcast_to_interview(interview_id, {
                        "type": "recording",
                        "action": message.get("action", "start"),  # start/stop/pause
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif message.get("type") == "ping":
                    # Keep-alive ping
                    await interview_manager.send_personal_message(interview_id, user_id, {
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                    
            except json.JSONDecodeError:
                await interview_manager.send_personal_message(interview_id, user_id, {
                    "type": "error",
                    "message": "Invalid message format",
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        interview_manager.disconnect(interview_id, user_id)
        
        # Notify other participants
        await interview_manager.broadcast_to_interview(interview_id, {
            "type": "participant_left",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        print(f"WebSocket error: {e}")
        interview_manager.disconnect(interview_id, user_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
