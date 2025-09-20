from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
import mysql.connector
from mysql.connector import Error
import requests
import os
import uuid
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Analytics & Reports Service",
    description="Provides analytics, reports, and business intelligence for recruitment",
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
        "http://localhost:8006"   # Communication service
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# Configure Gemini for AI insights
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

class ReportType(str, Enum):
    RECRUITMENT_DASHBOARD = "recruitment_dashboard"
    HIRING_METRICS = "hiring_metrics"
    CANDIDATE_ANALYTICS = "candidate_analytics"
    JOB_PERFORMANCE = "job_performance"
    AI_INSIGHTS = "ai_insights"
    BIAS_ANALYSIS = "bias_analysis"
    TIME_TO_HIRE = "time_to_hire"
    COST_PER_HIRE = "cost_per_hire"
    CONVERSION_FUNNEL = "conversion_funnel"
    DIVERSITY_REPORT = "diversity_report"

class DateRange(str, Enum):
    LAST_7_DAYS = "7d"
    LAST_30_DAYS = "30d"
    LAST_3_MONTHS = "3m"
    LAST_6_MONTHS = "6m"
    LAST_YEAR = "1y"
    CUSTOM = "custom"

class ExportFormat(str, Enum):
    JSON = "json"
    CSV = "csv"
    PDF = "pdf"
    EXCEL = "xlsx"

class ReportRequest(BaseModel):
    report_type: ReportType
    date_range: DateRange = DateRange.LAST_30_DAYS
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    filters: Optional[Dict[str, Any]] = {}
    export_format: ExportFormat = ExportFormat.JSON
    include_charts: bool = True
    group_by: Optional[str] = None

class MetricsResponse(BaseModel):
    total_applications: int
    total_jobs: int
    total_candidates: int
    active_jobs: int
    hired_candidates: int
    avg_time_to_hire: float
    avg_ai_score: float
    conversion_rate: float
    period: str

class ChartData(BaseModel):
    type: str  # bar, line, pie, scatter
    title: str
    data: Dict[str, Any]
    description: Optional[str] = None

class AnalyticsReport(BaseModel):
    report_id: str
    report_type: str
    generated_at: datetime
    date_range: Dict[str, Any]
    metrics: Dict[str, Any]
    charts: List[ChartData]
    insights: List[str]
    recommendations: List[str]
    raw_data: Optional[Dict[str, Any]] = None

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

def get_date_range_filter(date_range: DateRange, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None):
    """Get date filter based on range selection"""
    now = datetime.now()
    
    if date_range == DateRange.CUSTOM:
        if not start_date or not end_date:
            raise HTTPException(status_code=400, detail="Custom date range requires start_date and end_date")
        return start_date, end_date
    
    range_mapping = {
        DateRange.LAST_7_DAYS: 7,
        DateRange.LAST_30_DAYS: 30,
        DateRange.LAST_3_MONTHS: 90,
        DateRange.LAST_6_MONTHS: 180,
        DateRange.LAST_YEAR: 365
    }
    
    days = range_mapping.get(date_range, 30)
    start_date = now - timedelta(days=days)
    end_date = now
    
    return start_date, end_date

def create_tables():
    """Create necessary tables if they don't exist"""
    connection = get_db_connection()
    cursor = connection.cursor()
    
    try:
        # Analytics reports table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics_reports (
                id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
                report_type VARCHAR(50) NOT NULL,
                generated_by CHAR(36) NOT NULL,
                report_data JSON NOT NULL,
                filters JSON,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY (generated_by) REFERENCES users(id) ON DELETE CASCADE,
                INDEX idx_analytics_type (report_type),
                INDEX idx_analytics_generated (generated_at)
            )
        """)
        
        # Metrics cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics_cache (
                id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
                cache_key VARCHAR(255) UNIQUE NOT NULL,
                metric_data JSON NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                INDEX idx_cache_key (cache_key),
                INDEX idx_cache_expires (expires_at)
            )
        """)
        
        # User activity logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_activity_logs (
                id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
                user_id CHAR(36) NOT NULL,
                activity_type VARCHAR(50) NOT NULL,
                activity_data JSON,
                ip_address VARCHAR(45),
                user_agent TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                INDEX idx_activity_user (user_id),
                INDEX idx_activity_type (activity_type),
                INDEX idx_activity_timestamp (timestamp)
            )
        """)
        
        connection.commit()
        print("✅ Analytics service tables created successfully")
        
    except Error as e:
        print(f"❌ Database setup error: {e}")
    finally:
        cursor.close()
        connection.close()

# Initialize database
create_tables()

# =============================================================================
# ANALYTICS ENGINE
# =============================================================================

class AnalyticsEngine:
    def __init__(self):
        self.ai_model = genai.GenerativeModel('gemini-1.5-pro')
    
    async def generate_recruitment_dashboard(self, start_date: datetime, end_date: datetime, filters: Dict = None) -> AnalyticsReport:
        """Generate comprehensive recruitment dashboard"""
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        try:
            # Basic metrics
            metrics = await self._get_basic_metrics(cursor, start_date, end_date, filters)
            
            # Chart data
            charts = []
            
            # 1. Applications over time
            applications_chart = await self._get_applications_timeline(cursor, start_date, end_date)
            charts.append(applications_chart)
            
            # 2. Job performance
            job_performance_chart = await self._get_job_performance_chart(cursor, start_date, end_date)
            charts.append(job_performance_chart)
            
            # 3. AI scores distribution
            ai_scores_chart = await self._get_ai_scores_distribution(cursor, start_date, end_date)
            charts.append(ai_scores_chart)
            
            # 4. Application status breakdown
            status_chart = await self._get_application_status_breakdown(cursor, start_date, end_date)
            charts.append(status_chart)
            
            # Generate AI insights
            insights = await self._generate_ai_insights(metrics, charts)
            recommendations = await self._generate_recommendations(metrics, insights)
            
            return AnalyticsReport(
                report_id=str(uuid.uuid4()),
                report_type="recruitment_dashboard",
                generated_at=datetime.now(),
                date_range={"start": start_date, "end": end_date},
                metrics=metrics,
                charts=charts,
                insights=insights,
                recommendations=recommendations
            )
            
        finally:
            cursor.close()
            connection.close()
    
    async def _get_basic_metrics(self, cursor, start_date: datetime, end_date: datetime, filters: Dict = None) -> Dict:
        """Get basic recruitment metrics"""
        
        # Total applications in period
        cursor.execute("""
            SELECT COUNT(*) as total_applications
            FROM applications 
            WHERE created_at BETWEEN %s AND %s
        """, (start_date, end_date))
        total_applications = cursor.fetchone()['total_applications']
        
        # Total jobs in period
        cursor.execute("""
            SELECT COUNT(*) as total_jobs
            FROM jobs 
            WHERE created_at BETWEEN %s AND %s
        """, (start_date, end_date))
        total_jobs = cursor.fetchone()['total_jobs']
        
        # Active jobs
        cursor.execute("""
            SELECT COUNT(*) as active_jobs
            FROM jobs 
            WHERE status = 'active' AND created_at BETWEEN %s AND %s
        """, (start_date, end_date))
        active_jobs = cursor.fetchone()['active_jobs']
        
        # Total unique candidates
        cursor.execute("""
            SELECT COUNT(DISTINCT candidate_id) as total_candidates
            FROM applications 
            WHERE created_at BETWEEN %s AND %s
        """, (start_date, end_date))
        total_candidates = cursor.fetchone()['total_candidates']
        
        # Hired candidates
        cursor.execute("""
            SELECT COUNT(*) as hired_candidates
            FROM applications 
            WHERE status = 'hired' AND created_at BETWEEN %s AND %s
        """, (start_date, end_date))
        hired_candidates = cursor.fetchone()['hired_candidates']
        
        # Average time to hire (in days)
        cursor.execute("""
            SELECT AVG(DATEDIFF(updated_at, created_at)) as avg_time_to_hire
            FROM applications 
            WHERE status = 'hired' AND created_at BETWEEN %s AND %s
        """, (start_date, end_date))
        result = cursor.fetchone()
        avg_time_to_hire = float(result['avg_time_to_hire'] or 0)
        
        # Average AI score
        cursor.execute("""
            SELECT AVG(ai_score) as avg_ai_score
            FROM applications 
            WHERE ai_score IS NOT NULL AND created_at BETWEEN %s AND %s
        """, (start_date, end_date))
        result = cursor.fetchone()
        avg_ai_score = float(result['avg_ai_score'] or 0)
        
        # Conversion rate (hired/total applications)
        conversion_rate = (hired_candidates / total_applications * 100) if total_applications > 0 else 0
        
        return {
            "total_applications": total_applications,
            "total_jobs": total_jobs,
            "total_candidates": total_candidates,
            "active_jobs": active_jobs,
            "hired_candidates": hired_candidates,
            "avg_time_to_hire": round(avg_time_to_hire, 1),
            "avg_ai_score": round(avg_ai_score, 1),
            "conversion_rate": round(conversion_rate, 2)
        }
    
    async def _get_applications_timeline(self, cursor, start_date: datetime, end_date: datetime) -> ChartData:
        """Get applications timeline chart data"""
        cursor.execute("""
            SELECT 
                DATE(created_at) as application_date,
                COUNT(*) as application_count
            FROM applications 
            WHERE created_at BETWEEN %s AND %s
            GROUP BY DATE(created_at)
            ORDER BY application_date
        """, (start_date, end_date))
        
        results = cursor.fetchall()
        
        dates = [str(row['application_date']) for row in results]
        counts = [row['application_count'] for row in results]
        
        return ChartData(
            type="line",
            title="Applications Over Time",
            data={
                "labels": dates,
                "datasets": [{
                    "label": "Applications",
                    "data": counts,
                    "borderColor": "#3b82f6",
                    "backgroundColor": "rgba(59, 130, 246, 0.1)"
                }]
            },
            description="Daily application submission trends"
        )
    
    async def _get_job_performance_chart(self, cursor, start_date: datetime, end_date: datetime) -> ChartData:
        """Get job performance chart data"""
        cursor.execute("""
            SELECT 
                j.title,
                COUNT(a.id) as application_count,
                COUNT(CASE WHEN a.status = 'hired' THEN 1 END) as hired_count,
                AVG(a.ai_score) as avg_score
            FROM jobs j
            LEFT JOIN applications a ON j.id = a.job_id 
                AND a.created_at BETWEEN %s AND %s
            WHERE j.created_at BETWEEN %s AND %s
            GROUP BY j.id, j.title
            ORDER BY application_count DESC
            LIMIT 10
        """, (start_date, end_date, start_date, end_date))
        
        results = cursor.fetchall()
        
        job_titles = [row['title'][:30] + '...' if len(row['title']) > 30 else row['title'] for row in results]
        applications = [row['application_count'] for row in results]
        hired = [row['hired_count'] or 0 for row in results]
        
        return ChartData(
            type="bar",
            title="Top Job Performance",
            data={
                "labels": job_titles,
                "datasets": [
                    {
                        "label": "Applications",
                        "data": applications,
                        "backgroundColor": "#3b82f6"
                    },
                    {
                        "label": "Hired",
                        "data": hired,
                        "backgroundColor": "#10b981"
                    }
                ]
            },
            description="Application and hiring performance by job posting"
        )
    
    async def _get_ai_scores_distribution(self, cursor, start_date: datetime, end_date: datetime) -> ChartData:
        """Get AI scores distribution chart"""
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN ai_score >= 90 THEN '90-100'
                    WHEN ai_score >= 80 THEN '80-89'
                    WHEN ai_score >= 70 THEN '70-79'
                    WHEN ai_score >= 60 THEN '60-69'
                    WHEN ai_score >= 50 THEN '50-59'
                    ELSE 'Below 50'
                END as score_range,
                COUNT(*) as count
            FROM applications 
            WHERE ai_score IS NOT NULL 
                AND created_at BETWEEN %s AND %s
            GROUP BY score_range
            ORDER BY score_range DESC
        """, (start_date, end_date))
        
        results = cursor.fetchall()
        
        labels = [row['score_range'] for row in results]
        data = [row['count'] for row in results]
        
        return ChartData(
            type="pie",
            title="AI Score Distribution",
            data={
                "labels": labels,
                "datasets": [{
                    "data": data,
                    "backgroundColor": [
                        "#10b981", "#3b82f6", "#f59e0b", 
                        "#ef4444", "#8b5cf6", "#6b7280"
                    ]
                }]
            },
            description="Distribution of AI evaluation scores across all applications"
        )
    
    async def _get_application_status_breakdown(self, cursor, start_date: datetime, end_date: datetime) -> ChartData:
        """Get application status breakdown"""
        cursor.execute("""
            SELECT 
                status,
                COUNT(*) as count
            FROM applications 
            WHERE created_at BETWEEN %s AND %s
            GROUP BY status
            ORDER BY count DESC
        """, (start_date, end_date))
        
        results = cursor.fetchall()
        
        labels = [row['status'].replace('_', ' ').title() for row in results]
        data = [row['count'] for row in results]
        
        return ChartData(
            type="doughnut",
            title="Application Status Breakdown",
            data={
                "labels": labels,
                "datasets": [{
                    "data": data,
                    "backgroundColor": [
                        "#3b82f6", "#10b981", "#f59e0b", 
                        "#ef4444", "#8b5cf6", "#6b7280"
                    ]
                }]
            },
            description="Current status distribution of all applications"
        )
    
    async def _generate_ai_insights(self, metrics: Dict, charts: List[ChartData]) -> List[str]:
        """Generate AI insights from data"""
        try:
            prompt = f"""
            Analyze the following recruitment data and provide key insights:
            
            Metrics: {json.dumps(metrics)}
            
            Chart summaries:
            {json.dumps([{"title": chart.title, "description": chart.description} for chart in charts])}
            
            Provide 3-5 key insights about:
            1. Overall recruitment performance
            2. Trends and patterns
            3. Areas of concern or success
            4. Data-driven observations
            
            Keep each insight concise (1-2 sentences) and actionable.
            """
            
            response = self.ai_model.generate_content(prompt)
            insights_text = response.text.strip()
            
            # Split into individual insights
            insights = [insight.strip() for insight in insights_text.split('\n') if insight.strip() and not insight.strip().startswith('#')]
            
            return insights[:5]  # Limit to 5 insights
            
        except Exception as e:
            print(f"Error generating AI insights: {e}")
            return [
                f"Total of {metrics['total_applications']} applications received for {metrics['total_jobs']} jobs",
                f"Average time to hire is {metrics['avg_time_to_hire']} days",
                f"Conversion rate is {metrics['conversion_rate']}%",
                f"Average AI score is {metrics['avg_ai_score']}/100"
            ]
    
    async def _generate_recommendations(self, metrics: Dict, insights: List[str]) -> List[str]:
        """Generate actionable recommendations"""
        try:
            prompt = f"""
            Based on these recruitment metrics and insights, provide actionable recommendations:
            
            Metrics: {json.dumps(metrics)}
            Insights: {json.dumps(insights)}
            
            Provide 3-4 specific, actionable recommendations for:
            1. Improving recruitment efficiency
            2. Enhancing candidate quality
            3. Optimizing the hiring process
            4. Addressing any identified issues
            
            Each recommendation should be specific and implementable.
            """
            
            response = self.ai_model.generate_content(prompt)
            recommendations_text = response.text.strip()
            
            # Split into individual recommendations
            recommendations = [rec.strip() for rec in recommendations_text.split('\n') if rec.strip() and not rec.strip().startswith('#')]
            
            return recommendations[:4]  # Limit to 4 recommendations
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return [
                "Review job posting performance and optimize low-performing listings",
                "Implement strategies to improve candidate conversion rates",
                "Consider adjusting AI evaluation criteria based on hiring outcomes",
                "Focus recruitment efforts on channels with highest quality candidates"
            ]
    
    async def generate_bias_analysis(self, start_date: datetime, end_date: datetime) -> AnalyticsReport:
        """Generate comprehensive bias analysis report"""
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        try:
            # Get candidate demographics data
            cursor.execute("""
                SELECT 
                    cp.user_id,
                    cp.full_name,
                    cp.location,
                    cp.experience_years,
                    cp.skills,
                    cp.ai_score,
                    a.status,
                    a.created_at
                FROM candidate_profile cp
                JOIN applications a ON cp.user_id = a.candidate_id
                WHERE a.created_at BETWEEN %s AND %s
            """, (start_date, end_date))
            
            candidates = cursor.fetchall()
            
            # Analyze bias patterns
            bias_metrics = await self._analyze_bias_patterns(candidates)
            
            # Create bias visualization charts
            charts = await self._create_bias_charts(candidates, cursor, start_date, end_date)
            
            # Generate bias insights
            insights = await self._generate_bias_insights(bias_metrics, candidates)
            recommendations = await self._generate_bias_recommendations(bias_metrics, insights)
            
            return AnalyticsReport(
                report_id=str(uuid.uuid4()),
                report_type="bias_analysis",
                generated_at=datetime.now(),
                date_range={"start": start_date, "end": end_date},
                metrics=bias_metrics,
                charts=charts,
                insights=insights,
                recommendations=recommendations
            )
            
        finally:
            cursor.close()
            connection.close()
    
    async def _analyze_bias_patterns(self, candidates: List[Dict]) -> Dict:
        """Analyze potential bias patterns in hiring data"""
        if not candidates:
            return {"error": "No candidates data available"}
        
        df = pd.DataFrame(candidates)
        
        # Geographic bias analysis
        location_stats = df.groupby('location').agg({
            'ai_score': 'mean',
            'user_id': 'count',
            'status': lambda x: (x == 'hired').sum()
        }).round(2)
        
        # Experience level bias
        df['experience_level'] = pd.cut(df['experience_years'], 
                                      bins=[0, 2, 5, 10, float('inf')],
                                      labels=['Junior', 'Mid', 'Senior', 'Expert'])
        
        experience_stats = df.groupby('experience_level').agg({
            'ai_score': 'mean',
            'user_id': 'count',
            'status': lambda x: (x == 'hired').sum()
        }).round(2)
        
        # AI score bias analysis
        ai_score_quartiles = df['ai_score'].quantile([0.25, 0.5, 0.75]).to_dict()
        
        return {
            "total_candidates_analyzed": len(candidates),
            "location_bias": location_stats.to_dict() if not location_stats.empty else {},
            "experience_bias": experience_stats.to_dict() if not experience_stats.empty else {},
            "ai_score_distribution": ai_score_quartiles,
            "overall_hire_rate": (df['status'] == 'hired').sum() / len(df) * 100
        }
    
    async def _create_bias_charts(self, candidates: List[Dict], cursor, start_date: datetime, end_date: datetime) -> List[ChartData]:
        """Create bias analysis charts"""
        charts = []
        
        if not candidates:
            return charts
        
        df = pd.DataFrame(candidates)
        
        # 1. Hiring rate by experience level
        df['experience_level'] = pd.cut(df['experience_years'], 
                                      bins=[0, 2, 5, 10, float('inf')],
                                      labels=['Junior', 'Mid', 'Senior', 'Expert'])
        
        exp_hiring = df.groupby('experience_level').apply(
            lambda x: (x['status'] == 'hired').sum() / len(x) * 100 if len(x) > 0 else 0
        ).fillna(0)
        
        charts.append(ChartData(
            type="bar",
            title="Hiring Rate by Experience Level",
            data={
                "labels": exp_hiring.index.tolist(),
                "datasets": [{
                    "label": "Hire Rate (%)",
                    "data": exp_hiring.values.tolist(),
                    "backgroundColor": "#3b82f6"
                }]
            },
            description="Comparison of hiring rates across different experience levels"
        ))
        
        # 2. AI Score distribution by location (top 5 locations)
        top_locations = df['location'].value_counts().head(5).index
        location_scores = df[df['location'].isin(top_locations)].groupby('location')['ai_score'].mean()
        
        charts.append(ChartData(
            type="bar",
            title="Average AI Score by Location",
            data={
                "labels": location_scores.index.tolist(),
                "datasets": [{
                    "label": "Average AI Score",
                    "data": location_scores.values.tolist(),
                    "backgroundColor": "#10b981"
                }]
            },
            description="Average AI evaluation scores across different locations"
        ))
        
        return charts
    
    async def _generate_bias_insights(self, bias_metrics: Dict, candidates: List[Dict]) -> List[str]:
        """Generate insights about potential bias"""
        try:
            prompt = f"""
            Analyze this recruitment bias data and identify potential bias patterns:
            
            Bias Metrics: {json.dumps(bias_metrics)}
            Total Candidates: {len(candidates)}
            
            Look for:
            1. Disparities in hiring rates between groups
            2. AI score variations across demographics
            3. Geographic bias patterns
            4. Experience level bias
            
            Provide 3-4 specific insights about potential bias issues or positive patterns.
            Be objective and data-driven.
            """
            
            response = self.ai_model.generate_content(prompt)
            insights_text = response.text.strip()
            
            insights = [insight.strip() for insight in insights_text.split('\n') if insight.strip() and not insight.strip().startswith('#')]
            
            return insights[:4]
            
        except Exception as e:
            print(f"Error generating bias insights: {e}")
            return [
                "Bias analysis completed for recruitment data",
                "Review hiring patterns across different candidate segments",
                "Monitor AI scoring consistency across demographics",
                "Ensure fair representation in candidate evaluation process"
            ]
    
    async def _generate_bias_recommendations(self, bias_metrics: Dict, insights: List[str]) -> List[str]:
        """Generate bias mitigation recommendations"""
        try:
            prompt = f"""
            Based on this bias analysis, provide specific recommendations to improve fairness:
            
            Bias Metrics: {json.dumps(bias_metrics)}
            Insights: {json.dumps(insights)}
            
            Provide 3-4 actionable recommendations for:
            1. Reducing potential bias in the hiring process
            2. Improving diversity and inclusion
            3. Enhancing AI fairness
            4. Implementing bias monitoring
            
            Each recommendation should be specific and implementable.
            """
            
            response = self.ai_model.generate_content(prompt)
            recommendations_text = response.text.strip()
            
            recommendations = [rec.strip() for rec in recommendations_text.split('\n') if rec.strip() and not rec.strip().startswith('#')]
            
            return recommendations[:4]
            
        except Exception as e:
            print(f"Error generating bias recommendations: {e}")
            return [
                "Implement blind resume screening to reduce unconscious bias",
                "Regularly audit AI evaluation criteria for fairness",
                "Expand recruiting channels to improve candidate diversity",
                "Establish bias monitoring dashboards for ongoing oversight"
            ]

analytics_engine = AnalyticsEngine()

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
        "service": "analytics",
        "port": 8007,
        "database": db_status,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/analytics/metrics")
async def get_basic_metrics(
    date_range: DateRange = DateRange.LAST_30_DAYS,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: dict = Depends(verify_token)
):
    """Get basic recruitment metrics"""
    if current_user["user"]["role"] not in ["admin", "recruiter"]:
        raise HTTPException(status_code=403, detail="Access forbidden")
    
    start_dt, end_dt = get_date_range_filter(date_range, start_date, end_date)
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        metrics = await analytics_engine._get_basic_metrics(cursor, start_dt, end_dt)
        
        return MetricsResponse(
            total_applications=metrics["total_applications"],
            total_jobs=metrics["total_jobs"],
            total_candidates=metrics["total_candidates"],
            active_jobs=metrics["active_jobs"],
            hired_candidates=metrics["hired_candidates"],
            avg_time_to_hire=metrics["avg_time_to_hire"],
            avg_ai_score=metrics["avg_ai_score"],
            conversion_rate=metrics["conversion_rate"],
            period=f"{start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}"
        )
        
    finally:
        cursor.close()
        connection.close()

@app.post("/api/v1/analytics/reports/generate")
async def generate_report(
    report_request: ReportRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(verify_token)
):
    """Generate analytics report"""
    if current_user["user"]["role"] not in ["admin", "recruiter"]:
        raise HTTPException(status_code=403, detail="Access forbidden")
    
    start_date, end_date = get_date_range_filter(
        report_request.date_range, 
        report_request.start_date, 
        report_request.end_date
    )
    
    try:
        if report_request.report_type == ReportType.RECRUITMENT_DASHBOARD:
            report = await analytics_engine.generate_recruitment_dashboard(
                start_date, end_date, report_request.filters
            )
        elif report_request.report_type == ReportType.BIAS_ANALYSIS:
            report = await analytics_engine.generate_bias_analysis(start_date, end_date)
        else:
            # Generate basic report for other types
            report = await analytics_engine.generate_recruitment_dashboard(
                start_date, end_date, report_request.filters
            )
        
        # Store report in database
        await store_report(report, current_user["user"]["id"])
        
        # Handle export format
        if report_request.export_format != ExportFormat.JSON:
            background_tasks.add_task(
                export_report, 
                report, 
                report_request.export_format,
                current_user["user"]["id"]
            )
        
        return report
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Report generation failed: {str(e)}"
        )

@app.get("/api/v1/analytics/reports")
async def list_reports(
    report_type: Optional[ReportType] = None,
    limit: int = Query(20, ge=1, le=100),
    current_user: dict = Depends(verify_token)
):
    """List generated reports"""
    if current_user["user"]["role"] not in ["admin", "recruiter"]:
        raise HTTPException(status_code=403, detail="Access forbidden")
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        where_clause = "WHERE generated_by = %s"
        params = [current_user["user"]["id"]]
        
        if report_type:
            where_clause += " AND report_type = %s"
            params.append(report_type.value)
        
        cursor.execute(f"""
            SELECT id, report_type, generated_at, expires_at
            FROM analytics_reports 
            {where_clause}
            ORDER BY generated_at DESC 
            LIMIT %s
        """, params + [limit])
        
        reports = cursor.fetchall()
        
        return {
            "reports": reports,
            "total": len(reports)
        }
        
    finally:
        cursor.close()
        connection.close()

@app.get("/api/v1/analytics/reports/{report_id}")
async def get_report(
    report_id: str,
    current_user: dict = Depends(verify_token)
):
    """Get specific report"""
    if current_user["user"]["role"] not in ["admin", "recruiter"]:
        raise HTTPException(status_code=403, detail="Access forbidden")
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        cursor.execute("""
            SELECT * FROM analytics_reports 
            WHERE id = %s AND generated_by = %s
        """, (report_id, current_user["user"]["id"]))
        
        report = cursor.fetchone()
        
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        # Parse JSON data
        report['report_data'] = json.loads(report['report_data'])
        if report['filters']:
            report['filters'] = json.loads(report['filters'])
        
        return report
        
    finally:
        cursor.close()
        connection.close()

@app.get("/api/v1/analytics/charts/applications-timeline")
async def get_applications_timeline(
    date_range: DateRange = DateRange.LAST_30_DAYS,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: dict = Depends(verify_token)
):
    """Get applications timeline chart"""
    if current_user["user"]["role"] not in ["admin", "recruiter"]:
        raise HTTPException(status_code=403, detail="Access forbidden")
    
    start_dt, end_dt = get_date_range_filter(date_range, start_date, end_date)
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        chart = await analytics_engine._get_applications_timeline(cursor, start_dt, end_dt)
        return chart
        
    finally:
        cursor.close()
        connection.close()

@app.get("/api/v1/analytics/insights")
async def get_ai_insights(
    date_range: DateRange = DateRange.LAST_30_DAYS,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: dict = Depends(verify_token)
):
    """Get AI-generated insights"""
    if current_user["user"]["role"] not in ["admin", "recruiter"]:
        raise HTTPException(status_code=403, detail="Access forbidden")
    
    start_dt, end_dt = get_date_range_filter(date_range, start_date, end_date)
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Get basic metrics and charts
        metrics = await analytics_engine._get_basic_metrics(cursor, start_dt, end_dt)
        applications_chart = await analytics_engine._get_applications_timeline(cursor, start_dt, end_dt)
        
        # Generate insights
        insights = await analytics_engine._generate_ai_insights(metrics, [applications_chart])
        recommendations = await analytics_engine._generate_recommendations(metrics, insights)
        
        return {
            "insights": insights,
            "recommendations": recommendations,
            "generated_at": datetime.now().isoformat(),
            "period": f"{start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}"
        }
        
    finally:
        cursor.close()
        connection.close()

@app.get("/api/v1/analytics/diversity")
async def get_diversity_metrics(
    date_range: DateRange = DateRange.LAST_30_DAYS,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: dict = Depends(verify_token)
):
    """Get diversity and bias metrics"""
    if current_user["user"]["role"] not in ["admin", "recruiter"]:
        raise HTTPException(status_code=403, detail="Access forbidden")
    
    start_dt, end_dt = get_date_range_filter(date_range, start_date, end_date)
    
    try:
        bias_report = await analytics_engine.generate_bias_analysis(start_dt, end_dt)
        return {
            "diversity_metrics": bias_report.metrics,
            "bias_charts": bias_report.charts,
            "insights": bias_report.insights,
            "recommendations": bias_report.recommendations
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Diversity analysis failed: {str(e)}"
        )

@app.get("/api/v1/analytics/export/{report_id}/{format}")
async def export_report(
    report_id: str,
    format: ExportFormat,
    current_user: dict = Depends(verify_token)
):
    """Export report in specified format"""
    if current_user["user"]["role"] not in ["admin", "recruiter"]:
        raise HTTPException(status_code=403, detail="Access forbidden")
    
    # Get report data
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        cursor.execute("""
            SELECT report_data FROM analytics_reports 
            WHERE id = %s AND generated_by = %s
        """, (report_id, current_user["user"]["id"]))
        
        result = cursor.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Report not found")
        
        report_data = json.loads(result['report_data'])
        
        if format == ExportFormat.CSV:
            return export_to_csv(report_data)
        elif format == ExportFormat.PDF:
            return export_to_pdf(report_data)
        elif format == ExportFormat.EXCEL:
            return export_to_excel(report_data)
        else:
            return report_data
        
    finally:
        cursor.close()
        connection.close()

# =============================================================================
# HELPER FUNCTIONS FOR STORAGE AND EXPORT
# =============================================================================

async def store_report(report: AnalyticsReport, user_id: str):
    """Store report in database"""
    connection = get_db_connection()
    cursor = connection.cursor()
    
    try:
        expires_at = datetime.now() + timedelta(days=30)  # Reports expire after 30 days
        
        cursor.execute("""
            INSERT INTO analytics_reports 
            (id, report_type, generated_by, report_data, filters, expires_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            report.report_id,
            report.report_type,
            user_id,
            json.dumps(report.dict()),
            json.dumps({}),  # Filters would be stored here
            expires_at
        ))
        
        connection.commit()
        
    finally:
        cursor.close()
        connection.close()

def export_to_csv(report_data: Dict) -> Dict:
    """Export report data to CSV format"""
    try:
        # Create CSV from metrics
        metrics_df = pd.DataFrame([report_data.get('metrics', {})])
        csv_buffer = io.StringIO()
        metrics_df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        return {
            "content": csv_content,
            "content_type": "text/csv",
            "filename": f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV export failed: {str(e)}")

def export_to_pdf(report_data: Dict) -> Dict:
    """Export report data to PDF format"""
    # This would require a PDF generation library like reportlab
    # For now, return a placeholder
    return {
        "message": "PDF export not implemented yet",
        "content_type": "application/pdf"
    }

def export_to_excel(report_data: Dict) -> Dict:
    """Export report data to Excel format"""
    # This would require openpyxl library
    # For now, return a placeholder
    return {
        "message": "Excel export not implemented yet",
        "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)
