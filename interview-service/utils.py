import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import calendar

class InterviewScheduler:
    """Utility class for interview scheduling logic"""
    
    @staticmethod
    def find_optimal_slots(
        interviewer_availability: Dict,
        existing_interviews: List[Dict],
        duration_minutes: int,
        preferred_times: Optional[List[str]] = None
    ) -> List[Dict]:
        """Find optimal interview slots"""
        
        optimal_slots = []
        current_date = datetime.now().date()
        
        # Look ahead for next 14 days
        for i in range(14):
            check_date = current_date + timedelta(days=i)
            day_of_week = check_date.weekday()
            
            # Skip weekends unless specified
            if day_of_week >= 5:  # Saturday = 5, Sunday = 6
                continue
            
            # Check availability for this day
            if day_of_week in interviewer_availability:
                day_slots = interviewer_availability[day_of_week]
                
                for slot in day_slots:
                    slot_datetime = datetime.combine(
                        check_date, 
                        datetime.strptime(slot['start_time'], '%H:%M').time()
                    )
                    
                    # Check for conflicts
                    has_conflict = any(
                        abs((existing['scheduled_datetime'] - slot_datetime).total_seconds()) < (duration_minutes * 60)
                        for existing in existing_interviews
                    )
                    
                    if not has_conflict:
                        optimal_slots.append({
                            "datetime": slot_datetime.isoformat(),
                            "confidence_score": InterviewScheduler._calculate_confidence(
                                slot_datetime, preferred_times
                            ),
                            "available_interviewers": slot.get('interviewers', [])
                        })
        
        # Sort by confidence score
        optimal_slots.sort(key=lambda x: x['confidence_score'], reverse=True)
        return optimal_slots[:10]  # Return top 10 slots
    
    @staticmethod
    def _calculate_confidence(slot_datetime: datetime, preferred_times: Optional[List[str]]) -> float:
        """Calculate confidence score for a time slot"""
        base_score = 50.0
        
        # Prefer business hours (9 AM - 5 PM)
        hour = slot_datetime.hour
        if 9 <= hour <= 17:
            base_score += 30
        elif 8 <= hour <= 18:
            base_score += 20
        
        # Prefer mid-week (Tuesday-Thursday)
        weekday = slot_datetime.weekday()
        if 1 <= weekday <= 3:  # Tue-Thu
            base_score += 20
        elif weekday in [0, 4]:  # Mon, Fri
            base_score += 10
        
        # Consider preferred times if provided
        if preferred_times:
            time_str = slot_datetime.strftime('%H:%M')
            if time_str in preferred_times:
                base_score += 25
        
        return min(base_score, 100.0)

class InterviewAnalytics:
    """Utility class for interview analytics"""
    
    @staticmethod
    def calculate_interview_metrics(interviews: List[Dict]) -> Dict:
        """Calculate various interview metrics"""
        if not interviews:
            return {}
        
        total_interviews = len(interviews)
        completed_interviews = [i for i in interviews if i['status'] == 'completed']
        cancelled_interviews = [i for i in interviews if i['status'] == 'cancelled']
        no_shows = [i for i in interviews if i['status'] == 'no_show']
        
        # Calculate average duration (for completed interviews)
        avg_duration = sum(
            i.get('actual_duration_minutes', i.get('duration_minutes', 0))
            for i in completed_interviews
        ) / len(completed_interviews) if completed_interviews else 0
        
        # Calculate conversion rates
        completion_rate = len(completed_interviews) / total_interviews * 100 if total_interviews > 0 else 0
        no_show_rate = len(no_shows) / total_interviews * 100 if total_interviews > 0 else 0
        cancellation_rate = len(cancelled_interviews) / total_interviews * 100 if total_interviews > 0 else 0
        
        # Interview type distribution
        type_distribution = {}
        for interview in interviews:
            interview_type = interview.get('interview_type', 'unknown')
            type_distribution[interview_type] = type_distribution.get(interview_type, 0) + 1
        
        return {
            "total_interviews": total_interviews,
            "completed_interviews": len(completed_interviews),
            "completion_rate": round(completion_rate, 2),
            "no_show_rate": round(no_show_rate, 2),
            "cancellation_rate": round(cancellation_rate, 2),
            "average_duration_minutes": round(avg_duration, 1),
            "interview_type_distribution": type_distribution
        }
    
    @staticmethod
    def analyze_interviewer_performance(feedback_data: List[Dict]) -> Dict:
        """Analyze interviewer performance from feedback"""
        if not feedback_data:
            return {}
        
        interviewer_stats = {}
        
        for feedback in feedback_data:
            interviewer_id = feedback.get('interviewer_id')
            if interviewer_id not in interviewer_stats:
                interviewer_stats[interviewer_id] = {
                    "total_interviews": 0,
                    "average_ratings": {},
                    "recommendations": {"hire": 0, "reject": 0, "maybe": 0}
                }
            
            stats = interviewer_stats[interviewer_id]
            stats["total_interviews"] += 1
            
            # Calculate average ratings
            rating_fields = ['technical_skills', 'communication_skills', 'problem_solving', 'cultural_fit', 'overall_rating']
            for field in rating_fields:
                if feedback.get(field):
                    if field not in stats["average_ratings"]:
                        stats["average_ratings"][field] = []
                    stats["average_ratings"][field].append(feedback[field])
            
            # Track recommendations
            recommendation = feedback.get('recommendation')
            if recommendation in stats["recommendations"]:
                stats["recommendations"][recommendation] += 1
        
        # Calculate final averages
        for interviewer_id, stats in interviewer_stats.items():
            for field, ratings in stats["average_ratings"].items():
                stats["average_ratings"][field] = round(sum(ratings) / len(ratings), 2) if ratings else 0
        
        return interviewer_stats

class VideoMeetingIntegration:
    """Utility class for video meeting integrations"""
    
    @staticmethod
    def create_zoom_meeting(interview_data: Dict) -> Dict:
        """Create Zoom meeting (placeholder implementation)"""
        # In production, integrate with Zoom API
        meeting_data = {
            "meeting_id": f"zoom_{interview_data['interview_id']}",
            "join_url": f"https://zoom.us/j/{interview_data['interview_id']}",
            "password": "interview123",
            "host_key": "host123"
        }
        return meeting_data
    
    @staticmethod
    def create_teams_meeting(interview_data: Dict) -> Dict:
        """Create Microsoft Teams meeting (placeholder implementation)"""
        # In production, integrate with Microsoft Graph API
        meeting_data = {
            "meeting_id": f"teams_{interview_data['interview_id']}",
            "join_url": f"https://teams.microsoft.com/l/meetup-join/{interview_data['interview_id']}",
            "conference_id": interview_data['interview_id']
        }
        return meeting_data
    
    @staticmethod
    def create_google_meet(interview_data: Dict) -> Dict:
        """Create Google Meet meeting (placeholder implementation)"""
        # In production, integrate with Google Calendar API
        meeting_data = {
            "meeting_id": f"meet_{interview_data['interview_id']}",
            "join_url": f"https://meet.google.com/{interview_data['interview_id']}",
            "phone_numbers": ["+1-234-567-8900"]
        }
        return meeting_data
