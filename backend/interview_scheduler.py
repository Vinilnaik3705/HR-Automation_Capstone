import os
import json
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import uuid
from typing import List, Dict, Any, Optional
import toml

try:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    GOOGLE_CALENDAR_AVAILABLE = True
except ImportError:
    GOOGLE_CALENDAR_AVAILABLE = False

try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.mime.base import MIMEBase
    from email import encoders

    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

try:
    from zoomus import ZoomClient

    ZOOM_AVAILABLE = True
except ImportError:
    ZOOM_AVAILABLE = False


class InterviewScheduler:
    def __init__(self):
        self.setup_google_calendar()
        self.setup_email_config()
        self.setup_zoom_config()

    def setup_google_calendar(self):
        self.calendar_service = None
        self.credentials = None

        if not GOOGLE_CALENDAR_AVAILABLE:
            st.warning("📅 Google Calendar API not available - using database scheduling")
            return

        try:
            if os.path.exists("secrets.toml"):
                secrets = toml.load("secrets.toml")
                google_config = secrets.get('google_calendar', {})

                if not google_config:
                    return

            client_id = google_config.get('client_id')
            client_secret = google_config.get('client_secret')

            if not client_id or not client_secret:
                st.warning("⚠️ Google Calendar credentials incomplete - using database scheduling")
                return

            SCOPES = ['https://www.googleapis.com/auth/calendar']
            creds = None
            token_file = 'token.json'

            if os.path.exists(token_file):
                creds = Credentials.from_authorized_user_file(token_file, SCOPES)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    from google_auth_oauthlib.flow import Flow

                    flow = Flow.from_client_config(
                        {
                            "web": {
                                "client_id": client_id,
                                "client_secret": client_secret,
                                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                                "token_uri": "https://oauth2.googleapis.com/token"
                            }
                        },
                        scopes=SCOPES
                    )

                    flow.redirect_uri = 'http://localhost:8080/'

                    auth_url, _ = flow.authorization_url(prompt='consent')
                    st.info(f"🔗 Please authorize Google Calendar: [Click here]({auth_url})")
                    st.stop()

            self.credentials = creds
            self.calendar_service = build('calendar', 'v3', credentials=creds)
            st.success("✅ Google Calendar connected!")

        except Exception as e:
            st.warning(f"📅 Using database scheduling: {e}")

    def setup_email_config(self):
        self.email_config = {}

        if os.path.exists("secrets.toml"):
            secrets = toml.load("secrets.toml")
            self.email_config = secrets.get('email', {})

    def setup_zoom_config(self):
        self.zoom_config = {}
        self.zoom_client = None

        if os.path.exists("secrets.toml"):
            secrets = toml.load("secrets.toml")
            self.zoom_config = secrets.get('zoom', {})

            if ZOOM_AVAILABLE and self.zoom_config:
                try:
                    client_id = self.zoom_config.get('client_id')
                    client_secret = self.zoom_config.get('client_secret')
                    account_id = self.zoom_config.get('account_id')

                    if client_id and client_secret:
                        self.zoom_client = ZoomClient(client_id, client_secret, account_id)
                except Exception as e:
                    st.warning(f"Zoom configuration failed: {e}")

    def get_db_connection(self):
        try:
            if os.path.exists("secrets.toml"):
                secrets = toml.load("secrets.toml")
                db_config = secrets.get('database', {})
            else:
                db_config = {}

            conn = psycopg2.connect(
                host=db_config.get('host', os.getenv('DB_HOST', 'localhost')),
                database=db_config.get('name', os.getenv('DB_NAME', 'resume_analyzer')),
                user=db_config.get('user', os.getenv('DB_USER', 'postgres')),
                password=db_config.get('password', os.getenv('DB_PASSWORD', 'password')),
                port=db_config.get('port', os.getenv('DB_PORT', '5432'))
            )
            return conn
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            return None

    def get_available_interviewers(self):
        conn = self.get_db_connection()
        if not conn:
            return []

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, name, email, calendar_id, timezone, working_hours_start, 
                           working_hours_end, buffer_between_interviews_minutes
                    FROM interviewers 
                    WHERE is_active = TRUE
                """)
                return cur.fetchall()
        except Exception as e:
            st.error(f"Error fetching interviewers: {e}")
            return []
        finally:
            conn.close()

    def get_interviewer_availability(self, interviewer_id: int, date: datetime):
        if not self.calendar_service:
            return self.get_default_availability(date)

        try:
            interviewer = self.get_interviewer_by_id(interviewer_id)
            if not interviewer:
                return []

            calendar_id = interviewer.get('calendar_id', 'primary')

            start_time = datetime.combine(date, datetime.strptime(interviewer.get('working_hours_start', '09:00'),
                                                                  '%H:%M').time())
            end_time = datetime.combine(date, datetime.strptime(interviewer.get('working_hours_end', '17:00'),
                                                                '%H:%M').time())

            events_result = self.calendar_service.events().list(
                calendarId=calendar_id,
                timeMin=start_time.isoformat() + 'Z',
                timeMax=end_time.isoformat() + 'Z',
                singleEvents=True,
                orderBy='startTime'
            ).execute()

            events = events_result.get('items', [])

            available_slots = []
            current_time = start_time
            buffer_minutes = interviewer.get('buffer_between_interviews_minutes', 15)

            while current_time < end_time:
                slot_end = current_time + timedelta(minutes=30)

                conflict = False
                for event in events:
                    event_start = datetime.fromisoformat(
                        event['start'].get('dateTime', event['start'].get('date')).replace('Z', '+00:00'))
                    event_end = datetime.fromisoformat(
                        event['end'].get('dateTime', event['end'].get('date')).replace('Z', '+00:00'))

                    if (current_time < event_end and slot_end > event_start):
                        conflict = True
                        break

                if not conflict:
                    available_slots.append({
                        'start': current_time,
                        'end': slot_end,
                        'formatted': current_time.strftime('%I:%M %p')
                    })

                current_time += timedelta(minutes=30 + buffer_minutes)

            return available_slots

        except Exception as e:
            st.error(f"Error fetching availability: {e}")
            return self.get_default_availability(date)

    def get_default_availability(self, date: datetime):
        slots = []
        start_time = datetime.combine(date, datetime.strptime('09:00', '%H:%M').time())
        end_time = datetime.combine(date, datetime.strptime('17:00', '%H:%M').time())

        current_time = start_time
        while current_time < end_time:
            slot_end = current_time + timedelta(minutes=30)
            slots.append({
                'start': current_time,
                'end': slot_end,
                'formatted': current_time.strftime('%I:%M %p')
            })
            current_time += timedelta(minutes=45)

        return slots

    def get_interviewer_by_id(self, interviewer_id: int):
        conn = self.get_db_connection()
        if not conn:
            return None

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM interviewers WHERE id = %s
                """, (interviewer_id,))
                return cur.fetchone()
        except Exception as e:
            st.error(f"Error fetching interviewer: {e}")
            return None
        finally:
            conn.close()

    def get_meeting_platforms(self):
        platforms = []

        if self.zoom_client:
            platforms.append(("zoom", "🔴 Zoom Meeting"))

        platforms.append(("teams", "🔵 Microsoft Teams"))

        platforms.append(("generic", "📧 Generic - Link to be shared later"))

        return platforms

    def create_meeting_link(self, candidate_name: str, interviewer: Dict, slot: Dict, duration: int,
                            meeting_platform: str):
        try:
            if meeting_platform == "zoom" and self.zoom_client:
                return self.create_zoom_meeting(candidate_name, interviewer, slot, duration)

            elif meeting_platform == "teams":
                return self.generate_teams_link(candidate_name, slot)

            elif meeting_platform == "generic":
                return self.generate_generic_meeting_info()

            else:
                return self.generate_generic_meeting_info()

        except Exception as e:
            st.error(f"Meeting creation failed: {e}")
            return self.generate_generic_meeting_info()

    def create_zoom_meeting(self, candidate_name: str, interviewer: Dict, slot: Dict, duration: int):
        try:
            if not self.zoom_client:
                return None

            users_response = self.zoom_client.user.list()
            if users_response.status_code != 200:
                return self.generate_generic_meeting_info()

            users_data = users_response.json()
            if not users_data.get('users'):
                return self.generate_generic_meeting_info()

            user_id = users_data['users'][0]['id']

            meeting_details = {
                "topic": f"Interview: {candidate_name}",
                "type": 2,
                "duration": duration,
                "timezone": interviewer.get('timezone', 'UTC'),
                "settings": {
                    "host_video": True,
                    "participant_video": True,
                    "join_before_host": False,
                    "mute_upon_entry": True,
                    "waiting_room": True,
                }
            }

            response = self.zoom_client.meeting.create(user_id=user_id, **meeting_details)

            if response.status_code == 201:
                meeting_data = response.json()
                return meeting_data.get('join_url')
            else:
                return self.generate_generic_meeting_info()

        except Exception as e:
            return self.generate_generic_meeting_info()
    def _ensure_datetime(self, time_value):
        if isinstance(time_value, datetime):
            return time_value
        elif isinstance(time_value, str):
            if 'Z' in time_value:
                return datetime.fromisoformat(time_value.replace('Z', '+00:00'))
            else:
                return datetime.fromisoformat(time_value)
        else:
            raise ValueError(f"Unsupported time format: {type(time_value)}")

    def _format_datetime_for_display(self, time_value):
        dt = self._ensure_datetime(time_value)
        return {
            'date': dt.strftime('%B %d, %Y'),
            'time': dt.strftime('%I:%M %p'),
            'iso': dt.isoformat()
        }

    def generate_teams_link(self, candidate_name: str, slot: Dict):
        try:
            teams_config = self.email_config.get('teams', {})
            teams_link = teams_config.get('meeting_link', '')

            if teams_link:
                return teams_link
            else:
                meeting_id = f"interview-{uuid.uuid4().hex[:12]}"
                return f"https://teams.microsoft.com/l/meetup-join/0/0?subject=Interview with {candidate_name}"

        except Exception as e:
            return "Microsoft Teams meeting link will be shared separately"

    def generate_generic_meeting_info(self):
        return {
            'type': 'generic',
            'message': "The meeting link will be sent to you separately by the interviewer. Please ensure you have a stable internet connection and are ready to join 5 minutes before the scheduled time."
        }

    def schedule_interview(self, candidate_email: str, candidate_name: str,
                           interviewer_id: int, slot: Dict, interview_duration: int = 30,
                           meeting_platform: str = "generic"):
        try:
            interviewer = self.get_interviewer_by_id(interviewer_id)
            if not interviewer:
                return False, "Interviewer not found"

            meeting_link = self.create_meeting_link(candidate_name, interviewer, slot, interview_duration,
                                                    meeting_platform)

            event_id = None
            if self.calendar_service:
                event_id = self.create_calendar_event(
                    candidate_name, candidate_email, interviewer, slot, interview_duration, meeting_link
                )

            conn = self.get_db_connection()
            if conn:
                with conn.cursor() as cur:
                    scheduled_time = self._ensure_datetime(slot['start'])

                    cur.execute("""
                        INSERT INTO interview_schedules 
                        (candidate_name, candidate_email, interviewer_id, scheduled_time, 
                         duration_minutes, google_event_id, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        candidate_name,
                        candidate_email,
                        interviewer_id,
                        scheduled_time,
                        interview_duration,
                        event_id,
                        'scheduled'
                    ))

                    schedule_id = cur.fetchone()[0]
                    conn.commit()

            if self.email_config:
                email_sent = self.send_interview_invitation(
                    candidate_name, candidate_email, interviewer, slot, interview_duration, meeting_link,
                    meeting_platform
                )

                if not email_sent:
                    st.warning("Interview scheduled but email notification failed")

            self.log_scheduling_action(
                f"Scheduled interview for {candidate_name} with {interviewer['name']} via {meeting_platform}"
            )

            return True, "Interview scheduled successfully with meeting link"

        except Exception as e:
            return False, f"Scheduling failed: {str(e)}"

    def create_calendar_event(self, candidate_name: str, candidate_email: str,
                              interviewer: Dict, slot: Dict, duration: int, meeting_link: str):
        if not self.calendar_service:
            return None

        try:
            start_time = self._ensure_datetime(slot['start'])
            end_time = start_time + timedelta(minutes=duration)

            event = {
                'summary': f'Interview: {candidate_name}',
                'description': f'Interview with {candidate_name} for position.\n\nMeeting Link: {meeting_link}',
                'start': {
                    'dateTime': start_time.isoformat(),
                    'timeZone': interviewer.get('timezone', 'UTC'),
                },
                'end': {
                    'dateTime': end_time.isoformat(),
                    'timeZone': interviewer.get('timezone', 'UTC'),
                },
                'attendees': [
                    {'email': candidate_email},
                    {'email': interviewer['email']}
                ],
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'email', 'minutes': 24 * 60},
                        {'method': 'popup', 'minutes': 30},
                    ],
                },
            }

            if meeting_link:
                if isinstance(meeting_link, dict):
                    event[
                        'description'] += f"\n\nMeeting Details: {meeting_link.get('message', 'Check email for details')}"
                else:
                    event['description'] += f"\n\nJoin Meeting: {meeting_link}"

            calendar_id = interviewer.get('calendar_id', 'primary')
            created_event = self.calendar_service.events().insert(
                calendarId=calendar_id,
                body=event,
                sendUpdates='all'
            ).execute()

            return created_event['id']

        except Exception as e:
            st.error(f"Calendar event creation failed: {e}")
            return None

    def send_interview_invitation(self, candidate_name: str, candidate_email: str,
                                  interviewer: Dict, slot: Dict, duration: int,
                                  meeting_link: str, meeting_platform: str):
        if not EMAIL_AVAILABLE or not self.email_config:
            return False

        try:
            smtp_server = self.email_config.get('smtp_server', '')
            smtp_port = self.email_config.get('smtp_port', 587)
            sender_email = self.email_config.get('sender_email', '')
            sender_password = self.email_config.get('sender_password', '')

            if not all([smtp_server, sender_email, sender_password]):
                return False

            message = MIMEMultipart()
            message['From'] = sender_email
            message['To'] = candidate_email
            message['Subject'] = f'Interview Invitation - {candidate_name}'

            body = self._create_email_body(candidate_name, interviewer, slot, duration, meeting_link, meeting_platform)

            message.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(message)

            return True

        except Exception as e:
            st.error(f"Email sending failed: {e}")
            return False

    def _create_email_body(self, candidate_name: str, interviewer: Dict, slot: Dict,
                           duration: int, meeting_link: str, meeting_platform: str):

        time_info = self._format_datetime_for_display(slot['start'])

        base_body = f"""
Dear {candidate_name},

You have been scheduled for an interview with {interviewer['name']}.

Interview Details:
- Date: {time_info['date']}
- Time: {time_info['time']}
- Duration: {duration} minutes
- Interviewer: {interviewer['name']} ({interviewer['email']})
- Meeting Platform: {meeting_platform.replace('_', ' ').title()}
"""

        if meeting_platform == "zoom" and meeting_link:
            meeting_section = f"""
Meeting Link: {meeting_link}

To join the Zoom meeting:
1. Click the link above at the scheduled time
2. Download Zoom client if prompted
3. Test your audio and video before joining
4. Join 5-10 minutes early for setup
"""
        elif meeting_platform == "teams" and meeting_link:
            meeting_section = f"""
Meeting Link: {meeting_link}

To join the Microsoft Teams meeting:
1. Click the link above at the scheduled time
2. Use Teams web app or download the desktop app
3. Allow camera and microphone permissions
4. Join 5 minutes early to test your setup
"""
        elif meeting_platform == "generic":
            if isinstance(meeting_link, dict):
                meeting_section = f"""
Meeting Information:
{meeting_link.get('message', 'The meeting link will be shared separately by the interviewer.')}
"""
            else:
                meeting_section = f"""
Meeting Information:
{meeting_link}
"""
        else:
            meeting_section = f"""
Meeting Information:
The meeting link will be sent to you separately by the interviewer.
"""

        closing = """
Preparation Tips:
- Test your internet connection, microphone, and camera beforehand
- Find a quiet, well-lit place for the interview
- Have your resume and any relevant documents ready
- Prepare questions to ask the interviewer

We look forward to speaking with you!

Best regards,
Recruitment Team
"""

        return base_body + meeting_section + closing

    def reschedule_interview(self, schedule_id: int, new_slot: Dict):
        try:
            conn = self.get_db_connection()
            if not conn:
                return False, "Database connection failed"

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM interview_schedules WHERE id = %s
                """, (schedule_id,))
                schedule = cur.fetchone()

                if not schedule:
                    return False, "Schedule not found"

                interviewer = self.get_interviewer_by_id(schedule['interviewer_id'])

                if schedule['google_event_id'] and self.calendar_service:
                    self.update_calendar_event(
                        schedule['google_event_id'], interviewer, new_slot
                    )

                new_scheduled_time = self._ensure_datetime(new_slot['start'])

                cur.execute("""
                    UPDATE interview_schedules 
                    SET scheduled_time = %s, status = 'rescheduled', updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (new_scheduled_time, schedule_id))
                conn.commit()

            if self.email_config:
                self.send_rescheduling_notification(
                    schedule['candidate_name'], schedule['candidate_email'],
                    interviewer, new_slot
                )

            self.log_scheduling_action(
                f"Rescheduled interview for {schedule['candidate_name']}"
            )

            return True, "Interview rescheduled successfully"

        except Exception as e:
            return False, f"Rescheduling failed: {str(e)}"

    def update_calendar_event(self, event_id: str, interviewer: Dict, new_slot: Dict):
        if not self.calendar_service:
            return

        try:
            event = self.calendar_service.events().get(
                calendarId=interviewer.get('calendar_id', 'primary'),
                eventId=event_id
            ).execute()

            start_time = self._ensure_datetime(new_slot['start'])
            end_time = self._ensure_datetime(new_slot['end'])

            event['start']['dateTime'] = start_time.isoformat()
            event['end']['dateTime'] = end_time.isoformat()

            updated_event = self.calendar_service.events().update(
                calendarId=interviewer.get('calendar_id', 'primary'),
                eventId=event_id,
                body=event,
                sendUpdates='all'
            ).execute()

            return updated_event

        except Exception as e:
            st.error(f"Calendar event update failed: {e}")

    def send_rescheduling_notification(self, candidate_name: str, candidate_email: str,
                                       interviewer: Dict, new_slot: Dict):
        if not EMAIL_AVAILABLE or not self.email_config:
            return

        try:
            time_info = self._format_datetime_for_display(new_slot['start'])

            smtp_server = self.email_config.get('smtp_server', '')
            smtp_port = self.email_config.get('smtp_port', 587)
            sender_email = self.email_config.get('sender_email', '')
            sender_password = self.email_config.get('sender_password', '')

            message = MIMEMultipart()
            message['From'] = sender_email
            message['To'] = candidate_email
            message['Subject'] = f'Interview Rescheduled - {candidate_name}'

            body = f"""
            Dear {candidate_name},

            Your interview has been rescheduled.

            New Interview Details:
            - Date: {time_info['date']}
            - Time: {time_info['time']}
            - Interviewer: {interviewer['name']}

            Please update your calendar accordingly.

            Best regards,
            Recruitment Team
            """

            message.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(message)

        except Exception as e:
            st.error(f"Rescheduling email failed: {e}")

    def cancel_interview(self, schedule_id: int):
        try:
            conn = self.get_db_connection()
            if not conn:
                return False, "Database connection failed"

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM interview_schedules WHERE id = %s
                """, (schedule_id,))
                schedule = cur.fetchone()

                if not schedule:
                    return False, "Schedule not found"

                if schedule['google_event_id'] and self.calendar_service:
                    interviewer = self.get_interviewer_by_id(schedule['interviewer_id'])
                    self.cancel_calendar_event(
                        schedule['google_event_id'], interviewer
                    )

                cur.execute("""
                    UPDATE interview_schedules 
                    SET status = 'cancelled', updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (schedule_id,))
                conn.commit()

            if self.email_config:
                self.send_cancellation_notification(
                    schedule['candidate_name'], schedule['candidate_email']
                )

            self.log_scheduling_action(
                f"Cancelled interview for {schedule['candidate_name']}"
            )

            return True, "Interview cancelled successfully"

        except Exception as e:
            return False, f"Cancellation failed: {str(e)}"

    def cancel_calendar_event(self, event_id: str, interviewer: Dict):
        if not self.calendar_service:
            return

        try:
            self.calendar_service.events().delete(
                calendarId=interviewer.get('calendar_id', 'primary'),
                eventId=event_id,
                sendUpdates='all'
            ).execute()
        except Exception as e:
            st.error(f"Calendar event cancellation failed: {e}")

    def send_cancellation_notification(self, candidate_name: str, candidate_email: str):
        if not EMAIL_AVAILABLE or not self.email_config:
            return

        try:
            smtp_server = self.email_config.get('smtp_server', '')
            smtp_port = self.email_config.get('smtp_port', 587)
            sender_email = self.email_config.get('sender_email', '')
            sender_password = self.email_config.get('sender_password', '')

            message = MIMEMultipart()
            message['From'] = sender_email
            message['To'] = candidate_email
            message['Subject'] = f'Interview Cancelled - {candidate_name}'

            body = f"""
            Dear {candidate_name},

            Your scheduled interview has been cancelled.

            We apologize for any inconvenience. We will contact you if we need to reschedule.

            Best regards,
            Recruitment Team
            """

            message.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(message)

        except Exception as e:
            st.error(f"Cancellation email failed: {e}")

    def get_scheduled_interviews(self):
        conn = self.get_db_connection()
        if not conn:
            return []

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT iss.*, i.name as interviewer_name, i.email as interviewer_email
                    FROM interview_schedules iss
                    JOIN interviewers i ON iss.interviewer_id = i.id
                    WHERE iss.status IN ('scheduled', 'rescheduled')
                    ORDER BY iss.scheduled_time ASC
                """)
                return cur.fetchall()
        except Exception as e:
            st.error(f"Error fetching scheduled interviews: {e}")
            return []
        finally:
            conn.close()

    def log_scheduling_action(self, action: str):
        conn = self.get_db_connection()
        if not conn or 'user' not in st.session_state:
            return

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO scheduling_logs (user_id, action, ip_address, user_agent)
                    VALUES (%s, %s, %s, %s)
                """, (st.session_state.user['id'], action, "N/A", "Streamlit App"))
            conn.commit()
        except Exception as e:
            print(f"Logging error: {e}")
        finally:
            conn.close()


def show_schedule_interview_tab(scheduler: InterviewScheduler):
    st.subheader("Schedule New Interview")

    shortlisted_candidates = []
    if hasattr(st.session_state, 'scored_results') and st.session_state.scored_results:
        shortlisted_candidates = st.session_state.scored_results[:st.session_state.get('shortlist_count', 5)]

    if shortlisted_candidates:
        candidate_options = {f"{c['Name']} ({c['Email']})": c for c in shortlisted_candidates}
        selected_candidate_label = st.selectbox(
            "Select Candidate",
            options=list(candidate_options.keys()),
            help="Choose from shortlisted candidates"
        )
        selected_candidate = candidate_options[selected_candidate_label]
    else:
        col1, col2 = st.columns(2)
        with col1:
            candidate_name = st.text_input("Candidate Name")
        with col2:
            candidate_email = st.text_input("Candidate Email")
        selected_candidate = {'Name': candidate_name, 'Email': candidate_email}

    interviewers = scheduler.get_available_interviewers()
    if not interviewers:
        st.warning("No interviewers configured. Please set up interviewers first.")
        return

    interviewer_options = {f"{i['name']} ({i['email']})": i for i in interviewers}
    selected_interviewer_label = st.selectbox(
        "Select Interviewer",
        options=list(interviewer_options.keys())
    )
    selected_interviewer = interviewer_options[selected_interviewer_label]

    available_platforms = scheduler.get_meeting_platforms()
    platform_options = {display_name: platform_id for platform_id, display_name in available_platforms}

    selected_platform_label = st.selectbox(
        "Meeting Platform",
        options=list(platform_options.keys()),
        help="Select the video conferencing platform for the interview"
    )
    selected_platform = platform_options[selected_platform_label]

    interview_date = st.date_input(
        "Interview Date",
        min_value=datetime.now().date(),
        value=datetime.now().date() + timedelta(days=1)
    )

    available_slots = scheduler.get_interviewer_availability(
        selected_interviewer['id'], interview_date
    )

    if available_slots:
        slot_options = [f"{slot['formatted']}" for slot in available_slots]
        selected_slot_label = st.selectbox("Available Time Slots", options=slot_options)

        selected_slot = next(slot for slot in available_slots if slot['formatted'] == selected_slot_label)

        duration = st.selectbox("Interview Duration", [30, 45, 60], index=0)

        st.info(f"📹 Meeting will be conducted via {selected_platform_label}")

        if st.button("Schedule Interview", type="primary", use_container_width=True):
            if selected_candidate['Name'] and selected_candidate['Email']:
                success, message = scheduler.schedule_interview(
                    selected_candidate['Email'],
                    selected_candidate['Name'],
                    selected_interviewer['id'],
                    selected_slot,
                    duration,
                    selected_platform
                )

                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.error("Please provide candidate name and email")
    else:
        st.warning("No available time slots for selected date. Please choose another date.")


def show_scheduled_interviews_tab(scheduler: InterviewScheduler):
    st.subheader("Scheduled Interviews")

    scheduled_interviews = scheduler.get_scheduled_interviews()

    if not scheduled_interviews:
        st.info("No scheduled interviews found.")
        return

    df_data = []
    for interview in scheduled_interviews:
        df_data.append({
            'ID': interview['id'],
            'Candidate': interview['candidate_name'],
            'Email': interview['candidate_email'],
            'Interviewer': interview['interviewer_name'],
            'Scheduled Time': interview['scheduled_time'].strftime('%Y-%m-%d %I:%M %p'),
            'Duration': f"{interview['duration_minutes']} min",
            'Status': interview['status']
        })

    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)

    if st.button("Export to CSV", use_container_width=True):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="scheduled_interviews.csv",
            mime="text/csv",
            use_container_width=True
        )


def show_reschedule_cancel_tab(scheduler: InterviewScheduler):
    st.subheader("Reschedule or Cancel Interviews")

    scheduled_interviews = scheduler.get_scheduled_interviews()

    if not scheduled_interviews:
        st.info("No scheduled interviews found.")
        return

    interview_options = {
        f"{iv['candidate_name']} with {iv['interviewer_name']} on {iv['scheduled_time'].strftime('%Y-%m-%d %I:%M %p')}": iv
        for iv in scheduled_interviews
    }

    selected_interview_label = st.selectbox(
        "Select Interview to Modify",
        options=list(interview_options.keys())
    )
    selected_interview = interview_options[selected_interview_label]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Reschedule")
        st.write("Select new date and time:")

        interview_date = selected_interview['scheduled_time'].date()
        today = datetime.now().date()

        default_date = max(interview_date, today)

        new_date = st.date_input(
            "New Date",
            min_value=today,
            value=default_date,
            key="reschedule_date"
        )

        available_slots = scheduler.get_interviewer_availability(
            selected_interview['interviewer_id'], new_date
        )

        if available_slots:
            slot_options = [f"{slot['formatted']}" for slot in available_slots]
            new_slot_label = st.selectbox("New Time Slot", options=slot_options)
            new_slot = next(slot for slot in available_slots if slot['formatted'] == new_slot_label)

            if st.button("Reschedule Interview", use_container_width=True):
                success, message = scheduler.reschedule_interview(
                    selected_interview['id'], new_slot
                )

                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        else:
            st.warning("No available slots on selected date")

    with col2:
        st.subheader("Cancel")
        st.info("This action cannot be undone.")

        if st.button("Cancel Interview", type="secondary", use_container_width=True):
            success, message = scheduler.cancel_interview(selected_interview['id'])

            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)


def show_interviewer_setup_tab(scheduler: InterviewScheduler):
    st.subheader("Interviewer Management")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Add New Interviewer")

        with st.form("add_interviewer_form"):
            name = st.text_input("Interviewer Name")
            email = st.text_input("Email")
            calendar_id = st.text_input("Calendar ID (optional)", value="primary")
            timezone = st.selectbox("Timezone",
                                    ["UTC", "US/Eastern", "US/Central", "US/Pacific", "Europe/London", "Asia/Kolkata"])

            col1a, col2a = st.columns(2)
            with col1a:
                working_start = st.text_input("Working Hours Start", value="09:00")
            with col2a:
                working_end = st.text_input("Working Hours End", value="17:00")

            buffer_minutes = st.number_input("Buffer between interviews (minutes)", min_value=0, max_value=60, value=15)
            is_active = st.checkbox("Active", value=True)

            if st.form_submit_button("Add Interviewer", use_container_width=True):
                if name and email:
                    success = save_interviewer_to_db(scheduler, name, email, calendar_id, timezone,
                                                     working_start, working_end, buffer_minutes, is_active)
                    if success:
                        st.success("Interviewer added successfully!")
                    else:
                        st.error("Failed to add interviewer")
                else:
                    st.error("Name and email are required")

    with col2:
        st.subheader("Current Interviewers")
        interviewers = scheduler.get_available_interviewers()

        if interviewers:
            for interviewer in interviewers:
                with st.expander(f"{interviewer['name']} ({interviewer['email']})"):
                    st.write(f"Calendar ID: {interviewer.get('calendar_id', 'primary')}")
                    st.write(f"Timezone: {interviewer.get('timezone', 'UTC')}")
                    st.write(
                        f"Working Hours: {interviewer.get('working_hours_start', '09:00')} - {interviewer.get('working_hours_end', '17:00')}")
                    st.write(f"Status: {'Active' if interviewer.get('is_active', True) else 'Inactive'}")
        else:
            st.info("No interviewers configured")


def save_interviewer_to_db(scheduler: InterviewScheduler, name: str, email: str, calendar_id: str, timezone: str,
                           working_start: str, working_end: str, buffer_minutes: int, is_active: bool):
    conn = None
    try:
        conn = scheduler.get_db_connection()
        if conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO interviewers 
                    (name, email, calendar_id, timezone, working_hours_start, 
                     working_hours_end, buffer_between_interviews_minutes, is_active)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (name, email, calendar_id, timezone, working_start, working_end, buffer_minutes, is_active))
                conn.commit()
                return True
    except Exception as e:
        st.error(f"Error saving interviewer: {e}")
        return False
    finally:
        if conn:
            conn.close()


def show_interview_scheduling_section():
    st.header("📅 Interview Scheduling")

    scheduler = InterviewScheduler()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Schedule Interview",
        "View Scheduled",
        "Reschedule/Cancel",
        "Interviewer Setup"
    ])

    with tab1:
        show_schedule_interview_tab(scheduler)

    with tab2:
        show_scheduled_interviews_tab(scheduler)

    with tab3:
        show_reschedule_cancel_tab(scheduler)

    with tab4:
        show_interviewer_setup_tab(scheduler)


def create_scheduling_tables():
    conn = None
    try:
        scheduler = InterviewScheduler()
        conn = scheduler.get_db_connection()
        if conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS interviewers (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        email VARCHAR(255) NOT NULL UNIQUE,
                        calendar_id VARCHAR(255) DEFAULT 'primary',
                        timezone VARCHAR(50) DEFAULT 'UTC',
                        working_hours_start TIME DEFAULT '09:00',
                        working_hours_end TIME DEFAULT '17:00',
                        buffer_between_interviews_minutes INTEGER DEFAULT 15,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS interview_schedules (
                        id SERIAL PRIMARY KEY,
                        candidate_name VARCHAR(255) NOT NULL,
                        candidate_email VARCHAR(255) NOT NULL,
                        interviewer_id INTEGER REFERENCES interviewers(id),
                        scheduled_time TIMESTAMP NOT NULL,
                        duration_minutes INTEGER DEFAULT 30,
                        google_event_id VARCHAR(255),
                        status VARCHAR(50) DEFAULT 'scheduled',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS scheduling_logs (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER,
                        action TEXT,
                        ip_address VARCHAR(100),
                        user_agent TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                conn.commit()
                st.success("Database tables created successfully!")

    except Exception as e:
        st.error(f"Error creating tables: {e}")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    st.set_page_config(
        page_title="Interview Scheduler",
        page_icon="📅",
        layout="wide"
    )
    show_interview_scheduling_section()