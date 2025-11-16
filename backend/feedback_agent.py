import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import toml
from typing import Dict, List, Any, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


class FeedbackAgent:
    def __init__(self):
        self.email_config = self.load_email_config()

    def load_email_config(self):
        try:
            if os.path.exists("secrets.toml"):
                secrets = toml.load("secrets.toml")
                return secrets.get('email', {})
        except:
            return {}
        return {}

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

    def get_scheduled_interviews(self):
        conn = self.get_db_connection()
        if not conn:
            return []

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        iss.*, 
                        i.name as interviewer_name, 
                        i.email as interviewer_email,
                        COALESCE(iss.candidate_name, rd.candidate_name) as candidate_name,
                        COALESCE(iss.candidate_email, rd.candidate_email) as candidate_email,
                        rd.candidate_phone, 
                        rd.skills
                    FROM interview_schedules iss
                    JOIN interviewers i ON iss.interviewer_id = i.id
                    LEFT JOIN resume_data rd ON LOWER(rd.candidate_email) = LOWER(iss.candidate_email)
                    WHERE iss.status IN ('scheduled', 'completed')
                    ORDER BY iss.scheduled_time ASC
                """)
                return cur.fetchall()
        except Exception as e:
            st.error(f"Error fetching interviews: {e}")
            return []
        finally:
            conn.close()

    def save_feedback(self, interview_id: int, feedback_data: Dict, interviewer_id: int = None):
        conn = self.get_db_connection()
        if not conn:
            return False

        try:
            with conn.cursor() as cur:
                if interviewer_id is None:
                    cur.execute("SELECT interviewer_id FROM interview_schedules WHERE id = %s", (interview_id,))
                    result = cur.fetchone()
                    if result:
                        interviewer_id = result[0]
                    else:
                        return False

                cur.execute("""
                    INSERT INTO interview_feedback 
                    (interview_id, interviewer_id, technical_skills, communication_skills, 
                     problem_solving, cultural_fit, overall_rating, strengths, weaknesses,
                     recommendation, detailed_feedback, structured_feedback)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    interview_id, interviewer_id,
                    feedback_data.get('technical_skills'),
                    feedback_data.get('communication_skills'),
                    feedback_data.get('problem_solving'),
                    feedback_data.get('cultural_fit'),
                    feedback_data.get('overall_rating'),
                    feedback_data.get('strengths'),
                    feedback_data.get('weaknesses'),
                    feedback_data.get('recommendation'),
                    feedback_data.get('detailed_feedback'),
                    json.dumps(feedback_data)
                ))

                feedback_id = cur.fetchone()[0]

                cur.execute("""
                    UPDATE interview_schedules 
                    SET status = 'completed', feedback_submitted = TRUE
                    WHERE id = %s
                """, (interview_id,))

                cur.execute("SELECT candidate_email FROM interview_schedules WHERE id = %s", (interview_id,))
                email_result = cur.fetchone()
                candidate_email = email_result[0] if email_result else None

                if candidate_email:
                    cur.execute("""
                        UPDATE resume_data 
                        SET interview_status = %s, last_updated = CURRENT_TIMESTAMP
                        WHERE LOWER(candidate_email) = LOWER(%s)
                    """, (feedback_data.get('recommendation', 'hold'), candidate_email))

                conn.commit()

                email_sent = self.send_candidate_update_email(interview_id, feedback_data.get('recommendation', 'hold'),
                                                              feedback_data)

                if email_sent:
                    st.toast("📧 Feedback email sent to candidate", icon="✅")
                else:
                    st.toast("⚠️ Feedback saved but email failed", icon="⚠️")

                return True

        except Exception as e:
            st.error(f"Error saving feedback: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def send_candidate_update_email(self, interview_id: int, recommendation: str, feedback_data: Dict):
        if not self.email_config:
            return False

        try:
            conn = self.get_db_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        iss.candidate_name, 
                        iss.candidate_email,
                        i.name as interviewer_name
                    FROM interview_schedules iss
                    JOIN interviewers i ON iss.interviewer_id = i.id
                    WHERE iss.id = %s
                """, (interview_id,))
                interview = cur.fetchone()

            if not interview:
                return False

            candidate_name = interview['candidate_name']
            candidate_email = interview['candidate_email']
            interviewer_name = interview['interviewer_name']

            if recommendation == 'selected':
                subject, body = self._prepare_selection_email(candidate_name, interviewer_name, feedback_data)
            elif recommendation == 'rejected':
                subject, body = self._prepare_rejection_email(candidate_name, interviewer_name, feedback_data)
            else:
                subject, body = self._prepare_hold_email(candidate_name, interviewer_name, feedback_data)

            success = self._send_email_silent(candidate_email, subject, body)

            return success

        except Exception as e:
            return False

    def _send_email_silent(self, to_email: str, subject: str, body: str) -> bool:
        try:
            smtp_server = self.email_config.get('smtp_server', '')
            smtp_port = self.email_config.get('smtp_port', 587)
            sender_email = self.email_config.get('sender_email', '')
            sender_password = self.email_config.get('sender_password', '')

            if not all([smtp_server, sender_email, sender_password]):
                return False

            try:
                smtp_port = int(smtp_port)
            except ValueError:
                return False

            try:
                server = smtplib.SMTP(smtp_server, smtp_port, timeout=10)
                server.starttls()
                server.login(sender_email, sender_password)
                server.quit()
            except Exception:
                return False

            message = MIMEMultipart()
            message['From'] = sender_email
            message['To'] = to_email
            message['Subject'] = subject
            message.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(message)

            return True

        except Exception:
            return False

    def _prepare_selection_email(self, candidate_name: str, interviewer_name: str, feedback_data: Dict) -> tuple:
        subject = f"Congratulations! Next Steps - Interview Update"

        body = f"""
Dear {candidate_name},

We are pleased to inform you that you have been selected to move forward in our hiring process!

Your interview with {interviewer_name} was impressive, and we were particularly impressed with:
{feedback_data.get('strengths', 'Your skills and experience')}

Overall Rating: {feedback_data.get('overall_rating', 'N/A')}/5

Next Steps:
- Our HR team will contact you within 2-3 business days
- They will discuss the offer details and onboarding process
- Please keep an eye on your email for further instructions

We look forward to welcoming you to our team!

Best regards,
Recruitment Team
"""
        return subject, body

    def _prepare_rejection_email(self, candidate_name: str, interviewer_name: str, feedback_data: Dict) -> tuple:
        subject = f"Update on Your Application"

        body = f"""
Dear {candidate_name},

Thank you for taking the time to interview with {interviewer_name} for the position.

After careful consideration, we have decided to move forward with other candidates whose experience more closely matches our current needs.

We appreciate your interest in our company and wish you the best in your job search.

Thank you again for your time and consideration.

Best regards,
Recruitment Team
"""
        return subject, body

    def _prepare_hold_email(self, candidate_name: str, interviewer_name: str, feedback_data: Dict) -> tuple:
        subject = f"Update on Your Application Status"

        body = f"""
Dear {candidate_name},

Thank you for interviewing with {interviewer_name}. We are currently in the process of reviewing all candidates.

Your application is still under consideration, and we will provide you with an update within the next 7-10 business days.

We appreciate your patience during this process.

Best regards,
Recruitment Team
"""
        return subject, body

    def get_feedback_stats(self):
        conn = self.get_db_connection()
        if not conn:
            return {}

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_interviews,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                        COUNT(CASE WHEN status = 'scheduled' THEN 1 END) as scheduled,
                        COUNT(CASE WHEN feedback_submitted = TRUE THEN 1 END) as feedback_submitted
                    FROM interview_schedules
                """)
                stats = cur.fetchone()

                cur.execute("""
                    SELECT rd.interview_status, COUNT(*) as count
                    FROM resume_data rd
                    WHERE rd.interview_status IS NOT NULL
                    GROUP BY rd.interview_status
                """)
                recommendations = cur.fetchall()

                cur.execute("""
                    SELECT 
                        ROUND(AVG(technical_skills), 2) as avg_technical,
                        ROUND(AVG(communication_skills), 2) as avg_communication,
                        ROUND(AVG(problem_solving), 2) as avg_problem_solving,
                        ROUND(AVG(cultural_fit), 2) as avg_cultural_fit,
                        ROUND(AVG(overall_rating), 2) as avg_overall
                    FROM interview_feedback
                """)
                ratings = cur.fetchone()

                return {
                    'overall': stats,
                    'recommendations': recommendations,
                    'ratings': ratings
                }

        except Exception as e:
            st.error(f"Error fetching feedback stats: {e}")
            return {}
        finally:
            conn.close()


def show_feedback_section():
    st.header("📝 Interview Feedback Collection")

    feedback_agent = FeedbackAgent()

    tab1, tab2, tab3 = st.tabs([
        "Submit Feedback",
        "View Interviews",
        "Candidate Status"
    ])

    with tab1:
        show_submit_feedback_tab(feedback_agent)

    with tab2:
        show_view_interviews_tab(feedback_agent)

    with tab3:
        show_candidate_status_tab(feedback_agent)


def show_submit_feedback_tab(feedback_agent: FeedbackAgent):

    interviews = feedback_agent.get_scheduled_interviews()

    if not interviews:
        st.info("No interviews found for feedback submission.")
        return

    completed_interviews = [iv for iv in interviews if iv['status'] == 'completed']
    upcoming_interviews = [iv for iv in interviews if iv['status'] == 'scheduled']

    if upcoming_interviews:
        st.write("### Mark Interview as Completed")
        interview_options = {
            f"{iv['candidate_name']} - {iv['scheduled_time'].strftime('%Y-%m-%d %H:%M')}": iv
            for iv in upcoming_interviews
        }

        selected_interview_label = st.selectbox(
            "Select completed interview:",
            options=list(interview_options.keys())
        )
        selected_interview = interview_options[selected_interview_label]

        if st.button("Mark as Completed", use_container_width=True):
            conn = feedback_agent.get_db_connection()
            if conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE interview_schedules 
                        SET status = 'completed'
                        WHERE id = %s
                    """, (selected_interview['id'],))
                    conn.commit()
                st.success("Interview marked as completed! You can now submit feedback.")
                st.rerun()

    st.write("### Submit Feedback Form")

    feedback_interviews = [iv for iv in interviews if iv['status'] == 'completed']

    if not feedback_interviews:
        st.info("No completed interviews available for feedback.")
        return

    feedback_options = {
        f"{iv['candidate_name']} - {iv['scheduled_time'].strftime('%Y-%m-%d %H:%M')}": iv
        for iv in feedback_interviews
    }

    selected_feedback_label = st.selectbox(
        "Select interview for feedback:",
        options=list(feedback_options.keys())
    )
    selected_feedback = feedback_options[selected_feedback_label]

    with st.form("feedback_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Candidate:** {selected_feedback['candidate_name']}")
            st.write(f"**Interviewer:** {selected_feedback['interviewer_name']}")

        with col2:
            st.write(f"**Interview Date:** {selected_feedback['scheduled_time'].strftime('%Y-%m-%d %H:%M')}")
            st.write(f"**Candidate Email:** {selected_feedback['candidate_email']}")

        st.write("---")

        col1, col2 = st.columns(2)

        with col1:
            technical_skills = st.slider("Technical Skills", 1, 5, 3,
                                         help="Rate candidate's technical proficiency")
            communication_skills = st.slider("Communication Skills", 1, 5, 3,
                                             help="Rate verbal and written communication")

        with col2:
            problem_solving = st.slider("Problem Solving", 1, 5, 3,
                                        help="Rate analytical and problem-solving abilities")
            cultural_fit = st.slider("Cultural Fit", 1, 5, 3,
                                     help="Rate alignment with company culture")

        overall_rating = st.slider("Overall Rating", 1, 5, 3,
                                   help="Overall assessment of the candidate")

        strengths = st.text_area("Key Strengths",
                                 placeholder="What did the candidate do well? Specific skills, achievements...")

        weaknesses = st.text_area("Areas for Improvement",
                                  placeholder="Where can the candidate improve?")

        recommendation = st.selectbox(
            "Recommendation",
            options=["hold", "selected", "rejected"],
            format_func=lambda x: x.capitalize(),
            help="Final hiring recommendation"
        )

        detailed_feedback = st.text_area("Detailed Feedback",
                                         placeholder="Comprehensive feedback, specific examples, interview notes...",
                                         height=150)

        submitted = st.form_submit_button("Submit Feedback", use_container_width=True)

        if submitted:
            with st.spinner("Submitting feedback and sending email..."):
                feedback_data = {
                    'technical_skills': technical_skills,
                    'communication_skills': communication_skills,
                    'problem_solving': problem_solving,
                    'cultural_fit': cultural_fit,
                    'overall_rating': overall_rating,
                    'strengths': strengths,
                    'weaknesses': weaknesses,
                    'recommendation': recommendation,
                    'detailed_feedback': detailed_feedback
                }

                success = feedback_agent.save_feedback(
                    selected_feedback['id'], feedback_data
                )

                if success:
                    st.success("✅ Feedback submitted successfully!")
                else:
                    st.error("Failed to submit feedback")


def show_view_interviews_tab(feedback_agent: FeedbackAgent):
    st.subheader("All Interviews")

    interviews = feedback_agent.get_scheduled_interviews()

    if not interviews:
        st.info("No interviews scheduled.")
        return

    df_data = []
    for iv in interviews:
        df_data.append({
            'ID': iv['id'],
            'Candidate': iv['candidate_name'],
            'Email': iv['candidate_email'],
            'Interviewer': iv['interviewer_name'],
            'Scheduled Time': iv['scheduled_time'].strftime('%Y-%m-%d %H:%M'),
            'Status': iv['status'].capitalize(),
            'Feedback Submitted': 'Yes' if iv.get('feedback_submitted') else 'No'
        })

    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)


def show_candidate_status_tab(feedback_agent: FeedbackAgent):
    st.subheader("Candidate Status Overview")

    conn = feedback_agent.get_db_connection()
    if not conn:
        return

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT candidate_name, candidate_email, candidate_phone, 
                       skills, interview_status, last_updated
                FROM resume_data 
                WHERE interview_status IS NOT NULL
                ORDER BY last_updated DESC
            """)
            candidates = cur.fetchall()

        if not candidates:
            st.info("No candidate status data available.")
            return

        status_counts = {}
        for candidate in candidates:
            status = candidate['interview_status']
            status_counts[status] = status_counts.get(status, 0) + 1

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Candidates", len(candidates))
        with col2:
            st.metric("Selected", status_counts.get('selected', 0))
        with col3:
            st.metric("Rejected", status_counts.get('rejected', 0))
        with col4:
            st.metric("On Hold", status_counts.get('hold', 0))

        st.subheader("Candidate Details")
        df_data = []
        for candidate in candidates:
            df_data.append({
                'Name': candidate['candidate_name'],
                'Email': candidate['candidate_email'],
                'Phone': candidate['candidate_phone'],
                'Skills': candidate['skills'],
                'Status': candidate['interview_status'].capitalize(),
                'Last Updated': candidate['last_updated'].strftime('%Y-%m-%d %H:%M')
            })

        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"Error fetching candidate status: {e}")
    finally:
        conn.close()


def create_feedback_tables():
    conn = None
    try:
        conn = FeedbackAgent().get_db_connection()
        if conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS interview_feedback (
                        id SERIAL PRIMARY KEY,
                        interview_id INTEGER REFERENCES interview_schedules(id),
                        interviewer_id INTEGER REFERENCES interviewers(id),
                        technical_skills INTEGER CHECK (technical_skills >= 1 AND technical_skills <= 5),
                        communication_skills INTEGER CHECK (communication_skills >= 1 AND communication_skills <= 5),
                        problem_solving INTEGER CHECK (problem_solving >= 1 AND problem_solving <= 5),
                        cultural_fit INTEGER CHECK (cultural_fit >= 1 AND cultural_fit <= 5),
                        overall_rating INTEGER CHECK (overall_rating >= 1 AND overall_rating <= 5),
                        strengths TEXT,
                        weaknesses TEXT,
                        recommendation VARCHAR(20) CHECK (recommendation IN ('selected', 'rejected', 'hold')),
                        detailed_feedback TEXT,
                        structured_feedback JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                cur.execute("""
                    ALTER TABLE interview_schedules 
                    ADD COLUMN IF NOT EXISTS feedback_submitted BOOLEAN DEFAULT FALSE
                """)

                cur.execute("""
                    ALTER TABLE resume_data 
                    ADD COLUMN IF NOT EXISTS interview_status VARCHAR(20),
                    ADD COLUMN IF NOT EXISTS last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                """)

                conn.commit()
                st.success("Feedback tables created successfully!")

    except Exception as e:
        st.error(f"Error creating feedback tables: {e}")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    st.set_page_config(page_title="Feedback Agent", layout="wide")
    show_feedback_section()