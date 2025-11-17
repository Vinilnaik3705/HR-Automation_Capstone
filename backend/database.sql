CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS resume_files (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    filename VARCHAR(500) NOT NULL,
    file_size INTEGER,
    file_type VARCHAR(10),
    session_id VARCHAR(100),
    processed BOOLEAN DEFAULT FALSE,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS resume_data (
    id SERIAL PRIMARY KEY,
    resume_file_id INTEGER REFERENCES resume_files(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    candidate_name VARCHAR(255),
    candidate_email VARCHAR(255),
    candidate_phone VARCHAR(50),
    skills TEXT,
    extracted_text TEXT,
    raw_parsed_data JSONB,
    interview_status VARCHAR(20),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS upload_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    session_id VARCHAR(100) UNIQUE NOT NULL,
    total_files INTEGER,
    processed_files INTEGER DEFAULT 0,
    failed_files INTEGER DEFAULT 0,
    skills_text TEXT,
    status VARCHAR(50) DEFAULT 'processing',
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS matching_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    jd_text TEXT,
    required_skills TEXT,
    use_faiss BOOLEAN DEFAULT FALSE,
    skill_weight FLOAT DEFAULT 0.5,
    shortlist_count INTEGER DEFAULT 5,
    results JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    user_prompt TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    model_used VARCHAR(50) DEFAULT 'gpt-4o-mini',
    context_used TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

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
);

CREATE TABLE IF NOT EXISTS interview_schedules (
    id SERIAL PRIMARY KEY,
    candidate_name VARCHAR(255) NOT NULL,
    candidate_email VARCHAR(255) NOT NULL,
    interviewer_id INTEGER REFERENCES interviewers(id) ON DELETE CASCADE,
    scheduled_time TIMESTAMP NOT NULL,
    duration_minutes INTEGER DEFAULT 30,
    google_event_id VARCHAR(255),
    status VARCHAR(50) DEFAULT 'scheduled',
    feedback_submitted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS interview_feedback (
    id SERIAL PRIMARY KEY,
    interview_id INTEGER REFERENCES interview_schedules(id) ON DELETE CASCADE,
    interviewer_id INTEGER REFERENCES interviewers(id) ON DELETE CASCADE,
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
);

CREATE TABLE IF NOT EXISTS app_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    action VARCHAR(255) NOT NULL,
    description TEXT,
    ip_address VARCHAR(100),
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scheduling_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    action TEXT,
    ip_address VARCHAR(100),
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS job_descriptions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    required_skills TEXT,
    experience_level VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS candidate_status (
    id SERIAL PRIMARY KEY,
    resume_data_id INTEGER REFERENCES resume_data(id) ON DELETE CASCADE,
    status VARCHAR(50) NOT NULL,
    notes TEXT,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_resume_files_user_id ON resume_files(user_id);
CREATE INDEX IF NOT EXISTS idx_resume_data_user_id ON resume_data(user_id);
CREATE INDEX IF NOT EXISTS idx_resume_data_email ON resume_data(candidate_email);
CREATE INDEX IF NOT EXISTS idx_interview_schedules_interviewer_id ON interview_schedules(interviewer_id);
CREATE INDEX IF NOT EXISTS idx_interview_schedules_scheduled_time ON interview_schedules(scheduled_time);
CREATE INDEX IF NOT EXISTS idx_chat_history_user_id ON chat_history(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_created_at ON chat_history(created_at);
CREATE INDEX IF NOT EXISTS idx_upload_sessions_user_id ON upload_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_matching_sessions_user_id ON matching_sessions(user_id);
