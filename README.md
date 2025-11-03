# **Autonomous Recruitment Agent using GenAI** 🤖

## 📖 Overview

This project is an intelligent, **autonomous recruitment agent** powered by **Generative AI (GenAI)** and workflow orchestration. It automates the end-to-end hiring lifecycle—from resume screening and interview scheduling to feedback collection and candidate management. The system leverages Large Language Models (LLMs), vector embeddings, and real-world API integrations to create a seamless, efficient, and scalable HR Tech solution.

## 🎯 Key Features

### **🤖 AI-Powered Resume Screening**
- Automatically parses, extracts, and scores resumes against job descriptions
- Uses multiple similarity algorithms (embeddings, TF-IDF, Jaccard)
- FAISS integration for high-performance vector similarity search
- Customizable skills matching with configurable weights

### **📅 Autonomous Interview Scheduling**
- **Google Calendar API integration** for automatic event creation
- **Database scheduling fallback** (works without external APIs)
- Smart time slot management with conflict detection
- Automated email invitations to candidates
- Rescheduling and cancellation workflows

### **📝 Automated Feedback Collection** *(New)*
- **Structured evaluation forms** with 5-point rating scales
- **Automated candidate status updates** (Selected/Rejected/Hold)
- **Intelligent email triggers** based on interview outcomes
- **Analytics dashboard** for interview performance metrics
- **Workflow automation** reducing manual follow-ups

### **💬 AI Chat Assistant**
- Natural language queries about candidate profiles
- Context-aware responses using resume content
- Multiple AI model support (GPT-4, GPT-3.5, GPT-4o-mini)
- Conversation history and session management

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Frontend Framework** | Streamlit |
| **Database** | PostgreSQL |
| **AI & LLM** | OpenAI GPT-4/3.5, Sentence Transformers |
| **Vector Search** | FAISS |
| **File Parsing** | PyMuPDF, python-docx |
| **Calendar Integration** | Google Calendar API |
| **Email** | SMTP (Gmail/Outlook) |
| **Data Processing** | Pandas, NumPy, scikit-learn |
| **Configuration** | TOML |

## 📁 Project Structure

```
autonomous-recruitment-agent/
├── app_with_chat.py              # Main Streamlit application
├── interview_scheduler.py        # Interview scheduling module
├── feedback_agent.py            # Feedback collection module
├── requirements.txt             # Python dependencies
├── secrets.toml                # Configuration file
└── README.md                   # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL database
- OpenAI API key
- Google Cloud credentials (optional, for Calendar integration)

### Installation & Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd autonomous-recruitment-agent
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up Environment Variables:**
Create `secrets.toml` with your credentials:
```toml
[database]
host = "localhost"
name = "resume_analyzer"
user = "postgres"
password = "your_password"
port = 5432

[email]
smtp_server = "smtp.gmail.com"
smtp_port = 587
sender_email = "your_email@gmail.com"
sender_password = "your_app_password"

[google_calendar]
client_id = "your_client_id"
client_secret = "your_client_secret"
project_id = "your_project_id"

OPENAI_API_KEY = "your_openai_api_key"
```

4. **Run the Application:**
```bash
streamlit run app_with_chat.py
```

## 📋 Usage Guide

### 1. **Authentication & Setup**
- Register new account or login with existing credentials
- All data is user-specific and secure

### 2. **Resume Processing**
- Navigate to "Upload Resumes" section
- Upload multiple PDF/DOCX files simultaneously
- Monitor real-time processing progress
- View parsed candidate information

### 3. **AI-Powered Job Matching**
- Go to "Job Match" section
- Paste job description and configure matching preferences
- View ranked candidates with detailed score breakdowns
- Export results for further analysis

### 4. **Interview Scheduling** *(New)*
- Access "Interview Scheduling" module
- Select candidates from shortlisted results
- Choose interviewers and available time slots
- System automatically sends invitations and manages calendar

### 5. **Feedback Collection** *(New)*
- Use "Feedback Collection" module post-interview
- Submit structured evaluations with ratings and comments
- System automatically updates candidate status
- Triggers appropriate email communications

### 6. **AI Chat Assistant**
- Ask natural language questions about candidates
- Get insights based on resume content and matching scores
- Support for complex queries across multiple candidates

## 🔄 Complete Workflow

```
Resume Upload → AI Parsing → Database Storage → 
Job Matching → Shortlisting → Interview Scheduling → 
Interview Conducted → Feedback Submission → 
Status Auto-Update → Candidate Notification → Analytics
```

## 📊 System Capabilities

### **Automation Benefits:**
- **80% reduction** in manual resume screening time
- **Standardized** hiring evaluation process
- **Real-time** candidate status tracking
- **Automated** communication workflows
- **Data-driven** hiring decisions

### **Technical Features:**
- Multi-user authentication and data isolation
- Batch processing for high-volume recruitment
- Configurable matching algorithms
- Comprehensive analytics and reporting
- Email integration with multiple providers

## 👥 Contributors

- **D Vinil Naik** - Project Lead & Full Stack Development
- **Aarav Raj** - AI Integration & Backend Development

## 📄 License

This project is created for academic purposes as part of Capstone Project in NIT Jamshedpur.