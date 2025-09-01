# Autonomous Recruitment Agent using GenAI

## 📖 Overview

This project is an intelligent, **autonomous recruitment agent** powered by **Generative AI (GenAI)** and workflow orchestration. It automates the end-to-end hiring lifecycle—from resume screening and interview scheduling to feedback collection and onboarding coordination. The system leverages Large Language Models (LLMs), vector embeddings, and real-world API integrations to create a seamless, efficient, and scalable HR Tech solution.

## 🎯 Key Features

- **🤖 AI-Powered Resume Screening:** Automatically parses, extracts, and scores resumes against job descriptions using vector similarity search.
- **📅 Autonomous Interview Scheduling:** Integrates with calendar APIs (Google/Outlook) to find availability, schedule meetings, and send invites without human intervention.
- **🔄 Automated Feedback Collection:** Collects and parses structured feedback from interviewers, then updates candidate status and triggers communication.
- **📋 Streamlined Onboarding Coordination:** Sends offer letters, collects necessary documents, and nudges candidates to ensure a smooth onboarding process.
- **⚙️ Modular & Orchestrated Workflow:** All agents are integrated into a single, cohesive pipeline managed via a simple web UI.

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **LLM & AI** | OpenAI GPT-4o / GPT-3.5-turbo (Function Calling) |
| **Framework & Agents** | LangChain |
| **Vector Database** | ChromaDB |
| **Embeddings** | OpenAI `text-embedding-ada-002` |
| **Backend API** | FastAPI |
| **Database** | PostgreSQL |
| **Calendar & Email** | Google Workspace API / Gmail API |
| **Frontend (UI)** | Streamlit |
| **Deployment** | Render / Railway |
| **File Parsing** | PyPDF2, spaCy |

## 📁 Project Structure

```
autonomous-recruitment-agent/
├── agents/                 # Core AI Agent Modules
│   ├── resume_screener/
│   ├── scheduler/
│   ├── feedback_collector/
│   └── onboarding_agent/
├── backend/                # FastAPI Application
│   ├── main.py
│   ├── models.py
│   └── api/
├── frontend/               # Streamlit UI Application
│   └── app.py
├── database/               # Database models & migrations
├── scripts/                # Utility scripts & parsers
├── tests/                  # Unit and integration tests
├── requirements.txt        # Python dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- OpenAI API Key
- Google Cloud Project with Calendar and Gmail APIs enabled
- PostgreSQL Database
- (Optional) Pinecone account for alternative vector DB

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/autonomous-recruitment-agent.git
    cd autonomous-recruitment-agent
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables:**
    Create a `.env` file in the root directory and add your credentials:
    ```ini
    OPENAI_API_KEY=your_openai_api_key_here
    DATABASE_URL=your_postgresql_connection_string
    GOOGLE_CREDENTIALS_JSON=path/to/your/google-service-account-key.json
    ```

4.  **Run the Application:**
    - **Start the Backend (FastAPI):**
      ```bash
      cd backend
      uvicorn main:app --reload
      ```
      The API docs will be available at `http://localhost:8000/docs`

    - **Start the Frontend (Streamlit):**
      ```bash
      cd frontend
      streamlit run app.py
      ```
      The UI will be available at `http://localhost:8501`

## 📋 Usage

1.  **Access the Streamlit UI.**
2.  **Upload a Job Description** to set the criteria for screening.
3.  **Upload Candidate Resumes** (PDFs) in bulk. The Resume Screening Agent will parse, score, and shortlist them.
4.  **Schedule Interviews** for shortlisted candidates. The Scheduler Agent will propose times and send calendar invites.
5.  **Collect Feedback** via automated forms sent to interviewers after each meeting.
6.  **Initiate Onboarding** for selected candidates by sending offer letters and document requests.

## 📊 Evaluation Metrics

The system's performance is evaluated based on:
- **Resume Screening Accuracy:** Precision/Recall in matching skills and experience.
- **Scheduling Efficiency:** Reduction in time-to-schedule and number of manual interventions.
- **Feedback Loop Time:** Average time taken to collect and process interviewer feedback.
- **Onboarding Completion Rate:** Percentage of candidates who complete all onboarding steps without manual reminders.

## 👥 Contributors

- D Vinil Naik
- Aarav Raj

## 📄 License

This project is created for academic purposes as part of Capstone Project in NIT Jamshedpur.
