"""
=============================================================================
main.py — Resume Screening System Entry Point
=============================================================================
Run this file to execute the full pipeline on sample data.
Usage:  python main.py
=============================================================================
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from resume_screener import ResumeScreeningPipeline

# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE DATA: Job Description
# ─────────────────────────────────────────────────────────────────────────────

JOB_DESCRIPTIONS = {
    "Data Scientist": """
Job Title: Data Scientist
Department: AI & Analytics
Location: Hyderabad, India

About the Role:
We are looking for a passionate Data Scientist to join our growing AI team.
You will work on building ML models, NLP solutions, and data pipelines to
drive business decisions at scale.

Required Skills:
- Python (pandas, numpy, scikit-learn)
- Machine Learning (regression, classification, clustering)
- Natural Language Processing (NLP)
- SQL and database management
- TensorFlow or PyTorch
- Data visualization (matplotlib, seaborn, tableau)
- Git version control

Preferred Skills:
- Deep Learning and Neural Networks
- AWS or Azure cloud platforms
- Docker and containerization
- Apache Spark or distributed computing
- REST API development with Flask or FastAPI
- BERT, Transformers, Hugging Face

Experience: 2+ years of experience in data science or ML engineering
Education: B.Tech / B.E. / M.Tech in Computer Science, Data Science, or related field

Key Responsibilities:
- Build and deploy ML models into production systems
- Perform exploratory data analysis on large datasets
- Design and maintain data pipelines
- Collaborate with cross-functional teams on AI-driven features
- Present findings and model results to stakeholders
""",

    "Software Developer": """
Job Title: Software Developer (Full Stack)
Location: Bengaluru, India

Required Skills:
- Python or Java programming
- React or Angular frontend development
- Node.js or Django backend development
- REST API design and development
- SQL and NoSQL databases (MySQL, MongoDB)
- Git, GitHub, CI/CD pipelines
- HTML, CSS, JavaScript

Preferred Skills:
- Docker and Kubernetes
- AWS or GCP cloud services
- TypeScript
- Agile and Scrum methodology
- Redis caching
- Unit testing frameworks

Experience: 1+ years (Freshers with strong projects considered)
Education: Bachelor's degree in Computer Science or equivalent
""",
}

# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE DATA: Resumes (10 candidates with varied profiles)
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RESUMES = [
    {
        "id": "C001",
        "text": """
Arjun Sharma
arjun.sharma@email.com | +91 9876543210 | GitHub: github.com/arjunsharma

SUMMARY
Data Scientist with 3 years of experience in building ML models and NLP pipelines.
Strong background in Python, scikit-learn, TensorFlow, and AWS deployment.

SKILLS
Python, pandas, numpy, scikit-learn, TensorFlow, PyTorch, NLP, BERT, Transformers,
Hugging Face, SQL, PostgreSQL, MongoDB, Docker, AWS, Git, GitHub, matplotlib,
seaborn, tableau, Flask, REST API, deep learning, machine learning, data visualization

EXPERIENCE
Data Scientist — TechMahindra | 2021 – Present | 3 years of experience
- Built NLP sentiment analysis model with 94% accuracy on 1M+ customer reviews
- Developed recommendation engine using collaborative filtering (Python, scikit-learn)
- Deployed ML models on AWS SageMaker using Docker containers
- Reduced model inference time by 40% using feature selection and pruning

Machine Learning Intern — Infosys | 2020 (6 months)
- Built regression model to predict sales revenue with RMSE of 0.12
- Performed EDA on 50K records using pandas and seaborn

EDUCATION
B.Tech in Computer Science — JNTU Hyderabad, 2021 | CGPA: 8.7/10

CERTIFICATIONS
- AWS Certified Machine Learning Specialty
- Google Certified Professional Data Engineer
- TensorFlow Developer Certificate (Coursera)

PROJECTS
1. Resume Screening System — Python, TF-IDF, scikit-learn, spaCy
2. Fake News Detector — BERT, Hugging Face, Flask API
3. Customer Churn Prediction — XGBoost, SHAP, Tableau Dashboard
"""
    },
    {
        "id": "C002",
        "text": """
Priya Nair
priya.nair@gmail.com | +91 8765432109

SUMMARY
Fresh B.Tech graduate with strong foundation in machine learning and Python.
Completed multiple ML projects including image classification and NLP.

SKILLS
Python, pandas, numpy, scikit-learn, matplotlib, seaborn, SQL, MySQL,
machine learning, NLP, natural language processing, git, GitHub, HTML, CSS

EDUCATION
B.Tech Computer Science — Anna University | 2024 | CGPA: 8.2/10

PROJECTS
1. Sentiment Analysis on Twitter Data using NLTK and Logistic Regression (Accuracy: 87%)
2. House Price Prediction using XGBoost and feature engineering
3. Student Performance Management System — Python, SQLite, Tkinter

CERTIFICATIONS
- Machine Learning Specialization — Coursera (Andrew Ng)
- NPTEL Data Science Certificate

EXPERIENCE
No professional experience. 0 years.
"""
    },
    {
        "id": "C003",
        "text": """
Rahul Mehta
rahul.mehta@outlook.com | +91 9988776655

SUMMARY
5 years of experience as a Data Scientist at a fintech startup.
Expert in deep learning, NLP, and production ML systems on Azure.

SKILLS
Python, pandas, numpy, scikit-learn, TensorFlow, PyTorch, Keras, BERT, Transformers,
Hugging Face, deep learning, neural network, NLP, natural language processing,
SQL, PostgreSQL, Cassandra, Spark, Kafka, Docker, Kubernetes, Azure, GCP,
REST API, FastAPI, Git, CI/CD, Airflow, Tableau, Power BI, data pipeline, ETL

EXPERIENCE
Senior Data Scientist — Paytm | 2019 – Present | 5 years experience
- Led team of 4 data scientists building fraud detection models (saved ₹50Cr/yr)
- Built real-time ML inference pipeline using Kafka and Spark Streaming
- Fine-tuned BERT for financial document classification (F1: 0.96)
- Deployed 12 production ML models on Azure Kubernetes Service

Data Analyst — ICICI Bank | 2018–2019 | 1 year
- Built credit risk scoring model using Logistic Regression and XGBoost

EDUCATION
M.Tech in Data Science — IIT Bombay | 2018 | CGPA: 9.1/10
B.Tech in Computer Science — NIT Warangal | 2016 | CGPA: 8.8/10

CERTIFICATIONS
Microsoft Certified: Azure Data Scientist Associate
Google Certified Professional Data Engineer
AWS Certified Machine Learning Specialty
"""
    },
    {
        "id": "C004",
        "text": """
Sneha Kulkarni
sneha.k@yahoo.com

SKILLS
Java, Spring Boot, React, Angular, HTML, CSS, JavaScript, TypeScript,
Node.js, Express, SQL, MySQL, MongoDB, REST API, Git, Docker, AWS,
Agile, Scrum, Jenkins, CI/CD

EXPERIENCE
Software Developer — Wipro | 2022-2024 | 2 years of experience
- Built microservices with Spring Boot and deployed on AWS ECS
- Developed React frontend with Redux state management
- Wrote 80% test coverage using JUnit and Jest

EDUCATION
B.Tech CSE — VIT Vellore | 2022 | CGPA: 7.9/10
"""
    },
    {
        "id": "C005",
        "text": """
Karthik Reddy
karthik.reddy@gmail.com | Hyderabad

OBJECTIVE
Fresher seeking Data Scientist role. Strong interest in ML and AI.

SKILLS
Python, numpy, pandas, matplotlib, basic machine learning,
SQL, MySQL, HTML, CSS, JavaScript

EDUCATION
B.Tech CSE — JNTUH | 2024 | CGPA: 6.9/10

PROJECTS
1. Weather Prediction using Linear Regression
2. Basic Chatbot using NLTK
"""
    },
    {
        "id": "C006",
        "text": """
Divya Patel
divya.patel@email.com | +91 9123456789 | LinkedIn: linkedin.com/in/divyapatel

SUMMARY
Data Engineer with 4 years of experience in building scalable data pipelines,
ETL processes, and ML infrastructure on AWS and GCP.

SKILLS
Python, SQL, PostgreSQL, Apache Spark, Kafka, Airflow, Hadoop, ETL, data pipeline,
AWS, GCP, Docker, Kubernetes, Terraform, Git, CI/CD, pandas, numpy,
scikit-learn, data visualization, tableau, machine learning, TensorFlow

EXPERIENCE
Data Engineer — Amazon | 2020-Present | 4 years experience
- Built ETL pipelines processing 10TB/day using Spark and Airflow
- Designed ML feature store on AWS reducing training time by 60%
- Implemented data quality monitoring using Great Expectations

EDUCATION
B.Tech in Information Technology — Mumbai University | 2020 | CGPA: 8.5/10

CERTIFICATIONS
AWS Certified Data Analytics Specialty
Google Cloud Professional Data Engineer
"""
    },
    {
        "id": "C007",
        "text": """
Mohammed Ali
mohammedali@protonmail.com

SKILLS
Python, R, machine learning, scikit-learn, pandas, numpy, data analysis,
SQL, MySQL, matplotlib, seaborn, NLP, natural language processing,
TensorFlow, Git, GitHub, data visualization

EXPERIENCE
Junior Data Scientist — Startup | 2023-Present | 1.5 years experience
- Implemented text classification pipeline using TF-IDF and Logistic Regression
- Built A/B testing framework to evaluate model performance
- Created interactive dashboards using Plotly and Dash

EDUCATION
M.Sc Statistics — University of Hyderabad | 2022 | CGPA: 8.0/10
B.Sc Mathematics — Osmania University | 2020 | CGPA: 7.8/10
"""
    },
    {
        "id": "C008",
        "text": """
Ananya Singh
ananya.singh@gmail.com | New Delhi

SKILLS
Python, machine learning, deep learning, TensorFlow, PyTorch, computer vision,
OpenCV, YOLO, NLP, BERT, scikit-learn, pandas, numpy, Docker, AWS, Git,
Flask, REST API, data visualization, matplotlib

EXPERIENCE
AI Engineer — Samsung R&D | 2021-Present | 3 years experience
- Developed real-time object detection system using YOLO v8 (mAP: 0.94)
- Built OCR pipeline for document processing using TesseractOCR and BERT
- Optimized model inference to run on edge devices (Raspberry Pi, Jetson Nano)

EDUCATION
B.Tech Electronics & CSE — BITS Pilani | 2021 | CGPA: 9.0/10

CERTIFICATIONS
Deep Learning Specialization — Coursera (deeplearning.ai)
NPTEL Computer Vision and Image Processing
"""
    },
    {
        "id": "C009",
        "text": """
Ravi Kumar
ravikumar123@gmail.com

SKILLS
Excel, Power BI, SQL, basic Python, data analysis, reporting

EXPERIENCE
Business Analyst — Deloitte | 2020-Present | 4 years experience
- Created Power BI dashboards for executive reporting
- Wrote SQL queries for business intelligence reporting
- Managed data governance and documentation

EDUCATION
MBA — IIM Lucknow | 2020
BBA — Delhi University | 2018
"""
    },
    {
        "id": "C010",
        "text": """
Lavanya Rao
lavanya.rao@gmail.com | +91 9000000001 | GitHub: github.com/lavanyarao

SUMMARY
Final year B.Tech CSE student with strong ML portfolio and internship experience.
Passionate about NLP, computer vision, and building production-ready AI systems.

SKILLS
Python, pandas, numpy, scikit-learn, TensorFlow, NLP, natural language processing,
machine learning, deep learning, BERT, SQL, MySQL, Git, GitHub, Flask, REST API,
Docker, AWS basics, matplotlib, seaborn, data visualization, feature extraction

EXPERIENCE
Data Science Intern — HCL Technologies | Summer 2023 | 2 months
- Built text classification model for support ticket routing (Accuracy: 91%)
- Implemented data preprocessing pipeline for 100K+ records

EDUCATION
B.Tech Computer Science — Avanthi Institute of Engineering | 2024 | CGPA: 8.1/10

CERTIFICATIONS
Machine Learning Specialization — Coursera
NPTEL Python for Data Science

PROJECTS
1. Fake News Detection System — Logistic Regression, TF-IDF, Python (Accuracy: 92%)
2. Student Performance Management System — Python, SQLite, Tkinter
3. Resume Screening System — spaCy, scikit-learn, TF-IDF (this project!)
4. Image Classification using CNN — TensorFlow, CIFAR-10 (Accuracy: 89%)
"""
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "🔍 " * 20)
    print("   RESUME / CANDIDATE SCREENING SYSTEM")
    print("   Task 3 | ML + NLP | spaCy · scikit-learn · TF-IDF")
    print("🔍 " * 20)

    pipeline = ResumeScreeningPipeline(output_dir="outputs")

    # Run for Data Scientist role
    scores = pipeline.run(
        jd_title="Data Scientist",
        jd_text=JOB_DESCRIPTIONS["Data Scientist"],
        resume_data=SAMPLE_RESUMES,
        generate_charts=True,
    )

    print(f"\n✅ Done! Results saved in 'outputs/' folder.")
    print("   📄 ranking_Data_Scientist.csv")
    print("   📊 chart_ranking_Data_Scientist.png")
    print("   🔥 heatmap_Data_Scientist.png")
    print("   📉 skill_gap_Data_Scientist.png")
    print("   🗂️  results_Data_Scientist.json")
    print()
    print("=" * 60)
    print("TOP 3 CANDIDATES:")
    for cs in scores[:3]:
        print(f"  #{cs.rank} {cs.name} → {cs.final_score:.3f} — {cs.recommendation}")
    print("=" * 60)


if __name__ == "__main__":
    main()
