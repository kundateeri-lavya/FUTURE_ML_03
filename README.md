# 🔍 Resume / Candidate Screening System

> **Task 3** | ML-based system to automatically screen and rank resumes against a job description using TF-IDF, spaCy NER, and a multi-signal scoring engine.

---

## 📌 Project Overview

This project builds an end-to-end **Resume Screening & Ranking System** using **Natural Language Processing (NLP)** and **Machine Learning (ML)**. Given a job description and a set of resumes, the system:

1. **Parses** each resume to extract structured information (name, skills, experience, education)
2. **Cleans and vectorizes** text using TF-IDF with bi-gram support
3. **Scores** each candidate using a **5-component weighted scoring model**
4. **Ranks** candidates and identifies skill gaps
5. **Generates reports** — CSV, JSON, and 3 visualization charts

---

## 🧠 ML/NLP Techniques Used

| Technique | Purpose |
|-----------|---------|
| **TF-IDF Vectorization** | Represent resume & JD as numerical vectors |
| **Cosine Similarity** | Measure semantic closeness of resume to JD |
| **Named Entity Recognition (spaCy)** | Extract candidate names and organizations |
| **Regex + Pattern Matching** | Extract emails, phones, years of experience |
| **Skill Taxonomy Matching** | Map 80+ technical skills across 6 categories |
| **Custom Scoring Model** | Weighted multi-signal composite score |
| **MinMax Scaling** | Normalize scores to [0, 1] range |

---

## 📊 Scoring Model

```
Final Score = (TF-IDF × 0.35) + (Skill Match × 0.30)
            + (Experience × 0.15) + (Education × 0.10)
            + (Keyword Density × 0.10)
```

| Component | Weight | Description |
|-----------|--------|-------------|
| TF-IDF Cosine Similarity | 35% | Overall text relevance |
| Skill Match Score | 30% | Required + preferred skills matched |
| Experience Score | 15% | Years vs JD minimum |
| Education Score | 10% | Degree tier vs JD requirement |
| Keyword Density | 10% | JD keyword coverage in resume |

---

## 🗂️ Project Structure

```
resume_screening_system/
│
├── src/
│   └── resume_screener.py        # Core system — all 8 modules
│
├── data/
│   └── sample_resumes/           # Place .txt resume files here
│
├── outputs/                      # Auto-generated results
│   ├── ranking_*.csv
│   ├── results_*.json
│   ├── chart_ranking_*.png
│   ├── heatmap_*.png
│   └── skill_gap_*.png
│
├── notebooks/
│   └── analysis.ipynb            # Jupyter exploration notebook
│
├── main.py                       # Entry point
├── requirements.txt
├── setup.sh
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/YOUR_USERNAME/resume-screening-system.git
cd resume-screening-system
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run the System
```bash
python main.py
```

### 3. View Results
Check the `outputs/` folder for:
- `ranking_Data_Scientist.csv` — Full ranked candidate table
- `chart_ranking_*.png` — Visual ranking bar chart
- `heatmap_*.png` — Score component breakdown
- `skill_gap_*.png` — Matched vs missing skills

---

## 📦 Modules Explained

### `TextPreprocessor`
Handles all text normalization: URL/email removal, tokenization, stopword filtering, lemmatization (WordNet) and optional stemming (Porter). Returns clean strings ready for TF-IDF vectorization.

### `ResumeParser`
Extracts structured fields from raw resume text:
- **Name** → spaCy PERSON entities + regex fallback
- **Email/Phone** → Regex
- **Skills** → 80+ skill taxonomy across 6 categories (ML, Web, Cloud, Data, Soft Skills)
- **Experience** → Regex year extraction
- **Education** → Degree pattern matching
- **Certifications** → Known certification name matching
- **Sections** → Regex-based section detection

### `JobDescriptionParser`
Parses JDs for required/preferred skills, minimum experience, education level, and extracts top keywords via TF-IDF for keyword density scoring.

### `ScoringEngine`
The ML core — fits TF-IDF on corpus, computes cosine similarity, skill match ratio (required 70% + preferred 30%), experience ratio, education tier comparison, and keyword density. Combines into weighted final score.

### `ReportGenerator`
Produces:
- Terminal ranking table
- Detailed skill gap breakdown per candidate
- CSV export (pandas)
- JSON structured export
- 3 matplotlib/seaborn charts

### `ResumeScreeningPipeline`
Top-level orchestrator that chains Parse → Score → Report in a single `run()` call.

---

## 📈 Sample Output

```
==========================================================================================
  CANDIDATE RANKING REPORT — Role: Data Scientist
==========================================================================================
Rank  Name                   Final  TF-IDF   Skill    Exp   Edu    Gap%  Recommendation
------------------------------------------------------------------------------------------
1     Rahul Mehta            0.847   0.782   0.934   1.000  1.000   8.3%  🟢 STRONGLY RECOMMENDED
2     Arjun Sharma           0.784   0.731   0.876   1.000  0.900  12.5%  🟢 STRONGLY RECOMMENDED
3     Ananya Singh           0.712   0.668   0.823   1.000  0.900  18.2%  🟢 STRONGLY RECOMMENDED
4     Lavanya Rao            0.634   0.589   0.743   0.250  0.900  23.1%  🟡 RECOMMENDED
...
==========================================================================================
```

---

## 🛠️ Tools & Libraries

- **Python 3.8+**
- **spaCy** — NER for name extraction
- **NLTK** — Tokenization, stopwords, lemmatization
- **scikit-learn** — TF-IDF, cosine similarity, scaling
- **pandas / numpy** — Data handling and computation
- **matplotlib / seaborn** — Visualizations

---

## 🎯 Skills Demonstrated

- End-to-end NLP pipeline design
- Feature extraction from unstructured text
- Custom ML scoring model with domain knowledge
- Object-oriented Python with dataclasses
- Data visualization and reporting
- GitHub-ready project structure with documentation

---

## 👤 Author

**Lavya**  
B.Tech CSE — Avanthi Institute of Engineering and Technology, Hyderabad  
GitHub Portfolio | Placement 2024

---

## 📄 License

MIT License — free to use for educational and portfolio purposes.
