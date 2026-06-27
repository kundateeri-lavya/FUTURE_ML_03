"""
=============================================================================
Resume / Candidate Screening System
=============================================================================
Author      : Lavya
Project     : Task 3 - Resume/Candidate Screening System
Tools       : Python, spaCy/NLTK, Scikit-learn
Description : An ML-based system that automatically screens and ranks resumes
              based on a given job role using TF-IDF, cosine similarity,
              skill extraction, and a custom scoring engine.
=============================================================================
"""

import os
import re
import json
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import numpy as np
import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: SETUP & RESOURCE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def initialize_nlp_resources():
    """
    Downloads required NLTK resources if not already present and loads
    the spaCy English model for Named Entity Recognition and POS tagging.
    Returns the spaCy nlp object.
    """
    logger.info("Initializing NLP resources...")
    resources = ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger",
                 "punkt_tab", "omw-1.4"]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            logger.warning(f"Could not download {resource}: {e}")

    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy model loaded successfully.")
    except OSError:
        logger.warning("spaCy model not found. Using basic NLP pipeline.")
        nlp = None

    return nlp


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: DATA CLASSES (Schema Definition)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParsedResume:
    """
    Holds all structured information extracted from a single resume.
    Acts as the core data model throughout the pipeline.
    """
    candidate_id: str
    raw_text: str
    name: str = ""
    email: str = ""
    phone: str = ""
    skills: List[str] = field(default_factory=list)
    experience_years: float = 0.0
    education: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    projects: List[str] = field(default_factory=list)
    cleaned_text: str = ""
    section_texts: Dict[str, str] = field(default_factory=dict)


@dataclass
class JobDescription:
    """
    Structured representation of a job description with required/optional skills
    and weighted criteria used during candidate scoring.
    """
    title: str
    raw_text: str
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    min_experience: float = 0.0
    required_education: str = ""
    keywords: List[str] = field(default_factory=list)
    cleaned_text: str = ""


@dataclass
class CandidateScore:
    """
    Stores all computed scores for a candidate against a specific job.
    Provides a granular breakdown to support interview decision-making.
    """
    candidate_id: str
    name: str
    tfidf_similarity: float = 0.0
    skill_match_score: float = 0.0
    experience_score: float = 0.0
    education_score: float = 0.0
    keyword_density_score: float = 0.0
    final_score: float = 0.0
    rank: int = 0
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    skill_gap_percentage: float = 0.0
    recommendation: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: TEXT CLEANING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

class TextPreprocessor:
    """
    Handles all text normalization steps: noise removal, tokenization,
    stopword filtering, lemmatization and stemming.
    Modular design allows swapping individual steps independently.
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))
        # Domain-specific words to retain even if in stopwords
        self.domain_keep = {"not", "no", "nor", "none", "never", "without"}
        self.stop_words -= self.domain_keep

    def remove_noise(self, text: str) -> str:
        """Strips URLs, emails, special characters and normalizes whitespace."""
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"\S+@\S+", " EMAIL ", text)
        text = re.sub(r"\+?\d[\d\s\-().]{7,}\d", " PHONE ", text)
        text = re.sub(r"[^a-zA-Z0-9\s\+\#\.]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def tokenize_and_filter(self, text: str) -> List[str]:
        """Tokenizes and removes stopwords."""
        tokens = word_tokenize(text.lower())
        return [t for t in tokens if t.isalpha() and t not in self.stop_words and len(t) > 2]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Applies WordNet lemmatization to reduce morphological variation."""
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def stem(self, tokens: List[str]) -> List[str]:
        """Applies Porter stemming (optional, heavier normalization)."""
        return [self.stemmer.stem(t) for t in tokens]

    def clean(self, text: str, use_stemming: bool = False) -> str:
        """
        Full preprocessing pipeline:
        raw text → noise removal → tokenize → stopword removal → lemmatize
        Returns cleaned string ready for vectorization.
        """
        text = self.remove_noise(text)
        tokens = self.tokenize_and_filter(text)
        tokens = self.lemmatize(tokens)
        if use_stemming:
            tokens = self.stem(tokens)
        return " ".join(tokens)

    def extract_sections(self, raw_text: str) -> Dict[str, str]:
        """
        Identifies common resume sections using regex headers.
        Returns a dict mapping section names to their content blocks.
        Useful for targeted feature extraction (e.g., skills section only).
        """
        section_patterns = {
            "summary": r"(?i)(summary|objective|profile|about me)[:\n]+(.*?)(?=\n[A-Z]{3,}|\Z)",
            "experience": r"(?i)(experience|work history|employment)[:\n]+(.*?)(?=\n[A-Z]{3,}|\Z)",
            "education": r"(?i)(education|academic|qualification)[:\n]+(.*?)(?=\n[A-Z]{3,}|\Z)",
            "skills": r"(?i)(skills|technical skills|core competencies)[:\n]+(.*?)(?=\n[A-Z]{3,}|\Z)",
            "projects": r"(?i)(projects|portfolio|works)[:\n]+(.*?)(?=\n[A-Z]{3,}|\Z)",
            "certifications": r"(?i)(certifications|certificates|courses)[:\n]+(.*?)(?=\n[A-Z]{3,}|\Z)",
        }
        sections = {}
        for section, pattern in section_patterns.items():
            match = re.search(pattern, raw_text, re.DOTALL)
            if match:
                sections[section] = match.group(2).strip()[:800]
        return sections


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: RESUME PARSER
# ─────────────────────────────────────────────────────────────────────────────

class ResumeParser:
    """
    Extracts structured fields from raw resume text.
    Uses regex for contact info, spaCy NER for names/orgs,
    and a comprehensive skill dictionary for skill extraction.
    """

    # Master skill taxonomy — covers Data Science, Web Dev, Cloud, and Core CS
    SKILL_TAXONOMY = {
        "programming": [
            "python", "java", "c++", "c#", "javascript", "typescript", "r", "scala",
            "kotlin", "swift", "go", "rust", "php", "ruby", "perl", "matlab"
        ],
        "ml_ai": [
            "machine learning", "deep learning", "neural network", "nlp",
            "natural language processing", "computer vision", "reinforcement learning",
            "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost", "lightgbm",
            "bert", "transformers", "hugging face", "opencv", "yolo"
        ],
        "data": [
            "sql", "mysql", "postgresql", "mongodb", "cassandra", "redis",
            "elasticsearch", "hadoop", "spark", "hive", "kafka", "airflow",
            "pandas", "numpy", "matplotlib", "seaborn", "tableau", "power bi",
            "data analysis", "data visualization", "etl", "data pipeline"
        ],
        "web": [
            "html", "css", "react", "angular", "vue", "node.js", "express",
            "django", "flask", "fastapi", "spring boot", "rest api", "graphql",
            "bootstrap", "tailwind"
        ],
        "cloud_devops": [
            "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
            "jenkins", "git", "github", "gitlab", "ci/cd", "terraform",
            "ansible", "linux", "bash", "shell scripting"
        ],
        "soft": [
            "communication", "teamwork", "leadership", "problem solving",
            "agile", "scrum", "project management", "analytical"
        ]
    }

    ALL_SKILLS = [skill for group in SKILL_TAXONOMY.values() for skill in group]

    def __init__(self, nlp=None):
        self.nlp = nlp
        self.preprocessor = TextPreprocessor()

    def extract_email(self, text: str) -> str:
        match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
        return match.group(0) if match else ""

    def extract_phone(self, text: str) -> str:
        match = re.search(r"\+?[\d][\d\s\-().]{7,}[\d]", text)
        return match.group(0).strip() if match else ""

    def extract_name(self, text: str) -> str:
        """
        Tries spaCy PERSON entities first; falls back to first capitalized
        two-word sequence on first 3 lines of the resume.
        """
        if self.nlp:
            doc = self.nlp(text[:500])
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    return ent.text.strip()
        # Fallback: first line with 2+ capitalized words
        for line in text.split("\n")[:5]:
            line = line.strip()
            words = line.split()
            if 1 < len(words) <= 5 and all(w[0].isupper() for w in words if w.isalpha()):
                return line
        return "Unknown"

    def extract_skills(self, text: str) -> List[str]:
        """
        Scans full resume text for all skills in the master taxonomy.
        Uses lowercase matching and handles multi-word skills (e.g., 'machine learning').
        """
        text_lower = text.lower()
        found = []
        for skill in self.ALL_SKILLS:
            pattern = r"\b" + re.escape(skill) + r"\b"
            if re.search(pattern, text_lower):
                found.append(skill)
        return list(set(found))

    def extract_experience_years(self, text: str) -> float:
        """
        Parses experience years from patterns like:
        '3 years of experience', '2+ years', 'experienced for 5 years'
        Returns the maximum found value as a float.
        """
        patterns = [
            r"(\d+\.?\d*)\s*\+?\s*years?\s*(?:of\s+)?(?:experience|exp|work)",
            r"experience\s*(?:of\s+)?(\d+\.?\d*)\s*\+?\s*years?",
            r"(\d+\.?\d*)\s*\+?\s*yrs?\s*(?:of\s+)?(?:experience|exp)",
        ]
        years = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            years.extend([float(m) for m in matches])
        return max(years) if years else 0.0

    def extract_education(self, text: str) -> List[str]:
        """Finds degree mentions using common academic abbreviations."""
        degree_patterns = [
            r"\b(B\.?Tech|B\.?E\.?|B\.?Sc\.?|M\.?Tech|M\.?E\.?|M\.?Sc\.?|MBA|PhD|BCA|MCA)\b",
            r"\b(Bachelor|Master|Doctor|Graduate|Undergraduate)\b"
        ]
        degrees = []
        for pattern in degree_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            degrees.extend(matches)
        return list(set(degrees))

    def extract_certifications(self, text: str) -> List[str]:
        """Extracts known certification names from resume text."""
        cert_keywords = [
            "aws certified", "google certified", "microsoft certified", "oracle certified",
            "pmp", "cissp", "ceh", "comptia", "coursera", "udemy", "nptel",
            "tensorflow developer", "azure fundamentals", "data science professional"
        ]
        found = []
        text_lower = text.lower()
        for cert in cert_keywords:
            if cert in text_lower:
                found.append(cert)
        return found

    def parse(self, candidate_id: str, raw_text: str) -> ParsedResume:
        """
        Master parse function — runs all extractors in order and
        returns a fully populated ParsedResume dataclass.
        """
        resume = ParsedResume(candidate_id=candidate_id, raw_text=raw_text)
        resume.name = self.extract_name(raw_text)
        resume.email = self.extract_email(raw_text)
        resume.phone = self.extract_phone(raw_text)
        resume.skills = self.extract_skills(raw_text)
        resume.experience_years = self.extract_experience_years(raw_text)
        resume.education = self.extract_education(raw_text)
        resume.certifications = self.extract_certifications(raw_text)
        resume.section_texts = self.preprocessor.extract_sections(raw_text)
        resume.cleaned_text = self.preprocessor.clean(raw_text)
        return resume


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: JOB DESCRIPTION PARSER
# ─────────────────────────────────────────────────────────────────────────────

class JobDescriptionParser:
    """
    Parses job descriptions to extract structured requirements:
    required/preferred skills, experience thresholds, education,
    and high-frequency TF-IDF keywords used during scoring.
    """

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.resume_parser = ResumeParser()

    def extract_required_skills(self, text: str) -> List[str]:
        """Extracts skills from 'Required' or 'Must have' sections."""
        req_section = re.search(
            r"(?i)(required skills?|must have|mandatory)[:\n]+(.*?)(?=preferred|nice to have|\Z)",
            text, re.DOTALL
        )
        if req_section:
            return self.resume_parser.extract_skills(req_section.group(2))
        return self.resume_parser.extract_skills(text)

    def extract_preferred_skills(self, text: str) -> List[str]:
        """Extracts skills from 'Preferred' or 'Nice to have' sections."""
        pref_section = re.search(
            r"(?i)(preferred|nice to have|good to have)[:\n]+(.*?)(?=\n[A-Z]{3,}|\Z)",
            text, re.DOTALL
        )
        if pref_section:
            return self.resume_parser.extract_skills(pref_section.group(2))
        return []

    def extract_min_experience(self, text: str) -> float:
        """Parses minimum experience requirement from JD."""
        patterns = [
            r"(\d+\.?\d*)\s*\+?\s*years?\s*(?:of\s+)?(?:experience|exp)",
            r"minimum\s+(\d+\.?\d*)\s*years?",
            r"at least\s+(\d+\.?\d*)\s*years?"
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return 0.0

    def extract_keywords_tfidf(self, text: str, top_n: int = 30) -> List[str]:
        """
        Uses TF-IDF on the JD text to find the most discriminative terms.
        These are used as high-signal keywords during resume scoring.
        """
        cleaned = self.preprocessor.clean(text)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=100)
        try:
            tfidf_matrix = vectorizer.fit_transform([cleaned])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            top_indices = scores.argsort()[-top_n:][::-1]
            return [feature_names[i] for i in top_indices if scores[i] > 0]
        except Exception:
            return cleaned.split()[:top_n]

    def parse(self, title: str, raw_text: str) -> JobDescription:
        """Returns a populated JobDescription object from raw JD text."""
        jd = JobDescription(title=title, raw_text=raw_text)
        jd.required_skills = self.extract_required_skills(raw_text)
        jd.preferred_skills = self.extract_preferred_skills(raw_text)
        jd.min_experience = self.extract_min_experience(raw_text)
        jd.keywords = self.extract_keywords_tfidf(raw_text)
        jd.cleaned_text = self.preprocessor.clean(raw_text)
        return jd


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: FEATURE EXTRACTION & SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class ScoringEngine:
    """
    Core ML scoring module. Combines multiple signals to produce a weighted
    composite score for each candidate against the job description.

    Score Breakdown:
    ┌─────────────────────────┬────────┐
    │ Component               │ Weight │
    ├─────────────────────────┼────────┤
    │ TF-IDF Cosine Similarity│  35%   │
    │ Skill Match             │  30%   │
    │ Experience              │  15%   │
    │ Education               │  10%   │
    │ Keyword Density         │  10%   │
    └─────────────────────────┴────────┘
    """

    WEIGHTS = {
        "tfidf": 0.35,
        "skill": 0.30,
        "experience": 0.15,
        "education": 0.10,
        "keyword": 0.10,
    }

    EDUCATION_HIERARCHY = {
        "phd": 5, "doctorate": 5,
        "m.tech": 4, "mtech": 4, "m.e": 4, "master": 4, "m.sc": 4, "mba": 4, "mca": 4,
        "b.tech": 3, "btech": 3, "b.e": 3, "b.sc": 3, "bca": 3, "bachelor": 3,
        "diploma": 2, "12th": 1, "hsc": 1,
    }

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            min_df=1
        )
        self.scaler = MinMaxScaler()

    def compute_tfidf_similarity(self, jd: JobDescription, resumes: List[ParsedResume]) -> np.ndarray:
        """
        Fits TF-IDF on [JD + all resumes] corpus and computes cosine similarity
        of each resume vector against the JD vector.
        Returns array of similarity scores in [0, 1].
        """
        corpus = [jd.cleaned_text] + [r.cleaned_text for r in resumes]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        jd_vector = tfidf_matrix[0]
        resume_vectors = tfidf_matrix[1:]
        similarities = cosine_similarity(jd_vector, resume_vectors).flatten()
        return similarities

    def compute_skill_score(self, resume: ParsedResume, jd: JobDescription) -> Tuple[float, List[str], List[str]]:
        """
        Calculates skill match score as:
            (matched_required / total_required) * 0.7  +
            (matched_preferred / total_preferred) * 0.3

        Returns (score, matched_skills, missing_skills).
        """
        resume_skills_set = set(s.lower() for s in resume.skills)

        required = set(s.lower() for s in jd.required_skills)
        preferred = set(s.lower() for s in jd.preferred_skills)
        all_jd_skills = required | preferred

        matched = resume_skills_set & all_jd_skills
        missing = all_jd_skills - resume_skills_set

        req_matched = resume_skills_set & required
        pref_matched = resume_skills_set & preferred

        req_score = (len(req_matched) / len(required)) if required else 0.5
        pref_score = (len(pref_matched) / len(preferred)) if preferred else 0.5
        score = req_score * 0.7 + pref_score * 0.3

        return min(score, 1.0), list(matched), list(missing)

    def compute_experience_score(self, resume: ParsedResume, jd: JobDescription) -> float:
        """
        Scores candidate experience relative to JD minimum.
        Caps at 1.0 if experience exceeds requirement.
        Returns 0.3 baseline for freshers when no JD requirement given.
        """
        if jd.min_experience == 0:
            return 0.5 + min(resume.experience_years * 0.1, 0.5)
        ratio = resume.experience_years / jd.min_experience
        return min(ratio, 1.0)

    def compute_education_score(self, resume: ParsedResume, jd: JobDescription) -> float:
        """
        Maps candidate's detected degree to a numeric tier (1–5) and
        normalizes against the JD's required education tier.
        """
        candidate_level = 0
        for degree in resume.education:
            degree_lower = degree.lower().replace(".", "").replace(" ", "")
            for key, level in self.EDUCATION_HIERARCHY.items():
                if key.replace(".", "") in degree_lower:
                    candidate_level = max(candidate_level, level)

        required_level = 0
        if jd.required_education:
            req_lower = jd.required_education.lower()
            for key, level in self.EDUCATION_HIERARCHY.items():
                if key in req_lower:
                    required_level = level
                    break

        if required_level == 0:
            required_level = 3  # Default: assume Bachelor's required

        return min(candidate_level / required_level, 1.0)

    def compute_keyword_density_score(self, resume: ParsedResume, jd: JobDescription) -> float:
        """
        Counts how many of the JD's top TF-IDF keywords appear in the resume.
        Returns ratio of matched keywords to total JD keywords.
        """
        if not jd.keywords:
            return 0.5
        resume_lower = resume.cleaned_text.lower()
        matched = sum(1 for kw in jd.keywords if kw.lower() in resume_lower)
        return matched / len(jd.keywords)

    def score_candidates(self, jd: JobDescription, resumes: List[ParsedResume]) -> List[CandidateScore]:
        """
        Master scoring function. Runs all sub-scorers and combines results
        into a weighted final score. Assigns rank and recommendation label.
        Returns list of CandidateScore objects sorted by final_score descending.
        """
        logger.info(f"Scoring {len(resumes)} candidates for role: {jd.title}")

        tfidf_scores = self.compute_tfidf_similarity(jd, resumes)
        results = []

        for i, resume in enumerate(resumes):
            skill_score, matched, missing = self.compute_skill_score(resume, jd)
            exp_score = self.compute_experience_score(resume, jd)
            edu_score = self.compute_education_score(resume, jd)
            kw_score = self.compute_keyword_density_score(resume, jd)
            tfidf_score = float(tfidf_scores[i])

            final = (
                tfidf_score   * self.WEIGHTS["tfidf"]    +
                skill_score   * self.WEIGHTS["skill"]    +
                exp_score     * self.WEIGHTS["experience"]+
                edu_score     * self.WEIGHTS["education"] +
                kw_score      * self.WEIGHTS["keyword"]
            )

            total_jd_skills = len(set(jd.required_skills) | set(jd.preferred_skills))
            gap_pct = (len(missing) / total_jd_skills * 100) if total_jd_skills > 0 else 0.0

            cs = CandidateScore(
                candidate_id=resume.candidate_id,
                name=resume.name,
                tfidf_similarity=round(tfidf_score, 4),
                skill_match_score=round(skill_score, 4),
                experience_score=round(exp_score, 4),
                education_score=round(edu_score, 4),
                keyword_density_score=round(kw_score, 4),
                final_score=round(final, 4),
                matched_skills=matched,
                missing_skills=missing,
                skill_gap_percentage=round(gap_pct, 2),
            )
            results.append(cs)

        # Rank by final score
        results.sort(key=lambda x: x.final_score, reverse=True)
        for rank, cs in enumerate(results, start=1):
            cs.rank = rank
            cs.recommendation = self._assign_recommendation(cs.final_score, cs.skill_gap_percentage)

        return results

    def _assign_recommendation(self, score: float, gap_pct: float) -> str:
        """Maps final score + skill gap into a human-readable recommendation tier."""
        if score >= 0.70 and gap_pct <= 20:
            return "🟢 STRONGLY RECOMMENDED"
        elif score >= 0.55 and gap_pct <= 40:
            return "🟡 RECOMMENDED"
        elif score >= 0.40:
            return "🟠 MAYBE - NEEDS REVIEW"
        else:
            return "🔴 NOT RECOMMENDED"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class ReportGenerator:
    """
    Generates comprehensive output reports:
    - Terminal summary table
    - Detailed per-candidate skill gap analysis
    - CSV ranking export
    - Visualization charts (bar, radar, heatmap)
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def print_summary_table(self, scores: List[CandidateScore], jd_title: str):
        """Prints a formatted ranking table to console."""
        print("\n" + "=" * 90)
        print(f"  CANDIDATE RANKING REPORT — Role: {jd_title}")
        print("=" * 90)
        header = f"{'Rank':<5} {'Name':<22} {'Final':>7} {'TF-IDF':>7} {'Skill':>7} {'Exp':>6} {'Edu':>5} {'Gap%':>7}  Recommendation"
        print(header)
        print("-" * 90)
        for cs in scores:
            print(
                f"{cs.rank:<5} {cs.name[:21]:<22} {cs.final_score:>7.3f} "
                f"{cs.tfidf_similarity:>7.3f} {cs.skill_match_score:>7.3f} "
                f"{cs.experience_score:>6.3f} {cs.education_score:>5.3f} "
                f"{cs.skill_gap_percentage:>7.1f}%  {cs.recommendation}"
            )
        print("=" * 90)

    def print_skill_gap_analysis(self, scores: List[CandidateScore], top_n: int = 5):
        """Prints detailed skill gap for top N candidates."""
        print(f"\n{'─'*60}")
        print("  SKILL GAP ANALYSIS — TOP CANDIDATES")
        print(f"{'─'*60}")
        for cs in scores[:top_n]:
            print(f"\n#{cs.rank} {cs.name} (Score: {cs.final_score:.3f})")
            print(f"  ✅ Matched Skills ({len(cs.matched_skills)}): {', '.join(cs.matched_skills[:8]) or 'None'}")
            print(f"  ❌ Missing Skills ({len(cs.missing_skills)}): {', '.join(cs.missing_skills[:8]) or 'None'}")
            print(f"  📊 Skill Gap: {cs.skill_gap_percentage:.1f}%")
            print(f"  💡 {cs.recommendation}")

    def export_csv(self, scores: List[CandidateScore], jd_title: str) -> str:
        """Exports all candidate scores to a CSV file."""
        records = []
        for cs in scores:
            row = asdict(cs)
            row["matched_skills"] = "|".join(cs.matched_skills)
            row["missing_skills"] = "|".join(cs.missing_skills)
            records.append(row)
        df = pd.DataFrame(records)
        filename = self.output_dir / f"ranking_{jd_title.replace(' ', '_')}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"CSV exported to {filename}")
        return str(filename)

    def plot_ranking_chart(self, scores: List[CandidateScore], jd_title: str):
        """Generates a horizontal bar chart of final scores with color-coded tiers."""
        top = scores[:10]
        names = [f"#{cs.rank} {cs.name[:18]}" for cs in top]
        final_scores = [cs.final_score for cs in top]

        colors = []
        for cs in top:
            if cs.final_score >= 0.70: colors.append("#2ecc71")
            elif cs.final_score >= 0.55: colors.append("#f39c12")
            elif cs.final_score >= 0.40: colors.append("#e67e22")
            else: colors.append("#e74c3c")

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(names[::-1], final_scores[::-1], color=colors[::-1], edgecolor="white", height=0.65)

        for bar, score in zip(bars, final_scores[::-1]):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{score:.3f}", va="center", fontsize=9, fontweight="bold")

        ax.set_xlabel("Final Score", fontsize=11)
        ax.set_title(f"Candidate Rankings — {jd_title}", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlim(0, 1.1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3, linestyle="--")

        patches = [
            mpatches.Patch(color="#2ecc71", label="Strongly Recommended (≥0.70)"),
            mpatches.Patch(color="#f39c12", label="Recommended (≥0.55)"),
            mpatches.Patch(color="#e67e22", label="Maybe (≥0.40)"),
            mpatches.Patch(color="#e74c3c", label="Not Recommended (<0.40)"),
        ]
        ax.legend(handles=patches, loc="lower right", fontsize=8)
        plt.tight_layout()
        path = self.output_dir / f"chart_ranking_{jd_title.replace(' ', '_')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Bar chart saved to {path}")

    def plot_score_heatmap(self, scores: List[CandidateScore], jd_title: str):
        """Generates a heatmap of score components across top candidates."""
        top = scores[:8]
        data = {
            "TF-IDF": [cs.tfidf_similarity for cs in top],
            "Skill Match": [cs.skill_match_score for cs in top],
            "Experience": [cs.experience_score for cs in top],
            "Education": [cs.education_score for cs in top],
            "Keywords": [cs.keyword_density_score for cs in top],
        }
        names = [f"#{cs.rank} {cs.name[:14]}" for cs in top]
        df = pd.DataFrame(data, index=names)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(df, annot=True, fmt=".2f", cmap="YlOrRd",
                    linewidths=0.5, ax=ax, vmin=0, vmax=1,
                    annot_kws={"size": 9})
        ax.set_title(f"Score Component Heatmap — {jd_title}", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Score Component", fontsize=10)
        ax.set_ylabel("Candidate", fontsize=10)
        plt.tight_layout()
        path = self.output_dir / f"heatmap_{jd_title.replace(' ', '_')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Heatmap saved to {path}")

    def plot_skill_gap_chart(self, scores: List[CandidateScore], jd_title: str):
        """Plots matched vs missing skill counts for top candidates."""
        top = scores[:8]
        names = [f"#{cs.rank} {cs.name[:15]}" for cs in top]
        matched = [len(cs.matched_skills) for cs in top]
        missing = [len(cs.missing_skills) for cs in top]
        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(11, 5))
        b1 = ax.bar(x - width/2, matched, width, label="Matched Skills", color="#27ae60", alpha=0.85)
        b2 = ax.bar(x + width/2, missing, width, label="Missing Skills", color="#c0392b", alpha=0.85)

        ax.set_xlabel("Candidate", fontsize=11)
        ax.set_ylabel("Skill Count", fontsize=11)
        ax.set_title(f"Skill Match vs Gap — {jd_title}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout()
        path = self.output_dir / f"skill_gap_{jd_title.replace(' ', '_')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Skill gap chart saved to {path}")

    def export_json(self, scores: List[CandidateScore], jd: JobDescription):
        """Exports full analysis results as structured JSON."""
        output = {
            "job_title": jd.title,
            "total_candidates": len(scores),
            "required_skills": jd.required_skills,
            "candidates": [asdict(cs) for cs in scores],
        }
        path = self.output_dir / f"results_{jd.title.replace(' ', '_')}.json"
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"JSON results exported to {path}")
        return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class ResumeScreeningPipeline:
    """
    Top-level orchestrator that wires together all modules:
    Parser → Feature Extraction → Scoring → Reporting.
    Accepts text directly or from file paths.
    """

    def __init__(self, output_dir: str = "outputs"):
        self.nlp = initialize_nlp_resources()
        self.resume_parser = ResumeParser(nlp=self.nlp)
        self.jd_parser = JobDescriptionParser()
        self.scoring_engine = ScoringEngine()
        self.reporter = ReportGenerator(output_dir=output_dir)

    def load_resume_from_text(self, candidate_id: str, text: str) -> ParsedResume:
        return self.resume_parser.parse(candidate_id, text)

    def load_jd_from_text(self, title: str, text: str) -> JobDescription:
        return self.jd_parser.parse(title, text)

    def run(
        self,
        jd_title: str,
        jd_text: str,
        resume_data: List[Dict[str, str]],
        generate_charts: bool = True,
    ) -> List[CandidateScore]:
        """
        Full pipeline execution:
        1. Parse JD
        2. Parse all resumes
        3. Score and rank candidates
        4. Generate reports and charts

        Args:
            jd_title: Job role name
            jd_text: Raw job description text
            resume_data: List of {"id": str, "text": str} dicts
            generate_charts: Whether to save visualization plots

        Returns:
            Sorted list of CandidateScore objects
        """
        logger.info(f"Pipeline started for: {jd_title}")

        jd = self.load_jd_from_text(jd_title, jd_text)
        logger.info(f"JD parsed. Required skills: {jd.required_skills[:5]}...")

        resumes = []
        for item in resume_data:
            resume = self.load_resume_from_text(item["id"], item["text"])
            resumes.append(resume)
            logger.info(f"Parsed: {resume.name} | Skills: {len(resume.skills)} | Exp: {resume.experience_years}yr")

        scores = self.scoring_engine.score_candidates(jd, resumes)

        self.reporter.print_summary_table(scores, jd_title)
        self.reporter.print_skill_gap_analysis(scores, top_n=5)
        self.reporter.export_csv(scores, jd_title)
        self.reporter.export_json(scores, jd)

        if generate_charts:
            self.reporter.plot_ranking_chart(scores, jd_title)
            self.reporter.plot_score_heatmap(scores, jd_title)
            self.reporter.plot_skill_gap_chart(scores, jd_title)

        logger.info("Pipeline complete. ✅")
        return scores
