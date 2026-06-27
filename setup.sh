#!/bin/bash
# ============================================================
# setup.sh — One-command setup for Resume Screening System
# ============================================================
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Downloading spaCy English model..."
python -m spacy download en_core_web_sm

echo "Downloading NLTK resources..."
python -c "
import nltk
for r in ['punkt','stopwords','wordnet','averaged_perceptron_tagger','punkt_tab','omw-1.4']:
    nltk.download(r, quiet=True)
print('NLTK resources downloaded.')
"

echo ""
echo "✅ Setup complete! Now run:  python main.py"
