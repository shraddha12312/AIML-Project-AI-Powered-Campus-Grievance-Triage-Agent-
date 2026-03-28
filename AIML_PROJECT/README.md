# AI-Powered Campus Grievance Triage Agent 🎓🤖

## 🎯 The Real-World Problem
In university campuses, hundreds of student complaints (maintenance, academics, IT, hostel issues) are submitted daily as raw text. Administration typically processes these manually and sequentially. This leads to massive inefficiencies:
1. Critical or emergency issues get buried under trivial ones.
2. Routing to the wrong department causes delayed resolution.

## 💡 The AI Solution (BYOP Objective)
This project is an **Intelligent Agent** designed to automate the triage of incoming campus complaints. It reads the raw text of a student's grievance and automatically:
1. **Classifies** the appropriate department (Hostel, IT, Academics, Maintenance) using Machine Learning.
2. **Analyzes Sentiment** using NLP to gauge the student's frustration level and prioritize it.
3. **Applies Propositional Logic** to immediately flag critical emergency keywords (e.g., "fire", "blood", "spark") to bypass the AI and escalate instantly.

---

## 📚 Course Concepts Applied
This project meaningfully applies the following concepts from the **Fundamentals of AI and ML** syllabus:
*   **Intelligent Agents (CO1):** The `TriageAgent` class acts as a Problem-Solving Agent perceiving text inputs from its environment and making rationing routing decisions.
*   **Knowledge Representation & Logic (CO2):** Uses hard-coded propositional logic rules to handle edge-case emergencies that ML might misunderstand.
*   **Machine Learning Basics (CO4):** Uses **Supervised Learning** (Naive Bayes Classification) trained on historical complaint data to predict the target department. Handles text representation/feature learning via TF-IDF (Term Frequency-Inverse Document Frequency).
*   **Case Studies - NLP & Sentiment Analyzer (CO5):** Uses Natural Language Processing (NLTK's VADER) to extract emotional polarity from the text to dynamically adjust the priority of the ticket.

---

## 🚀 How to Set Up and Run This Project (For Beginners)

### 1. Prerequisites
Ensure you have Python (3.8 or higher) installed on your system.
You can check this by running `python --version` in your terminal.

### 2. Installation Steps
Open your terminal (Command Prompt / PowerShell) and navigate to the project folder, then run:

```bash
# 1. Install the required libraries
pip install -r requirements.txt

# 2. Download the NLP Lexicon required for Sentiment Analysis
python -c "import nltk; nltk.download('vader_lexicon')"
```

### 3. Usage Steps
The system operates in two phases: **Training the ML Model** and **Running the Intelligent Agent**.

**Step A: Train the Model**
Run this to train the classifier on the provided dataset. It will generate a model file (`department_classifier.pkl`) and a vectorizer (`tfidf_vectorizer.pkl`).
```bash
python src/model_training.py
```

**Step B: Run the Triage Agent**
Start the intelligent agent. You can input your own complaints and watch the AI route and prioritize them in real-time!
```bash
python src/triage_agent.py
```

---

## 📂 Project Structure
*   `data/complaints_dataset.csv` - The historical dataset used for supervised learning.
*   `src/logic_rules.py` - Contains the propositional logic for emergency keyword overrides.
*   `src/model_training.py` - Handles TF-IDF vectorization and Naive Bayes model training.
*   `src/sentiment_analyzer.py` - Handles NLP polarity scoring.
*   `src/triage_agent.py` - The core Intelligent Agent that glues everything together!
