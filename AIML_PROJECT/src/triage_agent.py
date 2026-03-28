"""
CO1/CO4: Intelligent Agents & ML Basics
The Core Problem-Solving Agent that coordinates Logic, ML, and NLP.
"""
import os
import pickle
from colorama import init, Fore, Style
from logic_rules import check_emergency_logic
from sentiment_analyzer import analyze_frustration

# Initialize terminal colors
init()

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'department_classifier.pkl')

class TriageAgent:
    def __init__(self):
        self.classifier_pipeline = None
        self._load_brain()
        
    def _load_brain(self):
        """Loads the pre-trained ML model."""
        if not os.path.exists(MODEL_PATH):
            print(Fore.RED + "Error: Model not found. Please run 'python src/model_training.py' first!" + Style.RESET_ALL)
            exit(1)
            
        with open(MODEL_PATH, 'rb') as f:
            self.classifier_pipeline = pickle.load(f)
            
    def process_complaint(self, text: str):
        print("\n" + "="*50)
        print(f"📋 Received New Complaint: '{text}'")
        print("="*50)
        
        # Step 1: Intelligent Agent applies Knowledge Representation (Logic Rules)
        print(Fore.CYAN + "[Agent Step 1] Applying Propositional Logic for Emergencies..." + Style.RESET_ALL)
        is_emergency = check_emergency_logic(text)
        
        if is_emergency:
             department = "🚨 CAMPUS SECURITY & EMERGENCY RESPONSE 🚨"
             urgency = "🔥 CRITICAL (Immediate Action Required)"
             print(Fore.RED + f" -> EMERGENCY DETECTED BY LOGIC PROTOCOL!" + Style.RESET_ALL)
             # Agent short-circuits ML steps to save time (Rationality)
             self._dispatch(department, urgency)
             return
             
        # Step 2: Use Machine Learning (Supervised Learning Classification)
        print(Fore.CYAN + "[Agent Step 2] Querying ML Model for Department Classification..." + Style.RESET_ALL)
        department_prediction = self.classifier_pipeline.predict([text])[0]
        # Get confidence/probability using predict_proba (Bayesian Statistics application)
        probabilities = self.classifier_pipeline.predict_proba([text])[0]
        confidence = max(probabilities) * 100
        print(f" -> Predicted Department: {department_prediction} (Confidence: {confidence:.1f}%)")
        
        # Step 3: NLP Sentiment Analysis (Case Study)
        print(Fore.CYAN + "[Agent Step 3] Running NLP Sentiment Analyzer to gauge frustration..." + Style.RESET_ALL)
        sentiment_data = analyze_frustration(text)
        
        urgency = f"Normal (Multiplier: {sentiment_data['priority_boost']}x)"
        if sentiment_data['frustration_level'] == "High Frustration":
            urgency = "HIGH PRIORITY (Student is very angry)"
            
        print(f" -> Sentiment: {sentiment_data['frustration_level']} (Score: {sentiment_data['compound_score']})")
        
        # Dispatch
        self._dispatch(department_prediction, urgency)
        
    def _dispatch(self, department: str, urgency: str):
         print(Fore.GREEN + f"\n🎯 Triage Complete! Dispatching Ticket..." + Style.RESET_ALL)
         print(f"   ► Target Department: {department}")
         print(f"   ► Priority Level:    {urgency}")
         print("="*50 + "\n")

if __name__ == "__main__":
    print(Fore.YELLOW + "Initializing AI Campus Triage Agent..." + Style.RESET_ALL)
    agent = TriageAgent()
    
    print("Welcome to the AI Campus Complaint Triage Console.")
    print("Type your complaint (or 'quit' to exit).")
    
    while True:
        user_input = input("\nEnter Student Complaint > ")
        if user_input.lower() in ['quit', 'exit']:
            print("Shutting down Triage Agent... Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        agent.process_complaint(user_input)
