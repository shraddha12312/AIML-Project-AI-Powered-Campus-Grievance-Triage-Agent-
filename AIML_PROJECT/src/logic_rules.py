"""
CO2: Problem Solving Methods & Knowledge Representation
This module uses hard-coded propositional logic to catch extreme
edge cases that might be missed by ML models.
"""

# Hardcoded propositions representing absolute True states
EMERGENCY_KEYWORDS = [
    "fire", "blood", "spark", "shock", "collapse", 
    "injured", "police", "ambulance", "fight", 
    "bleeding", "smoke", "short circuit"
]

def check_emergency_logic(complaint_text: str) -> bool:
    """
    Evaluates P1 OR P2 OR P3 ... OR Pn 
    Where Px is the presence of an emergency keyword.
    """
    text_lower = complaint_text.lower()
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in text_lower:
            return True # Logic matched, escalate immediately!
    return False
