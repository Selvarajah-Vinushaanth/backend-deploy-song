"""
Metaphor Classifier Model
This module classifies text to determine if it contains metaphors using Hugging Face models.
"""
import torch
from typing import Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load pre-trained model and tokenizer for metaphor classification
# Note: In a real implementation, you would use a model specifically fine-tuned for metaphor detection
# This example uses a sentiment model as a placeholder - you would replace with an actual metaphor classifier
MODEL_NAME = "vimosh-v/muril-large-metaphor"  # Replace with actual metaphor model
    
# Initialize tokenizer and model (lazy loading - will load on first use)
tokenizer = None
model = None

def load_model():
    """Load the model and tokenizer if not already loaded"""
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()  # Set model to evaluation mode

def classify_metaphor(text: str) -> Tuple[bool, float]:
    """
    Classify whether the given text contains metaphors using a pre-trained model.
    
    Args:
        text: The text to analyze
        
    Returns:
        Tuple of (is_metaphor, confidence_score)
    """
    # Load model if not already loaded
    load_model()
    
    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
    
    # Get the confidence score for positive class (index 1)
    # Note: In a real implementation, you would map the output to metaphor/non-metaphor
    confidence = probabilities[0][1].item()
    is_metaphor = confidence > 0.5
    
    return is_metaphor, confidence
