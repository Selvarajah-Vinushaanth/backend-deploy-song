"""
Metaphor Creator Model
This module generates creative metaphors based on Vehicle, Tenor, and Context using fine-tuned Tamil model.
"""
import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

# Define the model for text generation - using the fine-tuned model
MODEL_NAME = "Vinushaanth/metaphor-create-withoutlora"  # Your fine-tuned model

# Initialize tokenizer and model (lazy loading - will load on first use)
tokenizer = None
model = None
device = None

# Context mapping from English/Tamil to Tamil contexts
CONTEXT_MAPPING = {
    "poetic": "கவிதை",
    "கவிதை": "கவிதை",
    "short": "குறுகிய",
    "குறுகிய": "குறுகிய", 
    "romantic": "காதல்",
    "காதல்": "காதல்",
    "philosophical": "தத்துவம்",
    "தத்துவம்": "தத்துவம்",
    "humorous": "நகைச்சுவை",
    "நகைச்சுவை": "நகைச்சுவை",
    "general": "பொதுவான",
    "சுதந்திரம்": "சுதந்திரம்",
    "தடுப்பு": "தடுப்பு",
    "நீர் போல ஓடும்": "நீர் போல ஓடும்",
    "அதிர்ச்சி": "அதிர்ச்சி",
    "மறைந்தல்": "மறைந்தல்",
    "இன்பம்": "இன்பம்"
}

# Predefined metaphor templates for fallback
PREDEFINED_METAPHORS = {
    "கவிதை": [
        "{vehicle} போல் {tenor} மின்னுகிறது",
        "{tenor} ஒரு {vehicle} போன்று அழகாக இருக்கிறது",
        "{vehicle} என்ற {tenor} மனதில் நிற்கிறது"
    ],
    "காதல்": [
        "{vehicle} போல் என் {tenor} துடிக்கிறது",
        "{tenor} ஒரு {vehicle} போல் என் மனதில் வாழ்கிறது",
        "{vehicle} என்னும் {tenor} என் உயிரில் கலந்தது"
    ],
    "தத்துவம்": [
        "{vehicle} போல் {tenor} நிலையற்றது",
        "{tenor} ஒரு {vehicle} போன்று ஆழமானது",
        "{vehicle} என்ற {tenor} வாழ்க்கையின் உண்மை"
    ],
    "நகைச்சுவை": [
        "{vehicle} போல் {tenor} விநோதமாக இருக்கிறது",
        "{tenor} ஒரு {vehicle} போல் சுறுசுறுப்பாக இருக்கிறது"
    ],
    "குறுகிய": [
        "{vehicle} போல் {tenor}",
        "{tenor} = {vehicle}",
        "{vehicle} என்ற {tenor}"
    ]
}

def load_model():
    """Load the fine-tuned model and tokenizer if not already loaded"""
    global tokenizer, model, device
    if tokenizer is None or model is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            print(f"Model loaded successfully on {device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to a basic model if the fine-tuned one fails
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            device = torch.device("cpu")
            model.to(device)
            model.eval()

def generate_metaphor(source: str, target: str, Context: str = "கவிதை", count: int = 3) -> List[str]:
    """
    Generate creative metaphors using the fine-tuned model format.
    
    Args:
        source: The Vehicle (concrete concept) for the metaphor
        target: The Tenor (abstract concept) for the metaphor  
        Context: Context category for the metaphor
        count: Number of metaphors to generate
    
    Returns:
        List of generated metaphors
    """
    # Map the source to Vehicle and target to Tenor for clarity
    vehicle = source  # Concrete concept (source domain)
    tenor = target    # Abstract concept (target domain)
    
    # Map context to Tamil if needed
    tamil_context = CONTEXT_MAPPING.get(Context, Context)
    
    # Load model if not already loaded
    load_model()
    
    # Limit count to reasonable range
    count = max(1, min(5, count))
    
    metaphors = []
    
    # Generate metaphors using the fine-tuned model format
    for i in range(count):
        try:
            # Format input according to fine-tuned model's expected format
            input_text = f"Vehicle:{vehicle} Tenor:{tenor} Context:{tamil_context} <sep>"
            
            # Tokenize input
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            
            # Generate with the model
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=100,  # Increased for better metaphors
                    num_beams=5,
                    early_stopping=True,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode the generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prefix to get only generated part
            generated_part = generated_text.replace(input_text, "").strip()
            
            # Clean up the generated metaphor
            if generated_part:
                # Split by common Tamil sentence endings and take the first complete sentence
                sentences = []
                current_sentence = ""
                
                for char in generated_part:
                    current_sentence += char
                    if char in ['।', '.', '!', '?'] or (char == ' ' and len(current_sentence.strip()) > 30):
                        if len(current_sentence.strip()) > 10:  # Minimum length check
                            sentences.append(current_sentence.strip())
                            break
                        current_sentence = ""
                
                # If no sentence ending found, take first 50 characters and add period
                if not sentences and len(generated_part) > 10:
                    sentences.append(generated_part[:60].strip() + ".")
                
                if sentences:
                    metaphor = sentences[0]
                    # Ensure it doesn't repeat the input words too much
                    if not (metaphor.count(vehicle) > 3 or metaphor.count(tenor) > 3):
                        metaphors.append(metaphor)
                        continue
            
            # Fallback to template if generation fails or is poor quality
            template = random.choice(PREDEFINED_METAPHORS.get(tamil_context, PREDEFINED_METAPHORS["கவிதை"]))
            fallback_metaphor = template.format(vehicle=vehicle, tenor=tenor)
            metaphors.append(fallback_metaphor)
            
        except Exception as e:
            print(f"Error generating metaphor {i+1}: {e}")
            # Use predefined template as fallback
            template = random.choice(PREDEFINED_METAPHORS.get(tamil_context, PREDEFINED_METAPHORS["கவிதை"]))
            fallback_metaphor = template.format(vehicle=vehicle, tenor=tenor)
            metaphors.append(fallback_metaphor)
    
    # Remove duplicates while preserving order
    unique_metaphors = []
    for metaphor in metaphors:
        if metaphor not in unique_metaphors:
            unique_metaphors.append(metaphor)
    
    # If we don't have enough unique metaphors, add more using templates
    while len(unique_metaphors) < count:
        template = random.choice(PREDEFINED_METAPHORS.get(tamil_context, PREDEFINED_METAPHORS["கவிதை"]))
        new_metaphor = template.format(vehicle=vehicle, tenor=tenor)
        if new_metaphor not in unique_metaphors:
            unique_metaphors.append(new_metaphor)
    
    return unique_metaphors[:count]

# Alternative function name for backward compatibility
def generate_metaphor_by_topic(topic: str, Context: str = "கவிதை", count: int = 3, target: str = None) -> List[str]:
    """
    Backward compatibility function that maps old parameters to new format
    """
    if target is None:
        # If no target provided, use topic as both vehicle and tenor
        return generate_metaphor(topic, topic, Context, count)
    else:
        return generate_metaphor(topic, target, Context, count)
