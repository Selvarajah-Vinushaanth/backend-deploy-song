from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Vinushaanth/my-tamil-lyrics"
tokenizer = None
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()

def generate_lyrics_text(motion: str, seed: Optional[str] = "") -> str:
    load_model()
    seed = seed or " "

    try:
        # directly pass seed + emotion tags to model
        input_text = f"{seed} <emotion:{motion}> <sep>"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=128,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=1.0
            )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        clean_text = generated_text.split("<sep>")[-1].strip()

        return clean_text

    except Exception as e:
        print(f"Error generating lyrics: {str(e)}")
        return f"Could not generate lyrics for {motion} mood.\nLa la la, sing along!"
