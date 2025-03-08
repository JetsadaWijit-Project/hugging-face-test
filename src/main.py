from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from config import MODEL_NAME

def load_model():
    """Load the Hugging Face model and tokenizer."""
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer

def generate_response(prompt, model, tokenizer, max_length=100):
    """Generate a response using the model."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    model, tokenizer = load_model()
    
    while True:
        user_input = input("Enter a prompt (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        
        response = generate_response(user_input, model, tokenizer)
        print("\nAI Response:\n", response, "\n")
