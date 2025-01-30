from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load the pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Function to generate a new sentence
def generate_new_sentence(input_sentence, max_length=50, temperature=0.7):
    # Tokenize the input sentence
    input_ids = tokenizer.encode(input_sentence, return_tensors="pt")
    
    # Generate output using GPT-2
    output = model.generate(
        input_ids, 
        max_length=max_length, 
        temperature=temperature, 
        num_return_sequences=1, 
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and return the generated sentence
    generated_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_sentence

# Example usage
input_sentence = "Good in lab work, but theory needs more focus."
#Good in lab work, but theory needs more focus.
new_sentence = generate_new_sentence(input_sentence)
print("Generated Sentence:", new_sentence)
