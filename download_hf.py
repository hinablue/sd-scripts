from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your model and tokenizer
model_name = "google/gemma-2-2b"  # Replace with your desired model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the directory where you want to save the model
save_directory = "./gemma-2-2b"

# Save the model and tokenizer
model.save_pretrained(save_directory, safe_serialization=True, max_shard_size="30GB")

print(f"Model and tokenizer saved to {save_directory} in safetensors format.")
