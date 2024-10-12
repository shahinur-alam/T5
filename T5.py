from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load pretrained tokenizer and model
model_name = "t5-base"  # You can also use "t5-small", "t5-large", etc.
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Prepare input text
input_text = "translate English to German: The house is wonderful."

# Tokenize input
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# Generate output
outputs = model.generate(input_ids)

# Decode output
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded_output)