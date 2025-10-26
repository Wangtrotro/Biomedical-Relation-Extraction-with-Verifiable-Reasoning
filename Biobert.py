from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading BioGPT model...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")
model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT")

prompt = "Aspirin is used to treat"
inputs = tokenizer(prompt, return_tensors="pt")

print("Generating text...")
outputs = model.generate(**inputs, max_length=40)
print("\n🧠 Generated result:\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
