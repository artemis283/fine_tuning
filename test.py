from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login
import torch


login()

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# Load adapter weights
model = PeftModel.from_pretrained(base_model, "/root/fine_tuning/genz-llama-chat-adapter")

# Chat function
def chat(user_input):
    # Format the prompt to be more conversational
    prompt = f"[INST] You are a Gen Z friend having a casual conversation. Respond naturally and conversationally to: {user_input} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("Generating response...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.8,
        do_sample=True,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,  # Increased to prevent repetitive patterns
        no_repeat_ngram_size=3,
        num_return_sequences=1
    )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # More robust response extraction
    if "[/INST]" in full_response:
        response = full_response.split("[/INST]")[-1].strip()
    else:
        response = full_response.strip()
    
    # If response is empty or too short, provide a fallback
    if not response or len(response) < 5:
        response = "No cap, that's fire! What else you got?"
    
    return response

# Interactive chat loop
print("Chat Bot is ready! (type 'exit' to quit)")
while True:
    try:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        print("Processing...")
        response = chat(user_input)
        print(f"Bot: {response}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")