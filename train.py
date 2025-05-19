from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer
from datasets import Dataset
from datasets import load_dataset
import pandas as pd
import random
from huggingface_hub import login
import torch


login()

# Load model and tokenizer
model_id = "meta-llama/Llama-3.1-8B-Instruct"  
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Load model with proper device handling
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload"
)


# Load dataset
dataset = load_dataset("MLBtrio/genz-slang-dataset")
df = pd.DataFrame(dataset["train"])

# Create conversational examples
conversations = []

for _, row in df.iterrows():
    slang = row['Slang']
    definition = row['Description']
    example = row['Example']
    
    # Create natural conversational exchanges using the slang
    conversation_templates = [
        #Direct usage in conversation
        f"[INST] How was your day? [/INST]\n"
        f"My day was {random.choice(['pretty good', 'kinda rough', 'bussin', 'mid'])}, ngl. "
        f"I just {random.choice(['finished', 'completed', 'wrapped up'])} a project and it's {slang}. "
        f"{example}\n\n"
        f"[INST] What does {slang} mean? [/INST]\n"
        f"Oh, when I say \"{slang}\" I mean {definition}. It's Gen Z slang!",
        
        #Friend-to-friend chat
        f"[INST] Let's chat about the new movie that came out. What did you think? [/INST]\n"
        f"The movie was actually {slang}! {example} "
        f"I'd definitely recommend checking it out this weekend.",
        
        #Multiple slang terms in one response
        f"[INST] Tell me about your weekend plans [/INST]\n"
        f"Gonna hit up this party on Saturday, it's gonna be {slang}! "
        f"My fit is fire and I'm ready to vibe. No cap, "
        f"last time the DJ was bussin and the vibes were immaculate."
    ]
    
    conversations.extend(conversation_templates)

# Convert conversations to dataset format
dataset = Dataset.from_dict({"text": conversations})

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])


# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target attention modules
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./genz-llama-chat",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    warmup_steps=100,
    max_steps=-1,  # Let it run for full epochs
    dataloader_num_workers=4,
    dataloader_pin_memory=True
)

# Set up trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Train model
trainer.train()

# Save the LoRA adapter weights
model.save_pretrained("./genz-llama-chat-adapter")