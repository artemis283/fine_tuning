from datasets import load_dataset
import pandas as pd
import random

# Load dataset
dataset = load_dataset("MLBtrio/genz-slang-dataset")
df = pd.DataFrame(dataset["train"])

# Create conversational examples
conversations = []

for _, row in df.iterrows():
    slang = row['slang']
    definition = row['definition']
    example = row['example']
    
    # Create natural conversational exchanges using the slang
    conversation_templates = [
        # Template 1: Direct usage in conversation
        f"[INST] How was your day? [/INST]\n"
        f"My day was {random.choice(['pretty good', 'kinda rough', 'bussin', 'mid'])}, ngl. "
        f"I just {random.choice(['finished', 'completed', 'wrapped up'])} a project and it's {slang}. "
        f"{example}\n\n"
        f"[INST] What does {slang} mean? [/INST]\n"
        f"Oh, when I say \"{slang}\" I mean {definition}. It's Gen Z slang!",
        
        # Template 2: Friend-to-friend chat
        f"[INST] Let's chat about the new movie that came out. What did you think? [/INST]\n"
        f"The movie was actually {slang}! {example} "
        f"I'd definitely recommend checking it out this weekend.",
        
        # Template 3: Multiple slang terms in one response
        f"[INST] Tell me about your weekend plans [/INST]\n"
        f"Gonna hit up this party on Saturday, it's gonna be {slang}! "
        f"My fit is fire and I'm ready to vibe. No cap, "
        f"last time the DJ was bussin and the vibes were immaculate."
    ]
    
    conversations.extend(conversation_templates)

# Now you have natural conversations incorporating the slang terms