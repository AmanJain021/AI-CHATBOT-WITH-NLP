import spacy
import json
import random
import os

# Load SpaCy model
nlp = spacy.load("en_core_web_md")

# Load intents from file
base_dir = os.path.dirname(os.path.abspath(__file__))
intents_path = os.path.join(base_dir, "intents.json")
with open(intents_path, "r") as f:
    intents = json.load(f)

# Build intent index
intent_index = []
tag_responses = {}

for intent in intents["intents"]:
    tag = intent["tag"]
    responses = intent["responses"]
    tag_responses[tag] = responses

    for pattern in intent["patterns"]:
        intent_index.append({
            "tag": tag,
            "pattern": pattern,
            "doc": nlp(pattern.lower())
        })

def get_intent(user_input):
    user_doc = nlp(user_input.lower())
    scores = []

    for item in intent_index:
        score = user_doc.similarity(item["doc"])
        scores.append((item["tag"], score))

    best_match = max(scores, key=lambda x: x[1])
    return best_match[0] if best_match[1] > 0.70 else "default"

def chatbot():
    print("🤖 ChatBot: Hello! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("🤖 ChatBot: Goodbye! 👋")
            break

        tag = get_intent(user_input)
        response = random.choice(tag_responses.get(tag, ["Sorry, I didn't get that."]))
        print(f"🤖 ChatBot: {response}")

if __name__ == "__main__":
    chatbot()