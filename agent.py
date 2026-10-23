import logging
from openai import OpenAI
from memory import MemoryManager

logger = logging.getLogger(__name__)

BASE_PROMPT = "You are a helpful assistant with long-term memory"

def to_bullet(items):
    return "\n".join(f"- {item}" for item in items)

class Agent:
    def __init__(self, client, model='gpt-4o-mini'):
        self.client = OpenAI()
        self.memory = MemoryManager()
        self.model = model
        self.history = []

    def instruction(self, query):
        facts = self.memory.search_memories(query)
        logger.info("recalled facts: %s", facts)

        prompt = f"{BASE_PROMPT}\n# Relevant things you know about the user:\n{to_bullet(facts) or '(nothing)'}"
        logging.info(prompt)
        return prompt

    def chat(self, user_input):
        self.history.append({
            "role" : "user",
            "content" : user_input
        })
        response = self.client.responses.create(
            model = self.model,
            instructions = self.instruction(user_input),
            input = self.history,
        )
        self.history.append({
            "role": "assistant",
            "content" : response.output_text
        })
        return response.output_text

