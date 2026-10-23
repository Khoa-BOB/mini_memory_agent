import logging
import json
from openai import OpenAI
from memory import MemoryManager


logger = logging.getLogger(__name__)

BASE_PROMPT = "You are a helpful assistant with long-term memory"

TOOLS = [
    {
        "type": "function",
        "name": "save_memory",
        "description": (
            "Save or overwrite a long-term fact about the user (name, preferences, "
            "projects, important dates). Use a short snake_case key like 'user_city'. "
            "If a relevant fact is already listed in your instructions, reuse its key. "
            "Do NOT save small talk or things only relevant to this conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
            "required": ["key", "value"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "delete_memory",
        "description": "Forget a fact when the user asks you to, or when it is no longer true.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

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

    def run_tool(self, name, args):
        logger.info("tool call: %s %s", name, args)
        if name == "save_memory":
            self.memory.update_memory(args['key'], args['value'])
        elif name == "delete_memory":
            self.memory.delete_memory(args['key'])
        else:
            return f"error: unknown tool {name}"
        return "ok"

    def chat(self, user_input):
        self.history.append({
            "role" : "user",
            "content" : user_input
        })
        response = self.client.responses.create(
            model = self.model,
            instructions = self.instruction(user_input),
            input = self.history,
            tools=TOOLS
        )
        self.history += response.output
        logging.info(response.output)
        calls =[item for item in response.output if item.type == "function_call"]
        if not calls:
            return response.output_text

        
        for call in calls:
                result = self.run_tool(call.name, json.loads(call.arguments))
                self.history.append(
                    {"type": "function_call_output", "call_id": call.call_id, "output": result}
                )

