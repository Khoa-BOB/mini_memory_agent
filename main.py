from openai import OpenAI
from memory import MemoryManager
from agent import Agent
import logging

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(filename='logs/mini_memory_agent.log', level=logging.INFO)
    client = OpenAI()
    agent = Agent(client)
    # mem_manager = MemoryManager()

    chat_history = []
    # Create an agent loop
    while(True):
        user_input = input("User: ")
        if user_input.strip() in ['q','exit','quit']:
            break
        print("Agent: ",agent.chat(user_input))


if __name__ == "__main__":
    main()
