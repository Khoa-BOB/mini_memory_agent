from openai import OpenAI
from memory import MemoryManager
import logging

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(filename='logs/mini_memory_agent.log', level=logging.INFO)
    client = OpenAI()
    # mem_manager = MemoryManager()

    SYSTEM_PROMPT = """
    You are a friendly assistant.
    Give a short and clear answers.
    Use previous conversation context when relevant.
    """

    chat_history = []
    # Create an agent loop
    while(True):
        message = input("User: ")

        if message.strip() in ['q','exit','quit']:
            break

        chat_history.append({
                    'role' : 'user',
                    'content': message
                })
        
        response = client.responses.create(
            model="gpt-4o-mini",
            instructions=SYSTEM_PROMPT,
            input=chat_history
        )

        chat_history.append({
            'role':'assistant',
            'content' : response.output_text
        })

        print("Agent: ",response.output_text)


if __name__ == "__main__":
    main()
