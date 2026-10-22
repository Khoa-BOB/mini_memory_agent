from openai import OpenAI

def main():
    client = OpenAI()

    # Create an agent loop
    while(True):
        user_input = input("User: ")

        if user_input.strip() in ['q','exit','quit']:
            break

        response = client.responses.create(
            model="gpt-4o-mini",
            input=user_input,
        )
        print(response.output_text)


if __name__ == "__main__":
    main()
