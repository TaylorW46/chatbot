import ftbot
while True:
    user_input = input("You: ")

    response = ftbot.chatbot(user_input, max_length=50, num_return_sequences=1)    
    
    tokens = ftbot.tokenizer([user_input, response], padding=True, truncation=True)
    print(f"Bot: {response[0]['generated_text']}")