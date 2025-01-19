import openai

# Initialize OpenAI API
openai.api_key = 'YOUR OPENAI API KEY'

def analyze_with_chatgpt(software_description):
    """
    Simulate a conversation with ChatGPT to analyze a software's behavior.
    
    Parameters:
    software_description (str): A description of the software or its behavior.
    
    Returns:
    str: ChatGPT's opinion on whether the software is potentially malicious.
    """
    # Start a conversation with ChatGPT using the chat/completions endpoint
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"I've observed a software with the following behavior: {software_description}. Do you think this could be malware?"}
        ]
    )
    
    # Extract ChatGPT's response
    chatgpt_response = response.choices[0].message['content'].strip()
    return chatgpt_response

# Example usage
software_description = "The software tries to access user's contacts without permission and sends data to an unknown server."
result = analyze_with_chatgpt(software_description)
print(result)
