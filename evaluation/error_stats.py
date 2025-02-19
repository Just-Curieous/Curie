import os
from openai import AzureOpenAI

# Initialize Azure OpenAI client
    client = AzureOpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        azure_endpoint=os.getenv('OPENAI_API_BASE'),
        # api_version='2023-03-15-preview',
        api_version=os.getenv('API_VERSION'),
        organization=os.getenv('OPENAI_ORGANIZATION'),
    ) 

# Step 1: Load the file content
with open('common_errors_nonbug_report-1738044026.txt', 'r') as file:
    error_report = file.read()

# Step 2: Define the system and user prompts
system_prompt = (
    "You are an expert summarizer. "
)

user_prompt = f"""

Please do the following:
1. Categorize the errors into meaningful experimentation-related categories.
2. Provide a count of errors in each category. Provide the source file as well.
3. Output the results in a clear and concise format (e.g. Markdown).


Here is a log file containing error summaries from 30 files:
{error_report}

"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

# Step 3: Send the query to Azure OpenAI
response = client.chat.completions.create(
    model='gpt-4o',
    messages=messages,
    temperature=0.7,
    stop=None
)

# Step 4: Print the LLM's response


print(response.choices[0].message.content)





