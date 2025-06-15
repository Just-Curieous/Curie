from langchain_openai import AzureChatOpenAI
import sys, os
import ast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import tool
import model

# Read contents from file:
with open("assets/large_message_block.txt", 'r') as f:
    large_block = f.read()

    dict1 = ast.literal_eval(large_block)
    
    # print(dict1["scheduler"]["messages"][-1])

    gpt_4_llm = model.create_gpt_4()
    summarizer_llm = model.create_gpt_4()

    response = model.query_model_safe(gpt_4_llm, summarizer_llm, dict1["scheduler"]["messages"])