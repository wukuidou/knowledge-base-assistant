import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
       api_key=os.getenv("LLM_API_KEY"),
       base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
   )

response = client.chat.completions.create(
       model=os.getenv("LLM_MODEL", "deepseek-chat"),
       messages=[
           {"role": "system", "content": "你是个人知识库助手"},
           {"role": "user", "content": "请用三句话介绍你自己"},
       ],
       stream=False,
   )

print(response.choices[0].message.content)
