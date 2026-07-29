from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)
r = client.chat.completions.create(
    model=os.getenv("LLM_MODEL"),
    messages=[{"role": "user", "content": "Python列表推导式怎么写？你的回答只能用我知识库中的内容（但你并没有知识库，所以你应该说不知道）"}]
)
print(r.choices[0].message.content)
