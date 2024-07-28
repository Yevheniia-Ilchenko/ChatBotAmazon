from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI


model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4,)

response = model.invoke("Write a joke about people")
print(response)
