from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

template = """
You are PowerAI, an AI mentor helping students learn LangChain.
User question: {user_input}
Answer clearly in 3 short paragraphs.
"""

prompt = PromptTemplate(
    input_variables=["user_input"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

response = chain.run("What is conversational design in AI?")
print(response)
