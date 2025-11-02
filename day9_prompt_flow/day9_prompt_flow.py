from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# Step 1: Understand intent
intent_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="Analyze the intent of this user input and explain it briefly:\n\nUser: {user_input}"
)
intent_chain = LLMChain(llm=llm, prompt=intent_prompt, output_key="intent")

# Step 2: Generate structured outline
outline_prompt = PromptTemplate(
    input_variables=["intent"],
    template="Based on this intent: {intent}, create a 3-point outline explaining the steps clearly."
)
outline_chain = LLMChain(llm=llm, prompt=outline_prompt, output_key="outline")

# Step 3: Final conversational reply
response_prompt = PromptTemplate(
    input_variables=["outline"],
    template="Convert this outline into a friendly PowerAI-style response with emojis and clarity:\n\n{outline}"
)
response_chain = LLMChain(llm=llm, prompt=response_prompt, output_key="final_response")

# Combine all chains
overall_chain = SequentialChain(
    chains=[intent_chain, outline_chain, response_chain],
    input_variables=["user_input"],
    output_variables=["final_response"]
)

# Run it!
user_query = "How can I start learning LangChain for my own AI startup?"
result = overall_chain({"user_input": user_query})

print("🤖 PowerAI says:\n", result["final_response"])
