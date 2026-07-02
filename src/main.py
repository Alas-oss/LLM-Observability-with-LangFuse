import os
import time
from dotenv import load_dotenv
from langfuse import Langfuse
from openai import OpenAI

load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

openai_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

try:
    langfuse_prompt = langfuse.get_prompt("rag_system_prompt")
    SYSTEM_PROMPT = langfuse_prompt.get_langchain_prompt()
except Exception:
    SYSTEM_PROMPT = "You are a helpful assistant. Answer based on provided context: {context}"

def basic_retrieve(query: str):
    return [{"text": "Sample vector chunk context matching " + query, "score": 0.95}]

def rag_query(user_query: str, user_id: str) -> str:
    trace = langfuse.trace(
        name="rag_query", 
        input=user_query, 
        user_id=user_id
    )

    retrieval_span = trace.span(name="retrieval", input=user_query)
    chunks = basic_retrieve(user_query)
    retrieval_span.end(output={"chunks": chunks, "count": len(chunks)})

    context_str = "\n".join([c["text"] for c in chunks])
    formatted_prompt = f"Context: {context_str}\n\n Question: {user_query}"

    generation = trace.generation(
        name="llm_call",
        model="openai/gpt-oss-120b",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": formatted_prompt}
        ]
    )

    response = openai_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": formatted_prompt}
        ],
        temperature=0.1
    )

    answer = response.choices[0].message.content

    generation.end(
        output=answer,
        usage={
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens
        }
    )

    trace.update(output=answer)
    return answer

if __name__ == "__main__":
    test_queries = [f"Test user question number {i}" for i in range(1, 21)]
    for idx, q in enumerate(test_queries):
        print(f"Running query {idx+1}/20...")
        rag_query(q, user_id=f"user_{idx}")
