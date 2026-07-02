import os
import time
from dotenv import load_dotenv
from langfuse import Langfuse
from openai import OpenAI

load_dotenv()

langfuse = Langfuse(
    public_key=os.getnev("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)
openai_client = OpenAI(api_key=os.getnev("OPENAI_API_KEY"))

try:
    langfuse_prompt = langfuse.get_prompt("rag_system_prompt")
    SYSTEM_PROMPT = langfuse_prompt.get_langchain_prompt()
except:
    SYSTEM_PROMPT = "You are a helpful assistant. Answer based on provided context: {context}"

def basic_retrieve(query: str):
    return [{"text": "Sample vector chunk context matching " + query, "score": 0.95}]

def rag_query(user_query: str, user_id: str) -> str:
    trace = langfuse.trace(name="rag_query", input=user_query, user_query=user_id)

    retrieval_span = trace.span(name="retrieval", input=user_query)
    start_time = time.time()
    chunks = basic_retrieve(user_query)
    retrieval_span.end(output={"chunks": chunks, "count": len(chunks)})

    context_str = "\n".join([c["text"] for c in chunks])
    formatted_prompt = f"Context: {context_str}\n\n Question: {user_query}"

    generation = trace.generation(
        name="llm_call",
        model="gpt-4o",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": formatted_prompt}
        ]
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": formatted_prompt}
        ],
        temperature=0.1
    )

    generation.end(
        output=response.choices[0].message.content,
        usage={
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens
        }
    )

    trace.update(output=response.choices[0].message.content)
    return response.choices[0].message.content

if __name__=="__main__":
    test_queries = [f"Test user question number {i}" for i in range(1, 21)]
    for idx, q in enumerate(test_queries):
        print(f"Running query {idx+1}/20...")
        rag_query(q, user_id=f"user_{idx}")