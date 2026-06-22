import ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore


def get_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="saudi_vision_2030",
        url="http://localhost:6333",
    )


def retrieve_context(store, query, k=3):
    results = store.similarity_search(query, k=k)
    context_parts = []
    sources = []
    for doc in results:
        context_parts.append(doc.page_content)
        source = doc.metadata.get('source', 'unknown')
        if source not in sources:
            sources.append(source)
    return "\n\n".join(context_parts), sources


def build_prompt(context, question):
    return (
        "You are an expert analyst on Saudi Vision 2030 policy documents.\n"
        "Answer the question using ONLY the context provided below.\n"
        "If the answer is not in the context, say: "
        "I cannot find this information in the provided documents.\n"
        "Do not make up information that is not in the context.\n\n"
        "CONTEXT:\n"
        + context
        + "\n\nQUESTION:\n"
        + question
        + "\n\nANSWER:"
    )


def generate_answer(prompt):
    response = ollama.chat(
        model='llama3.2:1b',
        messages=[{'role': 'user', 'content': prompt}],
        options={
            'num_ctx': 2048,
            'num_predict': 512,
        }
    )
    return response['message']['content']


def ask(store, question):
    print(f"\nQuestion: {question}")
    print("-" * 60)
    context, sources = retrieve_context(store, question, k=3)
    prompt = build_prompt(context, question)
    answer = generate_answer(prompt)
    print(f"Answer:\n{answer}")
    print(f"\nSources used:")
    for s in sources:
        print(f"  - {s}")
    print("-" * 60)
    return answer


if __name__ == "__main__":
    print("Loading RAG pipeline...")
    store = get_vector_store()
    print("Pipeline ready. Running test questions...\n")

    test_questions = [
        "What are the main economic goals of Saudi Vision 2030?",
        "How does Vision 2030 plan to reduce dependence on oil?",
        "What role does the private sector play in Vision 2030?"
    ]

    for question in test_questions:
        ask(store, question)