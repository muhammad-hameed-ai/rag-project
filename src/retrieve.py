from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore


def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="saudi_vision_2030",
        url="http://localhost:6333",
    )
    return vector_store


if __name__ == "__main__":
    print("Starting retrieval test...")
    store = get_retriever()
    print("Connected to Qdrant collection.")

    query = "What are the main economic goals?"
    print(f"Searching for: {query}")

    results = store.similarity_search(query, k=3)
    print(f"Number of results returned: {len(results)}")

    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(doc.page_content[:300])
        print(f"Source: {doc.metadata.get('source', 'unknown')}")