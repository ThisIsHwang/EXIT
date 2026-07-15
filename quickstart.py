from exit_rag import ExitRAG, Document

# Initialize pipeline
rag = ExitRAG(
    compression_base_model="google/gemma-2b-it",
    compression_model="doubleyyh/exit-gemma-2b",
    reader_model="meta-llama/Llama-3.1-8B-Instruct",
)

# Example query and document
query = "How do solid-state drives (SSDs) improve computer performance?"
documents = [
    Document(
        title="Computer Storage Technologies",
        text="""
    Solid-state drives use flash memory to store data without moving parts.
    Unlike traditional hard drives, SSDs have no mechanical components.
    The absence of physical movement allows for much faster data access speeds.
    I bought my computer last week.
    SSDs significantly reduce boot times and application loading speeds.
    They consume less power and are more reliable than mechanical drives.
    The price of SSDs has decreased significantly in recent years.
    """,
    )
]

# Run RAG pipeline with compression
result = rag.run_rag(query, documents)

# Print results
print("\nQuery:", result["query"])
print("\nCompressed Context:", result["compressed_context"])
print("\nAnswer:", result["answer"])
print(f"\nCompression Time: {result['compression_time']:.2f}s")
print(f"Reading Time: {result['reading_time']:.2f}s")
print(f"Total Time: {result['total_time']:.2f}s")
