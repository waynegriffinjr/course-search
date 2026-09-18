import sys
from sentence_transformers import SentenceTransformer, util

sentences = [
    "In the AI Fundamentals & Orientation module, students learn how modern AI systems are designed, built, and applied in real-world scenarios.",
    "Advanced Python & Data Handling teaches techniques for working with data structures, file handling, and processing the data that powers AI applications.",
    "Databases & SQL covers how to design, query, and manage relational databases that form the foundation of production applications.",
    "REST API Fundamentals introduces HTTP protocols, API design principles, and methods for both consuming and building RESTful services.",
    "FastAPI Development focuses on creating high-performance, production-quality backend APIs using one of Python’s most modern frameworks.",
    "Web Essentials & Streamlit shows how to build interactive user interfaces for AI projects without needing traditional frontend experience.",
    "Applied AI: Embeddings & Retrieval teaches how to move beyond simple prompting by using embeddings and vector databases so systems understand meaning rather than just keywords.",
    "RAG & Docker Deployment covers building Retrieval-Augmented Generation systems grounded in real data and packaging them for professional deployment with Docker.",
    "The Capstone module requires students to design, develop, and deploy a complete, full-stack AI-powered application from scratch.",
    "Throughout the program, students gain hands-on experience with tools and technologies such as Python, FastAPI, SQLAlchemy, Streamlit, embeddings, vector databases, Docker, Git, and related AI engineering practices.",
]

print("Loading model...")

# ── Encode knowledge base ───────────────────────────────────────────────────
print(f"Embedding {len(sentences)} sentences...")

model = SentenceTransformer('all-MiniLM-L6-v2')

embedded_sentences = model.encode(sentences)


engaged = True

while engaged: 
    
    query = input("Write a sentence about what you have learned at Coding Temple: , or 'quit'")
    if query == "quit": sys.exit()

    query_embedding = model.encode(query)

    scores = util.cos_sim(query_embedding, embedded_sentences)[0]
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    print(f"Query: '{query}'\n")
    print("Top 3 results:")
    for rank, (idx, score) in enumerate(ranked[:3], 1):
        print(f"  {rank}. [{score:.4f}] {sentences[idx]}")
