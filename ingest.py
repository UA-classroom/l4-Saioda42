import os
import glob
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import pdfplumber

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from PDF."""
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text.append(f"--- Page {page_num + 1} ---\n{page_text}")
    return "\n".join(text)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append({
            "text": chunk.strip(),
            "start": start,
            "end": end
        })

        start = end - overlap

    return chunks

def find_pdf_files(data_dir: str = "data") -> list[str]:
    """Find all PDF files in data directory."""
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    return sorted(pdf_files)

def ingest_all_pdfs(data_dir: str = "data", db_path: str = "chroma_db"):
    """Process all PDFs in data directory and store in ChromaDB."""

    pdf_files = find_pdf_files(data_dir)

    if not pdf_files:
        print(f"No PDF files found in {data_dir}/")
        return

    print(f"Found {len(pdf_files)} PDF(s):\n")
    for pdf in pdf_files:
        print(f"  • {os.path.basename(pdf)}")
    print()

    print(f"Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Creating ChromaDB at {db_path}...")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name="cypher_rules",
        metadata={"hnsw:space": "cosine"}
    )

    total_chunks = 0
    chunk_id = 0

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"\n Processing: {filename}")

        try:
            text = extract_text_from_pdf(pdf_path)
            print(f"Extracted {len(text):,} characters")

            chunks = chunk_text(text)
            print(f"Created {len(chunks)} chunks")

            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                texts = [c["text"] for c in batch]
                ids = [f"chunk_{chunk_id + j}" for j in range(len(batch))]
                metadatas = [
                    {"source_file": filename, "chunk_index": i + j}
                    for j in range(len(batch))
                ]

                embeddings = model.encode(texts, show_progress_bar=False)

                collection.add(
                    embeddings=embeddings.tolist(),
                    documents=texts,
                    ids=ids,
                    metadatas=metadatas
                )

                chunk_id += len(batch)

            total_chunks += len(chunks)
            print(f"Stored {len(chunks)} chunks")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\n Done!")
    print(f"   Total chunks stored: {total_chunks}")
    print(f"   Location: {db_path}/")
    print(f"   Collection: 'cypher_rules'")

if __name__ == "__main__":
    ingest_all_pdfs()
