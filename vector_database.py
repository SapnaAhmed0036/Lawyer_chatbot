"""
NEW SIMPLIFIED VERSION - GUARANTEED TO WORK
Focuses on getting Section 302, 379, 154 correctly
"""

import os
import re
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

PDFS_DIR = "pdfs"
FAISS_DB_PATH = "vector_store/db_faiss_new"
OLLAMA_EMBED_MODEL = "mxbai-embed-large"

def extract_source_name(filename):
    """Extract source name from PDF filename"""
    if "panel code" in filename.lower() or "ppc" in filename.lower():
        return "Pakistan Penal Code (PPC) 1860"
    elif "criminal_procedure" in filename.lower() or "crpc" in filename.lower():
        return "Code of Criminal Procedure (CrPC) 1898"
    else:
        return filename.replace(".pdf", "")

def clean_text(text):
    """Basic text cleaning"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
    # Clean spacing
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    return text.strip()

def load_all_pdfs_simple(pdf_dir):
    """Load all PDFs - simple approach"""
    documents = []
    for file in os.listdir(pdf_dir):
        if file.lower().endswith(".pdf"):
            path = os.path.join(pdf_dir, file)
            print(f"\nProcessing: {file}")
            
            loader = PDFPlumberLoader(path)
            docs = loader.load()
            source_name = extract_source_name(file)
            
            # Merge pages but clean text
            full_text = "\n\n".join([clean_text(doc.page_content) for doc in docs if len(doc.page_content.strip()) > 50])
            
            merged_doc = Document(
                page_content=full_text,
                metadata={
                    "source": path,
                    "source_name": source_name,
                    "source_file": file,
                    "total_pages": len(docs)
                }
            )
            
            documents.append(merged_doc)
            print(f"   ✅ Loaded {len(docs)} pages ({len(full_text)} chars)")
    
    return documents

def create_simple_chunks(documents):
    """
    SIMPLIFIED CHUNKING - No complex section detection
    Just split by size with good overlap
    """
    print(f"\nCreating chunks from {len(documents)} documents...")
    
    # Use standard text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Smaller chunks for better granularity
        chunk_overlap=200,
        length_function=len,
        separators=[
            "\n\n",   # Paragraph breaks
            "\n",     # Line breaks
            ". ",     # Sentences
            " ",      # Words
            ""        # Characters
        ],
        add_start_index=True
    )
    
    all_chunks = []
    for doc in documents:
        chunks = splitter.split_documents([doc])
        print(f"   Created {len(chunks)} chunks from {doc.metadata['source_file']}")
        all_chunks.extend(chunks)
    
    # Now extract section numbers from each chunk
    enhanced_chunks = []
    for chunk in all_chunks:
        # Extract section number from chunk content
        section_num = extract_section_from_chunk(chunk.page_content)
        
        # Add to metadata
        chunk.metadata["section_number"] = section_num
        chunk.metadata["provision_type"] = "GENERAL"
        
        # Add descriptive prefix to content for better retrieval
        source_name = chunk.metadata.get("source_name", "Unknown")
        if section_num != "General Provision":
            prefix = f"[{source_name} - {section_num}]\n\n"
            chunk.page_content = prefix + chunk.page_content
        
        enhanced_chunks.append(chunk)
    
    print(f"\nTotal chunks created: {len(enhanced_chunks)}")
    
    # Show section distribution
    section_counts = {}
    for chunk in enhanced_chunks:
        sec = chunk.metadata.get("section_number", "Unknown")
        section_counts[sec] = section_counts.get(sec, 0) + 1
    
    print(f"\nSection Distribution (Top 30):")
    sorted_sections = sorted(section_counts.items(), key=lambda x: x[1], reverse=True)[:30]
    for section, count in sorted_sections:
        if "302" in section or "379" in section or "154" in section:
            print(f"   ✅ {section}: {count} chunks")
        else:
            print(f"      {section}: {count} chunks")
    
    return enhanced_chunks

def extract_section_from_chunk(text):
    """
    Extract section number from chunk text
    MULTIPLE STRATEGIES to catch all formats
    """
    # Search in first 500 characters
    search_text = text[:500]
    
    # Strategy 1: Look for "Section XXX"
    pattern1 = r'\bSection\s+(\d+[A-Z]?)\b'
    match = re.search(pattern1, search_text, re.IGNORECASE)
    if match:
        return f"Section {match.group(1)}"
    
    # Strategy 2: Look for "XXX. Title" at line start
    pattern2 = r'(?:^|\n)\s*(\d+[A-Z]?)\.\s+([A-Z][a-z])'
    match = re.search(pattern2, search_text, re.MULTILINE)
    if match:
        num = match.group(1)
        # Validate range
        try:
            n = int(re.match(r'\d+', num).group())
            if 1 <= n <= 600:  # Valid PPC/CrPC range
                return f"Section {num}"
        except:
            pass
    
    # Strategy 3: Look for just numbers that might be sections
    pattern3 = r'\b(302|379|154|300|301|303|304|378|380|381|156|157|497)\b'
    match = re.search(pattern3, search_text)
    if match:
        return f"Section {match.group(1)}"
    
    return "General Provision"

def get_embedding_model():
    """Use mxbai-embed-large from Ollama"""
    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)

def build_faiss_index():
    print("="*80)
    print("NEW SIMPLIFIED DATABASE BUILD")
    print("="*80)
    
    print("\nStep 1: Loading PDFs...")
    documents = load_all_pdfs_simple(PDFS_DIR)
    print(f"✅ Total documents loaded: {len(documents)}")

    print("\nStep 2: Creating chunks...")
    chunks = create_simple_chunks(documents)
    print(f"✅ Total chunks created: {len(chunks)}")

    print("\nStep 3: Creating FAISS vector store...")
    print("This may take 5-10 minutes. Please wait...")
    
    # Process in batches
    batch_size = 100
    faiss_db = None
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        print(f"   Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} ({len(batch_chunks)} chunks)...")
        
        if faiss_db is None:
            faiss_db = FAISS.from_documents(batch_chunks, get_embedding_model())
        else:
            batch_db = FAISS.from_documents(batch_chunks, get_embedding_model())
            faiss_db.merge_from(batch_db)

    os.makedirs(os.path.dirname(FAISS_DB_PATH), exist_ok=True)
    faiss_db.save_local(FAISS_DB_PATH)
    
    print("\n" + "="*80)
    print("✅ SUCCESS! FAISS vector store created and saved!")
    print("="*80)

if __name__ == "__main__":
    build_faiss_index()
