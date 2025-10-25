from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Path to your PDF file
pdf_path = "/home/mani-rathnam-bakaram/Documents/pdfqa/QuestNST LMS_ Complete User Guide by Role.pdf"

# Load the PDF
loader = PyPDFLoader(pdf_path)
docs = loader.load()

print(f"Total pages loaded: {len(docs)}")

# Recommended chunk settings for your use case
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # Each chunk has ~800 characters
    chunk_overlap=80,   # 12% overlap
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

# Split into chunks
chunks = text_splitter.split_documents(docs)

# Show some chunk details
print(f"Total chunks created: {len(chunks)}\n")

# Preview first few chunks
for i, chunk in enumerate(chunks[:5]):
    print(f"--- Chunk {i+1} ---")
    print(chunk.page_content[:300] + "...")
    print(f"Length: {len(chunk.page_content)} characters")
    print(f"Source page: {chunk.metadata.get('page', 'N/A')}\n")
