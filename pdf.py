"""
now i im about to do some pdf dataset processing and cleaning,ready for embedding and vectorization"""

from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
import json
pdf_path = r"F:\projects\wasteRAG\fol\MUT-all-Programmes-2.pdf"
loader = PyPDFLoader(pdf_path)
loader.raise_for_errors = True
docs = loader.load()
print(f"Loaded {len(docs)} documents from PDF.")


#1 split into school of

# 2️⃣ Combine all pages into one text
full_text = "\n".join(doc.page_content for doc in docs)

# 3️⃣ Split into "School of ..." chunks
schools = full_text.split("SCHOOL OF")
school_chunks = []
for school in schools:
    if not school.strip():
        continue
    school_chunks.append("SCHOOL OF" + school.strip())

# 4️⃣ Further split by "Department of ..."
final_chunks = []
for chunk in school_chunks:
    depts = chunk.split("Department of")
    header = depts[0]  # School heading
    for dept in depts[1:]:
        final_chunks.append(header + "\nDepartment of" + dept.strip())

# 5️⃣ Convert into LangChain Documents with metadata
documents = [
    Document(page_content=text_chunk, metadata={"source": pdf_path})
    for text_chunk in final_chunks
]

print(f"Created {len(documents)} logical chunks (School + Department)")

# 6️⃣ Save as TXT (only the text, each chunk separated by a line)
with open("chunks.txt", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc.page_content + "\n\n")  # blank line between chunks

# 7️⃣ Save as JSON (with metadata)
json_data = [
    {"content": doc.page_content, "metadata": doc.metadata}
    for doc in documents
]
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=4, ensure_ascii=False)

print("✅ Saved chunks to chunks.txt and chunks.json")