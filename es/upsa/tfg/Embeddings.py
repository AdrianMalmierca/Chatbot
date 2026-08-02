import json
import numpy as np
import faiss
from openai import OpenAI
import os
import pickle
import re
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "output_clean3.json")
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.bin")
DOCS_PATH = os.path.join(BASE_DIR, "doc_metadata.pkl")
CHUNK_MAP_PATH = os.path.join(BASE_DIR, "chunk_doc_map.pkl")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings.npy")
client = OpenAI(api_key=OPENAI_API_KEY)

#to balance precision, context and cost
CHUNK_SIZE = 1500      #max size of each chunk
CHUNK_OVERLAP = 200    #overlap between chunks

def cargar_json():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def dividir_en_chunks(texto, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Divide de text into chunks respecting the end of the sentence, with overlap between them."""
    if len(texto) <= chunk_size: #if its small return just one chunk
        return [texto]

    #Cut for phrase (point after space)
    frases = re.split(r'(?<=[.!?])\s+', texto)

    chunks = []
    actual = ""
    for frase in frases:
        if len(actual) + len(frase) + 1 <= chunk_size:
            actual = f"{actual} {frase}".strip() #if the phrase fix we add it
        else:
            if actual:
                chunks.append(actual) #save the chunk
            #the new chunk starts with the overlap of the last one
            solape = actual[-overlap:] if overlap and actual else "" #take the last 200 characters
            actual = f"{solape} {frase}".strip()
    if actual:
        chunks.append(actual) #saves the last chunk

    return chunks

def construir_chunks_documento(item):
    """Generate one or more text chunks for a document, with the title always included in each chunk
    (so that the embedding does not lose track of which document is being processed when the summary is long)."""
    titulo = item.get("title", "")
    autores = " ".join(item.get("authors", []))
    resumen = item.get("summary", "") or ""

    if not resumen.strip():
        #without summary, just one chunk with the information available
        return [f"{titulo}\n{autores}".strip()]

    trozos_resumen = dividir_en_chunks(resumen) #split the summary
    return [f"{titulo}\n{autores}\n{trozo}".strip() for trozo in trozos_resumen] #create a chunk for each part of the summary


def generar_embeddings(textos):
    embeddings = []
    for i, chunk in enumerate(textos):
        print(f"Embedding {i + 1}/{len(textos)}")
        emb = client.embeddings.create(input=chunk, model="text-embedding-3-small")
        vector = emb.data[0].embedding
        embeddings.append(vector)
    return np.array(embeddings, dtype="float32")


def main():
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        print("El índice ya existe. Elimina los archivos si deseas regenerarlo.")
        return

    data = cargar_json()

    #Generate the chunks y, in parallel, the chunk-to-parent-document-index map
    todos_los_chunks = [] #docs
    chunk_a_doc = [] #index

    for doc_idx, item in enumerate(data):
        chunks = construir_chunks_documento(item) #get the chunk of each document
        for chunk_texto in chunks:
            todos_los_chunks.append(chunk_texto) #save each chunk
            chunk_a_doc.append(doc_idx) #save to which document is the chunk

    print(f"{len(data)} documentos divididos en {len(todos_los_chunks)} chunks")

    embeddings = generar_embeddings(todos_los_chunks)

    print("Guardando FAISS index...")
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH) #save the index into the disk

    #doc_metadata.pkl saves the original documents
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(data, f)

    #chunk_doc_map.pkl saves for each row of the index FAISS, to which document is
    with open(CHUNK_MAP_PATH, "wb") as f:
        pickle.dump(chunk_a_doc, f)

    #save the embeddings
    np.save(EMBEDDINGS_PATH, embeddings)
    print("Index creado y guardado.")


if __name__ == "__main__":
    main()