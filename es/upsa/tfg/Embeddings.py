import json
import numpy as np
import faiss
from openai import OpenAI
import os
import pickle

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
JSON_PATH = "/Users/administrador/Documents/Phyton/RagLlm/es/upsa/tfg/output_clean3.json"
INDEX_PATH = "faiss_index.bin"
DOCS_PATH = "doc_metadata.pkl"
EMBEDDINGS_PATH = "embeddings.npy"
client = OpenAI(api_key=OPENAI_API_KEY)

def cargar_json():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

MAX_CHARS = 4000

def construir_documento(item):
    campos = [
        item.get("title", ""),
        item.get("summary", ""),
        " ".join(item.get("authors", []))
    ]
    texto = "\n".join([str(c) for c in campos if c])
    return texto[:MAX_CHARS]


def generar_embeddings(textos):
    embeddings = []
    for i, chunk in enumerate(textos):
        print(f"Embedding {i+1}/{len(textos)}")
        emb = client.embeddings.create(input=chunk, model="text-embedding-3-small")
        vector = emb.data[0].embedding
        embeddings.append(vector)
    return np.array(embeddings, dtype="float32")

def main():
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        print("El índice ya existe. Elimina los archivos si deseas regenerarlo.")
        return

    data = cargar_json()
    textos = [construir_documento(item) for item in data]
    embeddings = generar_embeddings(textos)

    print("Guardando FAISS index...")
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)

    with open(DOCS_PATH, "wb") as f:
        pickle.dump(data, f)

    np.save(EMBEDDINGS_PATH, embeddings)
    print("Index creado y guardado.")

if __name__ == "__main__":
    main()