import faiss
import numpy as np
import pickle
from openai import OpenAI
import re
import tiktoken

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_PATH = "/Users/administrador/Documents/Phyton/RagLlmTodo/es/upsa/tfg/faiss_index.bin"
DOCS_PATH = "/Users/administrador/Documents/Phyton/RagLlmTodo/es/upsa/tfg/doc_metadata.pkl"
MAX_TOKENS_CONTEXT = 18000
MAX_TOKENS_HISTORIAL = 32000 - MAX_TOKENS_CONTEXT - 2000  # reserva para system + pregunta

client = OpenAI(api_key=OPENAI_API_KEY)

def contar_tokens(messages, model="gpt-4-turbo"):
    enc = tiktoken.encoding_for_model(model)
    total = 0
    for m in messages:
        total += 4  # tokens por role y delimitadores
        total += len(enc.encode(m["content"]))
    return total


#Carga el índice y metadatos
def cargar_index():
    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)
    return index, docs

#Embedding de la pregunta
def embed_query(query):
    query_limpia = limpiar_entrada(query)
    emb = client.embeddings.create(input=query_limpia, model="text-embedding-3-small")
    return np.array(emb.data[0].embedding, dtype="float32").reshape(1, -1)


#Busca documentos más similares
def buscar_documentos(query, index, docs, max_tokens=MAX_TOKENS_CONTEXT, prev_docs=[]):
    def construir_resumen(doc):
        summary = doc.get('summary') or ''
        summary = summary.strip()
        if not summary:
            summary = 'Resumen no disponible. Usa la información del título, autores, año, congreso y tipo para razonar sobre el contenido.'
        return (
            f"TÍTULO: {doc.get('title', '')}\n"
            f"AUTORES: {', '.join(doc.get('authors', []))}\n"
            f"AÑO: {doc.get('year_of_publication', '')}\n"
            f"CONGRESO: {doc.get('congress', '')}\n"
            f"TIPO: {doc.get('type_of_publication', '')}\n"
            f"ISBN: {doc.get('isbn', '') or doc.get('isbn_issn', '')}\n"
            f"RESUMEN: {summary}\n\n"
        )

    def doc_id(doc):
        return f"{doc.get('title', '')}_{doc.get('year_of_publication', '')}".strip().lower()

    vec = embed_query(query)
    dists, indices = index.search(vec, len(docs))
    resultados = [docs[i] for i in indices[0] if i < len(docs)]

    usados = set(doc_id(d) for d in prev_docs)
    docs_en_contexto = []
    contexto = ""

    #Primero añade documentos anteriores si caben
    for doc in prev_docs:
        resumen = construir_resumen(doc)
        if len(contexto + resumen) < max_tokens:
            contexto += resumen
            docs_en_contexto.append(doc)
        else:
            break

    #Luego añade nuevos documentos
    for doc in resultados:
        uid = doc_id(doc)
        if uid in usados:
            continue
        resumen = construir_resumen(doc)
        if len(contexto + resumen) < max_tokens:
            contexto += resumen
            docs_en_contexto.append(doc)
            usados.add(uid)
        else:
            break

    return docs_en_contexto

#Limpia para evitar errores de codificación
def limpiar_texto(texto):
    return re.sub(r'[^\x00-\x7F]+', '', texto)

def limpiar_entrada(texto):
    if isinstance(texto, str):
        return texto.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return texto

#Construye contexto desde los artículos seleccionados
def construir_contexto(articulos):
    partes = []
    autores_acumulados = set()

    for art in articulos:
        autores = ", ".join(art.get('authors', []))
        autores_acumulados.update(art.get('authors', []))
        resumen = (
            f"TÍTULO: {art.get('title', '')}\n"
            f"AUTORES: {autores}\n"
            f"AÑO: {art.get('year_of_publication', '')}\n"
            f"CONGRESO: {art.get('congress', '')}\n"
            f"TIPO: {art.get('type_of_publication', '')}\n"
            f"ISBN: {art.get('isbn', '') or art.get('isbn_issn', '')}\n"
            f"RESUMEN: {art.get('summary', '')}"
        )
        partes.append(resumen)

    texto_total = "\n\n---\n\n".join(partes)
    return limpiar_texto(texto_total[:MAX_TOKENS_CONTEXT])

#Respuesta desde el LLM con historial
def obtener_respuesta(query, contexto, historial):
    query = limpiar_entrada(query)
    contexto = limpiar_entrada(contexto)

    prompt_instruccion = (
        f"Tienes acceso a los siguientes documentos académicos:\n\n{contexto}\n\n"
        f"Debes redactar tu respuesta únicamente usando los datos presentes en estos documentos, como los títulos, autores, resúmenes, etc.\n"
        f"Responde de forma natural, en lenguaje narrativo, como si estuvieras explicando la información a otra persona.\n"
        f"Evita usar listas, encabezados como 'TÍTULO:', 'AUTORES:' o formatos tipo ficha. Integra la información en párrafos completos.\n"
        f"Por ejemplo, en lugar de decir '**Título**: X', escribe 'El artículo titulado X, publicado en el año Y por los autores Z...'\n"
        f"\n"
        f"Tu respuesta debe considerar todos los documentos del contexto. "
        f"Si te preguntan por un autor, analiza todos los artículos en los que aparece y menciona sus coautores. "
        f"Si te preguntan por un tema (como 'educación', 'inteligencia artificial', etc.), busca esas palabras o conceptos relacionados en los títulos o resúmenes y menciona los autores asociados.\n"
        f"\n"
        f"No inventes nada que no esté en los textos. Si no puedes responder con los datos que tienes, dilo claramente.\n"
        f"\n"
        f"Cuando se te pregunte por un autor:\n"
        f"- Enumera TODOS los artículos en los que aparece, indicando título y año.\n"
        f"- Lista TODOS los coautores únicos de esos artículos.\n"
        f"- Si se te pregunta por una colaboración entre dos personas, confirma si existe, y cita los títulos donde ambos aparecen juntos. Si no hay coincidencias, indícalo.\n"
        f"- Si se menciona un autor y no hay resúmenes disponibles, intenta deducir su área de trabajo a partir de los títulos o los coautores.\n"
        f"- Al identificar autores, ten en cuenta que un mismo autor puede estar registrado de diferentes maneras (por ejemplo, 'Ana María Fermoso García', 'Ana Fermoso García', 'Ana M. Fermoso García' o 'María Fermoso García'). Intenta agrupar estas variantes si hay evidencia suficiente (coincidencia de apellidos, coautores comunes, títulos similares, etc.) y considera que podrían referirse a la misma persona. Haz esta agrupación con criterio razonable, sin asumir coincidencias si no hay apoyo en el contexto."
        f"- Lo mismo aplica a nombres invertidos (por ejemplo, 'Vidal Alonso Secades' vs 'Alonso Secades, Vidal') o abreviados. Usa los títulos, coautores o publicaciones para inferir si se trata de la misma persona, y acláralo en la respuesta si no estás seguro."
        f"-Por ejemplo, si en un documento aparece 'Ana María Fermoso García' y en otro 'Ana M. Fermoso García', y ambos artículos son de años cercanos y comparten coautores o temas, considera que podrían referirse a la misma persona. En ese caso, agrúpalos y acláralo como: 'Ana María Fermoso García (también mencionada como Ana M. Fermoso García)'"
        f"-Puedes agrupar nombres similares si existe suficiente evidencia en el contexto (como coincidencia parcial de nombre/apellidos, coautores, temas o publicaciones). No asumas que son diferentes solo porque el nombre no es idéntico."
        f"-Al identificar autores, analiza variantes del nombre que puedan referirse a la misma persona, usando pistas como coautores, año y temática. Por ejemplo, nombres abreviados o invertidos pueden ser la misma persona si coinciden en otros aspectos."
    f"\n"
        f"Organiza tu respuesta como una breve explicación, clara y coherente, sin inventar información que no esté en los documentos."
        f"Cuando se te pregunte por un tema o área (como 'inteligencia artificial', 'educación', etc.), debes identificar los artículos relevantes analizando los títulos y los resúmenes en busca de palabras clave relacionadas. Luego, menciona los autores de esos artículos, evitando repeticiones"
        "Si se te pide contar profesores, ivestigadores, artículos o colaboraciones (por ejemplo, '¿cuántos autores trabajan en IA?'), debes usar únicamente la información disponible en los documentos del contexto, y calcular el número a partir de los datos presentes (listas de autores, coincidencias, etc.)."
        "Si no tienes suficiente información para responder, acláralo explícitamente."
    )

    historial.append({"role": "system", "content": limpiar_entrada(prompt_instruccion)})
    historial.append({"role": "user", "content": query})

    #Limita el historial para no pasarte del máximo
    while contar_tokens(historial) > MAX_TOKENS_HISTORIAL and len(historial) > 2:
        historial.pop(1)

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=historial,
        max_tokens=1500,
        temperature=0.7
    )

    respuesta = response.choices[0].message.content.strip()
    historial.append({"role": "assistant", "content": respuesta})
    return respuesta

#Chat principal
def chatbot():
    print("Asistente Académico con RAG - Escribe 'salir' para terminar.")
    index, docs = cargar_index()

    historial = [
        {
            "role": "system",
            "content": (
                "Eres un asistente académico experto. "
                "Analiza el contexto proporcionado, que incluye artículos científicos con título, autores, resumen, etc. "
                "Tu objetivo es responder en español exclusivamente en base al contenido de estos documentos, sin inventar. "
                "Cuando se haga una pregunta sobre un tema (como 'inteligencia artificial', 'educación', etc.), "
                "busca términos relacionados en los títulos y resúmenes y menciona los autores de los artículos que los tratan. "
                "Cuando se pregunte por un autor, identifica todos los artículos en el contexto en los que aparece, "
                "y lista los coautores únicos, sin limitarte solo al último documento mencionado. "
                "No asumas que las preguntas siempre se refieren al último documento. "
                "Si no hay información suficiente, responde claramente que no está disponible en el contexto."
            )
        }
    ]

    docs_previos = []

    while True:
        query = input("Tú: ").strip()
        if query.lower() == "salir":
            print("¡Hasta luego!")
            break

        if len(query.split()) < 3:
            print("¡Hola! ¿En qué puedo ayudarte hoy?")
            continue

        #Buscar documentos incluyendo los previos
        nuevos_docs = buscar_documentos(query, index, docs, prev_docs=docs_previos)

        #Si no se encuentran documentos relevantes, evita generar respuesta
        if not nuevos_docs:
            print("No encontré documentos relevantes para eso. ¿Podrías reformular o preguntar sobre otro tema?")
            continue

        #Une sin duplicados
        vistos = set()
        docs_previos = []
        for d in nuevos_docs + docs_previos:
            uid = f"{d.get('title', '')}_{d.get('year_of_publication', '')}".strip().lower()
            if uid not in vistos:
                docs_previos.append(d)
                vistos.add(uid)

        contexto_str = construir_contexto(docs_previos)

        try:
            respuesta = obtener_respuesta(query, contexto_str, historial)
            print(f"\n{respuesta}\n")
        except Exception as e:
            print(f"\nError al generar la respuesta: {e}\n")

if __name__ == "__main__":
    chatbot()