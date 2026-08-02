import faiss
import numpy as np
import pickle
from openai import OpenAI
import re
import tiktoken
import os
from dotenv import load_dotenv
import time

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.bin")
DOCS_PATH = os.path.join(BASE_DIR, "doc_metadata.pkl")
CHUNK_MAP_PATH = os.path.join(BASE_DIR, "chunk_doc_map.pkl")
MAX_TOKENS_CONTEXT = 18000
MAX_TOKENS_HISTORIAL = 32000 - MAX_TOKENS_CONTEXT - 2000  #keeps for system + questions
TOP_K = 8 #nº of documents more relevant to recover on each request
MAX_DOCS_PREVIOS = 15  #max docs per session
client = OpenAI(api_key=OPENAI_API_KEY)
historiales = {} #session history
docs_previos = {} #docs returned on each session
ultima_actividad = {} #keeps the last time each session made a request
SESSION_TIMEOUT = 1800  #30 minutes of inactivity before clean a session

#for the messages of the chat
def contar_tokens(messages, model="gpt-4-turbo"):
    enc = tiktoken.encoding_for_model(model) #get the correct tokenizer for the model
    total = 0
    for m in messages:
        total += 4  #tokens per role, fixed number plus after depending the content
        total += len(enc.encode(m["content"])) #tokenize the message content and count the number of tokens
    return total

#count tokens in a string, to mesure the cost of the context when build it
def contar_tokens_texto(texto, model="gpt-4-turbo"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(texto))

#Charge the index and metadata
def cargar_index():
    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "rb") as f: #get the original documents
        docs = pickle.load(f)
    with open(CHUNK_MAP_PATH, "rb") as f: #load the map
        chunk_a_doc = pickle.load(f)
    return index, docs, chunk_a_doc

#Query embedding
def embed_query(query):
    query_limpia = limpiar_entrada(query)
    emb = client.embeddings.create(input=query_limpia, model="text-embedding-3-small") #API answer

    #fais needs float32
    return np.array(emb.data[0].embedding, dtype="float32").reshape(1, -1) #reshape because fais expects (n_queries, dim_embedding)
    #1 because theres only 1 embedding and -1 to calculate the dimension automatically

def puntuacion_lexica(query, doc):
    """Count how many words of the query appear literally into the title, authors or summary. To renforce the exacts coincidences
    (author names, titles called literally)"""
    query_palabras = set(re.findall(r'\w+', query.lower()))
    texto_doc = " ".join([
        doc.get('title', ''),
        ' '.join(doc.get('authors', [])),
        doc.get('summary', '') or ''
    ]).lower()

    contador = 0
    for palabra in query_palabras:
        if len(palabra) > 2 and palabra in texto_doc:
            contador += 1

    return contador

#Search the docs plus similar
def buscar_documentos(query, index, docs, chunk_a_doc, max_tokens=MAX_TOKENS_CONTEXT, prev_docs=[]):
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
    #index.search search on the CHUNKS (the more similar), so we ask for more candidates of the ones we want in TOP_K,
    #cause maybe some chunks are from the same document
    dists, indices = index.search(vec, min(TOP_K * 3, index.ntotal))

    vistos_idx_doc = set()
    resultados = [] #for the found documents
    for i in indices[0]:
        if i < 0 or i >= len(chunk_a_doc):
            continue
        doc_idx = chunk_a_doc[i]  #transform the chunk index into the document index
        if doc_idx in vistos_idx_doc:
            continue
        vistos_idx_doc.add(doc_idx)
        resultados.append(docs[doc_idx])
        if len(resultados) >= TOP_K:
            break

    #Combine the semantic order of FAISS with the lexical punctuation
    resultados.sort(key=lambda doc: puntuacion_lexica(query, doc), reverse=True)

    usados = set()
    docs_en_contexto = []
    contexto = ""

    #Before was on the contrary, before we add the previous and after the new ones:

    #First the new documents and important for this question
    #we check that the doc answer to the question yer or yes
    for doc in resultados:
        uid = doc_id(doc)
        resumen = construir_resumen(doc)
        if contar_tokens_texto(contexto + resumen) < max_tokens:
            contexto += resumen
            docs_en_contexto.append(doc)
            usados.add(uid)
        else:
            break

    #After, we have tokens, fill with previous docs
    #so the chatbot can keep talking about the previous question if necessary
    for doc in prev_docs:
        uid = doc_id(doc)
        if uid in usados:
            continue
        resumen = construir_resumen(doc)
        if contar_tokens_texto(contexto + resumen) < max_tokens:
            contexto += resumen
            docs_en_contexto.append(doc)
            usados.add(uid)
        else:
            break

    return docs_en_contexto

#Clean to avoid errors in the codification
def limpiar_texto(texto):
    return re.sub(r'[^\x00-\x7F]+', '', texto) #avoid everything that is not ASCII

def limpiar_entrada(texto):
    if isinstance(texto, str):
        #we transform to bytes and to string again to delete bad characters
        return texto.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return texto

#Build the context using the articles selected
def construir_contexto(articulos):
    partes = []

    for art in articulos:
        autores = ", ".join(art.get('authors', []))
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
    texto_limpio = limpiar_texto(texto_total)

    #Cut for tokens
    #is not necessary cause on buscar_documentos() we limite the number of tokens with contar_tokens_texto
    #so when this function receives this documents already filtered, the text is not bigger than the max of tokens
    #is just for security
    enc = tiktoken.encoding_for_model("gpt-4-turbo")
    tokens = enc.encode(texto_limpio)
    if len(tokens) > MAX_TOKENS_CONTEXT:
        tokens = tokens[:MAX_TOKENS_CONTEXT] #take only the first MAX_TOKENS_CONTEXT tokens
        texto_limpio = enc.decode(tokens) #transform the tokens into text

    return texto_limpio

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

    #Dont save the context on the persistent history, because it can consume a lot of tokens, with non relevant context
    #Create a temporal list only to this call to the API:
    # [original system] + [system with the actual context] + [previous turns] + [actual question]
    mensajes_api = (
            [historial[0]]  #original system with fix instructions (Eres un asistente académico experto...)
            + [{"role": "system", "content": limpiar_entrada(prompt_instruccion)}] #RAG context
            + historial[1:]  #previous turns
            + [{"role": "user", "content": query}] #actual question
    )

    while contar_tokens(mensajes_api) > (MAX_TOKENS_HISTORIAL + MAX_TOKENS_CONTEXT) and len(mensajes_api) > 3:
        """
        0 -> fix system
        1 -> system with RAG context
        2 -> first old message of the user
        3 -> old answer
        4 -> next message of the user
        ...
        """
        mensajes_api.pop(2)

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=mensajes_api,
            max_tokens=1500,
            temperature=0.7
        ) #if theres an error we go to the except
    except Exception as e:
        #We dont add to the history if the call fails to dont let a broken turn (user without assistant),
        # to dont disturb the next turn
        print(f"Error llamando a OpenAI: {e}")
        return "Hubo un problema generando la respuesta. Inténtalo de nuevo en unos segundos."

    respuesta = response.choices[0].message.content.strip()

    #On the persistant history only keep the turn user/assistant, never the message of the context
    historial.append({"role": "user", "content": query})
    historial.append({"role": "assistant", "content": respuesta})

    return respuesta

#Charge once the index
index, docs, chunk_a_doc = cargar_index()
print(f"Documentos cargados: {len(docs)}")

def limpiar_sesiones_inactivas():
    ahora = time.time() #actual time in seconds
    #Iterate through all sessions (sid) and their last activity time (t).
    #Store in inactivas only those sessions that have been inactive for more than SESSION_TIMEOUT seconds.
    #so if for example now is 1785354000 and  a = 1785350000, 1785354000 - 1785350000 = 4000, delete (4000>1800)
    #b = 1785354000, 1785354000 - 1785350000 = 500, keep it
    inactivas = [sid for sid, t in ultima_actividad.items() if ahora - t > SESSION_TIMEOUT]
    for sid in inactivas: #iterate each active session
        #none because if for some reason the key doesnt exist, doesnt give error
        historiales.pop(sid, None) #delete the conversation history of the session
        docs_previos.pop(sid, None) #delete the docs related to the session
        ultima_actividad.pop(sid, None) #deletes the record of the last activity from that session. It then disappears completely

def responder(question, session_id):
    limpiar_sesiones_inactivas() #clean the old sessions
    ultima_actividad[session_id] = time.time() #so each time a user send a message, renew the time of activity

    if session_id not in historiales:
        historiales[session_id] = [
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
        docs_previos[session_id] = []

    historial = historiales[session_id]
    prev_docs = docs_previos[session_id]

    nuevos_docs = buscar_documentos(question, index, docs, chunk_a_doc, prev_docs=prev_docs)

    if not nuevos_docs:
        return "No encontré documentos relevantes para eso."

    vistos = set()
    docs_actuales = []

    #the documents that couldn't enter into buscar_documentos because of the break, we add them here on the end
    #so we save them into prev_docs for future occasions
    for d in nuevos_docs + prev_docs:
        uid = f"{d.get('title', '')}_{d.get('year_of_publication', '')}".strip().lower()
        if uid not in vistos:
            docs_actuales.append(d)
            vistos.add(uid)

    #Only keep the latest docs, so the one of the latest queries
    docs_previos[session_id] = docs_actuales[:MAX_DOCS_PREVIOS]

    contexto = construir_contexto(docs_actuales)

    return obtener_respuesta(question, contexto, historial)
"""
#Chat principal
def chatbot():
    print("Asistente Académico con RAG - Escribe 'salir' para terminar.")

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

        #Search documents including the previous ones
        nuevos_docs = buscar_documentos(query, index, docs, chunk_a_doc, prev_docs=docs_previos)

        #Dont generate response if doesnt found relevant documents
        if not nuevos_docs:
            print("No encontré documentos relevantes para eso. ¿Podrías reformular o preguntar sobre otro tema?")
            continue

        #Join without duplicated
        vistos = set()
        docs_combinados = []
        for d in nuevos_docs + docs_previos:  #docs_previos with the value of the previous turn
            uid = f"{d.get('title', '')}_{d.get('year_of_publication', '')}".strip().lower()
            if uid not in vistos:
                docs_combinados.append(d)
                vistos.add(uid)
        docs_previos = docs_combinados  #update now, after use them

        contexto_str = construir_contexto(docs_previos)

        try:
            respuesta = obtener_respuesta(query, contexto_str, historial)
            print(f"\n{respuesta}\n")
        except Exception as e:
            print(f"\nError al generar la respuesta: {e}\n")

if __name__ == "__main__":
    chatbot()
"""