import pickle
import random
from collections import defaultdict

DOCS_PATH = "doc_metadata.pkl"
N_CASOS_TITULO = 10
N_CASOS_AUTOR = 5
SEED = 42

def cargar_docs():
    with open(DOCS_PATH, "rb") as f:
        return pickle.load(f)


def generar_casos_por_titulo(docs, n):
    random.seed(SEED)
    candidatos = [d for d in docs if d.get("title", "").strip()]
    #with min if we ask for 10, but theres only 7, takes 7
    muestra = random.sample(candidatos, min(n, len(candidatos))) #random sample choose the document without repeat
    return [
        {
            "pregunta": d["title"],
            "titulos_esperados": [d["title"]],
        }
        for d in muestra
    ]


def generar_casos_por_autor(docs, n):
    autor_a_titulos = defaultdict(list)
    for d in docs:
        for autor in d.get("authors", []):
            autor_a_titulos[autor.strip()].append(d["title"])

    #only authors with more than one publication
    autores_prolificos = {}
    for a, t in autor_a_titulos.items():
        if len(t) >= 2:
            autores_prolificos[a] = t

    random.seed(SEED)
    muestra_autores = random.sample(
        list(autores_prolificos.keys()), min(n, len(autores_prolificos)) #get the name of the authors
    )

    return [
        {
            "pregunta": f"¿Qué artículos ha publicado {autor}?",
            "titulos_esperados": autores_prolificos[autor],
        }
        for autor in muestra_autores
    ]


def main():
    docs = cargar_docs()
    casos_titulo = generar_casos_por_titulo(docs, N_CASOS_TITULO)
    casos_autor = generar_casos_por_autor(docs, N_CASOS_AUTOR)
    todos = casos_titulo + casos_autor

    print("CASOS_DE_PRUEBA = [")
    for caso in todos:
        print(f"    {{")
        print(f"        \"pregunta\": {caso['pregunta']!r},")
        print(f"        \"titulos_esperados\": {caso['titulos_esperados']!r},")
        print(f"    }},")
    print("]")


if __name__ == "__main__":
    main()