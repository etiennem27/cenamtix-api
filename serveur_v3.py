import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gensim.models import KeyedVectors

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_FILE = "frWac_no_postag_phrase_500_cbow_cut100.bin"
MODEL_URL = "http://embeddings.net/embeddings/frWac_no_postag_phrase_500_cbow_cut100.bin"

model = None

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_FILE):
        print("Téléchargement du modèle français (300 Mo)... Cela peut prendre quelques minutes.")
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_FILE, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Téléchargement terminé !")
    
    print("Chargement du modèle en mémoire...")
    model = KeyedVectors.load_word2vec_format(MODEL_FILE, binary=True)
    print("Modèle prêt ! Le serveur écoute sur http://localhost:8000")

@app.get("/score")
def get_score(target: str, word: str):
    global model
    if model is None:
        return {"error": "Modèle en cours de chargement", "sim": 0}
    
    target = target.lower().strip()
    word = word.lower().strip()
    
    if target == word:
        return {"sim": 1.0}

    # Fonction de secours pour gérer les noms composés et les espaces
    def compute_similarity(t, w):
        # Word2Vec gère souvent les phrases avec des underscores
        t_under = t.replace(' ', '_')
        w_under = w.replace(' ', '_')
        
        # 1. Essai direct avec underscores
        try:
            return model.similarity(t_under, w_under)
        except KeyError:
            pass

        # 2. Si le joueur tape un nom composé (ex: "simone weil") et la cible est simple ("weil")
        if ' ' in w and ' ' not in t:
            w_last = w.split()[-1] # On prend le dernier mot (le nom de famille)
            try:
                return model.similarity(t, w_last)
            except KeyError:
                pass

        # 3. Si la cible est composée (ex: "marc aurele") et le mot simple ("aurele")
        if ' ' in t and ' ' not in w:
            t_last = t.split()[-1]
            try:
                return model.similarity(t_last, w)
            except KeyError:
                pass

        # 4. Essai avec des espaces (au cas où)
        try:
            return model.similarity(t, w)
        except KeyError:
            return -1.0

    sim = compute_similarity(target, word)
    return {"sim": float(sim)}