# chatbot_utils.py
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# === Global Containers ===
_ASTRO_DATA = {}
_FAISS_INDEX = None
_CHUNKS = []
_EMBEDDER = None
_LLM = None

# === 1. Load everything (now accepts precomputed data) ===
def load_embeddings_and_model(
    embeddings: np.ndarray,
    chunks: list,
    astro_data: dict,
    embedder_name: str = 'all-MiniLM-L6-v2',
    llm_name: str = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
):
    """
    Initialize chatbot with precomputed embeddings, chunks, and astro data.
    Builds FAISS index in-memory and loads embedder and LLM.
    """
    global _ASTRO_DATA, _FAISS_INDEX, _CHUNKS, _EMBEDDER, _LLM

    # assign astro reference data
    _ASTRO_DATA = astro_data

    # assign chunks and embeddings
    _CHUNKS = chunks
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    _FAISS_INDEX = index

    # load embedder
    _EMBEDDER = SentenceTransformer(embedder_name)

    # load LLM pipeline
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    model = AutoModelForCausalLM.from_pretrained(llm_name)
    model.eval()
    _LLM = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device=-1
    )

# === 2. Formula calculation ===
def calculate_formula(question: str) -> str:
    # stubbed formula parser/evaluator
    try:
        return str(eval(question, {}, {}))
    except Exception:
        return "[could not calculate]"

# === 3. Intent Router + Query ===
def query_text(question: str) -> dict:
    """
    Route text query: uses astro data for specific intents, otherwise FAISS+LLM.
    Returns dict {type, answer}.
    """
    q = question.lower()

    # A) Calculation intent
    if re.search(r"\bcalculate\b|\bcompute\b|\benergy\b|\bohm\b", q):
        return {"type":"calculation", "answer": calculate_formula(question)}

    # B) Next solar/lunar eclipse
    if "next solar eclipse" in q:
        solar = _ASTRO_DATA.get("eclipses", {}).get("solar", {})
        return {"type":"eclipse", "answer": f"Next solar eclipse: {solar.get('date')}"}
    if "next lunar eclipse" in q:
        lunar = _ASTRO_DATA.get("eclipses", {}).get("lunar", {})
        return {"type":"eclipse", "answer": f"Next lunar eclipse: {lunar.get('date')}"}

    # C) Next full/new moon
    if "next full moon" in q:
        fm = next(p for p in _ASTRO_DATA.get("moon_phases", []) if p["phase"]=="Full Moon")
        return {"type":"moon_phase", "answer": f"Next full moon: {fm['datetime']}"}
    if "next new moon" in q:
        nm = next(p for p in _ASTRO_DATA.get("moon_phases", []) if p["phase"]=="New Moon")
        return {"type":"moon_phase", "answer": f"Next new moon: {nm['datetime']}"}

    # D) Upcoming events
    if "event" in q:
        evs = _ASTRO_DATA.get("events", [])[:5]
        lines = "\n".join(f"{e['date']}: {e.get('event',e)}" for e in evs)
        return {"type":"events", "answer": lines}

    # E) Tomorrow's forecast
    if "tomorrow" in q and "sky" in q:
        df = _ASTRO_DATA.get("forecast_daily", [])
        if len(df)>1:
            tomo = df[1]
            return {"type":"forecast", "answer": f"Tomorrow ({tomo['date']}): {tomo['label']} conditions"}

    # F) Fallback: FAISS + LLM
    if _EMBEDDER is None or _FAISS_INDEX is None:
        return {"type":"error", "answer": "Chatbot not initialized."}
    q_emb = _EMBEDDER.encode([question])
    D, I = _FAISS_INDEX.search(q_emb, 3)
    context = "\n".join(_CHUNKS[i] for i in I[0])
    prompt = (f"You are a helpful astronomy assistant.\n\nContext:\n{context}\n\n"
              f"Question: {question}\nAnswer briefly:")
    out = _LLM(prompt, max_new_tokens=150, do_sample=False)[0]['generated_text']
    answer = out.split("Answer:")[-1].strip()
    return {"type":"text", "answer": answer}
