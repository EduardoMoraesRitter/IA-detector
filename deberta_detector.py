"""
DeBERTa AI Text Detector — Camada de ML para deteccao de IA.

Usa modelo pre-treinado DeBERTa-v3 do Hugging Face para classificar
texto como AI-generated vs Human-written, por sentenca e por documento.

Complementa o detector estatistico (detector.py) com analise baseada em ML.
"""

import re
import os

# Cache dir para modelos (evita re-download)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(
    os.path.dirname(__file__), ".model_cache"
))

_model = None
_tokenizer = None
_MODEL_NAME = "vraj33/ai-text-detector-deberta"
_AVAILABLE = None


def is_available():
    """Verifica se PyTorch e transformers estao instalados."""
    global _AVAILABLE
    if _AVAILABLE is not None:
        return _AVAILABLE
    try:
        import torch
        import transformers
        _AVAILABLE = True
    except ImportError:
        _AVAILABLE = False
    return _AVAILABLE


def _load_model():
    """Carrega modelo DeBERTa (lazy loading — so carrega na primeira chamada)."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    if not is_available():
        raise RuntimeError("PyTorch/transformers nao instalado. Run: pip install transformers torch")

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    print(f"[DeBERTa] Carregando modelo {_MODEL_NAME}...")
    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    _model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
    _model.eval()
    print(f"[DeBERTa] Modelo carregado com sucesso.")
    return _model, _tokenizer


def detectar_ia(texto, threshold=0.5):
    """Classifica texto inteiro como AI ou Human usando DeBERTa.

    Retorna dict com:
    - label: 'ai' ou 'human'
    - ai_score: float 0-1 (probabilidade de ser AI)
    - human_score: float 0-1
    - confidence: 'high', 'medium', 'low'
    """
    import torch

    model, tokenizer = _load_model()

    inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

    # Pegar labels do modelo
    id2label = model.config.id2label if hasattr(model.config, 'id2label') else {0: 'human', 1: 'ai'}

    scores = {}
    for idx, label in id2label.items():
        scores[label.lower()] = probs[0][idx].item()

    ai_score = scores.get('ai', scores.get('ai-generated', scores.get('fake', 0)))
    human_score = scores.get('human', scores.get('human-written', scores.get('real', 0)))

    # Se labels nao batem, assumir idx 0=human, 1=ai
    if ai_score == 0 and human_score == 0:
        human_score = probs[0][0].item()
        ai_score = probs[0][1].item()

    label = 'ai' if ai_score > threshold else 'human'

    if max(ai_score, human_score) > 0.9:
        confidence = 'high'
    elif max(ai_score, human_score) > 0.7:
        confidence = 'medium'
    else:
        confidence = 'low'

    return {
        'label': label,
        'ai_score': ai_score,
        'human_score': human_score,
        'confidence': confidence,
        'score_pct': round(ai_score * 100, 1),
    }


def detectar_por_sentenca(texto):
    """Classifica cada sentenca individualmente com DeBERTa.

    Retorna lista de dicts, cada um com:
    - sentenca: str
    - label: 'ai' ou 'human'
    - ai_score: float 0-1
    - confidence: str
    - score_pct: float (0-100)
    """
    # Split em sentencas
    sentencas = re.split(r'(?<=[.!?])\s+', texto.strip())
    sentencas = [s.strip() for s in sentencas if s.strip() and len(s.split()) >= 3]

    if not sentencas:
        return []

    resultados = []
    for sent in sentencas:
        resultado = detectar_ia(sent)
        resultado['sentenca'] = sent
        resultados.append(resultado)

    return resultados


def detectar_completo(texto):
    """Analise completa: documento inteiro + por sentenca.

    Retorna dict com:
    - overall: resultado do documento inteiro
    - sentencas: lista de resultados por sentenca
    - stats: estatisticas agregadas
    """
    overall = detectar_ia(texto)
    sentencas = detectar_por_sentenca(texto)

    # Stats
    n_ai = sum(1 for s in sentencas if s['label'] == 'ai')
    n_human = sum(1 for s in sentencas if s['label'] == 'human')
    total = len(sentencas)

    return {
        'overall': overall,
        'sentencas': sentencas,
        'stats': {
            'total_sentencas': total,
            'ai_count': n_ai,
            'human_count': n_human,
            'ai_pct': round(n_ai / total * 100) if total else 0,
            'human_pct': round(n_human / total * 100) if total else 0,
        },
    }


# ---------------------------------------------------------------------------
# CLI para testes
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if not is_available():
        print("ERRO: PyTorch/transformers nao instalado.")
        print("Run: pip install transformers torch")
        sys.exit(1)

    if len(sys.argv) > 1:
        texto = ' '.join(sys.argv[1:])
    else:
        print("Cole o texto (linha vazia pra finalizar):")
        lines = []
        try:
            while True:
                line = input()
                if line == "" and lines:
                    break
                lines.append(line)
        except EOFError:
            pass
        texto = '\n'.join(lines).strip()

    if not texto:
        print("Texto vazio.")
        sys.exit(1)

    print(f"\n[*] Texto: {len(texto)} chars, {len(texto.split())} palavras\n")

    # Analise completa
    result = detectar_completo(texto)

    print("=" * 60)
    print("  DEBERTA AI DETECTION")
    print("=" * 60)
    print(f"\n  OVERALL: {result['overall']['score_pct']}% AI")
    print(f"  Label:   {result['overall']['label'].upper()}")
    print(f"  Conf:    {result['overall']['confidence']}")
    print(f"\n  Sentencas: {result['stats']['total_sentencas']}")
    print(f"  AI:     {result['stats']['ai_count']} ({result['stats']['ai_pct']}%)")
    print(f"  Human:  {result['stats']['human_count']} ({result['stats']['human_pct']}%)")
    print()

    for i, s in enumerate(result['sentencas'], 1):
        preview = s['sentenca'][:60] + '...' if len(s['sentenca']) > 60 else s['sentenca']
        label = "AI" if s['label'] == 'ai' else "Human"
        print(f"  {i:2d}. [{label:5s}] {s['score_pct']:5.1f}% | {preview}")

    print()
