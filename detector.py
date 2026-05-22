"""
Detector de Texto Gerado por IA — Analise Estatistica Pura v3
Zero dependencia de modelos. Usa apenas matematica + compressao.

Suporta TEXTO e CODIGO:
- Texto: 15 metricas (entropia, compressao, estilometria, etc.)
- Codigo: extrai linguagem natural (comentarios, strings) + metricas de codigo

Inspirado em: DetectGPT, GLTR, ZipPy (Thinkst), GPTZero, StyloAI
"""

import re
import math
import zlib
import unicodedata
from collections import Counter
from statistics import mean, stdev

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENT_RE = re.compile(r'(?<=[.!?])\s+')

def _frases(texto):
    partes = _SENT_RE.split(texto)
    return [f.strip() for f in partes if f.strip() and len(f.split()) >= 2]

def _palavras(texto):
    return re.findall(r'\b[a-zA-Z]+\b', texto.lower())

def _match_case(original, replacement):
    if not original or not replacement:
        return replacement
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement

def _contar_silabas(palavra):
    palavra = palavra.lower()
    if len(palavra) <= 2:
        return 1
    vogais = "aeiou"
    count = 0
    prev_vogal = False
    for ch in palavra:
        is_vogal = ch in vogais
        if is_vogal and not prev_vogal:
            count += 1
        prev_vogal = is_vogal
    if palavra.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)

# ---------------------------------------------------------------------------
# 1. Entropia Shannon
# ---------------------------------------------------------------------------

def calcular_entropia(texto):
    """
    Shannon entropy da distribuicao de palavras.
    IA: ~6-8 bits (previsivel) | Humano: ~8-10+ bits (mais diverso)
    """
    pals = _palavras(texto)
    if len(pals) < 10:
        return 0.0
    counter = Counter(pals)
    total = len(pals)
    return -sum((c / total) * math.log2(c / total) for c in counter.values())

# ---------------------------------------------------------------------------
# 2. Taxa de compressao (ZipPy)
# ---------------------------------------------------------------------------

def calcular_compressao(texto):
    """
    Ratio compressed/original usando zlib.
    IA comprime melhor porque e mais previsivel e repetitiva.
    IA: 0.25-0.40 | Humano: 0.40-0.60+
    """
    data = texto.encode('utf-8')
    if len(data) < 20:
        return 0.5
    compressed = zlib.compress(data, level=9)
    return len(compressed) / len(data)

# ---------------------------------------------------------------------------
# 3. Burstiness (CV do comprimento das frases)
# ---------------------------------------------------------------------------

def calcular_burstiness(texto):
    """
    Coeficiente de variacao do comprimento das frases.
    IA: 0.1-0.3 (uniforme) | Humano: 0.5-1.0+ (caotico)
    """
    frases = _frases(texto)
    if len(frases) < 3:
        return 0.5
    comprimentos = [len(f.split()) for f in frases]
    m = mean(comprimentos)
    if m == 0:
        return 0.0
    return stdev(comprimentos) / m

# ---------------------------------------------------------------------------
# 4. SD comprimento das frases (absoluto)
# ---------------------------------------------------------------------------

def calcular_sent_len_sd(texto):
    """
    Desvio padrao absoluto do comprimento das frases em palavras.
    THE metric mais discriminativa segundo pesquisas.
    IA: SD 2-5 palavras | Humano: SD 6-12+ palavras
    """
    frases = _frases(texto)
    if len(frases) < 3:
        return 5.0
    comprimentos = [len(f.split()) for f in frases]
    return stdev(comprimentos)

# ---------------------------------------------------------------------------
# 5. Variancia frases adjacentes
# ---------------------------------------------------------------------------

def calcular_var_adjacentes(texto):
    """
    Coeficiente de variacao das razoes entre frases consecutivas.
    IA mantem ritmo constante (ratio ~1.0). Humano varia muito.
    IA: CV < 0.3 | Humano: CV > 0.5
    """
    frases = _frases(texto)
    if len(frases) < 4:
        return 0.5
    comprimentos = [len(f.split()) for f in frases]
    ratios = []
    for i in range(1, len(comprimentos)):
        prev = comprimentos[i - 1]
        if prev > 0:
            ratios.append(comprimentos[i] / prev)
    if len(ratios) < 2:
        return 0.5
    m = mean(ratios)
    if m == 0:
        return 0.0
    return stdev(ratios) / m

# ---------------------------------------------------------------------------
# 6. TTR (Type-Token Ratio)
# ---------------------------------------------------------------------------

def calcular_ttr(texto):
    """
    Palavras unicas / total.
    IA: ~0.45 | Humano: ~0.55+
    """
    pals = _palavras(texto)
    if len(pals) < 5:
        return 0.5
    return len(set(pals)) / len(pals)

# ---------------------------------------------------------------------------
# 7. Hapax Legomena
# ---------------------------------------------------------------------------

def calcular_hapax_ratio(texto):
    """Palavras que aparecem exatamente 1 vez / total."""
    pals = _palavras(texto)
    if len(pals) < 5:
        return 0.5
    contagem = Counter(pals)
    hapax = sum(1 for c in contagem.values() if c == 1)
    return hapax / len(pals)

# ---------------------------------------------------------------------------
# 8. Ratio conteudo / funcao
# ---------------------------------------------------------------------------

FUNCTION_WORDS = frozenset([
    "the", "a", "an", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "is", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "it", "its", "this", "that", "these", "those", "there", "here",
    "not", "no", "nor", "so", "yet", "both", "either", "neither",
    "each", "every", "all", "any", "few", "more", "most", "some",
    "such", "than", "too", "very", "just", "about", "above", "after",
    "before", "between", "during", "into", "through", "under", "until",
    "up", "down", "out", "off", "over", "then", "once", "also", "how",
    "when", "where", "why", "what", "which", "who", "whom", "whose",
    "while", "because", "although", "though", "since", "unless", "whether",
    "am", "my", "me", "i", "you", "your", "yours", "he", "him", "his",
    "she", "her", "hers", "we", "us", "our", "ours", "they", "them",
    "their", "theirs", "itself", "himself", "herself", "themselves",
    "myself", "yourself", "ourselves", "yourselves",
])

def calcular_ratio_conteudo_funcao(texto):
    """
    Content words / function words.
    Humano: ~0.98 | IA: ~1.37 (muito mais noun-heavy)
    """
    pals = _palavras(texto)
    if len(pals) < 10:
        return 1.0
    funcao = sum(1 for p in pals if p in FUNCTION_WORDS)
    conteudo = len(pals) - funcao
    if funcao == 0:
        return 2.0
    return conteudo / funcao

# ---------------------------------------------------------------------------
# 9. Frequencia de contracoes
# ---------------------------------------------------------------------------

_CONTRACTION_RE = re.compile(
    r"\b(?:i'm|i've|i'll|i'd|isn't|aren't|wasn't|weren't|hasn't|haven't|"
    r"hadn't|won't|wouldn't|couldn't|shouldn't|didn't|doesn't|don't|"
    r"can't|mustn't|let's|that's|there's|here's|it's|he's|she's|"
    r"who's|what's|where's|when's|how's|they're|we're|you're|"
    r"they've|we've|you've|they'll|we'll|you'll|they'd|we'd|you'd|"
    r"ain't|gonna|wanna|gotta|y'all)\b",
    re.IGNORECASE,
)

def calcular_contracoes(texto):
    """
    Contracoes por 100 palavras.
    IA: < 0.5 (formal, evita contracoes) | Humano: 2-5+ (natural)
    """
    total_palavras = len(texto.split())
    if total_palavras < 10:
        return 0.0
    hits = len(_CONTRACTION_RE.findall(texto))
    return hits / (total_palavras / 100)

# ---------------------------------------------------------------------------
# 10. Entropia de pontuacao
# ---------------------------------------------------------------------------

def calcular_entropia_pontuacao(texto):
    """
    Shannon entropy dos intervalos entre pontuacoes.
    IA pontua em intervalos regulares (baixa entropia).
    IA: < 2.5 | Humano: > 3.0
    """
    intervalos = []
    palavras_desde_pont = 0
    for token in texto.split():
        palavras_desde_pont += 1
        if re.search(r'[.!?,;:\-]', token):
            intervalos.append(palavras_desde_pont)
            palavras_desde_pont = 0
    if len(intervalos) < 3:
        return 3.0
    counter = Counter(intervalos)
    total = len(intervalos)
    entropy = -sum((c / total) * math.log2(c / total) for c in counter.values())
    return entropy

# ---------------------------------------------------------------------------
# 11. Conectivos IA
# ---------------------------------------------------------------------------

AI_CONNECTORS = [
    "moreover", "furthermore", "additionally", "consequently",
    "nevertheless", "subsequently", "henceforth", "thereby",
    "wherein", "thereof", "notwithstanding", "aforementioned",
    "it is important to note", "it is worth mentioning",
    "it is worth noting", "it should be noted",
    "in today's world", "in today's society", "in today's digital",
    "plays a crucial role", "plays a vital role", "plays a pivotal role",
    "it is essential to", "it is imperative to",
    "has the potential to", "have the potential to",
    "in conclusion", "to summarize", "in summary",
    "on the other hand", "in contrast",
    "as a result", "due to the fact",
    "in order to", "with regard to", "in terms of",
    "a wide range of", "a variety of",
    "significantly", "substantially", "comprehensive",
    "multifaceted", "paradigm", "synergy",
    "leverage", "utilize", "facilitate", "delve",
    "foster", "underscore", "landscape", "realm",
    "embark", "endeavor", "commence",
    "robust", "streamline", "holistic",
    "encompasses", "enhancing", "transformative",
    "noteworthy", "groundbreaking", "pioneering",
    "paramount", "indispensable", "intricate",
    "pivotal", "fundamental", "imperative",
    "conversely", "correspondingly", "inherently",
    "particularly", "specifically", "essentially",
    "arguably", "undeniably", "inevitably",
    "navigating", "spearheading", "harnessing",
    "fostering", "cultivating", "bolstering",
    "underscoring", "epitomizing", "exemplifying",
    "revolutionizing", "transcending", "amalgamating",
]

def calcular_freq_conectivos(texto):
    """
    Conectivos/expressoes IA por sentenca.
    IA: > 0.5/frase | Humano: < 0.2/frase
    """
    texto_lower = texto.lower()
    frases = _frases(texto)
    if not frases:
        return 0.0
    total = sum(1 for c in AI_CONNECTORS if c in texto_lower)
    return total / len(frases)

# ---------------------------------------------------------------------------
# 12. Diversidade inicio de frase
# ---------------------------------------------------------------------------

def calcular_diversidade_inicio(texto):
    """
    Ratio de primeiras palavras unicas / total de frases.
    IA: < 0.6 | Humano: > 0.8
    """
    frases = _frases(texto)
    if len(frases) < 3:
        return 1.0
    inicios = []
    for f in frases:
        palavras = f.split()
        if palavras:
            inicios.append(palavras[0].lower())
    if not inicios:
        return 1.0
    return len(set(inicios)) / len(inicios)

# ---------------------------------------------------------------------------
# 13. Densidade de pronomes (1a pessoa)
# ---------------------------------------------------------------------------

FIRST_PERSON = frozenset(["i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"])
SECOND_PERSON = frozenset(["you", "your", "yours", "yourself", "yourselves"])

def calcular_densidade_pronomes(texto):
    """
    % de pronomes de 1a+2a pessoa no texto.
    IA: < 1.5% (evita perspectiva pessoal) | Humano: > 3%
    """
    pals = _palavras(texto)
    if len(pals) < 10:
        return 3.0
    pessoais = sum(1 for p in pals if p in FIRST_PERSON or p in SECOND_PERSON)
    return (pessoais / len(pals)) * 100

# ---------------------------------------------------------------------------
# 14. Frases tipicas IA (expandido)
# ---------------------------------------------------------------------------

def contar_frases_ia(texto):
    """
    Conta expressoes dead giveaway de texto IA.
    Retorna contagem por 100 palavras.
    """
    texto_lower = texto.lower()
    total_palavras = len(texto.split())
    if total_palavras == 0:
        return 0.0

    dead_giveaways = [
        # Aberturas formulaicas
        "it is important to note",
        "it's important to note",
        "it is worth mentioning",
        "it is worth noting",
        "it should be noted",
        "it is essential to understand",
        "it is crucial to recognize",
        "it goes without saying",
        "needless to say",

        # "In today's..." patterns
        "in today's rapidly",
        "in today's world",
        "in today's society",
        "in today's digital",
        "in today's fast-paced",
        "in today's interconnected",
        "in the modern era",
        "in the digital age",
        "in an era of",
        "in the realm of",
        "in the ever-evolving",
        "in the ever-changing",

        # "plays a ... role"
        "plays a crucial role",
        "plays a vital role",
        "plays a pivotal role",
        "plays an important role",
        "plays a significant role",
        "plays a key role",

        # Overused verbs
        "has revolutionized",
        "have revolutionized",
        "has transformed",
        "have transformed",
        "cannot be overstated",
        "should not be underestimated",

        # Hedging patterns
        "a testament to",
        "serves as a reminder",
        "paving the way",
        "at the forefront",
        "the landscape of",
        "navigating the complexities",
        "a holistic approach",
        "a comprehensive approach",
        "a nuanced understanding",
        "a deeper understanding",

        # "delve" family
        "let's delve",
        "let us delve",
        "delve into",
        "delving into",
        "dive into",
        "diving into",
        "embark on",
        "embarking on",

        # Closings formulaicos
        "in conclusion",
        "to sum up",
        "all in all",
        "at the end of the day",
        "moving forward",
        "going forward",
        "the bottom line is",
        "it remains to be seen",
        "only time will tell",

        # Misc IA-speak
        "a myriad of",
        "a plethora of",
        "a multitude of",
        "a wide array of",
        "a broad spectrum",
        "first and foremost",
        "last but not least",
        "by and large",
        "as we move forward",
        "when it comes to",
        "it is no secret that",
        "there is no denying",
        "stands as a testament",
        "represents a significant",
        "underscores the importance",
        "highlights the need",
        "sheds light on",
        "offers a unique perspective",
        "raises important questions",
    ]

    hits = sum(1 for phrase in dead_giveaways if phrase in texto_lower)
    return hits / (total_palavras / 100)

# ---------------------------------------------------------------------------
# 15. Padroes formulaicos (abertura + fechamento)
# ---------------------------------------------------------------------------

_OPENERS = [
    r"^in today'?s",
    r"^in the (?:modern|digital|contemporary|current)",
    r"^in an? (?:era|age|world)",
    r"^in the realm of",
    r"^in the ever",
    r"^(?:the|a) (?:rapid|growing|increasing|rising)",
    r"^when it comes to",
    r"^it is (?:no secret|widely|commonly|generally)",
    r"^throughout history",
    r"^since the dawn",
    r"^as (?:technology|society|the world)",
    r"^over the (?:past|last) (?:few|several|decade)",
]

_CLOSERS = [
    r"in conclusion",
    r"in summary",
    r"to (?:sum up|summarize|conclude)",
    r"all in all",
    r"(?:looking|moving) forward",
    r"it (?:is|remains) (?:clear|evident)",
    r"ultimately",
    r"by embracing",
    r"as we (?:move|look|navigate)",
    r"the (?:future|path) (?:holds|lies|remains)",
    r"only time will tell",
]

# ---------------------------------------------------------------------------
# Sentence-level AI detection patterns
# ---------------------------------------------------------------------------

_SENTENCE_AI_PHRASES = [
    "it is important to note", "it is worth mentioning", "it is worth noting",
    "it should be noted", "plays a crucial role", "plays a vital role",
    "plays a pivotal role", "in today's world", "in today's society",
    "in the modern era", "in the digital age", "delve into", "embark on",
    "a myriad of", "a plethora of", "has the potential to",
    "first and foremost", "last but not least", "it is essential to",
    "a holistic approach", "a comprehensive approach", "paving the way",
    "at the forefront", "in conclusion", "has revolutionized",
    "cannot be overstated", "in the realm of", "the landscape of",
    "a wide range of", "a variety of", "in order to",
    "navigating the complexities", "a nuanced understanding",
    "serves as a reminder", "a testament to",
]

_COLLOQUIAL_RE = re.compile(
    r'\b(?:like|you know|kinda|sorta|pretty much|a lot of|stuff|things|'
    r'honestly|basically|literally|whatever|anyway|i mean|right\?|'
    r'well,|so,|but hey|i guess|no way|oh well|come on|gonna|wanna|gotta)\b',
    re.IGNORECASE,
)

_SELF_CORRECTION_RE = re.compile(
    r'(?:well actually|or rather|I mean,|wait,|actually,|'
    r'— |no,|hmm|ugh|oops)',
    re.IGNORECASE,
)

_TRANSITION_STARTERS = re.compile(
    r'^\s*(?:moreover|furthermore|additionally|consequently|nevertheless|'
    r'subsequently|however|therefore|thus|hence|accordingly|'
    r'in addition|as a result|for instance|for example|'
    r'on the other hand|in contrast|similarly|likewise|'
    r'in particular|notably|importantly|significantly)\b',
    re.IGNORECASE,
)


def calcular_formulaicos(texto):
    """
    Score 0-1 para padroes formulaicos de abertura/fechamento.
    IA quase sempre abre com "In today's..." e fecha com "In conclusion..."
    """
    frases = _frases(texto)
    if len(frases) < 3:
        return 0.0

    score = 0.0
    primeira = frases[0].lower()
    ultima = frases[-1].lower()

    for pat in _OPENERS:
        if re.search(pat, primeira):
            score += 0.5
            break

    for pat in _CLOSERS:
        if re.search(pat, ultima):
            score += 0.5
            break

    # Bonus: check all sentences for closers (IA usa "furthermore" em todo paragrafo)
    mid_formulaic = 0
    for f in frases[1:-1]:
        f_low = f.lower().strip()
        if re.match(r'^(?:moreover|furthermore|additionally|consequently|in addition|similarly|likewise)\b', f_low):
            mid_formulaic += 1
    if len(frases) > 3 and mid_formulaic / (len(frases) - 2) > 0.3:
        score = min(score + 0.3, 1.0)

    return score

# ---------------------------------------------------------------------------
# Score combinado v2
# ---------------------------------------------------------------------------

def _score_entropia(entropia):
    """Baixa entropia = mais previsivel = mais IA"""
    if entropia < 5.0: return 92
    if entropia < 6.0: return 78
    if entropia < 7.0: return 60
    if entropia < 8.0: return 40
    if entropia < 9.0: return 25
    return 10

def _score_compressao(ratio):
    """Baixa ratio = comprime muito = mais IA"""
    if ratio < 0.28: return 95
    if ratio < 0.35: return 82
    if ratio < 0.42: return 65
    if ratio < 0.50: return 45
    if ratio < 0.58: return 30
    return 15

def _score_sent_sd(sd):
    """Baixo SD = frases uniformes = IA"""
    if sd < 2.5: return 95
    if sd < 4.0: return 80
    if sd < 5.5: return 62
    if sd < 7.0: return 42
    if sd < 9.0: return 25
    return 10

def _score_var_adj(cv):
    """Baixo CV adjacente = ritmo constante = IA"""
    if cv < 0.20: return 90
    if cv < 0.35: return 72
    if cv < 0.50: return 52
    if cv < 0.70: return 32
    return 12

def _score_burstiness(burst):
    if burst < 0.20: return 90
    if burst < 0.30: return 75
    if burst < 0.45: return 55
    if burst < 0.60: return 35
    if burst < 0.80: return 20
    return 8

def _score_ttr(ttr):
    if ttr < 0.35: return 80
    if ttr < 0.45: return 65
    if ttr < 0.55: return 50
    if ttr < 0.65: return 32
    return 15

def _score_hapax(hapax):
    if hapax < 0.25: return 78
    if hapax < 0.35: return 60
    if hapax < 0.45: return 42
    if hapax < 0.55: return 28
    return 12

def _score_ratio_cf(ratio):
    """Ratio alto = noun-heavy = IA"""
    if ratio > 1.50: return 92
    if ratio > 1.30: return 78
    if ratio > 1.15: return 60
    if ratio > 1.00: return 40
    if ratio > 0.85: return 22
    return 10

def _score_contracoes(freq):
    """Poucas contracoes = formal = IA"""
    if freq < 0.1: return 82
    if freq < 0.5: return 65
    if freq < 1.5: return 45
    if freq < 3.0: return 25
    return 8

def _score_ent_pontuacao(ent):
    """Baixa entropia pontuacao = mecanico = IA"""
    if ent < 1.5: return 88
    if ent < 2.2: return 72
    if ent < 2.8: return 55
    if ent < 3.3: return 38
    if ent < 3.8: return 22
    return 10

def _score_conectivos(freq):
    if freq > 1.2: return 98
    if freq > 0.8: return 88
    if freq > 0.5: return 72
    if freq > 0.3: return 55
    if freq > 0.15: return 38
    return 22  # ausencia nao prova humano — IA limpa tambem nao usa

def _score_div_inicio(div):
    if div < 0.45: return 88
    if div < 0.60: return 70
    if div < 0.75: return 48
    if div < 0.88: return 28
    return 10

def _score_pronomes(pct):
    """Poucos pronomes pessoais = impessoal = IA"""
    if pct < 0.5: return 85
    if pct < 1.5: return 68
    if pct < 3.0: return 48
    if pct < 5.0: return 28
    if pct < 10.0: return 15
    return 12  # muitos pronomes = provavelmente humano, mas IA 1a pessoa tambem

def _score_frases_ia(freq):
    if freq > 2.0: return 99
    if freq > 1.0: return 90
    if freq > 0.5: return 75
    if freq > 0.2: return 55
    if freq > 0.0: return 38
    return 22  # ausencia nao prova humano — IA limpa evita cliches

def _score_formulaicos(f):
    if f >= 0.8: return 95
    if f >= 0.5: return 78
    if f >= 0.3: return 55
    if f > 0.0: return 35
    return 18  # ausencia nao prova humano


def calcular_score(m):
    """Combina 15 metricas em score 0-100. Maior = mais provavel IA."""
    scores = {
        'entropia':     (_score_entropia(m['entropia']),          0.06),
        'compressao':   (_score_compressao(m['compressao']),      0.06),
        'sent_sd':      (_score_sent_sd(m['sent_sd']),            0.10),
        'var_adj':      (_score_var_adj(m['var_adj']),             0.06),
        'burstiness':   (_score_burstiness(m['burstiness']),      0.06),
        'ttr':          (_score_ttr(m['ttr']),                    0.02),
        'hapax':        (_score_hapax(m['hapax']),                0.02),
        'ratio_cf':     (_score_ratio_cf(m['ratio_cf']),          0.11),
        'contracoes':   (_score_contracoes(m['contracoes']),      0.11),
        'ent_pont':     (_score_ent_pontuacao(m['ent_pont']),     0.04),
        'conectivos':   (_score_conectivos(m['conectivos']),      0.10),
        'div_inicio':   (_score_div_inicio(m['div_inicio']),      0.03),
        'pronomes':     (_score_pronomes(m['pronomes']),          0.07),
        'frases_ia':    (_score_frases_ia(m['frases_ia']),        0.09),
        'formulaicos':  (_score_formulaicos(m['formulaicos']),    0.07),
    }

    base = sum(s * w for s, w in scores.values())

    # Consenso: sinais estruturais + lexicais fortes
    strong_ai = 0
    if m['conectivos'] > 0.5: strong_ai += 1
    if m['frases_ia'] > 0.5: strong_ai += 1
    if m['contracoes'] < 0.3: strong_ai += 1
    if m['pronomes'] < 1.0: strong_ai += 1
    if m['ratio_cf'] > 1.3: strong_ai += 1
    if m['sent_sd'] < 5.0: strong_ai += 1
    if m['burstiness'] < 0.25: strong_ai += 1
    # Novos sinais estruturais:
    if m['var_adj'] < 0.35: strong_ai += 1
    if m['entropia'] < 6.5: strong_ai += 1

    if strong_ai >= 7:
        base = max(base, 90)
    elif strong_ai >= 6:
        base = max(base, 82)
    elif strong_ai >= 5:
        base = max(base, 75)
    elif strong_ai >= 4:
        base = max(base, 65)
    elif strong_ai >= 3:
        base = max(base * 1.15, base)

    return min(max(base, 0.0), 99.5)


# ---------------------------------------------------------------------------
# Deteccao de codigo
# ---------------------------------------------------------------------------

_CODE_KEYWORDS = re.compile(
    r'^\s*(?:def |class |import |from |if |elif |else:|for |while |return |'
    r'print\(|input\(|try:|except |raise |with |async |await |'
    r'const |let |var |function |public |private |void |int |'
    r'#include|using namespace|System\.|Console\.)',
    re.MULTILINE,
)

_CODE_PATTERNS = re.compile(
    r'(?:= .*\(|\.append\(|\.split\(|\.strip\(|\.lower\(|\.upper\(|'
    r'f"[^"]*\{|f\'[^\']*\{|'
    r'\b\w+\s*=\s*\w+|'
    r'^\s{4,}\S|'  # indented lines
    r'#\s*\w)',
    re.MULTILINE,
)


def detectar_codigo(texto):
    """Retorna True se o texto parece conter codigo de programacao."""
    linhas = texto.strip().split('\n')
    if len(linhas) < 3:
        return False

    keyword_hits = len(_CODE_KEYWORDS.findall(texto))
    pattern_hits = len(_CODE_PATTERNS.findall(texto))
    indented = sum(1 for l in linhas if l.startswith('    ') or l.startswith('\t'))

    total_signals = keyword_hits + pattern_hits + indented
    ratio = total_signals / len(linhas)

    return ratio > 0.25 and keyword_hits >= 2


def extrair_nl_de_codigo(texto):
    """Extrai linguagem natural de codigo: comentarios, strings, docstrings."""
    partes_nl = []

    # Docstrings e strings multiline
    for m in re.finditer(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', texto, re.DOTALL):
        content = m.group(1) or m.group(2)
        if content.strip():
            partes_nl.append(content.strip())

    # Comentarios inline (#)
    for m in re.finditer(r'#\s*(.+)$', texto, re.MULTILINE):
        comment = m.group(1).strip()
        if len(comment) > 5 and not comment.startswith('!'):
            partes_nl.append(comment)

    # Comentarios // (JS, C, Java)
    for m in re.finditer(r'//\s*(.+)$', texto, re.MULTILINE):
        comment = m.group(1).strip()
        if len(comment) > 5:
            partes_nl.append(comment)

    # Strings com texto legivel (print, input, etc.)
    for m in re.finditer(r'(?:print|input|console\.log)\s*\(\s*["\'](.+?)["\']', texto):
        s = m.group(1).strip()
        if len(s) > 10 and re.search(r'[a-zA-Z]{3,}', s):
            partes_nl.append(s)

    # f-strings com texto legivel
    for m in re.finditer(r'f["\'](.+?)["\']', texto, re.DOTALL):
        s = re.sub(r'\{[^}]+\}', '', m.group(1)).strip()
        if len(s) > 15 and re.search(r'[a-zA-Z]{3,}', s):
            partes_nl.append(s)

    return ' '.join(partes_nl)


# ---------------------------------------------------------------------------
# Metricas de codigo IA
# ---------------------------------------------------------------------------

_PEDAGOGICAL_PHRASES = [
    "this function", "this method", "this program", "this script",
    "this code", "this class", "this module", "this variable",
    "this loop", "this condition", "this block",
    "here we", "here i", "we then", "we first", "we also",
    "we need to", "we want to", "we can", "we use",
    "i added", "i also", "i made", "i used", "i created",
    "i implemented", "i defined", "i wrote", "i changed",
    "i modified", "i updated", "i converted", "i included",
    "step 1", "step 2", "step 3", "step 4",
    "first, we", "next, we", "then, we", "finally, we",
    "first,", "second,", "third,", "finally,",
    "as you can see", "note that", "notice that", "remember that",
    "the following", "the above", "the below", "the result",
    "let's", "let us",
    "the program will", "the function will", "the code will",
    "the program choose", "the function return",
    "make the program", "made the program",
    "will prompt", "will ask", "will display", "will print",
    "will check", "will return", "will calculate", "will convert",
    "automatically", "the user", "user input", "user enters",
    "error handling", "handle the case", "edge case",
    "for example", "for instance",
    "ensure that", "make sure", "validate",
    "initialize", "initialize the",
    "store the", "stores the", "storing the",
    "check if", "checks if", "checking if", "checks whether",
    "convert the", "converts the", "converting",
    "iterate", "iterates", "iterating",
    "loop through", "loops through",
    "extra words", "added extra", "also made",
    # Label-style AI comments (AI labels its creative/bonus additions)
    "creativity:", "creative:", "bonus:", "extra credit:", "extra:",
    "added feature:", "feature:", "modification:", "enhancement:",
    # "Added" without "I" — AI explaining what was added
    "added a ", "added the ", "added an ", "added some",
    "added error", "added color", "added more", "added input",
]


_AI_COMMENT_RE = re.compile(
    r'#\s*(?:I\s+)?(?:added|also added|made|used|created|implemented|defined|wrote|changed|'
    r'modified|updated|converted|included|decided|chose|set|moved|put|renamed|'
    r'fixed|ensured|wanted|needed)',
    re.IGNORECASE,
)

_VAR_MATCHES_PROMPT_RE = re.compile(
    r'(\w+)\s*=\s*input\(\s*["\'](\w+)',
)


def _extrair_comentarios(texto):
    """Extrai APENAS comentarios (nao strings de print/input)."""
    partes = []
    for m in re.finditer(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', texto, re.DOTALL):
        content = m.group(1) or m.group(2)
        if content.strip():
            partes.append(content.strip())
    for m in re.finditer(r'#\s*(.+)$', texto, re.MULTILINE):
        comment = m.group(1).strip()
        if len(comment) > 3 and not comment.startswith('!'):
            partes.append(comment)
    for m in re.finditer(r'//\s*(.+)$', texto, re.MULTILINE):
        comment = m.group(1).strip()
        if len(comment) > 3:
            partes.append(comment)
    return ' '.join(partes)


def calcular_metricas_codigo(texto):
    """Metricas especificas para codigo AI-generated."""
    linhas = texto.strip().split('\n')
    linhas_nv = [l for l in linhas if l.strip()]

    # 1. Comment-to-code ratio
    comment_lines = sum(1 for l in linhas_nv if l.strip().startswith('#') or l.strip().startswith('//'))
    code_lines = len(linhas_nv) - comment_lines
    comment_ratio = comment_lines / max(code_lines, 1)

    # 2. Pedagogical phrases — only in COMMENTS, not in print/input strings
    comment_text = _extrair_comentarios(texto).lower()
    nl_text = extrair_nl_de_codigo(texto).lower()
    total_nl_words = len(nl_text.split()) if nl_text else 0
    pedagogical_hits = sum(1 for p in _PEDAGOGICAL_PHRASES if p in comment_text) if comment_text else 0

    # 3. "I [verb]" comment pattern — AI explaining its own work
    ai_comment_hits = len(_AI_COMMENT_RE.findall(texto))
    pedagogical_hits += ai_comment_hits * 2

    # 4. Repetitive function calls (many sequential input/print)
    func_sequences = re.findall(r'(?:^[ \t]*(?:input|print)\s*\(.*$\n?){3,}', texto, re.MULTILINE)
    repetitive_calls = len(func_sequences)
    if repetitive_calls > 0:
        pedagogical_hits += repetitive_calls * 2

    # Also count max consecutive same-function calls
    consecutive_input = len(re.findall(r'=\s*input\(', texto))
    if consecutive_input >= 4:
        pedagogical_hits += 3
    elif consecutive_input >= 3:
        pedagogical_hits += 1

    # 5. Variable name matches prompt text: adjective = input("adjective: ")
    var_prompt_matches = 0
    for m in _VAR_MATCHES_PROMPT_RE.finditer(texto):
        var_name = m.group(1).lower().rstrip('_').replace('_', '')
        prompt_word = m.group(2).lower().rstrip('_').replace('_', '')
        if var_name == prompt_word or var_name.startswith(prompt_word) or prompt_word.startswith(var_name):
            var_prompt_matches += 1
    if var_prompt_matches >= 3:
        pedagogical_hits += 4
    elif var_prompt_matches >= 2:
        pedagogical_hits += 2

    # 6. Comment explains rationale ("to make...", "so that...", "in order to...")
    rationale_comments = len(re.findall(
        r'#.*(?:to make|so that|in order to|to ensure|to allow|to handle|to keep|to avoid|to prevent|to improve|to add)',
        texto, re.IGNORECASE
    ))
    pedagogical_hits += rationale_comments * 2

    # 6b. Label-style comments: "# Label: explanation" — AI labels its sections
    label_comments = len(re.findall(
        r'#\s*(?:Creativity|Creative|Bonus|Extra|Feature|Modification|Enhancement|'
        r'Added|Note|Purpose|Description|Explanation|Summary|Overview|'
        r'Task|Assignment|Requirement|Challenge)\s*:',
        texto, re.IGNORECASE
    ))
    pedagogical_hits += label_comments * 3

    # 7. Line length uniformity (CV)
    if len(linhas_nv) >= 3:
        lens = [len(l) for l in linhas_nv]
        m = mean(lens)
        line_cv = stdev(lens) / m if m > 0 else 0
    else:
        line_cv = 0.5

    # 8. Variable name consistency
    identifiers = re.findall(r'\b([a-z][a-z0-9_]*)\b', texto)
    identifiers = [i for i in identifiers if len(i) > 1 and i not in (
        'if', 'in', 'or', 'is', 'as', 'do', 'of', 'to', 'no', 'up',
        'def', 'for', 'and', 'not', 'the', 'was', 'but', 'all', 'are',
        'else', 'from', 'with', 'true', 'none', 'self', 'that', 'this',
        'import', 'return', 'print', 'input', 'while', 'break', 'class',
        'false', 'raise', 'yield', 'async', 'await', 'super', 'lower',
        'upper', 'strip', 'split', 'append', 'range', 'capitalize',
    )]
    if len(identifiers) >= 3:
        id_lens = [len(i) for i in identifiers]
        m_id = mean(id_lens)
        id_cv = stdev(id_lens) / m_id if m_id > 0 else 0
    else:
        id_cv = 0.5

    # 9. Perfect formatting (consistent indentation)
    indent_levels = set()
    for l in linhas:
        stripped = l.lstrip()
        if stripped:
            indent = len(l) - len(stripped)
            if indent > 0:
                indent_levels.add(indent)
    uses_consistent_indent = all(i % 4 == 0 for i in indent_levels) if indent_levels else True

    # 10. Blank line regularity
    blank_positions = [i for i, l in enumerate(linhas) if not l.strip()]
    if len(blank_positions) >= 2:
        gaps = [blank_positions[i+1] - blank_positions[i] for i in range(len(blank_positions)-1)]
        blank_regularity = 1.0 - (stdev(gaps) / mean(gaps) if mean(gaps) > 0 else 0) if len(gaps) >= 2 else 0.5
    else:
        blank_regularity = 0.5

    return {
        'comment_ratio': comment_ratio,
        'pedagogical_hits': pedagogical_hits,
        'pedagogical_density': pedagogical_hits / max(total_nl_words / 100, 0.01) if total_nl_words > 0 else 0,
        'line_cv': line_cv,
        'id_cv': id_cv,
        'consistent_indent': uses_consistent_indent,
        'blank_regularity': blank_regularity,
        'var_prompt_matches': var_prompt_matches,
    }


def score_codigo(code_metricas, nl_score):
    """Score combinado para codigo. nl_score e o score do texto NL extraido."""
    cm = code_metricas

    s = 0.0

    # Comment ratio: AI over-comments (weight: up to 15)
    if cm['comment_ratio'] > 0.4: s += 15
    elif cm['comment_ratio'] > 0.25: s += 12
    elif cm['comment_ratio'] > 0.15: s += 8
    elif cm['comment_ratio'] > 0.05: s += 5
    else: s += 2

    # Pedagogical phrases: THE strongest signal for code (weight: up to 40)
    ph = cm['pedagogical_hits']
    if ph >= 6: s += 40
    elif ph >= 4: s += 34
    elif ph >= 3: s += 28
    elif ph >= 2: s += 22
    elif ph >= 1: s += 14
    else: s += 2

    # Variable-matches-prompt pattern (weight: up to 15)
    vpm = cm.get('var_prompt_matches', 0)
    if vpm >= 5: s += 15
    elif vpm >= 3: s += 11
    elif vpm >= 2: s += 7
    elif vpm >= 1: s += 3

    # Line length uniformity: AI = uniform (weight: up to 10)
    if cm['line_cv'] < 0.3: s += 10
    elif cm['line_cv'] < 0.5: s += 7
    elif cm['line_cv'] < 0.7: s += 4
    else: s += 1

    # Identifier name consistency (weight: up to 8)
    if cm['id_cv'] < 0.25: s += 8
    elif cm['id_cv'] < 0.4: s += 5
    else: s += 2

    # Consistent indentation (weight: up to 5)
    if cm['consistent_indent']:
        s += 5

    # Blank line regularity (weight: up to 5)
    if cm['blank_regularity'] > 0.7: s += 5
    elif cm['blank_regularity'] > 0.4: s += 3

    # NL text analysis (weight: up to ~25)
    if nl_score > 0:
        s += nl_score * 0.25

    # Consensus boost: many pedagogical + clean structure = definitely AI
    if ph >= 3 and cm['consistent_indent'] and cm['comment_ratio'] > 0.02:
        s = max(s, 65)
    if ph >= 5:
        s = max(s, 72)
    if ph >= 8:
        s = max(s, 78)
    if ph >= 12 or (ph >= 6 and vpm >= 3):
        s = max(s, 82)

    return min(s, 99.5)


# ---------------------------------------------------------------------------
# Funcao de avaliacao publica
# ---------------------------------------------------------------------------

def _avaliar_texto(texto):
    """Avalia texto puro (prosa) com 15 metricas."""
    metricas = {
        'entropia':     calcular_entropia(texto),
        'compressao':   calcular_compressao(texto),
        'burstiness':   calcular_burstiness(texto),
        'sent_sd':      calcular_sent_len_sd(texto),
        'var_adj':      calcular_var_adjacentes(texto),
        'ttr':          calcular_ttr(texto),
        'hapax':        calcular_hapax_ratio(texto),
        'ratio_cf':     calcular_ratio_conteudo_funcao(texto),
        'contracoes':   calcular_contracoes(texto),
        'ent_pont':     calcular_entropia_pontuacao(texto),
        'conectivos':   calcular_freq_conectivos(texto),
        'div_inicio':   calcular_diversidade_inicio(texto),
        'pronomes':     calcular_densidade_pronomes(texto),
        'frases_ia':    contar_frases_ia(texto),
        'formulaicos':  calcular_formulaicos(texto),
    }
    score = calcular_score(metricas)
    return score, metricas


def avaliar(texto):
    """Roda analise completa. Detecta automaticamente se e codigo ou texto."""
    is_code = detectar_codigo(texto)

    if is_code:
        nl_text = extrair_nl_de_codigo(texto)
        code_m = calcular_metricas_codigo(texto)

        nl_score = 0.0
        if nl_text and len(nl_text.split()) >= 5:
            nl_score, nl_metricas = _avaliar_texto(nl_text)
        else:
            nl_metricas = None

        final_score = score_codigo(code_m, nl_score)

        metricas = {
            'entropia':     calcular_entropia(texto),
            'compressao':   calcular_compressao(texto),
            'burstiness':   0.0,
            'sent_sd':      0.0,
            'var_adj':      0.0,
            'ttr':          calcular_ttr(texto),
            'hapax':        calcular_hapax_ratio(texto),
            'ratio_cf':     0.0,
            'contracoes':   calcular_contracoes(nl_text) if nl_text else 0.0,
            'ent_pont':     0.0,
            'conectivos':   0.0,
            'div_inicio':   0.0,
            'pronomes':     calcular_densidade_pronomes(nl_text) if nl_text else 0.0,
            'frases_ia':    0.0,
            'formulaicos':  0.0,
            'is_code':      True,
            'code_comment_ratio': code_m['comment_ratio'],
            'code_pedagogical': code_m['pedagogical_hits'],
            'code_line_cv': code_m['line_cv'],
            'code_nl_score': nl_score,
        }
        return final_score, metricas

    return _avaliar_texto(texto)


# ---------------------------------------------------------------------------
# Analise por trecho — identifica problemas e sugere correcoes
# ---------------------------------------------------------------------------

_EXPANDABLE_FORMS = [
    ("do not", "don't"),
    ("does not", "doesn't"),
    ("did not", "didn't"),
    ("is not", "isn't"),
    ("are not", "aren't"),
    ("was not", "wasn't"),
    ("were not", "weren't"),
    ("has not", "hasn't"),
    ("have not", "haven't"),
    ("had not", "hadn't"),
    ("will not", "won't"),
    ("would not", "wouldn't"),
    ("could not", "couldn't"),
    ("should not", "shouldn't"),
    ("cannot", "can't"),
]


_CONNECTOR_REPLACEMENTS = {
    "moreover": "also", "furthermore": "and", "additionally": "plus",
    "consequently": "so", "nevertheless": "still", "subsequently": "then",
    "thereby": "so", "henceforth": "from now on",
    "it is important to note": "worth noting",
    "it is worth mentioning": "also",
    "it is worth noting": "also",
    "it should be noted": "note that",
    "in today's world": "(remove)",
    "in today's society": "(remove)",
    "plays a crucial role": "matters a lot",
    "plays a vital role": "is important",
    "plays a pivotal role": "is key",
    "it is essential to": "you need to",
    "has the potential to": "could",
    "in conclusion": "(remove or use 'So,')",
    "in order to": "to",
    "with regard to": "about",
    "in terms of": "for",
    "a wide range of": "lots of",
    "a variety of": "different",
    "significantly": "a lot",
    "comprehensive": "full",
    "multifaceted": "complex",
    "paradigm": "approach",
    "leverage": "use",
    "utilize": "use",
    "facilitate": "help",
    "delve": "look into",
    "robust": "strong",
    "holistic": "overall",
    "landscape": "field",
    "encompasses": "includes",
    "transformative": "big",
    "paramount": "very important",
    "fundamental": "basic",
    "imperative": "urgent",
}


def analisar_problemas(texto):
    """Identifica trechos problematicos e sugere correcoes."""
    problemas = []
    is_code = detectar_codigo(texto)

    if is_code:
        _analisar_problemas_codigo(texto, problemas)
    else:
        _analisar_problemas_texto(texto, problemas)

    return problemas


def _analisar_problemas_texto(texto, problemas):
    """Analisa texto puro e encontra trechos com cara de IA."""
    texto_lower = texto.lower()

    for conn, fix in _CONNECTOR_REPLACEMENTS.items():
        idx = texto_lower.find(conn)
        if idx >= 0:
            end = idx + len(conn)
            if end < len(texto) and texto[end].isalpha():
                continue
            if idx > 0 and texto[idx - 1].isalpha():
                continue
            original = texto[idx:end]
            prob = {
                'tipo': 'conectivo_ia',
                'trecho': original,
                'motivo': 'Palavra/expressao tipica de IA',
                'sugestao': fix,
                'posicao': idx,
            }
            if not fix.startswith('('):
                prob['corrigido'] = _match_case(original, fix)
            problemas.append(prob)

    frases = _frases(texto)
    if len(frases) >= 3:
        comprimentos = [len(f.split()) for f in frases]
        for i in range(1, len(comprimentos)):
            diff = abs(comprimentos[i] - comprimentos[i-1])
            if diff <= 2 and comprimentos[i] > 10:
                problemas.append({
                    'tipo': 'uniformidade',
                    'trecho': frases[i][:80] + ('...' if len(frases[i]) > 80 else ''),
                    'motivo': 'Frases com tamanho muito parecido (IA escreve uniforme)',
                    'sugestao': 'Quebre em frases curtas + longas. Varie o ritmo.',
                    'posicao': texto.find(frases[i][:30]),
                })

    contracoes = _CONTRACTION_RE.findall(texto)
    total_palavras = len(texto.split())
    if total_palavras >= 20 and len(contracoes) == 0:
        expandable = []
        for expanded, contracted in _EXPANDABLE_FORMS:
            idx = texto_lower.find(expanded.lower())
            if idx >= 0:
                orig = texto[idx:idx+len(expanded)]
                fix = _match_case(orig, contracted)
                expandable.append({
                    'tipo': 'sem_contracoes',
                    'trecho': orig,
                    'motivo': 'Forma expandida - IA evita contracoes',
                    'sugestao': f'{orig} -> {fix}',
                    'corrigido': fix,
                    'posicao': idx,
                })
        if expandable:
            problemas.extend(expandable)
        else:
            problemas.append({
                'tipo': 'sem_contracoes',
                'trecho': '(texto inteiro)',
                'motivo': 'Zero contracoes - IA evita contracoes',
                'sugestao': 'Adicione contracoes: do not -> don\'t, it is -> it\'s',
                'posicao': 0,
            })

    for phrase in ["in today's", "in the modern era", "in the digital age",
                    "in conclusion", "to summarize", "in summary",
                    "first and foremost", "last but not least",
                    "plays a crucial role", "has the potential to",
                    "it is important to note", "it is worth noting",
                    "delve into", "a myriad of", "a plethora of"]:
        idx = texto_lower.find(phrase)
        if idx >= 0:
            original = texto[idx:idx+len(phrase)]
            if not any(p['trecho'].lower() == original.lower() for p in problemas):
                fix = _CONNECTOR_REPLACEMENTS.get(phrase, 'Reescreva de forma casual')
                prob = {
                    'tipo': 'frase_ia',
                    'trecho': original,
                    'motivo': 'Expressao dead giveaway de IA',
                    'sugestao': fix,
                    'posicao': idx,
                }
                if fix and not fix.startswith('(') and fix != 'Reescreva de forma casual':
                    prob['corrigido'] = _match_case(original, fix)
                problemas.append(prob)

    primeiro_pron = False
    for p in _palavras(texto):
        if p in FIRST_PERSON or p in SECOND_PERSON:
            primeiro_pron = True
            break
    if not primeiro_pron and total_palavras >= 30:
        problemas.append({
            'tipo': 'sem_pronomes',
            'trecho': '(texto inteiro)',
            'motivo': 'Sem pronomes pessoais — IA escreve de forma impessoal',
            'sugestao': 'Adicione perspectiva pessoal: "I think...", "you can see...", "we know..."',
            'posicao': 0,
        })

    if len(frases) >= 2:
        primeira = frases[0].lower()
        for pat in _OPENERS:
            if re.search(pat, primeira):
                problemas.append({
                    'tipo': 'abertura_formulaica',
                    'trecho': frases[0][:80] + ('...' if len(frases[0]) > 80 else ''),
                    'motivo': 'Abertura tipica de IA',
                    'sugestao': 'Comece de forma direta ou com uma opiniao/pergunta',
                    'posicao': 0,
                })
                break

    problemas.sort(key=lambda p: p['posicao'])


def _analisar_problemas_codigo(texto, problemas):
    """Analisa codigo e encontra sinais de IA."""
    linhas = texto.strip().split('\n')

    for i, linha in enumerate(linhas):
        stripped = linha.strip()

        if _AI_COMMENT_RE.search(linha):
            problemas.append({
                'tipo': 'comentario_ia',
                'trecho': stripped,
                'motivo': 'Comentario "I [verbo]" - IA explicando seu trabalho',
                'sugestao': 'Remova o comentario',
                'corrigido': '',
                'posicao': i,
            })
        elif stripped.startswith('#') and len(stripped) > 15:
            comment_lower = stripped.lower()
            for phrase in _PEDAGOGICAL_PHRASES:
                if phrase in comment_lower:
                    problemas.append({
                        'tipo': 'comentario_pedagogico',
                        'trecho': stripped,
                        'motivo': f'Comentario pedagogico (contem "{phrase}")',
                        'sugestao': 'Remova o comentario',
                        'corrigido': '',
                        'posicao': i,
                    })
                    break

    matches = list(_VAR_MATCHES_PROMPT_RE.finditer(texto))
    var_matches = []
    for m in matches:
        var_name = m.group(1).lower().rstrip('_').replace('_', '')
        prompt_word = m.group(2).lower().rstrip('_').replace('_', '')
        if var_name == prompt_word or var_name.startswith(prompt_word) or prompt_word.startswith(var_name):
            var_matches.append((m.group(1), m.group(2), m.start()))
    if len(var_matches) >= 2:
        nomes = ', '.join(f'{v}=input("{p}:")' for v, p, _ in var_matches[:4])
        problemas.append({
            'tipo': 'var_igual_prompt',
            'trecho': nomes,
            'motivo': f'{len(var_matches)} variaveis com nome igual ao prompt do input()',
            'sugestao': 'Renomeie: adjective->adj, exclamation->exc, animal->ani, verb1->v1',
            'posicao': var_matches[0][2],
        })

    input_lines = [(i, l) for i, l in enumerate(linhas)
                   if re.search(r'=\s*input\(', l.strip())]
    if len(input_lines) >= 4:
        problemas.append({
            'tipo': 'inputs_sequenciais',
            'trecho': f'{len(input_lines)} chamadas input() sequenciais',
            'motivo': 'Muitas chamadas input() repetitivas — padrao IA',
            'sugestao': 'Use um loop ou lista: prompts=["adj","animal",...]; answers=[input(p+": ") for p in prompts]',
            'posicao': input_lines[0][0],
        })

    print_lines = [(i, l) for i, l in enumerate(linhas)
                   if l.strip().startswith('print(') and len(l.strip()) > 10]
    consecutive = 0
    max_consecutive = 0
    for j in range(1, len(print_lines)):
        if print_lines[j][0] - print_lines[j-1][0] <= 2:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
    if max_consecutive >= 3:
        problemas.append({
            'tipo': 'prints_sequenciais',
            'trecho': f'{max_consecutive+1} prints seguidos',
            'motivo': 'Muitas chamadas print() sequenciais — padrao IA',
            'sugestao': 'Combine com \\n ou use uma string multiline',
            'posicao': print_lines[0][0],
        })

    rationale = re.findall(
        r'(#.*(?:to make|so that|in order to|to ensure|to allow|to handle|to keep|to avoid).*)',
        texto, re.IGNORECASE
    )
    for r in rationale:
        problemas.append({
            'tipo': 'comentario_razao',
            'trecho': r.strip(),
            'motivo': 'Comentario explicando razao - IA justifica suas escolhas',
            'sugestao': 'Remova completamente',
            'corrigido': '',
            'posicao': texto.find(r),
        })

    problemas.sort(key=lambda p: p['posicao'])


def aplicar_correcao(texto, problema):
    """Aplica uma correcao automatica ao texto. Retorna o texto corrigido."""
    tipo = problema['tipo']
    corrigido = problema.get('corrigido')
    if corrigido is None:
        return texto

    if tipo in ('comentario_ia', 'comentario_pedagogico', 'comentario_razao'):
        trecho = problema['trecho'].strip()
        lines = texto.split('\n')
        new_lines = []
        removed = False
        for line in lines:
            if not removed and trecho in line:
                removed = True
                continue
            new_lines.append(line)
        return '\n'.join(new_lines)

    pos = problema.get('posicao', -1)
    trecho = problema['trecho']
    if pos >= 0 and pos + len(trecho) <= len(texto) and texto[pos:pos+len(trecho)] == trecho:
        return texto[:pos] + corrigido + texto[pos+len(trecho):]
    return texto.replace(trecho, corrigido, 1)


def simular_correcao(texto, problema):
    """Simula uma correcao e retorna (novo_score, delta).
    delta e negativo quando o score melhora (cai)."""
    novo = aplicar_correcao(texto, problema)
    if novo == texto:
        return None, 0
    novo_score, _ = avaliar(novo)
    return novo_score, None


def simular_todas(texto, problemas):
    """Simula aplicar todas as correcoes fixaveis. Retorna (novo_score, delta)."""
    fixable = [p for p in problemas if 'corrigido' in p]
    if not fixable:
        return None, 0
    for prob in sorted(fixable, key=lambda p: p.get('posicao', 0), reverse=True):
        texto = aplicar_correcao(texto, prob)
    novo_score, _ = avaliar(texto)
    return novo_score, None


def gerar_texto_destacado(texto, problemas):
    """Gera HTML do texto com trechos problematicos destacados em cores."""
    if not problemas:
        return texto.replace('\n', '<br>')

    fixable = [p for p in problemas if 'corrigido' in p and p['trecho'] != '(texto inteiro)']
    fixable.sort(key=lambda p: p.get('posicao', 0))

    if not fixable:
        return texto.replace('\n', '<br>')

    result = []
    last_end = 0

    for prob in fixable:
        pos = prob.get('posicao', -1)
        trecho = prob['trecho']
        if pos < 0 or pos < last_end:
            continue
        end = pos + len(trecho)
        if end > len(texto):
            continue
        if texto[pos:end] != trecho:
            continue

        result.append(_html_escape(texto[last_end:pos]))
        result.append(
            f'<span style="background-color:#FBBF24;padding:1px 3px;border-radius:3px;'
            f'cursor:help" title="{_html_escape(prob["motivo"])}: {_html_escape(prob["sugestao"])}">'
            f'{_html_escape(trecho)}</span>'
        )
        last_end = end

    result.append(_html_escape(texto[last_end:]))
    return ''.join(result).replace('\n', '<br>')


def _html_escape(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def _revelar_chars_escondidos(texto):
    """Substitui TODOS os caracteres escondidos/invisiveis por badges coloridos visiveis.

    Mostra qualquer caractere que nao seja texto normal imprimivel:
    - Vermelho: watermarks conhecidos
    - Laranja: caracteres invisiveis desconhecidos (scanner generico)
    - Azul: tipograficos (aspas curvas, travessao, etc.)
    """
    result = []
    for ch in texto:
        cp = ord(ch)

        # Caracteres normais: ASCII imprimivel, letras, digitos, pontuacao comum, espacos normais
        if ch in ('\n', '\r', '\t', ' '):
            result.append(_html_escape(ch))
            continue

        # Verificar se eh um caractere "normal" (letra, digito, pontuacao basica)
        cat = unicodedata.category(ch)

        # Categorias normais: L=letra, N=numero, P=pontuacao, S=simbolo, Z=separador
        # Mas separadores invisiveis (Zl, Zp, Cf, Cc) devem ser mostrados
        if cat.startswith('L') or cat.startswith('N'):
            # Letras e numeros — normal
            result.append(_html_escape(ch))
            continue

        if cat in ('Pc', 'Pd', 'Ps', 'Pe', 'Pi', 'Pf', 'Po'):
            # Pontuacao — verificar se eh tipografica
            if ch in _TYPO_CHARS:
                sigla, nome = _TYPO_CHARS[ch]
                result.append(
                    f'<span style="background:#3B82F6;color:white;padding:0 3px;'
                    f'border-radius:3px;font-size:11px;cursor:help" '
                    f'title="{nome} (U+{cp:04X})">{_html_escape(ch)}</span>'
                )
            else:
                result.append(_html_escape(ch))
            continue

        if cat == 'Zs':
            # Separador de espaco
            if cp == 0x20:  # espaco normal
                result.append(' ')
            elif ch in _WATERMARK_CHARS:
                sigla, nome = _WATERMARK_CHARS[ch]
                result.append(
                    f'<span style="background:#EF4444;color:white;padding:0 3px;'
                    f'border-radius:3px;font-size:10px;font-weight:bold;cursor:help" '
                    f'title="{nome} (U+{cp:04X})">[{sigla}]</span>'
                )
            else:
                nome = unicodedata.name(ch, f'SPACE U+{cp:04X}')
                result.append(
                    f'<span style="background:#F59E0B;color:white;padding:0 3px;'
                    f'border-radius:3px;font-size:10px;font-weight:bold;cursor:help" '
                    f'title="{nome} (U+{cp:04X})">[U+{cp:04X}]</span>'
                )
            continue

        if cat in ('Sm', 'Sc', 'Sk', 'So'):
            # Simbolos matematicos, moeda, modificador, outro — normal
            result.append(_html_escape(ch))
            continue

        # Tudo abaixo sao categorias suspeitas: Cf, Cc, Mn, Mc, Me, Co, Cn, Zl, Zp
        if ch in _WATERMARK_CHARS:
            sigla, nome = _WATERMARK_CHARS[ch]
            result.append(
                f'<span style="background:#EF4444;color:white;padding:0 3px;'
                f'border-radius:3px;font-size:10px;font-weight:bold;cursor:help" '
                f'title="{nome} (U+{cp:04X})">[{sigla}]</span>'
            )
        elif cat in ('Cf', 'Cc', 'Co', 'Cn', 'Zl', 'Zp'):
            # Formato, controle, privado, nao-atribuido, separadores de linha/paragrafo
            nome = unicodedata.name(ch, f'CHAR U+{cp:04X}')
            result.append(
                f'<span style="background:#F59E0B;color:white;padding:0 3px;'
                f'border-radius:3px;font-size:10px;font-weight:bold;cursor:help" '
                f'title="{nome} (U+{cp:04X})">[U+{cp:04X}]</span>'
            )
        elif cat.startswith('M'):
            # Combining marks — mostrar em azul
            nome = unicodedata.name(ch, f'MARK U+{cp:04X}')
            result.append(
                f'<span style="background:#3B82F6;color:white;padding:0 3px;'
                f'border-radius:3px;font-size:11px;cursor:help" '
                f'title="{nome} (U+{cp:04X})">◌{_html_escape(ch)}</span>'
            )
        else:
            result.append(_html_escape(ch))

    return ''.join(result)


def gerar_texto_destacado_sentencas(texto, score, problemas):
    """Gera HTML com highlight por sentenca inteira (estilo QuillBot).

    - score >= 65%: todas as sentencas em dourado
    - score >= 45%: sentencas com problemas em amarelo claro
    - score < 45%: apenas trechos especificos
    """
    sentencas = re.split(r'(?<=[.!?])\s+', texto.strip())
    if not sentencas:
        return _html_escape(texto).replace('\n', '<br>')

    # Mapeia posicoes dos problemas
    prob_posicoes = set()
    for p in problemas:
        pos = p.get('posicao', -1)
        if pos >= 0:
            prob_posicoes.add(pos)

    result = []
    offset = 0
    for sent in sentencas:
        idx = texto.find(sent, offset)
        if idx < 0:
            idx = offset

        # Checa se a sentenca contem algum problema
        has_problem = False
        for p in problemas:
            pos = p.get('posicao', -1)
            trecho = p.get('trecho', '')
            if pos >= 0 and idx <= pos < idx + len(sent):
                has_problem = True
                break
            if trecho and trecho in sent:
                has_problem = True
                break

        # Decide highlight (estilo QuillBot — fundo dourado, sem borda)
        # Usa _revelar_chars_escondidos pra mostrar TODOS chars invisiveis
        sent_html = _revelar_chars_escondidos(sent)
        if score >= 65:
            # Score alto: tudo highlighted em dourado
            result.append(
                f'<span style="background:#F5DEB3;padding:2px 4px;'
                f'color:#1a1a1a;">{sent_html}</span> '
            )
        elif score >= 45 and has_problem:
            result.append(
                f'<span style="background:#FEF3C7;padding:2px 4px;'
                f'color:#1a1a1a;">{sent_html}</span> '
            )
        else:
            result.append(f'{sent_html} ')

        offset = idx + len(sent)

    html = ''.join(result)
    # Preserve paragraphs
    html = html.replace('\n\n', '</p><p style="margin-top:12px;">')
    html = html.replace('\n', '<br>')
    return f'<p>{html}</p>'


# ---------------------------------------------------------------------------
# Deteccao de Watermarks Unicode (caracteres invisiveis)
# ---------------------------------------------------------------------------

# Caracteres invisiveis usados como watermarks por LLMs
_WATERMARK_CHARS = {
    ' ': ('NNBSP', 'Narrow No-Break Space — principal watermark ChatGPT o3/o4-mini'),
    '​': ('ZWSP', 'Zero-Width Space — marker invisivel de largura zero'),
    '‌': ('ZWNJ', 'Zero-Width Non-Joiner — separador invisivel'),
    '‍': ('ZWJ', 'Zero-Width Joiner — conector invisivel'),
    '­': ('SHY', 'Soft Hyphen — hifenizacao invisivel'),
    '⁠': ('WJ', 'Word Joiner — impede quebra de linha'),
    '﻿': ('BOM', 'Byte Order Mark — marca de encoding'),
    '‎': ('LRM', 'Left-to-Right Mark — direcional invisivel'),
    '‏': ('RLM', 'Right-to-Left Mark — direcional invisivel'),
    ' ': ('EM SP', 'Em Space — espaco largo'),
    ' ': ('EN SP', 'En Space — espaco medio'),
    ' ': ('THIN', 'Thin Space — espaco fino'),
    ' ': ('HAIR', 'Hair Space — espaco minimo'),
    ' ': ('FIG SP', 'Figure Space — espaco de digito'),
    ' ': ('PUNC SP', 'Punctuation Space — espaco de pontuacao'),
    ' ': ('MMSP', 'Medium Mathematical Space'),
    ' ': ('NBSP', 'No-Break Space — espaco que nao quebra linha'),
    '᠎': ('MVS', 'Mongolian Vowel Separator'),
}

# Caracteres tipograficos que LLMs inserem (nao invisiveis, mas atipicos)
_TYPO_CHARS = {
    '—': ('EM DASH', 'Em Dash — traco longo (comum em IA)'),
    '–': ('EN DASH', 'En Dash — traco medio'),
    '‘': ('LSQUO', 'Left Single Quotation Mark — aspas curvas'),
    '’': ('RSQUO', 'Right Single Quotation Mark / apostrofo curvo'),
    '“': ('LDQUO', 'Left Double Quotation Mark — aspas duplas curvas'),
    '”': ('RDQUO', 'Right Double Quotation Mark — aspas duplas curvas'),
    '…': ('ELLIP', 'Horizontal Ellipsis — reticencias tipograficas'),
}

# Codepoints conhecidos (para nao duplicar no scan generico)
_KNOWN_CODEPOINTS = {ord(ch) for ch in _WATERMARK_CHARS} | {ord(ch) for ch in _TYPO_CHARS}

# Categorias Unicode suspeitas (invisiveis/format/control)
# Cf = format (zero-width, direction marks, BOM...)
# Cc = control chars
# Co = private use area
# Cn = unassigned
# Zs = space separator (exceto espaco normal U+0020)
# Zl = line separator
# Zp = paragraph separator
_SUSPICIOUS_CATEGORIES = {'Cf', 'Cc', 'Co', 'Cn', 'Zs', 'Zl', 'Zp'}

# Caracteres normais que NAO devem ser flagados
_ALLOWED_CHARS = {' ', '\n', '\r', '\t'}


def _is_anomalous_char(ch):
    """Verifica se um caractere e anomalo/invisivel usando categoria Unicode.
    Pega QUALQUER caractere suspeito, mesmo os que nao estao no dicionario.
    Returns: (is_anomalous: bool, nome: str or None)
    """
    cp = ord(ch)
    if ch in _ALLOWED_CHARS:
        return False, None
    if cp in _KNOWN_CODEPOINTS:
        return False, None  # ja tratado pelos dicts conhecidos

    cat = unicodedata.category(ch)
    if cat not in _SUSPICIOUS_CATEGORIES:
        return False, None

    # Pegar nome oficial do Unicode
    try:
        nome = unicodedata.name(ch, '')
    except ValueError:
        nome = ''
    if not nome:
        nome = f'Unknown ({cat})'

    return True, nome


def detectar_watermarks(texto):
    """
    Detecta caracteres Unicode invisiveis e tipograficos no texto.
    Usa dicionarios conhecidos + scan generico por categoria Unicode.
    Retorna dict com:
      - 'invisiveis': lista de {sigla, nome, char, contagem, posicoes}
      - 'tipograficos': lista de {sigla, nome, char, contagem, posicoes}
      - 'desconhecidos': lista de {sigla, nome, char, contagem, posicoes}
      - 'total_invisiveis': int (conhecidos + desconhecidos)
      - 'total_tipograficos': int
      - 'tem_watermark': bool (True se encontrou invisiveis)
      - 'texto_limpo': str (texto sem caracteres invisiveis)
      - 'resumo': str (descricao curta do resultado)
    """
    invisiveis = []
    tipograficos = []
    desconhecidos_map = {}  # cp -> {info}

    # 1) Scan caracteres conhecidos
    for char, (sigla, nome) in _WATERMARK_CHARS.items():
        posicoes = [i for i, c in enumerate(texto) if c == char]
        if posicoes:
            invisiveis.append({
                'sigla': sigla,
                'nome': nome,
                'char': repr(char),
                'unicode': f'U+{ord(char):04X}',
                'contagem': len(posicoes),
                'posicoes': posicoes[:20],
            })

    for char, (sigla, nome) in _TYPO_CHARS.items():
        posicoes = [i for i, c in enumerate(texto) if c == char]
        if posicoes:
            tipograficos.append({
                'sigla': sigla,
                'nome': nome,
                'char': char,
                'unicode': f'U+{ord(char):04X}',
                'contagem': len(posicoes),
                'posicoes': posicoes[:20],
            })

    # 2) Scan GENERICO — pega qualquer caractere anomalo nao listado
    for i, ch in enumerate(texto):
        is_anom, nome = _is_anomalous_char(ch)
        if not is_anom:
            continue
        cp = ord(ch)
        if cp not in desconhecidos_map:
            cat = unicodedata.category(ch)
            sigla = f'U+{cp:04X}'
            desconhecidos_map[cp] = {
                'sigla': sigla,
                'nome': f'{nome} (cat: {cat})',
                'char': repr(ch),
                'unicode': f'U+{cp:04X}',
                'contagem': 0,
                'posicoes': [],
            }
        entry = desconhecidos_map[cp]
        entry['contagem'] += 1
        if len(entry['posicoes']) < 20:
            entry['posicoes'].append(i)

    desconhecidos = list(desconhecidos_map.values())

    total_conhecidos = sum(w['contagem'] for w in invisiveis)
    total_desconhecidos = sum(w['contagem'] for w in desconhecidos)
    total_inv = total_conhecidos + total_desconhecidos
    total_tipo = sum(w['contagem'] for w in tipograficos)

    # Gerar texto limpo (remove conhecidos + desconhecidos)
    texto_limpo = texto
    for char in _WATERMARK_CHARS:
        cat = unicodedata.category(char)
        if cat == 'Zs':  # qualquer space separator -> espaco normal
            texto_limpo = texto_limpo.replace(char, ' ')
        else:
            texto_limpo = texto_limpo.replace(char, '')

    # Remover desconhecidos tambem
    for cp in desconhecidos_map:
        ch = chr(cp)
        cat = unicodedata.category(ch)
        if cat == 'Zs':
            texto_limpo = texto_limpo.replace(ch, ' ')
        else:
            texto_limpo = texto_limpo.replace(ch, '')

    # Gerar resumo
    if total_inv == 0 and total_tipo == 0:
        resumo = "Nenhum caractere Unicode suspeito detectado."
    elif total_inv == 0:
        resumo = f"{total_tipo} caractere(s) tipografico(s) encontrado(s) — comum em texto IA mas nao conclusivo."
    else:
        todos_siglas = [w['sigla'] for w in invisiveis] + [w['sigla'] for w in desconhecidos]
        tipos = ', '.join(todos_siglas[:8])
        if len(todos_siglas) > 8:
            tipos += f' +{len(todos_siglas) - 8} outros'
        resumo = f"ALERTA: {total_inv} caractere(s) invisivel(is) detectado(s) ({tipos}) — possivel copia direta de ChatGPT/LLM!"

    return {
        'invisiveis': invisiveis,
        'tipograficos': tipograficos,
        'desconhecidos': desconhecidos,
        'total_invisiveis': total_inv,
        'total_tipograficos': total_tipo,
        'tem_watermark': total_inv > 0,
        'texto_limpo': texto_limpo,
        'resumo': resumo,
    }


def gerar_texto_watermark_destacado(texto):
    """Gera HTML do texto com caracteres invisiveis destacados.
    Vermelho = conhecidos, Laranja = desconhecidos, Azul = tipograficos."""
    result = []
    for ch in texto:
        if ch in _WATERMARK_CHARS:
            sigla, nome = _WATERMARK_CHARS[ch]
            result.append(
                f'<span style="background-color:#EF4444;color:white;padding:1px 4px;'
                f'border-radius:3px;font-size:12px;font-weight:bold;cursor:help" '
                f'title="{nome} (U+{ord(ch):04X})">[{sigla}]</span>'
            )
        elif ch in _TYPO_CHARS:
            sigla, nome = _TYPO_CHARS[ch]
            result.append(
                f'<span style="background-color:#3B82F6;color:white;padding:1px 3px;'
                f'border-radius:3px;font-size:12px;cursor:help" '
                f'title="{nome} (U+{ord(ch):04X})">{_html_escape(ch)}</span>'
            )
        elif ch == '\n':
            result.append('<br>')
        else:
            # Scan generico para desconhecidos
            is_anom, nome = _is_anomalous_char(ch)
            if is_anom:
                sigla = f'U+{ord(ch):04X}'
                result.append(
                    f'<span style="background-color:#F59E0B;color:white;padding:1px 4px;'
                    f'border-radius:3px;font-size:12px;font-weight:bold;cursor:help" '
                    f'title="{nome} (U+{ord(ch):04X})">[{sigla}]</span>'
                )
            else:
                result.append(_html_escape(ch))
    return ''.join(result)

# ---------------------------------------------------------------------------
# Analise por sentenca individual (estilo QuillBot/GPTZero)
# ---------------------------------------------------------------------------

def _score_sentenca(sentenca):
    """Pontua uma sentenca individual para probabilidade de IA (0-100).

    Usa sinais lexicais e estruturais que funcionam em sentenca isolada:
    - Conectivos e frases tipicas de IA
    - Contracoes (ausencia = IA)
    - Pronomes pessoais (ausencia = IA)
    - Ratio conteudo/funcao
    - Padroes formulaicos
    - Expressoes coloquiais (sinal humano)
    """
    sentenca_lower = sentenca.lower()
    pals = _palavras(sentenca)
    words = sentenca.split()
    n_words = len(words)

    if n_words < 3:
        return {
            'sentenca': sentenca,
            'score': 25,
            'classificacao': 'human',
            'confianca': 'low',
            'triggers': [],
        }

    score = 40
    triggers = []

    # === SINAIS DE IA (aumentam score) ===

    # 1. Conectivos IA (sinal forte)
    for conn in AI_CONNECTORS:
        if conn in sentenca_lower:
            bonus = 15 if len(conn.split()) > 1 else 10
            score += bonus
            triggers.append(('ai', conn))
            break

    # 2. Frases dead giveaway de IA
    for phrase in _SENTENCE_AI_PHRASES:
        if phrase in sentenca_lower:
            score += 25
            triggers.append(('ai', phrase))
            break

    # 3. Sem contracoes em frase longa
    contractions = _CONTRACTION_RE.findall(sentenca)
    if not contractions and n_words >= 8:
        for expanded, contracted in _EXPANDABLE_FORMS:
            if expanded.lower() in sentenca_lower:
                score += 8
                triggers.append(('ai', f"No contraction: {expanded}"))
                break
        else:
            score += 3

    # 4. Sem pronomes pessoais
    personal = sum(1 for p in pals if p in FIRST_PERSON or p in SECOND_PERSON)
    if personal == 0 and n_words >= 8:
        score += 5

    # 5. Ratio conteudo/funcao alto (formal demais)
    if len(pals) >= 5:
        funcao = sum(1 for p in pals if p in FUNCTION_WORDS)
        conteudo = len(pals) - funcao
        if funcao > 0:
            ratio = conteudo / funcao
            if ratio > 1.6:
                score += 8
                triggers.append(('ai', 'High formality'))

    # 6. Abertura formulaica
    for pat in _OPENERS:
        if re.search(pat, sentenca_lower.strip()):
            score += 18
            triggers.append(('ai', 'Formulaic opener'))
            break

    # 7. Fechamento formulaico
    for pat in _CLOSERS:
        if re.search(pat, sentenca_lower.strip()):
            score += 18
            triggers.append(('ai', 'Formulaic closer'))
            break

    # 8. Comeca com transicao formal
    if _TRANSITION_STARTERS.match(sentenca.strip()):
        score += 8
        triggers.append(('ai', 'Formal transition'))

    # 9. Palavras longas em media
    if pals:
        avg_wlen = sum(len(p) for p in pals) / len(pals)
        if avg_wlen >= 6.5:
            score += 5

    # === SINAIS HUMANOS (diminuem score) ===

    # 10. Contracoes presentes
    if contractions:
        score -= 12
        triggers.append(('human', 'Uses contractions'))

    # 11. Pronomes pessoais
    if personal > 0:
        score -= 8
        triggers.append(('human', 'Personal pronouns'))

    # 12. Frase curta e direta
    if n_words <= 5:
        score -= 5

    # 13. Linguagem coloquial
    if _COLLOQUIAL_RE.search(sentenca):
        score -= 12
        triggers.append(('human', 'Colloquial language'))

    # 14. Auto-correcao
    if _SELF_CORRECTION_RE.search(sentenca):
        score -= 10
        triggers.append(('human', 'Self-correction'))

    # 15. Comeca com "And", "But", "So" (informal)
    first_word = words[0].lower().rstrip(',')
    if first_word in ('and', 'but', 'so', 'look', 'yeah', 'nah', 'ok', 'okay'):
        score -= 5
        triggers.append(('human', f'Informal start: "{first_word}"'))

    # Clamp 0-100
    score = max(0, min(100, score))

    # Classificacao
    if score >= 65:
        classificacao = 'ai'
        confianca = 'high' if score >= 80 else 'medium'
    elif score >= 45:
        classificacao = 'mixed'
        confianca = 'medium' if score >= 55 else 'low'
    else:
        classificacao = 'human'
        confianca = 'high' if score <= 20 else ('medium' if score <= 35 else 'low')

    return {
        'sentenca': sentenca,
        'score': score,
        'classificacao': classificacao,
        'confianca': confianca,
        'triggers': triggers,
    }


def avaliar_por_sentenca(texto):
    """Analisa texto sentenca por sentenca. Retorna lista de resultados.

    Cada resultado contem:
    - sentenca: str
    - score: int (0-100)
    - classificacao: 'ai' | 'mixed' | 'human'
    - confianca: 'high' | 'medium' | 'low'
    - triggers: list of (tipo, descricao) tuples
    """
    is_code = detectar_codigo(texto)
    if is_code:
        return []

    sentencas = _frases(texto)
    if not sentencas:
        return []

    resultados = []
    for sent in sentencas:
        resultado = _score_sentenca(sent)
        resultados.append(resultado)

    return resultados


def gerar_html_sentencas_individual(texto, resultados_sentenca):
    """Gera HTML com cores por sentenca baseado na analise individual.

    Cores:
    - Dourado: AI-generated (score >= 65)
    - Azul claro: AI-refined/mixed (score 45-65)
    - Sem cor: Human (score < 45)
    """
    if not resultados_sentenca:
        return _html_escape(texto).replace('\n', '<br>')

    result = []
    offset = 0

    for res in resultados_sentenca:
        sent = res['sentenca']
        score = res['score']
        classificacao = res['classificacao']

        idx = texto.find(sent, offset)
        if idx < 0:
            idx = offset

        if idx > offset:
            between = texto[offset:idx]
            result.append(_html_escape(between).replace('\n', '<br>'))

        sent_html = _revelar_chars_escondidos(sent)

        if classificacao == 'ai':
            bg = '#F5DEB3'
            border_color = '#D4A843'
        elif classificacao == 'mixed':
            bg = '#E8F0FE'
            border_color = '#93C5FD'
        else:
            bg = 'transparent'
            border_color = 'transparent'

        if classificacao != 'human':
            trigger_text = '; '.join(t[1] for t in res['triggers'][:3])
            title = f"Score: {score}% — {classificacao.upper()}"
            if trigger_text:
                title += f" | {trigger_text}"
            result.append(
                f'<span style="background:{bg};padding:2px 4px;'
                f'border-bottom:2px solid {border_color};color:#1a1a1a;'
                f'cursor:help;" title="{_html_escape(title)}">'
                f'{sent_html}</span> '
            )
        else:
            result.append(f'{sent_html} ')

        offset = idx + len(sent)

    if offset < len(texto):
        result.append(_html_escape(texto[offset:]).replace('\n', '<br>'))

    html = ''.join(result)
    html = html.replace('\n\n', '</p><p style="margin-top:12px;">')
    html = html.replace('\n', '<br>')
    return f'<p>{html}</p>'


# ---------------------------------------------------------------------------
# Exibicao
# ---------------------------------------------------------------------------

def barra(valor, maximo, largura=20):
    n = int((valor / maximo) * largura)
    n = max(0, min(n, largura))
    return "#" * n + "-" * (largura - n)


def rotulo_nivel(score):
    if score >= 85: return "MUITO PROVAVEL IA"
    if score >= 65: return "PROVAVEL IA"
    if score >= 45: return "INCONCLUSIVO"
    if score >= 25: return "PROVAVEL HUMANO"
    return "MUITO PROVAVEL HUMANO"


def exibir_resultado(score, metricas):
    m = metricas
    print("\n" + "=" * 70)
    print("         ANALISE DE DETECCAO DE CONTEUDO IA  v2")
    print("         (15 metricas | Estatistica + Compressao)")
    print("=" * 70)
    print()
    print(f"  Entropia Shannon:     {m['entropia']:>7.2f} bits")
    print(f"  Taxa compressao:      {m['compressao']:>7.4f}  {'ALTA (IA)' if m['compressao'] < 0.40 else 'OK'}")
    print(f"  SD compr. frases:     {m['sent_sd']:>7.2f}  {'BAIXA (IA)' if m['sent_sd'] < 5.0 else 'OK'}")
    print(f"  Var. frases adj.:     {m['var_adj']:>7.4f}  {'BAIXA (IA)' if m['var_adj'] < 0.4 else 'OK'}")
    print(f"  Burstiness (CV):      {m['burstiness']:>7.4f}  {'BAIXA (IA)' if m['burstiness'] < 0.3 else 'OK'}")
    print(f"  Diversidade (TTR):    {m['ttr']:>7.4f}")
    print(f"  Hapax Legomena:       {m['hapax']:>7.4f}")
    print(f"  Ratio cont./funcao:  {m['ratio_cf']:>7.4f}  {'ALTA (IA)' if m['ratio_cf'] > 1.2 else 'OK'}")
    print(f"  Contracoes/100p:      {m['contracoes']:>7.2f}  {'ZERO (IA)' if m['contracoes'] < 0.5 else 'OK'}")
    print(f"  Entropia pontuacao:   {m['ent_pont']:>7.2f}")
    print(f"  Conectivos IA/frase:  {m['conectivos']:>7.4f}  {'ALTA (IA)' if m['conectivos'] > 0.3 else 'OK'}")
    print(f"  Div. inicio frase:    {m['div_inicio']:>7.4f}")
    print(f"  Pronomes pessoais:    {m['pronomes']:>7.2f}%  {'ZERO (IA)' if m['pronomes'] < 1.5 else 'OK'}")
    print(f"  Frases tipicas IA:    {m['frases_ia']:>7.2f}/100p")
    print(f"  Padroes formulaicos:  {m['formulaicos']:>7.2f}")
    print()
    print("-" * 70)
    print(f"  SCORE FINAL:  {score:.1f}%  {barra(score, 100, 35)}")
    print(f"  VEREDICTO:    {rotulo_nivel(score)}")
    print("-" * 70)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Detector de Texto IA v2 (15 metricas)")
    parser.add_argument("--texto", "-t", type=str)
    parser.add_argument("--arquivo", "-a", type=str)
    args = parser.parse_args()

    if args.arquivo:
        with open(args.arquivo, "r", encoding="utf-8") as f:
            texto = f.read()
    elif args.texto:
        texto = args.texto
    else:
        print("Cole o texto para analisar (linha vazia para finalizar):\n")
        linhas = []
        try:
            while True:
                linha = input()
                if linha == "" and linhas:
                    break
                linhas.append(linha)
        except EOFError:
            pass
        texto = "\n".join(linhas).strip()

    if not texto or len(texto.split()) < 10:
        print("[!] Texto muito curto. Forneca pelo menos 10 palavras.")
        return

    print(f"[*] Texto: {len(texto)} caracteres, {len(texto.split())} palavras")
    score, metricas = avaliar(texto)
    exibir_resultado(score, metricas)


if __name__ == "__main__":
    main()
