"""
IA Detector — Interface Web
Streamlit app para gerar, detectar e humanizar texto IA.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import difflib
import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv
from detector import (avaliar, rotulo_nivel, detectar_codigo, analisar_problemas,
                      aplicar_correcao, simular_correcao, simular_todas, gerar_texto_destacado,
                      gerar_texto_destacado_sentencas,
                      detectar_watermarks, gerar_texto_watermark_destacado,
                      avaliar_por_sentenca, gerar_html_sentencas_individual)

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="IA Detector", page_icon="🔍", layout="wide")

GEMINI_CONFIGURADO = False


def configurar_gemini(api_key=None):
    global GEMINI_CONFIGURADO
    key = api_key or os.getenv("GEMINI_API_KEY")
    if key:
        genai.configure(api_key=key)
        GEMINI_CONFIGURADO = True
        return True
    return False


# ---------------------------------------------------------------------------
# Funcoes
# ---------------------------------------------------------------------------

def _html_escape_app(s):
    """HTML escape helper for app.py templates."""
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def gerar_texto(tema, palavras=250, temperature=0.7):
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = (
        f"Write a detailed, well-structured text about: {tema}\n"
        f"Write approximately {palavras} words. "
        f"Use a formal, informative tone typical of AI-generated content."
    )
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=palavras * 2,
        ),
    )
    return response.text


PROMPTS_HUMANIZAR = [
    # Nivel 1: Suave
    """Rewrite the following text to sound more naturally human-written.
Apply these techniques:
- Vary sentence lengths (mix short punchy sentences with longer ones)
- Replace formal connectors (moreover, furthermore, additionally) with casual transitions
- Use contractions where natural (don't, isn't, it's)
- Avoid starting consecutive sentences the same way
CRITICAL: Keep the same meaning. Only change HOW it reads.

Text:
{texto}""",

    # Nivel 2: Moderado
    """The following text still reads like AI. Rewrite it more naturally:
- Replace ALL academic phrases with conversational equivalents
- Add occasional hedging (maybe, probably, I think, arguably)
- Use contractions everywhere
- Vary sentence structure: some short, some long
- Avoid: moreover, furthermore, crucial, comprehensive, facilitate, leverage, utilize, paradigm
Keep the same core information.

Text:
{texto}""",

    # Nivel 3: Forte
    """This text is STILL detected as AI. Rewrite MORE aggressively:
- Use VERY varied sentence lengths: some 3 words, others 30+
- Add a personal touch or opinion
- Use rhetorical questions or self-corrections
- Replace ALL formal language with casual equivalents
- Avoid: moreover, furthermore, crucial, vital, comprehensive, multifaceted, leverage, utilize, facilitate, holistic, robust, landscape, paradigm, underscores, highlights
Keep the same core information.

Text:
{texto}""",

    # Nivel 4: Agressivo
    """This text is STILL detected as AI. Major transformation needed:
- Write like a real person — imperfect, natural, with personality
- Some sentences just 3-5 words, others rambling
- Use slang, idioms, informal expressions
- Start sentences with "And", "But", "So", "Look,"
- Add parenthetical asides (like this) or dashes — for emphasis
- Remove ANY "essay structure" — no "in conclusion", no "firstly/secondly"
- Sound opinionated, not neutral
Same information, completely different voice.

Text:
{texto}""",

    # Nivel 5: Maximo
    """This text is STILL flagged as AI after 4 rewrites. MAXIMUM transformation:
- Rewrite from scratch in your own casual voice
- Be messy, be real, be human
- Use incomplete thoughts, corrections mid-sentence
- Mix long rants with one-word reactions
- Throw in "honestly", "like", "you know", "right?"
- NO academic structure whatsoever
- Sound like a blog post or a text message, not an essay
Same core facts, totally different person writing it.

Text:
{texto}""",
]


PROMPTS_HUMANIZAR_CODE = [
    # Nivel 1: Suave
    """You are a code rewriter. Rewrite AI-generated code so it looks like a REAL STUDENT typed it.

MANDATORY changes:
1. DELETE every comment. ALL of them. No "# I added...", no "# This function...". ZERO comments.
2. RENAME variables to short names (adj, exc, v1, ani, p, f). Variable names must NOT match the input() prompt text exactly.
3. If there are 3+ sequential input() calls, use a different approach: a loop, a list comprehension, or a dict.
4. Add one small imperfection (extra blank line, inconsistent spacing, a temp variable).

CRITICAL RULES:
- The strings INSIDE print() and input() must stay EXACTLY the same. The user must see the same prompts and output.
- Only change variable names and code structure, NEVER the text the user sees.
- Output ONLY the rewritten code. No explanations. No markdown. No ``` fences.

{texto}""",

    # Nivel 2: Moderado
    """The code below is STILL detected as AI. Rewrite it differently.

MANDATORY:
1. ZERO comments
2. Short variable names (2-4 chars), must NOT match input prompt text
3. Use a loop or dict for repeated input() calls. Keep the ORIGINAL prompt strings.
4. Add a small quirk: unused variable, extra blank line, slightly weird structure
5. DO NOT change any string the user sees (print text, input prompts)

Output ONLY code. No text. No markdown. No ``` fences.

{texto}""",

    # Nivel 3: Forte
    """The code below is STILL detected as AI. Rewrite MORE aggressively.

MANDATORY:
1. ZERO comments
2. ALL variable names 1-3 characters (a, b, v1, e, p, f)
3. RESTRUCTURE: use a dict or loop for repeated input() calls. Keep the ORIGINAL prompt strings inside input().
4. Mix styles: some concatenation + some f-strings
5. Add a quirk: weird temp variable, unnecessary list, etc.
6. DO NOT change any string that the user sees (print text, input prompts). Only change code structure and variable names.

Output ONLY code. No text. No markdown. No ``` fences.

{texto}""",

    # Nivel 4: Agressivo
    """AGGRESSIVE rewrite. This code must look like a student who codes fast and messy.

Rules:
1. ZERO comments
2. Single-letter variables where possible
3. Completely different structure: dicts, loops, list comprehensions
4. Inconsistent style on purpose (mix concat and f-strings, inconsistent spacing)
5. Maybe combine prints with \\n
6. NEVER change the text inside print() or input() — same user-facing output
7. Same behavior

Output ONLY code. Nothing else. No ``` fences.

{texto}""",

    # Nivel 5: Maximo
    """MAXIMUM rewrite. 5th attempt. Make this code look completely different from the original.

Rules:
1. ZERO comments, ZERO docstrings
2. Single-letter variables ONLY (a, b, c, d, e, f, g, h)
3. Totally restructure: use a dict comprehension or enumerate loop for all inputs
4. Compress everything: fewer lines, dense code
5. Maybe use semicolons to put multiple statements on one line
6. Use string .format() or % formatting instead of f-strings for variety
7. Keep exact same user-facing behavior
8. Make it look like someone wrote it in 2 minutes

Output ONLY code. Nothing else. No ``` fences.

{texto}""",
]


def humanizar_texto(texto, iteracao=0):
    model = genai.GenerativeModel("gemini-2.0-flash")
    is_code = detectar_codigo(texto)

    if is_code:
        prompts = PROMPTS_HUMANIZAR_CODE
    else:
        prompts = PROMPTS_HUMANIZAR

    idx = min(iteracao, len(prompts) - 1)
    prompt = prompts[idx].format(texto=texto)
    temp = 1.0 + (iteracao * 0.15) if is_code else 0.85 + (iteracao * 0.15)
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=min(temp, 1.8),
            max_output_tokens=len(texto.split()) * 4,
        ),
    )
    result = response.text
    if is_code and result.startswith("```"):
        lines = result.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        result = '\n'.join(lines)
    return result


# Palavras banidas no rewrite (AI dead giveaways)
_BANNED_WORDS = [
    "moreover", "furthermore", "additionally", "consequently", "nevertheless",
    "comprehensive", "multifaceted", "paradigm", "leverage", "utilize",
    "facilitate", "delve", "robust", "holistic", "landscape", "realm",
    "encompass", "transformative", "paramount", "fundamental", "imperative",
    "crucial", "vital", "pivotal", "significantly", "substantially",
    "innovative", "groundbreaking", "pioneering", "noteworthy",
    "underscores", "highlights", "revolutionize", "streamline",
    "foster", "cultivate", "bolster", "harness", "spearhead",
    "navigate", "embark", "endeavor", "commence",
    "AI systems", "AI-powered", "machine learning algorithms",
    "pedagogical", "methodologies", "proficiency",
]
_BANNED_WORDS_STR = ", ".join(_BANNED_WORDS)


def gerar_reescrita_completa(texto, score):
    """Usa Gemini para gerar reescrita completa do texto, estilo QuillBot."""
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""You are a human ghostwriter. Your job is to rewrite this text so that NO AI detector can flag it.
The current AI detection score is {score:.0f}%.

CRITICAL RULES — follow ALL of them:

1. SENTENCE RHYTHM: Mix aggressively. Some sentences must be 3-6 words. Others 25+. Never put two sentences of similar length next to each other.

2. IMPERFECTIONS: Real humans are messy writers. Add:
   - Self-corrections ("well, actually...", "or rather,")
   - Parenthetical asides (like this one)
   - Sentence fragments. Just like this.
   - Start some sentences with "And", "But", "So", "Look,"

3. STRONG PERSONAL VOICE: Do NOT be neutral. Take a side. Use:
   - "I think", "I'd argue", "if you ask me", "from what I've seen"
   - Rhetorical questions ("But does that actually work?")
   - Opinions ("which is pretty wild when you think about it")

4. BANNED WORDS — never use these: {_BANNED_WORDS_STR}

5. CONTRACTIONS: Use them everywhere. "it is" → "it's", "do not" → "don't", "they are" → "they're". No exceptions.

6. COLLOQUIAL LANGUAGE: Write like a blog post, not an essay.
   - "a lot of" not "numerous"
   - "pretty much" not "essentially"
   - "messed up" not "problematic"
   - "huge" not "significant"
   - "stuff" or "things" not "elements" or "components"

7. NO ESSAY STRUCTURE: No "firstly", no "in conclusion", no "it is important to note". Just talk.

8. KEEP THE SAME MEANING: All core facts must stay. Only change HOW it reads.

9. Output ONLY the rewritten text. No intro, no explanation.

TEXT TO REWRITE:
{texto}"""

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=1.0,
            max_output_tokens=len(texto.split()) * 5,
        ),
    )
    return response.text.strip()


def gerar_reescrita_sentenca(sentenca, score):
    """Reescreve uma unica sentenca via Gemini para soar mais humana."""
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""Rewrite ONLY this single sentence to sound naturally human-written.
Current AI detection score: {score}%.

Rules:
- Use contractions (don't, it's, can't, they're)
- Use casual, conversational tone
- Add a personal touch if it fits
- Keep the EXACT same meaning
- NEVER use: moreover, furthermore, additionally, consequently, nevertheless, comprehensive, crucial, facilitate, leverage, utilize, delve, robust, holistic, landscape, paradigm
- Output ONLY the rewritten sentence. No quotes. No explanation.

Original: {sentenca}"""

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=1.0,
            max_output_tokens=200,
        ),
    )
    return response.text.strip().strip('"').strip("'")


def _gerar_diff_html(original, reescrito):
    """Gera HTML do texto reescrito com palavras alteradas destacadas.
    Vermelho = palavras novas/alteradas no texto reescrito."""

    orig_words = original.split()
    new_words = reescrito.split()
    sm = difflib.SequenceMatcher(None, orig_words, new_words)

    html_parts = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            html_parts.append(' '.join(new_words[j1:j2]))
        elif op == 'replace':
            for w in new_words[j1:j2]:
                html_parts.append(
                    f'<span style="color:#C0392B;font-weight:500;">{w}</span>'
                )
        elif op == 'insert':
            for w in new_words[j1:j2]:
                html_parts.append(
                    f'<span style="color:#2E86C1;text-decoration:underline;">{w}</span>'
                )
        # 'delete' — palavras removidas, nao mostramos no novo texto

    return ' '.join(html_parts)


def mostrar_metricas(score, metricas, prefix=""):
    m = metricas

    def cor_score(s):
        if s >= 85: return "🔴"
        if s >= 65: return "🟠"
        if s >= 45: return "🟡"
        return "🟢"

    st.markdown(f"### {prefix}{cor_score(score)} Score: **{score:.1f}%** — {rotulo_nivel(score)}")

    is_code = m.get('is_code', False)

    if is_code:
        st.caption("Modo: Codigo detectado — analisando comentarios + estrutura")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Compressao (ZipPy)", f"{m['compressao']:.4f}",
                       help="Taxa de compressao. Baixa = previsivel = IA")
            st.metric("Comentarios/codigo", f"{m.get('code_comment_ratio', 0):.2f}",
                       help="Ratio comentarios/codigo. Alta (>0.3) = IA over-comments")
        with col2:
            st.metric("Frases pedagogicas", f"{m.get('code_pedagogical', 0)}",
                       help="Frases como 'I added', 'This function', 'Here we'...")
            st.metric("Uniformidade linhas", f"{m.get('code_line_cv', 0):.3f}",
                       help="CV do comprimento das linhas. Baixo = codigo uniforme = IA")
        with col3:
            st.metric("Score NL extraido", f"{m.get('code_nl_score', 0):.1f}%",
                       help="Score do texto natural extraido (comentarios/strings)")
            st.metric("Entropia Shannon", f"{m['entropia']:.2f} bits",
                       help="Entropia da distribuicao de palavras")
    else:
        st.markdown("**Metricas principais**")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Compressao (ZipPy)", f"{m['compressao']:.4f}",
                       help="Taxa de compressao zlib. Baixa = texto previsivel = IA")
            st.metric("SD compr. frases", f"{m['sent_sd']:.2f}",
                       help="Desvio padrao do comprimento das frases. Baixa = IA")
            st.metric("Entropia Shannon", f"{m['entropia']:.2f} bits",
                       help="Entropia da distribuicao de palavras. Baixa = IA")

        with col2:
            st.metric("Conectivos IA/frase", f"{m['conectivos']:.3f}",
                       help="'moreover','furthermore','additionally'... Alta = IA")
            st.metric("Ratio cont./funcao", f"{m['ratio_cf']:.3f}",
                       help="Palavras de conteudo / funcao. Alta (>1.3) = IA")
            st.metric("Contracoes/100p", f"{m['contracoes']:.2f}",
                       help="Contracoes por 100 palavras. Zero = IA")

        with col3:
            st.metric("Frases tipicas IA", f"{m['frases_ia']:.2f}/100p",
                       help="Expressoes dead giveaway de IA")
            st.metric("Pronomes pessoais", f"{m['pronomes']:.2f}%",
                       help="1a/2a pessoa. Baixa (<1.5%) = IA")
            st.metric("Padroes formulaicos", f"{m['formulaicos']:.2f}",
                       help="Aberturas/fechamentos tipicos de IA (0-1)")

        with st.expander("Mais metricas"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Burstiness (CV)", f"{m['burstiness']:.4f}",
                           help="Variacao no comprimento das frases")
                st.metric("Var. frases adj.", f"{m['var_adj']:.4f}",
                           help="Variancia entre frases consecutivas. Baixa = IA")
            with c2:
                st.metric("Diversidade (TTR)", f"{m['ttr']:.4f}",
                           help="Palavras unicas / total")
                st.metric("Hapax Legomena", f"{m['hapax']:.4f}",
                           help="Palavras usadas 1x")
            with c3:
                st.metric("Entropia pontuacao", f"{m['ent_pont']:.2f}",
                           help="Entropia dos intervalos de pontuacao. Baixa = IA")
                st.metric("Div. inicio frase", f"{m['div_inicio']:.4f}",
                           help="Variedade no inicio das frases")

    st.progress(min(score / 100, 1.0))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Configuracao")

    api_key_input = st.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Pegue em: https://aistudio.google.com/apikey",
    )

    if api_key_input:
        configurar_gemini(api_key_input)
        st.success("API Key configurada")
    else:
        if configurar_gemini():
            st.success("API Key do .env carregada")
        else:
            st.warning("Configure a API Key do Gemini")

    st.divider()
    st.markdown("**Projeto IA-Detector**")
    st.caption("Gemini API + 15 metricas estatisticas")
    st.caption("Inspirado em DetectGPT, GLTR, ZipPy, GPTZero")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["📝 Gerador", "🔍 Detector", "🔄 Humanizador"])


# --- TAB 1: GERADOR ---
with tab1:
    st.header("Gerador de Texto IA")
    st.caption("Gera texto sobre qualquer tema usando Gemini")

    col_input, col_config = st.columns([3, 1])

    with col_input:
        tema = st.text_input("Tema", placeholder="Ex: The impact of AI on education")

    with col_config:
        palavras = st.slider("Palavras", 100, 500, 250, 50)
        temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1)

    if st.button("Gerar Texto", type="primary", disabled=not GEMINI_CONFIGURADO):
        if not tema:
            st.warning("Digite um tema")
        else:
            with st.spinner("Gerando com Gemini..."):
                texto = gerar_texto(tema, palavras, temperature)
                st.session_state["texto_gerado"] = texto

    if "texto_gerado" in st.session_state:
        st.text_area("Texto Gerado", st.session_state["texto_gerado"], height=300)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📋 Copiar para Detector"):
                st.session_state["texto_detector"] = st.session_state["texto_gerado"]
                st.success("Copiado para aba Detector")
        with col_b:
            if st.button("📋 Copiar para Humanizador"):
                st.session_state["texto_humanizar"] = st.session_state["texto_gerado"]
                st.success("Copiado para aba Humanizador")


# --- TAB 2: DETECTOR ---

def _aplicar_fix(prob_index):
    """Callback: aplica uma correcao e re-analisa."""
    prob = st.session_state.det_problemas[prob_index]
    texto = st.session_state.ta_detector
    texto = aplicar_correcao(texto, prob)
    st.session_state.ta_detector = texto
    score, metricas = avaliar(texto)
    st.session_state.det_score = score
    st.session_state.det_metricas = metricas
    st.session_state.det_problemas = analisar_problemas(texto)
    st.session_state.det_sentencas = avaliar_por_sentenca(texto)
    st.session_state.det_previews = {}
    st.session_state["ultimo_score"] = score
    st.session_state["ultimas_metricas"] = metricas


def _aplicar_todas():
    """Callback: aplica todas as correcoes auto-fixaveis."""
    fixable = [p for p in st.session_state.det_problemas if 'corrigido' in p]
    texto = st.session_state.ta_detector
    for prob in sorted(fixable, key=lambda p: p.get('posicao', 0), reverse=True):
        texto = aplicar_correcao(texto, prob)
    st.session_state.ta_detector = texto
    score, metricas = avaliar(texto)
    st.session_state.det_score = score
    st.session_state.det_metricas = metricas
    st.session_state.det_problemas = analisar_problemas(texto)
    st.session_state.det_sentencas = avaliar_por_sentenca(texto)
    st.session_state.det_previews = {}
    st.session_state["ultimo_score"] = score
    st.session_state["ultimas_metricas"] = metricas


def _aplicar_reescrita():
    """Callback: aplica reescrita completa e re-analisa."""
    reescrita = st.session_state.get("det_reescrita")
    if reescrita:
        st.session_state.ta_detector = reescrita
        score, metricas = avaliar(reescrita)
        st.session_state.det_score = score
        st.session_state.det_metricas = metricas
        problemas = analisar_problemas(reescrita)
        st.session_state.det_problemas = problemas
        st.session_state.det_sentencas = avaliar_por_sentenca(reescrita)
        st.session_state.det_previews = {}
        st.session_state["ultimo_score"] = score
        st.session_state["ultimas_metricas"] = metricas
        # Auto-gerar nova reescrita se score ainda alto
        if score >= 45 and GEMINI_CONFIGURADO:
            try:
                nova = gerar_reescrita_completa(reescrita, score)
                st.session_state.det_reescrita = nova
            except Exception:
                st.session_state.det_reescrita = None
        else:
            st.session_state.det_reescrita = None


def _tentar_novamente():
    """Callback: regenera reescrita via Gemini."""
    texto = st.session_state.ta_detector
    score = st.session_state.det_score
    try:
        reescrita = gerar_reescrita_completa(texto, score)
        st.session_state.det_reescrita = reescrita
    except Exception:
        st.session_state.det_reescrita = None


def _cancelar_reescrita():
    """Callback: cancela a reescrita."""
    st.session_state.det_reescrita = None


def _reescrever_sentenca(idx):
    """Callback: gera reescrita de uma sentenca individual via Gemini."""
    sentencas = st.session_state.get('det_sentencas', [])
    if idx >= len(sentencas):
        return
    s = sentencas[idx]
    try:
        reescrita = gerar_reescrita_sentenca(s['sentenca'], s['score'])
        if 'det_sent_rewrites' not in st.session_state:
            st.session_state.det_sent_rewrites = {}
        st.session_state.det_sent_rewrites[idx] = reescrita
    except Exception:
        pass


def _replace_sentenca(idx):
    """Callback: aplica reescrita de uma sentenca ao texto."""
    rewrites = st.session_state.get('det_sent_rewrites', {})
    if idx not in rewrites:
        return
    sentencas = st.session_state.get('det_sentencas', [])
    if idx >= len(sentencas):
        return

    original = sentencas[idx]['sentenca']
    reescrita = rewrites[idx]
    texto = st.session_state.ta_detector
    texto = texto.replace(original, reescrita, 1)

    st.session_state.ta_detector = texto
    score, metricas = avaliar(texto)
    st.session_state.det_score = score
    st.session_state.det_metricas = metricas
    st.session_state.det_problemas = analisar_problemas(texto)
    st.session_state.det_sentencas = avaliar_por_sentenca(texto)
    st.session_state.det_sent_rewrites = {}
    st.session_state.det_previews = {}
    st.session_state["ultimo_score"] = score
    st.session_state["ultimas_metricas"] = metricas


def _try_again_sentenca(idx):
    """Callback: regenera reescrita de uma sentenca."""
    _reescrever_sentenca(idx)


def _cancel_sentenca(idx):
    """Callback: cancela reescrita de uma sentenca."""
    rewrites = st.session_state.get('det_sent_rewrites', {})
    if idx in rewrites:
        del rewrites[idx]


with tab2:
    st.caption("15 metricas: entropia, compressao, estilometria — sem modelos pesados")

    if "texto_detector" in st.session_state:
        st.session_state.ta_detector = st.session_state.pop("texto_detector")
        for k in ("det_score", "det_metricas", "det_problemas", "det_previews"):
            st.session_state.pop(k, None)

    texto_detector = st.text_area(
        "Cole o texto para analisar",
        height=180,
        placeholder="Cole aqui o texto que deseja analisar...",
        key="ta_detector",
        label_visibility="collapsed",
    )

    analisar_clicked = st.button("Analisar", type="primary")

    if analisar_clicked:
        if not texto_detector or len(texto_detector.split()) < 10:
            st.warning("Texto muito curto. Minimo 10 palavras.")
        else:
            score, metricas = avaliar(texto_detector)
            problemas = analisar_problemas(texto_detector)
            watermarks = detectar_watermarks(texto_detector)
            sentencas = avaliar_por_sentenca(texto_detector)
            st.session_state.det_score = score
            st.session_state.det_metricas = metricas
            st.session_state.det_problemas = problemas
            st.session_state.det_watermarks = watermarks
            st.session_state.det_sentencas = sentencas
            st.session_state.det_sent_rewrites = {}
            st.session_state["ultimo_score"] = score
            st.session_state["ultimas_metricas"] = metricas

            previews = {}
            for i, prob in enumerate(problemas):
                if 'corrigido' in prob:
                    novo_score, _ = simular_correcao(texto_detector, prob)
                    if novo_score is not None:
                        previews[i] = novo_score
            all_score, _ = simular_todas(texto_detector, problemas)
            if all_score is not None:
                previews['all'] = all_score
            st.session_state.det_previews = previews

            # Gerar reescrita completa via Gemini (se score alto)
            if score >= 45 and GEMINI_CONFIGURADO:
                try:
                    reescrita = gerar_reescrita_completa(texto_detector, score)
                    st.session_state.det_reescrita = reescrita
                except Exception:
                    st.session_state.det_reescrita = None
            else:
                st.session_state.det_reescrita = None

    if "det_score" in st.session_state:
        score_atual = st.session_state.det_score
        metricas = st.session_state.get("det_metricas", {})
        problemas = st.session_state.get("det_problemas", [])
        previews = st.session_state.get("det_previews", {})

        def _cor_hex(s):
            if s >= 85: return "#EF4444"
            if s >= 65: return "#D97706"
            if s >= 45: return "#EAB308"
            return "#22C55E"

        def _label_en(s):
            if s >= 85: return "AI"
            if s >= 65: return "AI"
            if s >= 45: return "Mixed"
            return "Human"

        cor = _cor_hex(score_atual)

        # ====================================================
        # Layout QuillBot: 2 colunas
        # ====================================================
        col_left, col_right = st.columns([5, 3], gap="large")

        with col_left:
            # --- Texto com highlights por sentenca individual ---
            if texto_detector:
                sentencas_res = st.session_state.get('det_sentencas', [])
                if sentencas_res:
                    html_texto = gerar_html_sentencas_individual(
                        texto_detector, sentencas_res
                    )
                else:
                    html_texto = gerar_texto_destacado_sentencas(
                        texto_detector, score_atual, problemas
                    )
                n_words = len(texto_detector.split())
                n_sents = len(sentencas_res) if sentencas_res else 0

                # Legend
                legend = (
                    '<div style="display:flex;gap:16px;margin-bottom:12px;flex-wrap:wrap;">'
                    '<span style="font-size:12px;color:#666;">'
                    '<span style="background:#F5DEB3;padding:2px 8px;border-radius:3px;'
                    'border-bottom:2px solid #D4A843;">AI-generated</span></span>'
                    '<span style="font-size:12px;color:#666;">'
                    '<span style="background:#E8F0FE;padding:2px 8px;border-radius:3px;'
                    'border-bottom:2px solid #93C5FD;">AI-refined</span></span>'
                    '<span style="font-size:12px;color:#666;">'
                    '<span style="padding:2px 8px;">Human</span></span>'
                    '</div>'
                ) if sentencas_res else ''

                sent_info = f' | {n_sents} Sentences analyzed' if n_sents else ''
                st.markdown(
                    f'<div style="background:#FFFFFF;padding:24px;border-radius:10px;'
                    f'font-size:15px;line-height:2.0;color:#1a1a1a;border:1px solid #E5E7EB;'
                    f'min-height:120px;">'
                    f'{legend}'
                    f'{html_texto}'
                    f'<div style="border-top:1px solid #E5E7EB;margin-top:16px;padding-top:8px;'
                    f'display:flex;align-items:center;gap:16px;">'
                    f'<span style="color:#666;font-size:13px;">{n_words} Words{sent_info}</span>'
                    f'<span style="color:#22C55E;font-size:13px;">Analysis complete</span>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

            # --- Reescrita completa estilo QuillBot ---
            reescrita = st.session_state.get("det_reescrita")
            if reescrita:
                prob_level = "High Probability" if score_atual >= 65 else (
                    "Medium Probability" if score_atual >= 45 else "Low Probability")
                diff_html = _gerar_diff_html(texto_detector, reescrita)

                st.markdown(
                    f'<div style="background:#FFF8E1;border:1px solid #D4A843;'
                    f'border-radius:10px;padding:20px;margin:16px 0;">'
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">'
                    f'<span style="background:#D97706;color:white;padding:3px 12px;'
                    f'border-radius:4px;font-size:12px;font-weight:600;">AI-generated</span>'
                    f'<span style="color:#92400E;font-size:13px;">{prob_level}</span>'
                    f'</div>'
                    f'<p style="color:#78716C;font-size:13px;margin:0 0 8px 0;">'
                    f'Rewritten text:</p>'
                    f'<div style="color:#1a1a1a;font-size:14px;line-height:1.8;'
                    f'background:#FFFEF5;padding:12px;border-radius:6px;">'
                    f'{diff_html}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                col_try, col_cancel, col_replace = st.columns([1, 1, 1])
                with col_try:
                    st.button("🔄 Try again", on_click=_tentar_novamente,
                              use_container_width=True)
                with col_cancel:
                    st.button("✕ Cancel", on_click=_cancelar_reescrita,
                              use_container_width=True)
                with col_replace:
                    st.button("✓ Replace", type="primary",
                              on_click=_aplicar_reescrita,
                              use_container_width=True)

            # --- Correcoes individuais (colapsavel) ---
            fixable = [p for p in problemas if 'corrigido' in p]
            if fixable:
                with st.expander(
                    f"🔧 Individual fixes ({len(fixable)})", expanded=False
                ):
                    for i, prob in enumerate(problemas):
                        if 'corrigido' not in prob:
                            continue
                        preview_score = previews.get(i)
                        delta_txt = ""
                        if preview_score is not None:
                            delta = score_atual - preview_score
                            delta_txt = f' → {preview_score:.0f}% (-{delta:.0f}%)'

                        st.markdown(
                            f'<div style="background:#F9FAFB;border:1px solid #E5E7EB;'
                            f'border-radius:6px;padding:10px;margin:6px 0;">'
                            f'<p style="color:#666;font-size:12px;margin:0 0 4px 0;">'
                            f'{prob["motivo"]}{delta_txt}</p>'
                            f'<p style="color:#1a1a1a;font-size:13px;margin:0;'
                            f'font-style:italic;">'
                            f'{prob["corrigido"] if prob["corrigido"] else "(remover)"}'
                            f'</p></div>',
                            unsafe_allow_html=True,
                        )
                        st.button(
                            f"↻ Replace",
                            key=f"replace_{i}",
                            on_click=_aplicar_fix,
                            args=(i,),
                        )

                    if len(fixable) > 1:
                        all_preview = previews.get('all')
                        label = f"↻ Replace All ({len(fixable)})"
                        if all_preview is not None:
                            delta = score_atual - all_preview
                            label += f" → {all_preview:.0f}%"
                        st.button(label, type="secondary",
                                  on_click=_aplicar_todas,
                                  use_container_width=True)

            # Sugestoes nao-fixaveis (apenas dica) — deduplica
            non_fixable = [p for p in problemas if 'corrigido' not in p]
            if non_fixable:
                seen_sug = set()
                for prob in non_fixable:
                    sug_text = prob.get("sugestao", "")
                    if sug_text in seen_sug:
                        continue
                    seen_sug.add(sug_text)
                    st.markdown(
                        f'<div style="background:#EFF6FF;border:1px solid #93C5FD;'
                        f'border-radius:8px;padding:12px;margin:8px 0;">'
                        f'<span style="color:#1D4ED8;font-size:12px;font-weight:600;">'
                        f'Suggestion</span>'
                        f'<p style="color:#1E40AF;font-size:13px;margin:6px 0 0 0;">'
                        f'{sug_text}</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        with col_right:
            # --- Score grande estilo QuillBot (navy escuro) ---
            score_html = f'''
            <div style="text-align:center;padding:20px 0 10px 0;">
                <span style="font-size:72px;font-weight:700;color:#4A90C4;line-height:1;">
                    {score_atual:.0f}</span>
                <span style="font-size:24px;color:#4A90C4;vertical-align:super;">%</span>
                <p style="color:#999;font-size:15px;margin-top:4px;">
                    of text is likely <b style="color:#4A90C4;">{_label_en(score_atual)}</b>
                </p>
            </div>
            '''
            st.markdown(score_html, unsafe_allow_html=True)

            # --- Barra vertical AI vs Human (estilo QuillBot) ---
            ai_bar_pct = score_atual
            human_bar_pct = 100 - score_atual
            bar_html = f'''
            <div style="display:flex;justify-content:center;margin:0 0 10px 0;">
                <div style="display:flex;flex-direction:column;align-items:center;width:120px;">
                    <div style="width:80px;height:100px;background:#E5E7EB;border-radius:4px;
                                position:relative;overflow:hidden;">
                        <div style="position:absolute;bottom:0;width:100%;height:{ai_bar_pct}%;
                                    background:#D4A843;"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;width:120px;
                                margin-top:6px;">
                        <span style="color:#999;font-size:12px;">AI</span>
                        <span style="color:#999;font-size:12px;">Human</span>
                    </div>
                </div>
            </div>
            '''
            st.markdown(bar_html, unsafe_allow_html=True)

            # --- Breakdown (baseado em analise por sentenca) ---
            sentencas_res = st.session_state.get('det_sentencas', [])
            if sentencas_res:
                n_ai = sum(1 for s in sentencas_res if s['classificacao'] == 'ai')
                n_mixed = sum(1 for s in sentencas_res if s['classificacao'] == 'mixed')
                n_human = sum(1 for s in sentencas_res if s['classificacao'] == 'human')
                total_s = len(sentencas_res)
                ai_pct = round(n_ai / total_s * 100) if total_s else 0
                mixed_pct = round(n_mixed / total_s * 100) if total_s else 0
                human_pct = 100 - ai_pct - mixed_pct
            else:
                ai_pct = max(0, score_atual - 20) if score_atual >= 65 else 0
                mixed_pct = min(score_atual, 100) if 25 <= score_atual < 65 else (
                    20 if score_atual >= 65 else 0)
                human_pct = 100 - ai_pct - mixed_pct

            breakdown_html = f'''
            <div style="margin:10px 0 16px 0;">
                <div style="display:flex;align-items:center;margin:8px 0;gap:8px;">
                    <span style="color:#CCC;font-size:14px;flex:1;">AI-generated</span>
                    <span style="color:#FFF;font-size:14px;font-weight:700;">{ai_pct:.0f}%</span>
                    <div style="width:12px;height:12px;border-radius:50%;
                                background:#D4A843;flex-shrink:0;"></div>
                </div>
                <div style="display:flex;align-items:center;margin:8px 0;gap:8px;">
                    <span style="color:#CCC;font-size:14px;flex:1;">Human-written &amp; AI-refined</span>
                    <span style="color:#FFF;font-size:14px;font-weight:700;">{mixed_pct:.0f}%</span>
                    <div style="width:12px;height:12px;border-radius:50%;
                                background:#B0B8C4;flex-shrink:0;"></div>
                </div>
                <div style="display:flex;align-items:center;margin:8px 0;gap:8px;">
                    <span style="color:#CCC;font-size:14px;flex:1;">Human-written</span>
                    <span style="color:#FFF;font-size:14px;font-weight:700;">{human_pct:.0f}%</span>
                    <div style="width:12px;height:12px;border-radius:50%;
                                background:#D5D5D5;flex-shrink:0;"></div>
                </div>
            </div>
            '''
            st.markdown(breakdown_html, unsafe_allow_html=True)

            # --- Main AI Contributors (estilo QuillBot) ---
            st.markdown("---")
            st.markdown("**Main AI Contributors**")

            prob_label = "High Probability" if score_atual >= 65 else (
                "Medium Probability" if score_atual >= 45 else "Low Probability")
            prob_color = "#D4A843" if score_atual >= 65 else (
                "#EAB308" if score_atual >= 45 else "#22C55E")

            st.markdown(
                f'<div style="margin:8px 0 12px 0;">'
                f'<p style="color:#999;font-size:13px;margin:0 0 4px 0;">'
                f'How sure is our detector?</p>'
                f'<span style="color:{prob_color};font-size:15px;font-weight:600;">'
                f'{prob_label}</span></div>',
                unsafe_allow_html=True,
            )

            triggers = []
            if metricas.get('contracoes', 1) < 0.3:
                triggers.append("Zero contractions")
            if metricas.get('burstiness', 1) < 0.25:
                triggers.append("Uniform sentence lengths")
            if metricas.get('sent_sd', 10) < 5.0:
                triggers.append("Low sentence variation")
            if metricas.get('var_adj', 1) < 0.35:
                triggers.append("Similar sentence sizes")
            if metricas.get('conectivos', 0) > 0.3:
                triggers.append("AI connectives")
            if metricas.get('frases_ia', 0) > 0.2:
                triggers.append("Typical AI phrases")
            if metricas.get('pronomes', 5) < 1.5:
                triggers.append("No personal pronouns")
            if metricas.get('ratio_cf', 1) > 1.3:
                triggers.append("High content/function ratio")
            if metricas.get('entropia', 8) < 6.5:
                triggers.append("Low entropy")
            if metricas.get('formulaicos', 0) > 0.2:
                triggers.append("Formulaic patterns")

            if triggers:
                # "Why it triggers AI detection?" + tags/pills
                st.markdown(
                    '<p style="color:#999;font-size:13px;margin:8px 0 6px 0;">'
                    'Why it triggers AI detection?</p>',
                    unsafe_allow_html=True,
                )
                pills_html = '<div style="display:flex;flex-wrap:wrap;gap:6px;">'
                for t in triggers:
                    pills_html += (
                        f'<span style="background:#2A2A3E;border:1px solid #444;'
                        f'color:#CCC;padding:4px 12px;border-radius:16px;'
                        f'font-size:12px;white-space:nowrap;">{t}</span>'
                    )
                pills_html += '</div>'
                st.markdown(pills_html, unsafe_allow_html=True)
            else:
                st.markdown(
                    "<p style='color:#9CA3AF;font-size:13px;background:#1F2937;"
                    "padding:8px 12px;border-radius:6px;'>"
                    "Hard to detect / no clear markers</p>",
                    unsafe_allow_html=True,
                )

            # --- Per-sentence breakdown (estilo QuillBot) ---
            sentencas_res = st.session_state.get('det_sentencas', [])
            if sentencas_res:
                st.markdown("---")
                st.markdown("**Sentence Analysis**")
                st.caption(f"{len(sentencas_res)} sentences analyzed individually")

                rewrites = st.session_state.get('det_sent_rewrites', {})

                for idx_s, s_res in enumerate(sentencas_res):
                    s_score = s_res['score']
                    s_class = s_res['classificacao']
                    s_conf = s_res['confianca']
                    s_text = s_res['sentenca']
                    s_triggers = s_res['triggers']

                    if s_class == 'ai':
                        s_color = '#D4A843'
                        s_label = 'AI-generated'
                        s_badge = 'AI'
                        s_border = '#D4A843'
                        conf_label = 'High Probability' if s_conf == 'high' else 'Moderate'
                        conf_color = '#D4A843'
                    elif s_class == 'mixed':
                        s_color = '#60A5FA'
                        s_label = 'AI-refined'
                        s_badge = 'Mixed'
                        s_border = '#93C5FD'
                        conf_label = 'Moderate'
                        conf_color = '#60A5FA'
                    else:
                        s_color = '#22C55E'
                        s_label = 'Human-written'
                        s_badge = 'Human'
                        s_border = '#22C55E'
                        conf_label = ''
                        conf_color = '#22C55E'

                    preview = s_text[:60] + ('...' if len(s_text) > 60 else '')

                    # Header: score + preview + badge
                    st.markdown(
                        f'<div style="display:flex;align-items:flex-start;gap:8px;'
                        f'margin:6px 0 0 0;padding:8px 10px;'
                        f'border-left:3px solid {s_border};'
                        f'background:rgba(255,255,255,0.03);border-radius:0 6px 6px 0;">'
                        f'<span style="font-size:12px;color:{s_color};font-weight:700;'
                        f'min-width:32px;">{s_score}%</span>'
                        f'<span style="font-size:12px;color:#999;flex:1;'
                        f'line-height:1.4;">{_html_escape_app(preview)}</span>'
                        f'<span style="background:{s_color};color:white;'
                        f'padding:1px 8px;border-radius:10px;font-size:10px;'
                        f'font-weight:600;white-space:nowrap;">{s_badge}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Details for AI/mixed sentences
                    if s_class in ('ai', 'mixed'):
                        # Confidence + trigger tags
                        trigger_tags = ''.join(
                            f'<span style="background:#2A2A3E;border:1px solid #444;'
                            f'color:#CCC;padding:2px 6px;border-radius:10px;'
                            f'font-size:10px;">{_html_escape_app(t[1][:25])}</span>'
                            for t in s_triggers[:3]
                        )
                        st.markdown(
                            f'<div style="margin:2px 0 4px 42px;display:flex;'
                            f'flex-wrap:wrap;gap:4px;align-items:center;">'
                            f'<span style="color:{conf_color};font-size:11px;'
                            f'font-weight:600;">{conf_label}</span>'
                            f'{trigger_tags}</div>',
                            unsafe_allow_html=True,
                        )

                        # Rewrite section
                        if idx_s in rewrites:
                            rw_text = rewrites[idx_s]
                            st.markdown(
                                f'<div style="margin:4px 0 4px 12px;padding:8px 10px;'
                                f'background:#2A2A1E;border:1px solid {s_border};'
                                f'border-radius:6px;">'
                                f'<span style="color:#999;font-size:11px;">'
                                f'Rewritten text:</span><br>'
                                f'<span style="color:#E0E0E0;font-size:12px;'
                                f'line-height:1.5;">'
                                f'{_html_escape_app(rw_text)}</span></div>',
                                unsafe_allow_html=True,
                            )
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.button("🔄 Try", key=f"retry_s_{idx_s}",
                                          on_click=_try_again_sentenca,
                                          args=(idx_s,),
                                          use_container_width=True)
                            with c2:
                                st.button("✕ Cancel", key=f"cancel_s_{idx_s}",
                                          on_click=_cancel_sentenca,
                                          args=(idx_s,),
                                          use_container_width=True)
                            with c3:
                                st.button("✓ Replace", key=f"replace_s_{idx_s}",
                                          type="primary",
                                          on_click=_replace_sentenca,
                                          args=(idx_s,),
                                          use_container_width=True)
                        elif GEMINI_CONFIGURADO:
                            st.button(
                                f"✏️ Rewrite", key=f"rewrite_s_{idx_s}",
                                on_click=_reescrever_sentenca,
                                args=(idx_s,),
                                use_container_width=True,
                            )

        # --- Metricas colapsaveis (full width) ---
        with st.expander("Metricas detalhadas"):
            mostrar_metricas(score_atual, st.session_state.det_metricas)

        # --- Watermarks Unicode / Caracteres Especiais ---
        wm = st.session_state.get("det_watermarks")
        if wm:
            if wm['tem_watermark']:
                st.divider()
                st.markdown("### 🚨 Assinatura Unicode Detectada")
                st.error(wm['resumo'])

                html_wm = gerar_texto_watermark_destacado(texto_detector)
                st.markdown(
                    f'<div style="background:#1A1A2E;padding:16px;border-radius:8px;'
                    f'font-size:14px;line-height:1.8;color:#E0E0E0;border:1px solid #EF4444;">'
                    f'{html_wm}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "🔴 Vermelho = conhecidos | "
                    "🟠 Laranja = desconhecidos (scanner generico) | "
                    "🔵 Azul = tipograficos"
                )

                with st.expander("Detalhes dos caracteres encontrados", expanded=True):
                    for w in wm['invisiveis']:
                        st.markdown(
                            f"**🔴 {w['sigla']}** ({w['unicode']}) — {w['nome']}  \n"
                            f"Encontrado **{w['contagem']}x** nas posicoes: "
                            f"{w['posicoes'][:10]}{'...' if w['contagem'] > 10 else ''}"
                        )
                    for w in wm.get('desconhecidos', []):
                        st.markdown(
                            f"**🟠 {w['sigla']}** ({w['unicode']}) — {w['nome']}  \n"
                            f"Encontrado **{w['contagem']}x** nas posicoes: "
                            f"{w['posicoes'][:10]}{'...' if w['contagem'] > 10 else ''}  \n"
                            f"⚠️ *Caractere nao catalogado — detectado pelo scanner generico*"
                        )
                    for w in wm['tipograficos']:
                        st.markdown(
                            f"**🔵 {w['sigla']}** ({w['unicode']}) — `{w['char']}` — "
                            f"{w['nome']}  \n"
                            f"Encontrado **{w['contagem']}x**"
                        )

                st.text_area("Texto limpo (sem watermarks)", wm['texto_limpo'],
                             height=150, key="texto_limpo_wm")
                st.caption("Texto com todos os caracteres invisiveis removidos")

            elif wm['total_tipograficos'] > 0:
                with st.expander("🔵 Caracteres tipograficos detectados"):
                    st.info(wm['resumo'])
                    for w in wm['tipograficos']:
                        st.markdown(
                            f"**🔵 {w['sigla']}** ({w['unicode']}) — `{w['char']}` — "
                            f"{w['nome']} — **{w['contagem']}x**"
                        )
            else:
                st.caption("✅ Nenhum caractere Unicode suspeito detectado")


# --- TAB 3: HUMANIZADOR ---
with tab3:
    st.header("Humanizador de Texto/Codigo IA")
    st.caption("Reescreve texto ou codigo com Gemini + loop adversarial com detector como feedback")

    texto_humanizar = st.text_area(
        "Texto ou codigo para humanizar",
        value=st.session_state.get("texto_humanizar", ""),
        height=200,
        placeholder="Cole o texto ou codigo de IA que deseja humanizar...",
    )

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        max_iter = st.slider("Max iteracoes", 1, 10, 5)
    with col_h2:
        score_alvo = st.slider("Score alvo (%)", 5, 60, 35, 5)

    if st.button("Humanizar", type="primary", disabled=not GEMINI_CONFIGURADO):
        if not texto_humanizar or len(texto_humanizar.split()) < 10:
            st.warning("Texto muito curto. Minimo 10 palavras.")
        else:
            # Score original
            score_antes, metricas_antes = avaliar(texto_humanizar)

            st.subheader("Texto Original")
            st.caption(f"Score: {score_antes:.1f}% — {rotulo_nivel(score_antes)}")

            texto_atual = texto_humanizar
            melhor_texto = texto_humanizar
            melhor_score = score_antes

            progress = st.progress(0)
            status = st.empty()

            for it in range(max_iter):
                nivel = ["suave", "moderada", "forte", "agressiva", "maxima"][min(it, 4)]
                status.info(f"Iteracao {it + 1}/{max_iter} — reescrita {nivel}...")
                progress.progress((it + 1) / max_iter)

                texto_atual = humanizar_texto(texto_atual, it)
                score, metricas = avaliar(texto_atual)

                if score < melhor_score:
                    melhor_score = score
                    melhor_texto = texto_atual
                    metricas_depois = metricas

                if score <= score_alvo:
                    status.success(f"Iteracao {it + 1}: Score {score:.1f}% <= alvo {score_alvo}%!")
                    break

            progress.progress(1.0)

            # Resultado
            st.divider()
            col_antes, col_depois = st.columns(2)

            with col_antes:
                st.subheader("Antes")
                mostrar_metricas(score_antes, metricas_antes)
                st.text_area("Texto original", texto_humanizar, height=200,
                             key="txt_antes", disabled=True)

            with col_depois:
                st.subheader("Depois")
                if melhor_score < score_antes:
                    mostrar_metricas(melhor_score, metricas_depois)
                else:
                    mostrar_metricas(melhor_score, metricas_antes)
                st.text_area("Texto humanizado", melhor_texto, height=200,
                             key="txt_depois")

            # Delta
            reducao = score_antes - melhor_score
            pct = (reducao / score_antes * 100) if score_antes > 0 else 0

            if reducao > 0:
                st.success(f"Reducao: **{reducao:.1f} pontos ({pct:.0f}%)**")
            else:
                st.warning("O humanizador nao conseguiu reduzir o score.")

            st.session_state["texto_humanizado"] = melhor_texto
