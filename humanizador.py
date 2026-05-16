"""
Humanizador de Texto IA — Gemini + Loop Adversarial

Usa Gemini para reescrever texto e o detector estatistico como feedback.
Inspirado em:
- Adversarial-Paraphrasing (NeurIPS 2025): detector como feedback iterativo
- SICO (TMLR 2024): otimizacao de prompts para evadir detectores
- TempParaphraser (EMNLP 2025): reescrita com agressividade crescente
"""

import argparse
import os
import sys
import google.generativeai as genai
from dotenv import load_dotenv
from detector import avaliar, barra, rotulo_nivel

load_dotenv()


def configurar_api(api_key=None):
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        print("[!] GEMINI_API_KEY nao encontrada.")
        print("    Configure via: export GEMINI_API_KEY=sua_chave")
        print("    Ou crie um arquivo .env com: GEMINI_API_KEY=sua_chave")
        sys.exit(1)
    genai.configure(api_key=key)


# Prompts progressivamente mais agressivos (conceito do TempParaphraser)
PROMPTS_ITERACAO = [
    # Iteracao 1: reescrita suave
    """Rewrite the following text to sound more naturally human-written.
Apply these techniques:
- Vary sentence lengths (mix short punchy sentences with longer ones)
- Replace formal connectors (moreover, furthermore, additionally) with casual transitions
- Add occasional hedging language (maybe, probably, I think, arguably)
- Use contractions where natural
- Avoid starting consecutive sentences the same way

CRITICAL: Keep the same meaning and information. Only change HOW it reads.

Text:
{texto}""",

    # Iteracao 2: reescrita moderada
    """The following text still reads like it was written by AI. Rewrite it MORE aggressively:
- Use VERY varied sentence lengths: some as short as 3 words, others 30+
- Replace ALL academic/formal phrases with conversational equivalents
- Add a personal touch or opinion somewhere
- Break up any perfectly structured paragraphs
- Use rhetorical questions or self-corrections ("well, actually...")
- Avoid these words entirely: moreover, furthermore, crucial, vital, comprehensive, multifaceted, leverage, utilize, facilitate, holistic, robust, landscape, paradigm

Keep the same core information.

Text:
{texto}""",

    # Iteracao 3: reescrita agressiva
    """This text is STILL detected as AI-generated. Maximum transformation needed:
- Write like a real person typing quickly — imperfect, natural, with personality
- Some paragraphs should be just ONE sentence. Others should be long and rambling
- Use slang, idioms, or informal expressions where they fit
- Start some sentences with "And", "But", "So", "Look," or "Here's the thing"
- Add parenthetical asides (like this) or dashes — for interrupting yourself
- Remove ANY trace of "essay structure" — no "in conclusion", no "firstly/secondly"
- Sound opinionated, not neutral

Same information, completely different voice.

Text:
{texto}""",
]


def humanizar_texto(texto, modelo="gemini-2.0-flash", iteracao=0):
    """Reescreve texto usando Gemini com prompt da iteracao correspondente."""
    model = genai.GenerativeModel(modelo)
    idx = min(iteracao, len(PROMPTS_ITERACAO) - 1)
    prompt = PROMPTS_ITERACAO[idx].format(texto=texto)

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.9 + (iteracao * 0.15),
            max_output_tokens=len(texto.split()) * 3,
        ),
    )
    return response.text


def humanizar(texto, max_iteracoes=3, score_alvo=40.0, modelo="gemini-2.0-flash"):
    """
    Loop adversarial: reescreve com Gemini, avalia com detector, repete.
    A cada iteracao o prompt fica mais agressivo.
    """
    texto_atual = texto
    melhor_texto = texto
    melhor_score = 100.0

    for it in range(max_iteracoes):
        print(f"\n{'='*55}")
        print(f"  ITERACAO {it + 1}/{max_iteracoes}")
        print(f"{'='*55}")

        nivel = ["suave", "moderada", "agressiva"][min(it, 2)]
        print(f"  [*] Reescrevendo (modo {nivel})...")

        texto_atual = humanizar_texto(texto_atual, modelo, it)

        print(f"  [*] Avaliando com detector...")
        score, metricas = avaliar(texto_atual)

        print(f"\n  Resultado iteracao {it + 1}:")
        print(f"    Score:       {score:.1f}% — {rotulo_nivel(score)}")
        print(f"    Burstiness:  {metricas['burstiness']:.4f}")
        print(f"    Conectivos:  {metricas['conectivos']:.4f}/frase")
        print(f"    Frases IA:   {metricas['frases_ia']:.2f}/100p")

        if score < melhor_score:
            melhor_score = score
            melhor_texto = texto_atual

        if score <= score_alvo:
            print(f"\n  [+] Score {score:.1f}% <= alvo {score_alvo}%. Sucesso!")
            break

    return melhor_texto, melhor_score


def exibir_comparacao(texto_orig, texto_human, score_antes, score_depois,
                      metricas_antes, metricas_depois):
    ma, md = metricas_antes, metricas_depois

    print("\n" + "=" * 65)
    print("                  RESULTADO FINAL")
    print("=" * 65)

    print("\n  --- TEXTO ORIGINAL (IA) ---")
    preview = texto_orig[:400]
    print(f"  {preview}{'...' if len(texto_orig) > 400 else ''}")

    print("\n  --- TEXTO HUMANIZADO ---")
    preview = texto_human[:400]
    print(f"  {preview}{'...' if len(texto_human) > 400 else ''}")

    print(f"\n{'─'*65}")
    print(f"  {'METRICA':<22} {'ANTES':>10} {'DEPOIS':>10} {'DELTA':>10}")
    print(f"{'─'*65}")
    print(f"  {'Burstiness':<22} {ma['burstiness']:>10.4f} {md['burstiness']:>10.4f} {md['burstiness']-ma['burstiness']:>+10.4f}")
    print(f"  {'Conectivos IA/frase':<22} {ma['conectivos']:>10.4f} {md['conectivos']:>10.4f} {md['conectivos']-ma['conectivos']:>+10.4f}")
    print(f"  {'Frases tipicas IA':<22} {ma['frases_ia']:>10.2f} {md['frases_ia']:>10.2f} {md['frases_ia']-ma['frases_ia']:>+10.2f}")
    print(f"  {'Div. inicio frase':<22} {ma['div_inicio']:>10.4f} {md['div_inicio']:>10.4f} {md['div_inicio']-ma['div_inicio']:>+10.4f}")
    print(f"{'─'*65}")

    print(f"\n  SCORE ANTES:   {score_antes:>5.1f}%  {barra(score_antes, 100, 25)}  {rotulo_nivel(score_antes)}")
    print(f"  SCORE DEPOIS:  {score_depois:>5.1f}%  {barra(score_depois, 100, 25)}  {rotulo_nivel(score_depois)}")

    reducao = score_antes - score_depois
    pct = (reducao / score_antes * 100) if score_antes > 0 else 0
    print(f"\n  REDUCAO: {reducao:.1f} pontos ({pct:.0f}%)")
    print("=" * 65)
    print()


def main():
    parser = argparse.ArgumentParser(description="Humanizador de Texto IA — Gemini + Loop Adversarial")
    parser.add_argument("--texto", "-t", type=str, help="Texto para humanizar")
    parser.add_argument("--arquivo", "-a", type=str, help="Arquivo .txt para humanizar")
    parser.add_argument("--salvar", "-s", type=str, help="Salvar resultado em arquivo")
    parser.add_argument("--iteracoes", "-i", type=int, default=3, help="Max iteracoes (default: 3)")
    parser.add_argument("--score-alvo", type=float, default=40.0, help="Score alvo (default: 40.0)")
    parser.add_argument("--api-key", type=str, help="Gemini API key")
    args = parser.parse_args()

    if args.arquivo:
        with open(args.arquivo, "r", encoding="utf-8") as f:
            texto = f.read()
    elif args.texto:
        texto = args.texto
    else:
        print("Cole o texto para humanizar (linha vazia para finalizar):\n")
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

    configurar_api(args.api_key)

    print(f"[*] Texto: {len(texto)} caracteres, {len(texto.split())} palavras\n")

    # Score original
    print("[*] Avaliando texto original...")
    score_antes, metricas_antes = avaliar(texto)
    print(f"[*] Score original: {score_antes:.1f}% — {rotulo_nivel(score_antes)}")

    # Loop adversarial
    texto_humanizado, _ = humanizar(texto, max_iteracoes=args.iteracoes, score_alvo=args.score_alvo)

    # Metricas finais
    score_depois, metricas_depois = avaliar(texto_humanizado)

    # Comparacao
    exibir_comparacao(texto, texto_humanizado, score_antes, score_depois, metricas_antes, metricas_depois)

    if args.salvar:
        with open(args.salvar, "w", encoding="utf-8") as f:
            f.write(texto_humanizado)
        print(f"[+] Texto humanizado salvo em: {args.salvar}")


if __name__ == "__main__":
    main()
