"""
Gerador de Texto com Gemini API
Gera texto sobre qualquer tema usando Gemini 2.0 Flash.
"""

import argparse
import os
import sys
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


def configurar_api(api_key=None):
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        print("[!] GEMINI_API_KEY nao encontrada.")
        print("    Configure via: export GEMINI_API_KEY=sua_chave")
        print("    Ou crie um arquivo .env com: GEMINI_API_KEY=sua_chave")
        print("    Pegue sua chave em: https://aistudio.google.com/apikey")
        sys.exit(1)
    genai.configure(api_key=key)


def gerar_texto(tema, palavras=250, temperature=0.7, modelo="gemini-2.0-flash"):
    model = genai.GenerativeModel(modelo)
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


def main():
    parser = argparse.ArgumentParser(description="Gerador de Texto com Gemini API")
    parser.add_argument("--tema", "-t", type=str, help="Tema para gerar texto")
    parser.add_argument("--palavras", "-n", type=int, default=250, help="Numero aprox. de palavras (default: 250)")
    parser.add_argument("--temperature", "-temp", type=float, default=0.7, help="Temperature (default: 0.7)")
    parser.add_argument("--salvar", "-s", type=str, help="Salvar em arquivo")
    parser.add_argument("--api-key", type=str, help="Gemini API key (ou use GEMINI_API_KEY env var)")
    args = parser.parse_args()

    tema = args.tema or input("Digite o tema: ")

    configurar_api(args.api_key)

    print(f"\n{'='*60}")
    print(f"  Tema: {tema}")
    print(f"  Palavras: ~{args.palavras} | Temp: {args.temperature}")
    print(f"{'='*60}\n")

    texto = gerar_texto(tema, args.palavras, args.temperature)

    print(texto)
    print(f"\n{'='*60}")
    print(f"  {len(texto.split())} palavras geradas")
    print(f"{'='*60}")

    if args.salvar:
        with open(args.salvar, "w", encoding="utf-8") as f:
            f.write(texto)
        print(f"\n[+] Texto salvo em: {args.salvar}")


if __name__ == "__main__":
    main()
