import os
from dotenv import load_dotenv
import google.generativeai as genai
from detector import avaliar, rotulo_nivel

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

code_original = '''# I added one extra creative sentence at the end of the story using a place and a food.

print("Please enter the following:")
print()

adjective = input("adjective: ")
animal = input("animal: ")
verb1 = input("verb: ")
exclamation = input("exclamation: ")
verb2 = input("verb: ")
verb3 = input("verb: ")
place = input("place: ")
food = input("food: ")

exclamation = exclamation.capitalize()

print()
print("Your story is:")
print()

print(f"""The other day, I was really in trouble. It all started when I saw a very
{adjective} {animal} {verb1} down the hallway. "{exclamation}!" I yelled. But all
I could think to do was to {verb2} over and over. Miraculously,
that caused it to stop, but not before it tried to {verb3}
right in front of my family.
After that, we all went to the {place} and ate {food} like nothing had happened.""")'''

ai_text = """Artificial Intelligence has fundamentally transformed the landscape of modern education. Moreover, the integration of AI-powered tools into educational frameworks represents a comprehensive paradigm shift. Furthermore, these sophisticated systems facilitate personalized learning experiences that were previously unattainable. It is important to note that AI leverages vast datasets to identify patterns in student performance, thereby enabling educators to tailor their pedagogical approaches accordingly."""

prompts_code = [
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

""",
    """The code below is STILL detected as AI. Rewrite MORE aggressively.

MANDATORY:
1. ZERO comments
2. ALL variable names 1-3 characters (a, b, v1, e, p, f)
3. RESTRUCTURE: use a dict or loop for repeated input() calls. Keep the ORIGINAL prompt strings inside input().
4. Mix styles: some concatenation + some f-strings
5. Add a quirk: weird temp variable, unnecessary list, etc.
6. DO NOT change any string that the user sees (print text, input prompts). Only change code structure and variable names.

Output ONLY code. No text. No markdown. No ``` fences.

""",
    """MAXIMUM rewrite. Make it look like a lazy student typed this fast.

Rules:
1. ZERO comments
2. Single-letter variables
3. Completely different structure: dicts, loops, list comprehensions
4. Inconsistent style on purpose
5. Maybe combine prints with \\n
6. NEVER change the text inside print() or input() - same user-facing output
7. Same behavior

Output ONLY code. Nothing else. No ``` fences.

""",
]

prompts_text = [
    """Rewrite the following text to sound more naturally human-written.
Apply these techniques:
- Vary sentence lengths (mix short punchy sentences with longer ones)
- Replace formal connectors (moreover, furthermore, additionally) with casual transitions
- Add occasional hedging language (maybe, probably, I think, arguably)
- Use contractions where natural
- Avoid starting consecutive sentences the same way
CRITICAL: Keep the same meaning. Only change HOW it reads.

Text:
""",
    """The following text still reads like AI. Rewrite it MORE aggressively:
- Use VERY varied sentence lengths: some 3 words, others 30+
- Replace ALL academic phrases with conversational equivalents
- Add a personal touch or opinion
- Use rhetorical questions or self-corrections
- Avoid: moreover, furthermore, crucial, vital, comprehensive, multifaceted, leverage, utilize, facilitate, holistic, robust, landscape, paradigm
Keep the same core information.

Text:
""",
    """This text is STILL detected as AI. Maximum transformation:
- Write like a real person - imperfect, natural, with personality
- Some paragraphs just ONE sentence, others long and rambling
- Use slang, idioms, informal expressions
- Start sentences with "And", "But", "So", "Look,"
- Add parenthetical asides (like this) or dashes
- Remove ANY "essay structure" - no "in conclusion", no "firstly/secondly"
- Sound opinionated, not neutral
Same information, completely different voice.

Text:
""",
]

model = genai.GenerativeModel("gemini-2.0-flash")

def clean_response(text, is_code):
    if is_code and text.startswith("```"):
        lines = text.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = '\n'.join(lines)
    return text

def run_test(name, original, prompts, is_code):
    print("\n" + "=" * 60)
    print("TESTE: %s" % name)
    print("=" * 60)

    score_antes, m_antes = avaliar(original)
    print("ORIGINAL: %.1f%% - %s" % (score_antes, rotulo_nivel(score_antes)))
    if is_code:
        print("  pedagogical_hits: %s" % m_antes.get('code_pedagogical', 'N/A'))
    print()

    texto_atual = original
    melhor_texto = original
    melhor_score = score_antes

    for i, prompt_template in enumerate(prompts):
        prompt = prompt_template + texto_atual
        temp = 1.0 + (i * 0.2) if is_code else 0.9 + (i * 0.15)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=min(temp, 1.8), max_output_tokens=2000),
        )
        result = clean_response(response.text, is_code)
        texto_atual = result
        score, m = avaliar(result)

        if score < melhor_score:
            melhor_score = score
            melhor_texto = result

        print("--- Iteracao %d (temp=%.1f) ---" % (i+1, temp))
        print(result[:400])
        if len(result) > 400:
            print("...")
        print()
        ph = m.get('code_pedagogical', '-')
        print("Score: %.1f%% - %s | pedagogical: %s" % (score, rotulo_nivel(score), ph))
        print()

        if score <= 35:
            print(">>> ALVO ATINGIDO! <<<")
            break

    print("-" * 40)
    print("RESULTADO: %.1f%% -> %.1f%% (reducao: %.1f pts)" % (score_antes, melhor_score, score_antes - melhor_score))
    ok = "OK" if melhor_score <= 40 else "PRECISA MELHORAR"
    print("Status: %s" % ok)
    print()
    return melhor_score

# Run both tests
s1 = run_test("CODIGO IA (mad libs)", code_original, prompts_code, True)
s2 = run_test("TEXTO IA (education)", ai_text, prompts_text, False)

print("\n" + "=" * 60)
print("RESUMO FINAL")
print("=" * 60)
print("Codigo: %.1f%% - %s" % (s1, "PASSOU" if s1 <= 40 else "FALHOU"))
print("Texto:  %.1f%% - %s" % (s2, "PASSOU" if s2 <= 40 else "FALHOU"))
