import os
from dotenv import load_dotenv
import google.generativeai as genai
from detector import avaliar, rotulo_nivel, detectar_codigo

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

code_original = '''# I added a little extra sentence at the end with a place and a food to make the story longer.

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

print("The other day, I was really in trouble. It all started when I saw a very")
print(adjective + " " + animal + " " + verb1 + " down the hallway. " + '"' + exclamation + '!" I yelled. But all')
print("I could think to do was to " + verb2 + " over and over. Miraculously,")
print("that caused it to stop, but not before it tried to " + verb3)
print("right in front of my family.")
print("After that, we all went to the " + place + " and ate " + food + ".")'''

score_antes, _ = avaliar(code_original)
print("ORIGINAL: %.1f%% - %s\n" % (score_antes, rotulo_nivel(score_antes)))

model = genai.GenerativeModel("gemini-2.0-flash")
prompts = [
    """You are a code rewriter. Your job is to take AI-generated code and rewrite it so it looks like a REAL STUDENT typed it.

MANDATORY changes (do ALL of them):
1. DELETE every comment. Every single one. Students don't write comments like "# I added..." or "# This function...". Delete them ALL.
2. RENAME all variables to short/lazy names. Examples:
   - adjective -> adj
   - exclamation -> exc
   - verb1 -> v1
   - animal -> ani
   - place -> p
   - food -> f
   IMPORTANT: variable names must NOT match the input() prompt text.
3. If there are 3+ sequential input() calls, RESTRUCTURE them. Use a list or dict approach. Example:
   INSTEAD OF:
   adjective = input("adjective: ")
   animal = input("animal: ")
   verb1 = input("verb: ")

   DO THIS:
   prompts = ["adjective", "animal", "verb"]
   answers = [input(p + ": ") for p in prompts]
   adj, ani, v1 = answers

   OR just rename the variables so they don't match the prompts.
4. Add one small imperfection (extra blank line, slightly inconsistent spacing, a temp variable).
5. The program must produce the EXACT same output.

Output ONLY the rewritten code. No explanations. No markdown. No ``` fences.

""",
    """The code below is STILL detected as AI. Rewrite it MORE aggressively.

MANDATORY:
1. ZERO comments anywhere
2. ALL variable names must be 1-3 characters (a, b, v1, exc, p, f)
3. RESTRUCTURE repetitive patterns. If you see many input() calls, use a loop:
   words = {}
   for w in ["adjective","animal","verb","exclamation","verb","verb","place","food"]:
       words[w] = input(w + ": ")
4. Mix coding styles: some string concatenation + some f-strings
5. Add a quirk: maybe store something in a list first, or use a weird temp variable
6. Program must work identically

Output ONLY code. No text. No markdown.

""",
    """MAXIMUM rewrite. This code must look like a lazy student typed it fast.

Rules:
1. ZERO comments
2. Single-letter variables where possible
3. Use a completely different structure: dicts, lists, loops - anything but sequential repetitive calls
4. Inconsistent style on purpose
5. Maybe combine multiple prints into one with \\n
6. Same output behavior

Output ONLY code. Nothing else.

""",
]

texto_atual = code_original
melhor_texto = code_original
melhor_score = score_antes

for i, prompt_template in enumerate(prompts):
    prompt = prompt_template + texto_atual
    temp = 1.0 + (i * 0.2)
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=min(temp, 1.8), max_output_tokens=1500),
    )
    result = response.text
    if result.startswith("```"):
        lines = result.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        result = '\n'.join(lines)

    texto_atual = result
    score, m = avaliar(result)

    if score < melhor_score:
        melhor_score = score
        melhor_texto = result

    print("--- ITERACAO %d (temp=%.1f) ---" % (i+1, temp))
    print(result[:300])
    print("...")
    print("Score: %.1f%% - %s" % (score, rotulo_nivel(score)))
    print("pedagogical_hits:", m.get('code_pedagogical', 'N/A'))
    print()

    if score <= 40:
        print("Score alvo atingido!")
        break

print("=== RESULTADO FINAL ===")
print("Antes: %.1f%% -> Depois: %.1f%%" % (score_antes, melhor_score))
print("Reducao: %.1f pontos" % (score_antes - melhor_score))
print()
print(melhor_texto)
