import os
from dotenv import load_dotenv
import google.generativeai as genai
from detector import avaliar, rotulo_nivel, detectar_codigo

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

code = '''# I added a little extra sentence at the end with a place and a food to make the story longer.

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

score_antes, _ = avaliar(code)
print("ANTES: %.1f%% - %s" % (score_antes, rotulo_nivel(score_antes)))
print()

prompt = """You are a code rewriter. Your job is to take AI-generated code and rewrite it so it looks like a REAL STUDENT typed it.

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

""" + code

model = genai.GenerativeModel("gemini-2.0-flash")
response = model.generate_content(
    prompt,
    generation_config=genai.GenerationConfig(temperature=1.0, max_output_tokens=1500),
)

result = response.text
if result.startswith("```"):
    lines = result.split('\n')
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    result = '\n'.join(lines)

print("=== CODIGO HUMANIZADO ===")
print(result)
print()

score_depois, m = avaliar(result)
print("DEPOIS: %.1f%% - %s" % (score_depois, rotulo_nivel(score_depois)))
print("Reducao: %.1f pontos" % (score_antes - score_depois))
print()
print("is_code:", m.get('is_code', False))
print("pedagogical_hits:", m.get('code_pedagogical', 'N/A'))
