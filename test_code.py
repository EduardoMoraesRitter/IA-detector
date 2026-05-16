from detector import avaliar, rotulo_nivel

# 1. AI-generated code (user's test case)
code_ai = '''# I added extra words for a place and a food, and I also made the program choose "a" or "an" automatically before the animal.

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

first_letter = animal[0].lower()
if first_letter in "aeiou":
    article = "an"
else:
    article = "a"

print()
print("Your story is:")
print()

print(f"""The other day, I was really in trouble. It all started when I saw a very
{adjective} {animal} {verb1} down the hallway. "{exclamation}!" I yelled. But all
I could think to do was to {verb2} over and over. Miraculously,
that caused it to stop, but not before it tried to {verb3}
right in front of my family.
Later that day, I found {article} {animal} at the {place} eating {food}.""")'''

# 2. Human-written code (minimal comments, shortcuts, messy)
code_human = '''import sys

def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a+b
    return a

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
for i in range(n):
    print(fib(i), end=" ")
print()'''

# 3. AI prose
ai_prose = """Artificial Intelligence has fundamentally transformed the landscape of modern education. Moreover, the integration of AI-powered tools into educational frameworks represents a comprehensive paradigm shift. Furthermore, these sophisticated systems facilitate personalized learning experiences that were previously unattainable."""

# 4. Human text
human_text = """So I tried using ChatGPT for my homework last week and honestly? It was kind of a mess. Like, sure, it gave me a decent outline but the writing was so generic you could tell a robot wrote it."""

tests = [
    ("CODIGO IA (user test)", code_ai),
    ("CODIGO HUMANO (fibonacci)", code_human),
    ("TEXTO IA (prose)", ai_prose),
    ("TEXTO HUMANO (blog)", human_text),
]

for name, text in tests:
    score, m = avaliar(text)
    is_code = m.get('is_code', False)
    mode = "[CODE]" if is_code else "[TEXT]"
    print("%-35s %s  Score: %5.1f%% - %s" % (name, mode, score, rotulo_nivel(score)))
