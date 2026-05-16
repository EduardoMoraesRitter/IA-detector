from detector import analisar_problemas, avaliar, rotulo_nivel

code = '''# I added one extra creative sentence at the end of the story using a place and a food.

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
print(adjective + " " + animal + " " + verb1 + " down the hallway.")'''

score, _ = avaliar(code)
print("Score: %.1f%% - %s\n" % (score, rotulo_nivel(score)))

probs = analisar_problemas(code)
print("%d problemas encontrados:\n" % len(probs))
for i, p in enumerate(probs):
    print("%d. [%s] %s" % (i+1, p['tipo'], p['trecho'][:70]))
    print("   Motivo: %s" % p['motivo'])
    print("   Sugestao: %s" % p['sugestao'])
    print()

# Also test with AI prose
ai_text = """Artificial Intelligence has fundamentally transformed the landscape of modern education. Moreover, the integration of AI-powered tools into educational frameworks represents a comprehensive paradigm shift. Furthermore, these sophisticated systems facilitate personalized learning experiences."""

score2, _ = avaliar(ai_text)
print("=" * 50)
print("TEXTO IA: %.1f%% - %s\n" % (score2, rotulo_nivel(score2)))

probs2 = analisar_problemas(ai_text)
print("%d problemas encontrados:\n" % len(probs2))
for i, p in enumerate(probs2):
    print("%d. [%s] %s" % (i+1, p['tipo'], p['trecho'][:70]))
    print("   Motivo: %s" % p['motivo'])
    print("   Sugestao: %s" % p['sugestao'])
    print()
