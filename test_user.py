from detector import avaliar, rotulo_nivel, detectar_codigo, extrair_nl_de_codigo, calcular_metricas_codigo

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

print("=== DIAGNOSTICO ===")
print("detectar_codigo:", detectar_codigo(code))
nl = extrair_nl_de_codigo(code)
print("NL extraido:", repr(nl[:200]))
print("NL words:", len(nl.split()) if nl else 0)
cm = calcular_metricas_codigo(code)
print("code_metricas:", cm)
print()

score, m = avaliar(code)
print("Score: %.1f%% - %s" % (score, rotulo_nivel(score)))
for k, v in m.items():
    if isinstance(v, float):
        print("  %s: %.3f" % (k, v))
    else:
        print("  %s: %s" % (k, v))
