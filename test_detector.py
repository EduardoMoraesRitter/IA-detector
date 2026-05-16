from detector import avaliar, rotulo_nivel

ai_text = """Artificial Intelligence has fundamentally transformed the landscape of modern education. Moreover, the integration of AI-powered tools into educational frameworks represents a comprehensive paradigm shift. Furthermore, these sophisticated systems facilitate personalized learning experiences that were previously unattainable. It is important to note that AI leverages vast datasets to identify patterns in student performance, thereby enabling educators to tailor their pedagogical approaches accordingly. Additionally, the utilization of machine learning algorithms plays a crucial role in automating assessment processes. In today's rapidly evolving technological landscape, educational institutions must embrace these innovations to remain competitive. The multifaceted nature of AI applications in education encompasses everything from intelligent tutoring systems to automated content generation. Consequently, this holistic approach to educational technology has the potential to revolutionize how knowledge is disseminated and absorbed across diverse learning environments."""

human_text = """So I tried using ChatGPT for my homework last week and honestly? It was kind of a mess. Like, sure, it gave me a decent outline for my essay about climate change. But the actual writing was so generic - you could tell a robot wrote it from a mile away. My teacher wasn't fooled for a second. She pulled me aside and said look, I can tell this isn't yours. The sentences all sound the same, there's no personality in it. And she was right! I ended up rewriting the whole thing myself. Took me way longer but at least it sounded like ME, you know? I think AI is useful for brainstorming but don't let it write your stuff. You'll get caught and honestly the writing is just... bland."""

ai2_text = """The rapid advancement of technology has significantly impacted various sectors of society. One area that has seen substantial growth is the field of renewable energy. Solar and wind power have emerged as viable alternatives to traditional fossil fuels, offering a more sustainable approach to energy production. The implementation of these technologies has been facilitated by decreasing costs and improving efficiency. Governments around the world have recognized the importance of transitioning to clean energy sources and have implemented various policies to support this transition. The benefits of renewable energy extend beyond environmental considerations, encompassing economic growth and job creation. As research continues to advance, it is expected that renewable energy will play an increasingly important role in meeting global energy demands."""

score_ai, m_ai = avaliar(ai_text)
score_hum, m_hum = avaliar(human_text)
score_ai2, m_ai2 = avaliar(ai2_text)

print("=== TEXTO IA (blatant) ===")
print("Score: %.1f%% - %s" % (score_ai, rotulo_nivel(score_ai)))
print("  Conectivos: %.3f  Contracoes: %.2f  Pronomes: %.2f%%" % (m_ai['conectivos'], m_ai['contracoes'], m_ai['pronomes']))
print("  Ratio CF: %.3f  Sent SD: %.2f  Frases IA: %.2f" % (m_ai['ratio_cf'], m_ai['sent_sd'], m_ai['frases_ia']))
print()

print("=== TEXTO IA (subtle, sem buzzwords) ===")
print("Score: %.1f%% - %s" % (score_ai2, rotulo_nivel(score_ai2)))
print("  Conectivos: %.3f  Contracoes: %.2f  Pronomes: %.2f%%" % (m_ai2['conectivos'], m_ai2['contracoes'], m_ai2['pronomes']))
print("  Ratio CF: %.3f  Sent SD: %.2f  Frases IA: %.2f" % (m_ai2['ratio_cf'], m_ai2['sent_sd'], m_ai2['frases_ia']))
print()

print("=== TEXTO HUMANO ===")
print("Score: %.1f%% - %s" % (score_hum, rotulo_nivel(score_hum)))
print("  Conectivos: %.3f  Contracoes: %.2f  Pronomes: %.2f%%" % (m_hum['conectivos'], m_hum['contracoes'], m_hum['pronomes']))
print("  Ratio CF: %.3f  Sent SD: %.2f  Frases IA: %.2f" % (m_hum['ratio_cf'], m_hum['sent_sd'], m_hum['frases_ia']))
