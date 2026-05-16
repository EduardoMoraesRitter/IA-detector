# IA Detector

Detector de texto e codigo gerado por IA usando analise estatistica pura — zero modelos de machine learning.

Inspirado em: DetectGPT, GLTR, ZipPy (Thinkst), GPTZero, StyloAI, QuillBot AI Detector.

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```

Acesse `http://localhost:8501`.

Para usar o Gerador e Humanizador, configure a API Key do Gemini:
- No `.env`: `GEMINI_API_KEY=sua_chave_aqui`
- Ou direto na sidebar do app
- Pegue em: https://aistudio.google.com/apikey

## Funcionalidades

### 1. Detector (sem API, roda local)

Analisa texto ou codigo e retorna um score de 0-100% indicando probabilidade de ser gerado por IA.

**Auto-detecta** se a entrada e texto ou codigo e aplica metricas diferentes para cada um.

**Interface estilo QuillBot:**
- Texto com **highlights inline** (amarelo) nas frases detectadas como IA
- **Score visual** com barra de progresso colorida
- **Cards por problema** mostrando: trecho original, sugestao, e **previsao de % de melhoria**
- **Metricas detalhadas** em painel colapsavel
- Botao **"Aceitar"** por sugestao (aplica correcao individual)
- Botao **"Aceitar Todas"** com previsao do score final (ex: "Aceitar Todas -> 59% (-19%)")

### 2. Gerador (Gemini API)

Gera texto sobre qualquer tema usando Gemini 2.0 Flash. Controle de quantidade de palavras e temperature.

### 3. Humanizador (Gemini API + Detector)

Loop adversarial: reescreve o texto com Gemini e usa o proprio detector como feedback. Suporta texto e codigo. Repete ate atingir o score alvo.

---

## Como o Detector funciona

### Abordagem

O detector usa **analise estatistica pura** — nenhum modelo de ML, nenhuma API. Isso e a mesma abordagem usada por ferramentas como GPTZero e Originality.ai:

- **Perplexidade / Entropia** — texto IA e mais previsivel
- **Burstiness** — IA escreve frases de tamanho uniforme
- **Estilometria** — IA evita contracoes, pronomes pessoais, e usa vocabulario formal
- **Compressao (ZipPy)** — texto repetitivo comprime melhor com zlib

### Modo Texto — 15 metricas

| # | Metrica | O que mede | IA tipica | Humano tipico | Peso |
|---|---------|-----------|-----------|---------------|------|
| 1 | **Entropia Shannon** | Diversidade da distribuicao de palavras | 6-8 bits | 8-10+ bits | 0.05 |
| 2 | **Compressao (ZipPy)** | Taxa compressao zlib — texto previsivel comprime mais | 0.25-0.40 | 0.40-0.60+ | 0.06 |
| 3 | **Burstiness** | CV do comprimento das frases | 0.1-0.3 | 0.5-1.0+ | 0.04 |
| 4 | **SD compr. frases** | Desvio padrao absoluto do tamanho das frases | SD 2-5 | SD 6-12+ | 0.09 |
| 5 | **Var. adjacentes** | CV das razoes entre frases consecutivas | CV < 0.3 | CV > 0.5 | 0.04 |
| 6 | **TTR** | Type-Token Ratio (palavras unicas / total) | ~0.45 | ~0.55+ | 0.02 |
| 7 | **Hapax Legomena** | Palavras que aparecem 1 vez / total | baixo | alto | 0.02 |
| 8 | **Ratio conteudo/funcao** | Content words / function words | ~1.37 | ~0.98 | 0.12 |
| 9 | **Contracoes** | Contracoes por 100 palavras (don't, I'm...) | < 0.5 | 2-5+ | 0.10 |
| 10 | **Entropia pontuacao** | Shannon entropy dos intervalos de pontuacao | < 2.5 | > 3.0 | 0.04 |
| 11 | **Conectivos IA** | Frequencia de "moreover", "furthermore", "leverage"... | > 0.5/frase | < 0.2/frase | 0.14 |
| 12 | **Div. inicio frase** | Variedade das primeiras palavras | < 0.6 | > 0.8 | 0.03 |
| 13 | **Pronomes pessoais** | % de pronomes 1a/2a pessoa | < 1.5% | > 3% | 0.08 |
| 14 | **Frases tipicas IA** | Dead giveaways ("plays a crucial role", "delve into"...) | > 0.5/100p | ~0 | 0.12 |
| 15 | **Padroes formulaicos** | Aberturas/fechamentos tipicos ("In today's...", "In conclusion...") | 0.5-1.0 | ~0 | 0.05 |

**Mecanismo de consenso:** quando 5+ sinais fortes concordam, o score minimo e forcado (5 sinais = min 78%, 6+ = min 88%). Evita que metricas fracas diluam sinais claros.

### Modo Codigo — metricas especificas

Quando detecta codigo, o detector:
1. **Extrai linguagem natural** (comentarios, strings, docstrings, print/input)
2. **Analisa o NL extraido** com as 15 metricas de texto
3. **Aplica metricas de codigo:**

| Metrica | O que detecta | Peso |
|---------|--------------|------|
| **Comment ratio** | IA over-comments (ratio > 0.3) | ate 15 |
| **Frases pedagogicas** | "I added...", "This function...", "Here we...", "the user...", "will prompt..." | ate 40 |
| **Comentarios "I [verbo]"** | `# I added`, `# I made`, `# I used` — IA explicando seu trabalho | 2x por match |
| **Input() sequenciais** | 3+ chamadas `input()` seguidas | 1-3 pontos |
| **Var = input("var")** | `adjective = input("adjective: ")` — nome da variavel igual ao prompt | ate 15 |
| **Comentarios de razao** | "to make...", "so that...", "in order to..." | 2x por match |
| **Uniformidade linhas** | CV do comprimento das linhas (IA = uniforme) | ate 10 |
| **Consistencia IDs** | CV do tamanho dos nomes de variaveis | ate 8 |
| **Indentacao perfeita** | Toda indentacao multiplo de 4 | ate 5 |
| **Regularidade blank lines** | Linhas em branco em intervalos regulares | ate 5 |
| **Score NL** | Score do texto natural extraido | ate ~20 |

**Consenso codigo:** ph >= 8 = min 78%, ph >= 12 ou (ph >= 6 + vpm >= 3) = min 82%.

### Analise de Problemas e Sugestoes

O detector identifica **trechos especificos** com cara de IA e sugere correcoes:

| Tipo | Exemplo | Sugestao |
|------|---------|----------|
| **Conectivo IA** | "Moreover" | "Also" |
| **Conectivo IA** | "Furthermore" | "And" |
| **Conectivo IA** | "facilitate" | "help" |
| **Conectivo IA** | "comprehensive" | "full" |
| **Conectivo IA** | "paradigm" | "approach" |
| **Conectivo IA** | "landscape" | "field" |
| **Conectivo IA** | "It is important to note" | "Worth noting" |
| **Sem contracoes** | "do not" -> "don't" | Auto-correcao |
| **Frase IA** | "plays a crucial role" | "matters a lot" |
| **Uniformidade** | Frases com tamanho parecido | "Varie o ritmo" |
| **Sem pronomes** | Texto inteiro impessoal | "Adicione I think..., you can see..." |
| **Abertura formulaica** | "In today's world..." | "Comece de forma direta" |

Cada sugestao mostra **previsao de melhoria no score** quando aplicada.

### Niveis de score

| Score | Rotulo | Cor |
|-------|--------|-----|
| 85-100% | MUITO PROVAVEL IA | vermelho |
| 65-84% | PROVAVEL IA | laranja |
| 45-64% | INCONCLUSIVO | amarelo |
| 25-44% | PROVAVEL HUMANO | verde |
| 0-24% | MUITO PROVAVEL HUMANO | verde |

## Stack

- **Backend:** Python puro (math, zlib, re, statistics, collections)
- **Frontend:** Streamlit
- **API:** Google Gemini 2.0 Flash (gerador + humanizador)
- **ML models:** nenhum — 100% estatistico

## Estrutura

```
IA-detector/
├── app.py              # Interface Streamlit (3 tabs: Gerador, Detector, Humanizador)
├── detector.py         # Motor de deteccao (15 metricas texto + codigo + sugestoes)
├── gerador.py          # Modulo de geracao de texto (usado via app.py)
├── humanizador.py      # Modulo de humanizacao adversarial
├── test_detector.py    # Testes: AI prose vs human text
├── test_code.py        # Testes: AI code vs human code
├── test_sugestoes.py   # Testes: analise de problemas e sugestoes
├── test_loop.py        # Testes: loop adversarial com Gemini
├── test_user.py        # Teste diagnostico detalhado
├── requirements.txt    # streamlit, google-generativeai, python-dotenv
├── docs/
│   └── PESQUISA_WATERMARKS.md  # Pesquisa sobre watermarks Unicode em LLMs
└── .env                # GEMINI_API_KEY
```

## API do Detector (detector.py)

### Funcoes principais

```python
from detector import avaliar, rotulo_nivel, analisar_problemas, aplicar_correcao

# Avaliar texto
score, metricas = avaliar(texto)       # score 0-100, dict de metricas
label = rotulo_nivel(score)            # "PROVAVEL IA", "INCONCLUSIVO", etc.

# Analisar problemas
problemas = analisar_problemas(texto)  # lista de dicts com tipo, trecho, sugestao

# Aplicar correcao
novo_texto = aplicar_correcao(texto, problema)  # substitui trecho

# Simular correcao (preview)
novo_score, _ = simular_correcao(texto, problema)  # score apos 1 fix
novo_score, _ = simular_todas(texto, problemas)    # score apos todos os fixes

# HTML destacado
html = gerar_texto_destacado(texto, problemas)     # texto com <span> amarelos

# Detectar codigo
is_code = detectar_codigo(texto)       # True se parece codigo
```

## Pesquisa: Watermarks Unicode em LLMs

Ver documento completo em [`docs/PESQUISA_WATERMARKS.md`](docs/PESQUISA_WATERMARKS.md).

**Resumo:** Modelos como GPT-o3/o4-mini inserem caracteres Unicode invisiveis (NNBSP U+202F) nos textos. A OpenAI diz que e um artefato de treinamento, nao watermark intencional. Google usa SynthID (watermark estatistico). A maioria dos detectores de IA (GPTZero, Originality.ai) usa analise estatistica como metodo principal — mesma abordagem do nosso detector.
