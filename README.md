# IA Detector

AI-generated text and code detector using **pure statistical analysis** — zero ML models, zero API calls for detection. Runs entirely offline.

Built with the same core principles behind GPTZero, Originality.ai, and academic tools like DetectGPT and GLTR, but fully transparent and open-source.

## Features

- **15-metric detection engine** for prose (entropy, compression, stylometry)
- **Code detection mode** with comment analysis, pedagogical phrase detection, and structural metrics
- **Unicode watermark scanner** — detects invisible characters inserted by ChatGPT/LLMs (NNBSP, ZWSP, ZWJ, etc.)
- **QuillBot-style UI** — sentence-level highlighting, AI trigger breakdown, complete rewrite suggestions with word-level diff
- **One-click rewrite** — Gemini-powered full text rewrite with "Try again / Cancel / Replace" flow
- **Adversarial humanizer** — iterative rewrite loop using the detector itself as feedback
- **CLI tools** — standalone generator, detector, and humanizer scripts

## How Detection Works

### The Problem

AI-generated text has measurable statistical fingerprints. LLMs produce text that is more predictable, more uniform, and more formally structured than human writing. This detector exploits those differences using 15 independent metrics.

### Core Techniques

**1. Compression Analysis (ZipPy method)**

AI text is more repetitive and predictable, which means it compresses better with algorithms like zlib. The ratio of compressed size to original size is a strong AI signal.

```
AI text:   compressed/original = 0.25-0.40 (highly compressible)
Human text: compressed/original = 0.40-0.60+ (less compressible)
```

**2. Entropy Measurement**

Shannon entropy of the word distribution measures vocabulary diversity. AI tends to reuse words in predictable patterns, resulting in lower entropy.

```
AI text:   ~6-8 bits (predictable vocabulary)
Human text: ~8-10+ bits (diverse vocabulary)
```

**3. Sentence Rhythm Analysis (Burstiness)**

Humans write with "bursts" — mixing very short sentences with long, rambling ones. AI produces suspiciously uniform sentence lengths.

Three metrics capture this:
- **Standard deviation** of sentence lengths (most discriminative single metric)
- **Coefficient of variation** (burstiness)
- **Adjacent sentence variation** — ratio between consecutive sentences

**4. Stylometric Fingerprinting**

AI has specific writing habits that are statistically detectable:

| Signal | AI Pattern | Human Pattern |
|--------|-----------|---------------|
| Contractions | Avoids them (< 0.5/100 words) | Uses naturally (2-5+/100 words) |
| Personal pronouns | Almost none (< 1.5%) | Frequent (3%+) |
| Content/function word ratio | Heavy on content words (> 1.3) | Balanced (~1.0) |
| Sentence openers | Repetitive starts | Varied starts |

**5. AI Vocabulary Detection**

LLMs overuse specific words and phrases that are statistical dead giveaways:

- **Connectives**: "moreover", "furthermore", "additionally", "consequently"
- **Formal verbs**: "utilize", "facilitate", "leverage", "encompass"
- **Adjectives**: "comprehensive", "robust", "multifaceted", "transformative"
- **Phrases**: "plays a crucial role", "it is important to note", "in today's world"
- **Formulaic patterns**: "In today's..." openers, "In conclusion..." closers

**6. Punctuation Entropy**

AI punctuates at regular intervals. Humans are more erratic. Shannon entropy of the gaps between punctuation marks captures this.

**7. Consensus Mechanism**

When 4+ independent signals agree the text is AI-generated, the detector enforces minimum scores to prevent weak metrics from diluting strong signals:

```
4 strong signals → min 65%
5 strong signals → min 75%
6 strong signals → min 82%
7+ strong signals → min 90%
```

### Code Detection

When code is detected, the engine switches to specialized metrics:

| Metric | What it detects | Why it matters |
|--------|----------------|----------------|
| Comment ratio | AI over-comments (ratio > 0.3) | Real students rarely write comments |
| Pedagogical phrases | "I added...", "This function...", "Here we..." | AI explains its own work |
| "I [verb]" comments | `# I added`, `# I used`, `# I created` | AI narrating its process |
| Variable = input("variable") | `adjective = input("adjective:")` | AI names variables after prompts |
| Sequential input() calls | 3+ consecutive `input()` lines | AI follows patterns literally |
| Line length uniformity | Low CV of line lengths | AI writes uniform code |
| Perfect indentation | All indents are multiples of 4 | AI formats perfectly |

### Unicode Watermark Detection

Some LLMs (notably ChatGPT o3/o4-mini) embed invisible Unicode characters in their output. The detector scans for:

- **Known watermarks**: NNBSP (U+202F), ZWSP (U+200B), ZWNJ (U+200C), ZWJ (U+200D), SHY (U+00AD), WJ (U+2060), BOM (U+FEFF)
- **Unknown invisibles**: Generic scanner catches ANY character in suspicious Unicode categories (Cf, Cc, Co, Cn, Zl, Zp)
- **Typographic characters**: Em dash, curly quotes, ellipsis — common in AI output

All hidden characters are displayed inline with colored badges (red = watermark, orange = unknown, blue = typographic).

## Getting Started

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/IA-detector.git
cd IA-detector

# Install
pip install -r requirements.txt

# Run the web app
streamlit run app.py
```

Open `http://localhost:8501`.

### Gemini API (optional)

The **detector works entirely offline** — no API needed. The Generator, Humanizer, and Rewrite features require a Gemini API key:

1. Get a free key at [Google AI Studio](https://aistudio.google.com/apikey)
2. Create a `.env` file: `GEMINI_API_KEY=your_key_here`
3. Or paste it in the sidebar

## Usage

### Web Interface (Streamlit)

Three tabs:

1. **Detector** — Paste text, get a 0-100% AI probability score with sentence-level highlighting, trigger breakdown, and one-click rewrite suggestions
2. **Generator** — Generate AI text with Gemini (for testing the detector)
3. **Humanizer** — Adversarial rewrite loop: Gemini rewrites the text, the detector evaluates it, repeat until the target score is reached

### CLI

```bash
# Detect
python detector.py -a my_essay.txt

# Generate AI text
python gerador.py -t "impact of AI on education" -n 300

# Humanize (adversarial loop)
python humanizador.py -a ai_text.txt --iteracoes 5 --score-alvo 35
```

### Python API

```python
from detector import avaliar, rotulo_nivel, analisar_problemas, detectar_watermarks

# Score a text (0-100, higher = more likely AI)
score, metrics = avaliar(text)
label = rotulo_nivel(score)  # "PROVAVEL IA", "INCONCLUSIVO", etc.

# Get specific problems and suggestions
problems = analisar_problemas(text)
for p in problems:
    print(f"{p['tipo']}: {p['trecho']} -> {p.get('sugestao')}")

# Detect invisible Unicode watermarks
wm = detectar_watermarks(text)
if wm['tem_watermark']:
    print(f"Found {wm['total_invisiveis']} invisible characters!")
    print(wm['texto_limpo'])  # cleaned text
```

## Metrics Reference

### Text Mode — 15 Metrics

| # | Metric | Measures | AI | Human | Weight |
|---|--------|----------|-------|---------|--------|
| 1 | Shannon Entropy | Word distribution diversity | 6-8 bits | 8-10+ bits | 0.06 |
| 2 | Compression (ZipPy) | zlib compression ratio | 0.25-0.40 | 0.40-0.60+ | 0.06 |
| 3 | Burstiness (CV) | Sentence length variation | 0.1-0.3 | 0.5-1.0+ | 0.06 |
| 4 | Sentence SD | Absolute sentence length StdDev | SD 2-5 | SD 6-12+ | 0.10 |
| 5 | Adjacent Variation | Ratio between consecutive sentences | CV < 0.3 | CV > 0.5 | 0.06 |
| 6 | TTR | Type-Token Ratio (unique/total) | ~0.45 | ~0.55+ | 0.02 |
| 7 | Hapax Legomena | Words used exactly once | low | high | 0.02 |
| 8 | Content/Function Ratio | Content words / function words | ~1.37 | ~0.98 | 0.11 |
| 9 | Contractions | Contractions per 100 words | < 0.5 | 2-5+ | 0.11 |
| 10 | Punctuation Entropy | Shannon entropy of punctuation gaps | < 2.5 | > 3.0 | 0.04 |
| 11 | AI Connectives | "moreover", "furthermore"... per sentence | > 0.5 | < 0.2 | 0.10 |
| 12 | Sentence Opener Diversity | Unique first words / total sentences | < 0.6 | > 0.8 | 0.03 |
| 13 | Personal Pronouns | 1st/2nd person percentage | < 1.5% | > 3% | 0.07 |
| 14 | AI Phrases | Dead giveaway expressions per 100 words | > 0.5 | ~0 | 0.09 |
| 15 | Formulaic Patterns | Opening/closing templates (0-1) | 0.5-1.0 | ~0 | 0.07 |

### Score Levels

| Score | Label | Meaning |
|-------|-------|---------|
| 85-100% | MUITO PROVAVEL IA | Very likely AI-generated |
| 65-84% | PROVAVEL IA | Likely AI-generated |
| 45-64% | INCONCLUSIVO | Inconclusive |
| 25-44% | PROVAVEL HUMANO | Likely human-written |
| 0-24% | MUITO PROVAVEL HUMANO | Very likely human-written |

## Tech Stack

- **Detection Engine**: Pure Python (math, zlib, re, statistics, collections) — no ML
- **Frontend**: Streamlit
- **AI Features**: Google Gemini 2.0 Flash (generator, humanizer, rewrite)
- **Unicode Analysis**: unicodedata standard library

## Project Structure

```
IA-detector/
├── app.py              # Streamlit web interface
├── detector.py         # Detection engine (15 metrics + watermarks + suggestions)
├── gerador.py          # CLI text generator
├── humanizador.py      # CLI adversarial humanizer
├── test_detector.py    # Tests: AI vs human text
├── test_code.py        # Tests: AI vs human code
├── test_sugestoes.py   # Tests: problem analysis and suggestions
├── test_loop.py        # Tests: adversarial loop with Gemini
├── requirements.txt    # streamlit, google-generativeai, python-dotenv
├── docs/
│   └── PESQUISA_WATERMARKS.md  # Research on Unicode watermarks in LLMs
├── .env.example        # API key template
└── .gitignore
```

## Research References

This project draws from academic research on AI text detection:

- **DetectGPT** (Mitchell et al., 2023) — Perturbation-based detection using log probability
- **GLTR** (Gehrmann et al., 2019) — Statistical analysis of token predictions
- **ZipPy** (Thinkst) — Compression-based detection using zlib ratios
- **GPTZero** — Perplexity and burstiness as core metrics
- **StyloAI** — Stylometric fingerprinting for AI attribution
- **Adversarial Paraphrasing** (NeurIPS 2025) — Using detectors as feedback for iterative rewriting
- **SICO** (TMLR 2024) — Prompt optimization to evade detectors
- **TempParaphraser** (EMNLP 2025) — Progressive rewriting with increasing aggressiveness

## License

MIT
