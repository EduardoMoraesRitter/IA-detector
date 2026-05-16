# Pesquisa: Watermarks Unicode e Deteccao de Texto IA

Data: Maio 2025

## 1. O que sao watermarks Unicode em texto IA?

Modelos de linguagem (LLMs) podem inserir caracteres Unicode invisiveis nos textos que geram. Esses caracteres sao visualmente identicos a espacos normais, mas possuem codigos diferentes — detectaveis por editores de codigo ou ferramentas de analise.

### Caracteres mais comuns

| Caractere | Unicode | Nome | Uso |
|-----------|---------|------|-----|
| NNBSP | U+202F | Narrow No-Break Space | Principal "watermark" do ChatGPT |
| ZWSP | U+200B | Zero-Width Space | Marker invisivel (largura zero) |
| ZWNJ | U+200C | Zero-Width Non-Joiner | Separador invisivel |
| ZWJ | U+200D | Zero-Width Joiner | Conector invisivel |
| BOM | U+FEFF | Byte Order Mark | Marca de ordem de bytes |
| Em Space | U+2003 | Em Space | Espaco largo |
| Em Dash | U+2014 | Em Dash | Traco longo |
| Soft Hyphen | U+00AD | Soft Hyphen | Hifenizacao invisivel |
| Word Joiner | U+2060 | Word Joiner | Impede quebra de linha |

## 2. Quais LLMs fazem isso?

### ChatGPT (OpenAI) — Confirmado

- **Modelos afetados:** GPT-o3 e GPT-o4-mini (abril 2025)
- **Caractere:** Narrow No-Break Space (NNBSP, U+202F)
- **Condicoes:** Apenas respostas longas (ensaios, textos extensos)
- **Modelos NAO afetados:** GPT-4o (versoes anteriores)
- **Descoberto por:** Rumi (startup de AI academico)
- **Resposta OpenAI (22/abril/2025):** "Os caracteres especiais NAO sao watermark, sao um artefato de reinforcement learning em larga escala"
- **Status (23/abril/2025):** Parece ter sido corrigido — caracteres nao aparecem mais

### Google Gemini — SynthID (Oficial)

- **Tipo:** Watermark ESTATISTICO (nao Unicode)
- **Como funciona:** Altera sutilmente as probabilidades de selecao de tokens durante a geracao, criando um padrao estatistico detectavel
- **Deteccao:** Detector bayesiano (watermarked / not watermarked / uncertain)
- **Open source:** Sim! github.com/google-deepmind/synthid-text
- **Artefatos Unicode:** Tambem pode produzir alguns caracteres Unicode atipicos, mas nao e o mecanismo principal

### Claude (Anthropic) — Nao confirmado oficialmente

- **Relatos:** Ferramentas de terceiros detectam Zero-Width Space (U+200B) e outros caracteres em saidas do Claude
- **Posicao oficial:** Anthropic NAO confirmou nenhum watermark
- **Provavelmente:** Artefatos de tokenizacao, nao watermark intencional

### DeepSeek — Artefatos apenas

- **Tipo:** Nenhum watermark intencional
- **Artefatos:** Pode produzir caracteres Unicode atipicos como subproduto da tokenizacao
- **Posicao:** Sem mecanismo de watermark oficial

### Modelos Open Source (Llama, Mistral, etc.)

- **Watermark:** Nenhum built-in
- **Caracteres Unicode:** Minimo
- **Nota:** O SynthID do Google pode ser aplicado a qualquer modelo via HuggingFace

## 3. Os detectores de IA usam caracteres Unicode?

### Resposta curta: NAO como metodo principal

Os detectores de IA profissionais usam **analise estatistica de NLP** como metodo principal:

| Detector | Metodo Principal | Unicode como extra? |
|----------|-----------------|---------------------|
| **GPTZero** | Perplexidade + burstiness + ML | Nao |
| **Originality.ai** | ML classifier + metricas | Sim (ferramenta separada) |
| **Copyleaks** | ML + analise linguistica | Nao |
| **ZeroGPT** | Analise estatistica | Nao |
| **Nosso IA-Detector** | 15 metricas estatisticas | Possivel adicionar |

### Dado importante

Testes mostraram que **remover ou adicionar caracteres Unicode invisiveis NAO muda a deteccao** pelos detectores baseados em NLP. O conteudo e detectado como IA independentemente dos caracteres Unicode.

Isso confirma que a abordagem estatistica (perplexidade, burstiness, estilometria) e a que realmente funciona para deteccao de IA.

## 4. OpenAI: Ferramenta de 99.9% de precisao

A OpenAI desenvolveu internamente uma ferramenta de watermark ESTATISTICO (diferente dos caracteres Unicode) que altera a selecao de tokens para criar padroes detectaveis:

- **Precisao:** 99.9% em textos longos
- **Status:** NUNCA foi liberada ao publico
- **Razoes:** ~30% dos usuarios sao contra; watermark pode ser removido com traducao/edicao
- **Metodo:** Altera probabilidades de tokens (similar ao SynthID)
- **Diferente de:** Os caracteres Unicode (NNBSP) que sao artefatos

## 5. Projetos Open Source no GitHub

### Deteccao/remocao de caracteres Unicode

| Projeto | Tipo | Descricao |
|---------|------|-----------|
| [kovart/invisible-text](https://github.com/kovart/invisible-text) | Web app | Detecta e limpa caracteres Unicode invisiveis |
| [cronos3k/Text-Stealth-Watermark-Cleaner-Detector](https://github.com/cronos3k/Text-Stealth-Watermark-Cleaner-Detector) | Python | Detecta zero-width spaces, homoglyphs, NNBSP, control chars. Usa NFKC normalization |
| [juriku/hidden-characters-detector](https://github.com/juriku/hidden-characters-detector) | Python | Scan de arquivos/diretorios, integracao CI/CD, respeita .gitignore |
| [LeonardSEO/chatgpt-watermark-remover](https://github.com/LeonardSEO/chatgpt-watermark-remover) | Chrome ext | Remove U+202F, U+200B, control chars automaticamente |
| [ByteMastermind/Markless-GPT](https://github.com/ByteMastermind/Markless-GPT) | Chromium ext | Limpa automaticamente saidas do ChatGPT |
| [unixwzrd/UnicodeFix](https://github.com/unixwzrd/UnicodeFix) | CLI | Normaliza Unicode para ASCII (ChatGPT, Claude, Google) |
| [zenzired/UnicodeWatermarkCleaner](https://github.com/zenzired/UnicodeWatermarkCleaner) | Desktop app | Detecta e remove watermarks em documentos |

### Watermark estatistico

| Projeto | Tipo | Descricao |
|---------|------|-----------|
| [google-deepmind/synthid-text](https://github.com/google-deepmind/synthid-text) | Python | Watermark estatistico oficial do Google. Aplica-se a qualquer modelo via HuggingFace |

## 6. Como detectar caracteres Unicode (codigo)

### Detector simples em Python

```python
INVISIBLE_CHARS = {
    ' ': 'NNBSP (Narrow No-Break Space)',
    '​': 'ZWSP (Zero-Width Space)',
    '‌': 'ZWNJ (Zero-Width Non-Joiner)',
    '‍': 'ZWJ (Zero-Width Joiner)',
    '­': 'Soft Hyphen',
    '⁠': 'Word Joiner',
    '﻿': 'BOM (Byte Order Mark)',
    ' ': 'Em Space',
    ' ': 'En Space',
    ' ': 'Thin Space',
    ' ': 'Hair Space',
}

def detectar_unicode_watermark(texto):
    """Detecta caracteres Unicode invisiveis no texto."""
    encontrados = {}
    for i, ch in enumerate(texto):
        if ch in INVISIBLE_CHARS:
            nome = INVISIBLE_CHARS[ch]
            if nome not in encontrados:
                encontrados[nome] = {'codigo': repr(ch), 'contagem': 0, 'posicoes': []}
            encontrados[nome]['contagem'] += 1
            encontrados[nome]['posicoes'].append(i)
    return encontrados

def limpar_unicode(texto):
    """Remove todos os caracteres Unicode invisiveis."""
    for ch in INVISIBLE_CHARS:
        texto = texto.replace(ch, ' ' if ch in (' ', ' ', ' ', ' ', ' ') else '')
    return texto
```

### Como remover manualmente

1. **Bloco de Notas:** Cole o texto, salve como .txt (encoding UTF-8) e reabra
2. **Find & Replace:** No VS Code, use regex `[​‌‍ ﻿]` e substitua por nada
3. **Python:** `texto.encode('ascii', 'ignore').decode('ascii')`

## 7. Conclusoes

1. **Caracteres Unicode em texto IA sao reais**, mas na maioria dos casos sao artefatos de treinamento/tokenizacao, nao watermarks intencionais
2. **A OpenAI corrigiu** a insercao de NNBSP no GPT-o3/o4-mini apos a descoberta
3. **O unico watermark oficial e open source** e o SynthID do Google (estatistico, nao Unicode)
4. **Detectores de IA profissionais** (GPTZero, Originality.ai) usam analise estatistica, nao deteccao de Unicode
5. **Nosso detector** usa a mesma abordagem que funciona: 15 metricas estatisticas puras

## 8. Fontes

- [Rumi — New ChatGPT Models Seem to Leave Watermarks on Text](https://www.rumidocs.com/newsroom/new-chatgpt-models-seem-to-leave-watermarks-on-text)
- [WinBuzzer — OpenAI's o3/o4-mini Models Add Invisible Characters](https://winbuzzer.com/2025/04/21/openais-new-o3-o4-mini-models-add-invisible-characters-to-text-sparking-watermark-debate-xcxwbn/)
- [Clemens Jarnach — Find Invisible Unicode Characters aka "AI Watermarks"](https://clemensjarnach.github.io/02-articles/2025-04-24-article.html)
- [Google DeepMind — SynthID Text](https://github.com/google-deepmind/synthid-text)
- [Originality.ai — Invisible Text Detector](https://originality.ai/blog/invisible-text-detector-remover)
- [Oreate AI — OpenAI Watermark Technology 99.9%](https://www.oreateai.com/blog/openai-develops-highprecision-ai-text-watermarking-technology-new-breakthroughs-and-challenges-in-combating-cheating-in-education/5a1351ed287c7b08f643c4950a00e1ad)
- [kovart/invisible-text — GitHub](https://github.com/kovart/invisible-text)
- [cronos3k/Text-Stealth-Watermark-Cleaner-Detector — GitHub](https://github.com/cronos3k/Text-Stealth-Watermark-Cleaner-Detector)
