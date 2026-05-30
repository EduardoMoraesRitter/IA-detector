# DeBERTa — Como Funciona e Por Que Detecta IA Melhor

## O Que E DeBERTa?

**D**ecoding-**E**nhanced **B**ERT with **D**isentangled **A**ttention.
Criado pela Microsoft Research em 2020. Primeiro modelo a superar humanos
no benchmark SuperGLUE (89.9% vs humanos 89.8%).

## Arquitetura: Atencao Separada (Disentangled Attention)

### O Problema do BERT (modelo antigo)

No BERT, cada palavra entra no modelo e o significado + posicao sao
MISTURADOS num vetor so:

```
word_representation = content_embedding + position_embedding
```

E como misturar tinta azul (significado) e amarela (posicao) -> vira verde
e voce nao consegue separar mais.

### A Solucao do DeBERTa

DeBERTa mantém significado e posicao como DOIS VETORES INDEPENDENTES.
Cada token recebe:
- H_i: vetor de conteudo (o que a palavra significa)
- P_{i|j}: vetor de posicao relativa (onde esta em relacao a outra palavra)

### Tres Tipos de Atencao

O DeBERTa calcula 3 scores entre cada par de palavras:

1. Content-to-Content (C2C): "Como o significado de A se relaciona com B?"
2. Content-to-Position (C2P): "O significado de A importa pra posicao de B?"
3. Position-to-Content (P2C): "A posicao de A importa pro significado de B?"

Isso da uma compreensao MUITO mais rica do texto.

## Enhanced Mask Decoder (EMD)

- BERT: injeta posicao absoluta no INICIO e nunca mais revisita
- DeBERTa: usa posicao RELATIVA em todas as camadas, e so adiciona
  posicao absoluta NO FINAL, antes da predicao

Resultado: entendimento profundo de posicao relativa + consciencia
absoluta quando precisa.

## DeBERTa v3 — A Revolucao

### De MLM para RTD (Replaced Token Detection)

- v1/v2: treinado com MLM (preencher lacunas) — como BERT
- v3: treinado com RTD (detectar tokens substituidos) — estilo ELECTRA

Como funciona o RTD:
1. Uma IA pequena (Generator) troca algumas palavras no texto
2. O DeBERTa (Discriminator) detecta QUAIS foram trocadas
3. E um jogo GAN — Generator vs Discriminator

### Por Que RTD E Melhor

- MLM: modelo avalia apenas 15% dos tokens (os mascarados)
- RTD: modelo avalia 100% dos tokens (todos podem ser fakes)
- RTD e literalmente "detectar texto gerado por IA" — o pre-treino
  ja ensina isso!

### GDES (Gradient-Disentangled Embedding Sharing)

Problema do ELECTRA original: generator e discriminator compartilham
embeddings, mas seus losses puxam em direcoes opostas (cabo de guerra).
GDES resolve: bloqueia gradientes do discriminator de voltar pro generator.

## Por Que DeBERTa Detecta IA Melhor Que Estatisticas

### O que estatisticas (nosso detector) veem:
- Conta conectivos ("moreover", "furthermore")
- Mede uniformidade de frases (SD, burstiness)
- Detecta falta de contracoes
- 18 metricas numericas agregadas

### O que DeBERTa ve:
- Distribuicao de probabilidade de CADA token no contexto
- Padroes sutis na escolha de palavras
- Texto "perfeito demais" (perplexidade uniformemente baixa)
- Correlacoes entre posicao e vocabulario
- Coerencia cross-sentence (transicoes entre frases)
- Micro-estrutura que nenhuma metrica simples captura

### Analogia

Estatisticas = julgar uma pintura pela cor media
DeBERTa = analisar cada pincelada no contexto da obra inteira

## Comparacao Pratica

| Aspecto | Estatisticas (nosso) | DeBERTa (QuillBot) |
|---------|---------------------|---------------------|
| Parametros | 18 regras manuais | 304M parametros |
| Treinamento | Zero | Bilhoes de exemplos |
| Velocidade | Instantaneo | ~1-2s por texto |
| RAM | 0 | ~1.5GB (Large) |
| Precisao | ~60-70% | ~95-99% |
| Falso positivo | Raro | Muito raro |
| Texto generico | Falha | Detecta |
| Codigo IA | Bom | Excelente |

## Modelos Disponiveis (Gratis)

| Modelo | Base | Precisao | Uso |
|--------|------|----------|-----|
| desklib/ai-text-detector-v1.01 | v3 Large | RAID Best | Geral |
| vraj33/ai-text-detector-deberta | v3 Base | 99.4% | Leve |
| abhi099k/ai-text-detector-v-n4.0 | v3 Large | Alto | Diverso |
| LDKSolutions/chatgpt-qa-detector | v3 Large | Bom | Q&A |

## Como Usar

```python
from transformers import pipeline

classifier = pipeline("text-classification",
                      model="desklib/ai-text-detector-v1.01")

result = classifier("Your text here")
# -> [{'label': 'ai', 'score': 0.95}]
```

## Fontes

- DeBERTa Paper: https://arxiv.org/abs/2006.03654
- DeBERTa v3 Paper: https://arxiv.org/abs/2111.09543
- Microsoft Research: https://www.microsoft.com/en-us/research/publication/deberta
- GitHub: https://github.com/microsoft/DeBERTa
- Hugging Face: https://huggingface.co/microsoft/deberta-v3-large
