# sparse-vsas

TCC — Redes neurais **VSA esparsas** (*Sparse Vector Symbolic Architectures*) para representação composicional de cenas de vídeo com física de objetos, no estilo do dataset **MOVi** (Kubric).

A ideia central: em vez de um embedding denso e monolítico, cada rede codifica a cena em um **hipervetor discreto e esparso**, organizado em grupos que compartilham "unidades" físicas segundo um grafo fixo (o *scaffold*). Isso permite que redes treinadas **separadamente** (visão, física, decodificação de desfecho) sejam **coladas (Glue)** depois — alinhando seus códigos discretos por estatística, sem dados pareados nem re-treino conjunto.

## Estrutura

| Arquivo | Conteúdo |
|---|---|
| [base.py](base.py) | Arquitetura base: `SJConfig`, construção do *scaffold* esparso, e o módulo central `SparseVQCore` |
| [composed_nets.py](composed_nets.py) | Redes especialistas (visão, física, decodificador), o mecanismo de *Glue* entre arquiteturas, e utilitários de checkpoint |
| `movi_vision_sj_seed11.pkl` | Checkpoint treinado do especialista de visão |
| `movi_physics_sj_seed11.pkl` | Checkpoint treinado do especialista de física |
| `movi_decoder_sj_seed11.pkl` | Checkpoint treinado do decodificador de desfecho |

## Conceitos principais

### Scaffold (arquitetura esparsa)

`make_rigid_scaffold` gera uma máscara `[layers, groups, units]`: cada grupo (uma "população" de unidades) recebe um subconjunto fixo e esparso de unidades por camada. Dois grupos que compartilham uma unidade em alguma camada ficam conectados (`physical_edges`) — o resultado é um grafo esparso e estruturalmente distintivo, que funciona como uma espécie de "certificado de compatibilidade" entre redes (`architecture_intersection_tensor`, `structural_isomorphisms`). `permute_scaffold` embaralha os rótulos de grupos/unidades preservando esse grafo, para testar se duas arquiteturas são isomorfas.

### `SparseVQCore`

Módulo central (`base.py`). Para cada entrada:

1. Projeta a entrada em unidades e propaga por camadas (transições MLP).
2. Cada grupo lê apenas suas unidades físicas, produzindo um valor por grupo.
3. Faz **bind** (multiplicação elemento a elemento) do valor com um vetor de papel (`role`) fixo por grupo — semântica clássica de VSA.
4. Propaga mensagens unárias (por grupo) e de par (entre grupos conectados) de volta às unidades compartilhadas.
5. Quantiza cada grupo contra um *codebook* aprendido (estilo VQ-VAE, com straight-through estimator), produzindo códigos discretos por grupo.
6. Agrega tudo em um **program**: bundle dos vetores unários concatenado ao bundle dos vetores de par — a representação final da cena, ao mesmo tempo discreta (códigos) e graduada (vetor contínuo).

### Redes especialistas ([composed_nets.py](composed_nets.py))

- **`VisionSJSpecialist`**: CNN + GRU codifica frames de vídeo observados → `SparseVQCore` → decodifica frames RGB futuros.
- **`PhysicsSJSpecialist`**: MLP + GRU codifica a sequência de estados dos objetos → `SparseVQCore` → `CodeDynamics` prevê a distribuição de códigos futuros → decodifica o estado físico futuro.
- **`OutcomeDecoderSJ`**: a partir do estado futuro, prevê o desfecho do evento (classificação em 6 categorias).

### Glue: composição de redes independentes

`fit_sj_glue` recebe códigos discretos de duas redes com o mesmo *scaffold* (a menos de permutação):

1. Verifica compatibilidade estrutural via `structural_isomorphisms`.
2. Para cada par de grupos conectados no grafo, estima as distribuições conjuntas empíricas dos códigos e extrai a **componente pura de interação** (decomposição tipo ANOVA sobre o log da distribuição conjunta).
3. Resolve, por componente conexa do grafo, a permutação de códigos que minimiza o custo de alinhar essas interações entre as duas redes — sem qualquer par rotulado.
4. Retorna um `SJGlueResult` (mapeamento de grupos + mapeamento de valores por grupo, com margem de confiança), aplicável via `apply_glue_codes` para traduzir códigos de um modelo para o espaço de outro.

Isso é o que permite, por exemplo, traduzir os códigos do especialista de visão para o espaço de códigos do especialista de física, mesmo que as duas redes tenham sido treinadas de forma totalmente independente.

### Checkpoints

`save_checkpoint`/`load_checkpoint` serializam config, *scaffold* (masks), pesos e seed em um único arquivo (`format: sj-model-checkpoint-v1`). O campo `kind` (`"vision"`, `"physics"` ou `"decoder"`) determina qual classe é reconstruída ao carregar.

```python
from composed_nets import load_checkpoint

vision = load_checkpoint("movi_vision_sj_seed11.pkl")
physics = load_checkpoint("movi_physics_sj_seed11.pkl")
decoder = load_checkpoint("movi_decoder_sj_seed11.pkl")
```

## Como testar

O repositório não vem com dados reais nem script de treino — só os checkpoints. Para verificar que tudo está funcionando (dependências instaladas, checkpoints carregam, forward pass roda, Glue funciona), rode o smoke test com dados aleatórios:

```bash
python smoke_test.py
```

Saída esperada (aproximada):

```
[vision]  program=(2, 64) codes=(2, 4) future_rgb=(2, 4, 3, 32, 32)
[physics] program=(2, 64) codes=(2, 4) future_state=(2, 12, 10, 6)
[decoder] outcome_pred=(2, 6) codes=(2, 4)
[glue]    compatible=True group_map=(...) resolved_slots=(0, 1, 2, 3) objective=... margin=...
OK: all checkpoints loaded and ran successfully.
```

Como os dados são aleatórios (sem correlação real entre grupos), o `objective`/`margin` do Glue tendem a ficar perto de zero — isso é esperado e só significa que não há estrutura estatística real para alinhar. Para um teste de Glue significativo, é preciso gerar códigos a partir de dados reais do MOVi (mesmas cenas passadas pelos dois modelos), não deste script de sanity check.

### Teste de classificação real (MNIST)

`SparseVQCore` (o núcleo em [base.py](base.py)) não depende de vídeo nem de física — ele só recebe um vetor de embedding qualquer. [image_classification_test.py](image_classification_test.py) prova isso construindo um classificador de imagens do zero (CNN pequena → `SparseVQCore` → cabeça linear), sem reaproveitar nenhuma peça de [composed_nets.py](composed_nets.py), e treina/avalia em MNIST (baixado automaticamente via `torchvision` na primeira execução):

```bash
python image_classification_test.py
```

Resultado típico: ~93% de acurácia no teste depois de 3 épocas curtas — confirma que o gargalo VSA esparso (bind/bundle + VQ discreto) consegue aprender uma tarefa de classificação de imagens comum, não só as tarefas de vídeo/física do TCC.

Além do log de treino e da acurácia, o script imprime um `classification_report` (precisão/recall/F1 por classe) e salva dois PNGs:

- `confusion_matrix.png` — matriz de confusão do conjunto de teste.
- `sample_predictions.png` — grade de 16 imagens de teste com a predição e o rótulo verdadeiro (título verde se acertou, vermelho se errou).

## Requisitos

- Python 3.10+ (usa `X | Y` em type hints e `from __future__ import annotations`)
- `torch`
- `numpy`

Instalar tudo com um comando:

```bash
pip install -r requirements.txt
```
