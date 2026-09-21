# sparse-vsas

TCC — Redes neurais **VSA esparsas** (*Sparse Vector Symbolic Architectures*) para representação composicional de cenas de vídeo com física de objetos, no estilo do dataset **MOVi** (Kubric).

A ideia central: em vez de um embedding denso e monolítico, cada rede codifica a cena em um **hipervetor discreto e esparso**, organizado em grupos que compartilham "unidades" físicas segundo um grafo fixo (o *scaffold*). Isso permite que redes treinadas **separadamente** (visão, física, decodificação de desfecho) sejam **coladas (Glue)** depois — alinhando seus códigos discretos por estatística, sem dados pareados nem re-treino conjunto.

## Estrutura

| Arquivo | Conteúdo | estado |
|---|---|---|
| [base.py](base.py) | Arquitetura base: `SJConfig`, *scaffold* esparso, o core `SparseJointCore` e o modelo `FactorSJ` com treino e diagnósticos | ok |
| [glue.py](glue.py) | `SJGlue`, `fit_sj_glue` e `certify_sj_glue` | **não importa** |
| [sj_invariants.py](sj_invariants.py) | `code_probabilities` e `full_path_locality` | reconstrução temporária |
| [kfold_medmnist_test.py](kfold_medmnist_test.py) | Bateria k-fold em imagens médicas (MedMNIST) — ver [TESTES.md](TESTES.md) | ok |
| [plots.py](plots.py) | Matriz de confusão, predições de exemplo, métricas por fold, curva do trade-off | ok |
| [test_sj_invariants.py](test_sj_invariants.py) | 15 checks de contrato do `sj_invariants` | ok |
| [sj_image_classifier.py](sj_image_classifier.py) | Modelo/treino/relatórios dos testes de classificação (MNIST e FashionMNIST) | ok |
| [image_classification_test.py](image_classification_test.py) | Teste em MNIST — saídas em `results/mnist/` | ok |
| [fashion_classification_test.py](fashion_classification_test.py) | Teste em FashionMNIST — saídas em `results/fashion_mnist/` | ok |
| [composed_nets.py](composed_nets.py) | Redes especialistas (visão, física, decodificador) e utilitários de checkpoint | **API antiga** |
| [smoke_test.py](smoke_test.py) | Sanity check dos checkpoints MOVi | **API antiga** |
| `movi_*_sj_seed11.pkl` | Checkpoints treinados dos três especialistas | — |

### Pendências após a unificação das branches

O `base.py` foi reescrito (`SparseVQCore` virou `SparseJointCore`, o alfabeto passou
a ser por grupo e os campos de vídeo/MOVi saíram do `SJConfig`). Três consequências:

1. **`composed_nets.py` e `smoke_test.py` não rodam.** Eles dependem de
   `video_frames_obs`, `video_size`, `max_objects`, `state_dim`, `dyn_dim` e
   `state_embed`, todos removidos do `SJConfig`. Reativá-los é decisão de design:
   ou esses campos voltam, ou o pipeline MOVi é reescrito sobre o `FactorSJ`.
2. **`glue.py` não importa**, porque falta `sj_federated_networks_operator_v9.py`,
   que não foi commitado. É lá que está o algoritmo do Glue inteiro.
3. **`sj_invariants.py` é reconstrução**, feita a partir dos pontos de uso. O
   caminho de treino não passa por ele; só diagnósticos e as features do Glue.
   Ao receber o original, apague o atual e rode `python test_sj_invariants.py`.

## Conceitos principais

### Scaffold (arquitetura esparsa)

`make_rigid_scaffold` gera uma máscara `[layers, groups, units]`: cada grupo (uma "população" de unidades) recebe um subconjunto fixo e esparso de unidades por camada. Dois grupos que compartilham uma unidade em alguma camada ficam conectados (`physical_edges`) — o resultado é um grafo esparso e estruturalmente distintivo, que funciona como uma espécie de "certificado de compatibilidade" entre redes (`architecture_intersection_tensor`, `structural_isomorphisms`). `permute_scaffold` embaralha os rótulos de grupos/unidades preservando esse grafo, para testar se duas arquiteturas são isomorfas.

### `SparseJointCore`

Módulo central (`base.py`). Para cada entrada:

1. Projeta a entrada em unidades e propaga por camadas (transições MLP).
2. Cada grupo lê apenas suas unidades físicas, produzindo um valor por grupo.
3. Faz **bind** (multiplicação elemento a elemento) do valor com um vetor de papel (`role`) fixo por grupo — semântica clássica de VSA.
4. Propaga mensagens unárias (por grupo) e de par (entre grupos conectados) de volta às unidades compartilhadas.
5. Quantiza cada grupo contra um *codebook* aprendido (estilo VQ-VAE, com straight-through estimator), produzindo códigos discretos por grupo.
6. Agrega tudo em um **program**: bundle dos vetores unários concatenado ao bundle dos vetores de par — a representação final da cena, ao mesmo tempo discreta (códigos) e graduada (vetor contínuo).

### Redes especialistas ([composed_nets.py](composed_nets.py))

- **`VisionSJSpecialist`**: CNN + GRU codifica frames de vídeo observados → `SparseJointCore` → decodifica frames RGB futuros.
- **`PhysicsSJSpecialist`**: MLP + GRU codifica a sequência de estados dos objetos → `SparseJointCore` → `CodeDynamics` prevê a distribuição de códigos futuros → decodifica o estado físico futuro.
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

### Teste de classificação real (MNIST e FashionMNIST)

`SparseJointCore` (o núcleo em [base.py](base.py)) não depende de vídeo nem de física — ele só recebe um vetor de embedding qualquer. [sj_image_classifier.py](sj_image_classifier.py) prova isso construindo um classificador de imagens do zero (CNN pequena → `SparseJointCore` → cabeça linear), sem reaproveitar nenhuma peça de [composed_nets.py](composed_nets.py). Dois scripts finos rodam esse mesmo modelo em datasets diferentes:

```bash
python image_classification_test.py    # MNIST — saídas em results/mnist/
python fashion_classification_test.py  # FashionMNIST — saídas em results/fashion_mnist/
```

Resultados típicos depois de 6 épocas: ~96% de acurácia no MNIST, ~85% no FashionMNIST (mais difícil — mais variação visual dentro de cada classe) — confirma que o gargalo VSA esparso (bind/bundle + VQ discreto) consegue aprender uma tarefa de classificação de imagens comum, não só as tarefas de vídeo/física do TCC.

Cada execução salva, na pasta `results/<dataset>/` correspondente:

- `confusion_matrix.png` — matriz de confusão do conjunto de teste, com nome das classes.
- `sample_predictions.png` — grade de 16 imagens de teste com a predição e o rótulo verdadeiro (título verde se acertou, vermelho se errou).
- `metrics.txt` — log de treino, acurácia, `classification_report` (precisão/recall/F1 por classe) e o **relatório de especialização** (ver abaixo), tudo em texto.

### Penalidade de redundância entre grupos e especialização

Sem nenhum incentivo extra, os 8 grupos do `SparseJointCore` convergem para uma solução **redundante**: cada grupo, isoladamente, já aprende a "adivinhar" o dígito quase inteiro (informação mútua grupo↔rótulo alta e parecida entre todos os grupos), e nenhum par de grupos vizinhos no *scaffold* carrega mais informação junto do que o melhor grupo sozinho — ou seja, o mecanismo de troca de mensagens entre grupos conectados (pensado para viabilizar composicionalidade) não estava sendo usado de fato. Isso foi medido comparando a informação mútua normalizada (NMI, de `sklearn.metrics.normalized_mutual_info_score`) entre `codes[:, g]` e o rótulo, grupo a grupo, e entre pares de grupos conectados.

`pairwise_mi_penalty` (em [sj_image_classifier.py](sj_image_classifier.py)) resolve isso adicionando ao loss de treino uma estimativa, por lote, da informação mútua entre as distribuições de código (`out["probs"]`) de cada par de grupos, e penalizando quando dois grupos "sabem" a mesma coisa:

```python
loss = F.cross_entropy(...) + vq + 0.25 * commit + 0.05 * kl + REDUNDANCY_WEIGHT * pairwise_mi_penalty(out["probs"])
```

Com `REDUNDANCY_WEIGHT = 4.0` (a constante no topo do arquivo), o quadro muda por completo:

| | MNIST sem penalidade | MNIST com penalidade | FashionMNIST sem penalidade | FashionMNIST com penalidade |
|---|---|---|---|---|
| acurácia no teste | ~96% | ~96% (custo ≈0) | ~85% | ~85% (custo ≈0) |
| NMI(código, rótulo) por grupo | ~0.81 (uniforme) | ~0.42 (heterogêneo) | ~0.67 (uniforme) | ~0.31 (heterogêneo) |
| NMI entre códigos de grupos diferentes | ~0.84 (redundante) | ~0.08 (quase independente) | ~0.73 (redundante) | ~0.09 (quase independente) |
| ganho do par vs melhor grupo sozinho | −0.04 | **+0.19** | −0.05 | **+0.15** |

Ou seja: nenhum grupo sozinho fica confiante sobre a classe, mas pares de grupos conectados, combinados, carregam mais informação do que qualquer um isoladamente — especialização real via o grafo, não redundância, e o efeito se mantém em ambos os datasets. `specialization_report_text` calcula esses três números (NMI por grupo, NMI entre grupos, ganho do par) e eles vão pro `metrics.txt` de cada execução. `REDUNDANCY_WEIGHT = 0` reproduz o comportamento redundante original.

## Requisitos

- Python 3.10+ (usa `X | Y` em type hints e `from __future__ import annotations`)
- `torch`
- `numpy`

Instalar tudo com um comando:

```bash
pip install -r requirements.txt
```
