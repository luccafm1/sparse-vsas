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

## Requisitos

- Python 3.10+ (usa `X | Y` em type hints e `from __future__ import annotations`)
- `torch`
- `numpy`

Instalar tudo com um comando:

```bash
pip install -r requirements.txt
```
