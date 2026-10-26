# Teste e validação

## Protocolo unificado (use este)

```bash
pip install medmnist            # além do requirements.txt

python validation.py --suite smoke           # checagem rápida (~1 min)
python validation.py --suite generality      # o mesmo core em 4 domínios
python validation.py --suite encoder         # densa vs CNN vs sem core
python validation.py --suite specialization  # varredura da penalidade
python validation.py --suite all             # tudo (~2 h em RTX 3060)

python report.py                             # tabelas LaTeX + figuras do artigo
python test_sj_invariants.py                 # contrato do sj_invariants
```

Resultados em `results/validation/<config>.json`, um arquivo por configuração,
com métricas por fold, agregado, descrição do dataset e **proveniência**
(commit, se a árvore estava suja, versão do torch, GPU, timestamp). As tabelas
e figuras do artigo saem daí via `report.py`, em `results/validation/paper/` —
nenhum número do artigo é transcrito à mão.

### Por que o protocolo foi unificado

Antes existiam três caminhos de teste incompatíveis:

| caminho | dataset | validação | arquitetura |
|---|---|---|---|
| `sj_image_classifier.py` | MNIST / Fashion-MNIST | split único | `layers=2 units=32 hv_dim=64` |
| `kfold_medmnist_test.py` | MedMNIST | 5-fold | `layers=3 units=128 hv_dim=256` |
| `test_sj_invariants.py` | sintético | — | outra config ainda |

Ou seja: o ~96% do MNIST e o ~93% do BloodMNIST que circulavam **vieram de
arquiteturas diferentes** e nunca foram comparáveis. Qualquer afirmação do tipo
"o core é agnóstico ao domínio" baseada em comparar um com o outro era inválida.

O `validation.py` fixa uma arquitetura de referência (`BASE_ARCH`) e só varia os
eixos declarados como ablação: **encoder** (densa / CNN), **core**
(sparse-joint / sem core, como controle) e **λ** (peso da penalidade de
redundância). Além disso:

- as partições de fold vêm de semente fixa e são **as mesmas para toda
  configuração do mesmo dataset**, o que torna as comparações pareadas por fold;
- a padronização é ajustada só no fold de treino;
- todos os datasets são subamostrados de forma estratificada para o mesmo teto
  (17 mil), senão o MNIST (70k) receberia 4× o orçamento de treino do
  BloodMNIST (17k);
- comparações saem com diferença pareada, Cohen's *d* e Wilcoxon — com 5 folds
  o teste tem pouco poder, então a leitura principal é o tamanho de efeito.

### Controle sem o core

O `--core none` troca o gargalo VSA por uma densa de mesma dimensão de saída,
mantendo encoder e cabeça. É o que isola **o custo da discretização** do custo
do encoder — duas coisas que estavam sendo confundidas.

## Scripts anteriores (mantidos, mas superados)

`kfold_medmnist_test.py` continua funcionando e é de onde o `validation.py`
importa as métricas já validadas. `sj_image_classifier.py` e os dois testes de
MNIST/Fashion-MNIST idem. Para resultados que vão ao artigo, use o
`validation.py`: os outros usam configurações próprias e não são comparáveis
entre si.

```bash
python kfold_medmnist_test.py --encoder cnn --redundancy-weight 0.25
python image_classification_test.py
```

Cada run deles salva em `results/kfold/<nome>/`: `report.txt`, `metrics.json`,
`confusion_matrix.png`, `sample_predictions.png` e `per_fold_metrics.png`.

## O que a bateria mede

**Dataset.** MedMNIST v2 (Yang et al., *Scientific Data* 2023) — benchmark
padronizado de imagem médica em 28×28. Os splits oficiais são unidos e a
validação é por `StratifiedKFold`. ACC e AUC macro são as mesmas métricas que o
MedMNIST reporta, então os números são comparáveis aos baselines publicados. A
padronização é ajustada só no fold de treino, para não vazar.

**Classificação:** acurácia, AUC macro (OvR), acurácia balanceada, F1
macro/ponderado, precisão, recall, kappa de Cohen, MCC, relatório por classe e
matriz de confusão.

**Arquitetura:** NMI(código, rótulo) por grupo, NMI entre grupos (redundância),
ganho do par de grupos conectados vs o melhor grupo sozinho, fillers usados por
grupo, perplexidade do código e recuperação VSA (bundle → unbind → cleanup). As
três primeiras seguem as definições validadas na branch `add-tests-and-docs`.

**Encoder.** O `FactorSJ` monta o próprio encoder como uma única densa
(`Linear(input_dim → frame_embed)`) e o `train()` recebe vetor achatado. A flag
`--encoder cnn` substitui `model.encoder` por uma CNN pequena que remonta a
imagem internamente — o `forward` faz `self.sj(self.encoder(x))`, então basta
qualquer módulo `[B, input_dim] → [B, frame_embed]`. Nada no `base.py` muda.

## Resultados — BloodMNIST (8 classes, 17.092 imagens)

5 folds, 120 épocas, batch 1024, RTX 3060.

| | linear | **CNN** | CNN + penalidade 0,25 |
|---|---|---|---|
| acurácia | 0,8579 ± 0,008 | **0,9325 ± 0,009** | 0,9281 ± 0,006 |
| AUC macro | 0,9671 ± 0,003 | **0,9880 ± 0,002** | 0,9845 ± 0,002 |
| acurácia balanceada | 0,8343 ± 0,010 | **0,9226 ± 0,012** | 0,9196 ± 0,007 |
| F1 macro | 0,8372 ± 0,010 | **0,9236 ± 0,011** | 0,9180 ± 0,007 |
| NMI(código, rótulo) | 0,5279 | 0,6400 | 0,5030 |
| NMI entre grupos | 0,4305 | 0,4998 | **0,2380** |
| ganho do par | 0,0471 | 0,0602 | **0,1230** |
| recuperação VSA | 0,9840 | **0,9919** | 0,7548 |
| perplexidade | 5,12 | 4,90 | 3,63 |
| tempo/fold | 46 s | 91 s | 94 s |

Referência do MedMNIST para BloodMNIST 28×28: ResNet-18 ≈ 0,958 ACC / 0,998 AUC;
auto-sklearn ≈ 0,878 / 0,984.

## Leitura dos resultados

**1. O encoder era o gargalo dominante, não o núcleo VSA.** Trocar a densa sobre
pixel cru por uma CNN pequena vale +7,5 pp de acurácia (0,858 → 0,933) e +8,8 pp
de acurácia balanceada. Com isso a arquitetura encosta nos baselines publicados,
mantendo o gargalo discreto de 8 grupos no meio. Todo resultado anterior a esta
mudança subestimava o núcleo.

**2. A redundância entre grupos piora quando o encoder melhora** (0,43 → 0,50).
Faz sentido: com features boas, cada grupo sozinho já tem sinal suficiente para
prever bem, então todos convergem para a mesma informação. O problema não é
artefato de encoder fraco — ele se agrava conforme o resto melhora.

**3. A penalidade de redundância resolve isso, com um custo localizado.** Com
CNN + peso 0,25: a redundância cai pela metade (0,50 → 0,238), o **ganho do par
dobra (0,060 → 0,123)** — o maior valor observado em qualquer configuração
testada — e a acurácia praticamente não se move (0,9325 → 0,9281, −0,4 pp).
Em troca, a recuperação VSA cai de 0,992 para 0,755.

Esse custo **não era artefato de treino curto**: na primeira varredura (batch
256, 30 épocas) o cleanup caía para ~0,65, e havia a hipótese de que fosse
subtreino. Com 120 épocas e CNN o cleanup base sobe para 0,992, mas a penalidade
ainda o derruba para 0,755. O conflito é real e reprodutível entre encoders e
orçamentos de treino.

## Varredura do peso da penalidade

Feita antes, em **batch 256 / 30 épocas com encoder linear** — serve para a forma
da curva, não para comparação direta com a tabela acima. Figura em
`results/kfold/redundancy_sweep.png`.

| peso | acurácia | NMI entre grupos | ganho do par | cleanup |
|---|---|---|---|---|
| 0 | 0,8530 | 0,430 | +0,047 | 0,948 |
| 0,1 | 0,8530 | 0,222 | +0,092 | 0,727 |
| 0,25 | 0,8529 | 0,187 | **+0,096** | 0,660 |
| 0,5 | 0,8509 | 0,150 | **+0,097** | 0,643 |
| 1 | 0,8509 | 0,122 | +0,089 | 0,660 |
| 2 | 0,8496 | 0,107 | +0,075 | 0,679 |
| 4 | 0,8481 | 0,088 | +0,070 | 0,659 |

Duas coisas que a curva mostra: o ganho do par **satura em 0,25–0,5 e regride**
depois (o peso 4, usado na `add-tests-and-docs`, entrega menos composicionalidade
que 0,25); e o cleanup cai de degrau logo no menor peso, sem se recuperar com
pesos maiores — não dá para escapar do custo só ajustando o peso.

## DermaMNIST (7 classes, 10.015 imagens)

Rodado com encoder linear, batch 256, 30 épocas — desatualizado em relação ao
resto, mas o achado se sustenta: acurácia 0,716 com **acurácia balanceada
0,351**. O modelo estava essencialmente prevendo *melanocytic nevi* (≈67% do
dataset). Só aparece porque a bateria mede métricas balanceadas — a acurácia
sozinha esconderia isso em dataset médico desbalanceado. Vale refazer com CNN.

## Como a penalidade foi integrada

O `pairwise_mi_penalty` da `add-tests-and-docs` foi portado para o harness
(`kfold_medmnist_test.py`, flag `--redundancy-weight`). O `base.py` **não foi
alterado**: `train_supervised()` replica o loop de
`base.train(mode="acquisition", pairs=None)` e soma o termo. Com peso 0 o loop
reproduz o baseline, o que serve de controle da comparação.

A motivação é que, em `acquisition_losses`, os termos que forçariam
especialização (`response` e `invariance`) **são zerados quando `pairs=None`**:

```python
else:
    zero = out["program"].new_zeros(())
    losses["response"], losses["invariance"] = zero, zero
```

Classificação supervisionada não tem pares de intervenção, então sobram
`sufficiency + vq + support` — nenhum deles separa os grupos.

## Decisões em aberto (design, para o Lucca)

1. Verificar se o Glue ainda certifica com cleanup em ~0,75. Se sim, o peso
   0,25 com CNN é a melhor configuração encontrada (ganho do par 0,123, acurácia
   0,928).
2. Ou somar um termo de separação de codebook para contrabalançar — a queda de
   perplexidade junto com o cleanup sugere que a independência está sendo obtida
   afrouxando a separação entre entradas do codebook, e o cleanup é cosseno
   contra esse mesmo codebook.
3. Ou abandonar a penalidade e usar pares de intervenção (`response` /
   `invariance`), que é o mecanismo que a arquitetura nova já tem para isso. A
   penalidade de MI é um substituto para o cenário supervisionado, não o
   desenho original.

## Pendências

- **`sj_invariants.py` aqui é reconstrução temporária.** O original não foi
  commitado na `ver-real`. O `forward()` do core e os losses de treino não
  passam por ele, então os números acima não dependem da reconstrução — só os
  diagnósticos e as features de programa do Glue dependem. Quando o arquivo
  original chegar: apague o meu e rode `python test_sj_invariants.py` contra o
  dele; os 15 checks de contrato dizem se a semântica bate.
- **`sj_federated_networks_operator_v9.py` continua faltando**, então o
  `glue.py` não importa e nada de Glue foi testado.
- `composed_nets.py` ainda usa a API antiga (`SparseVQCore`, `cfg.state_embed`).
- A métrica de localidade do scaffold existe atrás de `--locality`, mas depende
  da definição de `full_path_locality` — confirmar contra o original antes de
  reportar esse número.
