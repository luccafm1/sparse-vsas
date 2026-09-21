# Resultados — especialistas colados vs. baselines

Cinco seeds, três modelos, cinco fatores generativos. O ponto do experimento é
uma comparação só: **colar dois SJ especialistas supera treinar um único modelo
monolítico?**

Todos os números abaixo são acurácia em porcentagem, exceto `MAE` (graus, menor
é melhor). `média ± desvio-padrão` sobre as 5 seeds.

## Os três modelos

| modelo | o que é |
|---|---|
| **MLP** | baseline denso, sem bottleneck simbólico |
| **SJ** | um único `SparseJointCore` — o bottleneck esparso/simbólico sozinho |
| **GLUED SPECIALISTS** | dois SJ treinados com seeds diferentes, com os códigos de um traduzidos para o outro via `SJGlue` (ver [glue.py](glue.py)) |

## Resumo

| Métrica | MLP | SJ | GLUED SPECIALISTS |
|---|---|---|---|
| shape | 58.06 ± 0.89 | 42.95 ± 0.24 | **77.88 ± 2.24** |
| scale | 49.07 ± 0.50 | 45.68 ± 1.04 | **78.13 ± 1.10** |
| orientation | 9.50 ± 0.42 | 2.73 ± 0.06 | **22.68 ± 8.78** |
| x | 78.97 ± 0.20 | 43.77 ± 1.70 | **80.56 ± 0.67** |
| y | 77.63 ± 0.05 | 41.98 ± 2.67 | **79.50 ± 0.40** |
| **macro** | 54.65 ± 0.24 | 35.42 ± 0.35 | **67.75 ± 1.64** |
| all-5 | 2.62 ± 0.33 | 0.07 ± 0.01 | **9.88 ± 3.30** |
| orient. within-1 | 24.39 ± 0.62 | 8.27 ± 0.07 | **32.53 ± 9.80** |
| orient. MAE (°) ↓ | 86.81 ± 0.54 | 88.57 ± 0.30 | **84.01 ± 3.14** |

`macro` é a média dos cinco fatores. `all-5` exige os cinco corretos no mesmo
exemplo — por ser um produto, ele é dominado pelo fator mais fraco.
`within-1` conta acerto se a orientação cai no bin correto ou num vizinho.

## Ganho da cola (pontos percentuais)

| Métrica | vs MLP | vs SJ |
|---|---|---|
| shape | +19.82 | +34.93 |
| scale | +29.06 | +32.45 |
| orientation | +13.18 | +19.96 |
| x | +1.59 | +36.79 |
| y | +1.87 | +37.52 |
| macro | +13.11 | +32.33 |
| all-5 | +7.26 | +9.81 |
| orient. within-1 | +8.14 | +24.25 |
| orient. MAE (°) | −2.80 | −4.56 |

## Leitura

### 1. O ganho vem da composição, não da capacidade do backbone

O SJ sozinho **perde do MLP em todos os cinco fatores** — em x e y por ~35 pp,
que é a maior distância da tabela. Ainda assim, colar dois SJ produz o melhor
modelo em todas as métricas. O backbone não ficou melhor entre uma linha e
outra; o que mudou foi haver dois deles e uma tradução entre os códigos.

Isso é o resultado central, e ele é mais forte do que "a cola ajuda": a cola
recupera muito mais do que a diferença SJ→MLP. Se o efeito fosse só o de um
ensemble, esperaríamos um ganho modesto sobre o *melhor* dos componentes, não
+13 pp de macro sobre um baseline que cada componente individualmente não
alcança.

### 2. x e y já estavam saturados; o ganho real é nos fatores categóricos

O MLP faz 78.97 / 77.63 em x e y, e a cola faz 80.56 / 79.50 — ganho de +1.6 e
+1.9 pp. Em compensação, shape ganha +19.8 e scale +29.1 pp. A cola está
resolvendo os fatores discretos e mal toca nos posicionais.

Vale notar quanto isso está acima do acaso. Assumindo a grade padrão de 5
fatores (3 formas, 6 escalas, 40 orientações, 32 posições em cada eixo):

| fator | acaso | MLP | SJ | GLUED |
|---|---|---|---|---|
| shape | 33.3 | 1.7× | 1.3× | 2.3× |
| scale | 16.7 | 2.9× | 2.7× | 4.7× |
| orientation | 2.5 | 3.8× | **1.1×** | 9.1× |
| x | 3.1 | 25.3× | 14.0× | 25.8× |
| y | 3.1 | 24.8× | 13.4× | 25.8× |

Em múltiplos do acaso, x e y são de longe o que todos os modelos fazem melhor —
a acurácia absoluta de ~79% engana porque o acaso ali é 3%. E o SJ sozinho está
**em cima do acaso na orientação** (2.73 contra 2.50): ele simplesmente não
aprende esse fator.

### 3. Orientação é o fator instável, e é ele que trava o `all-5`

Desvio-padrão de 8.78 no glue contra 0.42 do MLP. Por seed, a orientação vai de
12.64 a 31.94 — uma variação de 2.5× entre execuções que só diferem na seed:

| seed | orient. | within-1 | MAE° |
|---|---|---|---|
| 0 | 25.42 | 37.01 | 85.21 |
| 1 | 29.21 | 39.47 | 82.50 |
| 2 | **14.20** | **23.20** | **86.94** |
| 3 | **12.64** | **20.83** | **86.15** |
| 4 | 31.94 | 42.12 | 79.25 |

As três métricas se movem juntas (correlação de +0.996 entre acurácia e
within-1, e −0.878 com o MAE). Isso descarta ruído de medição: é o alinhamento
do fator de orientação entre os dois especialistas que ora acontece, ora não.
Seeds 2 e 3 são as ruins; 1 e 4, as boas.

Como o `all-5` é o produto dos cinco acertos, ele herda essa instabilidade
inteira — daí o ±3.30 sobre uma média de 9.88. Os 9.88% parecem baixos em
absoluto, mas são 3.8× o MLP, e o acaso aqui é 0.0001%.

### 4. A orientação continua angularmente ruim mesmo quando "acerta"

Um chute uniforme em 360° dá MAE de ~90°. Os três modelos ficam entre 84 e
89 — ou seja, **nenhum deles resolve orientação em termos angulares**, nem o
glue nas suas melhores seeds (79.25° na seed 4).

Isso convive com `within-1` de 32.5% (4.3× o acaso de 7.5%) porque os erros são
de cauda pesada: o modelo ou acerta de perto, ou erra por ~180°. A explicação
mais provável é simetria da forma — quadrado tem simetria de 90°, elipse de
180°, então boa parte dos ângulos é genuinamente indistinguível a partir do
pixel. Se for isso, o MAE bruto **não é a métrica certa** para esse fator, e o
número honesto é `within-1` ou um MAE calculado módulo a simetria da forma.
Vale confirmar antes de reportar o MAE em qualquer lugar.

## Dados por seed

| seed | modelo | shape | scale | orient | x | y | macro | all-5 | w/in-1 | MAE° |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | MLP | 58.77 | 48.52 | 9.57 | 79.05 | 77.61 | 54.70 | 2.88 | 24.64 | 87.00 |
| 0 | SJ | 43.03 | 44.05 | 2.63 | 45.51 | 40.84 | 35.21 | 0.09 | 8.26 | 88.26 |
| 0 | GLUED | 78.44 | 78.34 | 25.42 | 81.24 | 79.58 | 68.61 | 11.65 | 37.01 | 85.21 |
| 1 | MLP | 57.85 | 49.80 | 9.88 | 78.90 | 77.58 | 54.80 | 2.79 | 25.01 | 87.21 |
| 1 | SJ | 43.20 | 45.82 | 2.73 | 44.94 | 40.45 | 35.43 | 0.06 | 8.32 | 88.37 |
| 1 | GLUED | 75.52 | 78.12 | 29.21 | 79.55 | 79.54 | 68.39 | 13.26 | 39.47 | 82.50 |
| 2 | MLP | 57.54 | 48.71 | 9.77 | 79.08 | 77.69 | 54.56 | 2.60 | 24.64 | 87.23 |
| 2 | SJ | 42.58 | 46.31 | 2.80 | 43.66 | 44.30 | 35.93 | 0.08 | 8.30 | 88.74 |
| 2 | GLUED | 76.10 | 76.53 | 14.20 | 80.30 | 79.72 | 65.37 | 5.38 | 23.20 | 86.94 |
| 3 | MLP | 59.15 | 49.04 | 9.49 | 79.17 | 77.61 | 54.89 | 2.77 | 24.27 | 85.94 |
| 3 | SJ | 42.86 | 46.79 | 2.74 | 43.62 | 39.05 | 35.01 | 0.06 | 8.33 | 89.01 |
| 3 | GLUED | 81.20 | 79.62 | 12.64 | 81.04 | 79.85 | 66.87 | 7.49 | 20.83 | 86.15 |
| 4 | MLP | 56.99 | 49.27 | 8.81 | 78.66 | 77.68 | 54.28 | 2.05 | 23.39 | 86.66 |
| 4 | SJ | 43.07 | 45.45 | 2.73 | 41.11 | 45.28 | 35.53 | 0.06 | 8.16 | 88.47 |
| 4 | GLUED | 78.14 | 78.04 | 31.94 | 80.66 | 78.82 | 69.52 | 11.62 | 42.12 | 79.25 |

## Ressalvas

- **Cinco seeds é pouco** para o fator instável. Para shape, scale, x, y e macro
  os desvios são pequenos e a separação entre modelos é de dezenas de pp, então
  a conclusão principal aguenta. Para orientação, com 12.64 e 31.94 no mesmo
  conjunto de 5 pontos, a média de 22.68 não descreve bem a distribuição —
  rodar mais seeds antes de citar esse número isolado.
- **A instabilidade da orientação não foi diagnosticada.** Não se sabe ainda se
  é o grounding do glue que falha nesse grupo, se é o SJ que às vezes não
  aprende o fator (ele está no acaso sozinho), ou as duas coisas.
- **Os números acima são da arquitetura bipolar/Hadamard.** O `base.py` migrou
  para FHRR (`sparse-joint-fhrr-v1`) e o glue foi portado para a `main`, mas a
  bateria ainda não foi reexecutada sobre a arquitetura nova. Os resultados
  valem como baseline a reproduzir, não como estado atual.
- **A grade de fatores usada na tabela de acaso foi inferida** dos nomes das
  métricas (3/6/40/32/32). Se a configuração real diferir, aquela tabela muda —
  as demais, não.
