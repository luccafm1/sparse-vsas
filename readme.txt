================================================================================
 sparse-vsas - Artefatos do projeto
 Redes neurais VSA esparsas (Sparse Vector Symbolic Architectures) com nucleo
 "sparse-joint" e mecanismo de composicao (Glue) entre redes treinadas
 separadamente.
 Repositorio: https://github.com/luccafm1/sparse-vsas
================================================================================

1. VISAO GERAL
--------------------------------------------------------------------------------
O projeto propoe um nucleo neural (SparseJointCore) que codifica a entrada em
um hipervetor discreto e esparso, organizado em grupos que compartilham
unidades segundo um grafo fixo (scaffold). O nucleo e avaliado em
classificacao de imagens naturais (MNIST, Fashion-MNIST) e medicas
(BloodMNIST, DermaMNIST, do MedMNIST v2) sob um protocolo unico de validacao
cruzada k-fold. O Glue (glue.py) alinha os codigos discretos de duas redes
treinadas independentemente.

Fluxo principal para reproduzir os resultados do artigo:

    pip install -r requirements.txt
    pip install medmnist
    python validation.py --suite all      # treina e avalia todas as configs
    python report.py                      # gera tabelas LaTeX e figuras
    python make_samples_figure.py paper/figs/dataset_samples.pdf

Ambiente de referencia: Python 3.10+, PyTorch 2.x com CUDA, GPU RTX 3060
(a suite "all" leva cerca de 2 h nesse hardware).


2. DADOS
--------------------------------------------------------------------------------
Nenhum dado bruto e versionado: todos os datasets sao publicos e baixados
automaticamente na primeira execucao para o diretorio ./data/.

  MNIST, Fashion-MNIST   via torchvision.datasets (download=True)
  BloodMNIST, DermaMNIST via pacote medmnist (MedMNIST v2, Yang et al., 2023),
                         resolucao 28x28

datasets.py: carregador unificado. load_dataset(name, sample_cap, seed) une os
  splits oficiais, retorna imagens, rotulos, nomes de classes e formato, e
  aplica subamostragem estratificada ate sample_cap amostras (17.000 no
  protocolo), para igualar o orcamento de treino entre datasets.


3. CODIGO-FONTE - NUCLEO DA ARQUITETURA
--------------------------------------------------------------------------------
base.py: arquitetura base. SJConfig (hiperparametros), make_rigid_scaffold /
  permute_scaffold (grafo esparso de grupos e unidades), SparseJointCore
  (papeis e fillers em FHRR; propagacao por camadas, bind com papeis, mensagens unarias e de par,
  quantizacao por codebook, programa final), SparseJointNetwork /
  build_model (encoder + core + cabeca), rotinas de treino, encode e
  vsa_cleanup_accuracy.

sj_invariants.py: funcoes auxiliares code_probabilities e full_path_locality,
  usadas nos diagnosticos e nas features do Glue.

glue.py: mecanismo de composicao. SJGlue (traduz, mescla e compoe programas
  entre as algebras de duas redes), fit_sj_glue / fit_operator_sj_glue
  (estimam o alinhamento de grupos e de valores por operadores de acao, ou
  seja, a partir de como os codigos de cada rede reagem as mesmas
  transformacoes da entrada, sem pares rotulados) e certify_sj_glue (valida
  o certificado: mapa de grupos injetivo e mapas de filler como
  permutacoes).

plots.py: figuras da bateria k-fold - matriz de confusao, grade de predicoes
  de exemplo, metricas por fold e curva de trade-off da penalidade de
  redundancia.


4. CODIGO-FONTE - PROTOCOLO DE VALIDACAO (resultados do artigo)
--------------------------------------------------------------------------------
validation.py: protocolo unico de treino e teste usado no artigo.
  Arquitetura fixa (BASE_ARCH): groups=8, card=16, units=128, layers=3,
    topk=4, hv_dim=256, frame_embed=128, unit_width=8.
  Treino (BASE_TRAIN): epochs=120, batch=1024, lr=2e-3 (AdamW).
  Validacao: StratifiedKFold com 5 folds, seed=0 (mesmas particoes para
    todas as configuracoes do mesmo dataset), padronizacao ajustada apenas
    no fold de treino, SAMPLE_CAP=17000.
  Eixos de ablacao: encoder (linear | cnn), core (sparse_joint | none,
    controle sem o gargalo VSA) e redundancy_weight (lambda da penalidade
    de redundancia entre grupos).
  Parametros:
    --suite   smoke | generality | encoder | specialization | all
              smoke          checagem rapida (BloodMNIST, 2 folds, 2000 amostras)
              generality     mesmo core em MNIST, Fashion-MNIST, BloodMNIST,
                             DermaMNIST
              encoder        linear vs CNN vs sem core (Fashion-MNIST, BloodMNIST)
              specialization lambda em {0, 0.1, 0.25, 0.5, 1.0}
                             (Fashion-MNIST, BloodMNIST)
              all            uniao das tres suites acima, sem repeticao
    --only    filtra configuracoes por substring do nome (ex.: --only bloodmnist)
    --force   reexecuta configuracoes que ja possuem resultado
  Saida: results/validation/<config>.json, um arquivo por configuracao
    (nome = <dataset>_<encoder>[_sem-core][_w<lambda>]), contendo metricas por
    fold, agregado (media, desvio), descricao do dataset e proveniencia
    (commit, estado da arvore, versao do torch, GPU, timestamp). O progresso
    e as metricas de cada configuracao sao impressos no terminal.

report.py: converte os JSON de results/validation/ nas tabelas e figuras do
  artigo, incluindo comparacoes pareadas por fold (diferenca, Cohen's d,
  Wilcoxon) e os baselines publicados do MedMNIST v2.
  Parametros: --latex-only (gera apenas as tabelas).
  Saida em results/validation/paper/:
    table_generality.tex, table_encoder.tex, table_specialization.tex,
    table_stats.tex, fig_generality.pdf, fig_tradeoff.pdf
  Esses arquivos sao copiados para ./paper/ para compilar o artigo.

kfold_medmnist_test.py: bateria k-fold individual (metricas de classificacao
  e de arquitetura: NMI por grupo, redundancia entre grupos, ganho do par,
  uso do alfabeto, perplexidade, recuperacao VSA). Fornece as funcoes de
  treino e metricas reutilizadas pelo validation.py.
  Parametros: --dataset (padrao bloodmnist), --folds (5), --epochs (25),
    --groups (8), --card (0 = max(16, n_classes)), --units (128),
    --layers (3), --topk (4), --hv-dim (256), --frame-embed (128),
    --batch (256), --lr (2e-3), --seed (0), --limit (subamostra, 0 = tudo),
    --encoder (linear | cnn), --redundancy-weight (0.0), --locality,
    --out (padrao results/kfold/<dataset>).
  Saida: metrics.json, report.txt, confusion_matrix.png,
    sample_predictions.png, per_fold_metrics.png.

test_sj_invariants.py: 15 testes de contrato do sj_invariants.py.
  Uso: python test_sj_invariants.py


5. CODIGO-FONTE - EXPERIMENTOS DE GLUE
--------------------------------------------------------------------------------
glue_transfer_test.py: treina duas redes A e B independentes (mesmos dados,
  seeds diferentes) e mede se a cabeca de B le os codigos de A em tres
  condicoes: teto (B->B), sem cola e com cola (SJGlue.import_program).
  O alinhamento usa operadores de acao (transformacoes de imagem).
  Parametros principais: --dataset (bloodmnist), --encoder (cnn),
    --epochs (12), --groups (8), --card (8), --units (64), --layers (2),
    --topk (6), --hv-dim (128), --batch (256), --lr (2e-3),
    --redundancy-weight (0.0), --seed (0), --evidence (4000),
    --actions (geometric), --permutation-nulls (119),
    --bootstrap-replicates (16), --threads (2), --out (results/glue).
  Saida: <out>/<dataset>_<encoder>_<actions>_<tag>.json

kfold_glue_test.py: versao k-fold do teste anterior, com o controle adicional
  "fillers embaralhados" (isola o ganho do alinhamento de valores).
  Parametros principais: --dataset (bloodmnist), --folds (5),
    --encoder (cnn), --epochs (6), --batch (128), --actions (photometric),
    --seed (0), --evidence (1200), --permutation-nulls (39),
    --bootstrap-replicates (8), --out (results/kfold_glue); demais
    hiperparametros iguais ao glue_transfer_test.py.
  Saida: <out>/<stem>_report.txt e <out>/<stem>_metrics.json

RESULTADOS_GLUE.md: anotacoes organizadas dos resultados do Glue (dSprites,
  5 seeds): comparacao MLP vs SJ vs especialistas colados, por fator
  generativo, com media e desvio-padrao.


6. CODIGO-FONTE - FASE INICIAL (MNIST / Fashion-MNIST e MOVi)
--------------------------------------------------------------------------------
sj_image_classifier.py: modelo, laco de treino e relatorios dos primeiros
  testes de classificacao (CNN pequena -> SparseJointCore -> cabeca linear),
  com penalidade de redundancia entre grupos (REDUNDANCY_WEIGHT no topo do
  arquivo).

image_classification_test.py: executa o classificador em MNIST.
  Saida: results/mnist/ (confusion_matrix.png, sample_predictions.png,
  metrics.txt).

fashion_classification_test.py: executa o classificador em Fashion-MNIST.
  Saida: results/fashion_mnist/ (mesmos arquivos).

composed_nets.py: redes especialistas do cenario MOVi (VisionSJSpecialist,
  PhysicsSJSpecialist, OutcomeDecoderSJ) e utilitarios save_checkpoint /
  load_checkpoint, escritos sobre a API anterior do nucleo.

smoke_test.py: sanity check dos checkpoints MOVi com dados aleatorios
  (API anterior do nucleo).


7. MODELOS TREINADOS
--------------------------------------------------------------------------------
./movi_vision_sj_seed11.pkl   especialista de visao (MOVi), seed 11
./movi_physics_sj_seed11.pkl  especialista de fisica (MOVi), seed 11
./movi_decoder_sj_seed11.pkl  decodificador de desfecho (MOVi), seed 11
  Formato sj-model-checkpoint-v1: config, scaffold (mascaras), pesos e seed
  num unico arquivo; carregados com composed_nets.load_checkpoint.

Os modelos dos experimentos de classificacao sao treinados por fold dentro
de validation.py / kfold_medmnist_test.py, de forma deterministica a partir
das seeds registradas (seed do fold = seed + indice do fold).


8. RESULTADOS E LOGS
--------------------------------------------------------------------------------
./results/kfold/blood_linear/   BloodMNIST, encoder linear, lambda=0.0
./results/kfold/blood_cnn/      BloodMNIST, encoder CNN, lambda=0.0
./results/kfold/blood_cnn_w025/ BloodMNIST, encoder CNN, lambda=0.25
  Configuracao comum: 5 folds, groups=8, card=16, units=128, layers=3,
  topk=4, hv_dim=256, epochs=120, batch=1024, lr=2e-3, seed=0.
  Cada diretorio contem:
    report.txt              log do teste: configuracao, metricas de
                            classificacao e de arquitetura (media, desvio,
                            min, max), resultados e tempo de treino por fold
    metrics.json            dataset, folds, classes, config, resumo,
                            metricas por fold e matriz de confusao
    confusion_matrix.png    matriz de confusao normalizada
    sample_predictions.png  grade de predicoes de exemplo
    per_fold_metrics.png    metricas por fold

./results/validation/        saida do validation.py (ver secao 4)
./results/validation/paper/  saida do report.py (ver secao 4)


9. ARTIGO E FIGURAS
--------------------------------------------------------------------------------
./paper/article.tex      texto principal do artigo
./paper/methodology.tex  secao de metodologia (protocolo de validacao)
./paper/results.tex      secao de resultados; inclui as tabelas e figuras
                         geradas pelo report.py
./paper/standalone.tex   documento para compilar metodologia + resultados
                         isoladamente
./paper/bibtex.bib       referencias bibliograficas
./paper/figs/architecture.pdf         diagrama da arquitetura
./paper/figs/dataset_samples.pdf      amostras dos quatro datasets
./paper/figs/dataset_samples_all.pdf  variante com mais amostras

make_samples_figure.py: gera a figura de amostras dos datasets (uma linha
  por dataset, quatro classes por linha, 28x28, seed 0).
  Uso: python make_samples_figure.py [saida.pdf]
  (padrao: dataset_samples.pdf)


10. DOCUMENTACAO AUXILIAR
--------------------------------------------------------------------------------
README.md          descricao conceitual da arquitetura (scaffold, core,
                   Glue, checkpoints)
TESTES.md          descricao do protocolo de teste e validacao unificado
RESULTADOS_GLUE.md resultados dos experimentos de Glue
requirements.txt   dependencias (torch, numpy, torchvision, matplotlib,
                   scikit-learn, scipy); adicionalmente: medmnist
