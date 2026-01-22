#import "@preview/touying:0.6.1": *
#import themes.simple: *

#show: simple-theme.with(aspect-ratio: "16-9")
#set text(font: "Inria Sans", size: 21pt)

#show figure: X => {
  set align(center + horizon)
  X
  v(1em)
}


= EEG-based Classification of Cognitive Impairment

== Methodology

- Baseline with simple feature selection: Regional, vectors

- Baseline with simple methods: Random forest, SVM

- Evaluation pipeline: CV, recall, F1, binary classification

#v(2em)
- Improve

- Interpretability

== Feature extraction (1/3)

#figure(
  image("/experiments/plots/phase5/clustering_pca.png", width: 80%),
  caption: [PCA of true labels vs K-Means using the alpha band and regional feature selection (30 features). Silhouette=0.392, ARI=0.107],
)


== Feature extraction (2/3)
#figure(
  image("/experiments/plots/phase5/pca_vector_THETA.png", width: 80%),
  caption: [PCA of true labels vs K-Means using the theta band and vector feature selection. Silhouette=0.249, ARI=0.180],
)

== Feature extraction (3/3)

*Regional:* Mean on the rows: $30$ features.

*Vector:* Upper diagonal, $(30 times 29) / 2 = 435$ features.

== Model selection
#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    inset: 10pt,
    align: center,
    gutter: 0.5em,
    stroke: none,
    [*Strategy*], [*Band*], [*Selection*], [*Model*], [*Balanced Accuracy*],
    table.hline(stroke: 0.1em),
    [Regional], [Alpha], [ANOVA (k=20)], [Random Forest], [*58.22%*],
    [Vector], [Beta], [ANOVA (k=20)], [Random Forest], [56.89%],
    [Regional], [Alpha], [None], [Random Forest], [52.78%],
  ),
  caption: [Classification performance (nested CV)],
)

== Evaluation (1/2)

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    inset: 10pt,
    align: center,
    gutter: 0.5em,
    stroke: none,
    [*Strategy*], [*Band*], [*Selection*], [*Model*], [*Accuracy*],
    table.hline(stroke: 0.1em),
    [Regional], [Alpha], [ANOVA (k=10)], [Random Forest], [*65.56%*],
    [Vector], [Alpha], [None], [SVM], [61.11%],
    [Vector], [Alpha], [ANOVA (k=50)], [Random Forest], [61.11%],
  ),
  caption: [Classification performance with bad evaluation],
)

_ANOVA used on train and test set, high variance._

== Balanced accuracy

$
  "Accuracy" = "Correct" / "Total"
$

$
  "Balanced accuracy" = 1/3 ("Correct MCI" / "Total MCI" + "Correct SCI" / "Total SCI" + "Correct AD" / "Total AD")
$

== Evaluation (2/2)

Metrics:
- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1

Methods:
- LOO
- Mean, std

== Interpretability
#figure(
  image(
    "experiments/plots/interpretability/feature_importance.png",
    height: 72%,
  ),
  caption: [Most important electrodes when looking at the alpha band seems to be at the front.],
)

== Binary classification: AD vs SCI+MCI

- *Accuracy:* *71.56%*
- *Balanced Accuracy:* 58.67%

$->$ _Continuous transition between SCI and MCI?_

== Next
=== Improvements
- Test graph metrics.
- Test late fusion.
- Look for better feature selection.

=== Interpretability
- Graphs
- Investigate MCI heterogeneity.


== Failed experiments

- Use a sample of 28 SCI instead of all 40: on RF with the regional features and alpha band, it performed worse (51% acc vs 62%).
- Mutual information feature selection performs worse than ANOVA: on RF with the regional features and alpha band (53% acc vs 62%).
- Tree-based feature selection performs worse than ANOVA: on RF with the regional features and alpha band (60% vs 62%).
