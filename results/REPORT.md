# VoiceMood Results

## Headline metrics

| Feature | Model | Accuracy | 95% CI | Macro F1 | Brier |
|---|---|---|---|---|---|
| mfcc | logistic_regression | 0.674 | [0.618, 0.726] | 0.668 | 0.509 |
| mfcc | random_forest | 0.611 | [0.555, 0.670] | 0.594 | 0.640 |
| mfcc | svm_rbf | 0.719 | [0.663, 0.767] | 0.709 | 0.410 |
| wav2vec2 | logistic_regression | 0.698 | [0.649, 0.747] | 0.691 | 0.440 |
| wav2vec2 | random_forest | 0.483 | [0.424, 0.545] | 0.467 | 0.698 |
| wav2vec2 | svm_rbf | 0.653 | [0.594, 0.715] | 0.649 | 0.466 |
| yamnet | logistic_regression | 0.493 | [0.434, 0.552] | 0.485 | 0.771 |
| yamnet | random_forest | 0.403 | [0.344, 0.462] | 0.397 | 0.722 |
| yamnet | svm_rbf | 0.531 | [0.476, 0.587] | 0.526 | 0.610 |

## Pairwise McNemar tests (within feature group)

| Group | Model A | Model B | b (A only) | c (B only) | chi² | p |
|---|---|---|---|---|---|---|
| mfcc | mfcc__logistic_regression | mfcc__random_forest | 56 | 38 | 3.074 | 0.0795 |
| mfcc | mfcc__logistic_regression | mfcc__svm_rbf | 36 | 49 | 1.694 | 0.1931 |
| mfcc | mfcc__random_forest | mfcc__svm_rbf | 19 | 50 | 13.043 | 0.0003 |
| wav2vec2 | wav2vec2__logistic_regression | wav2vec2__random_forest | 79 | 17 | 38.760 | 0.0000 |
| wav2vec2 | wav2vec2__logistic_regression | wav2vec2__svm_rbf | 39 | 26 | 2.215 | 0.1366 |
| wav2vec2 | wav2vec2__random_forest | wav2vec2__svm_rbf | 7 | 56 | 36.571 | 0.0000 |
| yamnet | yamnet__logistic_regression | yamnet__random_forest | 59 | 33 | 6.793 | 0.0091 |
| yamnet | yamnet__logistic_regression | yamnet__svm_rbf | 34 | 45 | 1.266 | 0.2606 |
| yamnet | yamnet__random_forest | yamnet__svm_rbf | 16 | 53 | 18.783 | 0.0000 |

## Confusion matrices

### mfcc__logistic_regression

![confusion](plots\confusion_mfcc__logistic_regression.png)

### mfcc__random_forest

![confusion](plots\confusion_mfcc__random_forest.png)

### mfcc__svm_rbf

![confusion](plots\confusion_mfcc__svm_rbf.png)

### wav2vec2__logistic_regression

![confusion](plots\confusion_wav2vec2__logistic_regression.png)

### wav2vec2__random_forest

![confusion](plots\confusion_wav2vec2__random_forest.png)

### wav2vec2__svm_rbf

![confusion](plots\confusion_wav2vec2__svm_rbf.png)

### yamnet__logistic_regression

![confusion](plots\confusion_yamnet__logistic_regression.png)

### yamnet__random_forest

![confusion](plots\confusion_yamnet__random_forest.png)

### yamnet__svm_rbf

![confusion](plots\confusion_yamnet__svm_rbf.png)


## Calibration

### mfcc__logistic_regression

![calibration](plots\calibration_mfcc__logistic_regression.png)

### mfcc__random_forest

![calibration](plots\calibration_mfcc__random_forest.png)

### mfcc__svm_rbf

![calibration](plots\calibration_mfcc__svm_rbf.png)

### wav2vec2__logistic_regression

![calibration](plots\calibration_wav2vec2__logistic_regression.png)

### wav2vec2__random_forest

![calibration](plots\calibration_wav2vec2__random_forest.png)

### wav2vec2__svm_rbf

![calibration](plots\calibration_wav2vec2__svm_rbf.png)

### yamnet__logistic_regression

![calibration](plots\calibration_yamnet__logistic_regression.png)

### yamnet__random_forest

![calibration](plots\calibration_yamnet__random_forest.png)

### yamnet__svm_rbf

![calibration](plots\calibration_yamnet__svm_rbf.png)


_Generated in 111.9 s._
