# Random Forest
  Best Params: {'classifier__max_depth': None, 'classifier__max_features': 0.3, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 8, 'classifier__n_estimators': 500}

  accuracy                 : 0.9655 ± 0.0007
  balanced_accuracy        : 0.9554 ± 0.0010
  f1_weighted              : 0.9655 ± 0.0007
  f1_macro                 : 0.9571 ± 0.0010



# KNN

 Best Params: {'classifier__n_neighbors': 20, 'classifier__p': 2, 'classifier__weights': 'distance', 'pca__n_components': 30, 'scaler__scaler_op': StandardScaler()}

  accuracy                 : 0.4110 ± 0.0018
  balanced_accuracy        : 0.3117 ± 0.0024
  f1_weighted              : 0.3937 ± 0.0020
  f1_macro                 : 0.3226 ± 0.0028


# SVM

  Best Params: {'classifier__C': 100, 'classifier__gamma': 'scale', 'classifier__kernel': 'rbf', 'scaler__scaler_op': StandardScaler()}

  accuracy                 : 0.6356 ± 0.0089
  balanced_accuracy        : 0.5913 ± 0.0082
  f1_weighted              : 0.6319 ± 0.0083
  f1_macro                 : 0.5970 ± 0.0079


# FFNN

  Accuracy          : 0.9237
  Balanced Accuracy : 0.8889
  F1 (weighted)     : 0.9238
  F1 (macro)        : 0.8841

              precision    recall  f1-score   support

     Grade A       0.94      0.97      0.96      2654
     Grade B       0.96      0.93      0.94      3774
     Grade C       0.95      0.95      0.95      3708
     Grade D       0.94      0.92      0.93      2126
     Grade E       0.88      0.87      0.87      1208
     Grade F       0.76      0.71      0.74       753
     Grade G       0.74      0.86      0.80       608

    accuracy                           0.92     14831
   macro avg       0.88      0.89      0.88     14831
weighted avg       0.92      0.92      0.92     14831



# TABNET

  Accuracy          : 0.9284
  Balanced Accuracy : 0.9016
  F1 (weighted)     : 0.9282
  F1 (macro)        : 0.9000

              precision    recall  f1-score   support

     Grade A       0.94      0.98      0.96      2654
     Grade B       0.97      0.92      0.94      3774
     Grade C       0.94      0.96      0.95      3708
     Grade D       0.92      0.93      0.93      2126
     Grade E       0.90      0.88      0.89      1208
     Grade F       0.82      0.76      0.79       753
     Grade G       0.81      0.88      0.85       608

    accuracy                           0.93     14831
   macro avg       0.90      0.90      0.90     14831
weighted avg       0.93      0.93      0.93     14831




#  TabTransformer Test Set Evaluation
  Accuracy          : 0.8602
  Balanced Accuracy : 0.8036
  F1 (weighted)     : 0.8609
  F1 (macro)        : 0.7912

              precision    recall  f1-score   support

     Grade A       0.94      0.93      0.94      2654
     Grade B       0.91      0.91      0.91      3774
     Grade C       0.92      0.89      0.90      3708
     Grade D       0.84      0.85      0.85      2126
     Grade E       0.74      0.71      0.72      1208
     Grade F       0.55      0.50      0.52       753
     Grade G       0.59      0.83      0.69       608

    accuracy                           0.86     14831
   macro avg       0.79      0.80      0.79     14831
weighted avg       0.86      0.86      0.86     14831
