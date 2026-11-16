# Predicción del Rendimiento de GPUs (FP32) mediante Machine Learning

**Autores:** Neyder Fabián Peña González, Julio Cesar Sierra Jimenez, Samuel Mateo Chacon Ojeda

## Objetivo

Clasificar y predecir el rendimiento FP32 (TFLOPS) de GPUs a partir de sus especificaciones técnicas.

## Dataset

GPU Performance Prediction (Kaggle), 10,000 registros y 19 columnas: [Descargar dataset](https://www.kaggle.com/datasets/psychocoderxd/gpu-performance-prediction)

## Modelos

**Clasificación:**
* Gaussian Naive Bayes
* Decision Tree
* Random Forest
* Support Vector Machine (SVM), kernels: linear, RBF, polynomial

**Enfoques de clasificación:**
* Binario (bajo / alto)
* Multiclase (bajo / medio / alto)

**Regresión:**
* DecisionTreeRegressor
* RandomForestRegressor
* SVR (Support Vector Regression), kernels: linear, RBF, polynomial, sigmoid

**Hiperparámetros ajustados:**
* var_smoothing (Naive Bayes)
* max_depth, min_samples_split, min_samples_leaf (Decision Tree)
* n_estimators, max_depth, min_samples_leaf (Random Forest)
* kernel, C, gamma (SVM / SVR)

**Preprocesamiento:**
* IterativeImputer (MICE)
* StandardScaler
* Pipeline
* Análisis de correlación y multicolinealidad
* Manejo de outliers (conservados)

**Entrenamiento y validación:**
* Partición Train/Val/Test (60/20/20)
* Tuneo paso a paso (un hiperparámetro a la vez)
* Filtro de overfitting (gap Train-Validation)
* Validación cruzada (10 folds)
* Curvas de aprendizaje

**Evaluación de clasificación:**
* Accuracy, Precision, Recall, F1-Score
* Matriz de confusión

**Evaluación de regresión:**
* MSE, RMSE, MAE, R²
* Real vs. predicho
* Análisis de residuos
* Importancia de variables

## Enlaces

* Código (Colab): https://colab.research.google.com/drive/1U0GxtPr790DwN1rrcr_f7mW7ZT6TOT3D?usp=sharing
* Repositorio: https://github.com/Neyder2502/gpu-performance-ml

## Tecnologías Utilizadas

* **Lenguaje:** Python 3
* **Manipulación de datos:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Visualización:** Matplotlib, Seaborn
* **Entorno:** Google Colab / Jupyter Notebook
