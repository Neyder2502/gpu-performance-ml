# Predicción del Rendimiento de GPUs (FP32) mediante Machine Learning

**Autores:** Neyder Fabián Peña González, Julio Cesar Sierra Jimenez, Samuel Mateo Chacon Ojeda

## Objetivo

Clasificar el rendimiento en precisión simple (`fp32_performance`) de tarjetas gráficas en las categorías Bajo, Medio y Alto a partir de sus especificaciones técnicas de hardware.

## Dataset

GPU Performance Prediction (Kaggle), 10,000 registros y 19 columnas: [Descargar dataset](https://www.kaggle.com/datasets/psychocoderxd/gpu-performance-prediction)

## Modelos

**Algoritmos de clasificación:**
* Gaussian Naive Bayes
* Decision Tree
* Random Forest
* Support Vector Machine (SVM), kernel RBF

**Enfoques de clasificación:**
* Binario (bajo / alto)
* Multiclase (bajo / medio / alto)

**Preprocesamiento:**
* IterativeImputer (MICE)
* Análisis de correlación y multicolinealidad
* Manejo de outliers (conservados)

**Entrenamiento y validación:**
* Partición Train/Val/Test (60/20/20)
* Validación cruzada
* Búsqueda de hiperparámetros paso a paso

**Evaluación:**
* Accuracy, Precision, Recall, F1-Score
* Matriz de confusión
* Curvas de aprendizaje
* Análisis de overfitting (brecha Train-Validation)

## Enlaces

* Código (Colab): https://colab.research.google.com/drive/1U0GxtPr790DwN1rrcr_f7mW7ZT6TOT3D?usp=sharing
* Repositorio: https://github.com/Neyder2502/gpu-performance-ml

---

## Resultados Principales

* **Random Forest** y **Decision Tree** lograron un **F1-Score superior a 0.99** en validación y test, tanto en el enfoque binario como en el multiclase.
* **SVM** (kernel RBF) obtuvo un F1-Score cercano a 0.99, con mayor costo computacional.
* **Naive Bayes** fue el de peor desempeño (F1 ~ 0.89), afectado por la multicolinealidad entre variables, que viola su supuesto de independencia condicional.

## Tecnologías Utilizadas

* **Lenguaje:** Python 3
* **Manipulación de datos:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Visualización:** Matplotlib, Seaborn
* **Entorno:** Google Colab / Jupyter Notebook
