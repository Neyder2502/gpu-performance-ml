# Predicción del Rendimiento de GPUs (FP32) mediante Machine Learning

## Integrantes

* Neyder Fabián Peña González  Codigo: 2252004
* Julio Cesar Sierra Jimenez   Codigo: 2251503
* Samuel Mateo Chacon Ojeda    Codigo: 2251508

Este proyecto busca predecir y clasificar el rendimiento de tarjetas gráficas (GPUs) basándose en sus especificaciones técnicas de hardware

El objetivo principal es clasificar el rendimiento en precisión simple (`fp32_performance`) en las categorías Bajo, Medio y Alto,  permitiendo estimar la potencia de un hardware

## Dataset

Los datos fueron obtenidos de Kaggle: 
[GPU Performance Prediction](https://www.kaggle.com/datasets/psychocoderxd/gpu-performance-prediction).

* **Registros iniciales:** 10,000 filas, 19 columnas.
* **Variables destacadas:** `cores`, `tensor_cores`, `rt_cores`, `base_clock`, `boost_clock`, `ram`, `fp32_performance` (Target), entre otras

## Metodología y Flujo de Trabajo

El proyecto se desarrolló en un Jupyter Notebook estructurado en las siguientes fases:

1. **Análisis Exploratorio (EDA):**
   * Identificación de variables numéricas y nulas
   * Análisis de distribución del target
   * Análisis de correlación: Se detectó alta multicolinealidad entre `cores`, `rt_cores` y `fp16_performance`, y una fuerte correlación positiva con el target
   * **Outliers:** Se decidió **no eliminarlos**, ya que representan GPUs de gama alta legítimas y de interés para el modelo

2. **Limpieza de Datos:**
   * Eliminación de columnas vacías o casi vacías (`bf16_performance`, `tf32_performance`, `int8_performance`, `multi_gpu`)
   * Eliminación de filas con el target nulo
   * **Imputación multivariada:** Uso de `IterativeImputer` (MICE) para predecir y rellenar valores faltantes en las features restantes, preservando las relaciones matemáticas entre variables.

3. **Modelado (Clasificación):**
   Se plantearon dos enfoques:
   * **Binario:** GPUs de bajo rendimiento (≤ mediana) vs. alto rendimiento (> mediana).
   * **Multiclase:** GPUs de bajo (≤ p50), medio (p50 - p85) y alto (> p85) rendimiento.

   Se evaluaron 4 algoritmos con búsqueda paso a paso de hiperparámetros y validación cruzada (Train/Val/Test 60/20/20):
   * Gaussian Naive Bayes (GNB)
   * Decision Tree
   * Random Forest
   * Support Vector Machine (SVM)

4. **Evaluación:**
   * Uso de métricas robustas: Accuracy, Precision, Recall y F1-Score.
   * Análisis de Overfitting/Underfitting mediante la brecha (Gap) entre Train y Validation.
   * Matrices de confusión y curvas de aprendizaje.

## Resultados Principales

Los modelos basados en árboles dominaron el problema, alcanzando métricas casi perfectas:

* **Random Forest** y **Decision Tree** lograron un **F1-Score superior a 0.99** en validación y test, tanto en el enfoque binario como en el multiclase.
* **SVM** (con kernel RBF) obtuvo un F1-Score cercano a 0.99, pero con mayor costo computacional.
* **Naive Bayes** fue el modelo con peor desempeño (F1 ~ 0.89), afectado principalmente por la multicolinealidad entre las variables, lo cual viola su supuesto de independencia condicional.

## Tecnologías Utilizadas

* **Lenguaje:** Python 3
* **Manipulación de Datos:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Visualización:** Matplotlib, Seaborn
* **Entorno:** Google Colab / Jupyter Notebook
