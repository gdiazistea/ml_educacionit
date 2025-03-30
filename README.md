# Curso de Machine Learning - EducaciónIT

Este repositorio contiene apuntes, prácticas y material del curso de **Machine Learning** dictado en **EducaciónIT**. Se cubren desde fundamentos estadísticos hasta técnicas avanzadas de aprendizaje profundo y despliegue de modelos.

![Machine Learning](ml.jpg)

## Contenidos

### Fundamentos de Probabilidad y Estadística
- Probabilidad y Estadística.
- Variables Aleatorias y Distribuciones de Probabilidad.
- Asimetría y Curtosis.
- Población y Muestra.
- Correlación entre variables.
- Regresión Lineal.

### Modelos de Regresión y Clasificación
- K-Vecinos más cercanos (KNN).
- Regresión Polinómica y Logística.
- Árboles de decisión para regresión y clasificación.
- Evaluación de modelos: Underfitting, Overfitting, Validación cruzada.
- Matriz de Confusión, Curva ROC y Scores.
- Optimización de Hiperparámetros.
- Naive Bayes y Teorema de Bayes.

### Modelos Avanzados
- Sesgo y Varianza.
- Support Vector Machines (SVM).
- Ensambles de Modelos.

### Redes Neuronales y Deep Learning
- Redes Neuronales (Perceptrón, Multicapa).
- Descenso por Gradiente y funciones de activación.
- Backpropagation y Forward propagation.
- Regularización en Redes Neuronales.

### Redes Neuronales Avanzadas
- Redes Neuronales Recurrentes (RNN, LSTM).
- Redes Neuronales Convolucionales (CNN) y Visión por Computador.
- Neural Style Transfer y Redes Neuronales Adversarias (GANs).

### Procesamiento del Lenguaje Natural (PLN)
- Normalización de texto, Expresiones Regulares.
- Vectorización: Bag of Words, TF-IDF.
- Redes Transformer.
- Aprendizaje No Supervisado: Clustering (K-Means, DBSCAN).

### Reducción de Dimensionalidad y Deploy
- PCA, SVD.
- Sistemas de Recomendación.
- Publicación de Modelos y Pipelines.

## Requisitos
- Python 3.x
- Numpy, Pandas, Scikit-learn
- TensorFlow / PyTorch
- OpenCV, Matplotlib, Seaborn

## Instalación
```bash
# Clonar el repositorio
git clone https://github.com/gdiazistea/ml_educacionit.git
cd repo-machinelearning

# Crear un entorno virtual e instalar dependencias
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Uso
Los notebooks y scripts explicativos están organizados en carpetas según los módulos. 
Revisa `nbs/` para explorar ejemplos y ejercicios.
