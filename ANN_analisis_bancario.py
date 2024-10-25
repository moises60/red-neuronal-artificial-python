import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# 1. Cargar el dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# 2. Eliminar columnas irrelevantes
dataset = dataset.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# 3. Codificar variables categóricas
dataset = pd.get_dummies(dataset, columns=['Geography', 'Gender'], drop_first=True)

# 4. Separar características y variable objetivo
X = dataset.drop('Exited', axis=1)
y = dataset['Exited']

# 5. Dividir el dataset en conjunto de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 6. Escalado de características
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                    'EstimatedSalary']

X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# 7. Construir la Red Neuronal Artificial
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Inicializar el modelo
model = Sequential()

# Añadir capas
# Capa de entrada y primera capa oculta
model.add(Dense(units=64, kernel_initializer='he_uniform', activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.3))

# Segunda capa oculta
model.add(Dense(units=32, kernel_initializer='he_uniform', activation='relu'))
model.add(Dropout(0.3))

# Tercera capa oculta
model.add(Dense(units=16, kernel_initializer='he_uniform', activation='relu'))
model.add(Dropout(0.2))

# Capa de salida
model.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Implementar EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 8. Entrenar el modelo
history = model.fit(X_train, y_train, validation_split=0.1, batch_size=32, epochs=100, callbacks=[early_stopping])

# 9. Evaluar el modelo
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.30)

# Matriz de confusión y métricas de clasificación
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión:')
print(cm)
print('\nInforme de Clasificación:')
print(classification_report(y_test, y_pred))

# 10. Plotear la Matriz de Confusión con Seaborn
# Definir etiquetas
etiquetas = ['No Exited', 'Exited']

# Crear un DataFrame para la matriz de confusión
cm_df = pd.DataFrame(cm, index=etiquetas, columns=etiquetas)

plt.figure(figsize=(8,6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')

plt.title('Matriz de Confusión')
plt.ylabel('Clase Real')
plt.xlabel('Clase Predicha')
plt.show()


# 11. Visualizar la pérdida y precisión
plt.figure(figsize=(12,4))

# Pérdida
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

# Precisión
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.show()
