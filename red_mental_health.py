import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import messagebox


df = pd.read_csv("Mental_Health_and_Social_Media_Balance_Dataset.csv")



features = [
    "Age",
    "Daily_Screen_Time(hrs)",
    "Sleep_Quality(1-10)",
    "Stress_Level(1-10)",
    "Days_Without_Social_Media",
    "Exercise_Frequency(week)"
]

target = "Happiness_Index(1-10)"

X = df[features].values
y = df[target].values


scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo = Sequential()
modelo.add(Dense(10, activation='relu', input_dim=X_train.shape[1]))
modelo.add(Dense(5, activation='relu'))
modelo.add(Dense(1))  # salida

modelo.compile(optimizer='adam', loss='mean_squared_error')

modelo.fit(X_train, y_train, epochs=150, batch_size=8, verbose=1)


ventana = tk.Tk()
ventana.title("Predicción de Felicidad - Red Neuronal")
ventana.geometry("350x350")

labels = [
    "Edad",
    "Horas de pantalla",
    "Calidad del sueño (1-10)",
    "Estrés (1-10)",
    "Días sin redes",
    "Ejercicio por semana"
]

entries = []

for i, label in enumerate(labels):
    tk.Label(ventana, text=label, font=("Arial", 12)).pack()
    entry = tk.Entry(ventana, font=("Arial", 12))
    entry.pack()
    entries.append(entry)

def predecir():
    try:
        valores = [float(entry.get()) for entry in entries]
        valores_scaled = scaler.transform([valores])
        pred = modelo.predict(valores_scaled)[0][0]
        messagebox.showinfo("Resultado", f"Índice de Felicidad estimado: {pred:.2f}")

    except ValueError:
        messagebox.showerror("Error", "Por favor ingrese solo números válidos.")

tk.Button(
    ventana,
    text="Predecir Felicidad",
    font=("Arial", 14),
    command=predecir,
    bg="lightblue"
).pack(pady=15)

ventana.mainloop()
