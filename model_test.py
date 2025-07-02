import joblib
import pandas as pd
import numpy as np

# Cargar el modelo guardado
modelo_cargado = joblib.load('modelo_titanic_vc2.pkl')
print("Modelo cargado exitosamente.")

# 🟢 Función para transformar los datos del usuario
def transformar_datos(input_data):
    df = pd.DataFrame([input_data])  # Convertir JSON en DataFrame

    # 🟢 Crear Family_Size
    df['Family_Size'] = df['SibSp'] + df['Parch'] + 1  

    # 🟢 Mapear Family_Size_Grouped
    family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 
                  6: 'Medium', 7: 'Large', 8: 'Large', 9: 'Large', 11: 'Large'}
    df['Family_Size_Grouped'] = df['Family_Size'].map(family_map).fillna('Small')  

    # 🟢 Convertir Fare a Fare_cut
    bins = [-0.001, 7.775, 8.662, 14.454, 26, 52.369, 512.329, np.inf]
    labels = [0, 1, 2, 3, 4, 5, 6]
    df['Fare_cut'] = pd.cut(df['Fare'], bins=bins, labels=labels, include_lowest=True)

    # 🟢 Convertir Name_Length a Name_LengthGB
    bins_name = [0, 12, 18, 20, 23, 30, 38, np.inf]
    labels_name = [0, 1, 2, 3, 4, 5, 6]
    df['Name_LengthGB'] = pd.cut(df['Name_Length'], bins=bins_name, labels=labels_name, include_lowest=True)

    # 🟢 Convertir variables categóricas a numéricas
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).fillna(2)  

    # 🟢 Agregar las columnas que faltan con valores por defecto
    df['Name_Size'] = df['Name_Length'] // 5  # Ejemplo de asignación (ajustar si es necesario)
    df['Cabin_Assigned'] = 0  # Si el usuario no ingresa Cabina, asumimos que no tiene
    df['TicketNumberCounts'] = 1  # Asumimos que el ticket es único

    # 🟢 Incluir 'Fare' porque el modelo la necesita
    df['Fare'] = df['Fare']  

    # Seleccionar las columnas finales que el modelo espera
    columnas_finales = ['Pclass', 'Sex', 'Age', 'Fare', 'Fare_cut', 'Embarked', 'Family_Size_Grouped', 
                        'Name_LengthGB', 'Name_Size', 'Cabin_Assigned', 'TicketNumberCounts']
    df_final = df[columnas_finales]

    return df_final

# 🔵 Ejemplo de datos de usuario
input_usuario = {
    "Pclass": 3,
    "Sex": "female",
    "Age": 51,
    "SibSp": 1,
    "Parch": 3,
    "Fare": 30,
    "Embarked": "Q",
    "Name_Length": 19
}

# 🔵 Transformar los datos del usuario
df_transformado = transformar_datos(input_usuario)
print("\n📌 Datos transformados para el modelo:\n", df_transformado)

# 🔵 Hacer la predicción
prediccion = modelo_cargado.predict(df_transformado)
print("\n🔮 Predicción del modelo:", "Sobrevive" if prediccion[0] == 1 else "No sobrevive")
