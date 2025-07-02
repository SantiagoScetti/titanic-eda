"""
# Análisis del Titanic

Este script analiza el conjunto de datos del Titanic siguiendo la metodología CRISP-DM.
## Pasos:
1. Entendimiento del negocio
2. Entendimiento de los datos
3. Preparación de los datos
4. Modelado
5. Evaluación del modelo y visualización de datos
6. Despliegue
"""

# =============================================
# Importamos las Librerías
# =============================================
import pandas as pd                         # para manipulación de datos
import matplotlib.pyplot as ptl                 # data visualization
import seaborn as sns                           # data visualization
from sklearn.ensemble import RandomForestClassifier # modelo
from sklearn.impute import SimpleImputer   # Imputación de valores nulos
from sklearn.preprocessing import OrdinalEncoder # Codificación ordinal


# =============================================
# 2. ENTENDIMIENTO DE LOS DATOS
# =============================================

"""
## Leer y explorar los datos
- Carga de los conjuntos de entrenamiento y prueba.
- Exploración inicial del dataset para entender su estructura.
"""

## Cargar datasets
df_train=pd.read_csv('data/train.csv')
df_test=pd.read_csv('data/test.csv')

## Vista previa del conjunto de entrenamiento
print("Vista previa del conjunto de entrenamiento:")
print(df_train.sample(5))  # Muestra aleatoria de 5 filas

## Información general sobre el dataset
print("\nInformación del dataset de entrenamiento:")
print(df_train.info())

## ver duplicados
print("\nElementos duplicados:")
print(df_train.duplicated().sum())

## ver nulos
print("\nElementos nulos:")
print(df_train.isnull().sum().sort_values(ascending=False))


## ver valores unicos
print("\nElementos Únicos:")
print(df_train.nunique())



# =============================================
# Exploración de columnas categóricas y numéricas
# =============================================

"""
## Valores únicos en columnas categóricas y numéricas
- Exploración de valores únicos en columnas con menos de 10 valores distintos.
"""

# Columnas categóricas
columnas_categoricas=df_train.select_dtypes(include=['object']).columns
print("\n### Columnas categóricas:")
for column in columnas_categoricas:
    if df_train[column].nunique() <=10:
        print(f"{column}: {df_train[column].unique()}")


# Columnas numéricas
columnas_numericas = df_train.select_dtypes(include=['int64', 'float64']).columns
print("\n### Columnas numéricas:")
for column in columnas_numericas:
    if df_train[column].nunique() <= 10:
        print(f"{column}: {df_train[column].unique()}")


# =============================================
# ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =============================================

"""
## Análisis Exploratorio
- Visualización de las distribuciones de las variables de interés.
"""

# Distribución de la variable objetivo
sns.countplot(x='Survived', data=df_train)
ptl.title("Distribución de la variable objetivo: Supervivencia")
ptl.show()

# Relación entre 'Sex' y 'Survived'
sns.barplot(x='Sex', y='Survived', data=df_train)
ptl.title("Supervivencia por sexo")
ptl.show()

# =============================================
# CONCLUSIONES
# =============================================

"""
## Conclusiones preliminares
- Resumen de observaciones clave tras el análisis exploratorio.

° PassengerId, Cabin, Fare, Ticket y Name no aportan a la predicción -----> eliminar

"""
df_train = df_train.drop(columns=['Cabin', 'Fare', 'Ticket', 'Name'])
df_test = df_test.drop(columns=['Cabin', 'Fare', 'Ticket', 'Name'])
df_train.head()


#Preparación de los datos (3er paso)

## Separación de predictoras y targer
X = df_train.drop(['Survived'], axis= 1)
y = df_train.Survived

## pasar columnas categóricas(objetos) a numéricas(primitive Data Type)
s= (X.dtypes == 'object')
object_cols = list(s[s].index)

ordinal_encoder = OrdinalEncoder()
X[object_cols] = ordinal_encoder.fit_transform(X[object_cols])

print(X.head())

# Rellenamos valores nulos
imputer = SimpleImputer()
x_transformed= pd.DataFrame(imputer.fit_transform(X))
x_transformed.columns = X.columns
##verificamos que rellene correctamente
print(x_transformed.isnull().sum())


# MODEL (4to paso)
model = RandomForestClassifier()
model.fit(x_transformed, y)



## predicciones
df_test[object_cols] = ordinal_encoder.fit_transform(df_test[object_cols])

df_test_transformed = pd.DataFrame(imputer.transform(df_test))
df_test_transformed.columns= df_test.columns

predictions = model.predict(df_test_transformed)


# Submission
output = pd.DataFrame({'passengerId': df_test.PassengerId, 'survived': predictions})

output.to_csv('submission.csv', index = False)