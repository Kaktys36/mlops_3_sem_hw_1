import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load the dataset
file_path = '/app/creditdefault.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
data.head()


# 1.Общая информация о данных
data_info = data.info()


# 2.Статистический анализ
data_description = data.describe()


# 3.Проверка на пропуски
missing_values = data.isnull().sum()


# Построение диаграмм попарного распределения признаков (pairplot)
plt.figure(figsize=(12, 8))
sns.pairplot(data, hue="Default", palette="coolwarm")
plt.suptitle('Диаграммы попарного распределения признаков', y=1.02)
plt.show()

# Настроим стиль
sns.set(style="whitegrid")

# 1. Распределение дохода (Income)
plt.figure(figsize=(12, 6))
sns.histplot(data['Income'], bins=30, kde=True, color='skyblue')
plt.title('Распределение дохода')
plt.xlabel('Доход')
plt.ylabel('Частота')
plt.show()

# 2. Распределение возраста (Age)
plt.figure(figsize=(12, 6))
sns.histplot(data['Age'], bins=30, kde=True, color='lightgreen')
plt.title('Распределение возраста')
plt.xlabel('Возраст')
plt.ylabel('Частота')
plt.show()

# 3. Распределение суммы кредита (Loan)
plt.figure(figsize=(12, 6))
sns.histplot(data['Loan'], bins=30, kde=True, color='salmon')
plt.title('Распределение суммы кредита')
plt.xlabel('Сумма кредита')
plt.ylabel('Частота')
plt.show()

# 4. Корреляционная матрица
plt.figure(figsize=(8, 6))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Корреляционная матрица')
plt.show()

# Подсчет количества наблюдений для каждого класса
class_counts = data['Default'].value_counts()

# Визуализация распределения классов с помощью столбчатой диаграммы
plt.figure(figsize=(6, 4))
sns.barplot(
    x=class_counts.index,
    y=class_counts.values,
    palette='viridis',
    hue=class_counts.index)  # Добавлен параметр hue
plt.title('Распределение классов (дефолт/не дефолт)')
plt.xlabel('Класс (0 - не дефолт, 1 - дефолт)')
plt.ylabel('Количество')
plt.show()



data = data.dropna()  # На всякий случай, если есть пропуски

# Отделим целевую переменную (Default) от признаков
# Включаем признаки: доход, возраст, сумма кредита
X = data[['Income', 'Age', 'Loan']]
y = data['Default']  # Целевая переменная: дефолт (0 или 1)

# 2. Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 3. Обучение модели линейной регрессии
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# 5. Оценка модели
# Точность модели
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Матрица ошибок (confusion matrix)
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Отчет о классификации
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)

# Визуализация матрицы ошибок
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
