from typing import Annotated

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel

from fastapi import FastAPI, Body

app = FastAPI(
    title="group-analyzer",
    docs_url="/"
)

@app.post("/recognize_group")
def recognize_group(query: Annotated[str, Body()]) -> str | None:
    res = classify_topic(query)
    return res

# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка данных из CSV файла
try:
    df = pd.read_csv('dataset_.csv', sep=';', on_bad_lines='skip', quotechar='"', encoding='ISO-8859-1')
except pd.errors.ParserError as e:
    print(f"Error parsing CSV file: {e}")
    exit()

# Проверка наличия необходимых столбцов
required_columns = ['Topic', 'label']
if not all(col in df.columns for col in required_columns):
    print(f"CSV file is missing required columns: {required_columns}")
    exit()

# Удаление строк с пропущенными значениями
df.dropna(subset=required_columns, inplace=True)

# Подготовка данных
X = df['Topic'].tolist()
y = df['label'].tolist()

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Класс для обработки данных
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Инициализация токенизатора и модели BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')

# Создаем словарь меток, используя все уникальные метки из y_train и y_test
all_labels = set(y_train) | set(y_test)
label_to_id = {label: idx for idx, label in enumerate(all_labels)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

# Преобразование меток y_train и y_test в числовые форматы
y_train_ids = [label_to_id[label] for label in y_train]
y_test_ids = [label_to_id[label] for label in y_test]

# Создание наборов данных
train_dataset = TextDataset(X_train, y_train_ids, tokenizer)
test_dataset = TextDataset(X_test, y_test_ids, tokenizer)

# Инициализация модели для классификации
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_to_id)).to(device)

# Настройка параметров обучения
training_args = TrainingArguments(
    output_dir='results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='logs',
    logging_steps=10,
)

# Инициализация Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Обучение модели
# trainer.train()

# Функция для поиска похожего вопроса и метки
def find_similar_question(input_text, df):
    input_encoding = bert_model(**tokenizer(input_text, return_tensors="pt", truncation=True, padding=True))
    input_vector = input_encoding.last_hidden_state.mean(dim=1).detach()

    similarities = []
    for i, row in df.iterrows():
        topic_encoding = bert_model(**tokenizer(row['Topic'], return_tensors="pt", truncation=True, padding=True))
        topic_vector = topic_encoding.last_hidden_state.mean(dim=1).detach()
        similarity = cosine_similarity(input_vector, topic_vector)
        similarities.append((similarity[0][0], row['label']))

    # Найти наиболее похожий вопрос
    max_similarity, matched_label = max(similarities, key=lambda x: x[0])

    if max_similarity > 0.7:
        return matched_label  # Возвращаем метку, если совпадение достаточно высокое
    return None

# Функция для классификации пользовательского запроса
def classify_topic(new_topic) -> str | None:
    # Попытка найти похожий вопрос
    similar_label = find_similar_question(new_topic, df)
    if similar_label:
        print(f"\nPredicted label for the new topic: {similar_label}")
        return similar_label

    # Если похожий вопрос не найден, используем модель для классификации
    inputs = tokenizer(new_topic, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    predicted_label = id_to_label[predicted_class]

    print(f"\nPredicted label for the new topic: {predicted_label}")
    return predicted_label


