# Bolzhau.Tech — Локальный запуск

## 📁 Структура проекта

```
bolzhau-app/
├── frontend/
│   └── index.html          ← просто открой в браузере
├── backend/
│   ├── app.py              ← Flask сервер
│   ├── requirements.txt
│   ├── models/             ← СЮДА КЛАДЁШЬ СВОИ МОДЕЛИ
│   │   ├── bigrams.pkl     ← твоя bigram модель
│   │   ├── trigrams.pkl    ← твоя trigram модель
│   │   ├── keras_model.h5  ← твоя Keras модель (или .pkl)
│   │   └── tokenizer.pkl   ← твой токенизатор (если есть)
│   └── users.json          ← создаётся автоматически
```

## 🚀 Запуск

### Шаг 1 — Установи зависимости
```bash
cd backend
pip install flask flask-cors
# Если нужен Keras:
pip install tensorflow
```

### Шаг 2 — Положи модели в папку models/

**Форматы которые поддерживаются:**

| Файл | Формат |
|------|--------|
| `bigrams.pkl` | `dict: {слово: [(следующее, вероятность), ...]}` |
| `trigrams.pkl` | `dict: {(сл1, сл2): [(фраза, вероятность), ...]}` |
| `keras_model.h5` | Keras .h5 файл или pickle |
| `tokenizer.pkl` | Keras Tokenizer через pickle |

**Пример создания pkl:**
```python
import pickle

# Если у тебя bigrams = {"мен": [("бүгін", 0.9), ("кеше", 0.7)]}
with open('models/bigrams.pkl', 'wb') as f:
    pickle.dump(bigrams, f)

# Если у тебя bigrams = {"мен": {"бүгін": 0.9, "кеше": 0.7}}
# тоже работает — app.py обрабатывает оба формата
```

### Шаг 3 — Запусти Flask
```bash
cd backend
python app.py
# → http://localhost:5000
```

### Шаг 4 — Открой фронтенд
Просто открой `frontend/index.html` в браузере.

Сайт автоматически подключится к http://localhost:5000

## ✅ Проверка

Открой http://localhost:5000/health в браузере.
Должен показать:
```json
{
  "status": "ok",
  "models": {
    "bigrams": true,
    "trigrams": true,
    "keras": true
  }
}
```

## ⚠️ Если модели не загружаются

Сайт всё равно работает! Вместо твоих моделей показываются примерные казахские слова.
Это нормально для демо.

## 📧 Настройка email (опционально)

Создай файл `.env` в папке backend:
```
SMTP_USER=bolzhauai@gmail.com
SMTP_PASS=твой_app_password
```

Gmail App Password: myaccount.google.com → Security → App passwords
