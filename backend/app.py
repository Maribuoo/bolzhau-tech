"""
Bolzhau.Tech — Flask Backend (самодостаточный, без models.py/utils.py)

Структура папок:
  backend/
  ├── app.py  ← этот файл
  └── models/
      ├── kazakh_word_completion.pkl   (38 МБ)
      ├── bi_gram_model.pkl            (174 МБ)  ← или compressed версия
      └── tri_gram_model_mini.pkl      (430 МБ)  ← или compressed версия

Запуск:
  pip install flask flask-cors
  python app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import Counter
import pickle, os, json, hashlib, secrets
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ─── Пути к моделям ───────────────────────────────────────────────
BASE = os.path.dirname(__file__)

# Пробуем compressed сначала (быстрее грузятся), потом обычные
BIGRAM_PATHS = [
    os.path.join(BASE, 'models', 'compressed', 'compressed_bi_gram_model.pkl'),
    os.path.join(BASE, 'models', 'bi_gram_model.pkl'),
]
TRIGRAM_PATHS = [
    os.path.join(BASE, 'models', 'compressed', 'compressed_tri_gram_model.pkl'),
    os.path.join(BASE, 'models', 'tri_gram_model_mini.pkl'),
]
PREFIX_PATH = os.path.join(BASE, 'models', 'kazakh_word_completion.pkl')

# ─── Загрузка моделей ─────────────────────────────────────────────
print("⏳ Загружаем модели...")

# 1. Word completion (kazakh_word_completion.pkl)
# Формат: { "prefix_string": Counter({word: freq, ...}) }
PREFIX_MODEL = {}
try:
    with open(PREFIX_PATH, 'rb') as f:
        PREFIX_MODEL = pickle.load(f)
    print(f"✅ kazakh_word_completion.pkl  ({len(PREFIX_MODEL)} ключей)")
except FileNotFoundError:
    print("⚠️  kazakh_word_completion.pkl не найден")

# 2. Bigram (bi_gram_model.pkl)
# Формат в файле: { 'pair_counts': {(w1,w2): count}, 'word_counts': {w: count} }
# После обработки храним как: { (w1, w2): probability }
BIGRAM_MODEL = {}
for path in BIGRAM_PATHS:
    if os.path.exists(path):
        print(f"   Загружаем bigram: {os.path.basename(path)} ...")
        with open(path, 'rb') as f:
            raw = pickle.load(f)
        # Вычисляем условные вероятности P(w2|w1) = count(w1,w2) / count(w1)
        BIGRAM_MODEL = {
            (w1, w2): count / raw['word_counts'][w1]
            for (w1, w2), count in raw['pair_counts'].items()
            if w1 in raw['word_counts'] and raw['word_counts'][w1] > 0
        }
        print(f"✅ bigram загружен  ({len(BIGRAM_MODEL)} пар)")
        break
else:
    print("⚠️  bigram модель не найдена")

# 3. Trigram (tri_gram_model_mini.pkl)
# Формат: { 'triple_counts': {(w1,w2,w3): count}, 'bigram_counts': {...} }
TRIGRAM_MODEL = {}
for path in TRIGRAM_PATHS:
    if os.path.exists(path):
        print(f"   Загружаем trigram: {os.path.basename(path)} ...")
        with open(path, 'rb') as f:
            TRIGRAM_MODEL = pickle.load(f)
        print(f"✅ trigram загружен  ({len(TRIGRAM_MODEL.get('triple_counts', {}))} триграмм)")
        break
else:
    print("⚠️  trigram модель не найдена")

print("🚀 Готово!")

# ─── Функции предсказания ─────────────────────────────────────────

def predict_word_end(text: str, top_n: int = 5):
    """
    WORD_END — kazakh_word_completion.pkl
    Ищет prefix в Counter-словаре, возвращает топ слов
    """
    words  = text.strip().lower().split()
    prefix = words[-1] if words else text.strip().lower()

    counter = PREFIX_MODEL.get(prefix, Counter())
    if not counter:
        return []

    top   = counter.most_common(top_n)
    total = sum(freq for _, freq in top)
    if total == 0:
        return []

    return [
        {'text': f'...{word}', 'confidence': round(freq / total * 100)}
        for word, freq in top
    ]


def predict_next_word(text: str, top_n: int = 5):
    """
    NEXT_WORD — bi_gram_model.pkl
    P(w2 | last_word) из условных вероятностей
    """
    words     = text.strip().lower().split()
    last_word = words[-1] if words else ''

    candidates = {
        w2: prob
        for (w1, w2), prob in BIGRAM_MODEL.items()
        if w1 == last_word
    }

    if not candidates:
        return []

    top   = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_n]
    total = sum(p for _, p in top)
    if total == 0:
        return []

    return [
        {'text': word, 'confidence': round(prob / total * 100)}
        for word, prob in top
    ]


def predict_phrase(text: str, top_n: int = 5):
    """
    PHRASE — tri_gram_model_mini.pkl
    Ищет (w1,w2) → w3, потом (w2,w3) → w4, возвращает "w3 w4"
    """
    if not TRIGRAM_MODEL:
        return []

    words  = text.strip().lower().split()
    triple = TRIGRAM_MODEL.get('triple_counts', {})

    if len(words) < 2:
        # fallback на bigram
        return predict_next_word(text, top_n)

    word1, word2 = words[-2], words[-1]

    # Шаг 1: ищем w3 после (word1, word2)
    first = {
        w3: count
        for (w1, w2, w3), count in triple.items()
        if w1 == word1 and w2 == word2
    }
    if not first:
        return []

    first_sorted = sorted(first.items(), key=lambda x: x[1], reverse=True)[:top_n]

    results = []
    for w3, count1 in first_sorted:
        # Шаг 2: ищем w4 после (word2, w3)
        nexts = {
            w4: triple[(word2, w3, w4)]
            for (w1, w2, w3_), w4 in [
                ((w1, w2, w3_), w4_)
                for (w1, w2, w3_) in triple
                for w4_ in [None]          # placeholder
            ]
            if False  # заменяется ниже
        }
        # Простой поиск продолжения
        nexts = {
            w4: triple[(word2, w3, w4)]
            for (w1, w2, w3_) in triple
            if w1 == word2 and w2 == w3
            for w4 in [w3_]
        }

        if nexts:
            w4, count2 = max(nexts.items(), key=lambda x: x[1])
            results.append((f'{w3} {w4}', count1 * count2))
        else:
            results.append((w3, count1))

    results.sort(key=lambda x: x[1], reverse=True)
    total = sum(s for _, s in results) or 1

    return [
        {'text': phrase, 'confidence': round(score / total * 100)}
        for phrase, score in results[:top_n]
    ]


# ─── API роуты ────────────────────────────────────────────────────

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON required'}), 400

    text = data.get('text', '').strip()
    mode = data.get('mode', 'NEXT_WORD')

    if not text:
        return jsonify({'error': 'text is required'}), 400
    if mode not in ('WORD_END', 'NEXT_WORD', 'PHRASE'):
        return jsonify({'error': 'mode must be WORD_END, NEXT_WORD, or PHRASE'}), 400

    if mode == 'WORD_END':
        suggestions = predict_word_end(text)
    elif mode == 'NEXT_WORD':
        suggestions = predict_next_word(text)
    else:
        suggestions = predict_phrase(text)

    return jsonify({'suggestions': suggestions, 'model': mode, 'text': text})


@app.route('/api/feedback', methods=['POST'])
def feedback():
    data    = request.get_json() or {}
    name    = (data.get('name')    or '').strip()
    email   = (data.get('email')   or '').strip()
    message = (data.get('message') or '').strip()

    if not (name and email and message):
        return jsonify({'error': 'name, email and message are required'}), 400

    log_path = os.path.join(BASE, 'messages.json')
    msgs = json.load(open(log_path)) if os.path.exists(log_path) else []
    msgs.append({'name': name, 'email': email,
                 'message': message, 'time': datetime.now().isoformat()})
    with open(log_path, 'w') as f:
        json.dump(msgs, f, ensure_ascii=False, indent=2)

    # Email (опционально — нужны SMTP_USER и SMTP_PASS в .env)
    try:
        import smtplib
        from email.mime.text import MIMEText
        u, p = os.getenv('SMTP_USER',''), os.getenv('SMTP_PASS','')
        if u and p:
            msg = MIMEText(f"От: {name} <{email}>\n\n{message}")
            msg['Subject'] = f'Bolzhau — {name}'
            msg['From'] = u; msg['To'] = 'bolzhauai@gmail.com'
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
                s.login(u, p); s.send_message(msg)
    except Exception:
        pass

    return jsonify({'status': 'ok'}), 201


# ─── Авторизация ──────────────────────────────────────────────────
USERS_FILE = os.path.join(BASE, 'users.json')

def load_users():
    return json.load(open(USERS_FILE)) if os.path.exists(USERS_FILE) else {}

def save_users(u):
    json.dump(u, open(USERS_FILE,'w'), ensure_ascii=False, indent=2)

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()


@app.route('/api/auth/register', methods=['POST'])
def register():
    d = request.get_json() or {}
    name  = (d.get('name')     or '').strip()
    email = (d.get('email')    or '').strip().lower()
    pw    = (d.get('password') or '')

    if not (name and email and pw):
        return jsonify({'error': 'Барлық өрістерді толтырыңыз'}), 400
    if len(pw) < 6:
        return jsonify({'error': 'Кем дегенде 6 таңба'}), 400

    users = load_users()
    if email in users:
        return jsonify({'error': 'Бұл email тіркелген'}), 409

    token = secrets.token_hex(32)
    users[email] = {'name': name, 'password': hash_pw(pw), 'token': token}
    save_users(users)
    return jsonify({'token': token, 'name': name}), 201


@app.route('/api/auth/login', methods=['POST'])
def login():
    d     = request.get_json() or {}
    email = (d.get('email')    or '').strip().lower()
    pw    = (d.get('password') or '')

    users = load_users()
    user  = users.get(email)
    if not user or user['password'] != hash_pw(pw):
        return jsonify({'error': 'Email немесе пароль қате'}), 401

    return jsonify({'token': user['token'], 'name': user['name']}), 200


@app.route('/api/translate', methods=['POST'])
def translate():
    return jsonify({'result': '⚙️ Аударма қызметі жақында қосылады...'}), 200

@app.route('/api/chat', methods=['POST'])
def chat():
    return jsonify({'reply': '⚙️ Тілдесу қызметі жақында қосылады...'}), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'models': {
            'word_completion': len(PREFIX_MODEL) > 0,
            'bigram':          len(BIGRAM_MODEL) > 0,
            'trigram':         bool(TRIGRAM_MODEL),
        }
    })


if __name__ == '__main__':
    print("🌐 http://localhost:5000")
    app.run(debug=True, port=5000, host='0.0.0.0')
