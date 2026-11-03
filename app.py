import os
import sqlite3
from datetime import date, datetime
from functools import wraps

import joblib
import numpy as np
import pandas as pd
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "finance_assistant.db")
MODEL_PATH = os.path.join(BASE_DIR, "data", "expense_classifier.joblib")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-this-secret-key-before-deployment")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

CATEGORIES = ["Food", "Transport", "Shopping", "Bills", "Entertainment", "Health", "Education", "Other"]

TRAINING_EXAMPLES = {
    "Food": ["swiggy dinner", "zomato lunch", "restaurant meal", "grocery vegetables", "coffee and snacks",
             "pizza order", "breakfast", "supermarket food", "bakery", "food delivery"],
    "Transport": ["uber ride", "ola cab", "bus ticket", "metro recharge", "petrol fuel", "train fare",
                  "auto rickshaw", "parking fee", "flight ticket", "bike taxi"],
    "Shopping": ["amazon shopping", "clothes purchase", "shoes", "mobile accessories", "online order",
                 "mall shopping", "household items", "electronics purchase", "book purchase online"],
    "Bills": ["electricity bill", "water bill", "internet broadband", "mobile recharge", "gas bill",
              "rent payment", "phone bill", "subscription bill", "maintenance charge"],
    "Entertainment": ["movie tickets", "netflix subscription", "concert ticket", "game purchase",
                      "music subscription", " bowling", " amusement park", "streaming service"],
    "Health": ["pharmacy medicine", "doctor consultation", "hospital bill", "medical test",
               "dentist appointment", "health checkup", "vitamins", "clinic visit"],
    "Education": ["college fees", "online course", "tuition fee", "exam registration", "stationery",
                  "programming course", "certification fee", "textbook"],
    "Other": ["miscellaneous expense", "donation", "gift", "unexpected payment", "other expense",
              "personal expense", "service charge"]
}

def train_classifier():
    texts, labels = [], []
    for category, examples in TRAINING_EXAMPLES.items():
        for phrase in examples:
            texts.append(phrase)
            labels.append(category)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(texts, labels)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model

def get_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass
    return train_classifier()

model = get_model()

def db():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection

def init_db():
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            tx_date TEXT NOT NULL,
            description TEXT NOT NULL,
            amount REAL NOT NULL CHECK(amount > 0),
            tx_type TEXT NOT NULL CHECK(tx_type IN ('income','expense')),
            category TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS budgets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            month TEXT NOT NULL,
            category TEXT NOT NULL,
            limit_amount REAL NOT NULL CHECK(limit_amount > 0),
            UNIQUE(user_id, month, category),
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """)
init_db()

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped

def predict_category(description):
    prediction = model.predict([description])[0]
    probabilities = model.predict_proba([description])[0]
    confidence = float(np.max(probabilities))
    return prediction, confidence

def user_transactions(user_id, month=None):
    query = "SELECT * FROM transactions WHERE user_id = ?"
    params = [user_id]
    if month:
        query += " AND substr(tx_date,1,7) = ?"
        params.append(month)
    query += " ORDER BY tx_date DESC, id DESC"
    with db() as conn:
        return conn.execute(query, params).fetchall()

def month_summary(user_id, month):
    with db() as conn:
        row = conn.execute("""
            SELECT
              COALESCE(SUM(CASE WHEN tx_type='income' THEN amount ELSE 0 END),0) AS income,
              COALESCE(SUM(CASE WHEN tx_type='expense' THEN amount ELSE 0 END),0) AS expenses
            FROM transactions WHERE user_id=? AND substr(tx_date,1,7)=?
        """, (user_id, month)).fetchone()
        categories = conn.execute("""
            SELECT category, SUM(amount) AS total FROM transactions
            WHERE user_id=? AND tx_type='expense' AND substr(tx_date,1,7)=?
            GROUP BY category ORDER BY total DESC
        """, (user_id, month)).fetchall()
    income, expenses = float(row["income"]), float(row["expenses"])
    return income, expenses, income - expenses, categories

def build_insights(user_id, month):
    income, expenses, balance, categories = month_summary(user_id, month)
    messages = []
    if not categories:
        messages.append("Add a few expenses to start receiving personalized spending insights.")
    else:
        top = categories[0]
        messages.append(f"Your highest spending category this month is {top['category']} (₹{top['total']:,.2f}).")
    with db() as conn:
        budgets = conn.execute("SELECT category, limit_amount FROM budgets WHERE user_id=? AND month=?", (user_id, month)).fetchall()
    spent = {r["category"]: float(r["total"]) for r in categories}
    for budget in budgets:
        used = spent.get(budget["category"], 0)
        limit = float(budget["limit_amount"])
        if used >= limit:
            messages.append(f"Budget alert: {budget['category']} spending has reached or exceeded its ₹{limit:,.2f} limit.")
        elif used >= limit * 0.8:
            messages.append(f"Budget reminder: {budget['category']} spending has used {used/limit:.0%} of its limit.")
    if income > 0 and expenses > income:
        messages.append("Your expenses are higher than your recorded income this month. Review non-essential spending.")
    elif income > 0:
        messages.append(f"You have retained {balance/income:.0%} of your recorded income so far this month.")
    return messages

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not name or not email or len(password) < 8:
            flash("Enter your name and email, and use a password with at least 8 characters.", "danger")
            return render_template("register.html")
        try:
            with db() as conn:
                conn.execute("INSERT INTO users(name,email,password_hash) VALUES(?,?,?)",
                             (name, email, generate_password_hash(password)))
            flash("Account created. Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("An account with that email already exists.", "danger")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        with db() as conn:
            user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    month = request.args.get("month", date.today().strftime("%Y-%m"))
    income, expenses, balance, categories = month_summary(session["user_id"], month)
    transactions = user_transactions(session["user_id"], month)[:6]
    insights = build_insights(session["user_id"], month)
    category_labels = [r["category"] for r in categories]
    category_values = [float(r["total"]) for r in categories]
    # Monthly expense trend for the most recent six months with data.
    with db() as conn:
        trend = conn.execute("""
            SELECT substr(tx_date,1,7) AS month, SUM(amount) AS total
            FROM transactions WHERE user_id=? AND tx_type='expense'
            GROUP BY substr(tx_date,1,7) ORDER BY month DESC LIMIT 6
        """, (session["user_id"],)).fetchall()
    trend = list(reversed(trend))
    return render_template("dashboard.html", month=month, income=income, expenses=expenses,
                           balance=balance, categories=category_labels, category_values=category_values,
                           trend_labels=[r["month"] for r in trend],
                           trend_values=[float(r["total"]) for r in trend],
                           transactions=transactions, insights=insights)

@app.route("/transactions", methods=["GET", "POST"])
@login_required
def transactions():
    if request.method == "POST":
        description = request.form.get("description", "").strip()
        tx_date = request.form.get("tx_date", "")
        tx_type = request.form.get("tx_type", "expense")
        try:
            amount = float(request.form.get("amount", "0"))
        except ValueError:
            amount = 0
        if not description or not tx_date or amount <= 0 or tx_type not in ("income", "expense"):
            flash("Enter a description, valid date, positive amount and transaction type.", "danger")
            return redirect(url_for("transactions"))
        if tx_type == "expense":
            category, confidence = predict_category(description)
            selected = request.form.get("category", "").strip()
            if selected in CATEGORIES:
                category = selected
        else:
            category = request.form.get("category", "Income")
            confidence = 1.0
        with db() as conn:
            conn.execute("""INSERT INTO transactions(user_id,tx_date,description,amount,tx_type,category)
                            VALUES(?,?,?,?,?,?)""",
                         (session["user_id"], tx_date, description, amount, tx_type, category))
        flash(f"Transaction saved. Category: {category}.", "success")
        return redirect(url_for("transactions"))
    items = user_transactions(session["user_id"])
    return render_template("transactions.html", transactions=items, categories=CATEGORIES, today=date.today().isoformat())

@app.route("/transactions/<int:transaction_id>/delete", methods=["POST"])
@login_required
def delete_transaction(transaction_id):
    with db() as conn:
        conn.execute("DELETE FROM transactions WHERE id=? AND user_id=?", (transaction_id, session["user_id"]))
    flash("Transaction deleted if it belonged to your account.", "success")
    return redirect(url_for("transactions"))

@app.route("/transactions/<int:transaction_id>/edit", methods=["GET", "POST"])
@login_required
def edit_transaction(transaction_id):
    with db() as conn:
        item = conn.execute("SELECT * FROM transactions WHERE id=? AND user_id=?",
                            (transaction_id, session["user_id"])).fetchone()
    if not item:
        flash("Transaction not found.", "danger")
        return redirect(url_for("transactions"))
    if request.method == "POST":
        description = request.form.get("description", "").strip()
        tx_date = request.form.get("tx_date", "")
        try:
            amount = float(request.form.get("amount", "0"))
        except ValueError:
            amount = 0
        category = request.form.get("category", item["category"])
        if not description or not tx_date or amount <= 0:
            flash("Please enter valid transaction details.", "danger")
        else:
            if item["tx_type"] == "expense" and category == "Auto-detect":
                category, _ = predict_category(description)
            with db() as conn:
                conn.execute("""UPDATE transactions SET tx_date=?,description=?,amount=?,category=?
                                WHERE id=? AND user_id=?""",
                             (tx_date, description, amount, category, transaction_id, session["user_id"]))
            flash("Transaction updated.", "success")
            return redirect(url_for("transactions"))
    return render_template("edit_transaction.html", item=item, categories=CATEGORIES)

@app.route("/budgets", methods=["GET", "POST"])
@login_required
def budgets():
    current_month = date.today().strftime("%Y-%m")
    if request.method == "POST":
        month = request.form.get("month", current_month)
        category = request.form.get("category", "")
        try:
            limit_amount = float(request.form.get("limit_amount", "0"))
        except ValueError:
            limit_amount = 0
        if category not in CATEGORIES or limit_amount <= 0 or len(month) != 7:
            flash("Enter a valid month, category and positive budget amount.", "danger")
        else:
            with db() as conn:
                conn.execute("""INSERT INTO budgets(user_id,month,category,limit_amount)
                                VALUES(?,?,?,?)
                                ON CONFLICT(user_id,month,category) DO UPDATE SET limit_amount=excluded.limit_amount""",
                             (session["user_id"], month, category, limit_amount))
            flash("Budget saved.", "success")
        return redirect(url_for("budgets"))
    with db() as conn:
        rows = conn.execute("SELECT * FROM budgets WHERE user_id=? ORDER BY month DESC, category",
                            (session["user_id"],)).fetchall()
    budget_rows = []
    for row in rows:
        with db() as conn:
            spent = conn.execute("""SELECT COALESCE(SUM(amount),0) AS total FROM transactions
                WHERE user_id=? AND tx_type='expense' AND category=? AND substr(tx_date,1,7)=?""",
                (session["user_id"], row["category"], row["month"])).fetchone()["total"]
        limit = float(row["limit_amount"])
        budget_rows.append({"id": row["id"], "month": row["month"], "category": row["category"],
                            "limit": limit, "spent": float(spent), "percent": min(100, float(spent)/limit*100)})
    return render_template("budgets.html", budgets=budget_rows, categories=CATEGORIES, current_month=current_month)

@app.route("/budgets/<int:budget_id>/delete", methods=["POST"])
@login_required
def delete_budget(budget_id):
    with db() as conn:
        conn.execute("DELETE FROM budgets WHERE id=? AND user_id=?", (budget_id, session["user_id"]))
    flash("Budget deleted.", "success")
    return redirect(url_for("budgets"))

@app.route("/insights")
@login_required
def insights():
    month = request.args.get("month", date.today().strftime("%Y-%m"))
    messages = build_insights(session["user_id"], month)
    with db() as conn:
        history = conn.execute("""SELECT substr(tx_date,1,7) AS month, SUM(amount) AS total
            FROM transactions WHERE user_id=? AND tx_type='expense'
            GROUP BY substr(tx_date,1,7) ORDER BY month""", (session["user_id"],)).fetchall()
    values = [float(row["total"]) for row in history]
    # Simple transparent baseline forecast: average of up to the latest three available months.
    forecast = round(float(np.mean(values[-3:])), 2) if values else None
    if forecast is None:
        forecast_message = "Add expense history to calculate a spending estimate."
    else:
        forecast_message = f"Estimated spending for a future month: ₹{forecast:,.2f} (average of up to the latest 3 months with data)."
    return render_template("insights.html", month=month, insights=messages, forecast_message=forecast_message)

@app.route("/api/predict-category", methods=["POST"])
@login_required
def api_predict_category():
    data = request.get_json(silent=True) or {}
    description = str(data.get("description", "")).strip()
    if not description:
        return jsonify({"error": "Description is required."}), 400
    category, confidence = predict_category(description)
    return jsonify({"category": category, "confidence": round(confidence, 3)})

@app.route("/chatbot", methods=["POST"])
@login_required
def chatbot():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).lower()
    if any(word in message for word in ["budget", "limit"]):
        answer = "Open Budgets to set a monthly limit for each expense category. The dashboard and Insights page show budget warnings."
    elif any(word in message for word in ["add", "record", "transaction", "expense", "income"]):
        answer = "Open Transactions, enter the date, description, amount and type, then save. Expense categories are predicted automatically."
    elif any(word in message for word in ["balance", "dashboard", "summary"]):
        answer = "Your Dashboard displays income, expenses, available balance and category-wise spending for the selected month."
    elif any(word in message for word in ["predict", "forecast", "future"]):
        answer = "The Insights page estimates future expenses using the average of up to the latest three months with recorded expenses."
    else:
        answer = "I can help with this app's transactions, budgets, dashboard, expense categories and spending estimates. What would you like to know?"
    return jsonify({"answer": answer})

@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404

if __name__ == "__main__":
    app.run(debug=True)
