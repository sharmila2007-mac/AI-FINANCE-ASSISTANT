# AI Finance Assistant

A beginner-friendly personal finance web application built from the supplied AI Finance Assistant SRS.

## Features
- User registration, login and logout with hashed passwords
- User-specific SQLite records
- Income and expense CRUD (create, read, update, delete)
- ML expense categorization using TF-IDF + Logistic Regression
- Monthly and category-wise budgets with usage indicators
- Dashboard with income, expenses, balance, category chart and monthly trend
- Spending insights and a simple historical-average spending estimate
- In-app help chatbot for application-related questions
- JSON endpoint: `POST /api/predict-category`

## Technology
Python, Flask, SQLite, scikit-learn, pandas, NumPy, HTML, CSS, JavaScript and Chart.js.

## Run locally (macOS / Windows / Linux)
1. Install Python 3.10 or newer.
2. Extract this ZIP and open the project folder in VS Code or Terminal.
3. Create and activate a virtual environment:

   **macOS / Linux**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   **Windows**
   ```powershell
   py -m venv venv
   venv\\Scripts\\activate
   ```

4. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
5. Start the application:
   ```bash
   python app.py
   ```
6. Open `http://127.0.0.1:5000` in your browser and create an account.

The SQLite database is created automatically on first run. The ML model is trained from the small educational phrase dataset in `app.py` and saved under `data/` on first run.

## How to demonstrate
1. Register and log in.
2. Add an income transaction (for example, salary, ₹25000).
3. Add expenses such as `Swiggy dinner`, `Uber ride`, `electricity bill` and `pharmacy medicine`. Leave category as Auto-detect.
4. Open Dashboard to view totals and charts.
5. Create a monthly Food budget and add more food expenses to trigger a budget reminder.
6. Open AI Insights to see observations and a simple spending estimate.
7. Use the Ask Assistant button to ask how to add transactions or set budgets.

## API example
After logging in, send a JSON request to `/api/predict-category`:
```json
{"description": "swiggy dinner"}
```
The response includes a predicted category and model confidence.

## Project structure
```text
ai_finance_assistant/
├── app.py
├── requirements.txt
├── README.md
├── finance_assistant.db   # created automatically at runtime
├── data/
│   └── expense_classifier.joblib  # created automatically
├── templates/
│   ├── base.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── transactions.html
│   ├── edit_transaction.html
│   ├── budgets.html
│   ├── insights.html
│   └── 404.html
└── static/
    ├── style.css
    └── app.js
```

## Important notes
- The classifier is a compact demonstration model trained on example phrases, not a production financial model. Predictions may be wrong; users can choose a category manually.
- The spending estimate is a transparent average of up to the latest three months with expense data. It is not a sophisticated time-series model.
- The chatbot is rule-based and answers basic questions about using this application; it is not a general-purpose financial advisor.
- Chart.js and Google Fonts are loaded from CDNs, so those visual enhancements need an internet connection. The core Flask application runs locally.
- For deployment, set a strong random `SECRET_KEY`, turn off Flask debug mode, use HTTPS, and configure production hosting and database backups.
- This educational project does not connect to banks, move money, provide investment/tax advice, or replace a qualified financial professional.
