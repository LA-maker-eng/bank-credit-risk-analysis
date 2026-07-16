from flask import Flask, render_template, request, jsonify
from llm_analysis import get_model_result
from db_module import (
    init_database, import_csv_to_db, get_customer_overdue_stats,
    get_multi_table_analysis, get_high_risk_customers,
    get_finance_product_popularity, get_loan_type_distribution, execute_query
)
from rag_module import build_knowledge_base, query_knowledge_base
from data_validation import validate_customer_data, validate_batch_data, generate_validation_report
import pandas as pd
import os

app = Flask(__name__)

global_data = get_model_result()

if not os.path.exists("bank_credit.db"):
    init_database()
    import_csv_to_db()

if not os.path.exists("faiss_index"):
    build_knowledge_base()

@app.route("/")
def index():
    return render_template("index.html", data=global_data)

@app.route("/api/stats")
def get_stats():
    overdue_stats = get_customer_overdue_stats()
    high_risk = get_high_risk_customers()
    loan_dist = get_loan_type_distribution()
    
    return jsonify({
        "overdue_stats": overdue_stats.to_dict(orient="records"),
        "high_risk_count": len(high_risk),
        "loan_distribution": loan_dist.to_dict(orient="records")
    })

@app.route("/api/multi_table")
def get_multi_table():
    data = get_multi_table_analysis()
    return jsonify(data.to_dict(orient="records"))

@app.route("/api/finance_products")
def get_finance_products():
    data = get_finance_product_popularity()
    return jsonify(data.to_dict(orient="records"))

@app.route("/api/rag_query", methods=["POST"])
def rag_query():
    question = request.json.get("question", "")
    if not question:
        return jsonify({"error": "请输入问题"}), 400
    
    try:
        result = query_knowledge_base(question)
        return jsonify({
            "answer": result["answer"],
            "sources": result["sources"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/validate", methods=["POST"])
def validate_api():
    data = request.json.get("data", {})
    result = validate_customer_data(data)
    return jsonify(result)

@app.route("/api/validate_batch", methods=["POST"])
def validate_batch_api():
    records = request.json.get("records", [])
    if not records:
        return jsonify({"error": "请提供待校验数据"}), 400
    
    df = pd.DataFrame(records)
    result = validate_batch_data(df)
    result["report"] = generate_validation_report(result)
    return jsonify(result)

@app.route("/api/query", methods=["POST"])
def sql_query():
    query = request.json.get("query", "")
    if not query:
        return jsonify({"error": "请输入SQL查询语句"}), 400
    
    try:
        result = execute_query(query)
        return jsonify({
            "columns": result.columns.tolist(),
            "data": result.to_dict(orient="records")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)