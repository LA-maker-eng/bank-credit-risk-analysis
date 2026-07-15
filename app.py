from flask import Flask, render_template
from llm_analysis import get_model_result

app = Flask(__name__)

# 只加载一次模型数据
global_data = get_model_result()

@app.route("/")
def index():
    return render_template("index.html", data=global_data)

if __name__ == "__main__":
    app.run(debug=True)