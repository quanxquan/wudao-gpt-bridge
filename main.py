import os
import random
import requests
import pandas as pd
import io
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

API_BASE = os.environ.get('OPENAI_API_BASE', 'https://api.haojs.uk/v1')
API_KEY = os.environ.get('OPENAI_API_KEY')
MODEL = os.environ.get('MODEL_NAME', 'gpt-oss-120b')

@app.route('/get_random')
def get_random():
    try:
        # 直接使用你截图中的文件名规则 (p208p2002/wudaocorpus)
        # 从你截图的文件列表中选取几个确定的文件名
        parquet_files = [
            "part-2021009337.parquet",
            "part-2021011897.parquet",
            "part-2021012501.parquet"
        ]
        target_file = random.choice(parquet_files)
        
        # 构建 Raw 下载链接
        url = f"https://huggingface.co/datasets/p208p2002/wudaocorpus/resolve/main/{target_file}"
        
        # 只读取文件的前一小块以节省内存和时间
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        
        # 使用 pandas 读取 Parquet
        df = pd.read_parquet(io.BytesIO(resp.content))
        
        # 随机抽取一行
        random_row = df.sample(n=1).iloc[0]
        content = str(random_row.get('text') or random_row.get('content') or "内容解析为空")
        clean_content = content[:1200]

        # AI 分析逻辑
        analysis = "AI 考古专家正在解析..."
        if API_KEY:
            try:
                ai_r = requests.post(
                    f"{API_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": f"考古分析：{clean_content}"}]
                    },
                    timeout=20
                )
                analysis = ai_r.json()['choices'][0]['message']['content']
            except:
                analysis = "AI 连线超时，请重试。"

        return jsonify({
            "status": "success",
            "content": clean_content,
            "analysis": analysis,
            "meta": {"source": target_file}
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"挖掘失败: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
