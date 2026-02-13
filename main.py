import os
import random
import requests
from flask import Flask, jsonify
from flask_cors import CORS
from datasets import load_dataset

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# 环境变量
API_BASE = os.environ.get('OPENAI_API_BASE', 'https://api.haojs.uk/v1')
API_KEY = os.environ.get('OPENAI_API_KEY')
MODEL = os.environ.get('MODEL_NAME', 'gpt-oss-120b')

@app.route('/get_random')
def get_random():
    try:
        # 使用截图里的稳定源：p208p2002/wudaocorpus
        # streaming=True 是为了防止 Cloud Run 内存爆炸
        ds = load_dataset("p208p2002/wudaocorpus", split="train", streaming=True)
        
        # 随机跳过一段，增加考古的随机性
        shuffled_ds = ds.shuffle(seed=random.randint(0, 1000000), buffer_size=1000)
        item = next(iter(shuffled_ds))
        
        # 该数据集的字段通常是 'text'
        content = item.get('text', '') or item.get('content', '语料提取失败')
        clean_content = content[:1200]
        
        # 调用你的 120b AI 分析
        analysis = "AI 考古专家正在解析..."
        if API_KEY:
            try:
                r = requests.post(
                    f"{API_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json={
                        "model": MODEL,
                        "messages": [
                            {"role": "system", "content": "你是一位互联网考古学家，请分析这段文字的时代背景。"},
                            {"role": "user", "content": clean_content}
                        ]
                    },
                    timeout=20
                )
                analysis = r.json()['choices'][0]['message']['content']
            except:
                analysis = "AI 暂时无法解析这段深层语料。"

        return jsonify({
            "status": "success",
            "content": clean_content,
            "analysis": analysis,
            "meta": {"source": "p208p2002/wudaocorpus"}
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
