import os
import random
import requests
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
# 允许 Lovable 的所有预览域名跨域
CORS(app, resources={r"/*": {"origins": "*"}})

# 环境变量读取
API_BASE = os.environ.get('OPENAI_API_BASE', 'https://api.haojs.uk/v1')
API_KEY = os.environ.get('OPENAI_API_KEY')
MODEL = os.environ.get('MODEL_NAME', 'gpt-oss-120b')

# 极简语料池，确保 100% 返回数据
ARCHIVE_DATA = [
    "【2008年贴吧】如果有一天我消失了，谁会记得我？",
    "神马都是浮云，在这个赛博时代，我们都是数字游民。",
    "那是拨号上网的年代，一首 MP3 要下半个小时，但等待也是幸福的。"
]

@app.route('/get_random')
def get_random():
    content = random.choice(ARCHIVE_DATA)
    try:
        # 直接请求你的自建端点
        r = requests.post(
            f"{API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": f"考古分析：{content}"}]
            },
            timeout=10
        )
        analysis = r.json()['choices'][0]['message']['content']
    except:
        analysis = "AI 考古专家正在离线分析中..."

    return jsonify({
        "status": "success",
        "content": content,
        "analysis": analysis
    })

if __name__ == "__main__":
    # 必须监听 0.0.0.0 并使用 GCP 指定的 PORT
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
