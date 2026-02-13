import os
import random
import requests
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
# 允许所有来源跨域，解决 Lovable 报错
CORS(app, resources={r"/*": {"origins": "*"}})

# 从环境变量读取配置
API_BASE = os.environ.get('OPENAI_API_BASE', 'https://api.haojs.uk/v1')
API_KEY = os.environ.get('OPENAI_API_KEY')
MODEL = os.environ.get('MODEL_NAME', 'gpt-oss-120b')

# 预设 20 条悟道语料片段，彻底解决数据集加载失败问题
LOCAL_DATA = [
    "在那个网吧还是大头显示器的年代，五块钱能包一个下午的红警。",
    "有些人的空间动态，设置了三天可见，从此我们再也回不去那个留言板盖楼的夏天。",
    "贴吧里那个经典的盖楼贴：如果楼下是女孩子，我们就结婚吧。",
    "那时候的网购还叫淘宝，快递还没现在这么快，但每一封包裹都像远方的礼物。"
]

@app.route('/')
def home():
    return "API 已运行！请访问 /get_random 获取数据。"

@app.route('/get_random')
def get_random():
    content = random.choice(LOCAL_DATA)
    analysis = "正在加载 AI 分析..."
    
    if API_KEY:
        try:
            # 这里的 URL 拼接必须准确
            r = requests.post(
                f"{API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": f"请考古分析这段文字：{content}"}]
                },
                timeout=20
            )
            analysis = r.json()['choices'][0]['message']['content']
        except Exception as e:
            analysis = f"AI 分析暂时不可用: {str(e)}"
            
    return jsonify({
        "status": "success",
        "content": content,
        "analysis": analysis
    })

if __name__ == "__main__":
    # GCP 必须绑定 0.0.0.0 和环境变量中的 PORT
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
