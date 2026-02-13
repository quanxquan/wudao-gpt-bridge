import os
import random
import requests
from flask import Flask, jsonify
from flask_cors import CORS
from modelscope.hub.api import HubApi
from modelscope.msdatasets import MsDataset

app = Flask(__name__)
CORS(app)  # 允许 Lovable 前端跨域请求

api = HubApi()

# --- 从 GCP 环境变量读取配置 ---
DATASET_ID = os.environ.get('DATASET_ID', 'whynlp/WuDaoCorpus-200G-shuffled')
API_BASE = os.environ.get('OPENAI_API_BASE', 'https://api.haojs.uk/v1')
API_KEY = os.environ.get('OPENAI_API_KEY')
MODEL = os.environ.get('MODEL_NAME', 'gpt-oss-120b')

def ask_ai_archeologist(content):
    """调用自建端点的 AI 进行分析"""
    if not API_KEY:
        return "AI 配置缺失，无法分析。"
    
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system", 
                "content": "你是一位精通互联网历史的考古学家。请分析语料的年代感、社会背景及文风特征。"
            },
            {
                "role": "user", 
                "content": f"请对以下语料进行考古分析：\n\n{content}"
            }
        ],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        res_json = response.json()
        return res_json['choices'][0]['message']['content']
    except Exception as e:
        return f"AI 分析出错: {str(e)}"

@app.route('/get_random')
def get_random():
    try:
        # 1. 获取语料分片
        files = api.get_dataset_files(dataset_id=DATASET_ID, revision='master')
        data_files = [f for f in files if f.startswith('data/') and f.endswith('.jsonl')]
        
        if not data_files:
            return jsonify({"status": "error", "message": "未找到数据文件"}), 404
            
        target_file = random.choice(data_files)
        
        # 2. 读取随机语料
        ds = MsDataset.load(DATASET_ID, data_files=target_file, split='train')
        item = next(iter(ds))
        content = item.get('content', '内容为空')
        
        # 3. 直接在后端发起 AI 分析
        analysis = ask_ai_archeologist(content)
        
        return jsonify({
            "status": "success",
            "title": item.get('title', '无标题语料'),
            "content": content,
            "analysis": analysis,
            "meta": {
                "source": target_file,
                "model": MODEL,
                "dataset": DATASET_ID
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    # 动态绑定 GCP 端口
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
