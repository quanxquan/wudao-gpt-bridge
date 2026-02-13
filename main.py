import os
import random
import requests
import logging
from flask import Flask, jsonify
from flask_cors import CORS
from modelscope.hub.api import HubApi
from modelscope.msdatasets import MsDataset

# 配置日志，方便在 GCP Logs 中排查错误
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- 强化版 CORS 配置 ---
# 允许所有来源 (*) 访问所有接口，并允许特定的 Header
CORS(app, resources={r"/*": {"origins": "*"}}, 
     allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Origin"])

api = HubApi()

# 从 GCP 环境变量读取配置
DATASET_ID = os.environ.get('DATASET_ID', 'whynlp/WuDaoCorpus-200G-shuffled')
API_BASE = os.environ.get('OPENAI_API_BASE', 'https://api.haojs.uk/v1')
API_KEY = os.environ.get('OPENAI_API_KEY')
MODEL = os.environ.get('MODEL_NAME', 'gpt-oss-120b')

def ask_ai_archeologist(content):
    """调用自建端点的 AI 进行分析"""
    if not API_KEY:
        logger.warning("API_KEY is missing in environment variables")
        return "AI 配置缺失，无法分析。"
    
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "你是一位精通互联网历史的考古学家。"},
            {"role": "user", "content": f"请分析语料：\n\n{content}"}
        ]
    }
    
    try:
        # 设置超时，防止请求被 AI 响应卡死
        response = requests.post(url, json=payload, headers=headers, timeout=45)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"AI API Error: {str(e)}")
        return f"AI 分析暂不可用: {str(e)}"

@app.route('/get_random')
def get_random():
    logger.info("Received request for /get_random")
    try:
        # 1. 获取语料分片
        files = api.get_dataset_files(dataset_id=DATASET_ID, revision='master')
        data_files = [f for f in files if f.startswith('data/') and f.endswith('.jsonl')]
        
        if not data_files:
            return jsonify({"status": "error", "message": "No data files found"}), 404
            
        target_file = random.choice(data_files)
        logger.info(f"Selected file: {target_file}")
        
        # 2. 加载数据（设置内存和超时优化）
        ds = MsDataset.load(DATASET_ID, data_files=target_file, split='train')
        item = next(iter(ds))
        content = item.get('content', '内容为空')
        
        # 3. 后端同步发起 AI 分析
        analysis = ask_ai_archeologist(content)
        
        return jsonify({
            "status": "success",
            "title": item.get('title', '无标题语料'),
            "content": content,
            "analysis": analysis,
            "meta": {"source": target_file, "dataset": DATASET_ID}
        })
    except Exception as e:
        logger.error(f"Execution Error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    # 动态端口绑定，GCP 必须
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
