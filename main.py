import os
import random
import requests
import logging
from flask import Flask, jsonify
from flask_cors import CORS
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# 彻底放开跨域，确保 Lovable 预览和未来的 Cloudflare 部署都能通
CORS(app, resources={r"/*": {"origins": "*"}})

# 从 GCP 环境变量读取配置
API_BASE = os.environ.get('OPENAI_API_BASE', 'https://api.haojs.uk/v1')
API_KEY = os.environ.get('OPENAI_API_KEY')
MODEL = os.environ.get('MODEL_NAME', 'gpt-oss-120b')

def ask_ai(content):
    """请求你的自建 120b API"""
    if not API_KEY: 
        return "AI 配置未完成，请检查 GCP 环境变量。"
    try:
        url = f"{API_BASE}/chat/completions"
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "你是一位资深互联网考古学家，请分析这段语料的年代感、社会背景及文风特征。"},
                {"role": "user", "content": f"考古目标：\n\n{content}"}
            ]
        }
        res = requests.post(url, json=payload, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=25)
        return res.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"AI 分析暂不可用: {str(e)}"

@app.route('/get_random')
def get_random():
    try:
        # 使用流式加载，针对 TXT 版本的悟道 2.0
        # 这种方式不会下载整个 60G，而是只抓取一小段
        dataset = load_dataset("mdokl/WuDaoCorpora2.0-RefinedEdition60GTXT", split="train", streaming=True)
        
        # 随机跳过 0-5000 条来模拟真随机
        shuffled_dataset = dataset.skip(random.randint(0, 5000))
        item = next(iter(shuffled_dataset))
        
        # TXT 格式通常直接在 'text' 字段中
        content = item.get('text', '') or item.get('content', '内容获取失败')
        clean_content = content[:1200] # 截取前 1200 字，保证前端展示美观
        
        # 联动分析
        analysis = ask_ai(clean_content)
        
        return jsonify({
            "status": "success",
            "content": clean_content,
            "analysis": analysis,
            "meta": {
                "source": "HuggingFace/WuDao2.0-Refined",
                "model": MODEL
            }
        })
    except Exception as e:
        logger.error(f"HF Error: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": f"挖掘深度不足，请重试: {str(e)}"
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
