import os
import random
import requests
import logging
from flask import Flask, jsonify
from flask_cors import CORS

# 配置详细日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# 环境变量
API_BASE = os.environ.get('OPENAI_API_BASE', 'https://api.haojs.uk/v1')
API_KEY = os.environ.get('OPENAI_API_KEY')
MODEL = os.environ.get('MODEL_NAME', 'gpt-oss-120b')

@app.route('/get_random')
def get_random():
    logger.info("开始挖掘语料...")
    try:
        # 1. 随机选择分片
        file_num = random.randint(1, 354)
        file_name = f"train-{file_num:05d}-of-00354.txt"
        
        # 2. 构建 Hugging Face 原始文件链接
        # 确认这个 URL 在浏览器里是否能直接打开
        raw_url = f"https://huggingface.co/datasets/mdokl/WuDaoCorpora2.0-RefinedEdition60GTXT/resolve/main/data/{file_name}"
        
        logger.info(f"正在读取文件: {file_name}")
        
        # 3. 抓取内容 (不使用 Range，直接读取前 5000 字节，更稳健)
        resp = requests.get(raw_url, timeout=15)
        resp.raise_for_status() # 如果 404 或 403 会直接抛出异常
        
        full_text = resp.text
        # 随机取一段，模拟考古现场的“碎片感”
        start_idx = random.randint(0, max(0, len(full_text) - 2000))
        content = full_text[start_idx : start_idx + 1200]
        
        # 4. 联动 AI 分析
        analysis = "AI 考古专家正在解析..."
        if API_KEY:
            logger.info(f"正在调用 AI 模型: {MODEL}")
            try:
                ai_r = requests.post(
                    f"{API_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json={
                        "model": MODEL,
                        "messages": [
                            {"role": "system", "content": "你是一位精通互联网历史的考古学家，请分析这段语料的年代感和背景。"},
                            {"role": "user", "content": content}
                        ]
                    },
                    timeout=20
                )
                analysis = ai_r.json()['choices'][0]['message']['content']
            except Exception as ai_err:
                logger.error(f"AI 调用失败: {str(ai_err)}")
                analysis = "AI 连线中断，但这并不影响语料的价值。"

        return jsonify({
            "status": "success",
            "content": content,
            "analysis": analysis,
            "meta": {"source": file_name, "dataset": "WuDao 2.0 Refined"}
        })

    except Exception as e:
        logger.error(f"后端发生致命错误: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
