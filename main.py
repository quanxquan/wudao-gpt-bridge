import os
import random
import requests
import logging
from flask import Flask, jsonify
from flask_cors import CORS
from modelscope.hub.api import HubApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

api = HubApi()

# 尝试更换为维护更好的数据集
DATASET_ID = os.environ.get('DATASET_ID', 'LLM-Research/ChiS_Chinese_Common_Crawl')

@app.route('/get_random')
def get_random():
    try:
        # 1. 只获取文件列表
        files = api.get_dataset_files(dataset_id=DATASET_ID, revision='master')
        # 过滤出 jsonl 或 txt 文件
        data_files = [f for f in files if f.endswith(('.jsonl', '.txt', '.json'))]
        
        if not data_files:
            return jsonify({"status": "error", "message": "未找到有效语料文件"}), 404
            
        target_file = random.choice(data_files)
        
        # 2. 获取文件的直链（绕过 MsDataset 加载器）
        file_url = f"https://modelscope.cn/api/v1/datasets/{DATASET_ID}/repo/temp?FilePath={target_file}"
        
        # 3. 抓取部分内容
        # 使用 Stream 模式只读前 50KB，防止内存溢出
        response = requests.get(file_url, stream=True, timeout=10)
        content_chunk = response.raw.read(50000).decode('utf-8', errors='ignore')
        
        # 简单的清洗：取中间的一段，防止取到文件头的元数据
        lines = content_chunk.split('\n')
        final_content = lines[len(lines)//2] if len(lines) > 2 else content_chunk

        return jsonify({
            "status": "success",
            "content": final_content[:2000],
            "meta": {"source": target_file, "dataset": DATASET_ID}
        })
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"status": "error", "message": f"挖掘失败: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
