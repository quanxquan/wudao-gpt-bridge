import os
import random
import requests
import logging
from flask import Flask, jsonify
from flask_cors import CORS
from modelscope.hub.api import HubApi
from modelscope.msdatasets import MsDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# 彻底放开跨域限制，确保 Lovable 预览环境通行无阻
CORS(app, resources={r"/*": {"origins": "*"}})

api = HubApi()
DATASET_ID = os.environ.get('DATASET_ID', 'whynlp/WuDaoCorpus-200G-shuffled')

@app.route('/get_random')
def get_random():
    try:
        # 1. 仅获取元数据文件列表，不下载
        files = api.get_dataset_files(dataset_id=DATASET_ID, revision='master')
        data_files = [f for f in files if f.startswith('data/') and f.endswith('.jsonl')]
        
        if not data_files:
            return jsonify({"status": "error", "message": "No data files found"}), 404
            
        target_file = random.choice(data_files)
        logger.info(f"Targeting: {target_file}")
        
        # 2. 【核心修改】启用 use_streaming=True
        # 这会防止 ModelScope 在 Cloud Run 有限的磁盘里尝试下载整个分片
        ds = MsDataset.load(
            DATASET_ID, 
            data_files=target_file, 
            split='train', 
            use_streaming=True  # 👈 救命的一行
        )
        
        # 3. 只取第一条数据
        item = next(iter(ds))
        content = item.get('content', '内容为空')
        
        # 这里的 AI 调用部分你可以先注释掉测试数据，等数据通了再开
        # analysis = ask_ai_archeologist(content) 
        
        return jsonify({
            "status": "success",
            "content": content[:2000], # 截取前2000字防止 JSON 过大
            "meta": {"source": target_file}
        })
    except Exception as e:
        logger.error(f"Fatal Error: {str(e)}")
        return jsonify({"status": "error", "message": f"魔搭连接失败: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
