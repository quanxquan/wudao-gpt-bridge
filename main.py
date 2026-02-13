import os
import random
from flask import Flask, jsonify
from flask_cors import CORS
from modelscope.hub.api import HubApi
from modelscope.msdatasets import MsDataset

app = Flask(__name__)
CORS(app)  # 允许 Lovable 前端跨域请求

api = HubApi()
DATASET_ID = 'whynlp/WuDaoCorpus-200G-shuffled'

@app.route('/get_random')
def get_random():
    try:
        # 1. 获取所有数据分片列表
        files = api.get_dataset_files(dataset_id=DATASET_ID, revision='master')
        data_files = [f for f in files if f.startswith('data/') and f.endswith('.jsonl')]
        
        # 2. 随机抽取一个分片
        target_file = random.choice(data_files)
        
        # 3. 加载该分片的一条数据 (为了速度，我们只取第一条)
        ds = MsDataset.load(DATASET_ID, data_files=target_file, split='train')
        item = next(iter(ds))
        
        return jsonify({
            "status": "success",
            "title": item.get('title', '无标题语料'),
            "content": item.get('content', '内容为空'),
            "file_source": target_file
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
