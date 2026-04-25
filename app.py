from flask import Flask, request, jsonify, send_from_directory, Response
import csv
import json
import os
import hashlib
import secrets
from datetime import datetime
from agent import create_jobfit_agent, stream_jobfit_agent
from langchain.messages import HumanMessage

app = Flask(__name__, static_folder='frontend')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(BASE_DIR, 'database')
USERS_FILE = os.path.join(DATABASE_DIR, 'users.csv')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')

os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

agent_instance = None
agent_config = {"configurable": {"thread_id": "jobfit_user_session"}}


def get_agent():
    global agent_instance
    if agent_instance is None:
        agent_instance = create_jobfit_agent()
    return agent_instance


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def init_users_file():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['username', 'password_hash', 'created_at'])


def user_exists(username: str) -> bool:
    if not os.path.exists(USERS_FILE):
        return False
    with open(USERS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['username'] == username:
                return True
    return False


def verify_user(username: str, password: str) -> bool:
    if not os.path.exists(USERS_FILE):
        return False
    password_hash = hash_password(password)
    with open(USERS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['username'] == username and row['password_hash'] == password_hash:
                return True
    return False


@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')


@app.route('/index.html')
def index_page():
    return send_from_directory('frontend', 'index.html')


@app.route('/register.html')
def register_page():
    return send_from_directory('frontend', 'register.html')


@app.route('/dashboard')
def dashboard():
    return send_from_directory('frontend', 'dashboard.html')


@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({'error': '用户名和密码不能为空'}), 400

    if len(username) < 3:
        return jsonify({'error': '用户名至少需要3个字符'}), 400

    if len(password) < 6:
        return jsonify({'error': '密码至少需要6个字符'}), 400

    init_users_file()

    if user_exists(username):
        return jsonify({'error': '用户名已存在'}), 400

    password_hash = hash_password(password)
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(USERS_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([username, password_hash, created_at])

    return jsonify({'message': '注册成功'}), 200


@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({'error': '用户名和密码不能为空'}), 400

    init_users_file()

    if verify_user(username, password):
        token = secrets.token_hex(16)
        return jsonify({
            'token': token,
            'username': username,
            'message': '登录成功'
        }), 200
    else:
        return jsonify({'error': '用户名或密码错误'}), 401


@app.route('/api/agent/chat', methods=['POST'])
def agent_chat():
    data = request.get_json()
    message = data.get('message', '').strip()
    resume_path = data.get('resume_path', '')
    jd_path = data.get('jd_path', '')

    if not message:
        return jsonify({'error': '消息不能为空'}), 400

    user_input = message
    if resume_path and jd_path:
        user_input = f"{message}。简历路径: {resume_path}, JD路径: {jd_path}"
    elif resume_path:
        user_input = f"{message}。简历路径: {resume_path}"
    elif jd_path:
        user_input = f"{message}。JD路径: {jd_path}"

    import time
    print(f"[{time.strftime('%H:%M:%S')}] 开始处理请求: {message[:50]}...")

    def generate():
        try:
            agent = get_agent()
            chunk_count = 0
            for chunk in stream_jobfit_agent(agent, user_input, agent_config):
                chunk_count += 1
                print(f"[{time.strftime('%H:%M:%S')}] 收到chunk {chunk_count}: {str(chunk)[:100]}...")
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            print(f"[{time.strftime('%H:%M:%S')}] 流式响应完成，共 {chunk_count} 个chunk")
            yield f"data: {json.dumps({'type': 'done', 'content': ''}, ensure_ascii=False)}\n\n"
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] 错误: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/upload/<file_type>', methods=['POST'])
def upload_file(file_type):
    if file_type not in ['resume', 'jd']:
        return jsonify({'error': '无效的文件类型'}), 400

    if 'file' not in request.files:
        return jsonify({'error': '没有文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    username = request.form.get('username', 'anonymous')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{username}_{file_type}_{timestamp}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    file.save(filepath)

    return jsonify({
        'path': filepath,
        'filename': filename,
        'message': '上传成功'
    }), 200


if __name__ == '__main__':
    init_users_file()
    print('=' * 50)
    print('JobFit Agent 后端服务已启动')
    print('访问地址: http://localhost:5000')
    print('=' * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)
