import streamlit as st
import os
import sqlite3
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent

# ==========================================
# 0. 页面配置与 UI 设计
# ==========================================
st.set_page_config(page_title="车联网数据分析 Agent", page_icon="🚗", layout="wide")
st.title("🚗 问界智造 - 车联网与产线数据分析 Agent")
st.markdown("直接使用自然语言查询车辆测试数据，系统将自动生成 SQL 并进行深度分析。")

# ==========================================
# 1. 侧边栏：智能密钥配置
# ==========================================
with st.sidebar:
    st.header("⚙️ 系统配置")
    user_key = st.text_input("请输入 API Key (访客请留空，使用内置通道):", type="password", value="")
    base_url = "https://api.siliconflow.cn/v1"
    st.info("💡 提示：后台连接 SQLite 汽车制造数据库。使用 72B 旗舰模型保障逻辑推理能力。")

# 智能选择 Key
if user_key:
    api_key = user_key
else:
    try:
        api_key = st.secrets["SILICONFLOW_API_KEY"]
    except Exception:
        api_key = ""
        st.warning("⚠️ 未检测到系统内置密钥，请手动输入 API Key。")


# ==========================================
# 2. 核心功能：云端自动初始化数据库
# ==========================================
# 如果是在云端运行，找不到 db 文件，就自动建一个！
def ensure_database_exists():
    if not os.path.exists("car_data.db"):
        conn = sqlite3.connect("car_data.db")
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE vehicles (vin TEXT PRIMARY KEY, model TEXT, production_date DATE)''')
        cursor.execute(
            '''CREATE TABLE test_logs (log_id INTEGER PRIMARY KEY, vin TEXT, test_date DATE, battery_temp REAL, motor_speed INTEGER, test_result TEXT)''')

        # 插入模拟数据
        vehicles = [('VIN001', '问界M5', '2026-03-01'), ('VIN002', '问界M7', '2026-03-02'),
                    ('VIN003', '问界M9', '2026-03-02')]
        logs = [('VIN001', '2026-03-05', 45.2, 16000, 'Pass'), ('VIN002', '2026-03-06', 82.5, 15500, 'Fail'),
                ('VIN003', '2026-03-06', 38.0, 18000, 'Pass')]

        cursor.executemany("INSERT INTO vehicles VALUES (?, ?, ?)", vehicles)
        cursor.executemany(
            "INSERT INTO test_logs (vin, test_date, battery_temp, motor_speed, test_result) VALUES (?, ?, ?, ?, ?)",
            logs)
        conn.commit()
        conn.close()


# 确保数据库存在，然后连接
ensure_database_exists()


@st.cache_resource
def init_db():
    return SQLDatabase.from_uri("sqlite:///car_data.db")


db = init_db()

# ==========================================
# 3. 聊天交互界面
# ==========================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_question = st.chat_input("例如：帮我查一下，测试结果为 Fail 的车辆，平均电池温度是多少？")

if user_question:
    if not api_key:
        st.error("⚠️ 请先输入 API Key，或确保云端 Secrets 已配置！")
        st.stop()

    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.chat_history.append({"role": "user", "content": user_question})

    # 组装超级大脑
    llm = ChatOpenAI(api_key=api_key, base_url=base_url, model="Qwen/Qwen2.5-72B-Instruct", temperature=0)

    agent_executor = create_sql_agent(llm=llm, db=db, agent_type="tool-calling", verbose=True)

    # 强力 Prompt 注入
    augmented_question = f"{user_question}\n\n(系统指令：请必须使用中文，以专业的数据分析师口吻向业务人员汇报结果。直接给出最终答案，不要展示SQL语句。)"

    with st.chat_message("assistant"):
        with st.spinner("🤖 Agent 正在读取数据库结构并执行查询，请稍候..."):
            try:
                response = agent_executor.invoke({"input": augmented_question})
                ai_answer = response["output"]
                st.markdown(ai_answer)
                st.session_state.chat_history.append({"role": "assistant", "content": ai_answer})
            except Exception as e:
                st.error(f"分析失败: {e}")