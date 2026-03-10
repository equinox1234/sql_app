import streamlit as st
import os
import sqlite3
import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent

# ==========================================
# 0. 页面配置与 UI 设计
# ==========================================
st.set_page_config(page_title="车联网数据分析 Agent", page_icon="🚗", layout="wide")
st.title("🚗 问界智造 - 车联网与产线数据分析 Agent")
st.markdown("直接使用自然语言查询车辆测试数据，或**上传自定义 CSV 数据表**进行深度分析。")

# ==========================================
# 1. 侧边栏：智能密钥与数据导入
# ==========================================
with st.sidebar:
    st.header("⚙️ 系统配置")
    user_key = st.text_input("请输入 API Key (访客请留空):", type="password", value="")
    base_url = "https://api.siliconflow.cn/v1"
    
    st.divider() # 分割线
    
    # 新增：动态数据导入功能
    st.header("📂 导入业务数据")
    uploaded_file = st.file_uploader("上传您的 CSV 数据表", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # 读取 CSV
            df = pd.read_csv(uploaded_file)
            # 提取文件名作为表名
            table_name = uploaded_file.name.split('.')[0] 
            
            # 写入 SQLite 数据库
            conn = sqlite3.connect("car_data.db")
            df.to_sql(table_name, conn, if_exists='replace', index=False) 
            conn.close()
            
            st.success(f"✅ 数据表 `{table_name}` 导入成功！共 {len(df)} 行数据。")
            st.info("💡 Agent 已掌握该表结构，您可以直接提问了！")
        except Exception as e:
            st.error(f"导入失败: {e}")

# 智能选择 Key
if user_key:
    api_key = user_key
else:
    try:
        api_key = st.secrets["SILICONFLOW_API_KEY"]
    except Exception:
        api_key = ""

# ==========================================
# 2. 核心功能：云端自动初始化内置数据库
# ==========================================
def ensure_database_exists():
    if not os.path.exists("car_data.db"):
        conn = sqlite3.connect("car_data.db")
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE vehicles (vin TEXT PRIMARY KEY, model TEXT, production_date DATE)''')
        cursor.execute('''CREATE TABLE test_logs (log_id INTEGER PRIMARY KEY, vin TEXT, test_date DATE, battery_temp REAL, motor_speed INTEGER, test_result TEXT)''')
        
        vehicles = [('VIN001', '问界M5', '2026-03-01'), ('VIN002', '问界M7', '2026-03-02'), ('VIN003', '问界M9', '2026-03-02')]
        logs = [('VIN001', '2026-03-05', 45.2, 16000, 'Pass'), ('VIN002', '2026-03-06', 82.5, 15500, 'Fail'), ('VIN003', '2026-03-06', 38.0, 18000, 'Pass')]
        
        cursor.executemany("INSERT INTO vehicles VALUES (?, ?, ?)", vehicles)
        cursor.executemany("INSERT INTO test_logs (vin, test_date, battery_temp, motor_speed, test_result) VALUES (?, ?, ?, ?, ?)", logs)
        conn.commit()
        conn.close()

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

    # 组装超级大脑 (Agent)
    llm = ChatOpenAI(api_key=api_key, base_url=base_url, model="Qwen/Qwen2.5-72B-Instruct", temperature=0)
    agent_executor = create_sql_agent(llm=llm, db=db, agent_type="tool-calling", verbose=True)

    # 强力 Prompt 注入
    augmented_question = f"{user_question}\n\n(系统指令：请必须使用中文，以专业的数据分析师口吻向业务人员汇报结果。直接给出最终答案，不要展示SQL语句。)"

    with st.chat_message("assistant"):
        with st.spinner("🤖 Agent 正在探索数据库并执行查询，请稍候..."):
            try:
                response = agent_executor.invoke({"input": augmented_question})
                ai_answer = response["output"]
                st.markdown(ai_answer)
                st.session_state.chat_history.append({"role": "assistant", "content": ai_answer})
            except Exception as e:
                st.error(f"分析失败: {e}")
