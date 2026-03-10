import streamlit as st
import os
import sqlite3
import pandas as pd
import json
import re
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import plotly.express as px
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent

# ==========================================
# 0. 页面配置与 UI 设计
# ==========================================
st.set_page_config(page_title="车联网数据分析 Agent", page_icon="🚗", layout="wide")
st.title("🚗 问界智造 - 车联网与产线数据分析 Agent")
st.markdown("直接提问查询车辆测试数据，或上传 CSV 进行深度分析。系统支持 **语义路由** 与 **动态可视化**。")

# ==========================================
# 1. 侧边栏：配置与数据导入
# ==========================================
with st.sidebar:
    st.header("⚙️ 系统配置")
    user_key = st.text_input("请输入 API Key (访客请留空):", type="password", value="")
    base_url = "https://api.siliconflow.cn/v1"
    
    st.divider()
    st.header("📂 导入业务数据")
    uploaded_file = st.file_uploader("上传您的 CSV 数据表", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            table_name = uploaded_file.name.split('.')[0] 
            conn = sqlite3.connect("car_data.db")
            df.to_sql(table_name, conn, if_exists='replace', index=False) 
            conn.close()
            st.success(f"✅ 数据表 `{table_name}` 导入成功！")
        except Exception as e:
            st.error(f"导入失败: {e}")

# 智能选择 Key (优先使用用户输入的，其次使用系统 Secrets)
if user_key:
    api_key = user_key
else:
    api_key = st.secrets.get("SILICONFLOW_API_KEY", "")

# ==========================================
# 2. 数据库初始化 (如果不存在则自动建库)
# ==========================================
def ensure_database_exists():
    if not os.path.exists("car_data.db"):
        conn = sqlite3.connect("car_data.db")
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS vehicles (vin TEXT PRIMARY KEY, model TEXT, production_date DATE)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS test_logs (log_id INTEGER PRIMARY KEY, vin TEXT, test_date DATE, battery_temp REAL, motor_speed INTEGER, test_result TEXT)''')
        
        vehicles = [('VIN001', '问界M5', '2026-03-01'), ('VIN002', '问界M7', '2026-03-02'), ('VIN003', '问界M9', '2026-03-02')]
        logs = [('VIN001', '2026-03-05', 45.2, 16000, 'Pass'), ('VIN002', '2026-03-06', 82.5, 15500, 'Fail'), ('VIN003', '2026-03-06', 38.0, 18000, 'Pass')]
        
        cursor.executemany("INSERT OR REPLACE INTO vehicles VALUES (?, ?, ?)", vehicles)
        cursor.executemany("INSERT OR REPLACE INTO test_logs (vin, test_date, battery_temp, motor_speed, test_result) VALUES (?, ?, ?, ?, ?)", logs)
        conn.commit()
        conn.close()

ensure_database_exists()

@st.cache_resource
def get_db_connection():
    return SQLDatabase.from_uri("sqlite:///car_data.db")

db = get_db_connection()

# ==========================================
# 3. 聊天交互逻辑
# ==========================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_question = st.chat_input("例如：帮我画一个折线图，展示各车型的平均电池温度。")

if user_question:
    # --- 步骤 0: 初始化持久化记忆存储 ---
# 这会让 Agent 记住之前的对话，即使是在执行 SQL 时
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(
    chat_memory=msgs, 
    return_messages=True, 
    memory_key="chat_history", 
    output_key="output" # 必须指定，因为 SQL Agent 有多个输出键
)

# 渲染历史聊天记录
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if user_question:
    # --- 步骤 0: 初始化持久化记忆存储 ---
# 这会让 Agent 记住之前的对话，即使是在执行 SQL 时
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(
    chat_memory=msgs, 
    return_messages=True, 
    memory_key="chat_history", 
    output_key="output" # 必须指定，因为 SQL Agent 有多个输出键
)

# 渲染历史聊天记录
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if user_question:
    # 用户输入展示（msgs 会自动处理存储）
    st.chat_message("user").write(user_question)

    # --- 步骤 1: 语义路由 (路由不需要记忆，保持纯净) ---
    with st.chat_message("assistant"):
        with st.spinner("🔍 正在扫描业务表结构..."):
            router_llm = ChatOpenAI(api_key=api_key, base_url=base_url, model="Qwen/Qwen2.5-7B-Instruct", temperature=0)
            routing_prompt = f"当前有表: vehicles, test_logs。用户问题: {user_question}。请只输出需要的表名。"
            relevant_tables = router_llm.invoke(routing_prompt).content.strip()
            st.caption(f"🎯 路由锁定：`{relevant_tables}`")

        # --- 步骤 2: 满血 Agent 执行 (带记忆插件) ---
        with st.spinner("🤖 正在思考上下文并分析数据..."):
            llm_main = ChatOpenAI(api_key=api_key, base_url=base_url, model="Qwen/Qwen2.5-72B-Instruct", temperature=0)
            
            # 🌟 关键点：将 memory 传入 Agent
            agent_executor = create_sql_agent(
                llm=llm_main, 
                db=db, 
                agent_type="tool-calling", 
                verbose=True,
                memory=memory, # <--- 记忆插件注入！
                handle_parsing_errors=True
            )

            final_prompt = f"""
            用户问题: {user_question}
            当前参考表: {relevant_tables}
            
            指令:
            1. 中文汇报。若是对上文的追问（如“那M7呢？”），请结合历史 context 生成 SQL。
            2. 若需绘图，末尾附带 JSON：```json {{"type": "line/bar", "labels": [], "values": []}} ```
            """

            try:
                # 执行并存入记忆
                response = agent_executor.invoke({"input": final_prompt})
                full_res = response["output"]

                # --- 步骤 3: 结果清洗与展示 ---
                clean_text = re.sub(r'```json\n(.*?)\n```', '', full_res, flags=re.DOTALL)
                clean_text = re.sub(r'以下.*?JSON数据.*?[：:]', '', clean_text).strip()
                
                st.markdown(clean_text)
                # 注意：StreamlitChatMessageHistory 会在执行过程中自动记录对话，无需手动 append

                # --- 步骤 4: 动态图表渲染 (保持不变) ---
                json_match = re.search(r'```json\n(.*?)\n```', full_res, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(1))
                    df_plot = pd.DataFrame({"维度": data["labels"], "数值": data["values"]})
                    st.divider()
                    fig = px.line(df_plot, x="维度", y="数值", markers=True) if data.get("type") == "line" else px.bar(df_plot, x="维度", y="数值", color="维度")
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"分析出错: {e}")

