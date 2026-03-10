import streamlit as st
import os
import sqlite3
import pandas as pd
import json
import re
import plotly.express as px
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent

# ==========================================
# 0. 页面配置与 UI 设计
# ==========================================
st.set_page_config(page_title="车联网数据分析 Agent", page_icon="🚗", layout="wide")
st.title("🚗 问界智造 - 车联网与产线数据分析 Agent")
st.markdown("支持 **原生级多轮对话记忆**、**语义路由** 与 **动态可视化**。可直接提问或上传 CSV 业务表。")

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

if user_key:
    api_key = user_key
else:
    api_key = st.secrets.get("SILICONFLOW_API_KEY", "")

# ==========================================
# 2. 数据库初始化
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
# 3. 聊天交互逻辑 (原生记忆版)
# ==========================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_question = st.chat_input("例如：帮我查一下各车型的平均电池温度，并画个图。")

if user_question:
    if not api_key:
        st.error("⚠️ 请在左侧配置 API Key！")
        st.stop()

    st.chat_message("user").write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("🔍 扫描业务表..."):
            router_llm = ChatOpenAI(api_key=api_key, base_url=base_url, model="Qwen/Qwen2.5-7B-Instruct", temperature=0)
            routing_prompt = f"当前有表: vehicles, test_logs。用户提问: {user_question}。只需输出需要的表名。"
            relevant_tables = router_llm.invoke(routing_prompt).content.strip()
            st.caption(f"🎯 路由锁定：`{relevant_tables}`")

        with st.spinner("🤖 正在结合历史对话深度思考..."):
            llm_main = ChatOpenAI(api_key=api_key, base_url=base_url, model="deepseek-ai/DeepSeek-V3", temperature=0)
            
            # 移除了脆弱的 memory 模块
            agent_executor = create_sql_agent(
                llm=llm_main, 
                db=db, 
                agent_type="tool-calling", 
                verbose=True,
                handle_parsing_errors=True
            )

            # 🌟 硬核操作：手动提取最后 4 条聊天记录拼接成记忆字符串 🌟
            history_context = ""
            if len(st.session_state.chat_history) > 0:
                history_list = [f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-4:]]
                history_context = "\n".join(history_list)

            # 🌟 将记忆字符串直接注入 Prompt 🌟
            final_prompt = f"""
            【历史对话上下文】:
            {history_context if history_context else "这是第一轮对话，无历史上下文。"}

            【当前用户问题】: {user_question}
            【当前可用参考表】: {relevant_tables}
            
            【系统强制指令】:
            1. 请必须结合上述【历史对话上下文】来理解用户的【当前问题】（比如遇到“那M7呢”这种追问时，要知道上下文在聊什么）。
            2. 中文汇报结果，严禁展示 SQL。
            3. 若涉及画图，结尾附带 JSON：```json {{"type": "line/bar", "labels": [], "values": []}} ```
            """

            try:
                response = agent_executor.invoke({"input": final_prompt})
                full_res = response["output"]

                # 记录用户提问到列表 (放在请求成功后，防止脏数据)
                st.session_state.chat_history.append({"role": "user", "content": user_question})

                clean_text = re.sub(r'```json\n(.*?)\n```', '', full_res, flags=re.DOTALL)
                clean_text = re.sub(r'以下.*?JSON数据.*?[：:]', '', clean_text).strip()
                st.markdown(clean_text)
                
                # 记录 AI 回复到列表
                st.session_state.chat_history.append({"role": "assistant", "content": clean_text})

                json_match = re.search(r'```json\n(.*?)\n```', full_res, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(1))
                    df_plot = pd.DataFrame({"维度": data["labels"], "数值": data["values"]})
                    st.divider()
                    if data.get("type") == "line":
                        fig = px.line(df_plot, x="维度", y="数值", markers=True, text="数值", title="📈 趋势分析视图")
                    else:
                        fig = px.bar(df_plot, x="维度", y="数值", color="维度", text_auto='.2f', title="📊 分类统计视图")
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"分析出错: {e}")

