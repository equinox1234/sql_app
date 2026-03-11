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
st.set_page_config(page_title="车联网多智能体分析系统", page_icon="🚗", layout="wide")
st.title("🚗 问界智造 - 多智能体 (Multi-Agent) 数据大屏")
st.markdown("基于 **Supervisor -> Data Expert -> BI Analyst** 协作管线，支持原生多轮对话与动态可视化。")

# ==========================================
# 1. 系统配置与数据库初始化 (保持不变)
# ==========================================
with st.sidebar:
    st.header("⚙️ 系统配置")
    user_key = st.text_input("请输入 API Key (访客留空):", type="password", value="")
    base_url = "https://api.siliconflow.cn/v1"
    
    st.divider()
    st.header("📂 导入业务数据")
    uploaded_file = st.file_uploader("上传 CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            table_name = uploaded_file.name.split('.')[0] 
            conn = sqlite3.connect("car_data.db")
            df.to_sql(table_name, conn, if_exists='replace', index=False) 
            conn.close()
            st.success(f"✅ 表 `{table_name}` 导入成功！")
        except Exception as e:
            st.error(f"导入失败: {e}")

api_key = user_key if user_key else st.secrets.get("SILICONFLOW_API_KEY", "")

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
# 🌟 2. 核心架构：多智能体协作网络 (Multi-Agent Pipeline) 🌟
# ==========================================

class MultiAgentSystem:
    def __init__(self, api_key, base_url, db_instance):
        self.api_key = api_key
        self.base_url = base_url
        self.db = db_instance
        # 共享记忆流 (State)
        self.state = {
            "question": "",
            "history": "",
            "relevant_tables": "",
            "raw_analysis": "",
            "chart_json": None,
            "final_report": ""
        }

    def node_supervisor(self):
        """主管智能体：负责理解意图并降噪 (Schema Routing)"""
        router_llm = ChatOpenAI(api_key=self.api_key, base_url=self.base_url, model="Qwen/Qwen2.5-7B-Instruct", temperature=0)
        prompt = f"当前数据库有表: vehicles, test_logs。用户提问: {self.state['question']}。请只输出回答该问题需要用到的表名，逗号分隔。"
        self.state["relevant_tables"] = router_llm.invoke(prompt).content.strip()
        return self.state["relevant_tables"]

    def node_data_expert(self):
        """数据专家智能体：负责 ReAct 闭环与 SQL 查数"""
        llm_main = ChatOpenAI(api_key=self.api_key, base_url=self.base_url, model="deepseek-ai/DeepSeek-V3", temperature=0)
        agent_executor = create_sql_agent(llm=llm_main, db=self.db, agent_type="tool-calling", verbose=True, handle_parsing_errors=True)
        
        prompt = f"""
        【历史记忆】: {self.state['history'] if self.state['history'] else "无"}
        【用户问题】: {self.state['question']}
        【锁定表名】: {self.state['relevant_tables']}
        
        指令：
        1. 结合历史记忆理解用户问题。
        2. 查询数据库并用中文输出专业分析报告，严禁展示SQL。
        3. 如果用户要求画图，必须在末尾附带 JSON：```json {{"type": "line/bar", "labels": [], "values": []}} ```
        """
        response = agent_executor.invoke({"input": prompt})
        self.state["raw_analysis"] = response["output"]

    def node_bi_analyst(self):
        """BI 可视化智能体：负责清洗数据与提取图表配置"""
        raw_text = self.state["raw_analysis"]
        
        # 提取 JSON 并清洗文本
        json_match = re.search(r'```json\n(.*?)\n```', raw_text, re.DOTALL)
        if json_match:
            try:
                self.state["chart_json"] = json.loads(json_match.group(1))
            except Exception:
                pass
                
        clean_text = re.sub(r'```json\n(.*?)\n```', '', raw_text, flags=re.DOTALL)
        self.state["final_report"] = re.sub(r'以下.*?JSON数据.*?[：:]', '', clean_text).strip()

    def run_pipeline(self, user_question, chat_history):
        """执行多智能体工作流"""
        self.state["question"] = user_question
        
        # 组装记忆上下文
        if len(chat_history) > 0:
            self.state["history"] = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-4:]])
            
        # 1. 主管分发任务
        st.caption(f"👮‍♂️ **Supervisor 节点** 已启动，正在锁定数据库 Schema...")
        tables = self.node_supervisor()
        st.caption(f"🎯 Supervisor 路由结果：交由 Data Expert 分析 `{tables}` 表。")
        
        # 2. 数据专家执行
        st.caption(f"🧑‍💻 **Data Expert 节点** 接管任务，正在执行基于 ReAct 的推理与自纠错查询...")
        self.node_data_expert()
        
        # 3. BI 分析师处理视图
        st.caption(f"🎨 **BI Analyst 节点** 正在渲染可视化大屏与格式化报告...")
        self.node_bi_analyst()
        
        return self.state["final_report"], self.state["chart_json"]

# ==========================================
# 3. 聊天交互逻辑
# ==========================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_question = st.chat_input("例如：帮我查一下各车型的平均电池温度，并画个分类统计图。")

if user_question:
    if not api_key:
        st.error("⚠️ 请在左侧配置 API Key！")
        st.stop()

    st.chat_message("user").write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("🤖 多智能体协作网络 (Multi-Agent System) 正在运行..."):
            try:
                # 实例化并运行多智能体系统
                mas = MultiAgentSystem(api_key, base_url, db)
                final_report, chart_json = mas.run_pipeline(user_question, st.session_state.chat_history)

                # 记录用户问题
                st.session_state.chat_history.append({"role": "user", "content": user_question})

                # 展示 BI Analyst 整理后的最终报告
                st.markdown(final_report)
                st.session_state.chat_history.append({"role": "assistant", "content": final_report})

                # 渲染图表
                if chart_json:
                    df_plot = pd.DataFrame({"维度": chart_json["labels"], "数值": chart_json["values"]})
                    st.divider()
                    if chart_json.get("type") == "line":
                        fig = px.line(df_plot, x="维度", y="数值", markers=True, text="数值", title="📈 趋势分析视图")
                    else:
                        fig = px.bar(df_plot, x="维度", y="数值", color="维度", text_auto='.2f', title="📊 分类统计视图")
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"系统运行异常: {e}")

