import pandas as pd  # <--- 新增导入 pandas

# ==========================================
# 1. 侧边栏：智能密钥与数据导入
# ==========================================
with st.sidebar:
    st.header("⚙️ 系统配置")
    user_key = st.text_input("请输入 API Key (访客请留空):", type="password", value="")
    base_url = "https://api.siliconflow.cn/v1"
    
    st.divider() # 画一条分割线
    
    # 🌟 新增：动态数据导入功能 🌟
    st.header("📂 导入业务数据")
    uploaded_file = st.file_uploader("上传您的 CSV 数据表", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # 1. 用 pandas 读取上传的 CSV 表格
            df = pd.read_csv(uploaded_file)
            
            # 2. 提取文件名作为数据库里的表名（比如传了 battery.csv，表名就是 battery）
            table_name = uploaded_file.name.split('.')[0] 
            
            # 3. 将表格数据直接写入本地 SQLite 数据库
            conn = sqlite3.connect("car_data.db")
            # if_exists='replace' 表示如果同名表已存在，就覆盖它
            df.to_sql(table_name, conn, if_exists='replace', index=False) 
            conn.close()
            
            st.success(f"✅ 数据表 `{table_name}` 导入成功！共 {len(df)} 行数据。")
            st.info("💡 Agent 的视觉已更新，您可以直接提问该表的内容了！")
            
        except Exception as e:
            st.error(f"导入失败: {e}")

# (智能选择 Key 的逻辑保持不变...)
if user_key:
    api_key = user_key
else:
    try:
        api_key = st.secrets["SILICONFLOW_API_KEY"]
    except Exception:
        api_key = ""
