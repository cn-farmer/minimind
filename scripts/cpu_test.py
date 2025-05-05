import streamlit as st
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime

# 初始化数据存储
if 'cpu_data' not in st.session_state:
    st.session_state.cpu_data = pd.DataFrame(columns=['时间', 'CPU使用率'])

# Streamlit应用界面
st.title('本地CPU使用率实时监控')

# 创建图表和指标占位符
chart_placeholder = st.empty()
metric_placeholder = st.empty()

# 设置自动刷新间隔 (秒)
refresh_interval = st.slider('刷新间隔 (秒)', 0.5, 5.0, 1.0)

# 获取CPU核心数
cpu_count = psutil.cpu_count()
st.sidebar.markdown(f"**系统信息**\n- CPU核心数: {cpu_count}")

# 主循环
while True:
    try:
        # 获取当前CPU使用率
        cpu_percent = psutil.cpu_percent(interval=0.5)
        current_time = datetime.now().strftime('%H:%M:%S')

        # 更新数据
        new_row = pd.DataFrame([[current_time, cpu_percent]],
                               columns=['时间', 'CPU使用率'])
        st.session_state.cpu_data = pd.concat([st.session_state.cpu_data, new_row])

        # 只保留最近30个数据点
        if len(st.session_state.cpu_data) > 30:
            st.session_state.cpu_data = st.session_state.cpu_data.iloc[-30:]

        # 绘制图表
        fig, ax = plt.subplots(figsize=(10, 4))
        st.session_state.cpu_data.plot(x='时间', y='CPU使用率', ax=ax, marker='o')
        ax.set_title('CPU使用率实时监控')
        ax.set_ylabel('使用率 (%)')
        ax.set_ylim(0, 100)
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # 更新图表和指标
        chart_placeholder.pyplot(fig)
        metric_placeholder.metric("当前CPU使用率", f"{cpu_percent}%")
        plt.close(fig)

    except Exception as e:
        st.error(f"获取数据时出错: {e}")

    # 等待指定时间后刷新
    time.sleep(refresh_interval)