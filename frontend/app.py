import json

import requests
import streamlit as st

st.set_page_config(page_title="教学 Agent 演示", page_icon="🧠", layout="wide")
st.title("核桃智能教学诊断与干预 Agent 平台（Demo）")

api_url = st.text_input("FastAPI 地址", value="http://127.0.0.1:8000/agent/run")
request_text = st.text_area(
    "输入教师请求",
    value="请帮我诊断 student_id:1001 在循环知识点上的薄弱点，并给出干预与补练建议。",
    height=120,
)

if st.button("运行 Agent"):
    with st.spinner("处理中..."):
        try:
            resp = requests.post(api_url, json={"request_text": request_text}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            st.subheader("结构化结果")
            st.json(data)
            st.subheader("最终回复")
            st.markdown(data.get("final_response", ""))
            with st.expander("调试详情（debug_trace）", expanded=False):
                debug_trace = data.get("debug_trace", [])
                st.write(f"节点追踪条数: {len(debug_trace)}")
                st.json(debug_trace)
            with st.expander("证据摘要（evidence_summary）", expanded=False):
                st.json(data.get("evidence_summary", {}))
        except Exception as exc:
            st.error(f"调用失败: {exc}")

st.caption("说明：当前为本地规则 + 本地数据检索的可运行骨架。")
