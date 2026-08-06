"""
方太个性化膳食规划系统 - Gradio Web 界面（豆包风格）

用法：
    python gradio_app.py
启动后在浏览器打开 http://127.0.0.1:7861
"""

import json
import uuid
import requests
import gradio as gr

API_BASE = "http://127.0.0.1:5000"

# ============================================================
# CSS
# ============================================================

CUSTOM_CSS = """
/* 全局 */
footer { display: none !important; }
.gradio-container { max-width: 100% !important; }

/* 左侧边栏 - 浅色 */
.sidebar-col {
    background: #fafafa !important;
    border-right: 1px solid #e8e8e8 !important;
    padding: 20px 16px !important;
    min-height: 100vh;
}

/* 侧边栏中的文字 */
.sidebar-col label, .sidebar-col .gr-radio label span,
.sidebar-col p, .sidebar-col h3, .sidebar-col span {
    color: #333 !important;
}
.sidebar-col h1, .sidebar-col h2 {
    color: #111 !important;
}

/* 侧边栏 Radio */
.sidebar-col .gr-radio {
    background: transparent !important;
    border: none !important;
}
.sidebar-col input[type="radio"] {
    accent-color: #4f6ef7;
}

/* 侧边栏 Dropdown */
.sidebar-col select {
    background: #fff !important;
    color: #333 !important;
    border: 1px solid #ddd !important;
    border-radius: 6px !important;
}

/* 侧边栏按钮 */
.sidebar-btn {
    background: #fff !important;
    border: 1px solid #e0e0e0 !important;
    color: #555 !important;
    font-size: 13px !important;
    padding: 6px 12px !important;
    cursor: pointer;
    text-align: left !important;
    border-radius: 6px !important;
    transition: 0.15s;
}
.sidebar-btn:hover {
    background: #f0f0f0 !important;
    color: #333 !important;
}

/* 主对话区 */
.chat-col {
    background: #f8f8f8 !important;
    padding: 0 !important;
}

/* Chatbot */
.chatbot-panel {
    border: none !important;
    background: transparent !important;
}
.chatbot-panel .message-row {
    padding: 4px 0;
}

/* 用户气泡 */
.chatbot-panel .user {
    justify-content: flex-end;
}
.chatbot-panel .user .bubble {
    background: #4f6ef7 !important;
    color: #fff !important;
    border-radius: 14px 4px 14px 14px !important;
    padding: 10px 15px !important;
    font-size: 14px;
}

/* 助手气泡 */
.chatbot-panel .bot {
    justify-content: flex-start;
}
.chatbot-panel .bot .bubble {
    background: #fff !important;
    color: #222 !important;
    border-radius: 4px 14px 14px 14px !important;
    padding: 10px 15px !important;
    font-size: 14px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}

/* 输入区 */
.input-row {
    border-top: 1px solid #e8e8e8;
    background: #fff;
    padding: 12px 24px;
}
.input-row textarea {
    border: 1px solid #e0e0e0 !important;
    border-radius: 12px !important;
    padding: 10px 16px !important;
    font-size: 14px !important;
    background: #f5f5f5 !important;
    resize: none !important;
}
.input-row textarea:focus {
    border-color: #4f6ef7 !important;
    background: #fff !important;
    box-shadow: 0 0 0 3px rgba(79,110,247,0.1) !important;
}

/* 侧边栏分割线和说明 */
.sidebar-divider {
    border-top: 1px solid #2a2a30;
    margin: 12px 0;
}
"""

# ============================================================
# 后端通信
# ============================================================


class ChatSession:
    """管理单个用户的对话会话"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.session = requests.Session()

    def send(self, message: str) -> str:
        try:
            resp = self.session.post(
                f"{API_BASE}/api/dialog",
                json={"user_id": self.user_id, "message": message},
                timeout=60,
            )
            if resp.status_code != 200:
                return f"**[错误]** HTTP {resp.status_code}"

            data = resp.json()
            if not data.get("success"):
                return f"**[错误]** {data.get('error', '未知错误')}"

            info = data.get("data", {})
            intent = info.get("intent", "unknown")
            response = info.get("response", "")
            recommendations = info.get("recommendations", [])
            is_onboarding = info.get("is_onboarding", False)
            step = info.get("onboarding_step", 0)

            return self._format_response(intent, response, recommendations, is_onboarding, step)

        except requests.ConnectionError:
            return "**[错误]** 无法连接后端服务，请先启动 `python app.py`"
        except Exception as e:
            return f"**[错误]** {e}"

    def _format_response(self, intent: str, response: str,
                         recommendations: list, is_onboarding: bool,
                         step: int) -> str:
        lines = []

        if intent == "onboarding" or is_onboarding:
            step_labels = {1: "过敏信息", 2: "口味偏好", 3: "健康目标"}
            label = step_labels.get(step, "")
            if label:
                lines.append(f"### {label}")
            lines.append(response)
            return "\n\n".join(lines)

        intent_labels = {
            "recommend": "推荐", "add_constraint": "调整",
            "request_substitute": "替换", "reject_recommendation": "否定",
            "ask_clarification": "追问", "vague_query": "引导",
        }
        label = intent_labels.get(intent, "")

        if recommendations:
            lines.append(f"### {label} 推荐菜品\n")
            for i, r in enumerate(recommendations, 1):
                name = r.get("name", "未知")
                tags = r.get("tags", [])
                tag_str = " | ".join(tags[:4]) if tags else ""
                cals = r.get("nutrition", {}).get("calories", "")
                parts = []
                if cals:
                    parts.append(f"{cals}kcal")
                if tag_str:
                    parts.append(tag_str)
                detail = " | ".join(parts)
                lines.append(f"**{i}. {name}**  ")
                if detail:
                    lines.append(f"*{detail}*")
                lines.append("")

        if response:
            if recommendations:
                lines.append("---\n")
            lines.append(f"> {response}")

        return "\n".join(lines) if lines else response

    def reset(self):
        try:
            self.session.post(
                f"{API_BASE}/api/dialog",
                json={"user_id": self.user_id, "message": "reset", "reset": True},
                timeout=10,
            )
        except Exception:
            pass


_sessions: dict[str, ChatSession] = {}


def get_or_create_session(user_id: str) -> ChatSession:
    if user_id not in _sessions:
        _sessions[user_id] = ChatSession(user_id)
    return _sessions[user_id]


# ============================================================
# 事件函数
# ============================================================


# 检测 Gradio 版本，选择消息格式
try:
    from packaging.version import parse as _parse
    _USE_DICT = _parse(gr.__version__) >= _parse("5.0")
except Exception:
    _USE_DICT = False  # 旧版默认元组


def chat_fn(message: str, history: list, user_id: str):
    """处理消息"""
    if not message.strip():
        return history, ""

    history = list(history)
    try:
        session = get_or_create_session(user_id)
        reply = session.send(message.strip())
    except Exception as e:
        reply = f"发送失败: {e}"

    if _USE_DICT:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
    else:
        history.append((message, reply))
    return history, ""


def on_user_switch(user_type: str, preset_id: str):
    """切换用户，返回新 uid + 状态文本"""
    if user_type == "新用户":
        uid = "new_" + uuid.uuid4().hex[:8]
    else:
        uid = preset_id.strip() or "1"
        if not (uid.isdigit() and 1 <= int(uid) <= 50):
            uid = "1"
    if uid in _sessions:
        _sessions[uid].reset()
        del _sessions[uid]

    status = f"预置用户 {uid}" if user_type == "预置用户" else "新用户（自动引导填写偏好）"
    return uid, [], "", status


def on_reset(user_id: str):
    if user_id in _sessions:
        _sessions[user_id].reset()
        del _sessions[user_id]
    return [], ""


# ============================================================
# 界面
# ============================================================

with gr.Blocks(title="方太个性化膳食规划系统") as demo:

    current_user = gr.State("new_" + uuid.uuid4().hex[:8])

    with gr.Row(equal_height=True):
        # ===== 左侧边栏 =====
        with gr.Column(scale=1, min_width=220, elem_classes="sidebar-col"):
            gr.Markdown("## 方太膳食规划")

            user_type = gr.Radio(
                ["新用户", "预置用户"],
                label="用户类型",
                value="新用户",
                interactive=True,
            )
            preset_id = gr.Dropdown(
                choices=[str(i) for i in range(1, 51)],
                label="预置用户 ID",
                value="1",
                interactive=True,
                visible=False,
            )

            def toggle_preset(utype):
                return gr.Dropdown(visible=(utype == "预置用户"))

            user_type.change(toggle_preset, user_type, preset_id)

            switch_btn = gr.Button("切换用户", elem_classes="sidebar-btn")
            user_status = gr.Markdown("新用户（自动引导填写偏好）")

            reset_btn = gr.Button("新对话", elem_classes="sidebar-btn")

            gr.Markdown("""
            ---
            新用户直接对话即可，系统会自动引导填写过敏、口味、健康信息。

            预置用户选择 ID 后直接开始推荐。
            """)

        # ===== 主对话区 =====
        with gr.Column(scale=3, min_width=480, elem_classes="chat-col"):
            chatbot = gr.Chatbot(
                label="",
                height=580,
                show_label=False,
                elem_classes="chatbot-panel",
            )

            with gr.Row(elem_classes="input-row"):
                msg_input = gr.Textbox(
                    placeholder="输入你想吃的，比如「今晚吃什么？」...",
                    show_label=False,
                    scale=10,
                )

    # ===== 事件绑定 =====
    msg_input.submit(
        chat_fn, [msg_input, chatbot, current_user], [chatbot, msg_input]
    )

    switch_btn.click(
        on_user_switch, [user_type, preset_id],
        [current_user, chatbot, msg_input, user_status]
    )

    reset_btn.click(
        on_reset, [current_user], [chatbot, msg_input]
    )


if __name__ == "__main__":
    import traceback, sys

    print("=" * 50)
    print("  方太个性化膳食规划系统 - Web 界面（豆包风格）")
    print(f"  Python: {sys.version}")
    print(f"  后端: {API_BASE}")
    print("  界面: http://127.0.0.1:7861")
    print("=" * 50)
    try:
        import gradio as _gr
        print(f"  Gradio: {_gr.__version__}")
    except Exception:
        pass

    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=7861,
            share=False,
            show_error=True,
            css=CUSTOM_CSS,
        )
    except Exception as e:
        print(f"\n启动失败: {e}")
        traceback.print_exc()
        input("按回车退出...")
