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
/* ============ Codex 白色风格主题 ============ */
html, body {
    background: #fafafa !important;
    color: #1a1a1a !important;
}
:root {
    --body-background-fill: #fafafa !important;
    --body-text-color: #1a1a1a !important;
    --body-subtext-color: #6b7280 !important;
    --block-background-fill: #ffffff !important;
    --block-border-color: #e5e7eb !important;
    --block-border-width: 1px !important;
    --block-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
    --input-background-fill: #ffffff !important;
    --input-border-color: #d1d5db !important;
    --input-text-color: #1a1a1a !important;
    --input-placeholder-color: #9ca3af !important;
    --button-primary-background-fill: #111827 !important;
    --button-primary-text-color: #ffffff !important;
    --button-primary-background-fill-hover: #1f2937 !important;
    --button-secondary-background-fill: #ffffff !important;
    --button-secondary-text-color: #111827 !important;
    --button-secondary-background-fill-hover: #f3f4f6 !important;
    --button-secondary-border-color: #d1d5db !important;
    --border-color-primary: #e5e7eb !important;
    --checkbox-background-color: #ffffff !important;
    --checkbox-background-color-selected: #111827 !important;
    --checkbox-label-text-color: #111827 !important;
    --checkbox-label-text-color-selected: #ffffff !important;
    --checkbox-border-color: #d1d5db !important;
    --checkbox-border-color-selected: #111827 !important;
    --link-text-color: #2563eb !important;
    --body-font: "Inter", -apple-system, "Segoe UI", "PingFang SC",
        "Microsoft YaHei", "Helvetica Neue", sans-serif !important;
}
.gradio-container {
    background: #fafafa !important;
    color: #1a1a1a !important;
    max-width: 100% !important;
}
footer { display: none !important; }

/* 侧边栏 */
.sidebar-col {
    background: #ffffff !important;
    border-right: 1px solid #e5e7eb !important;
    padding: 22px 18px !important;
    min-height: 100vh;
}
.sidebar-col label, .sidebar-col .gr-radio label span,
.sidebar-col p, .sidebar-col h3, .sidebar-col span,
.sidebar-col h1, .sidebar-col h2 {
    color: #111827 !important;
}
.sidebar-col .gr-radio {
    background: transparent !important;
    border: none !important;
}
.sidebar-col input[type="radio"] { accent-color: #111827; }
/* 单选框选项（gradio6 用 label 渲染，选中项 class=selected） */
.sidebar-col label {
    background: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    padding: 6px 10px !important;
    transition: all 0.15s ease;
}
.sidebar-col label:hover { border-color: #9ca3af !important; }
.sidebar-col label.selected {
    background: #f3f4f6 !important;
    color: #111827 !important;
    border-color: #111827 !important;
}
.sidebar-col select {
    background: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    padding: 4px 8px !important;
}
.sidebar-btn {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    color: #111827 !important;
    font-size: 13px !important;
    padding: 8px 14px !important;
    border-radius: 8px !important;
    transition: all 0.15s ease;
}
.sidebar-btn:hover {
    background: #f3f4f6 !important;
    color: #000000 !important;
    border-color: #9ca3af !important;
}

/* 主对话区 */
.chat-col { background: #fafafa !important; padding: 0 !important; }
.chatbot-panel { border: none !important; background: transparent !important; }
.chatbot-panel .message-row { padding: 6px 0; }

/* 用户气泡 */
.chatbot-panel .user { justify-content: flex-end; }
.chatbot-panel .user .bubble {
    background: #e9edf2 !important;
    color: #111827 !important;
    border: 1px solid #dde3ea !important;
    border-radius: 12px 4px 12px 12px !important;
    padding: 10px 15px !important;
    font-size: 14px;
}

/* 助手气泡 */
.chatbot-panel .bot { justify-content: flex-start; }
.chatbot-panel .bot .bubble {
    background: #ffffff !important;
    color: #1f2937 !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 4px 12px 12px 12px !important;
    padding: 12px 16px !important;
    font-size: 14px;
    line-height: 1.65;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}

/* 输入区 */
.input-row {
    border-top: 1px solid #e5e7eb;
    background: #fafafa;
    padding: 14px 24px;
}
.input-row textarea {
    border: 1px solid #d1d5db !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    font-size: 14px !important;
    background: #ffffff !important;
    color: #111827 !important;
    resize: none !important;
}
.input-row textarea:focus {
    border-color: #111827 !important;
    background: #ffffff !important;
    box-shadow: 0 0 0 3px rgba(17, 24, 39, 0.06) !important;
}
.input-row textarea::placeholder { color: #9ca3af !important; }
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

            info = data.get("data") or data  # 兼容响应可能带/不带 data 包装
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
            "ask_clarification": "追问", "vague_query": "推荐",
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
