"""
方太个性化膳食规划系统 - Gradio Web 界面

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
/* 去除 Gradio 品牌水印/徽标（页脚 + 新版 gradio-lite badge + 右上 logo），
   避免界面上出现任何 AI/Gradio 生成水印 */
footer,
.gradio-container footer,
.gradio-container .gradio-lite,
.gradio-container .gradio-lite a,
.gradio-container [data-testid="gradio-logo"],
.gradio-container .logo-img {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}

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

/* ============ Codex × 膳食编辑风 · 精致增强 ============ */
html, body {
    background: #f6f6f3 !important;
}
:root {
    --ink: #141414 !important;
    --accent: #1f9d61 !important;
    --accent-soft: #e9f6ef !important;
    --paper: #f6f6f3 !important;
    --card: #ffffff !important;
    --line: #e8e7e2 !important;
    --muted: #7a7a76 !important;
    --mono: "JetBrains Mono", "SF Mono", ui-monospace, Consolas, monospace !important;
}
.gradio-container {
    background:
        radial-gradient(1200px 600px at 12% -8%, rgba(31,157,97,0.055), transparent 60%),
        radial-gradient(900px 500px at 100% 0%, rgba(20,20,20,0.035), transparent 55%),
        #f6f6f3 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
        "Microsoft YaHei", "Helvetica Neue", sans-serif !important;
}

/* ---- 侧边栏：杂志感品牌区 ---- */
.sidebar-col {
    background: rgba(255,255,255,0.72) !important;
    backdrop-filter: saturate(1.1) blur(2px);
    border-right: 1px solid var(--line) !important;
    padding: 26px 22px !important;
}
.sidebar-col > div:first-child > .markdown > p,
.sidebar-col h1, .sidebar-col h2, .sidebar-col h3,
.sidebar-col .gr-markdown { color: var(--ink) !important; }
.sidebar-col .gr-markdown h3 {
    font-size: 22px !important;
    letter-spacing: -0.02em !important;
    font-weight: 700 !important;
}
/* 标签上加绿色小徽标 */
.sidebar-col .gr-markdown p { font-size: 13px !important; color: var(--muted) !important; line-height: 1.7 !important; }
.sidebar-col .gr-block, .sidebar-col .gr-box, .sidebar-col .input-container { border: none !important; background: transparent !important; }
/* 控件标题 */
.sidebar-col label .wrap label, .sidebar-col .gr-form label span,
.sidebar-col label span { font-size: 11px !important; letter-spacing: 0.06em !important;
    text-transform: uppercase !important; color: var(--muted) !important; font-weight: 600 !important; }
/* 单选卡片 */
.sidebar-col label {
    background: #fff !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
    padding: 9px 13px !important;
    color: var(--ink) !important;
    font-size: 13px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
    transition: all 0.16s ease !important;
}
.sidebar-col label:hover { border-color: rgba(31,157,97,0.6) !important; transform: translateY(-1px); }
.sidebar-col label.selected {
    background: var(--accent-soft) !important;
    border-color: var(--accent) !important;
    color: var(--ink) !important;
    box-shadow: 0 0 0 3px rgba(31,157,97,0.10) !important;
}
.sidebar-col select {
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
    color: var(--ink) !important;
    font-size: 13px !important;
    padding: 8px 12px !important;
}
/* 按钮：墨黑主操作 + 精描边次操作 */
.sidebar-btn {
    background: #fff !important; border: 1px solid var(--line) !important;
    color: var(--ink) !important; border-radius: 10px !important;
    font-size: 13px !important; font-weight: 600 !important;
    padding: 10px 16px !important; box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
    transition: all 0.16s ease !important;
}
.sidebar-btn:hover {
    border-color: var(--ink) !important; background: var(--ink) !important; color: #fff !important;
    transform: translateY(-1px);
}
.sidebar-btn:active { transform: translateY(0); }

/* ---- 主对话区 ---- */
.chat-col { background: transparent !important; }

/* 顶部柔和标题条 */
.chat-col > div:first-child { position: relative; }

/* 聊天气泡 */
.chatbot-panel .message-row { padding: 8px 0 !important;
    animation: chatIn 0.32s ease both !important; }
@keyframes chatIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

/* 用户气泡：墨黑圆润 */
.chatbot-panel .user .bubble {
    background: linear-gradient(135deg, #1b1b1b, #2a2a2a) !important;
    color: #f5f5f2 !important;
    border: none !important;
    border-radius: 16px 6px 16px 16px !important;
    padding: 11px 16px !important;
    font-size: 14px !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.12) !important;
}
/* 助手气泡：白卡片 · 精致阴影 */
.chatbot-panel .bot .bubble {
    background: var(--card) !important;
    color: var(--ink) !important;
    border: 1px solid var(--line) !important;
    border-radius: 6px 18px 18px 18px !important;
    padding: 16px 18px !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
    box-shadow: 0 3px 12px rgba(0,0,0,0.05) !important;
    max-width: 760px !important;
}

/* 推荐菜品 markdown 卡片化 */
.chatbot-panel .bot .bubble h3 {
    font-size: 15px !important;
    letter-spacing: -0.01em !important;
    font-weight: 700 !important;
    color: var(--ink) !important;
    margin: 4px 0 10px !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}
.chatbot-panel .bot .bubble h3::before {
    content: "🍽";
    font-size: 15px;
}
.chatbot-panel .bot .bubble ol {
    margin: 0 !important;
    padding-left: 0 !important;
    list-style: none !important;
    display: grid !important;
    gap: 8px !important;
}
.chatbot-panel .bot .bubble ol > li {
    background: #fbfbf9 !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    padding: 10px 14px !important;
    font-size: 14px !important;
    position: relative !important;
    padding-left: 40px !important;
    transition: all 0.15s ease !important;
}
.chatbot-panel .bot .bubble ol > li:hover {
    border-color: rgba(31,157,97,0.55) !important;
    background: #fff !important;
    transform: translateX(2px);
}
.chatbot-panel .bot .bubble ol > li::before {
    content: counter(list-item);
    counter-increment: list-item;
    position: absolute !important;
    left: 12px !important; top: 50% !important;
    transform: translateY(-50%) !important;
    width: 20px !important; height: 20px !important;
    border-radius: 7px !important;
    background: var(--accent-soft) !important;
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    font-size: 11px !important; font-weight: 700 !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
}
.chatbot-panel .bot .bubble ol > li:first-child { counter-reset: list-item; }
.chatbot-panel .bot .bubble strong { font-weight: 650 !important; color: var(--ink) !important; }
/* 标签 chip：反引号 code → 甘草绿小胶囊 */
.chatbot-panel .bot .bubble code {
    background: var(--accent-soft) !important;
    color: #137a4a !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    padding: 2px 8px !important;
    border-radius: 20px !important;
    border: 1px solid rgba(31,157,97,0.18) !important;
    font-weight: 600 !important;
}
/* 热量：墨黑粗 mono */
.chatbot-panel .bot .bubble ol li strong {
    font-family: var(--mono) !important;
    color: var(--ink) !important;
}
/* 分隔线 */
.chatbot-panel .bot .bubble hr {
    border: none !important;
    border-top: 1px dashed #dcdbd5 !important;
    margin: 12px 0 !important;
}
/* 回复正文引用/正文 */
.chatbot-panel .bot .bubble blockquote {
    border-left: 3px solid var(--accent) !important;
    background: var(--accent-soft) !important;
    color: #3c3c3a !important;
    border-radius: 0 10px 10px 0 !important;
    padding: 8px 14px !important;
    margin: 8px 0 0 !important;
    font-size: 13px !important;
}
.chatbot-panel .bot .bubble p { margin: 8px 0 0 !important; color: #34342f !important; }
/* ---- 营养概览小卡片（紧凑） ---- */
.chatbot-panel .bot .bubble .nutrition-card {
    display: inline-block !important;
    background: #fafaf8 !important;
    border: 1px solid var(--line) !important;
    border-left: 3px solid var(--accent) !important;
    border-radius: 8px !important;
    padding: 4px 10px !important;
    margin: 8px 0 2px !important;
    font-size: 11.5px !important;
    line-height: 1.5 !important;
    color: #4a4a45 !important;
    box-shadow: none !important;
}
.chatbot-panel .bot .bubble .nutrition-card .nut-title {
    font-weight: 700 !important;
    color: var(--ink) !important;
    font-size: 11.5px !important;
}

/* ---- 输入区 ---- */
.input-row {
    border-top: 1px solid var(--line) !important;
    background: rgba(246,246,243,0.9) !important;
    backdrop-filter: blur(4px);
    padding: 16px 24px !important;
}
.input-row textarea {
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    padding: 13px 18px !important;
    font-size: 14px !important;
    background: #fff !important;
    color: var(--ink) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03) !important;
}
.input-row textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 4px rgba(31,157,97,0.12) !important;
    background: #fff !important;
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
            # 避免标题重复（如 label="推荐" 时不再拼成 "推荐 推荐菜品"）
            title = f"{label} 推荐菜品" if label and label != "推荐" else "推荐菜品"
            lines.append(f"### {title}\n")
            for i, r in enumerate(recommendations, 1):
                name = r.get("name", "未知")
                tags = r.get("tags", [])
                cals = r.get("nutrition", {}).get("calories", "")
                # CSS 把 <ol><li> 渲染为菜品卡片：编号徽章 + 菜名 + 绿色标签 chip + mono 热量
                dish = f"{i}. **{name}**"
                if cals:
                    dish += f"  **{ChatSession._format_calorie(cals)} kcal**"
                if tags:
                    dish += "  " + "  ".join(f"`{t}`" for t in tags[:4])
                lines.append(dish)
                lines.append("")

        if response:
            if recommendations:
                lines.append("---\n")
            lines.append(f"> {response}")

        return "\n".join(lines) if lines else response

    @staticmethod
    def _render_dish_section(intent: str, recommendations: list) -> str:
        """将流式返回的推荐列表渲染为菜品卡片区（只含菜名与标签，不拼接回复）。"""
        if not recommendations:
            return ""
        intent_labels = {
            "recommend": "推荐", "add_constraint": "调整",
            "request_substitute": "替换", "reject_recommendation": "否定",
            "ask_clarification": "追问", "vague_query": "推荐",
        }
        label = intent_labels.get(intent, "")
        title = f"{label} 推荐菜品" if label and label != "推荐" else "推荐菜品"
        lines = [f"### {title}\n"]
        for i, r in enumerate(recommendations, 1):
            name = r.get("name", "未知")
            tags = r.get("tags", [])
            cals = r.get("nutrition", {}).get("calories", "")
            dish = f"{i}. **{name}**"
            if cals:
                dish += f"  **{cals} kcal**"
            if tags:
                dish += "  " + "  ".join(f"`{t}`" for t in tags[:4])
            lines.append(dish)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_calorie(cals) -> str:
        """把卡路里归一化为干净的整数，如 250.0/250 -> '250'；空值返回 ''。"""
        if cals in (None, "", 0):
            return ""
        try:
            return str(int(float(cals)))
        except (TypeError, ValueError):
            return str(cals)

    @staticmethod
    def _embed_nutrition_card(md: str) -> str:
        """把回复末尾的"营养概览"至少包成一张更紧凑的小卡片，避免整段大字撑得过大。"""
        if '营养概览' not in md:
            return md
        lines = md.split('\n')
        out = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if '营养概览' in line:
                body = []
                j = i + 1
                while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith('#'):
                    body.append(lines[j].strip())
                    j += 1
                if body:
                    content = '<br>'.join(body)
                    out.append(f'<div class="nutrition-card"><span class="nut-title">营养概览</span><br>'
                               f'<span class="nut-body">{content}</span></div>')
                    i = j
                    continue
            out.append(line)
            i += 1
        return '\n'.join(out)

    def send_stream(self, message: str):
        """流式获取回复：推荐列表先出，回复文本随后逐段刷新（SSE）。

        Yields:
            每段完整的 markdown（推荐区 + 当前已累积的回复文本）。
        """
        rec_section, response_text = "", ""
        intent = "recommend"
        try:
            with self.session.post(
                f"{API_BASE}/api/dialog/stream",
                json={"user_id": self.user_id, "message": message},
                stream=True, timeout=120,
            ) as resp:
                if resp.status_code != 200:
                    yield f"**[错误]** HTTP {resp.status_code}"
                    return
                # 强制 UTF-8 解码 SSE，避免后端 text/event-stream 被 requests 按 Latin-1 解析成乱码
                resp.encoding = 'utf-8'
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw or not raw.startswith("data:"):
                        continue
                    payload = raw[5:].strip()
                    if payload in ("[DONE]", ""):
                        continue
                    try:
                        evt = json.loads(payload)
                    except Exception:
                        continue
                    etype = evt.get("event")
                    edata = evt.get("data", {}) or {}
                    if etype == "recommendations":
                        intent = edata.get("intent", intent)
                        rec_section = self._render_dish_section(
                            intent, edata.get("recommendations", []))
                        if rec_section:
                            yield rec_section
                    elif etype == "text":
                        txt = edata.get("text", "")
                        if not txt:
                            continue
                        response_text += txt
                        md = rec_section
                        if response_text:
                            if md:
                                md += "\n\n---\n"
                            md += f"> {response_text}"
                        yield md
                    elif etype == "complete":
                        full_resp = edata.get("response") or response_text
                        rec_section = self._render_dish_section(
                            intent, edata.get("recommendations", []))
                        md = rec_section
                        if full_resp:
                            if md:
                                md += "\n\n"
                            md += f"> {full_resp}"
                        yield self._embed_nutrition_card(md)
        except requests.ConnectionError:
            yield "**[错误]** 无法连接后端服务，请先启动 `python app.py`"
        except Exception as e:
            yield f"**[错误]** {e}"

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
    """处理消息（流式）：先追加用户气泡，再逐段刷新助手回复，推荐先出、文本后现。"""
    if not message.strip():
        yield history, ""
        return

    history = list(history)
    if _USE_DICT:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})
    else:
        history.append((message, ""))
    yield history, ""

    try:
        session = get_or_create_session(user_id)
        for md in session.send_stream(message.strip()):
            if _USE_DICT:
                history[-1]["content"] = md
            else:
                history[-1] = (message, md)
            yield history, ""
    except Exception as e:
        err = f"**[错误]** {e}"
        if _USE_DICT:
            history[-1]["content"] = err
        else:
            history[-1] = (message, err)
        yield history, ""


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
        # 必须启用 queue，chat_fn 的生成器 yield 才会逐段流式上屏：
        # 用户问题先显示（空助手气泡占位），随后推荐列表与回复文本逐段刷新。
        demo.queue(default_concurrency_limit=8)
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
