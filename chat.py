"""
方太个性化膳食规划 Agent - 终端对话界面

用法：
    python chat.py          # 新用户模式（会触发3步引导）
    python chat.py 1        # 预置用户1（海鲜过敏）
    python chat.py 3        # 预置用户3（高血压+高血糖）
    python chat.py 6        # 预置用户6（哺乳期+海鲜鸡蛋过敏）
"""

import sys
import json
import uuid
import requests

BASE_URL = "http://127.0.0.1:5000"


def main():
    # 确定用户ID
    if len(sys.argv) > 1:
        user_id = sys.argv[1]
        label = f"预置用户{user_id}"
    else:
        user_id = "new_" + uuid.uuid4().hex[:8]
        label = "新用户"

    session = requests.Session()
    print("=" * 50)
    print(f"  方太个性化膳食规划 Agent")
    print(f"  {label}")
    print(f"  输入 quit 退出 | reset 重置")
    print("=" * 50)
    print()

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见~")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("再见~")
            break

        if user_input.lower() == "reset":
            try:
                session.post(f"{BASE_URL}/api/dialog", json={
                    "user_id": user_id,
                    "message": "reset",
                    "reset": True
                }, timeout=10)
            except Exception:
                pass
            print("[对话已重置]")
            print()
            continue

        try:
            resp = session.post(f"{BASE_URL}/api/dialog", json={
                "user_id": user_id,
                "message": user_input
            }, timeout=60)

            if resp.status_code != 200:
                print(f"[错误] HTTP {resp.status_code}")
                continue

            data = resp.json()
            if not data.get("success"):
                print(f"[错误] {data.get('error', '未知错误')}")
                continue

            info = data.get("data", {})
            intent = info.get("intent", "unknown")
            response = info.get("response", "")
            recommendations = info.get("recommendations", [])
            is_onboarding = info.get("is_onboarding", False)

            # 显示引导状态
            if intent == "onboarding" or is_onboarding:
                step = info.get("onboarding_step", 0)
                step_labels = {1: "过敏", 2: "口味", 3: "健康"}
                step_label = step_labels.get(step, "")
                if step_label:
                    print(f"\n  [{step_label}] {response}\n")
                else:
                    print(f"\n  {response}\n")
                continue

            # 显示意图标签
            intent_labels = {
                "recommend": "推荐",
                "add_constraint": "追加约束",
                "request_substitute": "替换菜品",
                "reject_recommendation": "否定方案",
                "vague_query": "模糊查询",
                "ask_clarification": "追问",
            }
            tag = intent_labels.get(intent, intent)

            # 显示推荐菜品
            if recommendations:
                print(f"\n  [{tag}]")
                for i, r in enumerate(recommendations, 1):
                    name = r.get("name", "未知")
                    tags = r.get("tags", [])
                    tag_str = ", ".join(tags) if tags else ""
                    cals = r.get("nutrition", {}).get("calories", "")
                    extra = f" | {cals}kcal" if cals else ""
                    if tag_str:
                        extra += f" | {tag_str}"
                    print(f"    {i}. {name}{extra}")

            # 显示回复（去重：如果回复已经被推荐列表覆盖就不重复显示）
            if response:
                timing_info = info.get("timing", {})
                t_total = timing_info.get("t_total_ms", 0)
                t_llm = timing_info.get("t_llm_ms", 0)
                t_search = timing_info.get("t_search_ms", 0)
                timing_str = f" [{t_total:.0f}ms" if t_total else ""
                if t_search:
                    timing_str += f", 检索{t_search:.0f}ms"
                if t_llm:
                    timing_str += f", LLM{t_llm:.0f}ms"
                timing_str += "]" if timing_str else ""
                print(f"\n  {response}{timing_str}\n")
            else:
                print()

        except requests.ConnectionError:
            print("[错误] 无法连接服务，请先启动 python app.py")
        except Exception as e:
            print(f"[错误] {e}")


if __name__ == "__main__":
    main()
