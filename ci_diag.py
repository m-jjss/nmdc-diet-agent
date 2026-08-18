"""CI 诊断脚本：在容器内验证 dialog 接口（由 docker exec 调用）

用途：当 /api/dialog 在容器环境返回非 200 时，用 Flask test_client 直接调用
并打印异常 traceback，帮助定位容器环境特有的 500 根因。
"""
import traceback

import app as app_module

client = app_module.app.test_client()
try:
    resp = client.post('/api/dialog', json={
        'user_id': 'ci_test',
        'message': '推荐几道菜',
        'reset': True,
    })
    print('STATUS:', resp.status_code)
    print(resp.get_data(as_text=True)[:600])
except Exception:
    traceback.print_exc()
