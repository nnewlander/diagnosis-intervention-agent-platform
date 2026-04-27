import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests

from app.core.config import settings


def load_sample_requests() -> list[str]:
    sample_file = settings.PROJECT_ROOT / "data" / "sample_requests.json"
    if sample_file.exists():
        payload = json.loads(sample_file.read_text(encoding="utf-8"))
        return [item.get("request_text", "") for item in payload if item.get("request_text")]
    return [
        "请帮我定位 student_id:1001 在循环知识点的常见错误，并给出干预建议。",
        "我的学生总是编译报错，怎么快速讲清楚语法错误？",
        "请推荐适合函数知识点的补练题包。",
    ]


def validate_response(data: dict) -> tuple[bool, str]:
    required_keys = [
        "task_type",
        "evidence_summary",
        "diagnosis",
        "intervention_plan",
        "recommended_packages",
        "final_response",
        "debug_trace",
    ]
    missing = [k for k in required_keys if k not in data]
    if missing:
        return False, f"缺少字段: {missing}"
    return True, "OK"


def main() -> None:
    url = f"http://127.0.0.1:{settings.API_PORT}/agent/run"
    requests_list = load_sample_requests()[:5]
    print(f"[smoke] target={url}, cases={len(requests_list)}")

    for idx, req in enumerate(requests_list, start=1):
        try:
            response = requests.post(url, json={"request_text": req}, timeout=30)
            response.raise_for_status()
            data = response.json()
            valid, message = validate_response(data)
            print(f"\nCase#{idx}: {req}")
            print(f"- task_type: {data.get('task_type')}")
            print(f"- diagnosis: {data.get('diagnosis')}")
            print(f"- final_response: {data.get('final_response')[:120]}...")
            print(f"- structure_valid: {valid} ({message})")
        except Exception as exc:
            print(f"\nCase#{idx} FAILED: {exc}")


if __name__ == "__main__":
    main()
