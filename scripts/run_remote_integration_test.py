from pathlib import Path
import os
import socket
import subprocess
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _find_free_port(start_port: int) -> int:
    for port in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"未找到可用端口，起始端口={start_port}")


def _wait_for_health(url: str, timeout_seconds: int = 20) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _terminate_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def main() -> int:
    rag_port = _find_free_port(9002)
    kg_port = _find_free_port(9003 if rag_port != 9003 else 9004)
    if kg_port == rag_port:
        kg_port = _find_free_port(rag_port + 1)
    rag_base = f"http://127.0.0.1:{rag_port}"
    kg_base = f"http://127.0.0.1:{kg_port}"

    child_env = os.environ.copy()
    child_env["RAG_MOCK_PORT"] = str(rag_port)
    child_env["KG_MOCK_PORT"] = str(kg_port)

    smoke_env = os.environ.copy()
    smoke_env.update(
        {
            "RAG_PROVIDER": "remote",
            "KG_PROVIDER": "remote",
            "RAG_API_BASE": rag_base,
            "KG_API_BASE": kg_base,
        }
    )

    rag_cmd = [sys.executable, str(ROOT / "scripts" / "run_mock_rag.py")]
    kg_cmd = [sys.executable, str(ROOT / "scripts" / "run_mock_kg.py")]
    smoke_cmd = [sys.executable, str(ROOT / "scripts" / "smoke_test_remote.py")]

    rag_proc: subprocess.Popen | None = None
    kg_proc: subprocess.Popen | None = None

    try:
        print(f"[integration] 启动 mock RAG 服务: {rag_base}")
        rag_proc = subprocess.Popen(rag_cmd, cwd=str(ROOT), env=child_env)
        print(f"[integration] 启动 mock KG 服务: {kg_base}")
        kg_proc = subprocess.Popen(kg_cmd, cwd=str(ROOT), env=child_env)

        print("[integration] 等待服务健康检查...")
        rag_ready = _wait_for_health(f"{rag_base}/health")
        kg_ready = _wait_for_health(f"{kg_base}/health")
        if not rag_ready or not kg_ready:
            print("[integration] 服务健康检查失败。")
            print(f"[integration] rag_ready={rag_ready}, kg_ready={kg_ready}")
            return 1

        print("[integration] 健康检查通过，执行 remote smoke test...")
        result = subprocess.run(smoke_cmd, cwd=str(ROOT), env=smoke_env, check=False)
        if result.returncode == 0:
            print("[integration] 联调成功：remote smoke test 通过。")
        else:
            print(f"[integration] 联调失败：smoke test exit_code={result.returncode}")
        return result.returncode
    finally:
        print("[integration] 清理 mock 服务进程...")
        _terminate_process(rag_proc)
        _terminate_process(kg_proc)
        print("[integration] 清理完成。")


if __name__ == "__main__":
    raise SystemExit(main())
