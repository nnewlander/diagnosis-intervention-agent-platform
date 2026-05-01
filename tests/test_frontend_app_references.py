from pathlib import Path


def test_frontend_app_not_reference_old_formatter_names():
    app_path = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    content = app_path.read_text(encoding="utf-8")
    assert "_build_demo_final_response" not in content
    assert "_build_kg_conclusion" not in content
    assert "_select_kg_reference" not in content

