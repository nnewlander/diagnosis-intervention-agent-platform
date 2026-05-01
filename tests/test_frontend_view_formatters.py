from frontend.view_formatters import build_demo_final_response, build_kg_conclusion


def test_demo_final_response_nameerror_contains_classroom_phrase():
    text = build_demo_final_response(
        raw_final_response="",
        task_type="technical_qa",
        primary_task_type="technical_qa",
        parsed_error_type="NameError",
        rag_items=[{"source_id": "seed-faq-nameerror"}],
        kg_items=[{"entity": "NameError", "relation": "HAS_SOLUTION", "target": "检查变量或函数是否先定义后使用"}],
    )
    assert "程序找不到这个名字" in text


def test_kg_conclusion_has_space_after_nameerror():
    text = build_kg_conclusion(
        [
            {"entity": "NameError", "relation": "COMMON_MISUSE", "target": "变量名错误"},
            {"entity": "NameError", "relation": "RELATED_ERROR", "target": "未定义变量"},
            {"entity": "NameError", "relation": "HAS_SOLUTION", "target": "检查变量或函数是否先定义后使用"},
        ]
    )
    assert "NameError 常见问题" in text
    assert "NameError常见问题" not in text

