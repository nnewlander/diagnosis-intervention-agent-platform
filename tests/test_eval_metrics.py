from scripts.eval_offline import evaluate_task_aware_case


def test_technical_qa_empty_diagnosis_still_complete():
    state = {
        "final_response": "这是 NameError，通常变量未定义。",
        "need_clarify": False,
        "routing_mode": "technical_qa_short_path",
        "rag_evidence": [{"source_id": "x"}],
        "diagnosis": {},
        "intervention_plan": {},
        "recommended_packages": [],
    }
    result = evaluate_task_aware_case(state, expected_task="technical_qa", actual_primary_task="technical_qa")
    assert result["task_aware_structure_ok"] is True


def test_diagnosis_empty_should_fail():
    state = {"diagnosis": {}, "final_response": "ok"}
    result = evaluate_task_aware_case(state, expected_task="diagnosis", actual_primary_task="diagnosis")
    assert result["task_aware_structure_ok"] is False
    assert "diagnosis_empty" in result["failed_reasons"]


def test_technical_qa_need_clarify_true_should_fail():
    state = {
        "final_response": "解释内容",
        "need_clarify": True,
        "routing_mode": "technical_qa_short_path",
        "rag_evidence": [{"source_id": "x"}],
    }
    result = evaluate_task_aware_case(state, expected_task="technical_qa", actual_primary_task="technical_qa")
    assert result["task_aware_structure_ok"] is False
    assert "technical_qa_need_clarify_true" in result["failed_reasons"]


def test_technical_qa_short_path_true_pass():
    state = {
        "final_response": "解释内容",
        "need_clarify": False,
        "routing_mode": "technical_qa_short_path",
        "rag_evidence": [{"source_id": "x"}],
    }
    result = evaluate_task_aware_case(state, expected_task="technical_qa", actual_primary_task="technical_qa")
    assert result["task_aware_structure_ok"] is True
