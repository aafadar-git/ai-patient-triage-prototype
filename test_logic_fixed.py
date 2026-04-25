import json
import pytest
from unittest.mock import patch, MagicMock
import requests

from logic import (
    call_purdue_genai,
    call_purdue_genai_judge,
    local_fallback_judge,
    process_message_pipeline
)

# ==========================================
# TEST GROUP A: call_purdue_genai_judge
# ==========================================

@patch("logic.requests.post")
def test_a1_judge_pass(mock_post):
    """Assert a valid pass payload returns Success status."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps({
            "verdict": "pass",
            "corrected_urgency_label": "routine",
            "corrected_type_label": "admin",
            "corrected_route_label": "front desk",
            "corrected_confidence": 0.90,
            "corrected_draft_response": "Draft reply",
            "judge_rationale": "Looks good"
        })}}]
    }
    mock_post.return_value = mock_resp
    
    result = call_purdue_genai_judge("Test msg", {})
    assert result["status"] == "Success"
    assert result["data"]["verdict"] == "pass"

@patch("logic.requests.post")
def test_a2_judge_fail_missing_rationale(mock_post):
    """Assert a payload missing judge_rationale returns InvalidResponse."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps({
            "verdict": "pass",
            "corrected_urgency_label": "routine",
            "corrected_type_label": "admin",
            "corrected_route_label": "front desk",
            "corrected_confidence": 0.90,
            "corrected_draft_response": "Draft reply"
            # Missing judge_rationale
        })}}]
    }
    mock_post.return_value = mock_resp
    
    result = call_purdue_genai_judge("Test msg", {})
    assert result["status"] == "InvalidResponse"
    assert "judge_rationale" in result["error"]

@patch("logic.requests.post")
def test_a3_judge_fail_missing_confidence(mock_post):
    """Assert a payload missing corrected_confidence returns InvalidResponse."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps({
            "verdict": "pass",
            "corrected_urgency_label": "routine",
            "corrected_type_label": "admin",
            "corrected_route_label": "front desk",
            "corrected_draft_response": "Draft reply",
            "judge_rationale": "Looks good"
            # Missing corrected_confidence
        })}}]
    }
    mock_post.return_value = mock_resp
    
    result = call_purdue_genai_judge("Test msg", {})
    assert result["status"] == "InvalidResponse"
    assert "corrected_confidence" in result["error"]

@patch("logic.requests.post")
def test_a4_network_error_simulation(mock_post):
    """Assert a connection error is caught and returns an APIError status."""
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")
    
    result = call_purdue_genai_judge("Test msg", {}, max_retries=0)
    assert result["status"] == "APIError"
    assert "Connection failed" in result["error"]

# ==========================================
# TEST GROUP B: local_fallback_judge
# ==========================================

def test_b1_escalation():
    """Assert an acute message with routine label gets escalated to urgent."""
    candidate = {
        "urgency_label": "routine",
        "type_label": "symptom",
        "route_label": "nurse pool",
        "confidence": 0.85,
        "draft_response": "Have a nice day."
    }
    # "chest pain" is a known acute flag
    result = local_fallback_judge("I have severe chest pain right now.", candidate)
    
    assert result["corrected_urgency_label"] == "urgent"
    assert result["verdict"] == "fail"
    # Draft must be cleared for non-routine
    assert not result["corrected_draft_response"]

def test_b2_deescalation():
    """Assert a wellness message with urgent label gets de-escalated to routine."""
    candidate = {
        "urgency_label": "urgent",
        "type_label": "follow-up",
        "route_label": "physician",
        "confidence": 0.20,
        "draft_response": ""
    }
    # "diet" is a known wellness flag
    result = local_fallback_judge("I want to go on a diet and exercise.", candidate)
    
    assert result["corrected_urgency_label"] == "routine"
    assert result["verdict"] == "fail"

def test_b3_draft_removal_on_non_routine():
    """Assert that a non-empty draft is cleared when urgency is urgent."""
    candidate = {
        "urgency_label": "urgent",
        "type_label": "symptom",
        "route_label": "physician",
        "confidence": 0.90,
        "draft_response": "We will see you soon."
    }
    result = local_fallback_judge("Normal message", candidate)
    
    assert result["corrected_urgency_label"] == "urgent"
    assert not result["corrected_draft_response"]
    assert result["verdict"] == "fail"

def test_b4_clean_pass_through():
    """Assert that a valid routine payload passes without modification."""
    candidate = {
        "urgency_label": "routine",
        "type_label": "admin",
        "route_label": "front desk",
        "confidence": 0.50,
        "draft_response": "We will schedule you."
    }
    result = local_fallback_judge("I need an appointment.", candidate)
    
    assert result["corrected_urgency_label"] == "routine"
    assert result["corrected_confidence"] == 0.50
    assert result["corrected_draft_response"] == "We will schedule you."
    assert result["verdict"] == "pass"

# ==========================================
# TEST GROUP C: process_message_pipeline
# ==========================================

def test_c1_red_flag_pre_emption():
    """Assert that 'chest pain' immediately escalates and suppresses draft."""
    result = process_message_pipeline(
        "I have chest pain and shortness of breath.",
        inference_mode="Rules Only"
    )
    
    assert result["urgency_label"] == "urgent"
    assert result["escalation_reason"] is not None
    assert not result["draft_response"]

def test_c2_low_confidence_gate():
    """Assert that low confidence correctly suppresses the draft response."""
    # A very short message typically yields low confidence in stub_classifier
    result = process_message_pipeline(
        "Hi.",
        inference_mode="Rules Only"
    )
    
    # Confidence threshold in logic.py is 0.80.
    assert result["confidence"] < 0.80
    assert result["escalation_reason"] is not None
    assert not result["draft_response"]

def test_c3_routine_draft_generation():
    """Assert that a valid routine message produces a draft in Rules mode."""
    # stub_classifier gives admin/medication high confidence if long enough
    result = process_message_pipeline(
        "Can I get a refill on my blood pressure medication? It is lisinopril.",
        inference_mode="Rules Only"
    )
    
    assert result["urgency_label"] == "routine"
    assert result["escalation_reason"] is None
    # Wait, stub_classifier does not produce a draft! process_message_pipeline only produces
    # a draft if GenAI is Success, OR if dataset_row has one. We need to pass a mock dataset_row
    # to actually see a draft response in Rules Only mode, OR we can test the behavior.
    # We will pass a dataset_row to simulate the draft.
    result_with_draft = process_message_pipeline(
        "Can I get a refill on my blood pressure medication? It is lisinopril.",
        dataset_row={"draft_response": "Sure thing!"},
        inference_mode="Rules Only"
    )
    assert result_with_draft["urgency_label"] == "routine"
    assert result_with_draft["draft_response"] == "Sure thing!"

@patch("logic.call_purdue_genai")
def test_c4_genai_call_blocked_by_red_flag(mock_call_genai):
    """Assert that an acute red flag prevents the GenAI endpoint from being called."""
    process_message_pipeline(
        "I have severe shortness of breath.",
        inference_mode="Purdue GenAI Assisted"
    )
    
    assert mock_call_genai.call_count == 0

# ==========================================
# TEST GROUP D: call_purdue_genai edge cases
# ==========================================

@patch("logic.requests.post")
def test_d1_markdown_fence_stripping(mock_post):
    """Assert that markdown JSON fences are correctly stripped and parsed."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    
    raw_json_str = json.dumps({
        "urgency_label": "routine",
        "type_label": "admin",
        "route_label": "front desk",
        "confidence": 0.90,
        "draft_response": "Draft",
        "rationale": "OK"
    })
    
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": f"```json\n{raw_json_str}\n```"}}]
    }
    mock_post.return_value = mock_resp
    
    result = call_purdue_genai("Test")
    assert result["status"] == "Success"
    assert result["data"]["urgency_label"] == "routine"

@patch("logic.requests.post")
def test_d2_missing_urgency_label(mock_post):
    """Assert that missing urgency_label results in InvalidResponse."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps({
            "type_label": "admin",
            "route_label": "front desk",
            "confidence": 0.90,
            "draft_response": "Draft",
            "rationale": "OK"
            # Missing urgency_label
        })}}]
    }
    mock_post.return_value = mock_resp
    
    result = call_purdue_genai("Test")
    assert result["status"] == "InvalidResponse"
    assert "urgency_label" in result["error"]

@patch("logic.requests.post")
def test_d3_invalid_urgency_label_value(mock_post):
    """Assert that an unrecognized urgency_label fails validation."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps({
            "urgency_label": "Maybe",
            "type_label": "admin",
            "route_label": "front desk",
            "confidence": 0.90,
            "draft_response": "Draft",
            "rationale": "OK"
        })}}]
    }
    mock_post.return_value = mock_resp
    
    result = call_purdue_genai("Test")
    assert result["status"] == "InvalidResponse"
    assert "Invalid urgency label" in result["error"]
