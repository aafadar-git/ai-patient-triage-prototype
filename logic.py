import pandas as pd
import random
import re
import os
import json
import requests
import time

def call_purdue_genai(
    message: str,
    temperature: float = 0.0,
    api_key: str | None = None,
    model_name: str | None = None,
    request_timeout_seconds: int = 60,
    max_retries: int = 3
):
    if not api_key:
        api_key = os.environ.get("PURDUE_GENAI_API_KEY")
    if not api_key:
        return {"status": "MissingKey", "error": "PURDUE_GENAI_API_KEY environment variable is not set."}

    url = "https://genai.rcac.purdue.edu/api/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You are a patient portal triage assistant. Classify ONE incoming patient message conservatively and consistently.

Core triage policy:
1) emergency = immediate danger / severe acute symptoms needing immediate clinician escalation.
2) urgent = time-sensitive clinical concern, but not clearly life-threatening.
3) routine = non-acute questions, chronic care management, prevention/wellness, scheduling/admin, general guidance.

Important anti-over-escalation rule:
- Do NOT label lifestyle, wellness, diet, exercise, prevention, or general education questions as urgent/emergency unless there are clear acute red-flag symptoms.
- Example: "How can I have a healthier diet?" should be routine.

Invalid/non-clinical/figurative handling:
If the message is nonsensical, joking, non-medical, clearly invalid, romantic, or uses figurative slang (e.g., "refill my diet coke", "I'm dying 💀", "I can't breathe without you", "dying from laughter", "this is killing me lol"):
- Keep classification conservative (routine).
- Lower confidence (<0.5).
- Return empty draft_response.
- Explain why in rationale.

Output requirements:
Return ONLY valid JSON with exactly these keys (all mandatory), and no prose before/after:
"urgency_label": one of ["emergency", "urgent", "routine"]
"type_label": one of ["symptom", "medication", "admin", "follow-up"]
"route_label": one of ["nurse pool", "physician", "front desk"]
"confidence": float in [0.0, 1.0]
"draft_response": patient-facing draft only for valid routine clinical requests, else empty string
"rationale": concise explanation of why labels were chosen

Draft response rules (only when routine and valid):
- 2-4 sentences, empathetic and professional.
- No diagnosis, no unsafe advice, no mention of AI.
- If ambiguous, use conservative language and advise clinic follow-up.

Few-shot examples (guidance only; do not copy text literally):

Example A (routine wellness edge case):
Input: "How can I have a healthier diet? I want to improve my eating habits."
Output:
{{
  "urgency_label": "routine",
  "type_label": "follow-up",
  "route_label": "nurse pool",
  "confidence": 0.90,
  "draft_response": "Thanks for reaching out about improving your diet. We can share general nutrition guidance and help you set practical goals based on your health history. If you would like, we can also schedule a routine follow-up to discuss a personalized plan.",
  "rationale": "General wellness/preventive question without acute symptoms, so routine triage is appropriate."
}}

Example B (urgent symptom):
Input: "I have a high fever and rash that is getting worse since yesterday."
Output:
{{
  "urgency_label": "urgent",
  "type_label": "symptom",
  "route_label": "physician",
  "confidence": 0.88,
  "draft_response": "",
  "rationale": "Worsening acute symptoms are time-sensitive and should be escalated for clinician review."
}}

Example C (emergency symptom):
Input: "I have chest pain and shortness of breath right now."
Output:
{{
  "urgency_label": "emergency",
  "type_label": "symptom",
  "route_label": "physician",
  "confidence": 0.97,
  "draft_response": "",
  "rationale": "Potential life-threatening red-flag symptoms require immediate escalation."
}}

Example D (routine medication):
Input: "Can I get a refill on my lisinopril? I have 3 pills left."
Output:
{{
  "urgency_label": "routine",
  "type_label": "medication",
  "route_label": "nurse pool",
  "confidence": 0.93,
  "draft_response": "Thanks for your refill request. We have sent this to the care team for review and processing. Please continue taking your medication as previously directed, and let us know if you have any new symptoms.",
  "rationale": "Valid medication refill request without urgent symptoms; routine medication workflow."
}}

Example E (figurative/romantic language):
Input: "I can't breathe without you near me lol"
Output:
{{
  "urgency_label": "routine",
  "type_label": "admin",
  "route_label": "front desk",
  "confidence": 0.35,
  "draft_response": "",
  "rationale": "Uses figurative/romantic language ('can't breathe without you') rather than describing a genuine medical respiratory emergency. Routine/low confidence appropriate."
}}

Example F (figurative slang):
Input: "this workout is killing me I'm dying 💀"
Output:
{{
  "urgency_label": "routine",
  "type_label": "admin",
  "route_label": "front desk",
  "confidence": 0.40,
  "draft_response": "",
  "rationale": "Slang and exaggeration ('killing me', 'dying') with no true clinical symptoms. Routine/low confidence appropriate."
}}

Example G (invalid/non-clinical edge case):
Input: "Please refill my diet coke prescription lol."
Output:
{{
  "urgency_label": "routine",
  "type_label": "admin",
  "route_label": "front desk",
  "confidence": 0.25,
  "draft_response": "",
  "rationale": "Message appears non-clinical/joking and is not a valid healthcare request."
}}

Now classify this patient message:
{message}
"""
    try:
        if not model_name:
            model_name = os.environ.get("PURDUE_GENAI_MODEL", "llama3.1:latest")
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature)
        }
    except Exception as e:
        return {"status": "APIError", "error": f"Failed building API configuration: {str(e)}"}
    
    response = None
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=request_timeout_seconds)
            response.raise_for_status()
            last_error = None
            break
        except Exception as e:
            last_error = e
            # brief exponential backoff between retries
            if attempt < max_retries:
                time.sleep(1.5 * (2 ** attempt))

    if last_error is not None:
        return {
            "status": "APIError",
            "error": (
                f"Network or HTTP error after {max_retries + 1} attempt(s): {str(last_error)} "
                f"(timeout={request_timeout_seconds}s)"
            )
        }
        
    try:
        raw_json = response.json()
        print("RAW GENAI RESPONSE:", json.dumps(raw_json, indent=2))
        if not isinstance(raw_json, dict):
            status_code = getattr(response, "status_code", "Unknown")
            content_type = response.headers.get("Content-Type", "Unknown") if hasattr(response, "headers") else "Unknown"
            type_name = type(raw_json).__name__
            snippet = str(raw_json)[:500]
            return {
                "status": "InvalidAPIResponse", 
                "error": f"API returned non-dictionary JSON array or primitive.\nType: {type_name}\nStatus: {status_code}\nContent-Type: {content_type}\nPreview: {snippet}"
            }
    except Exception as e:
        status_code = getattr(response, "status_code", "Unknown") if 'response' in locals() else "Unknown"
        raw_text = response.text[:200] if 'response' in locals() and hasattr(response, "text") else "Unknown"
        return {"status": "InvalidAPIResponse", "error": f"API returned non-JSON body: {str(e)}\nStatus: {status_code}\nRaw Response: {raw_text}"}

    try:
        choices = raw_json.get("choices", [])
        if not choices:
            return {"status": "EmptyResponse", "error": f"Model returned no choices. Raw response: {raw_json}"}

        message_obj = choices[0].get("message", {})
        if not isinstance(message_obj, dict):
            return {"status": "InvalidAPIResponse", "error": "Message object is not a dictionary."}

        out_text = message_obj.get("content", "")
    except (AttributeError, KeyError, IndexError, TypeError) as e:
        return {"status": "InvalidAPIResponse", "error": f"Model JSON payload shape mismatch. Error: {str(e)}"}

    if not out_text or str(out_text).strip() == "":
         return {"status": "EmptyResponse", "error": f"Model returned empty message content. Raw response: {raw_json}"}

    try:
        if "```json" in out_text:
            out_text = out_text.split("```json")[1].split("```")[0]
        elif "```" in out_text:
            out_text = out_text.split("```")[1].split("```")[0]

        out_text = out_text.strip()
        if not out_text:
             return {"status": "EmptyResponse", "error": "Model content became empty after removing markdown fences."}

        parsed = json.loads(out_text)
    except Exception as e:
        return {"status": "ParseError", "error": f"Model returned invalid JSON format: {str(e)}\n\nExtracted Text: {out_text}"}

    if not isinstance(parsed, dict):
        return {"status": "ParseError", "error": "Model returned valid JSON, but it is not a dictionary object."}

    required_keys = ["urgency_label", "type_label", "route_label", "confidence", "draft_response", "rationale"]

    if "rationale" not in parsed or not str(parsed.get("rationale", "")).strip():
        parsed["rationale"] = "No rationale was returned by the model."
    if "draft_response" not in parsed or parsed["draft_response"] is None:
        parsed["draft_response"] = ""
    if "confidence" not in parsed or parsed["confidence"] is None:
        parsed["confidence"] = 0.5

    # urgency_label, type_label, and route_label intentionally have NO fallback.
    # If the model omits them, the pipeline hard-fails with InvalidResponse.
    # Only non-routing cosmetic fields (rationale, draft_response, confidence)
    # receive silent normalization because they do not affect clinical routing safety.
    missing = [k for k in required_keys if k not in parsed]
    if missing:
        return {"status": "InvalidResponse", "error": f"Missing required key(s): {', '.join(missing)}\n\nParsed JSON: {json.dumps(parsed)}"}

    valid_urgencies = ["emergency", "urgent", "routine"]
    valid_types = ["symptom", "medication", "admin", "follow-up"]
    valid_routes = ["nurse pool", "physician", "front desk"]

    if parsed.get("urgency_label") not in valid_urgencies: 
        return {"status": "InvalidResponse", "error": f"Invalid urgency label returned: {parsed.get('urgency_label')}"}
    if parsed.get("type_label") not in valid_types: 
        return {"status": "InvalidResponse", "error": f"Invalid type label returned: {parsed.get('type_label')}"}
    if parsed.get("route_label") not in valid_routes: 
        return {"status": "InvalidResponse", "error": f"Invalid route label returned: {parsed.get('route_label')}"}

    return {
        "status": "Success",
        "data": {
            "urgency_label": parsed["urgency_label"],
            "type_label": parsed["type_label"],
            "route_label": parsed["route_label"],
            "confidence": float(parsed["confidence"]),
            "draft_response": parsed["draft_response"],
            "rationale": parsed["rationale"],
            "prompt": prompt.strip(),
            "temperature": float(temperature)
        }
    }

def call_purdue_genai_judge(
    message: str,
    candidate_output: dict,
    api_key: str | None = None,
    model_name: str | None = None,
    request_timeout_seconds: int = 20,
    max_retries: int = 0
):
    """
    Secondary LLM judge pass to validate (and optionally correct) candidate triage output.
    """
    if not api_key:
        api_key = os.environ.get("PURDUE_GENAI_API_KEY")
    if not api_key:
        return {"status": "MissingKey", "error": "PURDUE_GENAI_API_KEY environment variable is not set."}

    if not model_name:
        model_name = os.environ.get("PURDUE_GENAI_MODEL", "llama3.1:latest")

    url = "https://genai.rcac.purdue.edu/api/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    judge_prompt = f"""
You are an independent clinical triage QA judge.
Review the candidate output for consistency, safety, confidence calibration, and response quality.

Patient message:
{message}

Candidate output JSON:
{json.dumps(candidate_output)}

Return ONLY one valid JSON object (no prose) with these keys:
- verdict: "pass" or "fail"
- corrected_urgency_label: one of ["emergency", "urgent", "routine"]
- corrected_type_label: one of ["symptom", "medication", "admin", "follow-up"]
- corrected_route_label: one of ["nurse pool", "physician", "front desk"]
- corrected_confidence: float in [0.0, 1.0]
- corrected_draft_response: string (empty if non-routine/high-risk)
- judge_rationale: concise explanation of validation/corrections

Rules:
- Do not over-escalate lifestyle/wellness/diet/prevention-only questions.
- Red-flag symptoms should not be routine.
- confidence should reflect certainty; lower it when ambiguous.
- Draft responses should exist only for routine, non-escalated situations.
"""

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": judge_prompt}],
        "temperature": 0.0
    }

    response = None
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=request_timeout_seconds)
            response.raise_for_status()
            last_error = None
            break
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1.0 * (2 ** attempt))

    if last_error is not None:
        return {
            "status": "APIError",
            "error": f"Judge network/HTTP error after {max_retries + 1} attempt(s): {str(last_error)}"
        }

    try:
        raw_json = response.json()
        choices = raw_json.get("choices", [])
        if not choices:
            return {"status": "EmptyResponse", "error": "Judge returned no choices."}
        out_text = choices[0].get("message", {}).get("content", "").strip()
        if not out_text:
            return {"status": "EmptyResponse", "error": "Judge returned empty content."}
        if "```json" in out_text:
            out_text = out_text.split("```json")[1].split("```")[0]
        elif "```" in out_text:
            out_text = out_text.split("```")[1].split("```")[0]

        parsed = json.loads(out_text.strip())
    except Exception as e:
        return {"status": "ParseError", "error": f"Judge parse error: {str(e)}"}

    required = [
        "verdict",
        "corrected_urgency_label",
        "corrected_type_label",
        "corrected_route_label",
        "corrected_confidence",
        "corrected_draft_response",
        "judge_rationale"
    ]
    missing = [k for k in required if k not in parsed]
    if missing:
        return {"status": "InvalidResponse", "error": f"Judge missing keys: {', '.join(missing)}"}

    return {"status": "Success", "data": parsed}

def local_fallback_judge(message: str, candidate_output: dict) -> dict:
    """
    Deterministic judge fallback when remote judge API times out/unavailable.
    Applies a few conservative consistency checks and returns judge-format output.
    """
    msg_l = (message or "").lower()
    corrected = dict(candidate_output or {})
    reasons = []
    changed = False

    acute_flags = [
        "chest pain", "shortness of breath", "severe bleeding", "faint", "fainted",
        "suicidal", "one-sided weakness", "allergic reaction", "high fever"
    ]
    wellness_flags = ["diet", "healthier", "nutrition", "exercise", "wellness", "prevention"]

    has_acute = any(flag in msg_l for flag in acute_flags)
    has_wellness = any(flag in msg_l for flag in wellness_flags)

    urgency = str(corrected.get("urgency_label", "routine"))
    conf = float(corrected.get("confidence", 0.5) or 0.5)

    # Avoid over-escalation for lifestyle/wellness-only prompts.
    if has_wellness and not has_acute and urgency in ["urgent", "emergency"]:
        corrected["urgency_label"] = "routine"
        corrected["type_label"] = corrected.get("type_label") if corrected.get("type_label") in ["admin", "follow-up"] else "follow-up"
        corrected["route_label"] = "nurse pool"
        corrected["confidence"] = min(conf, 0.75)
        reasons.append("Downgraded wellness/lifestyle-only message from urgent/emergency to routine.")
        changed = True

    # Ensure emergency-leaning acute flags are not routine.
    if has_acute and corrected.get("urgency_label") == "routine":
        corrected["urgency_label"] = "urgent"
        corrected["type_label"] = "symptom"
        corrected["route_label"] = "physician"
        corrected["confidence"] = max(float(corrected.get("confidence", 0.5) or 0.5), 0.8)
        reasons.append("Escalated routine label due to acute symptom red flags.")
        changed = True

    # No drafts for non-routine labels.
    if corrected.get("urgency_label") in ["urgent", "emergency"] and str(corrected.get("draft_response", "")).strip():
        corrected["draft_response"] = ""
        reasons.append("Removed draft response for non-routine triage.")
        changed = True

    return {
        "verdict": "fail" if changed else "pass",
        "corrected_urgency_label": corrected.get("urgency_label", "routine"),
        "corrected_type_label": corrected.get("type_label", "follow-up"),
        "corrected_route_label": corrected.get("route_label", "nurse pool"),
        "corrected_confidence": float(corrected.get("confidence", 0.5) or 0.5),
        "corrected_draft_response": str(corrected.get("draft_response", "") or ""),
        "judge_rationale": " | ".join(reasons) if reasons else "Local fallback judge found no consistency issues."
    }

RED_FLAGS = [
    "chest pain",
    "shortness of breath",
    "severe bleeding",
    "fainting",
    "fainted",
    "one-sided weakness",
    "allergic reaction",
    "high fever in infant",
    # Soft cardiopulmonary
    "chest tightness",
    "chest pressure",
    "elephant on chest",
    "hard to breathe",
    "can't catch my breath",
    "can't breathe",
    # Suicidality bucket
    "suicidal",
    "suicidal thoughts",
    "hurt myself",
    "want to die",
    "wish i wouldn't wake up",
    "end it all"
]

def is_obvious_nonclinical_figurative(message: str) -> bool:
    """
    Helper to detect clearly figurative/romantic/jokey language that might 
    cause false positive escalations in rule-based checks.
    """
    msg_l = message.lower().strip()
    
    strong_figurative = [
        "take my breath away", "dying laughing", "killing me", 
        "dying 💀", "dead 💀", "diet coke", "can't breathe laughing", 
        "lol", "lmao", "joke", "laughing", "dying from laughter"
    ]
    if any(phrase in msg_l for phrase in strong_figurative):
        return True
        
    if "without you" in msg_l and ("breathe" in msg_l or "breath" in msg_l):
        return True
        
    return False

def safety_check(message: str) -> str | None:
    """
    Checks a message for red-flag phrases. 
    Returns the reason for escalation if found, otherwise None.
    """
    message_lower = message.lower()
    flags_found = []
    
    figurative = is_obvious_nonclinical_figurative(message)
    respiratory_flags = ["shortness of breath", "hard to breathe", "can't catch my breath", "can't breathe", "trouble breathing"]
    
    for flag in RED_FLAGS:
        if re.search(r'\b' + re.escape(flag) + r'\b', message_lower):
            if figurative and flag in respiratory_flags:
                continue
            flags_found.append(flag)

    if flags_found:
        return f"Red flag detected: {', '.join(flags_found)}"
    return None

def stub_classifier(message: str):
    """
    Simulates ML classification for urgency, type, route, and confidence.
    Uses deterministic rules instead of random fallbacks.
    """
    msg_l = message.lower()

    # Pre-check for figurative or clearly non-clinical
    if is_obvious_nonclinical_figurative(message):
        return {
            "urgency_label": "routine",
            "type_label": "admin",
            "route_label": "front desk",
            "confidence": 0.25
        }

    # 1. Determine Urgency and Type
    if any(word in msg_l for word in ["blood", "pain", "hurt", "dying", "faint", "weakness", "fever", "allergic", "suicide", "suicidal"]):
        urgency = "emergency"
        msg_type = "symptom"
    elif any(word in msg_l for word in ["urgent", "infection", "rash", "tired", "worse"]):
        urgency = "urgent"
        msg_type = "symptom"
    elif any(word in msg_l for word in ["appointment", "schedule", "reschedule", "insurance", "form", "open", "time", "update"]):
        urgency = "routine"
        msg_type = "admin"
    elif any(word in msg_l for word in ["refill", "pill", "prescription", "medication", "lisinopril", "antibiotic"]):
        urgency = "routine"
        msg_type = "medication"
    elif any(word in msg_l for word in ["check", "result", "record", "follow up", "follow-up"]):
        urgency = "routine"
        msg_type = "follow-up"
    else:
        # Default fallback
        urgency = "routine"
        msg_type = "symptom"

    # 2. Determine Route based on explicit rules
    if msg_type == "admin":
        route = "front desk"
    elif msg_type == "medication" or (msg_type == "follow-up" and urgency == "routine") or (msg_type == "symptom" and urgency == "routine"):
        route = "nurse pool"
    elif (urgency in ["urgent", "emergency"]) and msg_type == "symptom":
        route = "physician"
    else:
        route = "nurse pool" # fallback for any uncovered combos (like urgent follow-up)

    # Deterministic pseudo-random confidence (stable based on message length & urgency)
    base_conf = 0.70
    if len(message) >= 30:
        base_conf += 0.15
    if urgency == "emergency":
        base_conf += 0.10

    confidence = round(min(0.99, max(0.50, base_conf)), 2)

    return {
        "urgency_label": urgency,
        "type_label": msg_type,
        "route_label": route,
        "confidence": confidence
    }

def process_message_pipeline(
    message: str,
    dataset_row=None,
    inference_mode="Rules Only",
    genai_temperature: float = 0.0,
    genai_api_key: str | None = None,
    genai_model_name: str | None = None,
    use_llm_judge: bool = True
):
    """
    Orchestrates the AI classification pipeline.
    If testing with the mock dataset, it uses the dataset values.
    Otherwise, it runs the stub classifier.
    """
    result = {}

    # 1. Safety / Rules Step
    escalation_reason = safety_check(message)
    manual_review_required = False
    review_reason = None

    genai_status = "None"
    res_genai = None
    genai_draft = ""
    result["genai_prompt"] = None
    result["genai_temperature"] = None
    result["judge_status"] = "None"
    result["judge_verdict"] = None
    result["judge_rationale"] = None
    result["judge_applied"] = False

    if inference_mode == "Purdue GenAI Assisted" and not escalation_reason:
        try_res = call_purdue_genai(
            message,
            temperature=genai_temperature,
            api_key=genai_api_key,
            model_name=genai_model_name
        )
        if try_res["status"] == "Success":
            res_genai = try_res["data"]
            genai_status = "Success"
        else:
            genai_status = try_res["status"]
            result["genai_error"] = try_res.get("error", "Unknown API error.")

    result["genai_status"] = genai_status

    if genai_status == "Success":
        result["urgency_label"] = res_genai["urgency_label"]
        result["type_label"] = res_genai["type_label"]
        result["route_label"] = res_genai["route_label"]
        result["confidence"] = res_genai["confidence"]
        result["rationale"] = res_genai.get("rationale", "")
        genai_draft = res_genai.get("draft_response", "")
        result["genai_prompt"] = res_genai.get("prompt")
        result["genai_temperature"] = res_genai.get("temperature")

        if use_llm_judge:
            judge_res = call_purdue_genai_judge(
                message=message,
                candidate_output={
                    "urgency_label": result["urgency_label"],
                    "type_label": result["type_label"],
                    "route_label": result["route_label"],
                    "confidence": result["confidence"],
                    "draft_response": genai_draft,
                    "rationale": result.get("rationale", "")
                },
                api_key=genai_api_key,
                model_name=genai_model_name
            )
            result["judge_status"] = judge_res.get("status", "Unknown")

            if judge_res.get("status") == "Success":
                judge_data = judge_res.get("data", {})
                verdict = str(judge_data.get("verdict", "")).strip().lower()
                result["judge_verdict"] = verdict
                result["judge_rationale"] = judge_data.get("judge_rationale", "")
                if verdict == "fail":
                    result["urgency_label"] = judge_data.get("corrected_urgency_label", result["urgency_label"])
                    result["type_label"] = judge_data.get("corrected_type_label", result["type_label"])
                    result["route_label"] = judge_data.get("corrected_route_label", result["route_label"])
                    try:
                        result["confidence"] = float(judge_data.get("corrected_confidence", result["confidence"]))
                    except Exception:
                        pass
                    genai_draft = str(judge_data.get("corrected_draft_response", genai_draft))
                    result["judge_applied"] = True
            else:
                result["judge_rationale"] = judge_res.get("error", "Judge call failed.")
                # Fallback: run deterministic local judge so validation still happens.
                fallback_data = local_fallback_judge(
                    message=message,
                    candidate_output={
                        "urgency_label": result["urgency_label"],
                        "type_label": result["type_label"],
                        "route_label": result["route_label"],
                        "confidence": result["confidence"],
                        "draft_response": genai_draft
                    }
                )
                result["judge_status"] = f"{result['judge_status']} (fallback-local)"
                result["judge_verdict"] = fallback_data.get("verdict")
                result["judge_rationale"] = (
                    f"{result.get('judge_rationale', '')} "
                    f"Using local fallback judge: {fallback_data.get('judge_rationale', '')}"
                ).strip()
                if fallback_data.get("verdict") == "fail":
                    result["urgency_label"] = fallback_data.get("corrected_urgency_label", result["urgency_label"])
                    result["type_label"] = fallback_data.get("corrected_type_label", result["type_label"])
                    result["route_label"] = fallback_data.get("corrected_route_label", result["route_label"])
                    try:
                        result["confidence"] = float(fallback_data.get("corrected_confidence", result["confidence"]))
                    except Exception:
                        pass
                    genai_draft = str(fallback_data.get("corrected_draft_response", genai_draft))
                    result["judge_applied"] = True
    else:
        # Fall back to rules classifier (stub_classifier), conservative rationale, empty draft
        if inference_mode == "Purdue GenAI Assisted" and genai_status != "None":
            stub_res = stub_classifier(message)
            result["urgency_label"] = stub_res["urgency_label"]
            result["type_label"] = stub_res["type_label"]
            result["route_label"] = stub_res["route_label"]
            result["confidence"] = stub_res["confidence"]
            result["rationale"] = "Fallback inference activated due to GenAI Studio failure."
            genai_draft = ""
        else:
            # 2 & 3. Classify and Route Steps (Rules Fallback)
            if dataset_row is not None:
                # Use existing mock knowledge for demo
                result["urgency_label"] = dataset_row.get("urgency_label")
                result["type_label"] = dataset_row.get("type_label")
                result["route_label"] = dataset_row.get("route_label")
                result["confidence"] = float(dataset_row.get("confidence", 0.0))
                # Override escalation reason if data has one explicitly (for the edge case ones)
                if pd.isna(escalation_reason) and pd.notna(dataset_row.get("escalation_reason")):
                     escalation_reason = dataset_row["escalation_reason"]
            else:
                # Use stub classifier for custom input
                stub_res = stub_classifier(message)
                result["urgency_label"] = stub_res["urgency_label"]
                result["type_label"] = stub_res["type_label"]
                result["route_label"] = stub_res["route_label"]
                result["confidence"] = stub_res["confidence"]

    # Always return a rationale string for UI consistency.
    rationale_raw = str(result.get("rationale", "")).strip()
    if (not rationale_raw) or rationale_raw.lower() == "no rationale was returned by the model.":
        if genai_status == "Success":
            result["rationale"] = (
                f"Model output summary: urgency={result.get('urgency_label')}, "
                f"type={result.get('type_label')}, route={result.get('route_label')}, "
                f"confidence={result.get('confidence', 0):.2f}. "
                "The model did not provide a detailed rationale string."
            )
        elif inference_mode == "Purdue GenAI Assisted" and escalation_reason:
            result["rationale"] = "GenAI call skipped because message matched safety escalation rules."
        elif inference_mode == "Purdue GenAI Assisted" and genai_status == "None":
            result["rationale"] = "GenAI was not invoked for this request."
        else:
            result["rationale"] = "Rationale not available for this inference path."

    # 3.5 Post-LLM Normalization Step
    if is_obvious_nonclinical_figurative(message) and not escalation_reason:
        # Figurative language with NO non-respiratory red flags present
        # Normalize AI classifications down from emergency/urgent
        if result.get("urgency_label") in ["urgent", "emergency"]:
            result["urgency_label"] = "routine"
            # Force confidence low to trigger manual review
            result["confidence"] = min(float(result.get("confidence", 0.0)), 0.49)
            result["rationale"] = (
                f"{result.get('rationale', '')} [SYSTEM NORMALIZED: "
                "Figurative/romantic language detected with no other clinical red flags.]"
            ).strip()

    # Low confidence override rule
    CONFIDENCE_THRESHOLD = 0.80
    if result["confidence"] < CONFIDENCE_THRESHOLD:
        manual_review_required = True
        if not review_reason:
            review_reason = f"Low confidence score ({result['confidence']:.2f} < {CONFIDENCE_THRESHOLD})"

    # Update urgency if escalated via clinical safety rules
    if escalation_reason and result["urgency_label"] == "routine":
         result["urgency_label"] = "urgent"

    result["escalation_reason"] = escalation_reason
    result["manual_review_required"] = manual_review_required
    result["review_reason"] = review_reason
    result["requires_clinician_review"] = bool(escalation_reason) or result["urgency_label"] in ["urgent", "emergency"]

    # 4. Drafting Step
    # Draft is ONLY generated for routine, non-escalated, high-confidence cases
    if result["urgency_label"] == "routine" and not result["requires_clinician_review"] and not manual_review_required:
        if genai_status == "Success" and genai_draft.strip() != "":
            result["draft_response"] = genai_draft
        elif dataset_row is not None and pd.notna(dataset_row.get("draft_response")) and dataset_row.get(
            "draft_response", "").strip() != "":
            result["draft_response"] = dataset_row["draft_response"]
        else:
            result["draft_response"] = None
    else:
        result["draft_response"] = None

    # Hard post-processing guard to clear draft_response if urgency is urgent/emergency or manual_review_required
    if result["requires_clinician_review"] or result["manual_review_required"]:
        result["draft_response"] = None

    return result

def evaluate_offline_dataset(csv_path: str, inference_mode: str = "Rules Only", max_samples: int = None, api_key: str = None, model_name: str = None):
    """
    Helper function to perform offline validation of the triage engine on a large dataset.
    This bypasses the Streamlit UI and tests the raw logic performance.
    """
    import os
    if not os.path.exists(csv_path):
        return {"status": "error", "message": f"Dataset {csv_path} not found."}

    df = pd.read_csv(csv_path)

    # Normalize schema if new format is provided
    rename_map = {
        "Patient_Message": "patient_message",
        "Urgency_Class": "urgency_label",
        "Type_Class": "type_label",
        "Draft_Response": "draft_response"
    }
    df.rename(columns=rename_map, inplace=True)

    required_cols = ["patient_message", "urgency_label", "type_label"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        return {"status": "error", "message": f"Missing required columns in dataset: {missing}"}

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    urgency_matches = 0
    type_matches = 0
    escalations = 0
    autodrafts = 0
    urgent_true_positives = 0
    urgent_false_negatives = 0
    total_urgent_expected = 0
    genai_fallbacks = 0

    successful_rows = 0

    for _, row in df.iterrows():
        try:
            res_pred = process_message_pipeline(
                row['patient_message'], 
                dataset_row=None, 
                inference_mode=inference_mode,
                genai_api_key=api_key,
                genai_model_name=model_name
            )
        except Exception as e:
            print(f"[evaluate_offline_dataset] Row failed: {type(e).__name__}: {e}")
            continue

        successful_rows += 1

        if inference_mode == "Purdue GenAI Assisted" and res_pred.get("genai_status") not in ["Success", "None"]:
            genai_fallbacks += 1

        if res_pred["urgency_label"] == row["urgency_label"]:
            urgency_matches += 1
        if res_pred["type_label"] == row["type_label"]:
            type_matches += 1

        is_expected_urgent = row["urgency_label"] in ["urgent", "emergency"]
        if is_expected_urgent:
            total_urgent_expected += 1
            if res_pred["urgency_label"] in ["urgent", "emergency"]:
                urgent_true_positives += 1
            elif res_pred["urgency_label"] == "routine":
                urgent_false_negatives += 1

        if res_pred.get("escalation_reason"):
            escalations += 1
        elif res_pred.get("urgency_label") == "routine" and res_pred.get("confidence") >= 0.80:
            autodrafts += 1

    if successful_rows == 0:
        return {"status": "error", "message": "All rows failed processing."}

    urgent_recall = urgent_true_positives / total_urgent_expected if total_urgent_expected > 0 else None

    return {
        "status": "success",
        "urgency_acc": urgency_matches / successful_rows,
        "type_acc": type_matches / successful_rows,
        "escalation_rate": escalations / successful_rows,
        "autodraft_rate": autodrafts / successful_rows,
        "urgent_recall": urgent_recall,
        "genai_fallbacks": genai_fallbacks,
        "urgent_false_negatives": urgent_false_negatives,
        "total_urgent_expected": total_urgent_expected,
        "total_messages": successful_rows
    }
