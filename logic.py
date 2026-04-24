import pandas as pd
import random
import re
import os
import json
import requests

def call_purdue_genai(message: str, temperature: float = 0.0, api_key: str | None = None, model_name: str | None = None):
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
Analyze the following patient portal message for triage.
First, assess whether the message is a valid, serious healthcare portal request. 
If the message is nonsensical, joking, non-medical, clearly invalid, or requests something inappropriate/non-clinical (e.g., "refill my diet coke"):
- Keep the triage classification conservative.
- Lower the confidence score appropriately (<0.5).
- Return "draft_response" as an empty string.
- Explain the invalidity strictly in the "rationale".

Return ONLY valid JSON with exactly these keys. All keys are mandatory. Do not omit any key. Return exactly one JSON object with no prose before or after the JSON. If uncertain about a classification, choose the closest valid label:
"urgency_label": (one of "emergency", "urgent", "routine")
"type_label": (one of "symptom", "medication", "admin", "follow-up")
"route_label": (one of "nurse pool", "physician", "front desk")
"confidence": (a float between 0.0 and 1.0)
"draft_response": (a patient-facing draft response if a valid routine clinical request, otherwise empty string)
"rationale": (a string explaining the reasoning)

Draft Response Rules for Valid Routine Messages:
- Assess the FULL context of the message—do not keyword-match a single term like "refill".
- If a refill request does not mention a plausible medication or sounds unserious/invalid, do NOT produce a normal refill response.
- Must be a realistic, patient-facing reply.
- 2-4 sentences maximum.
- Professional and empathetic tone.
- NO diagnosis.
- NO unsafe medical advice.
- Do NOT mention "AI-generated" or refer to an AI inside this text.
- If it is a valid refill/admin/follow-up request, acknowledge receipt and describe the next step appropriately.
- If the content is ambiguous, keep the message conservative and advise clinic follow-up.

Patient Message:
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
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return {"status": "APIError", "error": f"Network or HTTP error: {str(e)}"}
        
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
        
    required_keys = ["urgency_label", "type_label", "route_label", "confidence", "draft_response", "rationale"]
    
    if "rationale" not in parsed or not str(parsed.get("rationale", "")).strip():
        parsed["rationale"] = "No rationale was returned by the model."
    if "draft_response" not in parsed or parsed["draft_response"] is None:
        parsed["draft_response"] = ""
    if "confidence" not in parsed or parsed["confidence"] is None:
        parsed["confidence"] = 0.5
        
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
RED_FLAGS = [
    "chest pain",
    "shortness of breath",
    "suicidal thoughts",
    "severe bleeding",
    "fainting",
    "fainted",
    "one-sided weakness",
    "allergic reaction",
    "high fever in infant",
    "suicidal"
]

def safety_check(message: str) -> str | None:
    """
    Checks a message for red-flag phrases. 
    Returns the reason for escalation if found, otherwise None.
    """
    message_lower = message.lower()
    flags_found = []
    for flag in RED_FLAGS:
        if re.search(r'\b' + re.escape(flag) + r'\b', message_lower):
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
    genai_model_name: str | None = None
):
    """
    Orchestrates the AI classification pipeline.
    If testing with the mock dataset, it uses the dataset values.
    Otherwise, it runs the stub classifier.
    """
    result = {}
    
    # 1. Safety / Rules Step
    escalation_reason = safety_check(message)
    
    genai_status = "None"
    res_genai = None
    genai_draft = ""
    result["genai_prompt"] = None
    result["genai_temperature"] = None

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
    if not str(result.get("rationale", "")).strip():
        if inference_mode == "Purdue GenAI Assisted" and escalation_reason:
            result["rationale"] = "GenAI call skipped because message matched safety escalation rules."
        elif inference_mode == "Purdue GenAI Assisted" and genai_status == "None":
            result["rationale"] = "GenAI was not invoked for this request."
        else:
            result["rationale"] = "Rationale not available for this inference path."

    # Low confidence override rule
    CONFIDENCE_THRESHOLD = 0.80
    if result["confidence"] < CONFIDENCE_THRESHOLD:
        if not escalation_reason:
            escalation_reason = f"Low confidence score ({result['confidence']:.2f} < {CONFIDENCE_THRESHOLD})"

    # Update urgency if escalated via rules
    if escalation_reason and result["urgency_label"] == "routine":
         result["urgency_label"] = "urgent"

    result["escalation_reason"] = escalation_reason

    # 4. Drafting Step
    # Draft is ONLY generated for routine, non-escalated, high-confidence cases
    if result["urgency_label"] == "routine" and not escalation_reason:
        if genai_status == "Success" and genai_draft.strip() != "":
            result["draft_response"] = genai_draft
        elif dataset_row is not None and pd.notna(dataset_row.get("draft_response")) and dataset_row.get(
            "draft_response", "").strip() != "":
            result["draft_response"] = dataset_row["draft_response"]
        else:
            result["draft_response"] = None
    else:
        result["draft_response"] = None
        
    return result

def evaluate_offline_dataset(csv_path: str, inference_mode: str = "Rules Only", max_samples: int = None):
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
    total = len(df)
    
    for _, row in df.iterrows():
        try:
            res_pred = process_message_pipeline(row['patient_message'], dataset_row=None, inference_mode=inference_mode)
        except Exception as e:
            # Safe catch to ensure one row failure does not crash offline evaluation
            continue
            
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
            
    urgent_recall = urgent_true_positives / total_urgent_expected if total_urgent_expected > 0 else None
            
    return {
        "status": "success",
        "urgency_acc": urgency_matches / total,
        "type_acc": type_matches / total,
        "escalation_rate": escalations / total,
        "autodraft_rate": autodrafts / total,
        "urgent_recall": urgent_recall,
        "genai_fallbacks": genai_fallbacks,
        "urgent_false_negatives": urgent_false_negatives,
        "total_urgent_expected": total_urgent_expected,
        "total_messages": total
    }
