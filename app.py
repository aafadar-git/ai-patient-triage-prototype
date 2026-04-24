import os
import sys
import inspect
import importlib

# Force local directory to the absolute front of sys.path to strictly prioritize the local module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit as st
import datetime

import logic
# Force Streamlit to ignore cached sys.modules memory and absolutely re-parse logic.py
importlib.reload(logic)

# --- configuration & data ---
st.set_page_config(page_title="AI Patient Triage Prototype", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "mock_messages.csv")

@st.cache_data
def load_mock_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df.fillna('', inplace=True)
        return df
    return pd.DataFrame()

# --- state management ---
if 'review_log' not in st.session_state:
    st.session_state.review_log = []

# --- UI Header & Warning Banner ---
st.warning("⚠️ **DECISION SUPPORT ONLY**: This application is a human-in-the-loop AI prototype. It is NOT an autonomous clinical system. All AI outputs must be reviewed by qualified medical staff.")
st.title("Patient Portal Message Triage")

# Layout
with st.sidebar:
    inference_mode = st.radio("Inference Mode", ["Rules Only", "Purdue GenAI Assisted"])
    use_llm_judge = st.checkbox(
        "Enable LLM Judge Validation",
        value=True,
        help="Runs a second LLM pass to validate/correct classification, confidence, and draft response."
    )
    genai_temperature = st.slider(
        "GenAI Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Higher values increase response variability. Use this to validate confidence and prompt behavior."
    )
    st.divider()
    with st.expander("About this prototype", expanded=True):
        st.write("**The Problem:** Inbox overload leads to clinician burnout and risks missing urgent medical issues.")
        st.write("**The Solution:** A text-based generative & agentic AI tool for triaging messages and routing them appropriately.")
        st.write("**Safety Constraints:** Enforces strict red-flag rules, confidence thresholds, and clinician review for high-risk / low-confidence items.")

    st.divider()
    st.subheader("Deployment Readiness", anchor=False)
    
    # Check secrets securely, prioritizing os.environ
    api_key = os.environ.get("PURDUE_GENAI_API_KEY")
    if not api_key:
        try:
            if hasattr(st, "secrets"):
                api_key = st.secrets.get("PURDUE_GENAI_API_KEY")
        except Exception:
            api_key = None
            
    if api_key:
        st.success("✅ **GenAI API Key:** Configured")
    else:
        st.error("❌ **GenAI API Key:** Missing")
        
    model_name = os.environ.get("PURDUE_GENAI_MODEL")
    if not model_name:
        try:
            if hasattr(st, "secrets"):
                model_name = st.secrets.get("PURDUE_GENAI_MODEL")
        except Exception:
            model_name = None
            
    if not model_name:
        model_name = "llama3.1:latest"
        
    st.info(f"⚙️ **Model Target:** `{model_name}`")
    
    st.divider()
    st.caption(f"**Debug Path:** `{getattr(logic, '__file__', 'Unknown')}`")
    try:
        st.caption(f"**Debug Signature:** `{inspect.signature(logic.process_message_pipeline)}`")
    except Exception as e:
        st.caption(f"**Debug Signature Error:** `{e}`")

col1, col2 = st.columns([1, 1], gap="large")

# Load data
df_mock = load_mock_data()

with col1:
    st.subheader("1. Select or Enter Message")
    input_method = st.radio("Input Method", ["Load from Mock Dataset", "Enter Custom Message"], horizontal=True)
    
    selected_message = ""
    dataset_row = None
    
    if input_method == "Load from Mock Dataset" and not df_mock.empty:
        # Let user pick a message
        options = df_mock.apply(lambda row: f"{row['message_id']}: {row['patient_message'][:40]}...", axis=1).tolist()
        selected_option = st.selectbox("Select Mock Message", options)
        idx = options.index(selected_option)
        dataset_row = df_mock.iloc[idx].to_dict()
        selected_message = dataset_row["patient_message"]
        
        st.text_area("Patient Message Content", value=selected_message, height=150, disabled=True)
    else:
        selected_message = st.text_area("Enter Patient Message", height=150, placeholder="e.g. Can I get a refill on my lisinopril?")

with col2:
    st.subheader("2. AI Analysis & Triage")
    if selected_message.strip():
        # RUN AI PIPELINE
        result = logic.process_message_pipeline(
            selected_message,
            dataset_row=dataset_row,
            inference_mode=inference_mode,
            genai_temperature=genai_temperature,
            genai_api_key=api_key,
            genai_model_name=model_name,
            use_llm_judge=use_llm_judge
        )
        
        # Display Badges
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Urgency", result["urgency_label"].upper())
        c2.metric("Predicted Type", result["type_label"].title())
        c3.metric("Suggested Route", result["route_label"].title())
        
        # Confidence Score
        conf_color = "green" if result["confidence"] >= 0.80 else "red"
        st.markdown(f"**Confidence Score:** :{conf_color}[**{result['confidence']:.2f}**]")
        
        if result.get("genai_status") == "Success":
            st.caption("🤖 Model-assisted output via Purdue GenAI Studio")
            st.caption(f"Temperature Used: `{result.get('genai_temperature', 0.0):.2f}`")
            with st.expander("Prompt Used for GenAI", expanded=False):
                st.code(result.get("genai_prompt") or "Prompt not available.", language="text")
        elif inference_mode == "Purdue GenAI Assisted" and result.get("genai_status") not in ["Success", "None"]:
            st.warning("⚠️ **Purdue GenAI unavailable or failed; reverted to rules-only inference.**")
            with st.expander("Show Diagnostic Error", expanded=True):
                st.error(f"**Status Code:** {result.get('genai_status')}\n\n**Details:** {result.get('genai_error')}")

        rationale_text = str(result.get("rationale") or "").strip()
        if not rationale_text:
            if inference_mode == "Rules Only":
                rationale_text = (
                    "Rules-only mode is active. This output comes from deterministic rules/stub logic, "
                    "not a model-generated rationale."
                )
            elif result.get("genai_status") == "Success":
                rationale_text = "Model call succeeded but returned no rationale text."
            else:
                rationale_text = (
                    f"No model rationale available. GenAI status: {result.get('genai_status', 'Unknown')}."
                )

        with st.expander("AI Rationale", expanded=True):
            st.write(rationale_text)

        if inference_mode == "Purdue GenAI Assisted" and use_llm_judge:
            with st.expander("LLM Judge Validation", expanded=False):
                st.write(f"**Judge Status:** {result.get('judge_status', 'None')}")
                st.write(f"**Judge Verdict:** {result.get('judge_verdict') or 'N/A'}")
                st.write(f"**Judge Applied Corrections:** {result.get('judge_applied', False)}")
                st.write(result.get("judge_rationale") or "No judge rationale available.")
            
        st.divider()
        
        # Escalation / Drafting Panel
        if result.get("escalation_reason"):
            st.error(f"🚨 **ESCALATE TO CLINICIAN IMMEDIATELY**\n\n**Reason:** {result['escalation_reason']}")
            st.info("ℹ️ Auto-draft is disabled for escalated or non-routine messages.")
            draft = None
        else:
            st.success("✅ **Cleared for Auto-Drafting** (Routine issue, High Confidence)")
            if result.get("draft_response"):
                draft = st.text_area("Generated Draft Response", value=result["draft_response"], height=120)
            else:
                st.info(
                    "ℹ️ No auto-draft text was generated. Drafts are only produced for routine, "
                    "non-escalated, high-confidence messages."
                )
                draft = None
        
        st.divider()
        
        # Action Log buttons
        st.write("**Human-in-the-Loop Actions**")
        
        with st.form("human_override_form"):
            st.caption("Review & Override Labels")
            oc1, oc2 = st.columns(2)
            o_urg = oc1.selectbox("Final Urgency", ["routine", "urgent", "emergency"], index=["routine", "urgent", "emergency"].index(result["urgency_label"]))
            o_route = oc2.selectbox("Final Route", ["nurse pool", "physician", "front desk"], index=["nurse pool", "physician", "front desk"].index(result["route_label"]))
            
            c1, c2, c3, c4 = st.columns(4)
            app_btn = c1.form_submit_button("Approve Draft", disabled=bool(result.get("escalation_reason")))
            edit_btn = c2.form_submit_button("Edit Draft", disabled=bool(result.get("escalation_reason")))
            esc_btn = c3.form_submit_button("Escalate")
            rer_btn = c4.form_submit_button("Reroute")
            
        def handle_action(action_name):
            log_entry = {
                "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Message": selected_message[:50] + "...",
                "AI Urgency": result["urgency_label"],
                "Final Urgency": o_urg,
                "AI Route": result["route_label"],
                "Final Route": o_route,
                "Confidence": result["confidence"],
                "Escalation Reason": result.get("escalation_reason", "None"),
                "Reviewer Action": action_name
            }
            st.session_state.review_log.append(log_entry)
            st.success(f"Action '{action_name}' recorded.")

        if app_btn: handle_action("Approved Draft")
        if edit_btn: handle_action("Edited Draft")
        if esc_btn: handle_action("Escalated to Clinician")
        if rer_btn: handle_action("Rerouted Message")

st.divider()

# --- Dashboard & Review Log Sections ---
st.subheader("3. System Evaluation Dashboard & Audit Log")

tab_queue, tab1, tab2, tab3 = st.tabs(["Reviewer Queues", "Review Log", "Dashboard Analytics", "Governance & Safety"])

with tab_queue:
    st.markdown("### Initial Inbox Queues")
    if not df_mock.empty:
        escalated_queue = []
        routine_queue = []
        admin_queue = []
        
        for _, r in df_mock.iterrows():
            r_eval = logic.process_message_pipeline(
                r['patient_message'],
                dataset_row=r.to_dict(),
                genai_temperature=genai_temperature
            )
            item = {
                "Message ID": r['message_id'],
                "Patient Message": r['patient_message'],
                "Predicted Urgency": r_eval['urgency_label'],
                "Predicted Route": r_eval['route_label']
            }
            if r_eval.get("escalation_reason"):
                item["Escalation Reason"] = r_eval["escalation_reason"]
                escalated_queue.append(item)
            elif r_eval["route_label"] == "front desk":
                admin_queue.append(item)
            else:
                routine_queue.append(item)
                
        st.subheader("🚨 Escalated Review Queue", anchor=False)
        st.dataframe(pd.DataFrame(escalated_queue), use_container_width=True)
        
        st.subheader("📝 Routine Draft Queue", anchor=False)
        st.dataframe(pd.DataFrame(routine_queue), use_container_width=True)
        
        st.subheader("🏢 Admin / Front Desk Queue", anchor=False)
        st.dataframe(pd.DataFrame(admin_queue), use_container_width=True)
    else:
        st.info("No mock data available for queues.")

with tab1:
    if st.session_state.review_log:
        df_log = pd.DataFrame(st.session_state.review_log)
        st.dataframe(df_log, use_container_width=True)
    else:
        st.info("No actions recorded yet. Perform actions to see the review log.")

with tab2:
    if not df_mock.empty:
        total_msgs = len(df_mock)
        
        # Evaluate how many would be escalated
        # (Assuming the mock data contains the base info, apply our PIPELINE to it all)
        escalate_count = 0
        routine_draft_count = 0
        
        # Evaluation Counters
        urgency_matches = 0
        type_matches = 0
        
        for _, row in df_mock.iterrows():
            # Pipeline with dataset for demo tracking
            res = logic.process_message_pipeline(
                row['patient_message'],
                dataset_row=row.to_dict(),
                genai_temperature=genai_temperature
            )
            if res.get("escalation_reason"):
                escalate_count += 1
            elif res.get("urgency_label") == "routine" and res.get("confidence") >= 0.80:
                routine_draft_count += 1
            
            # Pure inference without dataset hints to test rule accuracy
            res_pred = logic.process_message_pipeline(
                row['patient_message'],
                dataset_row=None,
                genai_temperature=genai_temperature
            )
            if res_pred["urgency_label"] == row["urgency_label"]:
                urgency_matches += 1
            if res_pred["type_label"] == row["type_label"]:
                type_matches += 1
                
        d1, d2, d3 = st.columns(3)
        d1.metric("Total Messages Analyzed", total_msgs)
        d2.metric("Escalated to Human", escalate_count, help="Due to red flags or low confidence")
        d3.metric("Auto-Drafted (Routine)", routine_draft_count)
        
        st.divider()
        st.write("**Rules Engine Evaluation (Predicted vs Expected)**")
        e1, e2 = st.columns(2)
        e1.metric("Urgency Accuracy", f"{(urgency_matches / total_msgs) * 100:.0f}%", f"{urgency_matches}/{total_msgs} correct")
        e2.metric("Type Accuracy", f"{(type_matches / total_msgs) * 100:.0f}%", f"{type_matches}/{total_msgs} correct")
        
        st.caption("These analytics reflect the current mock dataset processed against the rules engine.")
        
        st.divider()
        st.write("**Offline Engine Evaluation**")
        
        def resolve_dataset_path():
            paths = [
                os.path.join(os.path.dirname(__file__), "data", "train_messages_v3_dynamic.csv"),
                os.path.join(os.getcwd(), "train_messages_v3_dynamic.csv"),
                "/Users/anasafadar/Downloads/train_messages_v3_dynamic.csv"
            ]
            for p in paths:
                if os.path.exists(p):
                    return p
            return None
            
        dataset_path = resolve_dataset_path()
        
        if dataset_path:
            st.caption(f"Resolved dataset path: `{dataset_path}`")
        else:
            st.caption("Dynamic training dataset not found in expected locations.")
            
        if st.button("Run Evaluation on Dynamic Training Set"):
            if dataset_path:
                with st.spinner("Running Offline Evaluation Engines..."):
                    eval_rules = logic.evaluate_offline_dataset(dataset_path, inference_mode="Rules Only")
                    eval_genai = logic.evaluate_offline_dataset(dataset_path, inference_mode="Purdue GenAI Assisted", max_samples=30)
                    
                st.write("---")
                c_rules, c_genai = st.columns(2)
                with c_rules:
                    st.write("**Rules Only (Full Dataset)**")
                    if eval_rules and eval_rules.get("status") == "success":
                        st.metric("Total Messages Analyzed", eval_rules["total_messages"])
                        e1, e2 = st.columns(2)
                        e1.metric("Urgency Acc.", f"{eval_rules['urgency_acc']*100:.1f}%")
                        e2.metric("Type Acc.", f"{eval_rules['type_acc']*100:.1f}%")
                        e3, e4 = st.columns(2)
                        e3.metric("Escalation Rate", f"{eval_rules['escalation_rate']*100:.1f}%")
                        e4.metric("Auto-Draft Rate", f"{eval_rules['autodraft_rate']*100:.1f}%")
                        
                        ur = f"{eval_rules['urgent_recall']*100:.1f}%" if eval_rules["urgent_recall"] is not None else "N/A"
                        st.metric("Urgent Recall (FN count)", f"{ur} ({eval_rules['urgent_false_negatives']} FNs)")
                    else:
                        st.error("Rules Evaluation failed.")
                        
                with c_genai:
                    st.write("**Purdue GenAI Assisted (30 Random Sample limit)**")
                    if eval_genai and eval_genai.get("status") == "success":
                        st.metric("Total Messages Analyzed", eval_genai["total_messages"])
                        e1, e2 = st.columns(2)
                        e1.metric("Urgency Acc.", f"{eval_genai['urgency_acc']*100:.1f}%")
                        e2.metric("Type Acc.", f"{eval_genai['type_acc']*100:.1f}%")
                        e3, e4 = st.columns(2)
                        e3.metric("Escalation Rate", f"{eval_genai['escalation_rate']*100:.1f}%")
                        e4.metric("Auto-Draft Rate", f"{eval_genai['autodraft_rate']*100:.1f}%")
                        
                        ur_gen = f"{eval_genai['urgent_recall']*100:.1f}%" if eval_genai["urgent_recall"] is not None else "N/A"
                        st.metric("Urgent Recall (FN count)", f"{ur_gen} ({eval_genai['urgent_false_negatives']} FNs)")
                    else:
                        st.warning("GenAI Evaluation skipped or failed gracefully.")
            else:
                st.warning("Dynamic training dataset not found in expected locations.")
                
    else:
        st.warning("Mock data not found.")

with tab3:
    st.markdown("### Pilot Deployment Readiness")
    st.info("**PHI / HIPAA Requirement**: All data handling operations must occur within a secure boundary.")
    st.info("**De-identification**: Input messages must be strictly de-identified before model training or inference on real data.")
    st.info("**Human Review Requirement**: Escalated, high-risk, and low-confidence cases enforce manual review.")
    st.info("**Audit Logging**: Comprehensive logging of algorithmic decisions and clinician actions occurs system-wide.")
    st.info("**Fairness Monitoring**: Continued safety audits will stratify metrics by demographic subgroup to ensure equitable outcomes.")
