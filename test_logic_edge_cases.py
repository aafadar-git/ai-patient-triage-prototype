"""
False positive / edge case / known limitations test suite for logic.py.
This suite documents and tests how the deterministic rules engine responds
to figurative language, slang, and mixed-signal inputs.
"""

import pytest
from logic import process_message_pipeline

# ==========================================
# TEST GROUP D — Figurative / Lyrical Language (False Positive Stress Tests)
# Purpose: Confirm safety invariants hold and document known behaviors
# where keyword matching triggers or avoids escalation on non-clinical language.
# ==========================================

def test_d1_figurative_breathe():
    """Assert figurative use of 'breathe' is routed as routine with no draft."""
    result = process_message_pipeline(
        "I can't breathe without you near me, I miss you so much.",
        inference_mode="Rules Only"
    )
    assert result["escalation_reason"] is None
    assert result["urgency_label"] == "routine"
    assert result["draft_response"] is None
    # NOTE: No false positive here — "breathe" is not in RED_FLAGS (only "shortness of breath" is)

def test_d2_figurative_heart():
    """Assert figurative use of 'heart' is routed as routine with no draft."""
    result = process_message_pipeline(
        "My heart skips a beat every time I think of my vacation.",
        inference_mode="Rules Only"
    )
    assert result["urgency_label"] == "routine"
    assert result["escalation_reason"] is None
    assert result["urgency_label"] != "emergency"
    assert result["draft_response"] is None
    # NOTE: "heart" alone does not trigger escalation in this rules engine

def test_d3_figurative_wellness_checkin():
    """Assert mild wellness check-in is not classified as emergency."""
    result = process_message_pipeline(
        "I've been feeling under the weather 🤧 but nothing serious, just wanted to check in.",
        inference_mode="Rules Only"
    )
    # Invariant 3: draft only on routine with no escalation
    if result.get("urgency_label") == "routine" and not result.get("escalation_reason"):
        assert result.get("draft_response") is None  # stub never produces draft text
    else:
        assert result.get("draft_response") is None
    # At minimum: no emergency classification for a mild wellness check-in
    assert result["urgency_label"] != "emergency"


# ==========================================
# TEST GROUP E — Emoji Slang (Ambiguous Intent Tests)
# Purpose: Confirm that emoji+slang use of clinical words triggers correct
# safety routing and documents that the rules engine cannot distinguish slang.
# ==========================================

def test_e1_emoji_slang_dying():
    """Assert slang use of 'dying' escalates to emergency with no draft."""
    result = process_message_pipeline(
        "I'm dying 💀💀 this waiting room music is so bad lol",
        inference_mode="Rules Only"
    )
    assert result["urgency_label"] == "emergency"
    assert result["escalation_reason"] is None
    assert result["draft_response"] is None
    # Known limitation: stub_classifier cannot distinguish "dying" (slang) from clinical use.
    # GenAI-assisted mode resolves this via full context window.

def test_e2_emoji_slang_dying_refill():
    """Assert 'dying' slang in refill request escalates to emergency."""
    result = process_message_pipeline(
        "omg I'm literally dying 💀 need my refill asap",
        inference_mode="Rules Only"
    )
    assert result["draft_response"] is None
    assert result["urgency_label"] == "emergency"
    assert result["escalation_reason"] is None
    # Known limitation: refill request buried in emoji slang gets flagged as emergency by stub.


# ==========================================
# TEST GROUP F — Mixed Signal Messages (Red Flag in Non-Clinical Context)
# Purpose: Confirm that red flags embedded in clearly non-clinical sentences
# trigger safety pre-emption (or fallback rules) and suppress the draft.
# ==========================================

def test_f1_mixed_signal_breathe():
    """Assert non-clinical 'can't breathe' is handled correctly."""
    result = process_message_pipeline(
        "I can't breathe I'm laughing so hard at this. Anyway, can I reschedule my appointment?",
        inference_mode="Rules Only"
    )
    # Safety invariant: if escalation_reason is present, draft must be None
    if result.get("escalation_reason"):
        assert result["draft_response"] is None
    else:
        # No escalation fired — "breathe" is not a RED_FLAG
        assert result["urgency_label"] == "routine"
        assert result["draft_response"] is None  # stub never produces draft text
    
    # In either case:
    assert result["draft_response"] is None
    # Note: This message does NOT trigger red flag escalation in current rules engine
    # because only the exact phrase "shortness of breath" is a RED_FLAG, not "breathe".
    # This is actually correct behavior — the message is clearly non-clinical.

def test_f2_mixed_signal_chest_hurts():
    """Assert non-clinical 'chest hurts' triggers emergency fallback in rules mode."""
    result = process_message_pipeline(
        "My chest hurts from laughing so much, I need a good doctor recommendation 😂",
        inference_mode="Rules Only"
    )
    assert result["draft_response"] is None
    assert result["urgency_label"] == "emergency"
    assert result["escalation_reason"] is None
    # Known false positive: stub maps "hurt" → emergency. GenAI mode handles context.
    # Safety property (no draft) is preserved regardless.
