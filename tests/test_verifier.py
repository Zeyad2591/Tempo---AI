from tempo.verifier.verifier import agreement, confidence, final_number, grounding, is_refusal


def test_final_number_prefers_final_answer_line():
    assert final_number("36 pens... Final answer: £10.20") == 10.2
    assert final_number("It is 1,250 kg") == 1250
    assert final_number("no numbers") is None


def test_agreement_numbers_and_words():
    assert agreement("Final answer: 129", "The sum is 129.") == 1.0
    assert agreement("Final answer: 129", "Final answer: 128") == 0.0
    assert agreement("Canberra.", "The capital of Australia is Canberra.") >= 0.5
    assert agreement("Canberra.", "Sydney.") == 0.0


def test_grounding():
    ctx = "Customers can request a full refund within 45 days of delivery."
    assert grounding("Customers have 45 days to request a refund.", ctx) > 0.6
    assert grounding("Customers have 90 days, says the CEO Bob.", ctx) < 0.6


def test_refusals_get_low_confidence():
    assert is_refusal("I don't know the answer.")
    conf, _ = confidence(["The context does not say; it is not mentioned in the documents."])
    assert conf < 0.3


def test_confidence_high_when_samples_agree():
    conf, why = confidence(["Canberra", "It is Canberra"])
    assert conf >= 0.6 and "agree" in why
    low, _ = confidence(["Canberra", "Sydney"])
    assert low < 0.6
