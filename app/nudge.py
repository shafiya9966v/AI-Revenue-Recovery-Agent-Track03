"""
Hinglish recovery nudge generator.

WHERE THIS FITS IN THE ARCHITECTURE:
This is the ONLY place in the entire pipeline that calls an LLM. Everything
else (diagnosis = ML classifier, decision = rules, action execution =
deterministic code) is deliberately NOT LLM-based, because those decisions
need to be reproducible, auditable, and immune to a model "changing its mind."

Natural-language customer messaging is different: it's low-stakes (worst case,
a slightly awkward sentence - not a wrong money decision) and it's exactly the
kind of task LLMs are good at that rule-based code is bad at. That's the whole
argument for using an LLM here and nowhere else in the pipeline.

BEHAVIOUR WITHOUT AN API KEY:
If MISTRAL_API_KEY is not set in .env, this module automatically falls back
to a small set of pre-written Hinglish templates - deterministic, no network
call - so the pipeline still runs end-to-end without needing live credentials.
This mirrors the same simulation-fallback pattern used in app/act.py for
Razorpay, for consistency.
"""
import os
from dotenv import load_dotenv
from app.schemas import Decision, PaymentFailedEvent, ActionType

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
NUDGE_SIMULATION_MODE = not MISTRAL_API_KEY

# Only these actions involve messaging a customer - no point generating
# a nudge for a silent retry or a human escalation.
MESSAGE_ELIGIBLE_ACTIONS = {
    ActionType.notify_customer,
    ActionType.request_new_instrument,
    ActionType.mandate_retry_sequence,
}

if not NUDGE_SIMULATION_MODE:
    from mistralai.client import Mistral
    _client = Mistral(api_key=MISTRAL_API_KEY)

# Deterministic fallback templates, nested by language and action type,
# used when no API key is configured or when the LLM call fails.
_FALLBACK_TEMPLATES = {
    "hinglish": {
        ActionType.notify_customer: (
            "Namaste! Aapka payment of Rs.{amount} complete nahi ho paya tha. "
            "Koi baat nahi, aap yaha click karke dobara try kar sakte hain: {link}"
        ),
        ActionType.request_new_instrument: (
            "Hi! Lagta hai aapka card ya payment method expire ho gaya hai. "
            "Rs.{amount} ka payment complete karne ke liye, please ek naya "
            "payment method add karein: {link}"
        ),
        ActionType.mandate_retry_sequence: (
            "Aapka subscription payment of Rs.{amount} is baar process nahi ho saka. "
            "Hum jald hi dobara try karenge - agar aap chahte hain to abhi bhi pay kar sakte hain: {link}"
        ),
    },
    "telugu_english": {
        ActionType.notify_customer: (
            "Namaskaram! Mee Rs.{amount} payment complete kaaledhu. "
            "No worries, yee link click chesi dobara try cheyandi: {link}"
        ),
        ActionType.request_new_instrument: (
            "Hi! Mee card/payment method expire ayyinattu undhi. "
            "Rs.{amount} payment complete cheyadaniki, please naya payment method add cheyandi: {link}"
        ),
        ActionType.mandate_retry_sequence: (
            "Mee subscription payment of Rs.{amount} process kaaledhu. "
            "Memu malli try chesthamu, kani direct ga pay cheyyadaniki yee link click cheyyandi: {link}"
        ),
    },
    "english": {
        ActionType.notify_customer: (
            "Hello! Your payment of Rs.{amount} could not be completed. "
            "You can try again by clicking here: {link}"
        ),
        ActionType.request_new_instrument: (
            "Hi! It looks like your card has expired. "
            "To complete your payment of Rs.{amount}, please add a new payment method: {link}"
        ),
        ActionType.mandate_retry_sequence: (
            "Your subscription payment of Rs.{amount} could not be processed. "
            "We will retry shortly, but you can also pay directly here: {link}"
        ),
    }
}


def generate_recovery_nudge(decision: Decision, event: PaymentFailedEvent, payment_link: str = "razorpay.me/xyz") -> str | None:
    """
    Returns a short, warm, localized nudge message for the customer, or None
    if this action type doesn't involve customer messaging.
    """
    if decision.action not in MESSAGE_ELIGIBLE_ACTIONS:
        return None

    lang = getattr(event, "preferred_language", "hinglish")
    if lang is None:
        lang = "hinglish"
    lang = lang.lower()
    if lang not in _FALLBACK_TEMPLATES:
        if "tel" in lang:
            lang = "telugu_english"
        elif lang == "english":
            lang = "english"
        else:
            lang = "hinglish"

    if NUDGE_SIMULATION_MODE:
        template = _FALLBACK_TEMPLATES[lang][decision.action]
        return template.format(amount=int(event.amount), link=payment_link)

    return _generate_with_mistral(decision, event, payment_link)


def _generate_with_mistral(decision: Decision, event: PaymentFailedEvent, payment_link: str) -> str:
    lang = getattr(event, "preferred_language", "hinglish")
    if lang is None:
        lang = "hinglish"
    lang = lang.lower()

    if "tel" in lang:
        language_style_instructions = """Write the message in TELUGU-ENGLISH (Telugish) - a natural mix of Telugu and English words written in Latin/Roman script. Do NOT use Telugu script. Write the way modern, urban Telugu speakers chat on WhatsApp/SMS.

Example of the STYLE required (do not copy this exact wording, just match the style):
"Ayyoo, mee Rs.1850 payment complete kaaledhu. No worries, yee link click chesi simple ga payment complete cheyyandi: razorpay.me/xyz" """
    elif lang == "english":
        language_style_instructions = """Write the message in warm, plain English. Keep it simple and clear.

Example of the STYLE required (do not copy this exact wording, just match the style):
"Oops, your payment of Rs.1850 failed. No worries, you can try again using this link: razorpay.me/xyz" """
    else:
        language_style_instructions = """Write the message in HINGLISH - a natural mix of Hindi and English written in Latin/Roman script. Do NOT use Devanagari script. Write the way modern, urban Hindi speakers chat on WhatsApp/SMS.

Example of the STYLE required (do not copy this exact wording, just match the style):
"Arey, aapka Rs.1850 ka payment complete nahi ho paya. Koi baat nahi, bas yahan click karke dobara try kar lijiye: razorpay.me/xyz" """

    prompt = f"""You are a customer messaging assistant for an Indian payments company.

{language_style_instructions}

Context:
- The customer's payment of Rs.{int(event.amount)} failed.
- Root cause: {decision.root_cause.value}
- Recommended action: {decision.action.value}
- Payment link to include: {payment_link}

Requirements:
- MUST follow the language instructions above. Do NOT write in plain English unless the language is explicitly English.
- Use a warm and reassuring tone.
- 1-2 sentences max, SMS/WhatsApp length.
- Include the payment link naturally.
- No emojis, no marketing hype.
- Output ONLY the message text, nothing else (no preamble, no quotes)."""

    try:
        response = _client.chat.complete(
            model=MISTRAL_MODEL,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content
        return text.strip().strip('"').strip("'")
    except Exception as e:
        # Graceful fallback: if the LLM call fails (rate limit, network,
        # malformed response), don't break the recovery flow - fall back
        # to the deterministic template instead of leaving the customer
        # with no message at all.
        lang_key = "telugu_english" if "tel" in lang else ("english" if lang == "english" else "hinglish")
        template = _FALLBACK_TEMPLATES[lang_key][decision.action]
        return template.format(amount=int(event.amount), link=payment_link)
