"""Streamlit health-information assistant with optional NVIDIA voice features."""
import base64
import binascii
import hashlib
import json
import logging
import os
import re
from pathlib import Path
import urllib.error
import urllib.request

import streamlit as st
from openai import APIStatusError, OpenAI, OpenAIError


GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_FALLBACK_MODEL = "gemini-3.1-flash-lite"
GEMINI_TTS_MODEL = "gemini-3.8-flash-tts"
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
GEMINI_TRANSCRIPTION_PROMPT = (
    "Transcribe the spoken words in this audio accurately. Return only the "
    "transcription, with no explanation."
)
LANGUAGES = {
    "English": {
        "code": "en-US",
        "ui": {
            "health_companion": "YOUR HEALTH COMPANION",
            "hero_description": "Clear, supportive health information whenever you need a place to start.",
            "not_diagnosis": "Text or voice | General guidance | Not a diagnosis",
            "profile": "Your health profile",
            "optional_context": "Optional context helps personalize general guidance.",
            "age": "Age (optional)",
            "conditions": "Known conditions (optional)",
            "medications": "Medications (optional)",
            "privacy": "Your health details and voice are sent to the AI provider. Avoid personal identifiers.",
            "photo_section": "Camera scan (optional)",
            "photo_help": "Capture the affected area. Capturing uploads the photo to this app; it goes to Gemini only when you consent and scan. Avoid faces and identifying details.",
            "photo_capture": "Capture photo",
            "photo_question": "What would you like to know about this photo?",
            "photo_question_placeholder": "For example: Describe the visible skin change",
            "photo_consent": "I agree to send this photo to Gemini for general information",
            "photo_scan": "Scan photo",
            "photo_missing": "Capture a photo first.",
            "photo_consent_required": "Confirm photo sharing before scanning.",
            "photo_provider_required": "Photo scan needs a Gemini API key. The photo was not sent.",
            "photo_default_question": "Describe only visible features in this photo and give general, non-diagnostic guidance. Mention uncertainty and when to seek professional care.",
            "provider_info": "Gemini is used for chat when GEMINI_API_KEY is set.",
            "voice_key_info": "Voice requires a Gemini key or an NVIDIA key with Riva access.",
            "voice_enabled": "Enable voice",
                        "voice_consent": "Send my recording to {provider} for transcription and a reply",
                        "microphone_permission_help": "Your browser will ask for microphone access when you press Record. If access was blocked before, allow it in this site's browser settings and reload.",
                        "voice_consent_required": "Confirm audio sharing to show the recorder.",
            "help": "How can I help?",
            "help_caption": "Describe what you’re feeling in your own words. Get general health information, not a diagnosis or a substitute for professional care.",
            "body_caption": "Choose the area that best matches your symptoms.",
            "body_area": "Where are the symptoms?",
            "body_areas": ("Head or face", "Eyes, ears, nose, or throat", "Chest or breathing", "Heart or circulation", "Stomach or abdomen", "Back or neck", "Arms or hands", "Legs or feet", "Skin", "Whole body or other"),
            "symptoms": "Describe your symptoms",
            "symptoms_placeholder": "For example: when they started and what makes them better or worse",
            "duration": "How long have you had them?",
            "durations": ("Today", "1-3 days", "4-7 days", "More than a week", "On and off"),
            "severity": "How severe do they feel? (0-10)",
            "guidance": "Get general guidance",
            "empty_symptoms": "Describe at least one symptom so I can provide relevant general guidance.",
            "record": "Speak your symptoms or ask a health question",
            "no_speech": "No speech was recognized. Check the recording and try again.",
            "retry": "Retry transcription",
            "chat": "Ask a health question or describe symptoms",
            "voice_error": "Voice transcription failed",
            "voice_playback_error": "Voice playback could not be generated",
            "retry_voice": "Retry spoken reply",
            "said": "You said",
            "reply_error": "Couldn't get a response",
            "thinking": "Preparing a careful response...",
        },
        "red_flag": "This could be an emergency. Please contact your local emergency services now or go to the nearest emergency department. If possible, ask someone nearby to stay with you. I can't assess or rule out an emergency in chat.",
    },
    "हिन्दी": {
        "code": "hi-IN",
        "ui": {
            "health_companion": "आपका स्वास्थ्य सहायक",
            "hero_description": "जब भी शुरुआत करने की ज़रूरत हो, सरल और सहायक स्वास्थ्य जानकारी पाएँ।",
            "not_diagnosis": "टेक्स्ट या आवाज़ | सामान्य जानकारी | निदान नहीं",
            "profile": "आपकी स्वास्थ्य जानकारी",
            "optional_context": "वैकल्पिक जानकारी से सामान्य सुझाव बेहतर हो सकते हैं।",
            "age": "उम्र (वैकल्पिक)",
            "conditions": "पहले से मौजूद स्वास्थ्य समस्याएँ (वैकल्पिक)",
            "medications": "दवाइयाँ (वैकल्पिक)",
            "privacy": "आपकी स्वास्थ्य जानकारी और आवाज़ AI सेवा को भेजी जाती है। नाम या पहचान बताने वाली जानकारी न दें।",
            "photo_section": "कैमरे से स्कैन करें (वैकल्पिक)",
            "photo_help": "प्रभावित हिस्से की फ़ोटो लें। फ़ोटो इस ऐप पर अपलोड होती है और सहमति देकर स्कैन करने पर ही Gemini को भेजी जाती है। चेहरा या पहचान बताने वाली जानकारी न दिखाएँ।",
            "photo_capture": "फ़ोटो लें",
            "photo_question": "इस फ़ोटो के बारे में क्या जानना चाहते हैं?",
            "photo_question_placeholder": "उदाहरण: त्वचा पर दिख रहे बदलाव का वर्णन करें",
            "photo_consent": "मैं सामान्य जानकारी के लिए यह फ़ोटो Gemini को भेजने की सहमति देता/देती हूँ",
            "photo_scan": "फ़ोटो स्कैन करें",
            "photo_missing": "पहले फ़ोटो लें।",
            "photo_consent_required": "स्कैन करने से पहले फ़ोटो भेजने की सहमति दें।",
            "photo_provider_required": "फ़ोटो स्कैन के लिए Gemini API key चाहिए। फ़ोटो नहीं भेजी गई।",
            "photo_default_question": "इस फ़ोटो में केवल दिखाई देने वाली बातों का वर्णन करें और सामान्य, गैर-निदानात्मक जानकारी दें। अनिश्चितता और डॉक्टर से कब संपर्क करना चाहिए, यह बताएँ।",
            "provider_info": "GEMINI_API_KEY सेट होने पर चैट के लिए Gemini का उपयोग होगा।",
            "voice_key_info": "आवाज़ के लिए Gemini कुंजी या Riva सुविधा वाली NVIDIA कुंजी चाहिए।",
            "voice_enabled": "आवाज़ चालू करें",
                        "voice_consent": "रिकॉर्डिंग को ट्रांसक्रिप्शन और जवाब के लिए {provider} को भेजें",
                        "microphone_permission_help": "रिकॉर्ड करें दबाने पर ब्राउज़र माइक्रोफ़ोन की अनुमति माँगेगा। पहले रोक दिया हो तो साइट की ब्राउज़र सेटिंग में अनुमति देकर पेज फिर से लोड करें।",
                        "voice_consent_required": "रिकॉर्डर दिखाने के लिए ऑडियो भेजने की सहमति दें।",
            "help": "मैं आपकी कैसे मदद करूँ?",
            "help_caption": "अपने लक्षण अपने शब्दों में बताएँ। सामान्य स्वास्थ्य जानकारी पाएँ—यह निदान या डॉक्टर की सलाह का विकल्प नहीं है।",
            "body_caption": "लक्षणों के अनुसार शरीर का हिस्सा चुनें।",
            "body_area": "लक्षण कहाँ हैं?",
            "body_areas": ("सिर या चेहरा", "आँख, कान, नाक या गला", "छाती या साँस", "दिल या रक्त संचार", "पेट", "पीठ या गर्दन", "बाँह या हाथ", "पैर या पाँव", "त्वचा", "पूरा शरीर या अन्य"),
            "symptoms": "अपने लक्षण बताएँ",
            "symptoms_placeholder": "जैसे: लक्षण कब शुरू हुए और किससे बेहतर या बदतर होते हैं",
            "duration": "लक्षण कितने समय से हैं?",
            "durations": ("आज", "1-3 दिन", "4-7 दिन", "एक सप्ताह से अधिक", "कभी-कभी"),
            "severity": "लक्षण कितने गंभीर हैं? (0-10)",
            "guidance": "सामान्य सुझाव पाएँ",
            "empty_symptoms": "कृपया कम से कम एक लक्षण बताएँ, ताकि मैं उपयोगी सामान्य जानकारी दे सकूँ।",
            "record": "अपने लक्षण या स्वास्थ्य संबंधी सवाल बोलें",
            "no_speech": "आवाज़ समझ नहीं आई। रिकॉर्डिंग जाँचकर फिर कोशिश करें।",
            "retry": "आवाज़ फिर से पहचानें",
            "chat": "स्वास्थ्य संबंधी सवाल पूछें या लक्षण बताएँ",
            "voice_error": "आवाज़ पहचानने में समस्या",
            "voice_playback_error": "आवाज़ में जवाब तैयार नहीं हो सका",
            "retry_voice": "बोला हुआ जवाब फिर से चलाएँ",
            "said": "आपने कहा",
            "reply_error": "जवाब नहीं मिल सका",
            "thinking": "सावधानी से जवाब तैयार हो रहा है...",
        },
        "red_flag": "यह आपात स्थिति हो सकती है। अभी अपनी स्थानीय आपातकालीन सेवा से संपर्क करें या नज़दीकी आपातकालीन विभाग जाएँ। हो सके तो किसी व्यक्ति को अपने पास रहने के लिए कहें। मैं चैट में आपात स्थिति का आकलन या उसे खारिज नहीं कर सकता।",
    },
}
SYSTEM_PROMPT = """You are DrHealth AI, a friendly, calm health-information assistant.
You are not a doctor and cannot diagnose disease or predict a user's health.
Offer general educational information and help the user decide what to discuss
with a licensed clinician. Ask focused follow-up questions when important details
are missing. Clearly distinguish possibilities from facts, mention uncertainty,
and never recommend starting, stopping, or changing prescription medication.
When asked about symptoms, discuss a few possible explanations without claiming a
diagnosis, suggest only low-risk general self-care, say when to contact a clinician,
and list relevant warning signs that need urgent care.
For potentially urgent symptoms, advise prompt professional evaluation rather
than trying to rule out an emergency. Keep answers concise, empathetic, and
plain-language. If an image is provided, describe only visible features, do not
diagnose from the image, and say when the image is unclear. Do not claim that
you examined the user or reviewed records."""

DOCTOR_ILLUSTRATION = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 360" role="img">
<rect x="8" y="8" width="204" height="344" rx="28" fill="#f0f7fa"/>
<circle cx="110" cy="69" r="38" fill="#d99b72"/>
<path d="M72 67Q73 26 110 29Q147 30 148 67L137 54Q111 64 81 51Z" fill="#34495e"/>
<path d="M95 101h30v28H95z" fill="#d99b72"/>
<path d="M71 124Q110 108 149 124L179 174L157 190L148 166V300H72V166L63 190L41 174Z" fill="#fff" stroke="#9eb8c4" stroke-width="3"/>
<path d="M95 122l15 23 15-23 15 12-13 36H93l-13-36z" fill="#d9eef5"/>
<path d="M110 145v31q26 0 26 25t-26 25q-22 0-22-18" fill="none" stroke="#3b7890" stroke-width="5" stroke-linecap="round"/>
<circle cx="88" cy="208" r="6" fill="#3b7890"/>
<path d="M86 181h18v27H86z" fill="#f7fbfc" stroke="#9eb8c4" stroke-width="2"/>
<path d="M80 300v38M140 300v38" stroke="#34495e" stroke-width="16" stroke-linecap="round"/>
<path d="M72 340h24M128 340h24" stroke="#34495e" stroke-width="10" stroke-linecap="round"/>
<circle cx="97" cy="69" r="3" fill="#34495e"/><circle cx="123" cy="69" r="3" fill="#34495e"/>
<path d="M101 85q9 7 18 0" fill="none" stroke="#8d4e43" stroke-width="2" stroke-linecap="round"/>
</svg>"""
DOCTOR_HOSPITAL_IMAGE = Path(__file__).parent / "assets" / "doctor_hospital.svg"

HERO_STYLES = """
<style>
.stApp {
  background: linear-gradient(145deg, #f5fbfd 0%, #edf6f8 55%, #f7fafc 100%);
}
.main .block-container {
  max-width: 1180px;
  padding-top: 1.6rem;
  padding-bottom: 3rem;
}
.stSidebar {
  background: #f1f8fa;
}
.st-key-hero_panel {
  overflow: hidden;
  padding: 1rem 1.4rem;
  border: 1px solid #d8e9ee;
  border-radius: 26px;
  background: linear-gradient(112deg, #fff 0%, #effaff 52%, #dceff4 100%);
  box-shadow: 0 18px 48px rgba(37, 94, 112, .12);
}
.st-key-hero_panel [data-testid="stHorizontalBlock"] { align-items: center; }
.st-key-hero_panel h1 {
  color: #173d50;
  letter-spacing: -.04em;
  font-size: clamp(1.5rem, 3vw, 2.4rem);
  white-space: nowrap;
}
.st-key-hero_panel p { color: #56717e; }
.st-key-hero_panel [data-testid="stImage"] img { border-radius: 18px; }
.st-key-hero_panel [data-testid="stVerticalBlock"] { gap: .7rem; }
.st-key-hero_panel .stAlert { border-radius: 14px; }
.st-key-hero_panel .stCaption {
  color: #28726d;
}
@media (max-width: 600px) {
  .st-key-hero_panel [data-testid="stHorizontalBlock"] { flex-direction: column; }
}
</style>
"""


def _emergency_message(text, language="English"):
    """Return urgent-care guidance for explicit red-flag symptom phrases."""
    patterns = (
        r"\b(chest pain|chest pressure|crushing chest)\b",
        r"\b(shortness of breath|can't breathe|cannot breathe|difficulty breathing)\b",
        r"\b(face droop|one-sided weakness|sudden weakness|slurred speech)\b",
        r"\b(unconscious|not waking|seizure|severe bleeding)\b",
        r"\b(overdose|poisoning|suicide attempt|trying to kill myself)\b",
        r"(सीने में दर्द|साँस नहीं आ रही|सांस नहीं आ रही|बेहोश|दौरा|बहुत खून बहना)",
    )
    normalized = text.lower()
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            preceding_words = normalized[max(0, match.start() - 24):match.start()]
            if re.search(r"\b(no|not|without|denies|denied)\s+(?:any\s+)?$", preceding_words):
                continue
            following_words = normalized[match.end():match.end() + 18]
            if pattern.startswith("(सीने") and re.match(r"\s*नहीं", following_words):
                continue
            return LANGUAGES[language]["red_flag"]
    return None

def _gemini_user_content(text, image):
    if image is None:
        return text
    image_type = image.type or "image/jpeg"
    encoded_image = base64.b64encode(image.getvalue()).decode("ascii")
    return [
        {"type": "text", "text": text},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{image_type};base64,{encoded_image}"},
        },
    ]


def _assistant_reply(messages, profile, model=None, language="English", image=None):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    if image is not None and not gemini_api_key:
        raise RuntimeError("Photo review needs a Gemini API key. The photo was not sent.")
    if gemini_api_key:
        api_url, api_key = GEMINI_API_URL, gemini_api_key
        selected_model = model or GEMINI_MODEL
    elif nvidia_api_key:
        api_url, api_key = NVIDIA_API_URL, nvidia_api_key
        selected_model = model or NVIDIA_MODEL
    else:
        raise RuntimeError(
            "AI chat is not configured. Create a Gemini API key at "
            "https://aistudio.google.com/apikey, set GEMINI_API_KEY, and restart the app."
        )

    context = (
        "\nReply only in Hindi (हिन्दी), using clear, natural language."
        if language == "हिन्दी"
        else "\nReply only in English."
    )
    if profile:
        context += "\nUser-provided context (not verified): " + "; ".join(profile)
    request_messages = messages[-12:]
    if image is not None and request_messages:
        request_messages = list(request_messages)
        last_message = request_messages[-1]
        request_messages[-1] = {
            **last_message,
            "content": _gemini_user_content(last_message["content"], image),
        }
    client = OpenAI(base_url=api_url, api_key=api_key, timeout=45.0, max_retries=1)
    response_started = False
    try:
        stream = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + context},
                *request_messages,
            ],
            temperature=0.3,
            max_tokens=1000,
            top_p=0.95,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                response_started = True
                yield content
    except APIStatusError as exc:
        if (
            gemini_api_key
            and selected_model == GEMINI_MODEL
            and exc.status_code in {429, 500, 502, 503, 504}
            and not response_started
        ):
            yield from _assistant_reply(
                messages,
                profile,
                GEMINI_FALLBACK_MODEL,
                language,
                image,
            )
            return
        if response_started:
            logging.getLogger(__name__).warning(
                "AI response stream ended early (HTTP %s).", exc.status_code
            )
            yield "\n\nThe response was interrupted. Please try again."
            return
        raise

    if not response_started:
        raise RuntimeError(
            "The AI provider returned an empty reply. Please try again in a moment."
        )


def _profile_context(age, conditions, medications):
    profile = []
    if age:
        profile.append(f"age {age}")
    if conditions.strip():
        profile.append(f"conditions: {conditions.strip()}")
    if medications.strip():
        profile.append(f"medications: {medications.strip()}")
    return profile


def _symptom_prompt(body_area, symptoms, duration, severity, language="English"):
    return (
        "I want general guidance about these symptoms. "
        f"Body area: {body_area}. "
        f"What I feel: {symptoms.strip()}. "
        f"How long: {duration}. "
        f"Severity I selected: {severity}/10. "
        "Please explain a few possible causes without diagnosing me, suggest "
        "only low-risk general self-care if appropriate, identify warning signs "
        "that need urgent care, and tell me when to contact a clinician. "
        "Ask a short follow-up question if important details are missing. "
        + (
            "Reply only in Hindi (हिन्दी), using clear, natural language."
            if language == "हिन्दी"
            else "Reply only in English."
        )
    )


def _gemini_request(url, api_key, payload, timeout=45):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Gemini request failed ({exc.code}): {detail[:400]}"
        ) from exc


def _interaction_text(result):
    for output in result.get("outputs", []):
        if output.get("type") == "text" and output.get("text"):
            return output["text"].strip()
    for step in reversed(result.get("steps", [])):
        if step.get("type") == "model_output":
            text = "".join(
                part.get("text", "")
                for part in step.get("content", [])
                if part.get("type") == "text"
            ).strip()
            if text:
                return text
    raise RuntimeError("Gemini returned no text for this request.")


def _transcribe_with_gemini(audio_file, language="English"):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY to transcribe audio with Gemini.")

    audio_bytes = audio_file.getvalue()
    mime_type = audio_file.type or "audio/wav"
    upload_request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/upload/v1beta/files",
        data=json.dumps({"file": {"display_name": "DrHealth AI voice input"}}).encode("utf-8"),
        headers={
            "x-goog-api-key": api_key,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(audio_bytes)),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(upload_request, timeout=45) as response:
            upload_url = response.headers.get("x-goog-upload-url")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Gemini audio upload failed ({exc.code}): {detail[:400]}"
        ) from exc
    if not upload_url:
        raise RuntimeError("Gemini did not provide an audio upload URL.")

    finalize_request = urllib.request.Request(
        upload_url,
        data=audio_bytes,
        headers={
            "Content-Length": str(len(audio_bytes)),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
            "Content-Type": mime_type,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(finalize_request, timeout=45) as response:
            uploaded = json.loads(response.read().decode("utf-8")).get("file", {})
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Gemini could not finish uploading the recording ({exc.code}): {detail[:400]}"
        ) from exc

    file_name = uploaded.get("name")
    file_uri = uploaded.get("uri")
    uploaded_mime = uploaded.get("mimeType", mime_type)
    if not file_name or not file_uri:
        raise RuntimeError("Gemini upload completed without audio file metadata.")

    try:
        interaction_url = "https://generativelanguage.googleapis.com/v1beta/interactions"
        input_content = [
            {
                "type": "text",
                "text": GEMINI_TRANSCRIPTION_PROMPT
                + (
                    " The speaker is speaking Hindi; preserve the original Hindi words and script."
                    if language == "हिन्दी"
                    else " The speaker is speaking English."
                ),
            },
            {"type": "audio", "uri": file_uri, "mime_type": uploaded_mime},
        ]
        try:
            result = _gemini_request(
                interaction_url,
                api_key,
                {"model": GEMINI_MODEL, "input": input_content, "store": False},
                timeout=60,
            )
        except RuntimeError as exc:
            if "(503)" not in str(exc):
                raise
            result = _gemini_request(
                interaction_url,
                api_key,
                {
                    "model": GEMINI_FALLBACK_MODEL,
                    "input": input_content,
                    "store": False,
                },
                timeout=60,
            )
        return _interaction_text(result)
    finally:
        delete_request = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/{file_name}",
            headers={"x-goog-api-key": api_key},
            method="DELETE",
        )
        try:
            with urllib.request.urlopen(delete_request, timeout=15):
                pass
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            error_code = exc.code if isinstance(exc, urllib.error.HTTPError) else "network error"
            logging.getLogger(__name__).warning(
                "Could not delete temporary Gemini audio upload; provider expiry will apply (%s).",
                error_code,
            )


def _transcribe_audio(audio_file, language="English"):
    if os.getenv("GEMINI_API_KEY"):
        return _transcribe_with_gemini(audio_file, language)
    if os.getenv("NVIDIA_API_KEY"):
        from Nvidia_Speech import transcribe

        return transcribe(audio_file, language_code=LANGUAGES[language]["code"])
    raise RuntimeError(
        "Voice transcription needs GEMINI_API_KEY or an NVIDIA_API_KEY with "
        "Riva speech access. Set a Gemini key to use the same provider as chat."
    )


def _synthesize_with_gemini(text, language="English"):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY to generate voice playback with Gemini.")

    payload = {
        "model": GEMINI_TTS_MODEL,
        "input": [
            {
                "type": "user_input",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                        "annotations": [
                            {
                                "type": "speech_metadata",
                                "style": (
                                    "calm and friendly, speak naturally in Hindi"
                                    if language == "हिन्दी"
                                    else "calm and friendly, speak naturally in English"
                                ),
                            }
                        ],
                    }
                ],
            }
        ],
        "response_format": {"type": "audio"},
        "generation_config": {"speech_config": [{"voice": "Kore"}]},
    }
    request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/interactions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Gemini voice generation failed ({exc.code}): {detail[:400]}"
        ) from exc

    for output in reversed(result.get("outputs", [])):
        if output.get("type") == "audio" and output.get("data"):
            try:
                return base64.b64decode(output["data"], validate=True)
            except binascii.Error as exc:
                raise RuntimeError("Gemini returned invalid audio data.") from exc
    for step in reversed(result.get("steps", [])):
        for part in reversed(step.get("content", [])):
            if part.get("type") == "audio" and part.get("data"):
                try:
                    return base64.b64decode(part["data"], validate=True)
                except binascii.Error as exc:
                    raise RuntimeError("Gemini returned invalid audio data.") from exc
    raise RuntimeError("Gemini returned no audio for voice playback.")


def _speech_text(text):
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+", "", text)
    text = re.sub(r"[*_~`]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _synthesize_reply(text, language="English"):
    text = _speech_text(text)
    if os.getenv("GEMINI_API_KEY"):
        return _synthesize_with_gemini(text, language)
    if os.getenv("NVIDIA_API_KEY"):
        from Nvidia_Speech import synthesize

        return synthesize(text, language_code=LANGUAGES[language]["code"])
    raise RuntimeError(
        "Voice playback needs GEMINI_API_KEY or an NVIDIA_API_KEY with Riva access."
    )


st.set_page_config(page_title="DrHealth AI", page_icon="🩺", layout="wide")
st.html(HERO_STYLES)
with st.sidebar:
    language = st.selectbox("Language / भाषा", ("English", "हिन्दी"))
ui = LANGUAGES[language]["ui"]
with st.container(key="hero_panel", border=True):
    hero_copy, hero_art = st.columns([1, 1.15], vertical_alignment="center")
    with hero_copy:
        st.caption(ui["health_companion"])
        st.title("DrHealth AI")
        st.write(ui["hero_description"])
        st.caption(ui["not_diagnosis"])
    with hero_art:
        st.image(
            str(DOCTOR_HOSPITAL_IMAGE),
            width="stretch",
            alt="A dimensional doctor in a bright hospital room with medical windows and a cross.",
        )

with st.sidebar:
    st.subheader(ui["profile"])
    st.caption(ui["optional_context"])
    age = st.number_input(ui["age"], min_value=0, max_value=120, value=0)
    conditions = st.text_input(ui["conditions"])
    medications = st.text_input(ui["medications"])
    st.caption(ui["privacy"])
    if not os.getenv("GEMINI_API_KEY"):
        st.caption(ui["provider_info"])
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("NVIDIA_API_KEY"):
        st.caption(ui["voice_key_info"])
    voice_enabled = st.toggle(ui["voice_enabled"], value=False)
    voice_consent = False
    if voice_enabled:
        st.caption(ui["microphone_permission_help"])
        voice_provider = "Gemini" if os.getenv("GEMINI_API_KEY") else "NVIDIA"
        voice_consent = st.checkbox(
            ui["voice_consent"].format(provider=voice_provider),
            key="voice_provider_consent",
        )
    voice_enabled = voice_enabled and voice_consent

st.subheader(ui["help"])
st.caption(ui["help_caption"])
photo_version = st.session_state.get("photo_input_version", 0)
with st.expander(ui["photo_section"], expanded=False):
    st.caption(ui["photo_help"])
    captured_photo = st.camera_input(
        ui["photo_capture"],
        key=f"photo_capture_{photo_version}",
    )
    photo_question = st.text_input(
        ui["photo_question"],
        placeholder=ui["photo_question_placeholder"],
    )
    photo_consent = st.checkbox(
        ui["photo_consent"],
        key=f"photo_consent_{photo_version}",
    )
    scan_photo = st.button(
        ui["photo_scan"],
        icon=":material/document_scanner:",
        key=f"scan_photo_{photo_version}",
    )

if scan_photo:
    if captured_photo is None:
        st.error(ui["photo_missing"])
    elif not photo_consent:
        st.error(ui["photo_consent_required"])
    elif not os.getenv("GEMINI_API_KEY"):
        st.error(ui["photo_provider_required"])
    else:
        st.session_state.pending_prompt = (
            photo_question.strip() or ui["photo_default_question"]
        )
        st.session_state.pending_photo_scan = True
        st.session_state.photo_input_version = photo_version + 1

body_map, symptom_form = st.columns([1, 2])
with body_map:
    st.image(
        DOCTOR_ILLUSTRATION,
        width=220,
        alt="Illustration of a doctor wearing a white coat and stethoscope.",
    )
    st.caption(ui["body_caption"])

with symptom_form:
    with st.form("symptom_checker"):
        body_area = st.selectbox(
            ui["body_area"],
            LANGUAGES["English"]["ui"]["body_areas"],
            format_func=lambda value: (
                LANGUAGES["हिन्दी"]["ui"]["body_areas"][
                    LANGUAGES["English"]["ui"]["body_areas"].index(value)
                ]
                if language == "हिन्दी"
                else value
            ),
        )
        symptom_details = st.text_area(
            ui["symptoms"],
            placeholder=ui["symptoms_placeholder"],
            max_chars=1200,
        )
        duration = st.selectbox(
            ui["duration"],
            LANGUAGES["English"]["ui"]["durations"],
            format_func=lambda value: (
                LANGUAGES["हिन्दी"]["ui"]["durations"][
                    LANGUAGES["English"]["ui"]["durations"].index(value)
                ]
                if language == "हिन्दी"
                else value
            ),
        )
        severity = st.slider(ui["severity"], 0, 10, 3)
        assess_symptoms = st.form_submit_button(
            ui["guidance"],
            icon=":material/health_and_safety:",
        )

if assess_symptoms:
    if not symptom_details.strip():
        st.error(ui["empty_symptoms"])
    else:
        st.session_state.pending_prompt = _symptom_prompt(
            body_area,
            symptom_details,
            duration,
            severity,
            language,
        )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if voice_enabled:
    audio = st.audio_input(
        ui["record"],
        sample_rate=16000,
        key="symptom_audio",
    )
    if audio:
        st.audio(audio, format="audio/wav")
        audio_digest = hashlib.sha256(audio.getvalue()).hexdigest()
        if st.session_state.get("last_audio_digest") != audio_digest:
            try:
                prompt = _transcribe_audio(audio, language)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                st.session_state.voice_error = str(exc)
                prompt = ""
            else:
                st.session_state.voice_error = ""
            st.session_state.last_audio_digest = audio_digest
            if prompt:
                st.session_state.last_transcript_digest = audio_digest
                st.info(f"{ui['said']}: {prompt}")
                st.session_state.pending_prompt = prompt
        if st.session_state.get("last_transcript_digest") != audio_digest:
            if st.session_state.get("voice_error"):
                st.error(f"{ui['voice_error']}: {st.session_state.voice_error}")
            else:
                st.warning(ui["no_speech"])
            if st.button(ui["retry"], key="retry_transcription"):
                st.session_state.last_audio_digest = None
                st.rerun()

prompt = st.chat_input(ui["chat"])
if not prompt:
    prompt = st.session_state.pop("pending_prompt", None)
photo_for_prompt = (
    captured_photo
    if st.session_state.pop("pending_photo_scan", False)
    else None
)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    urgent_reply = _emergency_message(prompt, language)
    if urgent_reply:
        reply = urgent_reply
        with st.chat_message("assistant"):
            st.markdown(reply)
    else:
        try:
            with st.chat_message("assistant"):
                with st.spinner(ui["thinking"]):
                    reply = st.write_stream(
                        _assistant_reply(
                            st.session_state.messages,
                            _profile_context(age, conditions, medications),
                            language=language,
                            image=photo_for_prompt,
                        )
                    )
        except (RuntimeError, OpenAIError) as exc:
            error_label = "जवाब नहीं मिल सका" if language == "हिन्दी" else ui["reply_error"]
            st.error(f"{error_label}: {exc}")
            reply = None

    if reply:
        st.session_state.messages.append({"role": "assistant", "content": reply})
        if voice_enabled:
            try:
                audio_reply = _synthesize_reply(reply, language)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                st.session_state.voice_reply_pending = reply
                st.session_state.voice_reply_language = language
                st.error(f"{ui['voice_playback_error']}: {exc}")
            else:
                st.audio(audio_reply, format="audio/wav", autoplay=True)
                st.session_state.pop("voice_reply_pending", None)
                st.session_state.pop("voice_reply_language", None)

if st.session_state.get("voice_reply_pending"):
    if st.button(ui["retry_voice"], key="retry_voice_reply"):
        try:
            audio_reply = _synthesize_reply(
                st.session_state.voice_reply_pending,
                st.session_state.voice_reply_language,
            )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            st.error(f"{ui['voice_playback_error']}: {exc}")
        else:
            st.audio(audio_reply, format="audio/wav", autoplay=True)
            del st.session_state["voice_reply_pending"]
            del st.session_state["voice_reply_language"]
