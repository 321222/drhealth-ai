# DrHealth AI

A Streamlit health-information chat app with optional NVIDIA speech-to-text and
text-to-speech. It can discuss symptoms and health questions, but it does not
diagnose disease or predict an individual's health.

## Run locally

1. Install Python 3.10 or later.
2. From this folder, install dependencies:

   ```powershell
   py -3.12 -m pip install -r requirements.txt
   ```

3. Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/apikey),
   then start the app with its secure key prompt:

   ```powershell
   .\run.ps1
   ```

   The launcher reads the key without displaying it and saves it outside the
   project in your Windows account's `.streamlit\secrets.toml`, with access
   restricted to your account. It reuses the saved key on later launches. If
   `GEMINI_API_KEY` is already present in the environment, the launcher saves it
   there automatically the first time.

For Streamlit Community Cloud, deploy `Chatbot.py` from the project repository
and add `GEMINI_API_KEY` in the app's **Settings → Secrets**. Do not commit API
keys or a `secrets.toml` file.

Chat defaults to Gemini 3.8 Flash through Google's OpenAI-compatible API. With a
Gemini key, voice input and voice replies use Gemini audio models and do not need
an NVIDIA Riva key. If `GEMINI_API_KEY` is not set, the app can instead use
NVIDIA Nemotron and NVIDIA Riva when `NVIDIA_API_KEY` has access to the configured
speech functions. Free-tier model availability and usage are subject to account,
region, and rate limits.
If Gemini is temporarily overloaded, chat and transcription retry with
`gemini-3.1-flash-lite`.

The interface uses a hospital-themed background, a dimensional doctor illustration,
and a symptom checker with a body-area selector. The optional camera scan also
accepts uploaded JPG, PNG, or WebP images; images are sent to Gemini only after
the user consents and selects Scan. Choose English or Hindi from the sidebar; chat
replies, symptom guidance, voice transcription, and spoken replies follow the
selected language.

For voice chat, enable voice, record a question, and stop the recording to send it.
The assistant transcribes the recording and automatically attempts to play its answer
aloud; the audio player remains available if the browser blocks autoplay. This is a
record-and-reply flow, not continuous hands-free streaming. Voice requires a working
internet connection, a configured provider key, and provider availability; temporary
service, microphone, or network failures can still occur. Transcription and spoken
reply each offer a retry when available. Gemini spoken replies use an original,
low-pitched, measured AI-assistant style; press Play in the audio player if the
browser blocks automatic playback. When voice is enabled, Chrome's built-in speech
engine also provides a **Speak reply** control that does not require a separate TTS
service or microphone permission.

The app returns general health information and possible self-care suggestions, not
a diagnosis or a prescription. Seek professional care for concerning or worsening
symptoms.

Do not enter names, identifying information, or sensitive medical records.
Chat messages and any optional health-profile details are sent to the selected
AI provider. Free-tier prompts may be used by the provider to improve services,
depending on its terms. This
prototype is not a medical device, does not replace a clinician, and must not
be used for emergencies. For urgent symptoms, contact local emergency services.
