"""NVIDIA Riva speech (hosted on build.nvidia.com): speech-to-text and text-to-speech."""
import io
import os
import re
import wave

import grpc

SERVER = "grpc.nvcf.nvidia.com:443"
ASR_FUNCTION_ID = "1598d209-5e27-4d3c-8079-4751568b1081"
TTS_FUNCTION_ID = "877104f7-e885-42b9-8de8-f6e4c6303969"
VOICES = {
    "en-US": "Magpie-Multilingual.EN-US.Aria",
    "hi-IN": "Magpie-Multilingual.HI-IN.Aria",
}
TTS_RATE = 22050


def _auth(function_id):
    import riva.client

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Voice needs NVIDIA_API_KEY. Set it in the environment before starting the app."
        )

    return riva.client.Auth(
        None,
        True,
        SERVER,
        [
            ["function-id", function_id],
            ["authorization", f"Bearer {api_key}"],
        ],
    )


def _speech_error(exc):
    status = exc.code()
    if status == grpc.StatusCode.UNAUTHENTICATED:
        return RuntimeError(
            "NVIDIA Riva rejected speech authentication. Check that NVIDIA_API_KEY "
            "is valid and that your account/key has access to the configured Riva "
            "speech function. Access to NVIDIA's chat API does not necessarily "
            "include Riva speech access."
        )
    if status == grpc.StatusCode.PERMISSION_DENIED:
        return RuntimeError(
            "NVIDIA Riva denied access to this speech function. Check your account's "
            "Riva function access and the configured function ID."
        )
    status_name = status.name if status else "UNKNOWN"
    return RuntimeError(f"NVIDIA Riva speech request failed ({status_name}).")


def transcribe(audio_file, language_code="en-US"):
    """Recorded WAV (from st.audio_input) -> text, using NVIDIA Parakeet."""
    import riva.client

    with wave.open(io.BytesIO(audio_file.getvalue())) as w:
        rate, channels, pcm = w.getframerate(), w.getnchannels(), w.readframes(w.getnframes())
    config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=rate,
        audio_channel_count=channels,
        language_code=language_code,
        max_alternatives=1,
        enable_automatic_punctuation=True,
    )
    asr = riva.client.ASRService(_auth(ASR_FUNCTION_ID))
    try:
        resp = asr.offline_recognize(pcm, config)
    except grpc.RpcError as exc:
        raise _speech_error(exc) from exc
    return " ".join(r.alternatives[0].transcript for r in resp.results if r.alternatives).strip()


def _chunks(text, limit=300):
    """Split long replies into sentence groups the TTS model handles well."""
    out, cur = [], ""
    for s in re.split(r"(?<=[.!?])\s+", text.strip()):
        if cur and len(cur) + len(s) > limit:
            out.append(cur)
            cur = ""
        cur += (" " if cur else "") + s
    if cur:
        out.append(cur)
    return out


def pcm_to_wav(pcm, rate=TTS_RATE):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def synthesize(text, language_code="en-US"):
    """Text -> WAV bytes, using NVIDIA Magpie TTS."""
    import riva.client

    tts = riva.client.SpeechSynthesisService(_auth(TTS_FUNCTION_ID))
    pcm = b""
    try:
        for piece in _chunks(text):
            for response in tts.synthesize_online(
                piece,
                VOICES.get(language_code, VOICES["en-US"]),
                language_code=language_code,
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hz=TTS_RATE,
            ):
                pcm += response.audio
    except grpc.RpcError as exc:
        raise _speech_error(exc) from exc
    return pcm_to_wav(pcm)
