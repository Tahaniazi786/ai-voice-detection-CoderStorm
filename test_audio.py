from audio_utils import decode_base64_audio, extract_features
import base64

# Put any small mp3 file in project folder
with open("sample.mp3", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()

audio, sr = decode_base64_audio(audio_base64)

if audio is None:
    print("Audio decode FAILED")
else:
    features = extract_features(audio, sr)
    print("Audio decode OK")
    print("Feature shape:", features.shape)
