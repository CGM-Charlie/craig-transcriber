import sys
import os
import glob
import numpy as np
import soundfile as sf
import platform
import torch
import whisper

if platform.processor() == "arm" or platform.machine() == "arm64":
    DEVICE = "cpu"
    FP16 = False
elif torch.cuda.is_available():
    DEVICE = "cuda"
    FP16 = True
else:
    DEVICE = "cpu"
    FP16 = False

def load_track(path):
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    target_sr = 16000
    if sr != target_sr:
        indices = np.round(np.linspace(0, len(data) - 1, int(len(data) * target_sr / sr))).astype(int)
        data = data[indices]
    return data

def speaker_from_filename(path):
    name = os.path.basename(path)
    # "1-cgm2qp.flac" -> "cgm2qp"
    return name.split("-", 1)[1].rsplit(".", 1)[0]

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 transcript.py <craig_dir> [output_dir] [model]")
        sys.exit(1)

    craig_dir = os.path.abspath(sys.argv[1])
    default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcripts")
    output_dir = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else default_out
    model_name = sys.argv[3] if len(sys.argv) > 3 else "base"
    os.makedirs(output_dir, exist_ok=True)

    tracks = sorted(glob.glob(os.path.join(craig_dir, "[0-9]-*.flac")))
    if not tracks:
        sys.exit("No Craig audio tracks found in " + craig_dir)

    print(f"Loading '{model_name}' model on {DEVICE}...")
    model = whisper.load_model(model_name, device=DEVICE)

    all_segments = []
    for t in tracks:
        speaker = speaker_from_filename(t)
        print(f"Transcribing {speaker}...")
        audio = load_track(t)
        result = model.transcribe(audio, fp16=FP16)
        for seg in result["segments"]:
            text = seg["text"].strip()
            if text:
                all_segments.append((seg["start"], seg["end"], speaker, text))

    all_segments.sort(key=lambda s: s[0])

    lines = []
    for start, end, speaker, text in all_segments:
        ts = f"[{int(start)//60:02d}:{int(start)%60:02d}]"
        lines.append(f"{ts} {speaker}: {text}")

    session = os.path.basename(craig_dir)
    transcript_path = os.path.join(output_dir, f"{session}_transcript.txt")
    with open(transcript_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Transcript saved to: {transcript_path}")

if __name__ == "__main__":
    main()
