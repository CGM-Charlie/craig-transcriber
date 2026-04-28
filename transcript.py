import sys
import os
import glob
import argparse
import platform
import numpy as np
import soundfile as sf

IS_APPLE_SILICON = platform.machine() == "arm64" and platform.system() == "Darwin"

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
    return name.split("-", 1)[1].rsplit(".", 1)[0]

def transcribe_track_mps(audio, model_name):
    from whisper_mps.whisper import transcribe
    return transcribe(audio, model=model_name)

def transcribe_track_default(audio, model, fp16):
    return model.transcribe(audio, fp16=fp16)

def main():
    parser = argparse.ArgumentParser(description="Transcribe Craig Discord recordings")
    parser.add_argument("craig_dir", help="Path to extracted Craig recording folder")
    parser.add_argument("output_dir", nargs="?", default=None, help="Output directory (default: transcripts/)")
    parser.add_argument("--model", "-m", default="base", help="Whisper model: tiny, base, small, medium, large (default: base)")
    args = parser.parse_args()

    craig_dir = os.path.abspath(args.craig_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcripts")
    model_name = args.model
    os.makedirs(output_dir, exist_ok=True)

    tracks = sorted(glob.glob(os.path.join(craig_dir, "[0-9]-*.flac")))
    if not tracks:
        sys.exit("No Craig audio tracks found in " + craig_dir)

    if IS_APPLE_SILICON:
        print(f"Loading '{model_name}' model on Apple Silicon (MPS via whisper-mps)...")
        transcribe_fn = lambda audio: transcribe_track_mps(audio, model_name)
    else:
        import torch
        import whisper
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fp16 = device == "cuda"
        print(f"Loading '{model_name}' model on {device}...")
        model = whisper.load_model(model_name, device=device)
        transcribe_fn = lambda audio: transcribe_track_default(audio, model, fp16)

    all_segments = []
    for t in tracks:
        speaker = speaker_from_filename(t)
        print(f"Transcribing {speaker}...")
        audio = load_track(t)
        result = transcribe_fn(audio)
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
