# Craig Transcriber

Transcribes [Craig](https://craig.chat/) Discord bot recordings into speaker-attributed text using OpenAI's [Whisper](https://github.com/openai/whisper).

Craig records each participant as a separate audio track. This tool transcribes each track individually, tags segments with the speaker's Discord username, and merges them chronologically into a single timestamped transcript.

## Requirements

- Python 3.8+
- [openai-whisper](https://github.com/openai/whisper): `pip install openai-whisper`
- soundfile: `pip install soundfile`

## Usage

1. Download your Craig recording (the self-processing zip) and extract it.
2. Run the script pointing to the extracted folder:

```bash
# Default: saves to transcripts/ in this repo
python3 transcript.py /path/to/craig-recording

# Custom output directory
python3 transcript.py /path/to/craig-recording /path/to/output

# Custom output directory + Whisper model (tiny, base, small, medium, large)
python3 transcript.py /path/to/craig-recording /path/to/output small
```

## Output

The transcript is saved as a text file with timestamped, speaker-labeled lines:

```
[00:00] alice: Hey everyone, let's get started
[00:03] bob: Sounds good, I have some updates
[00:08] charlie: Same here
```

## Summarizing

A prompt template for summarizing transcripts with an LLM is available in `prompts/prompt_v1.txt`. Paste your transcript after the prompt to get a topic-based summary.
