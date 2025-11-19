CosyVoice WebServer
===================

Start the server
----------------
1) Install dependencies (in your Python environment):
   - `pip install -r requirements.txt`
   - ffmpeg is required, but does **not** need to be on PATH. Place the executable at one of:
     - `R:\LLM\ffmpeg\ffmpeg.exe`
     - `R:\LLM\CosyVoice\ffmpeg.exe`
     - `R:\LLM\ffmpeg\bin\ffmpeg.exe`
     - or set the environment variable `FFMPEG_BINARY` (or `FFMPEG_PATH`) to the full executable path.

2) Run the server:
   - `python CosyVoice/webserver.py`
   - It listens on http://0.0.0.0:8000 by default.

Endpoints
---------
- `POST /generate` (multipart form):
  - `file`: uploaded audio/video
  - `transcript`: what was said in the uploaded media
  - `desired_tts`: the message you want to synthesize
  - Returns: `{"audio_url": "/generated/<uuid>.wav", "audio_name": "<uuid>.wav"}`

- `GET /generated/{filename}`:
  - Serves audio files from `GeneratedAudio`.

Data flow
---------
1) Upload saved to `UserAudio/<uuid>_orig.ext`
2) Audio extracted/converted to 16-bit 16kHz mono WAV at `UserAudio/<uuid>.wav`
3) Invokes `run_cozyvoice.py` with:
   - `--audio_name=<uuid>`
   - `--transcript=<transcript>`
   - `--desired_tts=<desired_tts>`
4) Expects `GeneratedAudio/<uuid>.wav` to be produced, which is then served back.

Notes
-----
- The server attempts to locate `run_cozyvoice.py` either in repository root or in `CosyVoice/`.
- Ensure `UserAudio/` and `GeneratedAudio/` exist (the server will create them if missing).
- Open firewall for TCP 8000 if accessing over LAN.


