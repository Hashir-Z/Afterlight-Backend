import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torchaudio

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

# Add Matcha-TTS to path
COSYVOICE_DIR = Path(__file__).resolve().parent
sys.path.append(str(COSYVOICE_DIR / 'third_party' / 'Matcha-TTS'))
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav


def get_project_root() -> Path:
	# This file lives in <root>/CosyVoice/webserver.py
	return Path(__file__).resolve().parents[1]


ROOT_DIR = get_project_root()
# COSYVOICE_DIR already defined above
USER_AUDIO_DIR = COSYVOICE_DIR / "UserAudio"
GENERATED_DIR = COSYVOICE_DIR / "GeneratedAudio"
ROOT_GENERATED_DIR = ROOT_DIR / "GeneratedAudio"  # in case the script writes here

# Ensure directories exist
USER_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
ROOT_GENERATED_DIR.mkdir(parents=True, exist_ok=True)

# Global model instance (will be initialized at startup)
cosyvoice: Optional[CosyVoice2] = None

app = FastAPI(title="CosyVoice WebServer", version="1.0.0")

# Allow LAN usage from any origin by default
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
	"""Initialize CosyVoice model at server startup."""
	global cosyvoice
	print("Loading CosyVoice2 model...")
	try:
		model_path = COSYVOICE_DIR / "pretrained_models" / "CosyVoice2-0.5B"
		cosyvoice = CosyVoice2(
			str(model_path),
			load_jit=False,
			load_trt=False,
			load_vllm=False,
			fp16=False
		)
		print("CosyVoice2 model loaded successfully!")
	except Exception as e:
		print(f"Failed to load CosyVoice2 model: {e}")
		raise


def _find_python_executable() -> str:
	# Prefer current interpreter
	return sys.executable or "python"


def _resolve_run_script_path() -> Path:
	# Try both locations: root and inside CosyVoice (user-specified path)
	candidates = [
		ROOT_DIR / "run_cozyvoice.py",
		ROOT_DIR / "CosyVoice" / "run_cozyvoice.py",
	]
	for c in candidates:
		if c.exists():
			return c
	raise FileNotFoundError("run_cozyvoice.py not found in expected locations.")


def _candidate_ffmpeg_paths() -> list[Path]:
	env_candidate = os.environ.get("FFMPEG_BINARY") or os.environ.get("FFMPEG_PATH")
	candidates: list[Path] = []
	if env_candidate:
		env_path = Path(env_candidate)
		if env_path.is_dir():
			# If a directory is provided, try typical filenames inside it
			candidates.append(env_path / "ffmpeg.exe")
			candidates.append(env_path / "ffmpeg")
			candidates.append(env_path / "bin" / "ffmpeg.exe")
			candidates.append(env_path / "bin" / "ffmpeg")
		else:
			candidates.append(env_path)

	# Common local placements (drop prebuilt ffmpeg here to avoid touching PATH)
	potential_dirs = [
		ROOT_DIR,
		ROOT_DIR / "CosyVoice",
		ROOT_DIR / "ffmpeg",
		ROOT_DIR / "ffmpeg" / "bin",
		ROOT_DIR / "bin",
		Path(sys.executable).parent,  # Python env Scripts/ or bin/
	]

	for base in potential_dirs:
		candidates.append(base / "ffmpeg.exe")
		candidates.append(base / "ffmpeg")

	# Only keep files, not directories
	return [p for p in candidates if p.is_file()]


def _resolve_ffmpeg_executable() -> str:
	for candidate in _candidate_ffmpeg_paths():
		try:
			if candidate and candidate.is_file():
				return str(candidate)
		except Exception:
			continue

	# Fall back to PATH lookup
	from shutil import which

	path_hit = which("ffmpeg")
	if path_hit:
		return path_hit

	raise FileNotFoundError(
		"ffmpeg executable not found. Place ffmpeg/ffmpeg.exe inside the project (e.g. R:\\LLM\\ffmpeg\\ffmpeg.exe) "
		"or set the FFMPEG_BINARY environment variable to point to it."
	)


def _save_upload_to_useraudio(upload: UploadFile, dest_path: Path) -> None:
	# Ensure read cursor at start; avoid partial writes
	try:
		upload.file.seek(0)
	except Exception:
		pass
	with dest_path.open("wb") as f:
		shutil.copyfileobj(upload.file, f)
	# Close to release file handle on Windows before ffmpeg touches it
	try:
		upload.file.close()
	except Exception:
		pass


def _extract_16bit_wav(input_path: Path, output_path: Path) -> None:
	ffmpeg_exec = _resolve_ffmpeg_executable()
	# Convert to 16-bit PCM WAV, 48kHz, preserve channel count (mono stays mono, stereo stays stereo)
	cmd = [
		ffmpeg_exec, "-y",
		"-i", str(input_path),
		"-ar", "48000",
		"-acodec", "pcm_s16le",
		str(output_path),
	]
	proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	if proc.returncode != 0:
		raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")


def _create_wav_header(sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
	"""Create a WAV file header with placeholder file size (0xFFFFFFFF for streaming)."""
	# WAV header structure
	# RIFF header
	chunk_id = b'RIFF'
	chunk_size = 0xFFFFFFFF  # Placeholder for streaming (unknown size)
	format = b'WAVE'
	
	# fmt subchunk
	subchunk1_id = b'fmt '
	subchunk1_size = 16  # PCM format
	audio_format = 1  # PCM
	byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
	block_align = num_channels * (bits_per_sample // 8)
	
	# data subchunk
	subchunk2_id = b'data'
	subchunk2_size = 0xFFFFFFFF  # Placeholder for streaming (unknown size)
	
	# Pack header
	header = struct.pack('<4sI4s4sIHHIIHH4sI',
		chunk_id, chunk_size, format,
		subchunk1_id, subchunk1_size, audio_format, num_channels, sample_rate,
		byte_rate, block_align, bits_per_sample,
		subchunk2_id, subchunk2_size)
	
	return header


def _generate_audio_stream(audio_name_no_ext: str, transcript: str, desired_tts: str):
	"""Generator function that yields audio chunks as bytes for streaming."""
	global cosyvoice
	
	if cosyvoice is None:
		raise RuntimeError("CosyVoice model not initialized. Please restart the server.")
	
	# Load reference audio (16kHz as required by CosyVoice)
	filename = USER_AUDIO_DIR / f"{audio_name_no_ext}.wav"
	if not filename.exists():
		raise FileNotFoundError(f"Reference audio file not found: {filename}")
	
	prompt_speech_16k = load_wav(str(filename), 16000)
	
	# Get sample rate from model
	sample_rate = cosyvoice.sample_rate
	
	# Yield WAV header first
	yield _create_wav_header(sample_rate, num_channels=1, bits_per_sample=16)
	
	# Run zero-shot inference with streaming enabled
	for i, j in enumerate(cosyvoice.inference_zero_shot(
		desired_tts,
		transcript,
		prompt_speech_16k,
		stream=True)):  # Enable streaming
		# Convert audio tensor to int16 bytes
		# tts_speech shape is typically (channels, samples), flatten to 1D for WAV
		audio_np = j['tts_speech'].numpy()
		tts_audio = (audio_np * (2 ** 15)).astype(np.int16).tobytes()
		yield tts_audio


@app.post("/generate")
async def generate(
	transcript: str = Form(...),
	desired_tts: str = Form(...),
	file_hash: str = Form(...),
	file: Optional[UploadFile] = File(default=None),
):
	if not transcript or not desired_tts or not file_hash:
		raise HTTPException(status_code=400, detail="transcript, desired_tts, and file_hash are required.")
	
	audio_name_no_ext = file_hash
	
	# Check if file was actually provided (not empty blob)
	file_provided = False
	file_size = 0
	if file is not None:
		try:
			filename = getattr(file, 'filename', None)
			if filename:
				# Check file size to see if it's a real file or empty blob
				file.file.seek(0, 2)  # Seek to end
				file_size = file.file.tell()
				file.file.seek(0)  # Reset to beginning
				if file_size > 0:
					file_provided = True
		except Exception:
			pass
	
	# Check if original file already exists for this hash
	
	converted_wav_path = USER_AUDIO_DIR / f"{audio_name_no_ext}.wav"
	
	if converted_wav_path.exists():
		# File already exists - reuse existing converted WAV
		print(f"Reusing existing file for hash: {audio_name_no_ext}")
	elif file_provided:
		# New file upload - process it
		print(f"Processing new file upload for hash: {audio_name_no_ext}")
		original_ext = Path(file.filename or "").suffix or ""
		
		# Save original upload to UserAudio/<hash>_orig<ext>
		original_save_path = USER_AUDIO_DIR / f"{audio_name_no_ext}_orig{original_ext}"
		_save_upload_to_useraudio(file, original_save_path)
		
		# Extract/convert to 16-bit WAV: UserAudio/<hash>.wav
		try:
			_extract_16bit_wav(original_save_path, converted_wav_path)
		except Exception as e:
			raise HTTPException(status_code=500, detail=f"Audio extraction failed: {e}")
	else:
		# No file provided and no existing file found
		raise HTTPException(status_code=400, detail=f"File not found for hash {file_hash}. Please upload the file.")

	# Stream audio chunks directly to the client
	try:
		# Return streaming response with audio chunks
		return StreamingResponse(
			_generate_audio_stream(
				audio_name_no_ext=audio_name_no_ext,
				transcript=transcript,
				desired_tts=desired_tts
			),
			media_type="audio/wav",
			headers={
				"Content-Type": "audio/wav",
				"Cache-Control": "no-cache",
				"Connection": "keep-alive",
			}
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@app.get("/generated/{filename}")
async def get_generated(filename: str):
	target = GENERATED_DIR / filename
	if not target.exists() or not target.is_file():
		raise HTTPException(status_code=404, detail="File not found.")
	media_type = "audio/wav" if target.suffix.lower() == ".wav" else "application/octet-stream"
	return FileResponse(path=str(target), media_type=media_type, filename=target.name)


@app.delete("/cleanup/{file_hash}")
async def cleanup(file_hash: str):
	"""Delete only the original uploaded file (not converted/generated WAV files)."""
	try:
		# Sanitize file_hash for filename matching
		safe_hash = file_hash.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
		
		# Find and delete only the original uploaded file
		for orig_file in USER_AUDIO_DIR.glob(f"{safe_hash}_orig.*"):
			try:
				orig_file.unlink()
				print(f"Deleted original file: {orig_file}")
			except Exception as e:
				print(f"Failed to delete {orig_file}: {e}")
		
		# Note: We do NOT delete the converted WAV file or generated audio
		# as they may be needed for future requests with the same file
		
		return {"status": "success", "message": f"Cleaned up original file for hash {file_hash}"}
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@app.get("/health")
async def health():
	return {"status": "ok"}


def _default_host() -> str:
	return os.environ.get("HOST", "0.0.0.0")


def _default_port() -> int:
	try:
		return int(os.environ.get("PORT", "8000"))
	except Exception:
		return 8000


if __name__ == "__main__":
	# Run with: python CosyVoice/webserver.py
	import uvicorn
	# When executed from within the CosyVoice directory, use local module path
	# e.g., `python webserver.py`
	try:
		uvicorn.run("webserver:app", host=_default_host(), port=_default_port(), reload=False)
	except Exception:
		# Fallback for running from repo root with `python CosyVoice/webserver.py`
		uvicorn.run("CosyVoice.webserver:app", host=_default_host(), port=_default_port(), reload=False)


