# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import logging
import tempfile
import subprocess
from pathlib import Path
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


def convert_to_wav(input_file_path, output_file_path=None, sample_rate=16000, channels=1):
    """
    Convert any audio/video file to WAV format using ffmpeg.
    
    Supports: mp4, mov, mp3, mpg, mpeg, wmv, mkv, avi, wav, flac, ogg, and many more.
    
    Args:
        input_file_path: Path to input file
        output_file_path: Path to output WAV file (optional, auto-generated if None)
        sample_rate: Target sample rate (default: 16000 for CosyVoice)
        channels: Number of channels (1=mono, 2=stereo)
    
    Returns:
        Path to output WAV file
    """
    if output_file_path is None:
        base_name = os.path.splitext(input_file_path)[0]
        output_file_path = f"{base_name}.wav"
    
    # Build ffmpeg command
    cmd = [
        'ffmpeg',
        '-i', input_file_path,           # Input file
        '-acodec', 'pcm_s16le',         # 16-bit PCM codec
        '-ac', str(channels),            # Audio channels
        '-ar', str(sample_rate),         # Sample rate
        '-y',                            # Overwrite output
        output_file_path                 # Output file
    ]
    
    try:
        # Run ffmpeg (suppress output)
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return output_file_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Error converting {input_file_path}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to convert audio file: {e}")
    except FileNotFoundError:
        raise HTTPException(
            status_code=500, 
            detail="ffmpeg not found. Please install ffmpeg to support audio/video conversion."
        )


async def process_audio_file(upload_file: UploadFile, target_sr=16000):
    """
    Process uploaded audio/video file and convert to WAV if needed.
    
    Args:
        upload_file: FastAPI UploadFile object
        target_sr: Target sample rate
    
    Returns:
        torch.Tensor: Audio tensor ready for CosyVoice processing
    """
    # Get file extension
    filename = upload_file.filename or "audio"
    file_ext = Path(filename).suffix.lower()
    
    # Supported formats that don't need conversion
    wav_formats = {'.wav', '.flac', '.ogg', '.aiff'}
    
    # Create temporary files
    temp_input_path = None
    wav_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_input:
            # Save uploaded file to temp location
            content = await upload_file.read()
            temp_input.write(content)
            temp_input_path = temp_input.name
        
        # Convert to WAV if needed
        if file_ext not in wav_formats:
            logging.info(f"Converting {file_ext} file to WAV format")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
                temp_wav_path = temp_wav.name
            convert_to_wav(temp_input_path, temp_wav_path, sample_rate=target_sr, channels=1)
            wav_path = temp_wav_path
        else:
            # Already a supported format, use directly
            wav_path = temp_input_path
        
        # Load audio using existing load_wav function
        audio_tensor = load_wav(wav_path, target_sr)
        
        return audio_tensor
    
    finally:
        # Clean up temporary files
        try:
            if temp_input_path and os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
            if wav_path and file_ext not in wav_formats and wav_path != temp_input_path and os.path.exists(wav_path):
                os.unlink(wav_path)
        except Exception as e:
            logging.warning(f"Failed to clean up temp files: {e}")


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = await process_audio_file(prompt_wav, target_sr=16000)
    model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = await process_audio_file(prompt_wav, target_sr=16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_speech_16k = await process_audio_file(prompt_wav, target_sr=16000)
    model_output = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='iic/CosyVoice-300M',
                        help='local path or modelscope repo id')
    parser.add_argument('--load_trt',
                        action='store_true',
                        default=False,
                        help='Enable TensorRT acceleration for flow decoder estimator')
    parser.add_argument('--load_jit',
                        action='store_true',
                        default=False,
                        help='Enable JIT optimization for text encoder and flow encoder')
    parser.add_argument('--fp16',
                        action='store_true',
                        default=False,
                        help='Use FP16 precision (requires CUDA)')
    parser.add_argument('--trt_concurrent',
                        type=int,
                        default=4,
                        help='Number of concurrent TensorRT inference contexts (default: 4)')
    parser.add_argument('--load_vllm',
                        action='store_true',
                        default=False,
                        help='Enable vLLM for CosyVoice2 (requires TensorRT-LLM)')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    
    try:
        if args.load_vllm:
            # CosyVoice2 with vLLM
            cosyvoice = CosyVoice2(args.model_dir, 
                                 load_jit=args.load_jit, 
                                 load_trt=args.load_trt, 
                                 load_vllm=args.load_vllm,
                                 fp16=args.fp16, 
                                 trt_concurrent=args.trt_concurrent)
        else:
            # Try CosyVoice first
            cosyvoice = CosyVoice(args.model_dir,
                                load_jit=args.load_jit,
                                load_trt=args.load_trt,
                                fp16=args.fp16,
                                trt_concurrent=args.trt_concurrent)
    except Exception as e:
        try:
            # Fallback to CosyVoice2
            logging.info(f"Failed to load as CosyVoice, trying CosyVoice2: {e}")
            cosyvoice = CosyVoice2(args.model_dir,
                                load_jit=args.load_jit,
                                load_trt=args.load_trt,
                                load_vllm=args.load_vllm,
                                fp16=args.fp16,
                                trt_concurrent=args.trt_concurrent)
        except Exception as e2:
            raise TypeError(f'no valid model_type! Error: {e2}')
    
    if args.load_trt:
        logging.info(f"TensorRT acceleration enabled with {args.trt_concurrent} concurrent contexts")
    if args.fp16:
        logging.info("FP16 precision enabled")
    if args.load_jit:
        logging.info("JIT optimization enabled")
    
    logging.info(f"FastAPI server starting on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
