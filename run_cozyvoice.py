import sys
import argparse
import os
import torchaudio

# Add Matcha-TTS to path
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# Argument parser setup
parser = argparse.ArgumentParser(description="CosyVoice2 zero-shot TTS generation")
parser.add_argument('--audio_name', type=str, required=True, help='Path to reference audio file (e.g., audio.wav)')
parser.add_argument('--transcript', type=str, required=True, help='Text spoken in the reference audio')
parser.add_argument('--desired_tts', type=str, required=True, help='Text to synthesize using the reference voice')
args = parser.parse_args()

# Load reference audio
filename = os.path.join("UserAudio", args.audio_name + ".wav")
prompt_speech_16k = load_wav(filename, 16000)

# # Initialize CosyVoice2
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B',
                       load_jit=False, load_trt=True, load_vllm=False, fp16=True)

# # Run zero-shot inference
# for i, j in enumerate(cosyvoice.inference_sft(
#     args.desired_tts,
#     args.transcript,
#     prompt_speech_16k,
#     stream=False)):
#     savepath  = os.path.join("GeneratedAudio", args.audio_name + ".wav")
#     torchaudio.save(savepath, j['tts_speech'], cosyvoice.sample_rate)

# Run zero-shot inference
for i, j in enumerate(cosyvoice.inference_zero_shot(
    args.desired_tts,
    args.transcript,
    prompt_speech_16k,
    stream=False)):
    savepath  = os.path.join("GeneratedAudio", args.audio_name + ".wav")
    torchaudio.save(savepath, j['tts_speech'], cosyvoice.sample_rate)

# Accent control usage
# Say this sentence in a Southern American accent.
# for i, j in enumerate(cosyvoice.inference_instruct2(
#     args.target_text,
#     args.prompt_text,
#     prompt_speech_16k,
#     stream=False)):
#     torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
