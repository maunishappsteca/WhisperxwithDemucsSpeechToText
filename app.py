import os
import uuid
import subprocess
import whisperx
import runpod
import boto3
import gc
import json
from typing import Optional
from botocore.exceptions import ClientError
import logging
from datetime import datetime
import shutil
import tempfile

# --- Configuration ---
COMPUTE_TYPE = "float16"
BATCH_SIZE = 16
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
MODEL_CACHE_DIR = os.getenv("WHISPER_MODEL_CACHE", "/app/models")

# Demucs config
DEMUCS_MODEL = os.getenv("DEMUCS_MODEL", "htdemucs")
VOCAL_GAIN_DB_DEFAULT = float(os.getenv("VOCAL_GAIN_DB", "6"))
DEMUCS_CACHE_DIR = os.getenv("DEMUCS_CACHE_DIR", os.path.join(MODEL_CACHE_DIR, "demucs"))
os.environ["DEMUCS_CACHE"] = DEMUCS_CACHE_DIR

# Logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# S3 client
s3 = boto3.client('s3') if S3_BUCKET else None

def ensure_model_cache_dir():
    try:
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        test_file = os.path.join(MODEL_CACHE_DIR, "test.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)

        os.makedirs(DEMUCS_CACHE_DIR, exist_ok=True)
        test_file2 = os.path.join(DEMUCS_CACHE_DIR, "test.tmp")
        with open(test_file2, "w") as f:
            f.write("test")
        os.remove(test_file2)

        return True
    except Exception as e:
        logger.error(f"Model cache directory error: {str(e)}")
        return False

def convert_to_wav(input_path: str) -> str:
    try:
        output_path = f"/tmp/{uuid.uuid4()}.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le",
            "-loglevel", "error",
            output_path
        ], check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {str(e)}")
        raise RuntimeError(f"FFmpeg conversion failed: {str(e)}")
    except Exception as e:
        logger.error(f"Audio conversion error: {str(e)}")
        raise RuntimeError(f"Audio conversion error: {str(e)}")

def separate_vocals_with_demucs(input_wav_path: str, model_name: str) -> str:
    outdir = f"/tmp/demucs_{uuid.uuid4()}"
    os.makedirs(outdir, exist_ok=True)
    logger.info(f"Running Demucs '{model_name}' to isolate vocals...")
    try:
        subprocess.run(
            [
                "python", "-m", "demucs.separate",
                "-n", model_name,
                "--two-stems", "vocals",
                "-o", outdir,
                input_wav_path
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        vocals_path = None
        for root, _, files in os.walk(outdir):
            if "vocals.wav" in files:
                vocals_path = os.path.join(root, "vocals.wav")
                break
        if not vocals_path:
            raise RuntimeError("Demucs completed but vocals.wav not found.")
        return vocals_path, outdir
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="ignore")
        logger.error(f"Demucs failed: {err[:5000]}")
        shutil.rmtree(outdir, ignore_errors=True)
        raise RuntimeError("Demucs separation failed.")
    except Exception as e:
        shutil.rmtree(outdir, ignore_errors=True)
        logger.error(f"Demucs error: {str(e)}")
        raise

def boost_and_resample(input_path: str, gain_db: float) -> str:
    boosted_path = f"/tmp/{uuid.uuid4()}_vocals_boosted.wav"
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", input_path,
            "-vn",
            "-filter:a", f"volume={gain_db}dB",
            "-ac", "1",
            "-ar", "16000",
            "-acodec", "pcm_s16le",
            "-loglevel", "error",
            boosted_path
        ], check=True)
        return boosted_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg boost failed: {str(e)}")
        raise RuntimeError(f"FFmpeg boost failed: {str(e)}")

def prepare_audio_for_transcription(base_wav_path: str, use_vocal_isolation: bool,
                                    vocal_gain_db: float, demucs_model: str) -> str:
    if not use_vocal_isolation:
        return base_wav_path

    demucs_tmp = None
    try:
        vocals_path, demucs_tmp = separate_vocals_with_demucs(base_wav_path, demucs_model)
        boosted_path = boost_and_resample(vocals_path, vocal_gain_db)
        if demucs_tmp:
            shutil.rmtree(demucs_tmp, ignore_errors=True)
        return boosted_path
    except Exception as e:
        logger.warning(f"Vocal isolation failed, falling back: {str(e)}")
        if demucs_tmp:
            shutil.rmtree(demucs_tmp, ignore_errors=True)
        return base_wav_path

def load_model(model_size: str, language: Optional[str]):
    if not ensure_model_cache_dir():
        raise RuntimeError("Model cache directory is not accessible")
    return whisperx.load_model(
        model_size,
        device="cuda",
        compute_type=COMPUTE_TYPE,
        download_root=MODEL_CACHE_DIR,
        language=language if language and language != "-" else None
    )

def load_alignment_model(language_code: str):
    try:
        return whisperx.load_align_model(language_code=language_code, device="cuda")
    except Exception:
        fallback_models = {
            "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
            "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
            "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
        }
        if language_code in fallback_models:
            return whisperx.load_align_model(
                model_name=fallback_models[language_code], device="cuda"
            )
        raise RuntimeError(f"No alignment model available for {language_code}")

def transcribe_audio(audio_path: str, model_size: str, language: Optional[str], align: bool):
    try:
        model = load_model(model_size, language)

        # --- FIX 1: Always request word timestamps
        result = model.transcribe(
            audio_path,
            batch_size=BATCH_SIZE,
            language=language if language and language != "-" else None,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False
        )

        # --- FIX 2 & 3: language fallback if unknown
        detected_language = result.get("language")
        if not detected_language or detected_language == "unknown":
            detected_language = language if language else "en"

        # Alignment
        if align:
            aligned = None
            try:
                align_model, metadata = load_alignment_model(detected_language)
                aligned = whisperx.align(
                    result["segments"], align_model, metadata,
                    audio_path, device="cuda", return_char_alignments=False
                )
            except Exception as e:
                logger.error(f"Alignment failed: {str(e)}")
                result["alignment_error"] = str(e)

            if aligned:
                result = aligned
            else:
                logger.warning("Using non-aligned segments (may lack words).")

        # --- FIX 4: normalize schema → every segment has "words"
        for seg in result["segments"]:
            if "words" not in seg:
                seg["words"] = None

        return {
            "text": " ".join(seg["text"] for seg in result["segments"]),
            "segments": result["segments"],
            "language": detected_language,
            "model": model_size,
            "alignment_success": "alignment_error" not in result
        }
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise RuntimeError(f"Transcription failed: {str(e)}")

def handler(job):
    try:
        if not job.get("input"):
            return {"error": "No input provided"}
        input_data = job["input"]
        file_name = input_data.get("file_name")
        if not file_name:
            return {"error": "No file_name provided"}

        # 1. Download
        local_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(file_name)}"
        try:
            s3.download_file(S3_BUCKET, file_name, local_path)
        except Exception as e:
            return {"error": f"S3 download failed: {str(e)}"}

        # 2. Convert to WAV
        try:
            audio_path = convert_to_wav(local_path)
            os.remove(local_path)
        except Exception as e:
            return {"error": f"Audio processing failed: {str(e)}"}

        # 3. Vocal isolation
        use_isolation = bool(input_data.get("use_vocal_isolation", True))
        vocal_gain_db = float(input_data.get("vocal_gain_db", VOCAL_GAIN_DB_DEFAULT))
        demucs_model = input_data.get("demucs_model", DEMUCS_MODEL)
        processed_audio_path = prepare_audio_for_transcription(
            audio_path, use_isolation, vocal_gain_db, demucs_model
        )

        # 4. Transcribe
        try:
            result = transcribe_audio(
                processed_audio_path,
                input_data.get("model_size", "large-v3"),
                input_data.get("language", None),
                input_data.get("align", False)
            )
        except Exception as e:
            return {"error": str(e)}
        finally:
            if processed_audio_path and os.path.exists(processed_audio_path) and processed_audio_path != audio_path:
                os.remove(processed_audio_path)
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
            gc.collect()
            # Delete the file from S3
            #s3.delete_object(Bucket=S3_BUCKET, Key=file_name)
        return result
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

if __name__ == "__main__":
    print("Starting WhisperX cuda Endpoint with Translation + Demucs Vocal Isolation...")
    if not ensure_model_cache_dir():
        raise RuntimeError("Model cache directory is not accessible")

    if os.environ.get("RUNPOD_SERVERLESS_MODE") == "true":
        runpod.serverless.start({"handler": handler})
    else:
        test_result = handler({
            "input": {
                "file_name": "test.wav",
                "model_size": "base",
                "language": "hi",
                "align": True,
                "use_vocal_isolation": True,
                "vocal_gain_db": 6.0,
                "demucs_model": "htdemucs"
            }
        })
        print("Test Result:", json.dumps(test_result, indent=2))
