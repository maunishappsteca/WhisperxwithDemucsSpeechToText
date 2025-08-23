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
import shutil  # ### NEW
import tempfile  # ### NEW

# --- Configuration ---
COMPUTE_TYPE = "float16"  # Changed to float16 for better cuda compatibility
BATCH_SIZE = 16  # Reduced batch size for cuda
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
MODEL_CACHE_DIR = os.getenv("WHISPER_MODEL_CACHE", "/app/models")

# ### NEW: Demucs-related config
DEMUCS_MODEL = os.getenv("DEMUCS_MODEL", "htdemucs")
VOCAL_GAIN_DB_DEFAULT = float(os.getenv("VOCAL_GAIN_DB", "6"))
DEMUCS_CACHE_DIR = os.getenv("DEMUCS_CACHE_DIR", os.path.join(MODEL_CACHE_DIR, "demucs"))
os.environ["DEMUCS_CACHE"] = DEMUCS_CACHE_DIR  # help demucs cache its models

# Configure logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Log to console
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Initialize S3 client
s3 = boto3.client('s3') if S3_BUCKET else None

def ensure_model_cache_dir():
    """Ensure model cache directory exists and is accessible"""
    try:
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        # Test if directory is writable
        test_file = os.path.join(MODEL_CACHE_DIR, "test.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)

        # ### NEW: ensure demucs subdir too
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
    """Convert media file to 16kHz mono WAV"""
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
        logger.error(f"FFmpeg conversion failed error: {str(e)}")
        raise RuntimeError(f"FFmpeg conversion failed: {str(e)}")
    except Exception as e:
        logger.error(f"Audio conversion error: {str(e)}")
        raise RuntimeError(f"Audio conversion error: {str(e)}")

# ### NEW: Run Demucs to separate vocals (two-stems to save compute)
def separate_vocals_with_demucs(input_wav_path: str, model_name: str) -> str:
    """
    Runs Demucs separation and returns path to vocals.wav.
    Raises on failure.
    """
    # Demucs writes to: {outdir}/{model}/{basename}/vocals.wav
    outdir = f"/tmp/demucs_{uuid.uuid4()}"
    os.makedirs(outdir, exist_ok=True)
    logger.info(f"Running Demucs '{model_name}' to isolate vocals...")
    try:
        # Use two-stems=vocals for speed/memory savings
        proc = subprocess.run(
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
        # Find vocals.wav
        vocals_path = None
        for root, _, files in os.walk(outdir):
            if "vocals.wav" in files:
                vocals_path = os.path.join(root, "vocals.wav")
                break
        if not vocals_path:
            raise RuntimeError("Demucs completed but vocals.wav not found.")

        logger.info(f"Demucs vocals found at: {vocals_path}")
        return vocals_path, outdir  # return tmp dir for later cleanup
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="ignore")
        logger.error(f"Demucs failed: {err[:5000]}")
        # Clean the output dir on failure
        shutil.rmtree(outdir, ignore_errors=True)
        raise RuntimeError("Demucs separation failed.")
    except Exception as e:
        shutil.rmtree(outdir, ignore_errors=True)
        logger.error(f"Demucs error: {str(e)}")
        raise

# ### NEW: Boost vocal volume and ensure 16 kHz mono again
def boost_and_resample(input_path: str, gain_db: float) -> str:
    """
    Boosts the audio (dB) and ensures 16k mono PCM16.
    """
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

# ### NEW: Orchestrator for your new flow
def prepare_audio_for_transcription(
    base_wav_path: str,
    use_vocal_isolation: bool,
    vocal_gain_db: float,
    demucs_model: str
) -> str:
    """
    Given a 16k mono WAV, optionally run Demucs -> boost -> return path to processed wav.
    If Demucs fails, returns the original base_wav_path.
    """
    if not use_vocal_isolation:
        return base_wav_path

    demucs_tmp = None
    try:
        vocals_path, demucs_tmp = separate_vocals_with_demucs(base_wav_path, demucs_model)
        boosted_path = boost_and_resample(vocals_path, vocal_gain_db)
        # Clean Demucs temp tree (large)
        if demucs_tmp:
            shutil.rmtree(demucs_tmp, ignore_errors=True)
        return boosted_path
    except Exception as e:
        logger.warning(f"Vocal isolation failed, falling back to original audio: {str(e)}")
        if demucs_tmp:
            shutil.rmtree(demucs_tmp, ignore_errors=True)
        return base_wav_path

def load_model(model_size: str, language: Optional[str]):
    """Load Whisper model with GPU optimization"""
    try:
        if not ensure_model_cache_dir():
            logger.error(f"Model cache directory is not accessible")
            raise RuntimeError("Model cache directory is not accessible")
            
        return whisperx.load_model(
            model_size,
            device="cuda",
            compute_type=COMPUTE_TYPE,
            download_root=MODEL_CACHE_DIR,
            language=language if language and language != "-" else None
        )
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")

def load_alignment_model(language_code: str):
    """Load alignment model with fallback options"""
    try:
        # Try to load the default model first
        return whisperx.load_align_model(language_code=language_code, device="cuda")
    except Exception as e:
        logger.warning(f"Failed to load default alignment model for {language_code}, trying fallback: {str(e)}")
        
        # Define fallback models for specific languages
        fallback_models = {
            "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",  # Hindi
            "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese", # Portuguese
            "he": "imvladikon/wav2vec2-xls-r-300m-hebrew", # Hebrew
        }
        
        if language_code in fallback_models:
            try:
                # Try to load the fallback model
                return whisperx.load_align_model(
                    model_name=fallback_models[language_code],
                    device="cuda"
                )
            except Exception as fallback_e:
                logger.error(f"Failed to load fallback alignment model for {language_code}: {str(fallback_e)}")
                raise RuntimeError(f"Alignment model loading failed for {language_code}")
        else:
            logger.error(f"No alignment model available for language: {language_code}")
            raise RuntimeError(f"No alignment model available for language: {language_code}")
        

def transcribe_audio(audio_path: str, model_size: str, language: Optional[str], align: bool):
    """Core transcription logic with optional translation"""
    try:
        model = load_model(model_size, language)
        result = model.transcribe(audio_path, batch_size=BATCH_SIZE)
        #result = model.transcribe(audio_path, batch_size=BATCH_SIZE, language=language if language and language != "-" else None, word_timestamps=True, vad_filter=True, condition_on_previous_text=False)
        detected_language = result.get("language", language if language else "en")
        
        if align and detected_language != "unknown":
            try:
                align_model, metadata = load_alignment_model(detected_language)
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    metadata,
                    audio_path,
                    device="cuda",
                    return_char_alignments=False
                )
            except Exception as e:
                logger.error(f"Alignment skipped: {str(e)}")
                # Continue without alignment if it fails
                result["alignment_error"] = str(e)

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
    """RunPod serverless handler"""
    try:
        # Validate input
        if not job.get("input"):
            return {"error": "No input provided"}
            
        input_data = job["input"]
        file_name = input_data.get("file_name")
        
        if not file_name:
            return {"error": "No file_name provided in input"}
        
        # 1. Download from S3
        local_path = f"/tmp/{uuid.uuid4()}_{os.path.basename(file_name)}"
        try:
            s3.download_file(S3_BUCKET, file_name, local_path)
        except Exception as e:
            return {"error": f"S3 download failed: {str(e)}"}
        
        # 2. Convert to WAV (16 kHz mono) if needed
        try:
            if not file_name.lower().endswith('.wav'):
                audio_path = convert_to_wav(local_path)
                os.remove(local_path)
            else:
                # even if input is wav, re-encode to guarantee 16k mono
                audio_path_16k = convert_to_wav(local_path)
                os.remove(local_path)
                audio_path = audio_path_16k
        except Exception as e:
            return {"error": f"Audio processing failed: {str(e)}"}

        # 2b. ### NEW: Demucs vocal isolation + ffmpeg boost (optional, defaults on)
        use_isolation = bool(input_data.get("use_vocal_isolation", True))
        vocal_gain_db = float(input_data.get("vocal_gain_db", VOCAL_GAIN_DB_DEFAULT))
        demucs_model = input_data.get("demucs_model", DEMUCS_MODEL)

        try:
            processed_audio_path = prepare_audio_for_transcription(
                base_wav_path=audio_path,
                use_vocal_isolation=use_isolation,
                vocal_gain_db=vocal_gain_db,
                demucs_model=demucs_model
            )
        except Exception as e:
            # prepare_audio already logs and falls back; this is just extra safety
            logger.warning(f"prepare_audio_for_transcription error: {str(e)}")
            processed_audio_path = audio_path

        # 3. Transcribe (NOTE: we pass the processed path)
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
            # 4. Cleanup
            # Remove the processed file if it is not the original audio_path
            if processed_audio_path and os.path.exists(processed_audio_path) and processed_audio_path != audio_path:
                try:
                    os.remove(processed_audio_path)
                except Exception:
                    pass
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception:
                    pass
            gc.collect()
        
        # 5. Keep response structure IDENTICAL
        return result
        
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

if __name__ == "__main__":
    print("Starting WhisperX cuda Endpoint with Translation + Demucs Vocal Isolation...")

    # Verify model cache directory at startup
    if not ensure_model_cache_dir():
        print("ERROR: Model cache directory is not accessible")
        if os.environ.get("RUNPOD_SERVERLESS_MODE") == "true":
            # In serverless mode, we want to fail fast if model dir isn't available
            raise RuntimeError("Model cache directory is not accessible")
    
    if os.environ.get("RUNPOD_SERVERLESS_MODE") == "true":
        runpod.serverless.start({"handler": handler})
    else:
        # Test with mock input (demucs off for quick test)
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
