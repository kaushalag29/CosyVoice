"""
CosyVoice3 Server
Hosts the Fun-CosyVoice3-0.5B model for cross-lingual voice cloning via HTTP API.
"""
import os
import sys
import torch
import torchaudio
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Add third_party/Matcha-TTS to path (required by CosyVoice)
matcha_tts_path = Path(__file__).parent / "third_party" / "Matcha-TTS"
sys.path.insert(0, str(matcha_tts_path))

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import load_wav

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instance
cosyvoice_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    initialize_model()
    yield
    # Shutdown (cleanup if needed)
    pass

app = FastAPI(lifespan=lifespan)

class GenerateAudioRequest(BaseModel):
    text: str
    reference_audio_path: str
    reference_text: str  # Original subtitle text from reference audio
    output_path: str
    stream: bool = False
    language_id: str = "en"  # Target language (2-letter ISO code)

def initialize_model():
    """Initialize CosyVoice3 model on startup"""
    global cosyvoice_model

    logger.info("Initializing CosyVoice3 model...")
    
    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    logger.info(f"Using device: {device}")
    
    try:
        model_dir = Path(__file__).parent / "pretrained_models" / "Fun-CosyVoice3-0.5B"

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Fun-CosyVoice3-0.5B model not found at {model_dir}. "
                "Please download it first using the instructions in README.md"
            )

        # Initialize CosyVoice3 model using AutoModel
        # AutoModel automatically detects model version based on cosyvoice3.yaml
        # load_jit=False, load_trt=False, load_vllm=False for standard inference
        cosyvoice_model = AutoModel(
            model_dir=str(model_dir),
            load_trt=torch.cuda.is_available(),
            load_vllm=torch.cuda.is_available(),
            fp16=False  # Note: FP16 has issues with CosyVoice3's DiT architecture
        )

        logger.info(f"CosyVoice3 model initialized successfully")
        logger.info(f"Model sample rate: {cosyvoice_model.sample_rate}")
        
    except Exception as e:
        logger.error(f"Failed to initialize CosyVoice3 model: {e}", exc_info=True)
        raise

@app.post("/generate-audio")
async def generate_audio(request: GenerateAudioRequest):
    """
    Generate audio using CosyVoice3 cross-lingual voice cloning.

    This endpoint uses inference_cross_lingual for cross-language dubbing
    (e.g., Japanese audio -> English speech while preserving voice characteristics).
    CosyVoice3 offers superior content consistency, speaker similarity, and prosody naturalness.
    """
    # Language tag mapping for CosyVoice3
    # CosyVoice3 supports 9 languages and 18+ Chinese dialects
    # Language tags: <|zh|> (Chinese), <|en|> (English), <|jp|> (Japanese),
    # <|ko|> (Korean), <|yue|> (Cantonese), plus German, Spanish, French, Italian, Russian
    LANGUAGE_TAG_MAP = {
        "zh": "<|zh|>",
        "en": "<|en|>",
        "ja": "<|jp|>",  # Note: CosyVoice uses "jp" internally, not "ja"
        "ko": "<|ko|>",
        "yue": "<|yue|>"
    }
    
    try:
        logger.info("Received generate-audio request")
        logger.info(f"Text length: {len(request.text)} chars")
        logger.info(f"Reference audio: {request.reference_audio_path}")
        logger.info(f"Reference text: {request.reference_text[:100]}...")
        logger.info(f"Target language: {request.language_id}")
        
        # Validate inputs
        if not os.path.exists(request.reference_audio_path):
            raise HTTPException(
                status_code=400, 
                detail=f"Reference audio file not found: {request.reference_audio_path}"
            )
        
        # Ensure output directory exists
        output_dir = os.path.dirname(request.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Load reference audio at 16kHz (CosyVoice3 requirement)
        logger.info("Loading reference audio...")
        prompt_speech_16k = load_wav(request.reference_audio_path, 16000)
        
        # CosyVoice3 requires <|endofprompt|> token format
        # Format: "system_prompt<|endofprompt|>language_tag + text_to_speak"
        # IMPORTANT: Only text AFTER <|endofprompt|> will be spoken
        language_tag = LANGUAGE_TAG_MAP.get(request.language_id, "<|en|>")
        tagged_text = f"You are a helpful assistant.<|endofprompt|>{language_tag}{request.text}"

        logger.info("Starting CosyVoice3 generation...")
        logger.info(f"  Generate text: '{request.text[:100]}...'")
        logger.info(f"  Language tag: '{language_tag}'")
        logger.info(f"  Tagged text: '{tagged_text[:100]}...'")

        # Use cross_lingual inference for cross-language voice cloning
        # This allows Japanese voice -> English speech while preserving voice characteristics
        # The <|endofprompt|> token ensures only the actual text is spoken
        audio_chunks = []

        for i, result in enumerate(cosyvoice_model.inference_cross_lingual(
            tagged_text,  # Text with system prompt and language specification
            prompt_speech_16k,
            stream=request.stream
        )):
            logger.info(f"Generated chunk {i}")
            audio_chunks.append(result['tts_speech'])
        
        # Concatenate all audio chunks
        if len(audio_chunks) > 1:
            final_audio = torch.cat(audio_chunks, dim=1)
        else:
            final_audio = audio_chunks[0]
        
        # Save the generated audio
        torchaudio.save(
            request.output_path,
            final_audio.cpu(),
            cosyvoice_model.sample_rate
        )
        
        logger.info(f"Audio generated successfully: {request.output_path}")
        return {
            "status": "success",
            "output_path": request.output_path,
            "sample_rate": cosyvoice_model.sample_rate,
            "duration": final_audio.shape[-1] / cosyvoice_model.sample_rate,
            "chunks": len(audio_chunks)
        }
        
    except Exception as e:
        logger.error(f"Error generating audio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if cosyvoice_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    return {
        "status": "healthy",
        "model": "Fun-CosyVoice3-0.5B",
        "sample_rate": cosyvoice_model.sample_rate
    }

@app.post("/shutdown")
async def shutdown():
    """Shutdown endpoint for graceful termination"""
    logger.info("Shutdown request received")
    return {"status": "shutting down"}

if __name__ == "__main__":
    # Port is configurable via environment variable
    port = int(os.environ.get("PORT", 8018))
    logger.info(f"Starting CosyVoice server on port {port}")

    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
