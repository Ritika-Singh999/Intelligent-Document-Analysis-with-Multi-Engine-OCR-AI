from typing import Tuple
import logging

logger = logging.getLogger(__name__)

_STATE = {
    "loaded": False,
    "processor": None,
    "model": None,
    "model_name": "naver-clova-ix/donut-base-finetuned-docvqa",
}


def get_donut() -> Tuple[object, object]:
    """Return (processor, model). Loads them on first call.

    Returns:
        Tuple[DonutProcessor, VisionEncoderDecoderModel] or (None, None) if loading fails
    """
    if _STATE["loaded"]:
        return _STATE["processor"], _STATE["model"]

    try:
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        logger.info("Loading Donut model for document understanding...")
        processor = DonutProcessor.from_pretrained(_STATE["model_name"])
        model = VisionEncoderDecoderModel.from_pretrained(_STATE["model_name"])
        model.eval()

        _STATE["processor"] = processor
        _STATE["model"] = model
        _STATE["loaded"] = True
        return processor, model
    except Exception as e:
        logger.error(f"Failed to initialize Donut model: {e}")
        # Return None instead of raising to allow graceful degradation
        return None, None
