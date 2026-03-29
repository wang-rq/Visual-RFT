from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

try:
    from transformers import AriaForConditionalGeneration
except ImportError:
    AriaForConditionalGeneration = None

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

try:
    from transformers import Qwen2VLForConditionalGeneration
except ImportError:
    Qwen2VLForConditionalGeneration = None

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None


def is_qwen_vl_model(model_id: str) -> bool:
    return any(name in model_id for name in ("Qwen2-VL", "Qwen2.5-VL", "Qwen3-VL"))


def is_vision_language_model(model_id: str) -> bool:
    return is_qwen_vl_model(model_id) or "Aria" in model_id


def load_generation_model(model_id: str, model_init_kwargs: dict):
    qwen_vl_kwargs = dict(model_init_kwargs)
    qwen_vl_kwargs.pop("use_cache", None)

    if "Qwen2-VL" in model_id and Qwen2VLForConditionalGeneration is not None:
        return Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
    if "Qwen2.5-VL" in model_id and Qwen2_5_VLForConditionalGeneration is not None:
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **qwen_vl_kwargs)
    if "Qwen3-VL" in model_id:
        if Qwen3VLForConditionalGeneration is not None:
            return Qwen3VLForConditionalGeneration.from_pretrained(model_id, **qwen_vl_kwargs)
        if AutoModelForImageTextToText is not None:
            return AutoModelForImageTextToText.from_pretrained(model_id, **qwen_vl_kwargs)
    if "Aria" in model_id and AriaForConditionalGeneration is not None:
        aria_kwargs = dict(model_init_kwargs)
        aria_kwargs.pop("use_cache", None)
        return AriaForConditionalGeneration.from_pretrained(model_id, **aria_kwargs)
    return AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)


def load_processing_class(model_id: str, max_pixels: int, min_pixels: int):
    if is_vision_language_model(model_id):
        processing_class = AutoProcessor.from_pretrained(model_id)
        pad_token_id = processing_class.tokenizer.pad_token_id
        processing_class.pad_token_id = pad_token_id
        processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
        image_processor = getattr(processing_class, "image_processor", None)
        if image_processor is not None:
            if hasattr(image_processor, "max_pixels"):
                image_processor.max_pixels = max_pixels
            if hasattr(image_processor, "min_pixels"):
                image_processor.min_pixels = min_pixels
        return processing_class, pad_token_id

    processing_class = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    pad_token_id = processing_class.pad_token_id
    return processing_class, pad_token_id
