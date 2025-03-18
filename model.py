import os
from transformers import VisionEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer

# Disable WandB logging
os.environ["WANDB_DISABLED"] = "true"

def setup_model():
    """Set up the vision-language model and tokenizer."""
    image_encoder = "google/vit-base-patch16-224-in21k"
    text_decoder = "gpt2"

    # Load the model
    vision_text_model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        image_encoder, text_decoder
    )

    # Load feature extractor and tokenizer
    extractor = AutoFeatureExtractor.from_pretrained(image_encoder)
    tokenizer = AutoTokenizer.from_pretrained(text_decoder)

    # GPT-2 lacks decoder_start/pad tokens
    tokenizer.pad_token = tokenizer.eos_token

    # Update model configuration
    vision_text_model.config.eos_token_id = tokenizer.eos_token_id
    vision_text_model.config.decoder_start_token_id = tokenizer.bos_token_id
    vision_text_model.config.pad_token_id = tokenizer.pad_token_id
    vision_text_model.generation_config.pad_token_id = tokenizer.pad_token_id

    model_path = "vit-gpt2-model"
    vision_text_model.save_pretrained(model_path)
    extractor.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    return vision_text_model, tokenizer, extractor

