import os
import datasets
from PIL import Image

def download_and_process_data(tokenizer, extractor):
    """Download dataset and process images and captions."""
    coco_urls = [
        "http://images.cocodataset.org/zips/train2017.zip",
        "http://images.cocodataset.org/zips/val2017.zip",
        "http://images.cocodataset.org/zips/test2017.zip",
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "http://images.cocodataset.org/annotations/image_info_test2017.zip"
    ]

    for url in coco_urls:
        os.system(f"wget {url}")

    dataset_path = "/content/"
    dataset = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", data_dir=dataset_path)
    

    def tokenize_captions(captions, max_length):
        """Tokenize text captions."""
        return tokenizer(captions, padding="max_length", max_length=max_length).input_ids

    def extract_features(image_paths, validate_images=True):
        """Extract image features using ViT."""
        processed_images = []
        valid_indices = []

        if validate_images:
            for img_path in image_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    processed_images.append(img)
                    valid_indices.append(True)
                except Exception:
                    valid_indices.append(False)
        else:
            processed_images = [Image.open(img).convert("RGB") for img in image_paths]

        encoder_inputs = extractor(images=processed_images, return_tensors="np")
        return encoder_inputs.pixel_values

    def process_samples(examples, max_length, validate=True):
        """Process dataset examples for training."""
        img_paths = examples['image_path']
        text_captions = examples['caption']

        return {
            'labels': tokenize_captions(text_captions, max_length),
            'pixel_values': extract_features(img_paths, validate)
        }

    processed_data = dataset.map(
        function=process_samples,
        batched=True,
        fn_kwargs={"max_length": 128},
        remove_columns=dataset['train'].column_names
    )

    return processed_data

