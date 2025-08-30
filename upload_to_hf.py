from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="model/model.safetensors",
    path_in_repo="model.safetensors",
    repo_id="Sxhni/deepfake-detector-vit",
    repo_type="model"
)

api.upload_file(
    path_or_fileobj="model/config.json",
    path_in_repo="config.json",
    repo_id="Sxhni/deepfake-detector-vit",
    repo_type="model"
)

api.upload_file(
    path_or_fileobj="model/preprocessor_config.json",
    path_in_repo="preprocessor_config.json",
    repo_id="Sxhni/deepfake-detector-vit",
    repo_type="model"
)
