import torch

model_path = "models/pharmbert_p7_best.pt"

model = torch.load(
    model_path,
    map_location="cpu"
)

print(type(model))

print("Model loaded successfully")