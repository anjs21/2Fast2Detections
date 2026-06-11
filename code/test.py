import torch
from models import get_backbone

def test_load_aide():
    print("Attempting to load the AIDE backbone...")
    try:
        model, num_features, shared_layer = get_backbone("aide")
        print("AIDE model loaded successfully!")
        print(f"Number of features: {num_features}")
        print(f"Shared layer: {shared_layer}")
        
        # Test forward pass with dummy input
        print("Running forward pass with a dummy tensor...")
        dummy_input = torch.randn(2, 3, 256, 256)
        output = model(dummy_input)
        print(f"Forward pass completed successfully! Output shape: {output.shape}")
    except Exception as e:
        print("An error occurred while loading or running AIDE model:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_load_aide()