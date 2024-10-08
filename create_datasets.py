from transformer_lens import HookedTransformer
from load_dataset import create_sva_datasets

if __name__ == "__main__":
    # Load the model
    print("Loading model...")
    model = HookedTransformer.from_pretrained(
        "gemma-2b",
        center_unembed=True,
        center_writing_weights=False,
        fold_ln=False,
        fold_value_biases=False,
    )
    print("Model loaded successfully.")
    # Create SVA datasets
    print("Creating SVA datasets...")
    create_sva_datasets(model)
    print("SVA datasets created successfully.")


