from src.train import main as train_main
from src.predict import main as predict_main
from src.analyze import main as analyze_main


def main() -> None:
    """Run the full project pipeline."""
    print("Step 1/3: train model")
    train_main()

    print("\nStep 2/3: generate predictions")
    predict_main()

    print("\nStep 3/3: create analysis artifacts")
    analyze_main()

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()