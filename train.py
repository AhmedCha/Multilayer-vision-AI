# Entry point for training models or checking data integrity.
# Delegates execution to the hiervision trainer.

from hiervision import trainer

def main() -> None:
    trainer.main()

if __name__ == "__main__":
    main()