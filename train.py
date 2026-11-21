# Entry point for training models or checking data integrity.
# Delegates execution to the hiervision trainer or data validation modules.

import sys
from hiervision import trainer, data

def main() -> None:
    # A standalone check flag intercept to run data validation without starting training.
    if "--check" in sys.argv:
        data.check_data()
        print("Data check passed. Hierarchy is valid.")
        return

    trainer.main()

if __name__ == "__main__":
    main()