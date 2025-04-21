"""
Miscellaneous utilities.
"""
import pandas as pd


def save_results_to_csv(results: list, csv_path: str) -> None:
    """Save list of dicts to CSV."""
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

```}
