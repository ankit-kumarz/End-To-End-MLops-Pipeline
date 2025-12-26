"""
Generate Sample Dataset

Creates a sample dataset for demonstrating the MLOps pipeline.
This would typically be replaced with real data in production.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_sample_data(n_samples: int = 1000, output_path: str = "data/raw/dataset.csv"):
    """
    Generate synthetic dataset for classification task.
    
    Args:
        n_samples: Number of samples to generate
        output_path: Path where to save the CSV file
    """
    np.random.seed(42)
    
    # Generate features
    feature1 = np.random.randn(n_samples)
    feature2 = np.random.randn(n_samples)
    feature3 = np.random.randn(n_samples)
    feature4 = np.random.uniform(0, 100, n_samples)
    feature5 = np.random.randint(1, 10, n_samples)
    
    # Generate target with some relationship to features
    target = (
        (feature1 > 0).astype(int) +
        (feature2 > 0.5).astype(int) +
        (feature4 > 50).astype(int)
    ) > 1
    target = target.astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'feature_1': feature1,
        'feature_2': feature2,
        'feature_3': feature3,
        'feature_4': feature4,
        'feature_5': feature5,
        'target': target
    })
    
    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Generated {n_samples} samples")
    print(f"✅ Saved to: {output_path}")
    print(f"✅ File size: {output_file.stat().st_size / 1024:.2f} KB")
    print(f"\nDataset shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    
    return df


if __name__ == "__main__":
    df = generate_sample_data()
    print("\n📊 Sample data preview:")
    print(df.head())
