"""
inspect_data.py - Visualize and validate the dataset before training
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pathlib import Path

from dataset import get_datasets, get_dataset_info
from augmentation import visualize_augmentations
from config import PROCESSED_DATA_DIR


def plot_class_distribution(info: dict, save_path: Path = None):
    """Plot the distribution of samples across classes."""
    import pandas as pd
    
    # Load all CSVs to get class counts
    splits_data = {}
    for split in ['train', 'val', 'test']:
        csv_path = PROCESSED_DATA_DIR / f"processed_{split}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            class_counts = df['label'].value_counts().sort_index()
            splits_data[split] = class_counts
    
    if not splits_data:
        print("No data found!")
        return
    
    # Get all unique classes
    all_classes = set()
    for counts in splits_data.values():
        all_classes.update(counts.index)
    all_classes = sorted(all_classes)
    
    # Prepare data for grouped bar chart
    x = np.arange(len(all_classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = {'train': '#2E86AB', 'val': '#A23B72', 'test': '#F18F01'}
    
    for i, (split, counts) in enumerate(splits_data.items()):
        values = [counts.get(cls, 0) for cls in all_classes]
        offset = width * (i - 1)
        ax.bar(x + offset, values, width, label=split.capitalize(), 
               color=colors.get(split, 'gray'))
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Dataset Class Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved class distribution to {save_path}")
    
    plt.show()


def visualize_batch(dataset, class_names: list = None, num_images: int = 16):
    """Visualize a batch of images from the dataset."""
    # Get one batch
    for images, labels in dataset.take(1):
        n = min(num_images, len(images))
        
        grid_size = int(np.ceil(np.sqrt(n)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(n):
            img = images[i].numpy()
            label_idx = labels[i].numpy()
            
            # Display image
            axes[i].imshow(img)
            axes[i].axis('off')
            
            # Add label
            if class_names and label_idx < len(class_names):
                title = class_names[label_idx]
            else:
                title = f"Class {label_idx}"
            axes[i].set_title(title, fontsize=8)
        
        # Hide extra subplots
        for i in range(n, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        break


def compare_original_vs_augmented(dataset, num_examples: int = 3):
    """Show original images alongside their augmented versions."""
    from augmentation import visualize_augmentations
    
    for images, labels in dataset.take(1):
        for idx in range(min(num_examples, len(images))):
            img = images[idx]
            label = labels[idx].numpy()
            
            # Generate augmented versions
            aug_images = visualize_augmentations(img, num_examples=8)
            
            # Plot
            fig, axes = plt.subplots(3, 3, figsize=(9, 9))
            axes = axes.flatten()
            
            # Original in center
            axes[4].imshow(img.numpy())
            axes[4].set_title(f'ORIGINAL (Class {label})', 
                             fontweight='bold', color='red')
            axes[4].axis('off')
            
            # Augmented versions around it
            positions = [0, 1, 2, 3, 5, 6, 7, 8]
            for i, pos in enumerate(positions):
                axes[pos].imshow(aug_images[i].numpy())
                axes[pos].set_title(f'Aug {i+1}')
                axes[pos].axis('off')
            
            plt.suptitle('Original vs Augmented Versions', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
        break


def check_image_statistics(dataset, num_batches: int = 10):
    """Compute statistics about the images in the dataset."""
    print("Computing image statistics...")
    
    pixel_values = []
    
    for i, (images, _) in enumerate(dataset.take(num_batches)):
        pixel_values.append(images.numpy().flatten())
        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1}/{num_batches} batches...")
    
    pixel_values = np.concatenate(pixel_values)
    
    print("\n" + "="*60)
    print("IMAGE STATISTICS")
    print("="*60)
    print(f"Min pixel value:    {pixel_values.min():.4f}")
    print(f"Max pixel value:    {pixel_values.max():.4f}")
    print(f"Mean pixel value:   {pixel_values.mean():.4f}")
    print(f"Std pixel value:    {pixel_values.std():.4f}")
    print(f"Median pixel value: {np.median(pixel_values):.4f}")
    
    # Plot histogram
    plt.figure(figsize=(10, 4))
    plt.hist(pixel_values, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Pixel Values Across Dataset')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    """Run all inspection visualizations."""
    print("="*60)
    print("DATASET INSPECTION")
    print("="*60)
    
    # 1. Get dataset info
    print("\n1. Loading dataset information...")
    info = get_dataset_info()
    print(f"\nDataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 2. Plot class distribution
    print("\n2. Plotting class distribution...")
    plot_class_distribution(info)
    
    # 3. Load datasets
    print("\n3. Loading datasets...")
    train_ds, val_ds, test_ds = get_datasets(batch_size=32)
    
    # 4. Visualize training batch
    print("\n4. Visualizing training batch...")
    class_names = info.get('class_names', None)
    visualize_batch(train_ds, class_names=class_names)
    
    # 5. Compare original vs augmented
    print("\n5. Comparing original vs augmented images...")
    compare_original_vs_augmented(train_ds)
    
    # 6. Check image statistics
    print("\n6. Computing image statistics...")
    check_image_statistics(train_ds)
    
    print("\n" + "="*60)
    print("INSPECTION COMPLETE!")


if __name__ == "__main__":
    main()