import random

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Splits a dataset into training, validation, and test sets.

    Args:
        dataset (list): The dataset to be split.
        train_ratio (float): Ratio of training data.
        val_ratio (float): Ratio of validation data.
        test_ratio (float): Ratio of test data.

    Returns:
        tuple: Three lists corresponding to training, validation, and test sets.
    """
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("The sum of train_ratio, val_ratio, and test_ratio must equal 1.0")

    random.shuffle(dataset)  # Shuffle the dataset randomly
    total = len(dataset)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_set = dataset[:train_end]
    val_set = dataset[train_end:val_end]
    test_set = dataset[val_end:]

    return train_set, val_set, test_set

# Example usage
if __name__ == "__main__":
    # Replace this with your dataset (e.g., a list of file paths or data entries)
    dataset = [f"data_{i}" for i in range(100)]  # Example dataset

    # Split the dataset
    train_set, val_set, test_set = split_dataset(dataset)

    # Output the results
    print("Training Set:", train_set)
    print("Validation Set:", val_set)
    print("Test Set:", test_set)

    # Optionally save to files
    with open("train_set.txt", "w") as train_file:
        train_file.write("\n".join(train_set))

    with open("val_set.txt", "w") as val_file:
        val_file.write("\n".join(val_set))

    with open("test_set.txt", "w") as test_file:
        test_file.write("\n".join(test_set))
