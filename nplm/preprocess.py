from datasets import Dataset as HFDataset, DatasetDict
from torch.utils.data import Dataset as TorchDataset
def read_dataset(dataset_path):
    """This function will read a dataset from disk. It should have been downloaded 
    with the `download_hf_data` function from the `download` module.
    
    Args:
        dataset_path (str): The path to the dataset on disk.

    Returns:
        DatasetDict: The loaded dataset.
    """
    from datasets import load_from_disk
    return load_from_disk(dataset_path)


def generate_documents(dataset : DatasetDict | HFDataset):
    """This function will take a dataset or dataset dict object and generate documents from it.
    The function should handle both DatasetDict and Dataset objects.
    The function should be able to handle multiple types of 'documents' within the dataset:

         - words 
         - sentences
         - paragraphs
         - articles 
         - groups thereof
    This function should return a simple list of documents. 

    Args:
        dataset (DatasetDict | HFDataset): The dataset or dataset dict to generate documents from.

    Returns:
        list: A list of generated documents.
    """
    return

def split_dataset(dataset : DatasetDict | HFDataset, test_size=0.2):
    """This function will split a dataset or dataset dict into training and testing sets.

    Args:
        dataset (DatasetDict | HFDataset): The dataset or dataset dict to split.
        test_size (float): The proportion of the dataset to include in the test split.

    Returns:
        tuple: A tuple containing the training and testing datasets.
    """
    from sklearn.model_selection import train_test_split
    if isinstance(dataset, HFDataset):
        train_indices, test_indices = train_test_split(range(len(dataset)), test_size=test_size)
        return dataset.select(train_indices), dataset.select(test_indices)
    elif isinstance(dataset, DatasetDict):
        train_dataset = {}
        test_dataset = {}
        for key, ds in dataset.items():
            train_indices, test_indices = train_test_split(range(len(ds)), test_size=test_size)
            train_dataset[key] = ds.select(train_indices)
            test_dataset[key] = ds.select(test_indices)
        return DatasetDict(train_dataset), DatasetDict(test_dataset)