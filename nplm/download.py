

import argparse
import datasets
from datasets import DatasetDict

def download_hf_data(dataset_name, output):
    """Downloads a hugging face dataset from the given URL and saves it to the specified output file."""
    dataset = datasets.load_dataset(dataset_name)
    return dataset

def save_hf_data(dataset : DatasetDict , output):
    """Saves a hugging face dataset to the specified output file."""
    dataset.save_to_disk(output)
    
def parse_args():
    
    parser = argparse.ArgumentParser(description="Download NPLM data")
    parser.add_argument("--dataset", type=str, required=True, help="name of the dataset to download")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    return parser.parse_args()

def main():
    """Main function to download NPLM data based on command-line arguments.
    
    It should parse the command-line arguments and download the specified data.
        --dataset: The name of the dataset to download.
        --output: The path to the output file where the dataset will be saved.
    It should call a function to download a dataset from hugging face. 
    It should write the dataset to a specified output file."""
    args = parse_args()
    dataset = download_hf_data(args.dataset, args.output)
    save_hf_data(dataset, args.output)
if __name__ == "__main__":
    main()