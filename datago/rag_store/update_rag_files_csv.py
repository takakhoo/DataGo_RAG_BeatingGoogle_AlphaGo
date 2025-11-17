import os
import csv
from pathlib import Path

# Define the path to the RAG data directory
# RAG_DATA_DIR = "../../katago_repo/KataGo/cpp/datago_implement/rag_data"
RAG_DATA_DIR = "../../../alphago_project/build/rag_data"

# Output CSV file path (in the same directory as this script)
OUTPUT_CSV = "rag_files_list.csv"


def update_rag_files_csv():
    """
    Scans RAG_DATA_DIR for JSON files starting with 'RAG_raw',
    and writes all filenames to a CSV file.
    """
    # Get absolute path to RAG data directory
    script_dir = Path(__file__).parent
    rag_dir = (script_dir / RAG_DATA_DIR).resolve()

    # Check if directory exists
    if not rag_dir.exists():
        print(f"Error: RAG data directory does not exist: {rag_dir}")
        return

    # Find all JSON files starting with "RAG_raw"
    rag_files = sorted([f.name for f in rag_dir.glob("RAG_raw*.json")])

    if not rag_files:
        print(f"No RAG files found in {rag_dir}")
        return

    # Write to CSV
    csv_path = script_dir / OUTPUT_CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename'])  # Header
        for filename in rag_files:
            writer.writerow([filename])

    print(f"Updated {csv_path} with {len(rag_files)} RAG files")


if __name__ == "__main__":
    update_rag_files_csv()