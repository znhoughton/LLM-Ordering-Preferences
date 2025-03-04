import subprocess
import csv

# Path to your CSV file with checkpoints
csv_file = "../Data/target_checkpoints.csv"

# Base command to run your script
base_command = ["python", "get_ordering_prefs.py", 
                "--model", "allenai/OLMo-7B-0424-hf", 
                "--wordlist", "../Data/nonce_binoms.csv"]

# Read checkpoint values from the CSV file
with open(csv_file, mode="r") as file:
    reader = csv.reader(file)
    for row in reader:
        checkpoint = row[0]  # Get the checkpoint value from the row

        # Build the full command with the current checkpoint
        command = base_command + ["--checkpoint", checkpoint]
        print(f"Running: {' '.join(command)}")  # Optional: Print the command

        # Run the command
        subprocess.run(command)
