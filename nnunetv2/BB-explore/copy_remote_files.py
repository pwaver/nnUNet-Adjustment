import os
import re
import subprocess
1+1
# Configuration
remote_host = "ubuntu@remote-dl-dev.angiowavedata.com"
ssh_key_path = os.path.expanduser("~/.ssh/AWI-remote-dev.pem")
local_base_dir = "/Users/billb/Projects/AWI/NetExploration/nnUNet"
log_file = os.path.join(local_base_dir, "training_logs.txt")

# To populate the log file, run the following command on the remote system:
# find . -type f -name "training_*.txt" -size +20k -ls > training_logs.txt

# ssh -i ~/.ssh/AWI-remote-dev.pem ubuntu@remote-dl-dev.angiowavedata.com 'find . -type f -name "training_*.txt" -size +20k -ls > training_logs.txt'

# Expand the ~ in the path
ssh_key_path = os.path.expanduser('~/.ssh/AWI-remote-dev.pem')

ssh_command = [
    'ssh',
    '-i', ssh_key_path,
    'ubuntu@remote-dl-dev.angiowavedata.com',
    'find . -type f -name "training_*.txt" -size +20k -ls'
]

try:
    result = subprocess.run(
        ssh_command,
        check=True,
        text=True,
        capture_output=True
    )
    print("Command executed successfully")
    if result.stdout:
        with open(log_file, 'w') as f:
            f.write(result.stdout)
        print(f"Output written to {log_file}")
except subprocess.CalledProcessError as e:
    print("Error executing command:", e)
    if e.stderr:
        print("Error output:", e.stderr)


def extract_paths_from_log(log_file):
    """Extract file paths from the log file."""
    paths = set()
    with open(log_file, 'r') as f:
        content = f.read()
        # This pattern looks for file paths - adjust if needed
        file_paths = re.findall(r'(?:/[^\s]+)+(?:\.[^\s]+)?', content)
        paths.update(file_paths)
    return paths

def create_local_directory(path):
    """Create local directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def copy_file(remote_path, local_path):
    """Copy file from remote to local using scp."""
    local_dir = os.path.dirname(local_path)
    create_local_directory(local_dir)
    
    scp_command = [
        "scp",
        "-i", ssh_key_path,
        "-o", "StrictHostKeyChecking=no",
        f"{remote_host}:/home/ubuntu/{remote_path}",
        local_path
    ]
    
    try:
        subprocess.run(scp_command, check=True)
        print(f"Successfully copied: {remote_path} -> {local_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to copy {remote_path}: {e}")

def main():
    # Extract paths from log file
    remote_paths = extract_paths_from_log(log_file)
    
    # Process each path
    for remote_path in remote_paths:
        # Skip if it doesn't look like a file path
        if not remote_path.startswith('/'):
            continue
            
        # Create local path
        local_path = os.path.join(local_base_dir, remote_path.lstrip('/'))
        
        # Copy the file
        copy_file(remote_path, local_path)

if __name__ == "__main__":
    main()
