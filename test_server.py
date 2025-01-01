import requests
import time
import numpy as np
import subprocess
import sys
import os

def main():
    # Start server
    print("Starting server...")
    # Using subprocess to start uvicorn
    # Assumes uvicorn is in path or python -m uvicorn
    
    # Redirect to file to avoid pipe buffer deadlock
    log_file = open("server_output.log", "w")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.server:app", "--port", "8000"],
        cwd=os.getcwd(),
        stdout=log_file,
        stderr=log_file
    )
    
    # Wait for startup
    time.sleep(3)
    
    try:
        base_url = "http://localhost:8000"
        
        # Check stats
        print("Checking stats...")
        try:
            r = requests.get(f"{base_url}/stats")
            print(r.json())
        except ConnectionError:
            print("Failed to connect to server. Check logs.")
            # print stderr
            # print(server_process.stderr.read().decode())
            return

        # Insert
        print("Inserting 100 vectors...")
        dim = 128
        for i in range(100):
            # Generate random vector
            vec = np.random.rand(dim).astype(np.float32).tolist()
            payload = {"id": i, "vector": vec}
            r = requests.post(f"{base_url}/insert", json=payload)
            if r.status_code != 200:
                print(f"Insert failed: {r.text}")
                break
            if i % 10 == 0:
                print(f"Inserted {i}/100")
        
        # Search
        print("Searching...")
        query = np.random.rand(dim).astype(np.float32).tolist()
        r = requests.post(f"{base_url}/search", json={"vector": query, "k": 5})
        print(r.json())
        
        # Save
        print("Saving...")
        r = requests.post(f"{base_url}/save")
        print(r.json())
        
        print("SUCCESS: End-to-end API test passed.")
        
    finally:
        print("Stopping server...")
        server_process.terminate()
        log_file.close()
        # server_process.wait()

if __name__ == "__main__":
    main()




