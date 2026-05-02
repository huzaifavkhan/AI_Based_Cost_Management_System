"""
Start all 3 FL client servers (ports 8001, 8002, 8003).

Run from the cost-management-system/ directory:
    python start_fl_clients.py

Each client is a real separate process. It loads its own data slice at startup
and exposes a /train endpoint. Raw invoice data never leaves the process —
the main server only ever sends and receives model weight vectors.

Then start the main server in a separate terminal:
    uvicorn app.main:app --reload

Then open the dashboard:
    streamlit run dashboard/app.py
"""
import subprocess
import sys
import os
import time
import signal

CLIENTS = [
    ("client_a", 8001),
    ("client_b", 8002),
    ("client_c", 8003),
]

procs: list[subprocess.Popen] = []


def start_clients():
    for client_id, port in CLIENTS:
        env = {**os.environ, "CLIENT_ID": client_id}
        cmd = [
            sys.executable, "-m", "uvicorn",
            "app.federated.client_server:app",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--log-level", "warning",   # quieter logs; change to "info" to debug
        ]
        p = subprocess.Popen(cmd, env=env)
        procs.append(p)
        print(f"  Started {client_id:10s}  →  http://localhost:{port}  (PID {p.pid})")

    print()
    print("All 3 FL client servers running.")
    print("Coordinator will auto-detect them when you click 'Start Federated Training'.")
    print()
    print("Press Ctrl+C to stop all clients.\n")


def shutdown(signum=None, frame=None):
    print("\nStopping FL client servers …")
    for p in procs:
        p.terminate()
    for p in procs:
        p.wait()
    print("All clients stopped.")
    sys.exit(0)


if __name__ == "__main__":
    print("\nStarting FL client servers …\n")
    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    start_clients()

    # Wait for all child processes
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        shutdown()
