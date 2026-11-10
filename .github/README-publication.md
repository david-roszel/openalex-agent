# Publication checks

The workflow installs dependencies, scans exact tracked/index blobs, verifies the public manifest, then runs the current app and isolated frozen R7 synthetic suites with real IP networking blocked. It never reads a grading key, starts a hosted research run or creates an experimental approval. The matrix is configured for Python3.11/3.14; CI results are available only after an approved push.
