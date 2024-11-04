# Change directory to the project folder
cd C:\Users\j48h1\AI-Startup\Repos\whisper-writer

# Activate the virtual environment and run the Python script in the background
Start-Process powershell -ArgumentList {
    .\venv\Scripts\activate
    python run.py
} -NoNewWindow

# Exit the current PowerShell session
exit