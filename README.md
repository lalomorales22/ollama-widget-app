# Ollama Advanced Chat Desktop App

**Version:** 1.1.0


A feature-rich desktop application for interacting with local Ollama models. This Python-based app provides a tabbed interface for managing multiple conversations, supports vision (image attachments), RAG (PDF/TXT document context), speech-to-text (Whisper/Sphinx), and stores all chat history persistently in an SQLite database.

![Screenshot 2025-05-09 at 9 57 58 AM](https://github.com/user-attachments/assets/c0fda04f-23b8-45f1-8015-791e6090ffbc)

## Features

* **Multiple Chat Sessions:** Manage different conversations in separate, closable, and reorderable tabs.
* **Persistent Storage:** All chats and messages are automatically saved to a local SQLite database (`ollama_chat.db`).
* **Model Selection per Chat:** Choose different Ollama models for each conversation.
* **Vision Support:** Attach images (PNG, JPG, JPEG) to your messages for interaction with vision-enabled Ollama models (e.g., LLaVA). Images are displayed as thumbnails in the chat.
* **RAG (Retrieval Augmented Generation):** Attach PDF or TXT files. The content of these documents is used as context for your next message to the Ollama model.
* **Speech-to-Text:** Use your microphone to dictate messages. The app attempts to use OpenAI's Whisper (local) and can fall back to CMU Sphinx.
* **Chat Management:**
    * Rename chat tabs.
    * Permanently delete chats and their history from the database.
* **Export Chats:** Export individual chat histories to JSON or plain text files.
* **Configurable Ollama URL:** Set the API endpoint for your Ollama instance via the settings menu.
* **Modern UI:** A clean, Nord-inspired dark theme.

## Prerequisites

* **Python 3.7+**
* **Ollama:** Ensure Ollama is installed and running on your system. You can download it from [ollama.com](https://ollama.com/).
* **Ollama Models:** Pull some models using the Ollama CLI (e.g., `ollama pull llama3`, `ollama pull llava` for vision).

## Installation

1.  **Clone the repository (or download the script):**
    ```bash
    git clone https://github.com/lalomorales22/ollama-widget-app.git
    cd ollama-widget-app
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For speech recognition, `PyAudio` is a common dependency for microphone access. If you encounter issues, you might need to install system-level audio libraries like `portaudio` (`sudo apt-get install portaudio19-dev` on Debian/Ubuntu, `brew install portaudio` on macOS).*

## Usage

1.  **Ensure Ollama is running:**


2.  **Run the application:**
    ```bash
    python app.py
    ```


3.  **Interacting with the App:**
    * Use the "➕ New Chat" button to start new conversations.
    * Right-click on a tab to rename or delete the chat permanently.
    * Use the model dropdown in each chat tab to select the Ollama model.
    * Click the "📎" (paperclip) button to attach images (for vision) or documents (for RAG).
    * Click the "🎤" (microphone) button to record audio for speech-to-text.
    * Go to "File" > "⚙️ Settings" to change the Ollama API URL if needed.
    * Go to "File" > "📤 Export Current Chat" to save a chat's history to a file.

## Contributing

Contributions, issues, and feature requests are welcome! Please feel free to check the [issues page](https://github.com/lalomorales22/ollama-widget-app/issues).

## License

This project is open-source. (Consider adding a LICENSE file, e.g., MIT License).
