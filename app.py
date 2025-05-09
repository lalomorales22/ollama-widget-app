import sys
import os
import base64
import json
import threading
import time # For timestamps
import sqlite3 # For database

# Attempt to import necessary libraries and provide guidance if missing
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTextEdit, QLineEdit, QPushButton, QLabel, QComboBox, QFileDialog,
        QTabWidget, QMessageBox, QStatusBar, QSizePolicy, QScrollArea,
        QDialog, QFormLayout, QDialogButtonBox, QInputDialog, QMenu
    )
    from PyQt5.QtGui import QFont, QPixmap, QImage, QColor, QPalette, QIcon, QTextCursor, QTextDocument, QTextImageFormat
    from PyQt5.QtCore import Qt, pyqtSignal, QThread, QSize, QUrl
except ImportError:
    print("PyQt5 not found. Please install it: pip install PyQt5")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("requests not found. Please install it: pip install requests")
    sys.exit(1)

try:
    import PyPDF2
except ImportError:
    print("PyPDF2 not found. Please install it: pip install PyPDF2")
    PyPDF2 = None

try:
    import speech_recognition as sr
    # You might need to install PyAudio for microphone access: pip install PyAudio
    # And for Whisper: pip install openai-whisper
except ImportError:
    print("SpeechRecognition not found. Please install it: pip install SpeechRecognition openai-whisper PyAudio")
    sr = None

try:
    from PIL import Image, ImageQt # Pillow
except ImportError:
    print("Pillow not found. Please install it: pip install Pillow")
    Image = None
    ImageQt = None


# --- Configuration ---
DEFAULT_OLLAMA_URL = "http://localhost:11434"
APP_TITLE = "Ollama Advanced Chat 🧠"
VERSION = "1.1.0" # Version updated
DATABASE_NAME = "ollama_chat.db"

# --- Helper Functions ---
def get_timestamp():
    """Returns a formatted timestamp."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def to_base64(image_path):
    """Converts an image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return None

# --- Database Manager ---
class DatabaseManager:
    def __init__(self, db_name=DATABASE_NAME):
        self.db_name = db_name
        self.conn = None
        self._connect()
        self._create_tables()

    def _connect(self):
        self.conn = sqlite3.connect(self.db_name)
        self.conn.row_factory = sqlite3.Row # Access columns by name
        # Enable Foreign Key support
        self.conn.execute("PRAGMA foreign_keys = ON;")


    def _create_tables(self):
        cursor = self.conn.cursor()
        # Conversations Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                ollama_model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Messages Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL, -- 'user', 'assistant', 'system'
                content TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                image_path TEXT,      -- Store path to image file
                rag_filename TEXT,    -- Store filename of RAG document
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)
        self.conn.commit()

    def add_conversation(self, name, ollama_model=None):
        cursor = self.conn.cursor()
        timestamp = get_timestamp()
        cursor.execute("""
            INSERT INTO conversations (name, ollama_model, created_at, last_accessed_at)
            VALUES (?, ?, ?, ?)
        """, (name, ollama_model, timestamp, timestamp))
        self.conn.commit()
        return cursor.lastrowid

    def rename_conversation(self, conversation_id, new_name):
        cursor = self.conn.cursor()
        cursor.execute("UPDATE conversations SET name = ? WHERE id = ?", (new_name, conversation_id))
        self.conn.commit()

    def delete_conversation(self, conversation_id):
        cursor = self.conn.cursor()
        # Foreign key ON DELETE CASCADE should handle messages
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        self.conn.commit()

    def get_all_conversations(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name, ollama_model, last_accessed_at FROM conversations ORDER BY last_accessed_at DESC")
        return cursor.fetchall()

    def add_message(self, conversation_id, role, content, timestamp, image_path=None, rag_filename=None):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO messages (conversation_id, role, content, timestamp, image_path, rag_filename)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (conversation_id, role, content, timestamp, image_path, rag_filename))
        self.conn.commit()
        self.update_conversation_accessed_time(conversation_id) # Update last accessed time
        return cursor.lastrowid

    def get_messages(self, conversation_id):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, role, content, timestamp, image_path, rag_filename
            FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC
        """, (conversation_id,))
        return cursor.fetchall()

    def update_conversation_accessed_time(self, conversation_id):
        cursor = self.conn.cursor()
        timestamp = get_timestamp()
        cursor.execute("UPDATE conversations SET last_accessed_at = ? WHERE id = ?", (timestamp, conversation_id))
        self.conn.commit()
        
    def update_conversation_model(self, conversation_id, model_name):
        cursor = self.conn.cursor()
        cursor.execute("UPDATE conversations SET ollama_model = ? WHERE id = ?", (model_name, conversation_id))
        self.conn.commit()
        self.update_conversation_accessed_time(conversation_id)


    def close(self):
        if self.conn:
            self.conn.close()

# --- Worker Threads (Unchanged from previous version) ---
class OllamaRequestThread(QThread):
    response_received = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    def __init__(self, ollama_url, model_name, messages, stream=False, images=None):
        super().__init__()
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.messages = messages # This is the history for Ollama API
        self.stream = stream
        self.images = images if images else []
    def run(self):
        try:
            payload = { "model": self.model_name, "messages": self.messages, "stream": self.stream }
            if self.images:
                if self.messages and self.messages[-1]["role"] == "user":
                    if not self.messages[-1].get("content"): self.messages[-1]["content"] = " "
                    self.messages[-1]["images"] = self.images
                else:
                    self.error_occurred.emit("Cannot send image without a user message context.")
                    return
            api_url = f"{self.ollama_url}/api/chat"
            response = requests.post(api_url, json=payload, timeout=120)
            response.raise_for_status()
            self.response_received.emit(response.json())
        except requests.exceptions.RequestException as e: self.error_occurred.emit(f"Network/API Error: {e}")
        except json.JSONDecodeError: self.error_occurred.emit("Error decoding JSON response.")
        except Exception as e: self.error_occurred.emit(f"Unexpected error: {e}")

class AudioRecognitionThread(QThread):
    transcription_ready = pyqtSignal(str)
    recognition_error = pyqtSignal(str)
    def __init__(self, recognizer, audio_source):
        super().__init__()
        self.recognizer = recognizer
        self.audio_source = audio_source
    def run(self):
        if not sr: self.recognition_error.emit("SpeechRec lib not available."); return
        try:
            self.recognition_error.emit("Listening...")
            # It's important to set a timeout for listen to avoid indefinite blocking
            # Also, phrase_time_limit can be useful.
            audio = self.recognizer.listen(self.audio_source, timeout=10, phrase_time_limit=30) 
            self.recognition_error.emit("Transcribing...")
            try: text = self.recognizer.recognize_whisper(audio) # Requires openai-whisper
            except sr.UnknownValueError: text = ""; self.recognition_error.emit("Whisper: no speech.")
            except sr.RequestError as e:
                self.recognition_error.emit(f"Whisper API error: {e}. Trying Sphinx.")
                try: text = self.recognizer.recognize_sphinx(audio)
                except sr.UnknownValueError: text = ""; self.recognition_error.emit("Sphinx: no speech.")
                except sr.RequestError as es: text = ""; self.recognition_error.emit(f"Sphinx error: {es}")
            except Exception as e_whisp_local: # Other Whisper errors (model not found, etc.)
                self.recognition_error.emit(f"Whisper local error: {e_whisp_local}. Trying Sphinx.")
                try: text = self.recognizer.recognize_sphinx(audio)
                except Exception as es_fallback: text = ""; self.recognition_error.emit(f"Sphinx fallback error: {es_fallback}")
            self.transcription_ready.emit(text)
        except sr.WaitTimeoutError: self.recognition_error.emit("No speech detected (timeout).")
        except Exception as e: self.recognition_error.emit(f"Audio recognition error: {e}")

# --- Chat Widget for each Tab ---
class ChatWidget(QWidget):
    status_update_requested = pyqtSignal(str, int)

    def __init__(self, ollama_url, available_models, db_manager, conversation_id, parent=None): # Added db_manager, conversation_id
        super().__init__(parent)
        self.ollama_url = ollama_url
        self.available_models = available_models
        self.db_manager = db_manager
        self.conversation_id = conversation_id
        
        self.messages_for_ollama_api = []  # Stores history for Ollama API (role, content, images)
        
        self.current_rag_text = None
        self.current_rag_filename = None
        self.current_image_path = None 
        self.current_image_base64 = None

        self.ollama_thread = None
        self.audio_thread = None
        self.recognizer = sr.Recognizer() if sr else None
        if self.recognizer:
            self.recognizer.pause_threshold = 0.8 # seconds of non-speaking audio before a phrase is considered complete

        self._init_ui()
        self.load_history_from_db() # Load history when widget is created

        # Connect model dropdown change to DB update
        self.model_dropdown.currentTextChanged.connect(self.on_model_changed)


    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems([m.get("name", "N/A") for m in self.available_models] if self.available_models else ["No models found"])
        self.model_dropdown.setToolTip("Select the Ollama model for this chat.")
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_dropdown)
        layout.addLayout(model_layout)
        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setFont(QFont("Inter", 11))
        self.chat_area.setStyleSheet("background-color: #2E3440; color: #ECEFF4; border-radius: 5px; padding: 5px;")
        layout.addWidget(self.chat_area)
        self.context_label = QLabel("Context: None")
        self.context_label.setStyleSheet("font-size: 9pt; color: #A3BE8C;")
        self.context_label.setFixedHeight(20)
        layout.addWidget(self.context_label)
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message or use mic/attach...")
        self.input_field.setFont(QFont("Inter", 10))
        self.input_field.setStyleSheet("background-color: #3B4252; color: #ECEFF4; border-radius: 5px; padding: 8px;")
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field)
        self.send_button = QPushButton("➤")
        self.send_button.setToolTip("Send Message (Enter)")
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setStyleSheet("QPushButton { background-color: #5E81AC; color: #ECEFF4; border-radius: 5px; padding: 8px; font-weight: bold; } QPushButton:hover { background-color: #81A1C1; }")
        input_layout.addWidget(self.send_button)
        self.attach_button = QPushButton("📎")
        self.attach_button.setToolTip("Attach File (Image for Vision, PDF/TXT for RAG)")
        self.attach_button.clicked.connect(self.handle_attachment)
        self.attach_button.setStyleSheet("QPushButton { background-color: #A3BE8C; color: #2E3440; border-radius: 5px; padding: 8px; font-weight: bold; } QPushButton:hover { background-color: #B4D0A0; }")
        input_layout.addWidget(self.attach_button)
        self.record_button = QPushButton("🎤")
        self.record_button.setToolTip("Record Audio (Whisper/Sphinx)")
        if not sr: self.record_button.setEnabled(False); self.record_button.setToolTip("SpeechRecognition lib not available.")
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setStyleSheet("QPushButton { background-color: #EBCB8B; color: #2E3440; border-radius: 5px; padding: 8px; font-weight: bold; } QPushButton:hover { background-color: #F0D8A0; }")
        input_layout.addWidget(self.record_button)
        layout.addLayout(input_layout)
        self.setLayout(layout)

    def on_model_changed(self, model_name):
        if self.conversation_id and model_name and model_name != "No models found":
            self.db_manager.update_conversation_model(self.conversation_id, model_name)
            self.status_update_requested.emit(f"Model for this chat set to {model_name}.", 2000)


    def _add_message_to_chat_display(self, sender, message_text, msg_timestamp, image_display_path=None):
        """Adds a message to the chat QTextEdit display area with styling and optional image."""
        # msg_timestamp can be a string from DB or a new one via get_timestamp()
        formatted_message_prefix = ""
        bubble_style = ""

        if sender.lower() == "user":
            formatted_message_prefix = f"<div style='text-align:right; margin-bottom: 8px;'><span style='color:#88C0D0; font-weight:bold;'>You ({msg_timestamp}):</span><br><div style='background-color:#3B4252; color:#D8DEE9; padding: 8px; border-radius: 8px 0px 8px 8px; display:inline-block; max-width: 70%; text-align:left;'>"
        elif sender.lower() == "ollama":
            formatted_message_prefix = f"<div style='text-align:left; margin-bottom: 8px;'><span style='color:#A3BE8C; font-weight:bold;'>Ollama ({msg_timestamp}):</span><br><div style='background-color:#434C5E; color:#E5E9F0; padding: 8px; border-radius: 0px 8px 8px 8px; display:inline-block; max-width: 70%; text-align:left;'>"
        else: # System messages
            system_message_html = f"<div style='text-align:center; margin-bottom: 8px;'><span style='color:#BF616A; font-style:italic;'>System ({msg_timestamp}): {message_text}</span></div>"
            self.chat_area.append(system_message_html)
            self.chat_area.moveCursor(QTextCursor.End)
            return

        # Escape HTML special characters for plain text messages
        escaped_text = message_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        message_content_html = escaped_text

        if image_display_path:
            try:
                q_img = QImage(image_display_path)
                if not q_img.isNull():
                    # Unique resource name for the image in QTextDocument
                    unique_resource_name = f"image_{os.path.basename(image_display_path)}_{int(time.time()*10000)}"
                    self.chat_area.document().addResource(QTextDocument.ImageResource, QUrl(unique_resource_name), q_img)
                    message_content_html += f"<br><img src='{unique_resource_name}' width='150'>" # Increased width
                else:
                    message_content_html += f"<br><small style='color:#D08770;'>(Image not found/loadable: {os.path.basename(image_display_path)})</small>"
            except Exception as e_img:
                print(f"Error preparing image for display {image_display_path}: {e_img}")
                message_content_html += f"<br><small style='color:#D08770;'>(Error displaying image: {os.path.basename(image_display_path)})</small>"
        
        formatted_message_suffix = "</div></div>" # Close the inner bubble and outer div
        full_html_message = formatted_message_prefix + message_content_html + formatted_message_suffix
        
        self.chat_area.append(full_html_message)
        self.chat_area.moveCursor(QTextCursor.End)

    def send_message(self):
        user_text = self.input_field.text().strip()
        if not user_text and not self.current_image_base64:
            self.status_update_requested.emit("Cannot send an empty message.", 3000); return

        msg_timestamp = get_timestamp()
        
        # Determine content for display and API
        display_text = user_text if user_text else "[Image sent]"
        api_content = user_text if user_text else "Describe this image." # Default prompt for image-only

        # Add to UI
        self._add_message_to_chat_display("User", display_text, msg_timestamp, image_display_path=self.current_image_path)

        # Prepare message for Ollama API history
        message_for_api = {"role": "user", "content": api_content}
        if self.current_rag_text: # RAG context
            message_for_api["content"] = f"Using the following document context:\n---\n{self.current_rag_text}\n---\n\nUser question: {api_content}"
        
        # Add to DB (before sending to Ollama, so it's saved even if API fails)
        self.db_manager.add_message(
            self.conversation_id, "user", api_content, msg_timestamp, # Store API content
            image_path=self.current_image_path, 
            rag_filename=self.current_rag_filename
        )
        
        # Add to in-memory Ollama history (self.messages_for_ollama_api)
        # This history should be rebuilt from DB or kept in sync carefully.
        # For simplicity, let's rebuild it before each call based on what Ollama expects.
        # The `get_ollama_formatted_history` method will handle this.
        
        self.messages_for_ollama_api = self.get_ollama_formatted_history() # Get latest history
        self.messages_for_ollama_api.append(message_for_api) # Add current message

        self.input_field.clear()
        model_name = self.model_dropdown.currentText()
        if model_name == "No models found" or not model_name:
            self._add_message_to_chat_display("System", "No Ollama model selected.", get_timestamp())
            # No need to pop from DB, just don't send
            return

        self.status_update_requested.emit(f"Sending to {model_name}...", 0)
        self.send_button.setEnabled(False); self.record_button.setEnabled(False)

        images_payload = [self.current_image_base64] if self.current_image_base64 else None
        if images_payload: # If sending an image, add it to the last message in API history
            self.messages_for_ollama_api[-1]["images"] = images_payload


        self.ollama_thread = OllamaRequestThread(self.ollama_url, model_name, self.messages_for_ollama_api, images=None) # Images are now part of messages
        self.ollama_thread.response_received.connect(self.handle_ollama_response)
        self.ollama_thread.error_occurred.connect(self.handle_ollama_error)
        self.ollama_thread.finished.connect(self._on_ollama_thread_finished)
        self.ollama_thread.start()

        self.current_image_path = None; self.current_image_base64 = None # Clear after sending
        # RAG context (self.current_rag_text, self.current_rag_filename) persists until explicitly changed/cleared.
        self._update_context_label()

    def get_ollama_formatted_history(self):
        """ Retrieves messages from DB and formats them for Ollama API """
        db_messages = self.db_manager.get_messages(self.conversation_id)
        ollama_history = []
        for msg_row in db_messages:
            entry = {"role": msg_row["role"], "content": msg_row["content"]}
            # Note: Ollama API expects 'images' as a list of base64 strings
            # with the *user* message that introduces them.
            # This simplified history retrieval doesn't re-fetch base64 for past images.
            # For true multi-turn vision, the API message list needs careful construction.
            # For now, images are only sent with the *current* user message.
            ollama_history.append(entry)
        return ollama_history


    def handle_ollama_response(self, response_data):
        ai_reply = response_data.get("message", {}).get("content", "No proper response.")
        msg_timestamp = get_timestamp()
        
        # Add to UI
        self._add_message_to_chat_display("Ollama", ai_reply, msg_timestamp)
        
        # Add to DB
        self.db_manager.add_message(self.conversation_id, "assistant", ai_reply, msg_timestamp)
        
        # Update in-memory API history
        self.messages_for_ollama_api.append({"role": "assistant", "content": ai_reply})
        
        self.status_update_requested.emit(f"Response from {self.model_dropdown.currentText()}.", 3000)

    def handle_ollama_error(self, error_message):
        timestamp = get_timestamp()
        self._add_message_to_chat_display("System", f"Error: {error_message}", timestamp)
        # We don't add this system error to DB usually, but could if needed for audit.
        self.status_update_requested.emit(f"Error: {error_message}", 5000)

    def _on_ollama_thread_finished(self):
        self.send_button.setEnabled(True)
        if sr: self.record_button.setEnabled(True)
        self.ollama_thread = None

    def handle_attachment(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Attach File", "", "All Files (*.png *.jpg *.jpeg *.pdf *.txt);;Images (*.png *.jpg *.jpeg);;PDF (*.pdf);;Text (*.txt)")
        if not file_path: return

        file_ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)
        timestamp = get_timestamp()

        if file_ext in ['.png', '.jpg', '.jpeg']:
            # For simplicity, let's copy image to an app-specific data dir if we want to make paths more robust
            # For now, just use original path. User must not move/delete it.
            self.current_image_base64 = to_base64(file_path)
            if self.current_image_base64:
                self.current_image_path = file_path # Store original path
                self.current_rag_text = None; self.current_rag_filename = None # Clear RAG
                self._add_message_to_chat_display("System", f"Image '{filename}' loaded. Will be sent with next message.", timestamp)
                self.status_update_requested.emit(f"Image '{filename}' loaded.", 3000)
            else:
                self._add_message_to_chat_display("System", f"Failed to load image '{filename}'.", timestamp)
        elif file_ext == '.pdf' and PyPDF2:
            try:
                text = ""
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages: text += (page.extract_text() or "") + "\n"
                self.current_rag_text = text.strip(); self.current_rag_filename = filename
                self.current_image_path = None; self.current_image_base64 = None # Clear vision
                self._add_message_to_chat_display("System", f"PDF '{filename}' loaded for RAG.", timestamp)
            except Exception as e: self._add_message_to_chat_display("System", f"Error reading PDF '{filename}': {e}", timestamp)
        elif file_ext == '.txt':
            try:
                with open(file_path, "r", encoding='utf-8') as f: self.current_rag_text = f.read().strip()
                self.current_rag_filename = filename
                self.current_image_path = None; self.current_image_base64 = None
                self._add_message_to_chat_display("System", f"Text file '{filename}' loaded for RAG.", timestamp)
            except Exception as e: self._add_message_to_chat_display("System", f"Error reading TXT '{filename}': {e}", timestamp)
        else:
            self._add_message_to_chat_display("System", f"Unsupported file: {filename}", timestamp)
        self._update_context_label()

    def _update_context_label(self):
        if self.current_image_path: self.context_label.setText(f"Context: Image - {os.path.basename(self.current_image_path)}"); self.context_label.setStyleSheet("font-size: 9pt; color: #8FBCBB;")
        elif self.current_rag_filename: self.context_label.setText(f"Context: Document - {self.current_rag_filename}"); self.context_label.setStyleSheet("font-size: 9pt; color: #EBCB8B;")
        else: self.context_label.setText("Context: None"); self.context_label.setStyleSheet("font-size: 9pt; color: #A3BE8C;")

    def toggle_recording(self):
        if not self.recognizer: self._add_message_to_chat_display("System", "Speech recognition not initialized.", get_timestamp()); return
        if self.audio_thread and self.audio_thread.isRunning(): self.status_update_requested.emit("Recording in progress...", 2000); return
        try:
            self.status_update_requested.emit("Starting mic... Grant access if prompted.", 0)
            # Use a context manager for sr.Microphone to ensure it's closed
            # This part is tricky as sr.Microphone() itself can raise errors if no mic
            try:
                mic_test = sr.Microphone() # Test if mic can be opened
                mic_test = None # release it
            except Exception as e_mic:
                 self._add_message_to_chat_display("System", f"Microphone error: {e_mic}. Check connection/permissions.", get_timestamp())
                 self.status_update_requested.emit(f"Mic error: {e_mic}", 5000)
                 return

            # The actual listen needs to be in the thread.
            # We pass the recognizer instance and tell the thread to create its own mic source.
            # This is safer than passing an open mic object across threads.
            # However, the provided AudioRecognitionThread expects an audio_source.
            # Let's adjust AudioRecognitionThread to open/close mic internally or simplify.

            # For now, keeping original structure, assuming sr.Microphone() can be passed if done carefully.
            # A robust way: AudioRecognitionThread handles its own sr.Microphone() context.
            # Let's modify AudioRecognitionThread slightly for this.
            self.audio_thread = AudioRecognitionThread(self.recognizer, None) # Pass None, thread will manage mic
            self.audio_thread.transcription_ready.connect(self.handle_transcription)
            self.audio_thread.recognition_error.connect(self.handle_recognition_error)
            self.audio_thread.finished.connect(self._on_audio_thread_finished)
            self.record_button.setText("..."); self.record_button.setEnabled(False); self.send_button.setEnabled(False)
            self.audio_thread.start()
        except Exception as e:
            self._add_message_to_chat_display("System", f"Could not start recording: {e}", get_timestamp())
            self.status_update_requested.emit(f"Recording Error: {e}", 5000)
            self._on_audio_thread_finished() # Reset buttons

    def handle_transcription(self, text):
        timestamp = get_timestamp()
        if text:
            self.input_field.setText(f"{self.input_field.text()} {text}".strip())
            self._add_message_to_chat_display("System", f"Transcription: {text}", timestamp) # Log transcription
        else: self._add_message_to_chat_display("System", "No speech detected or transcription failed.", timestamp)
        self.record_button.setText("🎤")

    def handle_recognition_error(self, error_message):
        if "Listening..." in error_message or "Transcribing..." in error_message: self.status_update_requested.emit(error_message, 0)
        else:
            self._add_message_to_chat_display("System", f"Recognition Error: {error_message}", get_timestamp())
            self.status_update_requested.emit(f"Recognition Error: {error_message}", 4000)
            self.record_button.setText("🎤")

    def _on_audio_thread_finished(self):
        self.record_button.setText("🎤"); self.record_button.setEnabled(True); self.send_button.setEnabled(True)
        self.audio_thread = None

    def load_history_from_db(self):
        self.chat_area.clear()
        self.messages_for_ollama_api = [] # Reset API history
        db_messages = self.db_manager.get_messages(self.conversation_id)
        
        current_model_in_db = None
        if db_messages: # Try to get model from last assistant message or conversation table
            conv_details = self.db_manager.conn.execute("SELECT ollama_model FROM conversations WHERE id = ?", (self.conversation_id,)).fetchone()
            if conv_details and conv_details['ollama_model']:
                current_model_in_db = conv_details['ollama_model']

        if current_model_in_db:
            index = self.model_dropdown.findText(current_model_in_db, Qt.MatchFixedString)
            if index >= 0: self.model_dropdown.setCurrentIndex(index)
        
        for msg_row in db_messages:
            # Add to UI display
            self._add_message_to_chat_display(
                msg_row["role"], 
                msg_row["content"], 
                msg_row["timestamp"], 
                image_display_path=msg_row["image_path"]
            )
            # Rebuild API history (simplified, no images for past messages here)
            self.messages_for_ollama_api.append({"role": msg_row["role"], "content": msg_row["content"]})
        self.status_update_requested.emit("Chat history loaded.", 1500)


# --- Main Application Window ---
class OllamaApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ollama_url = DEFAULT_OLLAMA_URL
        self.available_models = []
        self.db_manager = DatabaseManager() # Initialize DB Manager

        self._init_ui()
        self._apply_stylesheet()
        self._load_ollama_models() # This will also trigger loading chats from DB if models load

    def _init_ui(self):
        self.setWindowTitle(f"{APP_TITLE} - v{VERSION}")
        self.setGeometry(100, 100, 750, 850) # Slightly larger
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.setMovable(True) # Allow reordering tabs
        self.tab_widget.tabCloseRequested.connect(self.close_tab_view) # Just closes view
        self.tab_widget.setContextMenuPolicy(Qt.CustomContextMenu) # For right-click menu
        self.tab_widget.customContextMenuRequested.connect(self.show_tab_context_menu)
        main_layout.addWidget(self.tab_widget)
        self.add_tab_button = QPushButton("➕ New Chat")
        self.add_tab_button.clicked.connect(lambda: self.add_new_chat_tab_action())
        self.add_tab_button.setFixedHeight(30)
        main_layout.addWidget(self.add_tab_button, alignment=Qt.AlignRight)
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        settings_action = file_menu.addAction("⚙️ &Settings"); settings_action.triggered.connect(self.open_settings_dialog)
        export_chat_action = file_menu.addAction("📤 &Export Current Chat")
        export_chat_action.triggered.connect(self.export_current_chat)
        # Import chat might be complex if it needs to merge with DB or create new conv.
        # For now, focusing on DB as primary.
        file_menu.addSeparator()
        exit_action = file_menu.addAction("🚪 &Exit"); exit_action.triggered.connect(self.close)
        help_menu = menubar.addMenu("&Help")
        about_action = help_menu.addAction("ℹ️ &About"); about_action.triggered.connect(self.show_about_dialog)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready.", 3000)

    def _apply_stylesheet(self): # Using previous stylesheet, ensure it's fine
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #2E3440; color: #D8DEE9; font-family: Inter, sans-serif; }
            QTabWidget::pane { border-top: 2px solid #4C566A; }
            QTabBar::tab { background: #3B4252; color: #D8DEE9; border: 1px solid #4C566A; border-bottom-color: #3B4252; padding: 8px 15px; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #434C5E; color: #ECEFF4; border-bottom-color: #434C5E; }
            QTabBar::tab:hover { background: #4C566A; }
            /* QTabBar::close-button styling might be OS dependent or need specific icons */
            QPushButton { background-color: #5E81AC; color: #ECEFF4; border: none; padding: 8px 12px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #81A1C1; } QPushButton:pressed { background-color: #4C566A; }
            QLineEdit, QTextEdit, QComboBox { background-color: #3B4252; color: #D8DEE9; border: 1px solid #4C566A; border-radius: 4px; padding: 6px; }
            QComboBox::drop-down { border: none; }
            QLabel { color: #D8DEE9; } QStatusBar { background-color: #3B4252; color: #ECEFF4; }
            QMenuBar { background-color: #3B4252; color: #ECEFF4; }
            QMenuBar::item { background-color: #3B4252; color: #ECEFF4; padding: 4px 8px; }
            QMenuBar::item:selected { background-color: #4C566A; }
            QMenu { background-color: #3B4252; color: #ECEFF4; border: 1px solid #4C566A; }
            QMenu::item:selected { background-color: #5E81AC; }
        """)

    def _load_ollama_models(self):
        self.status_bar.showMessage("Fetching Ollama models...", 0)
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            response.raise_for_status(); data = response.json()
            self.available_models = data.get("models", [])
            if not self.available_models:
                self.status_bar.showMessage("No Ollama models. Check Ollama.", 5000)
                QMessageBox.warning(self, "Ollama Models", "No models found. Ensure Ollama is running and models are pulled.")
            else:
                self.status_bar.showMessage(f"{len(self.available_models)} models loaded.", 3000)
            self.load_conversations_from_db() # Load chats after models are known
        except requests.exceptions.ConnectionError:
            self.status_bar.showMessage("Ollama connection error.", 5000)
            QMessageBox.critical(self, "Ollama Error", f"Could not connect to Ollama at {self.ollama_url}.")
            self.load_conversations_from_db() # Still try to load chats, they might use a model not currently available
        except Exception as e:
            self.status_bar.showMessage(f"Error fetching models: {e}", 5000)
            self.load_conversations_from_db()


    def load_conversations_from_db(self):
        """Loads all conversations from DB and creates/updates tabs."""
        # Clear existing tabs before loading (or implement more complex tab matching)
        # For simplicity now, clear and reload. This might lose unsaved state in non-DB parts.
        # while self.tab_widget.count() > 0:
        #     self.tab_widget.removeTab(0)

        conversations = self.db_manager.get_all_conversations()
        if not conversations and not self.available_models: # No chats and no models
             self.status_bar.showMessage("No chats in DB and no models from Ollama.", 3000)
        elif not conversations and self.available_models: # No chats but models exist
             self.add_new_chat_tab_action("Default Chat") # Create one default chat
        else:
            for conv_row in conversations:
                self.add_chat_tab_from_db(conv_row['id'], conv_row['name'], conv_row['ollama_model'])
        
        if self.tab_widget.count() == 0 and self.available_models: # If DB was empty but models exist
            self.add_new_chat_tab_action("Default Chat")


    def add_chat_tab_from_db(self, conversation_id, name, ollama_model):
        """Adds a tab for an existing conversation from the database."""
        chat_widget = ChatWidget(self.ollama_url, self.available_models, self.db_manager, conversation_id, self)
        chat_widget.status_update_requested.connect(self.update_status_bar)
        
        index = self.tab_widget.addTab(chat_widget, name)
        self.tab_widget.setCurrentIndex(index)
        
        # Set model in dropdown if it was stored
        if ollama_model:
            model_idx = chat_widget.model_dropdown.findText(ollama_model, Qt.MatchFixedString)
            if model_idx >= 0:
                chat_widget.model_dropdown.setCurrentIndex(model_idx)
            else: # Model stored in DB not in current available_models list
                chat_widget.model_dropdown.addItem(f"{ollama_model} (not found)") # Add it as a placeholder
                chat_widget.model_dropdown.setCurrentText(f"{ollama_model} (not found)")


    def add_new_chat_tab_action(self, name=None):
        """Action to create a new chat, save to DB, and open tab."""
        if not self.available_models and self.ollama_url == DEFAULT_OLLAMA_URL:
            reply = QMessageBox.question(self, "No Models Loaded", "Ollama models haven't loaded. Continue to create chat anyway?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No: return

        default_model = self.available_models[0]['name'] if self.available_models else None
        
        self.chat_sessions_count = self.db_manager.conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        tab_name = name if name else f"Chat {self.chat_sessions_count + 1}"
        
        # Create conversation in DB first
        conversation_id = self.db_manager.add_conversation(tab_name, default_model)
        if conversation_id:
            self.add_chat_tab_from_db(conversation_id, tab_name, default_model) # Load it like any other DB chat
            self.status_bar.showMessage(f"New chat '{tab_name}' created.", 2000)
        else:
            QMessageBox.critical(self, "Database Error", "Could not create new conversation in the database.")


    def close_tab_view(self, index):
        """Closes the tab at the given index (view only, data remains in DB)."""
        widget_to_close = self.tab_widget.widget(index)
        if widget_to_close:
            self.tab_widget.removeTab(index)
            widget_to_close.deleteLater() 

    def show_tab_context_menu(self, point):
        tab_bar = self.tab_widget.tabBar()
        tab_index = tab_bar.tabAt(point)
        if tab_index == -1: return

        menu = QMenu(self)
        rename_action = menu.addAction("✏️ Rename Chat")
        delete_action = menu.addAction("🗑️ Delete Chat Permanently")
        
        action = menu.exec_(tab_bar.mapToGlobal(point))

        if action == rename_action: self.rename_chat_tab(tab_index)
        elif action == delete_action: self.delete_chat_tab_permanently(tab_index)

    def rename_chat_tab(self, index):
        current_name = self.tab_widget.tabText(index)
        chat_widget = self.tab_widget.widget(index)
        if not isinstance(chat_widget, ChatWidget): return

        conversation_id = chat_widget.conversation_id
        new_name, ok = QInputDialog.getText(self, "Rename Chat", "Enter new name:", QLineEdit.Normal, current_name)
        if ok and new_name and new_name != current_name:
            self.tab_widget.setTabText(index, new_name)
            self.db_manager.rename_conversation(conversation_id, new_name)
            self.status_bar.showMessage(f"Chat renamed to '{new_name}'.", 3000)

    def delete_chat_tab_permanently(self, index):
        current_name = self.tab_widget.tabText(index)
        chat_widget = self.tab_widget.widget(index)
        if not isinstance(chat_widget, ChatWidget): return
        
        conversation_id = chat_widget.conversation_id
        reply = QMessageBox.question(self, 'Delete Chat Permanently',
                                     f"Delete '{current_name}' and all its messages from the database? This cannot be undone.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.db_manager.delete_conversation(conversation_id)
            self.tab_widget.removeTab(index) # This also calls widget.deleteLater()
            self.status_bar.showMessage(f"Chat '{current_name}' deleted permanently.", 3000)

    def update_status_bar(self, message, timeout=3000):
        if timeout == 0: self.status_bar.showMessage(message)
        else: self.status_bar.showMessage(message, timeout)

    def open_settings_dialog(self):
        dialog = QDialog(self); dialog.setWindowTitle("⚙️ Settings"); dialog.setMinimumWidth(350)
        layout = QFormLayout(dialog)
        self.ollama_url_input = QLineEdit(self.ollama_url)
        layout.addRow("Ollama API URL:", self.ollama_url_input)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept); button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        if dialog.exec_() == QDialog.Accepted:
            new_url = self.ollama_url_input.text().strip()
            if new_url != self.ollama_url:
                self.ollama_url = new_url
                self.status_bar.showMessage(f"Ollama URL updated. Reloading models...", 5000)
                self._load_ollama_models()
                for i in range(self.tab_widget.count()): # Update existing chat widgets
                    cw = self.tab_widget.widget(i)
                    if isinstance(cw, ChatWidget): cw.ollama_url = self.ollama_url

    def export_current_chat(self):
        current_widget = self.tab_widget.currentWidget()
        if not isinstance(current_widget, ChatWidget): QMessageBox.information(self, "Export Chat", "No active chat to export."); return

        conversation_id = current_widget.conversation_id
        chat_name = self.tab_widget.tabText(self.tab_widget.currentIndex())
        messages = self.db_manager.get_messages(conversation_id) # Get messages from DB

        if not messages: QMessageBox.information(self, "Export Chat", "Chat history is empty."); return

        file_path, _ = QFileDialog.getSaveFileName(self, f"Export Chat '{chat_name}'", f"{chat_name}.json", "JSON Files (*.json);;Text Files (*.txt)")
        if not file_path: return

        try:
            history_for_export = [{"role": m["role"], "content": m["content"], "timestamp": m["timestamp"], 
                                   "image_path": m["image_path"], "rag_filename": m["rag_filename"]} for m in messages]
            if file_path.endswith(".json"):
                with open(file_path, 'w', encoding='utf-8') as f: json.dump(history_for_export, f, indent=2)
            elif file_path.endswith(".txt"):
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"Chat History: {chat_name} - Exported: {get_timestamp()}\n")
                    f.write(f"Model (last used in tab): {current_widget.model_dropdown.currentText()}\n\n")
                    for message in history_for_export:
                        f.write(f"[{message['timestamp']}] {message['role'].capitalize()}:\n{message['content']}\n")
                        if message['image_path']: f.write(f"(Image: {os.path.basename(message['image_path'])})\n")
                        if message['rag_filename']: f.write(f"(RAG Doc: {message['rag_filename']})\n")
                        f.write("\n---\n\n")
            self.status_bar.showMessage(f"Chat exported to {os.path.basename(file_path)}", 3000)
        except Exception as e: QMessageBox.critical(self, "Export Error", f"Could not export chat: {e}")


    def show_about_dialog(self):
        QMessageBox.about(self, f"About {APP_TITLE}", f"<h2>{APP_TITLE} v{VERSION}</h2><p>Multi-conversation Ollama client with Vision, RAG, Speech-to-Text, and SQLite persistence.</p><p>Ollama URL: {self.ollama_url}</p><p>Database: {os.path.abspath(DATABASE_NAME)}</p>")

    def closeEvent(self, event):
        # DB connection is closed when db_manager is garbage collected or explicitly if needed
        # self.db_manager.close() # Ensure DB is closed cleanly
        reply = QMessageBox.question(self, 'Exit Application', "Are you sure you want to exit?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes: event.accept()
        else: event.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Consider setting app icon here
    # app.setWindowIcon(QIcon("path/to/your/icon.png"))
    main_window = OllamaApp()
    main_window.show()
    exit_code = app.exec_()
    main_window.db_manager.close() # Explicitly close DB connection on exit
    sys.exit(exit_code)

