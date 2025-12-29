# CSM (Conversational Speech Model) - Complete Setup & Documentation

## 📋 Overview

CSM is an advanced **Speech Generation Model** developed by Sesame that converts text and audio inputs into natural-sounding conversational speech. It uses a Llama backbone combined with a Mimi audio decoder to generate high-quality RVQ (Residual Vector Quantized) audio codes.

This project includes:
- **`run_csm.py`** - Standalone demo for generating multi-speaker conversations
- **`server_voice.py`** - FastAPI-based voice chat server with AI-powered responses
- **`voice.html`** - Browser interface for real-time voice interaction

---

## 🛠️ Setup Instructions Using Conda

### Prerequisites
- **Windows/Linux/Mac** with PowerShell, Command Prompt, or Bash
- **CUDA 12.4+** compatible GPU (NVIDIA recommended)
- **Conda** installed (Download from [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/projects/miniconda/releases.html))
- **FFmpeg** (optional but recommended for audio operations)
- **Hugging Face Account** with access to:
  - `meta-llama/Llama-3.2-1B-Instruct`
  - `sesame/csm-1b`

### Step 1: Create Conda Environment

```bash
# Open terminal/PowerShell and navigate to the project folder
cd c:\Users\info\Documents\charangit\csm

# Create a new conda environment with Python 3.10
conda create -n csm-env python=3.10 -y

# Activate the environment
conda activate csm-env
```

### Step 2: Install Dependencies

```bash
# Upgrade pip for better package resolution
pip install --upgrade pip

# Install requirements from requirements.txt
pip install -r requirements.txt

# Windows-specific: Install triton-windows instead of triton
pip install triton-windows

# Install Hugging Face CLI for model authentication
pip install huggingface-hub

# Optional: Install FFmpeg via conda
conda install ffmpeg -c conda-forge -y
```

### Step 3: Authenticate with Hugging Face

```bash
# Login to Hugging Face to access gated models
huggingface-cli login

# You'll be prompted to paste your HF token from:
# https://huggingface.co/settings/tokens
```

### Step 4: Disable Torch Compilation (Required)

```bash
# This is crucial for the Mimi audio tokenizer to work properly
# Set environment variable to disable lazy compilation
set NO_TORCH_COMPILE=1        # Windows Command Prompt
$env:NO_TORCH_COMPILE="1"     # Windows PowerShell
export NO_TORCH_COMPILE=1     # Linux/Mac Bash
```

### Step 5: Verify Installation

```bash
# Test by running a simple Python import check
python -c "import torch, torchaudio, fastapi, transformers; print('All packages installed!')"
```

---

## ▶️ Running the Code

### Option 1: Standalone Demo - `run_csm.py`

Generates a multi-speaker conversation without a server or UI.

```bash
# Make sure the environment is active
conda activate csm-env

# Disable Torch compilation (if not already set globally)
set NO_TORCH_COMPILE=1  # Windows
export NO_TORCH_COMPILE=1  # Linux/Mac

# Run the demo
python run_csm.py
```

**What it does:**
1. Loads the CSM model onto GPU/CPU
2. Downloads speaker prompt audio samples from Hugging Face
3. Generates a 4-turn conversation between 2 speakers
4. Saves the full conversation as audio files
5. Prints transcriptions to console

**Output:**
- Console logs showing each generated utterance
- Audio files saved to disk (conversation_log.wav or similar)

---

### Option 2: Interactive Voice Chat Server - `server_voice.py`

Launches a FastAPI server with a web interface for real-time voice conversations.

```bash
# Make sure the environment is active
conda activate csm-env

# Set environment variable
set NO_TORCH_COMPILE=1  # Windows
export NO_TORCH_COMPILE=1  # Linux/Mac

# Start the server
python server_voice.py
```

**Server Info:**
- **URL:** `http://localhost:8000`
- **Endpoint:** `POST /talk` (processes voice input)
- **Endpoint:** `GET /` (serves voice.html interface)

**Output:**
- Running server on `0.0.0.0:8000`
- Console logs for each interaction
- `conversation_log.wav` file updated with each exchange

Open your browser to `http://localhost:8000` to start chatting!

---

## 🧠 Component Explanation

### 1. **`server_voice.py`** - Voice Chat Server

This is the main application that orchestrates three AI models working together:

#### **Architecture Overview:**
```
User Microphone
    ↓
voice.html (Browser Recording)
    ↓
POST /talk (Server)
    ↓
┌─────────────────────────────────────┐
│  PIPELINE (server_voice.py)         │
├─────────────────────────────────────┤
│ 1. STT (Speech-to-Text)             │
│    Model: Whisper Tiny              │
│    Input: WAV audio from mic        │
│    Output: Text transcription       │
│                                     │
│ 2. LLM (Text Generation)            │
│    Model: Llama-3.2-1B-Instruct     │
│    Input: Chat history + new text   │
│    Output: AI response text         │
│                                     │
│ 3. TTS (Text-to-Speech)             │
│    Model: CSM-1B                    │
│    Input: AI text + speaker voice   │
│    Output: Audio WAV                │
└─────────────────────────────────────┘
    ↓
Browser Audio Playback
    ↓
Speaker
```

#### **Detailed Component Breakdown:**

##### **Global Variables & State Management:**

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Where models run (GPU/CPU)

CONVERSATION_FILE = "conversation_log.wav"
# Saves full conversation history

MAX_CONTEXT = 3
# Keeps last 3 turns for voice style consistency

history_text = [{"role": "system", "content": "..."}]
# Chat history for LLM (system prompt + turn-by-turn messages)

history_audio_segments = []
# Audio context for CSM to maintain speaker voice characteristics

full_audio_buffer = []
# All audio generated so far, saved to disk
```

##### **A. Speech-to-Text (STT) - Whisper Model:**

```python
stt_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny.en",  # Can upgrade to 'base.en' or 'small.en'
    device=DEVICE
)
```

**What it does:**
- Converts user's voice recording into text
- Input: WAV audio from microphone (any sample rate)
- Output: String transcription (e.g., "Hey how are you doing?")
- Model options:
  - `whisper-tiny.en` - Fastest, lower accuracy (~1-2s processing)
  - `whisper-base.en` - Balanced (slower but more accurate)
  - `whisper-small.en` - High accuracy (slowest)

**Processing steps in `/talk` endpoint:**
1. Receive audio bytes from client
2. Load as tensor with `torchaudio.load()`
3. Resample to 16kHz if needed (Whisper requirement)
4. Run through STT pipeline
5. Extract text from result

##### **B. Language Model (LLM) - Llama 3.2:**

```python
chat_pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,  # Memory efficient
    device_map=DEVICE,
)
```

**What it does:**
- Generates intelligent responses based on conversation history
- Input: Chat history (list of dicts with role/content)
- Output: Natural language response
- Maintains context from previous turns

**Key Features:**
- **System Prompt:** "You are a helpful AI. Keep answers short and spoken naturally."
- **Temperature:** 0.7 (balanced creativity and consistency)
- **Max Tokens:** 100 (keeps responses concise for TTS)
- **Chat Format:**
  ```python
  [
    {"role": "system", "content": "system instructions..."},
    {"role": "user", "content": "user's text"},
    {"role": "assistant", "content": "ai's response"}
  ]
  ```

##### **C. Text-to-Speech (TTS) - CSM Model:**

```python
csm_generator = load_csm_1b(DEVICE)
```

**What it does:**
- Converts text into natural-sounding speech audio
- Uses speaker voice characteristics from context
- Maintains conversation flow and speaker personality
- Input: Text + speaker context audio
- Output: WAV audio tensor (24kHz sample rate)

**Key Parameters in `generate()` call:**
```python
ai_audio_tensor = csm_generator.generate(
    text=ai_text,           # What to say
    speaker=0,              # Speaker ID (0=AI, 1=User)
    context=recent_segments, # Last 3 turns for style
    max_audio_length_ms=30_000,  # Max 30 seconds per response
    temperature=0.6,        # Lower temp = more stable
    topk=20                 # Top-20 token sampling
)
```

#### **The `/talk` Endpoint - Complete Flow:**

```python
@app.post("/talk")
async def audio_talk(file: UploadFile = File(...)):
```

**Step-by-Step Processing:**

1. **Receive & Load Audio:**
   ```python
   user_audio_bytes = await file.read()
   user_tensor, user_sr = torchaudio.load(io.BytesIO(user_audio_bytes))
   ```
   - Reads WAV data from form submission
   - Converts to PyTorch tensor

2. **Speech-to-Text:**
   ```python
   result = stt_pipe("temp_input.wav")
   user_text = result["text"]
   print(f"User Said: {user_text}")
   ```
   - Transcribes audio to text
   - Handles resampling to 16kHz if needed

3. **Update History:**
   ```python
   history_text.append({"role": "user", "content": user_text})
   history_audio_segments.append(user_segment)  # For CSM context
   ```
   - Stores user input in chat history
   - Stores audio representation for speaker context

4. **Generate AI Response:**
   ```python
   outputs = chat_pipe(
       history_text,
       max_new_tokens=100,
       temperature=0.7,
   )
   ai_text = outputs[0]["generated_text"][-1]["content"]
   history_text.append({"role": "assistant", "content": ai_text})
   ```
   - Uses LLM to generate contextual response
   - Appends to chat history

5. **Generate Audio from Text:**
   ```python
   recent_segments = [history_audio_segments[0]] + history_audio_segments[-MAX_CONTEXT:]
   ai_audio_tensor = csm_generator.generate(
       text=ai_text,
       speaker=0,
       context=recent_segments,
       temperature=0.6,
   )
   ```
   - CSM generates speech audio
   - Uses sliding window context for consistency
   - Maintains speaker characteristics

6. **Save Conversation:**
   ```python
   conversation_tensor = torch.cat(full_audio_buffer, dim=0)
   torchaudio.save(CONVERSATION_FILE, conversation_tensor.unsqueeze(0), 
                   csm_generator.sample_rate)
   ```
   - Concatenates all audio generated so far
   - Saves to `conversation_log.wav` for replay

7. **Return Audio to Client:**
   ```python
   buffer = io.BytesIO()
   torchaudio.save(buffer, ai_audio_tensor.unsqueeze(0).cpu(), 
                   csm_generator.sample_rate, format="wav")
   return StreamingResponse(buffer, media_type="audio/wav")
   ```
   - Converts tensor to WAV format in memory
   - Streams back to browser for playback

---

### 2. **`voice.html`** - Browser Interface

Interactive web UI for real-time voice chat.

#### **Layout & Styling:**

```html
<body>
  <button id="micBtn">Hold to Talk</button>
  <div id="status">Ready</div>
</body>
```

**Visual Elements:**
- **Microphone Button:** 150px circular button with state changes
  - **Default:** Gray (#444) - "Ready"
  - **Recording:** Red (#d63333) with glow - "Listening..."
  - **Processing:** Blue (#007bff) with pulse animation - "Thinking..."
- **Status Text:** Shows current operation state

#### **Key JavaScript Functions:**

##### **1. `initAudio()` - Setup Microphone**
```javascript
async function initAudio() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);
    };
    mediaRecorder.onstop = sendAudio;
}
```
- Requests microphone access from browser
- Creates `MediaRecorder` to capture audio chunks
- Called once when page loads

##### **2. `startRecording()` - When User Presses Button**
```javascript
function startRecording() {
    audioChunks = [];
    mediaRecorder.start();
    btn.classList.add('recording');
    btn.innerText = "Listening...";
}
```
- Clears previous audio chunks
- Starts recording microphone stream
- Updates button color to red
- Changes text to "Listening..."

##### **3. `stopRecording()` - When User Releases Button**
```javascript
function stopRecording() {
    mediaRecorder.stop();
    btn.classList.remove('recording');
    btn.classList.add('processing');
    btn.innerText = "Thinking...";
}
```
- Stops recording
- Switches button to processing state (blue)
- Triggers `sendAudio()` when recording ends

##### **4. `sendAudio()` - Upload to Server**
```javascript
async function sendAudio() {
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const formData = new FormData();
    formData.append("file", audioBlob, "input.wav");

    const response = await fetch('/talk', {
        method: 'POST',
        body: formData
    });

    const blob = await response.blob();
    const audio = new Audio(URL.createObjectURL(blob));
    audio.play();
}
```
- Combines recorded audio chunks into single WAV blob
- Sends to server `/talk` endpoint as multipart form data
- Waits for server response
- Converts response blob to audio
- Plays AI response through speaker

#### **User Interaction Flow:**

```
User Action          Button State    Status              Processing
──────────────────────────────────────────────────────────────────
User presses button  RECORDING       "Recording..."      Mic captures voice
User releases        PROCESSING      "Processing & Gen"  
                                                         → STT converts to text
                                                         → LLM generates response
                                                         → CSM creates audio
                                                         → Server streams audio
                                     "Speaking..."       Browser plays AI voice
Audio finishes                       "Ready"             Ready for next input
```

#### **CSS Animations:**

- **Recording State:** Subtle scale animation to indicate active recording
- **Processing State:** `@keyframes pulse` - Creates expanding box-shadow effect
- **Responsive:** Flexbox centered on all screen sizes
- **Dark Theme:** Black background with white text for low-light environments

---

## 📊 Input/Output Specifications

### **Input Format**

#### **server_voice.py `/talk` Endpoint:**

**Request Type:** `POST multipart/form-data`

**Parameter:**
```
file: <audio/wav binary data>
Sample Rate: Any (auto-resampled)
Channels: Mono or Stereo (handled)
Duration: Typically 1-30 seconds
```

**Example request:**
```javascript
const formData = new FormData();
formData.append("file", audioBlob, "input.wav");
fetch('/talk', { method: 'POST', body: formData });
```

#### **STT (Whisper) Input:**
- **Type:** WAV audio
- **Sample Rate:** Any (Whisper handles resampling to 16kHz internally)
- **Duration:** Typically < 30 seconds for good accuracy

#### **LLM (Llama) Input:**
- **Type:** Chat history (JSON list)
- **Format:**
  ```python
  [
    {"role": "system", "content": "You are helpful..."},
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": "I don't have..."}
  ]
  ```
- **Max Context:** Entire conversation history (handled by Llama's 2048 token limit)

#### **CSM (TTS) Input:**
- **Type:** Text + Audio context
- **Text:** String to convert to speech
- **Context:** List of `Segment` objects (previous turns)
  ```python
  Segment(speaker=0, text="Hello", audio=torch.Tensor)
  ```
- **Speaker ID:** Integer (0=AI, 1=User)

---

### **Output Format**

#### **STT (Whisper) Output:**
```python
{
    "text": "What's the weather today?",
    "chunks": [...]  # Optional timing info
}
```

#### **LLM (Llama) Output:**
```python
[
    {
        "generated_text": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "The weather today is..."}
        ]
    }
]
```

#### **CSM (TTS) Output:**
- **Type:** `torch.Tensor` (floating point audio)
- **Shape:** `(num_samples,)` - 1D tensor
- **Sample Rate:** 24,000 Hz (fixed by CSM)
- **Range:** [-1.0, 1.0] (normalized audio)

**Conversion to WAV:**
```python
audio_tensor = csm_generator.generate(...)  # Returns tensor
torchaudio.save("output.wav", audio_tensor.unsqueeze(0), 24000, format="wav")
```

#### **Server Response (/talk endpoint):**
```
Content-Type: audio/wav
Body: Binary WAV data
Sample Rate: 24,000 Hz
Duration: Variable (depends on response length)
```

---

### **Data Flow Diagram**

```
┌─────────────────────────────────────────────────────────────┐
│ Browser (voice.html)                                        │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ startRecording()                                     │   │
│ │ → mediaRecorder.start()                              │   │
│ │ → audioChunks = [blob1, blob2, ...]                  │   │
│ └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
              ↓ stopRecording() + fetch('/talk')
┌─────────────────────────────────────────────────────────────┐
│ Server (server_voice.py) - POST /talk                       │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ 1. Receive: multipart/form-data (WAV blob)          │   │
│ │ 2. STT: torchaudio.load() → whisper-tiny → text     │   │
│ │    Input:  audio 16kHz                              │   │
│ │    Output: "What time is it?"                        │   │
│ ├──────────────────────────────────────────────────────┤   │
│ │ 3. LLM: chat_pipe(history_text) → response           │   │
│ │    Input:  [{"role": "user", "content": "..."}]     │   │
│ │    Output: "It's 3:30 PM"                            │   │
│ ├──────────────────────────────────────────────────────┤   │
│ │ 4. TTS: csm_generator.generate(text, context)        │   │
│ │    Input:  text="It's 3:30 PM", speaker=0           │   │
│ │    Output: torch.Tensor (24kHz audio)               │   │
│ ├──────────────────────────────────────────────────────┤   │
│ │ 5. Save: torchaudio.save(CONVERSATION_FILE, ...)    │   │
│ │ 6. Return: StreamingResponse(WAV bytes)              │   │
│ └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
              ↓ Response headers: audio/wav
┌─────────────────────────────────────────────────────────────┐
│ Browser (voice.html)                                        │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ sendAudio() continuation                             │   │
│ │ → const blob = await response.blob()                 │   │
│ │ → const audio = new Audio(URL.createObjectURL(blob)) │   │
│ │ → audio.play()                                       │   │
│ └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
              ↓ Audio plays through speaker
```

---

## 🔧 Configuration & Tuning

### **Model Parameters in `server_voice.py`:**

```python
# STT Configuration
stt_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny.en",  # Options: base.en, small.en
    device=DEVICE
)
# Trade-off: tiny=fast, base=better, small=best quality

# LLM Configuration
chat_pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,  # Use float32 if bfloat16 causes issues
    device_map=DEVICE,
)

# LLM Generation Parameters
outputs = chat_pipe(
    history_text,
    max_new_tokens=100,        # Response length (words)
    do_sample=True,            # Use sampling (not greedy)
    temperature=0.7,           # 0.1=deterministic, 1.0=creative
)

# CSM Configuration
MAX_CONTEXT = 3                # Keep last 3 turns (increase for more context)

ai_audio_tensor = csm_generator.generate(
    text=ai_text,
    speaker=0,
    context=recent_segments,
    max_audio_length_ms=30_000, # Max 30s per utterance
    temperature=0.6,            # CSM temperature (lower=stable)
    topk=20                      # Nucleus sampling size
)
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| **CUDA out of memory** | Use `torch_dtype=torch.float32` or reduce `max_new_tokens` |
| **"No module named 'triton'"** | Windows: `pip install triton-windows` |
| **Whisper loading slowly** | Use `model="openai/whisper-tiny.en"` (smallest) |
| **Audio quality poor** | Upgrade STT: use `base.en` or `small.en` |
| **Microphone not working** | Check browser permissions, use HTTPS for non-localhost |
| **Models downloading slowly** | Set `HF_HOME` environment variable to custom cache folder |

---

## 📁 File Structure

```
csm/
├── run_csm.py          # Standalone conversation demo
├── server_voice.py     # FastAPI voice chat server
├── voice.html          # Browser UI for voice chat
├── generator.py        # CSM model loading & generation
├── models.py           # Llama model architecture definitions
├── watermarking.py     # Audio watermarking utilities
├── requirements.txt    # Python dependencies
├── conversation_log.wav # Output: Saved conversations
└── README_DETAILED.md  # This file
```

---

## 🚀 Quick Start Commands

```bash
# 1. Activate environment
conda activate csm-env

# 2. Set environment variable
set NO_TORCH_COMPILE=1

# 3a. Run standalone demo
python run_csm.py

# 3b. OR start voice server
python server_voice.py

# 4. Open browser
# http://localhost:8000
```

---

## 📚 Additional Resources

- **CSM Paper:** https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice
- **Model Repos:**
  - CSM: https://huggingface.co/sesame/csm-1b
  - Llama 3.2: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
  - Whisper: https://huggingface.co/openai/whisper-tiny.en
- **FastAPI Docs:** https://fastapi.tiangolo.com
- **Web Audio API:** https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API

---

**Created:** December 2025  
**Framework:** FastAPI + PyTorch + Hugging Face Transformers  
**License:** Check original repository for license details
