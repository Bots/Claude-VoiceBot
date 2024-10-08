# ClaudeVoiceBot
ClaudeVoiceBot is an intelligent voice assistant powered by Anthropic's Claude AI. It uses hotword detection, speech recognition, and text-to-speech capabilities to create a seamless voice interaction experience.

# Features
- Hotword detection using Porcupine
- Speech recognition with OpenAI's Whisper model
- Natural language processing with Claude-3-Opus
- Text-to-speech using ElevenLabs.
- Voice activity detection for improved audio processing

# Prerequisites
Python 3.8+
CUDA-compatible GPU (for Whisper model)

# Installation
1. Clone the repository: 
```bash
git clone https://github.com/Bots/Claude-VoiceBot.git
```
2. Install the required packages: 
```bash
pip install -r requirements.txt
```
3. Set up environment variables: Copy the env.example file to .env (note the .) and fill in the required values:
```bash
cp env.example .env
```

# (Optional) Create a custom hotword using Porcupine:
1. Visit the Picovoice website.
2. Sign up or log in to your account.
3. Navigate to the Porcupine section and click on "Create Keyword".
4. Enter your desired hotword and select the languages and platforms you want to support.
5. Train the model and download the generated keyword file (.ppn).
6. Create a "keywords" folder in the project root and place the downloaded .ppn file in it. Name the file "keyword.ppn".

# Usage
1. Run the main script: python main.py
2. Choose whether to use ElevenLabs or pyTTSx3. ElevenLabs is a cloud service and therefore takes a little more time than the much lower quality, on-device pyTTSx3.
3. Say the hotword (default is 'hey computer') to activate the bot, after you hear the notification, speak your query. The bot will process your speech, generate a response using Claude, and speak the answer back to you.

# Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

# License
This project is licensed under the MIT License - see the LICENSE file for details.