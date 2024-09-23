# Class Notes Manager

Class Notes Manager is an advanced, AI-powered note-taking application designed for students and educators. It provides a seamless interface for creating, managing, and interacting with class notes, leveraging cutting-edge AI technologies for enhanced learning experiences.

## Features

1. **User Authentication**: Secure login system to protect user data and provide personalized experiences.

2. **Class Management**: Create and manage multiple classes, keeping notes organized by subject.

3. **Rich Note-Taking**: 
   - Text-based note creation and editing
   - File attachment support (images, PDFs, PowerPoint presentations)
   - Audio file upload with automatic transcription

4. **AI-Powered Teaching Assistant (AI-TA)**:
   - Each class has its own AI-TA trained on the notes and materials of that specific class
   - Ask questions about the class content and receive intelligent responses

5. **Optical Character Recognition (OCR)**:
   - Automatically extract text from images and scanned PDFs
   - Convert PowerPoint presentations to text

6. **Audio Transcription**:
   - Automatically transcribe uploaded audio files to text

7. **Voting System**:
   - Upvote and downvote notes to highlight the most useful content
   - Notes are displayed in order of popularity

8. **Cloud Storage Integration**:
   - Seamlessly store and retrieve files using Amazon S3

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Database**: SQLite
- **AI and Machine Learning**:
  - OpenAI's GPT for the AI Teaching Assistant
  - Tesseract OCR for text extraction from images
  - OpenAI's Whisper for audio transcription
- **Cloud Storage**: Amazon S3
- **Authentication**: Streamlit-Authenticator

## System-Level Dependencies

Before installing the Python packages, ensure you have the following system-level dependencies installed:

- **Tesseract OCR**: Required for text extraction from images
- **Poppler**: Required for PDF processing
- **FFmpeg**: Required for audio processing

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/class-notes-manager.git
   cd class-notes-manager
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   Create a `.env` file in the project root and add the following:
   ```
   OPENAI_API_KEY=your_openai_api_key
   AWS_ACCESS_KEY_ID=your_aws_access_key
   AWS_SECRET_ACCESS_KEY=your_aws_secret_key
   S3_BUCKET_NAME=your_s3_bucket_name
   ```

4. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

## Usage

1. Log in using your credentials.
2. Create a new class or select an existing one from the sidebar.
3. Add notes by typing in the text area, uploading files, or recording audio.
4. Use the AI-TA to ask questions about the class content.
5. Vote on notes to help prioritize the most useful information.
6. Edit or delete notes as needed.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
