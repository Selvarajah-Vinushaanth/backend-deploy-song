# Song Analysis API

A FastAPI backend for the Song-React application that provides three main functionalities:
1. Metaphor creation
2. Metaphor classification
3. Lyrics generation

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

## API Endpoints

### 1. Create Metaphors
- **URL**: `/api/create-metaphors`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "topic": "love",
    "style": "romantic",
    "count": 3
  }
  ```
- **Response**:
  ```json
  {
    "metaphors": [
      "Your love is like a flame that warms my heart",
      "Love is a journey that never ends",
      "Your heart is like the moon that guides me through darkness"
    ]
  }
  ```

### 2. Predict Metaphor
- **URL**: `/api/predict`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "text": "Life is a journey"
  }
  ```
- **Response**:
  ```json
  {
    "is_metaphor": true,
    "confidence": 0.89
  }
  ```

### 3. Generate Lyrics
- **URL**: `/api/generate-lyrics`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "theme": "summer love",
    "style": "pop",
    "length": 3
  }
  ```
- **Response**:
  ```json
  {
    "lyrics": "Verse 1:\nUnder the summer sky...\n\nChorus:\nSummer love, summer love...\n\n..."
  }
  ```

## Notes

- The backend uses Hugging Face transformer models for all functionalities
- All models are loaded on-demand to save memory
- For production use, consider deploying with proper resource allocation for the ML models
