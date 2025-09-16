import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import time
from datetime import datetime

# Import model modules
from models.metaphor_creator import generate_metaphor
from models.metaphor_classifier import classify_metaphor
from models.lyric_generator import generate_lyrics_text
from models.masking_predict import predict_masked_tokens

# Initialize Firebase Admin safely
try:
    # Check if Firebase app is already initialized
    app = firebase_admin.get_app()
except ValueError:
    # If not, initialize it
    cred = credentials.Certificate("../song-writing-assistant-4cd39-firebase-adminsdk-fbsvc-8ddefda8ac.json")
    app = firebase_admin.initialize_app(cred)

# Get Firestore client
db = firestore.client()

app = FastAPI(title="Song Analysis API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","https://song-react-with-firestore.vercel.app","https://deploy-frontend-sand.vercel.app"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def validate_api_key(
    api_key: Optional[str] = Header(None), 
    x_api_key: Optional[str] = Depends(api_key_header)
):
    """Validate the API key from either header."""
    key = api_key or x_api_key
    
    print(f"Validating API key: {key}")  # optional, remove in production

    if not key:
        raise HTTPException(status_code=401, detail="API Key is missing")
    
    try:
        api_keys_ref = db.collection('apiKeys')
        query = api_keys_ref.where('key', '==', key)
        results = query.get()
        
        if len(results) > 0:
            key_doc = results[0]
            key_data = key_doc.to_dict()
            
            if 'userId' in key_data:
                print(f"Found valid key for user {key_data['userId']}")  # optional, remove in prod
                return key_data['userId']  # Return userId for logging
            else:
                print(f"API key exists but has no userId")  # optional
                raise HTTPException(status_code=403, detail="Invalid API Key - No associated user")
        
        # No valid key found
        print(f"No valid key found")  # optional
        raise HTTPException(status_code=403, detail="Invalid API Key")
        
    except Exception as e:
        print(f"API key validation error: {str(e)}")  # optional
        raise HTTPException(status_code=500, detail="Error validating API Key")

async def log_api_request(user_id: str, endpoint: str, status_code: int, latency_ms: int, payload_size: int = 0):
    """Log API request to Firestore for analytics"""
    try:
        log_data = {
            'userId': user_id,
            'endpoint': endpoint,
            'status': status_code,
            'latency': latency_ms,
            'payloadSize': payload_size,
            'timestamp': firestore.SERVER_TIMESTAMP,
            'date': datetime.now().strftime('%Y-%m-%d')
        }
        
        # Add to apiRequests collection
        db.collection('apiRequests').add(log_data)
        print(f"Logged API request for user {user_id}: {endpoint} - {status_code}")
        
    except Exception as e:
        print(f"Failed to log API request: {str(e)}")
        # Don't raise exception - logging failure shouldn't break the API

async def log_api_request_test(user_id: str, endpoint: str, status_code: int, latency_ms: int, payload_size: int = 0):
    """Log API request to Firestore for analytics"""
    try:
        log_data = {
            'userId': user_id,
            'endpoint': endpoint,
            'status': status_code,
            'latency': latency_ms,
            'payloadSize': payload_size,
            'timestamp': firestore.SERVER_TIMESTAMP,
            'date': datetime.now().strftime('%Y-%m-%d')
        }
        
        # Add to apiRequests collection
        db.collection('apiRequests_test').add(log_data)
        print(f"Logged API request for user {user_id}: {endpoint} - {status_code}")
        
    except Exception as e:
        print(f"Failed to log API request: {str(e)}")
        # Don't raise exception - logging failure shouldn't break the API
# Models for request/response data
class MetaphorRequest(BaseModel):
    source: str
    target: str
    # Prefer Context from frontend (supports Tamil/English); keep emotion for backward compatibility
    Context: Optional[str] = None
    emotion: Optional[str] = None
    count: Optional[int] = 2
    # userId:str
class MetaphorRequestReal(BaseModel):
    source: str
    target: str
    # Prefer Context from frontend (supports Tamil/English); keep emotion for backward compatibility
    Context: Optional[str] = None
    emotion: Optional[str] = None
    count: Optional[int] = 2
    userId:str
class MetaphorResponse(BaseModel):
    metaphors: List[str]
    
class PredictionRequestReal(BaseModel):
    text: str
    userId:str
  
class PredictionRequest(BaseModel):
    text: str
    # userId:str    
class PredictionResponse(BaseModel):
    is_metaphor: bool
    confidence: float
class MaskingRequest(BaseModel):
    text: str
    top_k: Optional[int] = 5  # Updated default value to be more reasonable
    # userId:str    
class MaskingRequestReal(BaseModel):
    text: str
    top_k: Optional[int] = 5  # Updated default value to be more reasonable
    userId:str
class MaskingResponse(BaseModel):
    suggestions: List[str]


# API Routes
@app.post("/api/create-metaphors", response_model=MetaphorResponse)
async def create_metaphors(request: MetaphorRequestReal):
    try:
        # Prefer explicit Context from client; else derive from emotion for backward compatibility
        emotion_to_context = {
            "positive": "romantic",
            "negative": "philosophical",
            "neutral": "poetic",
        }
        normalized_context = (
            (request.Context or "").strip()
            or emotion_to_context.get((request.emotion or "").strip().lower(), "poetic")
        )

        # Call new signature: generate_metaphor(source, target, Context, count)
        metaphors = generate_metaphor(
            source=request.source,
            target=request.target,
            Context=normalized_context,
            count=request.count or 2,
        )

        # Deduplicate
        unique_metaphors = []
        for m in metaphors:
            if m not in unique_metaphors:
                unique_metaphors.append(m)
        await log_api_request(request.userId, "metaphor-creator", 200, 500)
        return {"metaphors": unique_metaphors}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating metaphors: {str(e)}")

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_metaphor(request: PredictionRequestReal):
    try:
        is_metaphor, confidence = classify_metaphor(request.text)
        await log_api_request(request.userId, "metaphor-classifier", 200,500)
        return {"is_metaphor": is_metaphor, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting metaphor: {str(e)}")

class LyricsRequest(BaseModel):
    motion: str
    seed: Optional[str] = ""
    count: Optional[int] = 3  # Add count parameter with default value
    # userId:str
class LyricsRequestReal(BaseModel):
    motion: str
    seed: Optional[str] = ""
    count: Optional[int] = 3  # Add count parameter with default value
    userId:str    
class LyricsResponse(BaseModel):
    lyrics: List[str]
    # suggestions: Optional[List[str]] = None

@app.post("/api/generate-lyrics", response_model=LyricsResponse)
async def create_lyrics(request: LyricsRequestReal):
    try:
        print(f"Received lyric request: Motion={request.motion}, Seed={request.seed}, Count={request.count}")

        # generate main lyric
        main_lyric = generate_lyrics_text(motion=request.motion, seed=request.seed)

        # generate additional lyrics based on count (count - 1 more lyrics)
        more_lyrics = [generate_lyrics_text(motion=request.motion, seed="") for _ in range(request.count - 1)]

        # combine all lyrics in a single list
        all_lyrics = [main_lyric] + more_lyrics

        print(f"Generated total {len(all_lyrics)} lyrics")
        await log_api_request(request.userId, "lyric-generator", 200,500)
        return {"lyrics": all_lyrics}

    except Exception as e:
        print(f"Error generating lyrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating lyrics: {str(e)}")

@app.post("/api/predict-mask")
async def predict_mask(request: MaskingRequestReal):
    try:
        if "[mask]" not in request.text:
            raise HTTPException(status_code=400, detail="Text must contain [mask] token")
        print(request)
        # Ensure top_k is within reasonable bounds
        top_k = max(1, min(15, request.top_k))  # Allow up to 15 suggestions
        user_id=request.userId
        print(user_id)
        suggestions = predict_masked_tokens(request.text, top_k=top_k)
        await log_api_request(user_id, "masking-predict", 200, 500)
        
        return {"suggestions": suggestions}
    except Exception as e:
        print(f"Error predicting masked tokens: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error predicting masked tokens: {str(e)}")

@app.post("/api/v1/predict-mask")
async def predict_mask_v1(request: MaskingRequest, user_id: str = Depends(validate_api_key)):
    """V1 endpoint for mask prediction with API key validation"""
    start_time = time.time()
    
    try:
        if "[mask]" not in request.text:
            raise HTTPException(status_code=400, detail="Text must contain [mask] token")
        
        # Ensure top_k is within reasonable bounds
        top_k = max(1, min(15, request.top_k))  # Allow up to 15 suggestions
        
        suggestions = predict_masked_tokens(request.text, top_k=top_k)
        
        # Log successful request
        latency_ms = int((time.time() - start_time) * 1000)
        await log_api_request_test(user_id, "/api/v1/predict-mask", 200, latency_ms, len(request.text))
        
        return {"suggestions": suggestions}
    except HTTPException as he:
        # Log failed request
        latency_ms = int((time.time() - start_time) * 1000)
        await log_api_request_test(user_id, "/api/v1/predict-mask", he.status_code, latency_ms)
        raise he
    except Exception as e:
        print(f"Error predicting masked tokens: {str(e)}")
        # Log error request
        latency_ms = int((time.time() - start_time) * 1000)
        await log_api_request_test(user_id, "/api/v1/predict-mask", 500, latency_ms)
        raise HTTPException(status_code=500, detail=f"Error predicting masked tokens: {str(e)}")

@app.post("/api/v1/classify-metaphor")
async def classify_metaphor_v1(request: PredictionRequest, user_id: str = Depends(validate_api_key)):
    """V1 endpoint for metaphor classification with API key validation"""
    start_time = time.time()
    
    try:
        is_metaphor, confidence = classify_metaphor(request.text)
        
        # Log successful request
        latency_ms = int((time.time() - start_time) * 1000)
        await log_api_request_test(user_id, "/api/v1/classify-metaphor", 200, latency_ms, len(request.text))
        
        return {"is_metaphor": is_metaphor, "confidence": confidence}
    except Exception as e:
        # Log error request
        latency_ms = int((time.time() - start_time) * 1000)
        await log_api_request_test(user_id, "/api/v1/classify-metaphor", 500, latency_ms)
        raise HTTPException(status_code=500, detail=f"Error predicting metaphor: {str(e)}")

@app.post("/api/v1/create-metaphors")
async def create_metaphors_v1(request: MetaphorRequest, user_id: str = Depends(validate_api_key)):
    """V1 endpoint for metaphor creation with API key validation"""
    start_time = time.time()
    
    try:
        emotion_to_context = {
            "positive": "romantic",
            "negative": "philosophical",
            "neutral": "poetic",
        }
        normalized_context = (
            (request.Context or "").strip()
            or emotion_to_context.get((request.emotion or "").strip().lower(), "poetic")
        )

        metaphors = generate_metaphor(
            source=request.source,
            target=request.target,
            Context=normalized_context,
            count=request.count or 2,
        )

        unique_metaphors = []
        for m in metaphors:
            if m not in unique_metaphors:
                unique_metaphors.append(m)

        latency_ms = int((time.time() - start_time) * 1000)
        payload_size = len(request.source) + len(request.target or "")
        await log_api_request_test(user_id, "/api/v1/create-metaphors", 200, latency_ms, payload_size)

        return {"metaphors": unique_metaphors}
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        await log_api_request_test(user_id, "/api/v1/create-metaphors", 500, latency_ms)
        raise HTTPException(status_code=500, detail=f"Error generating metaphors: {str(e)}")

@app.post("/api/v1/generate-lyrics")
async def generate_lyrics_v1(request: LyricsRequest, user_id: str = Depends(validate_api_key)):
    """V1 endpoint for lyric generation with API key validation"""
    start_time = time.time()
    
    try:
        print(f"Received lyric request: Motion={request.motion}, Seed={request.seed}, Count={request.count}")

        # generate main lyric
        main_lyric = generate_lyrics_text(motion=request.motion, seed=request.seed)

        # generate additional lyrics based on count (count - 1 more lyrics)
        more_lyrics = [generate_lyrics_text(motion=request.motion, seed="") for _ in range(request.count - 1)]

        # combine all lyrics in a single list
        all_lyrics = [main_lyric] + more_lyrics

        print(f"Generated total {len(all_lyrics)} lyrics")
        
        # Log successful request
        latency_ms = int((time.time() - start_time) * 1000)
        payload_size = len(request.motion) + len(request.seed or "")
        await log_api_request_test(user_id, "/api/v1/generate-lyrics", 200, latency_ms, payload_size)
        
        return {"lyrics": all_lyrics}

    except Exception as e:
        print(f"Error generating lyrics: {str(e)}")
        # Log error request
        latency_ms = int((time.time() - start_time) * 1000)
        await log_api_request_test(user_id, "/api/v1/generate-lyrics", 500, latency_ms)
        raise HTTPException(status_code=500, detail=f"Error generating lyrics: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Song Analysis API"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
