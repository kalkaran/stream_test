from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
import os
from contextlib import asynccontextmanager
import uvicorn
import logging

from conversation_controller import ConversationController

controller = ConversationController()

logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Application startup: Starting background conversion.")
    controller.start_background_conversion(check_interval=5, parallel_limit=10)
    controller.start_background_whisper_conversion(check_interval=5, parallel_limit=10)
    logging.info("Background conversion thread started.")
    yield
    logging.info("Application shutdown: Stopping background conversion.")
    controller.stop_background_conversions()
    logging.info("Background conversion thread stopped.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust allowed origins as needed.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dictionary to track processed chunks for each session.
# Structure: { session_id: set(chunk_numbers) }
received_chunks = {}

def save_audio(audio_bytes: bytes, session_id: str, chunk_number: int, chunk_type: str):
    """Save the received audio chunk to the 'uploads' subfolder without converting the format."""
    try:
        if len(audio_bytes) == 0:
            logging.warning("Received empty audio data for session %s, chunk %d. Skipping save.", session_id, chunk_number)
            return
        uploads_dir = "uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        filename = f"{session_id}_chunk{chunk_number}_{chunk_type}_{uuid.uuid4().hex}.webm"
        filepath = os.path.join(uploads_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(audio_bytes)
        logging.info("Saved file: %s (size: %d bytes)", filepath, len(audio_bytes))
        controller.handle_chunk(
            session_id=session_id, 
            chunk_number=chunk_number,
            chunk_file_path=filepath, 
            chunk_name=filename, 
            chunk_type=chunk_type
        )
    except Exception as e:
        logging.error("Error saving audio for session %s chunk %d: %s", session_id, chunk_number, e)

@app.get("/")
def read_root():
    return {"message": "Hello, world!"}



@app.post("/register_conversation")
async def register_conversation():
    #TODO enforce registration
    try:
        # Generate a unique session id.
        session_id = "session-" + uuid.uuid4().hex
        response_data = {"status": "registered", "session_id": session_id}
        return JSONResponse(content=response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/conversation_status_all")
async def conversation_status_all():
    try:
        # Generate a unique session id.
        response_data=controller.get_all_conversation_summary()
        return JSONResponse(content=response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/conversation_status/{conversation_id}")
async def conversation_status(conversation_id):
    try:
        # Generate a unique session id.
        response_data=controller.get_conversation_data(conversation_id)
        return JSONResponse(content=response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/upload_audio")
async def upload_audio(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    session_id: str = Form(...),
    chunk_number: int = Form(...),
    chunk_type: str = Form(...)
):
    if not audio:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    # Initialize session tracking if needed.
    if session_id not in received_chunks:
        received_chunks[session_id] = set()
    
    # If this chunk has already been processed, return a duplicate response.
    if chunk_number in received_chunks[session_id]:
        return JSONResponse(content={"status": "duplicate", "message": "Chunk already processed"})
    
    # Mark the chunk as received.
    received_chunks[session_id].add(chunk_number)
    
    try:
        # Read the uploaded audio data.
        audio_bytes = await audio.read()
        
        # Schedule saving the audio in the background.
        background_tasks.add_task(save_audio, audio_bytes, session_id, chunk_number, chunk_type)
        
        response_data = {
            "status": "accepted",
            "session_id": session_id,
            "chunk_number": chunk_number,
            "chunk_type": chunk_type
        }
        
        # If this is the final chunk, check that all expected chunks have been received.
        if chunk_type.lower() == "final":
            # Assume chunk numbering starts at 0; hence final chunk's number + 1 is the total expected.
            expected_chunks = set(range(chunk_number + 1))
            if received_chunks[session_id] == expected_chunks:
                del received_chunks[session_id]
                response_data["cleanup"] = "session cleaned up"
            else:
                missing = sorted(list(expected_chunks - received_chunks[session_id]))
                response_data["cleanup"] = f"not cleaned up, missing chunks: {missing}"
                #TODO - Find a way to handle it so it waits in case the final chunk arrives before every chunk has been recieved.
        return JSONResponse(content=response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=8888, reload=True)
