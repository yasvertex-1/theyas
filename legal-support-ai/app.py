import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.chat_utils import session_manager, chat_interface,text_to_speech  


# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# FastAPI Setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Pydantic model for incoming request
class UserProfile(BaseModel):
    firstName: str
    lastName: str
    country: str

# API Endpoints
class InterviewRequest(BaseModel):
    profile: dict

@app.post("/start_interview")
async def start_interview(request: InterviewRequest):
    try:
        # Log the incoming request data for debugging
        logger.info(f"Received profile data: {request.profile}")

                # Validate the profile data with Pydantic
        try:
            profile = UserProfile(**request.profile).model_dump()
            logger.info(f"Validated profile: {profile}")
        except Exception as e:
            logger.error(f"Profile validation error: {e}")
            raise HTTPException(400, f"Invalid profile data: {str(e)}")
  

        # Create session and store profile
        try:
            session_id = session_manager.create_session(profile)
            logger.info(f"Created session: {session_id}")
        except Exception as e:
            logger.error(f"Session creation error: {e}")
            raise HTTPException(500, f"Failed to create session: {str(e)}")
        
        # Extract profile information
        first_name = profile["firstName"]
        
        # Create initial prompt with profile information
        initial_prompt = f""" This is  my name {first_name}"""
        

        # Invoke the chain with the session
        try:        
            response = chat_interface(initial_prompt, session_id=session_id)
            logger.info(f"Generated AI response: {response}")
        except Exception as e:
            logger.error(f"Chain invocation error: {e}")
            raise HTTPException(500, f"Failed to generate interview response: {str(e)}")
        #audio = text_to_speech(response)
        #logger.info(f"Generated audio for response")
        #if not audio:
            #logger.error("Failed to generate audio response")
            #raise HTTPException(500, "Failed to generate audio response")
        
        return JSONResponse({#"audio": audio,
            "session_id": session_id,
            "text": response })
    except Exception as e:
        logger.error(f"Start interview error: {e}", exc_info=True)
        raise HTTPException(500, f"Internal server error: {str(e)}")
    
@app.post("/process_text_response")
async def process_text_response(response: dict):
    try:
        session_id = response.get("session_id")
        user_input = response.get("text", "")
        
        if not session_id:
            raise HTTPException(400, "session_id required")

        ai_response = chat_interface(user_input, session_id=session_id)
        logger.info(f"AI response generated: {ai_response}")
        if not ai_response:
            raise HTTPException(500, "Failed to generate AI response")
        
        #audio = text_to_speech(ai_response)
        #logger.info(f"Generated audio for AI response")
        #if not audio:
            #logger.error("Failed to generate audio response")
            #raise HTTPException(500, "Failed to generate audio response")
        
        return JSONResponse({#"audio": audio,
            "text": ai_response})
    except Exception as e:
        logger.error(f"Text processing error: {e}")
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)