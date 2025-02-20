from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union, Dict, Any
import uvicorn
import logging
import gc
import os
import resource
from webui import run_with_stream

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Increase the soft limit for file descriptors
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))

app = FastAPI()

class RunRequest(BaseModel):
    data: List[Union[str, int, float, bool]]

class RunResponse(BaseModel):
    final_result: str = ""
    errors: str = ""
    model_actions: str = ""
    model_thoughts: str = ""
    trace_file: str = ""
    history_file: str = ""

@app.post("/run", response_model=RunResponse)
async def run_agent(request: RunRequest):
    try:
        logger.info(f"Received request with data: {request.data}")
        
        # Initialize response with empty values
        response = RunResponse()
        
        # Call run_with_stream and process the generator
        try:
            async for update in run_with_stream(*request.data):
                logger.info(f"Got update: {update}")
                
                # If update is a tuple/list, unpack it
                if isinstance(update, (tuple, list)):
                    if len(update) > 0:
                        response.final_result = str(update[0])
                    if len(update) > 1:
                        response.errors = str(update[1])
                    if len(update) > 2:
                        response.model_actions = str(update[2])
                    if len(update) > 3:
                        response.model_thoughts = str(update[3])
                    if len(update) > 4:
                        response.trace_file = str(update[4])
                    if len(update) > 5:
                        response.history_file = str(update[5])
                else:
                    # If update is a single value, update final_result
                    response.final_result = str(update)
        finally:
            # Force garbage collection
            gc.collect()
            
        return response
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7788)
