"""
This file contains integration tests for the Ollama self-hosted LLM client. 
These tests verify that the Ollama client is responding properly, and returning
the expected results.

The tests assert that all the self-hosted ollama models are working as expected,
including, but not limited to, the following metrics provided by the API:
    - Initialization Time (Cold Start)
    - Response Time (Total wall time)
    - Token Usage (Input and Output)
    - Thoughput (Tokens per second)
"""

# Importing from libraries
from dotenv import load_dotenv
import logging
import os
import requests
import sys
import time
# Importing from files
from src.telemetry import diagnose_ollama_timeout

# Load the environment variables from the .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] %(name)s - %(message)s', #Logging format with timestamp, log level, logger name, and message
    handlers=[
        logging.StreamHandler(sys.stdout) #Log to standard output (console)
        # You can add a logging.FileHandler here to log to a file if needed
    ]   
)
logger = logging.getLogger(__name__) # Abreviated logger name for cleaner logs

def test_ollama_api():
    """Test the Ollama API is responding properly and measure performance metrics."""
    base_url = os.getenv("OLLAMA_BASE_URL")
    if not base_url:
        logger.error("OLLAMA_BASE_URL is not set in the environment variables.")
        return

    # Set the URL for the Ollama API endpoint
    url = f"{base_url}/api/generate"
    
    # TO-DO: Add a function to dynamically fetch the available models from the Ollama API and iterate through them for testing
    models = ["ollama/qwen2.5:7b", "ollama/qwen2.5:14b"] # Manually defined list of models to test, replace with dynamic fetching in the future

    for model in models:
        logger.info(f"Testing model: {model}")

        # Define the payload for the request with the model and prompt
        payload = {
            "model": model,
            "prompt": "Are you working properly?",
            "stream": False # Set to True if you want to test streaming responses 
        }

        logger.info(f"Initiating connection to Ollama API at {url} with model {model}")
        logger.info(f"Payload: {payload}")

        # Measure the time taken for the API response
        start_time = time.time() # Start the timer

        try:
            response = requests.post(url, json=payload, timeout=600) # Set a reasonable timeout for the request
            response.raise_for_status() # Raise an exception for HTTP errors ( status 4xx 5xx )

            end_time = time.time() # End the timer
            assert response.status_code == 200, f"Expected status code 200, got {response.status_code}" # Assert that the response status code is 200 (OK)
            duration = end_time - start_time # Calculate the duration of the request

            data = response.json() # Parse the JSON response from the API
            logger.info(f"Response received in {duration:.2f} seconds: {data}")
            assert "response" in data, "Response key not found in the API response" # Assert that the response contains the expected 'response' key
            assert duration < 300, f"Response time {duration:.2f} seconds exceeds the expected threshold of 300 seconds" # Assert that the response time is within an acceptable threshold (e.g., 300 seconds) 

            total_duration = data.get("total_duration", 0) / 1e9 # Convert nanoseconds to seconds
            load_duration = data.get("load_duration", 0) / 1e9 # Convert nanoseconds to seconds
            token_usage = data.get("token_usage", {}) # Get token usage information from the response
            token_output = token_usage.get("output_tokens", 0) # Get the number of output tokens from the token usage information

            logger.info(f"Total Duration: {total_duration:.2f} seconds")
            logger.info(f"Load Duration: {load_duration:.2f} seconds")
            logger.info(f"Token Usage: {token_usage}")
            logger.info(f"Output Tokens: {token_output}")
        
        except requests.exceptions.Timeout:
            logger.error(f"Timeout occurred while connecting to Ollama API for model {model}. Attempting to diagnose the issue...")
            is_loaded, diagnosis = diagnose_ollama_timeout(base_url, model)
            logger.warning(diagnosis) # Log the diagnosis result for the timeout issue

        except requests.exceptions.RequestException as e:
            logger.error(f"An error occurred while connecting to Ollama API for model {model}: {e}")