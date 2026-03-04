"""
This File contains diagnostic functions for Ollama self-hosted LLM clients.

These functions must obtain complementary data to help diagnose issues with 
Ollama self-hosted LLM clients.

These data must be obtained by providing functions to acces endpoints or logs
that are specifically oriented to debug, providing informations during errors,
after errors or in bad states.
"""

import requests
import logging

logger = logging.getLogger(__name__)

def diagnose_ollama_timeout(base_url: str, target_model: str) -> tuple[bool, str]:
    """
    Investigates ths state of the Ollama server after a timeout.
    Returns a diagnostic string explaining the probable cause.
    """
    ps_url = f"{base_url}/api/ps"

    try:
        ps_response = requests.get(ps_url, timeout=5) # Short timeout for diagnostics
        
        if ps_response.status_code == 200:
            loaded_models = ps_response.json().get("models", []) # Get the list of loaded models from the response
            
            is_loaded = False # Assume the model is not loaded until we find it in the list

            for model in loaded_models: # Iterate through the loaded models to check if the target model is among them
                if model.get("name") == target_model: # Check if the model name matches the target model
                    is_loaded = True # If we find the target model in the list of loaded models, we can conclude that the model is loaded and the issue might be related to something else (e.g., resource constraints, network issues, etc.)
                    break

            if is_loaded:
                return (is_loaded, f"DIAGNOSIS: The model '{target_model}' is loaded on the Ollama server." 
                            "The problem is likely due to low resourcers of CPU capabilities.")
                
            else:
                return (is_loaded, f"DIAGNOSIS: The model '{target_model}' is NOT loaded on the Ollama server. "
                        "It probably failed duringn the Cold Start phase, which is the most resource intensive phase. ")
        else:
            return (False, f"DIAGNOSIS: Failed to diagnose Ollama timeout. Status code: {ps_response.status_code}")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"DIAGNOSIS: Error while diagnosing Ollama timeout: {e}")
        return (False, f"DIAGNOSIS: Error while diagnosing Ollama timeout: {e}")