import os
import requests
import base64
import logging
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging configuration
logging.basicConfig(level=logging.INFO)

# Configuration constants
USER_ID = os.getenv("USER_ID")
API_BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{USER_ID}/ai/run/"
API_TOKEN = os.getenv("API_TOKEN")
if not API_TOKEN:
    logging.error("API_TOKEN not found in environment variables!")
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}





def llama(prompt, max_retries=3, retry_delay=2):
    """
    Sends a prompt to the Llama language model endpoint and returns the response.
    
    Args:
        prompt (str): The user prompt to send.
        max_retries (int): Maximum number of attempts.
        retry_delay (int): Seconds to wait between attempts.
    
    Returns:
        dict or None: The JSON response from the Llama API or None if all attempts fail.
    """
    categories = ["music", "science"]
    payload = {
        "messages": [
            {"role": "system", "content": f"You are a friendly assistant that helps categorise conversations and summarises if there is any of these categories in the text: {str(categories)}"},
            {"role": "user", "content": prompt}
        ]
    }
    model_endpoint = "@cf/meta/llama-3-8b-instruct"
    url = f"{API_BASE_URL}{model_endpoint}"
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=HEADERS, json=payload)
            logging.info("Llama attempt %d: response code %d", attempt+1, response.status_code)
            response.raise_for_status()
            result = response.json()
            logging.info("Llama response: %s", result)
            return result.get("result").get("response")
        except Exception as e:
            logging.error("Error in llama API call on attempt %d: %s", attempt+1, e)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    return None

def base64encodewavfile(file_path):
    """
    Reads a WAV file and returns its base64 encoded string.
    
    Args:
        file_path (str): The path to the WAV file.
    
    Returns:
        str or None: Base64 encoded string of the file contents, or None if an error occurs.
    """
    try:
        with open(file_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return encoded
    except Exception as e:
        logging.error("Error encoding WAV file '%s': %s", file_path, e)
        return None

def extract_final_text(response):
    """
    Extracts and returns the final transcription text from a Whisper API response.

    Args:
        response (dict): The JSON response from the Whisper API.

    Returns:
        str or None: The transcription text if available, otherwise None.
    """
    try:
        if response.get("success", False):
            result = response.get("result", {})
            text = result.get("text")
            if text is not None:
                return text.strip()
            else:
                logging.error("No 'text' key found in result: %s", result)
                return None
        else:
            logging.error("Response unsuccessful: %s", response)
            return None
    except Exception as e:
        logging.error("Error extracting final text: %s", e)
        return None

def whisper(file_path, max_retries=3, retry_delay=2):
    """
    Sends a WAV file (encoded in base64) to the Whisper endpoint and returns the transcription.
    
    Args:
        file_path (str): The path to the WAV file.
        max_retries (int): Maximum number of attempts.
        retry_delay (int): Seconds to wait between attempts.
    
    Returns:
        str or None: The transcription text from the Whisper API or None if all attempts fail.
    """
    base64string = base64encodewavfile(file_path)
    if base64string is None:
        logging.error("Failed to encode file: %s", file_path)
        return None
    
    payload = {
        "audio": base64string,
    }
    model_endpoint = "@cf/openai/whisper-large-v3-turbo"
    url = f"{API_BASE_URL}{model_endpoint}"
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=HEADERS, json=payload)
            logging.info("Whisper attempt %d: response code %d", attempt+1, response.status_code)
            response.raise_for_status()
            result = response.json()
            logging.info("Whisper response: %s", result)
            final_text = extract_final_text(result)
            return final_text
        except Exception as e:
            logging.error("Error in whisper API call for file '%s' on attempt %d: %s", file_path, attempt+1, e)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    return None

if __name__ == "__main__":
    # Example usage for Llama
    prompt = "Write a short story about a llama that goes on a journey to find an orange cloud"
    llama_result = llama(prompt)
    print("Llama result:", llama_result)
    
    # Example usage for Whisper
    wav_file_path = "./test/0014.wav"  # Replace with your actual WAV file path.
    whisper_result = whisper(wav_file_path)
    print("Whisper result:", whisper_result)
