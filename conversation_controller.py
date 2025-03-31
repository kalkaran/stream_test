import uuid
import concurrent.futures
import threading
import time
import logging
from utils.convert_all_formats_to_wav import AudioConverter
from utils.api import whisper, llama

logging.basicConfig(level=logging.INFO)

class Conversation:
    """
    Represents a conversation that tracks uploaded chunks.
    
    Attributes:
        conversation_id (str): The unique ID for the conversation.
        chunks (dict): A mapping from chunk number to a dictionary containing:
            - file_path: The path where the chunk file is stored.
            - file_name: The name of the chunk file.
            - chunk_type: The type of the chunk ("first", "middle", or "final").
        final_chunk_received (bool): Indicates whether the final chunk has been received.
    """
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.chunks = {}  
        self.final_chunk_received = False
        self.category_results = []  # Each element: {"prompt": ..., "result": ...}

    def add_chunk(self, chunk_number: int, chunk_file_path: str, chunk_name: str, chunk_type: str):
        self.chunks[chunk_number] = {
            "file_path": chunk_file_path,
            "file_name": chunk_name,
            "chunk_type": chunk_type,
            "chunk_converted": False,   # For WAV conversion
            "whisper_converted": False, # For Whisper processing
            "whisper_output": None      # To store the API output
        }
        if chunk_type.lower() == "final":
            self.final_chunk_received = True

    def is_complete(self) -> bool:
        """
        Determines if the conversation is complete.
        Assumes that if the final chunk has been received, then the expected number of chunks is (max_chunk_number + 1).
        
        Returns:
            bool: True if all chunks are present, otherwise False.
        """
        if not self.final_chunk_received:
            return False
        expected_chunks = set(range(max(self.chunks.keys()) + 1))
        return set(self.chunks.keys()) == expected_chunks

    def get_missing_chunks(self):
        """
        Returns a sorted list of any missing chunk numbers, if the final chunk has been received.
        
        Returns:
            list: Missing chunk numbers if the final chunk was received; otherwise, an empty list.
        """
        if not self.final_chunk_received:
            return []
        expected_chunks = set(range(max(self.chunks.keys()) + 1))
        missing = expected_chunks - set(self.chunks.keys())
        return sorted(missing)

    def has_pending_conversion(self) -> bool:
        """Returns True if there's at least one chunk not yet converted."""
        for chunk in self.chunks.values():
            if not chunk["chunk_converted"]:
                return True
        return False

    def has_pending_whisper(self) -> bool:
        """Returns True if there's at least one chunk not yet processed by Whisper."""
        for chunk in self.chunks.values():
            if not chunk.get("whisper_converted", False):
                return True
        return False

    def convert_next_chunk(self):
        """
        Finds and converts the next unconverted chunk using AudioConverter.
        Returns a descriptive string indicating the result.
        """
        for chunk_number in sorted(self.chunks.keys()):
            chunk = self.chunks[chunk_number]
            if not chunk["chunk_converted"]:
                try:
                    converter = AudioConverter(input_file=chunk["file_path"], output_folder="converted_wav")
                    converted_path = converter()  # __call__ returns the converted file path.
                    chunk["chunk_converted"] = True
                    logging.info(f"Converted chunk {chunk_number} to WAV at {converted_path}")
                    return f"Converted chunk {chunk_number}"
                except Exception as e:
                    logging.error(f"Failed to convert chunk {chunk_number}: {e}")
                    chunk["chunk_converted"] = "Failed"
                    return f"Conversion failed for chunk {chunk_number}"
        return "No pending conversion"


    def convert_next_chunk_whisper(self):
        """
        Finds and processes the next unprocessed chunk using the whisper() API function.
        The output from the API is stored in the chunk's 'whisper_output' field.
        Returns a descriptive string indicating the result.
        """
        for chunk_number in sorted(self.chunks.keys()):
            chunk = self.chunks[chunk_number]
            if not chunk.get("whisper_converted", False):
                try:
                    output = whisper(chunk["file_path"])  # Call your API function on the file.
                    chunk["whisper_output"] = output
                    chunk["whisper_converted"] = True
                    logging.info(f"Whisper processed chunk {chunk_number} with output: {output}")
                    return f"Whisper processed chunk {chunk_number}"
                except Exception as e:
                    logging.error(f"Whisper conversion failed for chunk {chunk_number}: {e}")
                    chunk["whisper_converted"] = "Failed"
                    return f"Whisper conversion failed for chunk {chunk_number}"
        return "No pending Whisper conversion"

    
    def categorize(self, max_segment_words=20, overlap=10):
        """
        Collects words from all whisper outputs, creates overlapping segments, 
        calls the llama API on each segment, and stores the prompt and result.
        
        - The first segment is the first max_segment_words.
        - Subsequent segments are created with an overlap (last 'overlap' words from previous segment plus next words).
        
        Returns:
            list: List of dictionaries with keys 'prompt' and 'result'.
            Returns None if insufficient words are available.
        """
        # Concatenate whisper outputs in order.
        all_text = ""
        for chunk_number in sorted(self.chunks.keys()):
            chunk = self.chunks[chunk_number]
            if chunk.get("whisper_converted") is True and chunk.get("whisper_output"):
                all_text += " " + chunk["whisper_output"]
        all_text = all_text.strip()
        if not all_text:
            logging.info("No whisper output available for categorization.")
            return None
        words = all_text.split()
        if len(words) < max_segment_words:
            logging.info("Not enough words to categorize. Need at least %d, got %d", max_segment_words, len(words))
            return None

        segments = []
        # First segment: first max_segment_words.
        segments.append({"prompt": " ".join(words[:max_segment_words]), "result": None})
        last_index = max_segment_words

        # Create sliding windows.
        while last_index < len(words):
            start_index = max(last_index - overlap, 0)
            end_index = start_index + max_segment_words
            if end_index > len(words):
                break
            segments.append({"prompt": " ".join(words[start_index:end_index]), "result": None})
            last_index = end_index

        results = []
        for seg in segments:
            try:
                logging.info("Sending categorization prompt: %s", seg["prompt"])
                response = llama(seg["prompt"])
                seg["result"] = response
                results.append(seg)
            except Exception as e:
                logging.error("Error categorizing prompt '%s': %s", seg["prompt"], e)
                seg["result"] = "Error"
                results.append(seg)
        self.category_results = results
        return results

    def get_category_details(self):
            """Returns the list of categorization details (prompt and result) if available."""
            return self.category_results if self.category_results else None


class ConversationController:
    """
    Controller class to manage multiple conversations.
    
    Attributes:
        conversations (dict): A mapping from session_id to Conversation objects.
    """
    def __init__(self):
        logging.info("Started Conversion Controller.")
        self.conversations = {}  # {session_id: Conversation}
        self._conversion_thread = None
        self._whisper_thread = None
        self._category_thread = None
        self._stop_event = threading.Event()
    
    ## AUDIO CONVERTER:

    def convert_chunks_to_wav(self, parallel_limit=10):
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_limit) as executor:
            futures = []
            for conversation in self.conversations.values():
                if conversation.has_pending_conversion():
                    future = executor.submit(conversation.convert_next_chunk)
                    futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    # Only log as INFO if a conversion occurred or failed; otherwise, log at DEBUG.
                    if result not in ("No pending conversion",):
                        logging.info(f"Conversion task result: {result}")
                    else:
                        logging.debug(f"Conversion task result: {result}")
                except Exception as e:
                    logging.error(f"Conversion task raised an exception: {e}")

    def _background_conversion_loop(self, check_interval, parallel_limit):
        while not self._stop_event.is_set():
            self.convert_chunks_to_wav(parallel_limit=parallel_limit)
            time.sleep(check_interval)

    def start_background_conversion(self, check_interval=5, parallel_limit=10):
        """
        Starts a background thread that periodically checks and converts unconverted chunks.
        
        Args:
            check_interval (int): Number of seconds to wait between checks.
            parallel_limit (int): Maximum number of concurrent conversion tasks.
        """
        if self._conversion_thread is None or not self._conversion_thread.is_alive():
            self._stop_event.clear()
            self._conversion_thread = threading.Thread(
                target=self._background_conversion_loop,
                args=(check_interval, parallel_limit),
                daemon=True
            )
            self._conversion_thread.start()
            logging.info("Started background conversion thread.")

    # def stop_background_conversion(self):
    #     """Stops the background conversion thread gracefully."""
    #     self._stop_event.set()
    #     if self._conversion_thread is not None:
    #         self._conversion_thread.join()
    #         logging.info("Stopped background conversion thread.")

    ## WHISPER:

    def convert_chunks_to_whisper(self, parallel_limit=10):
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_limit) as executor:
            futures = []
            for conversation in self.conversations.values():
                if conversation.has_pending_whisper():
                    future = executor.submit(conversation.convert_next_chunk_whisper)
                    futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    # Only log as INFO if a conversion occurred or failed; otherwise, log at DEBUG.
                    if result not in ("No pending conversion",):
                        logging.info(f"Conversion task result: {result}")
                    else:
                        logging.debug(f"Conversion task result: {result}")
                except Exception as e:
                    logging.error(f"Conversion task raised an exception: {e}")

    def start_background_whisper_conversion(self, check_interval=5, parallel_limit=10):
        """
        Starts a background thread that periodically checks and processes unconverted chunks with Whisper.
        """
        # We create a separate thread for Whisper conversion.
        self._whisper_thread = threading.Thread(
            target=self._background_whisper_conversion_loop,
            args=(check_interval, parallel_limit),
            daemon=True
        )
        self._whisper_thread.start()
        logging.info("Started background Whisper conversion thread.")

    def stop_background_conversions(self):
        """Stops the background conversion threads gracefully."""
        self._stop_event.set()
        if self._conversion_thread is not None:
            self._conversion_thread.join()
            logging.info("Stopped background WAV conversion thread.")
        if hasattr(self, '_whisper_thread') and self._whisper_thread is not None:
            self._whisper_thread.join()
            logging.info("Stopped background Whisper conversion thread.")


    def _background_whisper_conversion_loop(self, check_interval, parallel_limit):
        while not self._stop_event.is_set():
            self.convert_chunks_to_whisper(parallel_limit=parallel_limit)
            time.sleep(check_interval)


        # Category conversion methods:
    def convert_chunks_to_category(self, parallel_limit=5):
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_limit) as executor:
            futures = []
            for conversation in self.conversations.values():
                # Only attempt categorization if there is new whisper output.
                futures.append(executor.submit(conversation.categorize))
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        logging.info("Category conversion task result: %s", result)
                except Exception as e:
                    logging.error("Category conversion task raised an exception: %s", e)

    def _background_category_loop(self, check_interval, parallel_limit):
        while not self._stop_event.is_set():
            self.convert_chunks_to_category(parallel_limit=parallel_limit)
            time.sleep(check_interval)

    def start_background_category_conversion(self, check_interval=10, parallel_limit=5):
        if self._category_thread is None or not self._category_thread.is_alive():
            self._stop_event.clear()
            self._category_thread = threading.Thread(
                target=self._background_category_loop,
                args=(check_interval, parallel_limit),
                daemon=True
            )
            self._category_thread.start()
            logging.info("Started background category conversion thread.")

    def stop_background_conversions(self):
        self._stop_event.set()
        if self._conversion_thread is not None:
            self._conversion_thread.join()
            logging.info("Stopped background WAV conversion thread.")
        if self._whisper_thread is not None:
            self._whisper_thread.join()
            logging.info("Stopped background Whisper conversion thread.")
        if self._category_thread is not None:
            self._category_thread.join()
            logging.info("Stopped background category conversion thread.")



    def handle_chunk(self, *, session_id: str, chunk_number: int, chunk_file_path: str, chunk_name: str, chunk_type: str) -> Conversation:
        """
        Handles an incoming chunk: if a conversation with the given session_id exists, it adds the chunk to it;
        otherwise, it creates a new conversation and then adds the chunk.
        
        Args:
            session_id (str): The session identifier.
            chunk_number (int): The number of the chunk.
            chunk_file_path (str): The file path where the chunk is stored.
            chunk_name (str): The file name of the chunk.
            chunk_type (str): The type of the chunk ("first", "middle", or "final").
        
        Returns:
            Conversation: The conversation with the chunk added.
        """
        conversation = self.conversations.get(session_id)
        if conversation is None:
            conversation = Conversation(session_id)
            self.conversations[session_id] = conversation

        conversation.add_chunk(chunk_number, chunk_file_path, chunk_name, chunk_type)
        
        # Optionally, if the conversation is complete, remove it from the controller.
        if conversation.is_complete():
            logging.info(f"Conversation {session_id} is complete. Chunks: {conversation.chunks}")
            # Uncomment the following line to remove completed conversations:
            # del self.conversations[session_id]
        
        return conversation
    
    def get_conversation_data(self, session_id: str):
        """
        Returns the data for a given conversation if it exists.
        
        Args:
            session_id (str): The unique session id of the conversation.
        
        Returns:
            dict or None: A dictionary containing the conversation's details, or None if not found.
        """
        conversation = self.conversations.get(session_id)
        if conversation is None:
            return None
        return {
            "conversation_id": conversation.conversation_id,
            "chunks": conversation.chunks,
            "final_chunk_received": conversation.final_chunk_received,
            "missing_chunks": conversation.get_missing_chunks()
        }

    def get_whisper_outputs(self):
        """
        Returns a list of dictionaries for each chunk across all conversations,
        containing the conversation ID, chunk number, and the whisper output.

        Returns:
            list: A list of dictionaries, each with keys:
                'conversation_id', 'chunk_number', and 'whisper_output'.
        """
        whisper_dict = {}
        try:
            for conv_id, conversation in self.conversations.items():
                outputs = []
                for chunk_number, chunk in conversation.chunks.items():
                    outputs.append({
                        #"conversation_id": conv_id,
                        #"chunk_number": chunk_number,
                        "whisper_output": chunk.get("whisper_output")
                    })
                whisper_dict[conv_id] = outputs
        except Exception as e:
            logging.error("Error retrieving whisper outputs: %s", e)
        return whisper_dict

    # def get_all_conversation_summary(self):
    #     try:
    #         summary = []
    #         for conv_id, conversation in self.conversations.items():
    #             converted_count = sum(1 for chunk in conversation.chunks.values() if chunk["chunk_converted"] is True)
    #             whisper_count = sum(1 for chunk in conversation.chunks.values() if chunk.get("whisper_converted") is True)
    #             summary.append({
    #                 "conversation_id": conv_id,
    #                 "file_count": len(conversation.chunks),
    #                 "converted_count": converted_count,
    #                 "whisper_count": whisper_count
    #             })
    #             summary.append(self.get_whisper_outputs().get(conv_id))
    #             #summary.append(self.)    
    #         return summary
    #     except Exception as e:
    #         logging.error(f"Error in get_all_conversation_summary: {e}")
    #         return []



    def get_all_conversation_summary(self):
        try:
            summary = []
            for conv_id, conversation in self.conversations.items():
                converted_count = sum(1 for chunk in conversation.chunks.values() if chunk["chunk_converted"] is True)
                whisper_count = sum(1 for chunk in conversation.chunks.values() if chunk.get("whisper_converted") is True)
                summary.append({
                    "conversation_id": conv_id,
                    "file_count": len(conversation.chunks),
                    "converted_count": converted_count,
                    "whisper_count": whisper_count,
                    "whisper_outputs": self.get_whisper_outputs().get(conv_id),
                    "category_details": conversation.get_category_details()
                })
            return summary
        except Exception as e:
            logging.error(f"Error in get_all_conversation_summary: {e}")
            return []



# Example usage:
if __name__ == "__main__":
    controller = ConversationController()
    
    # Simulate handling chunks for a conversation.
    session_id = "session-" + uuid.uuid4().hex
    
    # First chunk
    filepath1 = "uploads/file1.webm"
    filename1 = "file1.webm"
    controller.handle_chunk(session_id=session_id, chunk_number=0, chunk_file_path=filepath1, chunk_name=filename1, chunk_type="first")
    
    # Middle chunk
    filepath2 = "uploads/file2.webm"
    filename2 = "file2.webm"
    controller.handle_chunk(session_id=session_id, chunk_number=1, chunk_file_path=filepath2, chunk_name=filename2, chunk_type="middle")
    
    # Final chunk
    filepath3 = "uploads/file3.webm"
    filename3 = "file3.webm"
    conversation = controller.handle_chunk(session_id=session_id, chunk_number=2, chunk_file_path=filepath3, chunk_name=filename3, chunk_type="final")
    
    if conversation.is_complete():
        print("Conversation is complete!")
    else:
        print("Missing chunks:", conversation.get_missing_chunks())
    
    print("Conversation data:", controller.get_conversation_data(session_id))
