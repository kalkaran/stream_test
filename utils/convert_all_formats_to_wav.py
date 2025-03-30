import os
import shutil
import subprocess
from pydub import AudioSegment
from pydub.utils import mediainfo
import re
import sys


class AudioConverter:
    """
    As there are a few different possible ways the sound files will be passed through the system.
    I have created a few different ways to use this.

    Behaviour:
        Requires: input_file.
            Will create and save wav file with same name to same folder as input file (xx.mp4) begets xx.wav
        Optional: output_folder.
            Will set output directory for the wav file (xx.mp3, /folder/) will create a wav with same name in the /folder/
        Optional: id
            Will move the input_file to a folder named the id parameter, and create the wav file there too.
        Optional: output_folder, id
            Will move the input_file to a folder named the id parameter at the path passed in the output_folder parameter,
             and create the wav file there too.


    A class for converting audio and extracting sound from video files to WAV format.

    Supports all audio and video formats that FFmpeg can handle.
    Checks which system can support.
    The class checks the file format of media (via pydub)
    attempts to convert it to WAV if the format is supported.

    """

    def __init__(self, input_file, output_folder=None, id=None):
        self.input_file = input_file
        self.id = id
        self.supported_formats = self._get_supported_formats()

        # Determine the output folder: if no output_folder is provided and an ID is given, use the input file's directory.
        if output_folder is not None:
            self.output_folder = output_folder
        elif self.id:
            self.output_folder = os.path.dirname(self.input_file)
        else:
            self.output_folder = os.path.dirname(self.input_file)

        if self.id:
            self._create_id_subfolder()

        self.file_format = None
        self.audio_codec = None

        self.output_file = self._get_output_file_path()
        self._check_ffmpeg_installed()
        self.file_format = self._get_file_format()

    def __call__(self):
        """
        Initializes an instance of AudioConverter.

        Parameters:
        ----------
        input_file : str
            The path to the input file (audio or video).
        output_folder : str, optional
            The folder where the converted WAV file will be saved. If None, defaults to the input file's directory.
        id : str, optional
            An optional identifier used to create a subfolder for organizing files.
        """
        self.convert_to_wav()
        return self.input_file, self.output_file, self.output_folder, self.output_file_name

    def _check_ffmpeg_installed(self):
        """Check if FFmpeg is installed on the system."""
        try:
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("FFmpeg is installed.")
        except FileNotFoundError:
            raise EnvironmentError("FFmpeg is not installed. Please install FFmpeg to use this tool.")

    def _create_id_subfolder(self):
        """Creates a subfolder named after the ID and moves the input file to it."""
        self.id_folder = os.path.join(self.output_folder, self.id)
        os.makedirs(self.id_folder, exist_ok=True)

        new_input_path = os.path.join(self.id_folder, os.path.basename(self.input_file))
        shutil.move(self.input_file, new_input_path)
        self.input_file = new_input_path
        print(f"Input file moved to {self.input_file}")

        self.output_folder = self.id_folder

    def _get_output_file_path(self):
        """Generates the output file path for the converted WAV file."""
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        filename = os.path.basename(self.input_file)
        self.output_file_name = os.path.splitext(filename)[0] + '.wav'
        return os.path.join(self.output_folder, self.output_file_name)

    def _get_supported_formats(self):
        """Dynamically fetch the formats supported by FFmpeg."""
        try:
            result = subprocess.run(
                ['ffmpeg', '-formats'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            output = result.stdout + result.stderr

            # Parse the output to find supported formats
            demuxers = set()
            muxers = set()
            start_parsing = False

            # Regular expression pattern to match 'D', 'E', or 'DE' anywhere in the line
            # Regular expression pattern to match 'D', 'E', or 'DE' at the start of the line
            pattern = re.compile(r'^\s*(D?E?)\s+(\w+)')

            for line in output.splitlines():
                match = pattern.match(line)
                if match:
                    flag = match.group(1)  # 'D', 'E', or 'DE'
                    format_name = match.group(2)  # The actual format name

                    if 'D' in flag:
                        demuxers.add(format_name)
                    if 'E' in flag:
                        muxers.add(format_name)

            # Debug output to verify supported formats
            # print("Supported demuxers:", demuxers)
            # print("Supported muxers:", muxers)

            return {
                'demuxers': demuxers,
                'muxers': muxers
            }
        except Exception as e:
            print(f"Error checking FFmpeg supported formats: {e}")
            return None

    def _get_file_format(self):
        """
        Get the actual file format using the mediainfo function from pydub.
        This method reads the metadata from the file itself and checks if the format matches a known supported format.
        """
        try:
            info = mediainfo(self.input_file)
            if 'format_name' in info and info['format_name']:
                format_list = [fmt.strip().lower() for fmt in info['format_name'].split(',')]
            else:
                format_list = []
            
            if format_list:
                print(f"The file media info matches the following formats: {', '.join(format_list)}")
            else:
                print("Warning: No media information was found. Proceeding with conversion anyway.")
            self.file_formats = format_list
            
            # Extract audio codec
            audio_codec = None
            ffprobe_cmd = ['ffprobe', '-i', self.input_file, '-show_streams', '-select_streams', 'a', '-loglevel', 'error']
            ffprobe_output = subprocess.run(ffprobe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            
            for line in ffprobe_output.stdout.splitlines():
                if "codec_name=" in line:
                    audio_codec = line.split("=")[1].strip()
                    break
            
            # #Check if any format from mediainfo matches a supported demuxer format from FFmpeg
            # compatible_formats = [fmt for fmt in format_list if fmt in (fmt.lower() for fmt in self.supported_formats['demuxers'])]
            
            if audio_codec:
                print(f"Decoding Codec matches: {audio_codec}")
            self.audio_codec = audio_codec
        except Exception as e:
            print(f"Unable to determine file decoder: {e}")


    def convert_to_wav(self):
        """
        Convert the input file to WAV format or extract audio if it's a video file.
        """
        if self.file_formats is None:
            print(f"Unrecognized format for {self.input_file}. Trying conversion anyway...")

        self._convert_audio_to_wav()

    def _convert_audio_to_wav(self):
        """Convert any audio or video file to WAV format while preserving correct codec handling and channel count."""
        try:
            # Detect original channel count
            ffprobe_cmd = ['ffprobe', '-i', self.input_file, '-show_streams', '-select_streams', 'a', '-loglevel', 'error']
            ffprobe_output = subprocess.run(ffprobe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            
            channels = 1  # Default to mono
            for line in ffprobe_output.stdout.splitlines():
                if "channels=" in line:
                    channels = line.split("=")[1].strip()
                    break
        except Exception as e:
            print(f"Error detecting channel count for {self.input_file}: {e}")
            channels = 1  # Fallback to mono if detection fails

        # Now proceed with converting the file
        try:
            #cmd = ['ffmpeg', '-i', self.input_file, '-ar', '16000', '-ac', str(channels), '-b:a', '256k', '-f', 'wav', self.output_file]
            
            # Apply codec-specific handling
            codec_map = {
                "opus": "libopus",
                "aac": "aac",
                "vorbis": "vorbis",
                "flac": "flac",
                "pcm_s16le": "pcm_s16le",
                "amr_nb": "amrnb",
                "amr_wb": "amrwb",
                "wma": "wmav2"
            }
            
            # cmd = ['ffmpeg', '-y', '-i', self.input_file]
            # if self.audio_codec in codec_map:
            #     cmd.extend(['-c:a', codec_map[self.audio_codec]])
            #cmd.extend(['-ar', '16000', '-ac', str(channels), '-b:a', '256k', '-f', 'wav', self.output_file])
            #cmd.extend(['-ar', '16000', '-ac', "1", '-b:a', '256k', '-f', 'wav', self.output_file])
            #cmd.extend(['-c:a', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-b:a', '256k', '-f', 'wav', self.output_file])
            # cmd.extend(['-c:a', 'pcm_s16le', '-ar', '16000', '-ac', str(channels), '-b:a', '256k', '-f', 'wav', self.output_file])
            cmd = ['ffmpeg', '-y']
            if self.audio_codec in codec_map:
                # Force the decoder from the codec map for the input
                cmd.extend(['-c:a', codec_map[self.audio_codec]])
            cmd.extend(['-i', self.input_file])
            cmd.extend(['-c:a', 'pcm_s16le', '-ar', '16000', '-ac', str(channels), '-b:a', '256k', '-f', 'wav', self.output_file])





            
            print(f"Converting with command: {cmd}")
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            print(f"Converted {self.input_file} to WAV format.")
        except Exception as e:
            print(f"Error converting {self.input_file} to WAV: {e}")




if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Only input_file provided
        input_file = sys.argv[1]
        converter = AudioConverter(input_file)()
    elif len(sys.argv) == 3:
        # input_file and output_folder provided
        input_file = sys.argv[1]
        output_folder = sys.argv[2]
        converter = AudioConverter(input_file, output_folder)()
    else:
        print("Usage: python3 convert_file_to_wave.py <input_file>")
        print("Usage: python3 convert_file_to_wave.py <input_file> <output_folder>")
        sys.exit(1)

