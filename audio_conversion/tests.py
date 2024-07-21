import unittest
import os
from pydub.generators import Sine
from audio_conversion import convert_audio


class TestAudioConverter(unittest.TestCase):

    def setUp(self):
        """
        Creates a 20 second audio file used for testing purposes
        """
        self.input_file = "test_audio.mp3"
        self.output_file_wav = "test_audio.wav"
        self.output_file_invalid = "test_audio.invalid"

        if not os.path.isfile(self.input_file):
            sine_wave = Sine(440).to_audio_segment(duration=20000)
            sine_wave.export(self.input_file, format="mp3")

    def tearDown(self):
        """
        Cleans up after the setUp
        """
        if os.path.isfile(self.input_file):
            os.remove(self.input_file)
        if os.path.isfile(self.output_file_wav):
            os.remove(self.output_file_wav)
        if os.path.isfile(self.output_file_invalid):
            os.remove(self.output_file_invalid)

    def test_1(self):
        """
        Tests for successful audio conversion
        """
        convert_audio(self.input_file, "wav")
        self.assertTrue(os.path.isfile(self.output_file_wav))

    def test_2(self):
        """Tests if the file does not exists"""
        with self.assertRaises(FileNotFoundError):
            convert_audio("do_not_exist_file.mp3", "wav")

    def test_3(self):
        """
        Tests for correct file format
        """
        with self.assertRaises(ValueError):
            convert_audio(self.input_file, "invalid_format")


if __name__ == "__main__":
    unittest.main()
