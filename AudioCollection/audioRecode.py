# import required libraries
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import os
from datetime import datetime

# Sampling frequency
freq = 44400

# Recording duration in seconds
duration = 10

# dataset_folder
folder_name = "dataset/"
if not os.path.exists(folder_name):
	os.mkdir(folder_name)

# Save File
for i in range(30):
	# to record audio from
	# sound-device into a Numpy
	recording = sd.rec(int(duration * freq),
					samplerate = freq, channels = 2)

	# Wait for the audio to complete
	sd.wait()

	# using wavio to save the recording in .wav format
	# This will convert the NumPy array to an audio
	# file with the given sampling frequency

	file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+".wav"
	wv.write(folder_name+file_name, recording, freq, sampwidth=2)

	print(file_name)

print("Done!")