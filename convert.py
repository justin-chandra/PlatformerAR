import pyaudio
import wave
import sys

CHUNK = 1024

if len(sys.argv) < 2:
	print("Plays a wave file.\n\Usage: %s filename.wav" % sys.argv[0])
	sys.exit(-1)

file = scipy.io.wavfile.read(sys.argv[1])
print(a[1])


