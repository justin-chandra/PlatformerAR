import pyaudio
import wave

wtf = wave.open('test.wav', 'rb')
data = wtf.readframes(1024)
print(data)