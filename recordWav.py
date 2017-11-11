import pyaudio
import wave
 
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
# WAVE_OUTPUT_FILENAME = "file.wav"

filename_dict = {
    0: "Hello",
    1: "Mouse",
    2: "Dog",
    3: "Omar",
    4: "Stinks",
}
  
for i in range(5):
    if i == 0:
        print("Say \'Hello\'")
    elif i == 1:
        print("Say \'Mouse\'")
    elif i == 2:
        print("Say \'Dog\'")
    elif i == 3:
        print("Say \'Omar\'")
    else:
        print("Say \'Stinks\'")
        
    for j in range(10):
        audio = pyaudio.PyAudio()
        # start Recording
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
        print("recording...")
        frames = []
         
        # for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        ui_key = "0"
        while(ui_key != ""):
            ui_key = input()
            data = stream.read(CHUNK)
            frames.append(data)
        print("finished recording insance number {}".format(j + 1))
         
         
        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()
         
        WAVE_OUTPUT_FILENAME = "samples/" + filename_dict[i] + "_{}".format(j) + ".wav"
         
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()