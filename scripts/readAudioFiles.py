import wave

wav_file = '../data/wav_files/DVA1A.wav'
obj = wave.open(wav_file, 'rb')
print(obj.getnchannels())
length = obj.getnframes() / obj.getframerate()
print(length)



obj_new = wave.open(wav_file, 'wb')
obj_new.setnchannels(obj.getnchannels())
obj_new.setsampwidth(obj.getsampwidth())
obj_new.setframerate(obj.getframerate())
obj_new.write(obj.readframes(-1))

obj.close()
obj_new.close()