import simpleaudio as sa

class Alarm:
    def play(self):
        while True:
            wave_obj = sa.WaveObject.from_wave_file("./alarm.wav")
            play_obj = wave_obj.play()
            play_obj.wait_done()