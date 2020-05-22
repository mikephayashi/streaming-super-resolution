import simpleaudio as sa


class Alarm:

    def play(self, song_path):
        while True:
            wave_obj = sa.WaveObject.from_wave_file(song_path)
            play_obj = wave_obj.play()
            play_obj.wait_done()

    def hooray(self):
        self.play("./alarm.wav")

    def oops(self):
        self.play("./police.mp3")
