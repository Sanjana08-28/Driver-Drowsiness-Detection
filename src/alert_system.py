import pygame


class AlertSystem:

    def __init__(self):

        pygame.mixer.init()

        self.alarm_playing = False

        self.alarm_sound = pygame.mixer.Sound(
            'app/static/alarm/alarm.wav'
        )

    def play_alert(self):

        if not self.alarm_playing:

            self.alarm_sound.play()

            self.alarm_playing = True

    def stop_alert(self):

        self.alarm_sound.stop()

        self.alarm_playing = False