import pygame
import time

pygame.mixer.init()

sound = pygame.mixer.Sound(
    'app/static/alarm/alarm.wav'
)

sound.play()

time.sleep(5)