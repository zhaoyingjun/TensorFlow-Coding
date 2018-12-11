from aip import AipSpeech
import pygame
import time
APP_ID ="15119080"
API_KEY ="2Xkrh0GvmRAGZsAOXliUNG5G"
SECRET_KEY ="GSaFjq9y3ZkXLZI2Dkw5cmmLDomWdiUS"
def read(strline):
    client = AipSpeech(APP_ID,API_KEY,SECRET_KEY)
    result = client.synthesis(strline,"zh",1,{
    "spd":1,
    "vol":5,
    "pit":5,
    "per":0
    })
 
#识别正确返回语言二进制 错误返回dict 参照下面错误码.
    if not isinstance(result,dict):
       with open("audio.mp3","wb")as f:
            f.write(result)
#print(result)
    pygame.mixer.init()

    file="audio.mp3"

    track = pygame.mixer.music.load(file)
    pygame.mixer.music.play()
   
