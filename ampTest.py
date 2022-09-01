# import imp
# from msilib.schema import Class
# from os import getcwd
# from re import X
# import librosa.display
# import matplotlib.pyplot as plt
import math
from unicodedata import name
import librosa
import os
import numpy as np
import tensorflow as tf
from pydub import AudioSegment
import pandas as pd
# from decimal import *
# import cv2

class AmplitudeTester:
    
    def __init__(self) -> None:
        self.__volumeIncrement = 0
        self.__isLoadFile = False
        self.__fileName:str = ""
        self.__fileData:list = []
        self.__inputDatas = []
        self.__strLen = 0
        self.loudness = "Not load WAV file yet!"
        self.__result ={}
        self.__fileNameList = []
        self.__resultTable = [
            "Fire Alarm",
            "Clapping",
            "Chatting",
            "Fire Alarm Voice",
            "Baby Cry",
            "Cat",
            "Snore",
            "Rain",
            "Door Knock",
            "quiet",
            "Ambulance",
            "Fire Truck",
            "Police Car",
            "Car Horn",
            "Polive Horn",
            "Thunder",
            "TrafficNoise",
            "TrafficAccidence",
            "TrashCar",
            "TempleActivity",
            "ContructionSite",
            "Dog",
            "LoudEngine",
            "TrainWhistle",
            ]
        self.__detail = {}
        for i in range(24):
            self.__detail[i] = []

    def volumeAdjustByAmp(self,db:int):

        if not self.__isLoadFile:
            print("Don't run volumeAdjust before loadFile!")
            return
        # amplitude is from db's formula -> 20*log(a/b)=db -> log(a/b)=db/20 -> a/b=10^(db/20)
        amp = math.pow(10,db/20)
        for index,value in enumerate(self.__fileData):
            if value*amp >= 1.0:
                self.__fileData[index] = 1.0
            elif value*amp <= -1.0:
                self.__fileData[index] = -1.0
            else:
                self.__fileData[index] = value*amp
        self.__volumeIncrement = db


    def loadWavFile(self,filePath:str,samplerate = 16000):  
        self.__fileName = filePath[filePath.rfind("\\")+1:]

        if self.__fileName == "" or self.__fileName[-4:] != '.wav':
            print("Path error!")
            return

        y,sr = librosa.load(path = filePath,sr = samplerate)
        self.__fileData = y

        if len(self.__fileData) == 0:
            print("didn't load av file,please check your path or file")
            return

        if len(self.__fileData) < 32000:
            print("File is too short,the WAV file needs 2 secounds at least!")
            return
        
        sound = AudioSegment.from_wav(filePath)
        self.loudness = sound.dBFS
        self.__isLoadFile = True

    def doLibrosa(self,stepLength = 3200,repeatTimes = 6):

        if not self.__isLoadFile:
            print("Don't doLibrosa before loadFile!")
            return 

        for i in range(repeatTimes):
            input = []
            pos = i*stepLength
            data = self.__fileData[pos:pos+15999]
            melSpectrogram = librosa.feature.melspectrogram(y=data, sr=16000,n_fft=1024 ,n_mels=128,hop_length=256)
            xAxis = len(melSpectrogram)
            yAxis = len(melSpectrogram[0])
            for xIndex in range(xAxis):
                for yIndex in range(yAxis):
                    input.append(melSpectrogram[xIndex][yIndex])
            input = np.array(input)
            input = input.reshape(1,128,63,1)
            input = input.astype(np.float32)
            self.__inputDatas.append(input)
        self.__putInModel()

    def __putInModel(self):
        maxProbability= 0
        result = -1
        interpreter = tf.lite.Interpreter(model_path="keras_model.tflite")
        for indexOfInputDatas,input in enumerate(self.__inputDatas):
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            index = input_details[0]['index']
            interpreter.set_tensor(index,input)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            for index,probability in enumerate(output_data[0]):
                maxProbability = max(probability,maxProbability)
                if maxProbability == probability:
                    result = index
                self.__detail[index].append(probability)
            #     print(index," : ",probability)
            # print("-"*5,"以上為第{}次辨識結果".format(indexOfInputDatas+1),"-"*5)
        
        strPattern0 = f"檔名 : {self.__fileName}"
        strPattern1 = f"聲音響度 : {round(self.loudness,4)}db"
        if self.__volumeIncrement >= 0:
            strPattern2 = f"聲音增幅 : +{round(self.__volumeIncrement,4)}db"
        else:
            strPattern2 = f"聲音增幅 : {round(self.__volumeIncrement,4)}db"
        strPattern3 = f"最終辨識結果 : {self.__resultTable[result]}({result})"
        
        strPattern4 = f"最大機率為 : {round(float(maxProbability),4)}"
    
        self.__result[self.__fileName] = [strPattern0,strPattern1,strPattern2,strPattern3,strPattern4]
        self.__strLen = max(self.__strLen,len(strPattern0),len(strPattern1),len(strPattern2),len(strPattern3))
        self.__fileNameList.append(self.__fileName[0:-4])
        self.__reset()

    def showResult(self):
        for key in self.__result:
            for fragment in self.__result[key]:
                fragment = fragment.ljust(self.__strLen+2)
                print(fragment,end="")
            print()

    def exportResult(self):
        frame = pd.DataFrame(self.__detail).T
        columns = list(frame.columns)
        
        
        # print(self.__fileNameList)
        for index,value in enumerate(columns):
            columns[index] = value%6+1
        frame.columns = columns
        
        # frame.insert(loc=0,column = self.__fileNameList[0],value=None)
        frame.insert(loc=0,column=self.__fileNameList[0],value=None)
        
        nameIndex = 1
        
        print(f'----{len(frame.columns)}')
        for i in range(len(frame.columns)+len(self.__fileNameList)-2):
            # print(frame.columns[i])
            if frame.columns[i] == 6:
                # frame.insert(loc=i+1,column=self.__fileNameList[nameIndex],value=None)
                frame.insert(loc=i+1,column=self.__fileNameList[nameIndex],value=None)
                print(nameIndex)
                nameIndex +=1
        print(f'----{len(frame.columns)}')
        isNoneColumnAdd = False
        for i in range(len(frame.columns)+len(self.__fileNameList)-2):
            if isNoneColumnAdd:
                isNoneColumnAdd = False
                continue
            if type(frame.columns[i+1]) != int:
                frame.insert(loc=i+1,column=None,value=None,allow_duplicates=True)
                isNoneColumnAdd = True

        try:
            with pd.ExcelWriter('Result.xlsx') as writer:
                frame.to_excel(writer)
        except Exception as e:
            print(f"error occur when save data to excel file,error message : {e}")

    def __reset(self):
        self.__fileData = []
        self.__fileName = ""
        self.__inputDatas.clear()
        # self.__detail.clear()
        # self.__result.clear()
        self.__isLoadFile = False
        self.__volumeIncrement = 0
        self.loudness = "Not load WAV file yet!"


# Entry Point
amplitudeTester = AmplitudeTester()
path = r"C:\Users\Aurismart_Ray\AppData\Local\librosa\librosa\Cache\\"
roots = os.walk(path)
# sampleBit = input("input the sample bit of these audio data:\n")
# soundLevel = input("input the loudness level you want:\n")

for parent, dirnames, filenames in roots:
    for filename in filenames:
        if(filename[-3:] == "wav"):
            filePath = path + filename
            amplitudeTester.loadWavFile(filePath)
            if amplitudeTester.loudness != -20:
                amplitudeTester.volumeAdjustByAmp(-15-amplitudeTester.loudness)
            amplitudeTester.doLibrosa()
amplitudeTester.exportResult()
amplitudeTester.showResult()


