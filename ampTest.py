# import imp
# from msilib.schema import Class
# from os import getcwd
# from re import X
# import librosa.display
# import matplotlib.pyplot as plt
from json import load
import math
import librosa
import os
import numpy as np
import tensorflow as tf
from pydub import AudioSegment
# import cv2





class AmplitudeTester:
    
    def __init__(self) -> None:
        self.__isLoadFile = False
        self.__fileNameList = []
        self.__fileDataList = []
        self.__inputDatas = []
        self.__resultTable = [
            "住警器",
            "拍手聲",
            "說話聲",
            "住警器V",
            "Baby Cry",
            "Cat",
            "Snore",
            "Rain",
            "Door Knock",
            "quiet",
            "警示",
            "警示",
            "警示",
            "喇叭聲",
            "移車",
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
        
    # def volumeAdjustByAudioSegment(self,db:int):
    #     print(self.__fileNameList[0])

    #     if not self.__isLoadFile:
    #         print("Don't run volumeAdjust before loadFile!")
    #         return

    #     for file in self.__fileNameList:
    #         sound = AudioSegment.from_wav(file)
            
    #         sound = sound + db
    #         file = file[0:-3]
    #         print(file)
    #         # sound.export(file,"wav")

    
    def volumeAdjustByAmp(self,db:int):

        if not self.__isLoadFile:
            print("Don't run volumeAdjust before loadFile!")
            return
        # amplitude is from db's formula -> 20*log(a/b)=db -> log(a/b)=db/20 -> a/b=10^(db/20)
        amp = math.pow(10,db/20)
        


    def loadWavFile(self,path:str = r"C:\Users\Aurismart_Ray\AppData\Local\librosa\librosa\Cache" ):
        roots = os.walk(path)
        self.__isLoadFile = True
        for parent, dirnames, filenames in roots:
            for filename in filenames:
                if(filename[-3:] == "wav"):
                    print(filename[-3:])
                    y,sr = librosa.load(path + '\\'+filename,sr=16000)
                    self.__fileNameList.append(path + '\\'+filename)
                    self.__fileDataList.append(y)
        if self.__fileDataList == None:
            print("didn't find and wav file,please check your path")


    def doLibrosa(self) -> int:

        if not self.__isLoadFile:
            print("Don't doLibrosa before loadFile!")
            return 

        for y in self.__fileDataList:
            for i in range(6):
                input = []
                pos = i*3200
                data = y[pos:pos+15999]
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
                print(index," : ",probability)
            print("-"*5,"以上為第{}次辨識結果".format(indexOfInputDatas+1),"-"*5)
        print(f"最終辨識結果 : {self.__resultTable[result]}({result})    最大機率為 : {maxProbability}")

amplitudeTester = AmplitudeTester()
amplitudeTester.loadWavFile()
amplitudeTester.volumeAdjustByAudioSegment(3)
# amplitudeTester.doLibrosa()


