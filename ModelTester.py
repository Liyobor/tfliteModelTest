import math
import librosa
import os
import numpy as np
import tensorflow as tf
from pydub import AudioSegment
import pandas as pd
import struct

class ModelTester():
    
    def __init__(self,model_path,outputName:str) -> None:
        self.__volumeIncrement = 0
        self.__isLoadFile = False
        self.__isLoadBin = False
        self.__fileName:str = ""
        self.__fileData:list = []
        self.__inputDatas = []
        self.__strLen = 0
        self.loudness = "Not load WAV file yet!"
        self.max_dBFS = "Not load WAV file yet!"
        self.__normalizedDb = None
        self.__result ={}
        self.__fileNameList = []
        self.__outPutName = outputName
        # Load model
        self.__interpreter = tf.lite.Interpreter(model_path)
        self.__interpreter.allocate_tensors()
        self.__repeatTimes = 6

        # get input/output information
        input_details = self.__interpreter.get_input_details()
        self.__modelInputIndex = input_details[0]['index']
        output_details = self.__interpreter.get_output_details()
        self.__outPutTensorIndex = output_details[0]['index']
        self.model_inputDetails = self.__interpreter.get_input_details()

        
        self.__resultTable = [
            "住警器A",                  #0
            "拍手聲",                   #1
            "說話聲",                   #2
            "住警器Voice",              #3
            "Baby Cry",                #4
            "Cat",                     #5
            "Snore",                   #6
            "Rain",                    #7
            "Door Knock",              #8
            "quiet",                   #9
            "救護車",                   #10
            "消防車",                   #11
            "警車",                    #12
            "喇叭聲",                  #13
            "警車-喇叭",                #14
            "Thunder",                #15
            "brakingSound",           #16
            "MotorVehicleCrash",      #17
            "TrashCar",               #18
            "TempleActivity",         #19
            "ConstructionSite",       #20
            "Dog",                    #21
            "LoudEngine",             #22
            "TrainWhistle",           #23
            "TurningWarning",         #24
            "Whistle",                #25
            "ElectricMotorEngine",    #26
            "喇叭聲",                  #27
            "喇叭聲",                  #28
            "警車",                    #29
            "警車",                    #30
            "GlassBreaking",          #31
            "GarbageTruckToAlice",    #32
            "住警器B",                 #33
            "住警器C",                 #34
            "住警器D",                 #35
            ]
        self.__detail = {}
        for i in range(len(self.__interpreter.get_tensor(self.__outPutTensorIndex)[0])):
            self.__detail[i] = []

    def volumeAdjustByAmp(self,db:int,callByNormalized:bool = False):

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
        if not callByNormalized:
            self.__volumeIncrement = db

    def NormalizedMaxDBFS(self,targetDb:int):
        if not self.__isLoadFile:
            print("Don't run volumeNormalizedDb before loadFile!")
            return

        if targetDb == self.max_dBFS:
            return
        
        if targetDb>0:
            print("targetDb must < 0 !")
            return

        self.volumeAdjustByAmp(targetDb - self.max_dBFS,callByNormalized=True)
        self.max_dBFS = targetDb
        self.__normalizedDb = targetDb

    def loadBinFile(self,filePath:str):
        self.__fileName = filePath[filePath.rfind("\\")+1:]
        if self.__fileName == "" or self.__fileName[-4:] != '.bin':
            print("Path error!")
            return
        data = []
        bin = open(filePath,'rb')
        for i in range(int(os.path.getsize(filePath)/4)):
            floatTuple = struct.unpack('f',bin.read(4))
            # print("tuple = ",floatTuple[0])
            # print("type = ",type(floatTuple[0]))
            floatData = floatTuple[0]
            data.append(np.float32(floatData))
        data = np.array(data).reshape(1,128,63,1)
        data = data.astype(np.float32)
        self.__inputDatas.append(data)
        self.loudness = None
        self.max_dBFS = None
        self.__isLoadBin = True

    def loadWavFile(self,filePath:str,samplerate:int = 16000):  
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
        # print("file name = ",self.__fileName)
        # print("max_dBFS = ",sound.max_dBFS)
        self.loudness = sound.dBFS
        self.max_dBFS = sound.max_dBFS
        # print("loudness = ",self.loudness)
        # print()
        self.__isLoadFile = True


    def showMelSpectrogramArray(self,stepLength = 1600,repeatTimes = 10):
        if not self.__isLoadFile:
            print("Don't doLibrosa before loadFile!")
            return 
        for i in range(repeatTimes):
            pos = i*stepLength
            data = self.__fileData[pos:pos+15999]
            melSpectrogram = librosa.feature.melspectrogram(y=data, sr=16000,n_fft=1024 ,n_mels=128,hop_length=256)
            xAxis = len(melSpectrogram)
            yAxis = len(melSpectrogram[0])
            count = 0
            print(f"filename:{self.__fileName[0:-4]}_{i}{self.__fileName[-4:]}"+"-"*10)
            for xIndex in range(xAxis):
                for yIndex in range(yAxis):
                    if count <= 20:
                        print(f"({melSpectrogram[xIndex][yIndex]})")
                        count +=1

    def doLibrosa(self,stepLength = 3200,repeatTimes = 6):

        if not self.__isLoadFile and not self.__isLoadBin:
            print("Don't doLibrosa before loadFile!")
            return 
        self.__repeatTimes = repeatTimes
        if self.__isLoadFile:
            for i in range(repeatTimes):
                input = []
                pos = i*stepLength
                data = self.__fileData[pos:pos+15999]
                melSpectrogram = librosa.feature.melspectrogram(y=data, sr=16000,n_fft=1024 ,n_mels=128,hop_length=256)
                xAxis = len(melSpectrogram)
                yAxis = len(melSpectrogram[0])
                count = 0
                for xIndex in range(xAxis):
                    for yIndex in range(yAxis):
                        if count <51:
                            # print(f"filename:{self.__fileName} , count {count}:{melSpectrogram[xIndex][yIndex]}")
                            count +=1
                        input.append(melSpectrogram[xIndex][yIndex])
                # print("----next round----")
                input = np.array(input)
                input = input.reshape(1,128,63,1)
                input = input.astype(np.float32)
                self.__inputDatas.append(input)
            self.__putInModel()
        elif self.__isLoadBin:
            self.__putInModel()

    

    


    def __putInModel(self):
        maxProbability= 0
        probabilityList = [0.0]*len(self.__interpreter.get_tensor(self.__outPutTensorIndex)[0])
        result = -1
        
        for indexOfInputDatas,input in enumerate(self.__inputDatas):
            
            # input_details = self.__interpreter.get_input_details()
            # output_details = self.__interpreter.get_output_details()
            # index = input_details[0]['index']
            # print('input = ',input.shape)
            self.__interpreter.set_tensor(self.__modelInputIndex,input)
            self.__interpreter.invoke()
            # output_data = self.__interpreter.get_tensor(output_details[0]['index'])
            output_data = self.__interpreter.get_tensor(self.__outPutTensorIndex)
            # print("output_data[0] len = ",len(output_data[0]))
            for index,probability in enumerate(output_data[0]):
                probabilityList[index] += probability
                # print(f'index = {index}, prob = {probability}')

                # maxProbability = max(probability,maxProbability)
                # if maxProbability == probability:
                #     result = index
                self.__detail[index].append(probability)

            #     print(index," : ",probability)
            # print("-"*5,"以上為第{}次辨識結果".format(indexOfInputDatas+1),"-"*5)
        # print(self.__fileName)
        # print(probabilityList)
        
        
        strPattern0 = f"{self.__fileName}"
        if(self.max_dBFS != None):
            strPattern1 = f"{round(self.max_dBFS+self.__volumeIncrement,4)}db"
        else:
            strPattern1 = "NaN"
        if self.__volumeIncrement >= 0:
            strPattern2 = f"+{round(self.__volumeIncrement,4)}db"
        else:
            strPattern2 = f"{round(self.__volumeIncrement,4)}db"
        strPattern3 = f"{self.__resultTable[probabilityList.index(max(probabilityList))]}({probabilityList.index(max(probabilityList))})"
        # print(f'result  = {strPattern3}')
       
        
        strPattern4 = f"{round(float(max(probabilityList)),4)}"
        # print(f'max probability  = {strPattern4}')

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
        frame = pd.DataFrame(self.__result)
        # file = 'Result.xlsx'
        file = self.__outPutName
        frame.index = ['File Name','Max dBFS','Volume Increment','Result','Max Probability']
        try:
            if os.path.isfile(file):
                with pd.ExcelWriter(file,mode="a") as writer:
                    if self.__volumeIncrement >= 0:
                        frame.to_excel(excel_writer=writer,sheet_name="+"+str(self.__volumeIncrement)+"db")
                    else:
                        frame.to_excel(excel_writer=writer,sheet_name=str(self.__volumeIncrement)+"db")
            else:
                with pd.ExcelWriter(file) as writer:
                    if self.__volumeIncrement >= 0:
                        frame.to_excel(excel_writer=writer,sheet_name="+"+str(self.__volumeIncrement)+"db")
                    else:
                        frame.to_excel(excel_writer=writer,sheet_name=str(self.__volumeIncrement)+"db")
        except Exception as e:
            print(f"error occur when save data to {file} file,error message : {e}")
        

    def exportDeatils(self):
        frame = pd.DataFrame(self.__detail).T
        columns = list(frame.columns)
        
        
        # print(self.__fileNameList)
        for index,value in enumerate(columns):
            columns[index] = value%(self.__repeatTimes)+1
        frame.columns = columns
        
        # frame.insert(loc=0,column = self.__fileNameList[0],value=None)
        print(self.__fileNameList)
        frame.insert(loc=0,column=self.__fileNameList[0],value=None)
        
        nameIndex = 1
        
        # print(f'----{len(frame.columns)}')
        for i in range(len(frame.columns)+len(self.__fileNameList)-2):
        
            if frame.columns[i] == (self.__repeatTimes):
                # frame.insert(loc=i+1,column=self.__fileNameList[nameIndex],value=None)
                frame.insert(loc=i+1,column=self.__fileNameList[nameIndex],value=None)
                # print(nameIndex)
                nameIndex +=1
        # print(f'----{len(frame.columns)}')
        isNoneColumnAdd = False
        for i in range(len(frame.columns)+len(self.__fileNameList)-2):
            if isNoneColumnAdd:
                isNoneColumnAdd = False
                continue
            if type(frame.columns[i+1]) != int:
                frame.insert(loc=i+1,column=None,value=None,allow_duplicates=True)
                isNoneColumnAdd = True
        # file = 'Details_Of_Result.xlsx'
        file = f'Details_Of_{self.__outPutName}'
        try:
            if os.path.isfile(file):
                with pd.ExcelWriter(file,mode="a") as writer:
                    if self.__volumeIncrement >= 0:
                        frame.to_excel(excel_writer=writer,sheet_name="+"+str(self.__volumeIncrement)+"db")
                    else:
                        frame.to_excel(excel_writer=writer,sheet_name=str(self.__volumeIncrement)+"db")
                    
            else:
                with pd.ExcelWriter(file) as writer:
                    if self.__volumeIncrement >= 0:
                        frame.to_excel(excel_writer=writer,sheet_name="+"+str(self.__volumeIncrement)+"db")
                    else:
                        frame.to_excel(excel_writer=writer,sheet_name=str(self.__volumeIncrement)+"db")
        except Exception as e:
            print(f"error occur when save data to Details_Of_Result.xlsx file,error message : {e}")
        

    def __reset(self):
        self.__fileData = []
        self.__fileName = ""
        self.__inputDatas.clear()
        self.__isLoadFile = False
        self.loudness = "Not load WAV file yet!"
        self.max_dBFS = "Not load WAV file yet!"

    def reset(self):
        self.__volumeIncrement = 0
        self.__strLen = 0
        self.__normalizedDb = None
        self.__result.clear()
        self.__fileNameList.clear()
        self.__detail = {}
        for i in range(len(self.__interpreter.get_tensor(self.__outPutTensorIndex)[0])):
            self.__detail[i] = []



            


