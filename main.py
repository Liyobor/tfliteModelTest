import os
from ModelTester import ModelTester
from alive_progress import alive_bar



# Entry Point
# path = r"C:\Users\Aurismart_Ray\AppData\Local\librosa\librosa\Cache\\"


# sampleBit = input("input the sample bit of these audio data:\n")
# soundLevel = input("input the loudness level you want:\n")

examplePath = fr"C:\Users\{os.getlogin()}\Desktop\python"
path = input(f"Please enter wav file location e.g. {examplePath} \n")
roots = os.walk(path)
resultFileName = input('enter the result xlsx file name e.g. Result_AMB.xlsx \n')
amplitudeTester = ModelTester(model_path='keras_model.tflite',outputName=resultFileName)
for parent, dirnames, filenames in roots:
    for i in range(5):
        print(f"round {i+1} start,increase decibel is {i*3} db!")
        with alive_bar(len(filenames)) as bar:
            for filename in filenames:
                if filename[-3:] == "wav":
                    filePath = path + '\\' + filename
                    amplitudeTester.loadWavFile(filePath)
                    # print(amplitudeTester.max_dBFS)
                    amplitudeTester.NormalizedMaxDBFS(-20)
                    amplitudeTester.volumeAdjustByAmp(i*3)
                    amplitudeTester.doLibrosa()
                bar()
            amplitudeTester.exportDeatils()
            # amplitudeTester.showResult()
            amplitudeTester.exportResult()
            amplitudeTester.reset()
            