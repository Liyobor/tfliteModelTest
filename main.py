import os
from SimpleFileExplorer import SimpleFileExplorer
from ModelTester import ModelTester
from alive_progress import alive_bar




# Entry Point
# path = r"C:\Users\Aurismart_Ray\AppData\Local\librosa\librosa\Cache\\"


# sampleBit = input("input the sample bit of these audio data:\n")
# soundLevel = input("input the loudness level you want:\n")

examplePath = fr"C:\Users\{os.getlogin()}\Desktop\python"
path = input(f"Please enter root location e.g. {examplePath} \n")
fileExplorer = SimpleFileExplorer(path= path)
roots = fileExplorer.getDirs()
for root in roots:
    print('root = ',root)
    amplitudeTester = ModelTester(model_path='keras_model.tflite',outputName=root[-3:]+'.xlsx')
    for parent, dirnames, filenames in os.walk(root):
        for i in range(5):
            print(f"round {i+1} start,increase decibel is {i*3} db!")
            with alive_bar(len(filenames)) as bar:
                for filename in filenames:
                    if filename[-3:] == "wav":
                        filePath = root + '\\' + filename
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
