from doctest import testfile
from ModelTester import ModelTester
from SimpleFileExplorer import SimpleFileExplorer

path = input("input\n")
explorer = SimpleFileExplorer(path)
files = explorer.getFiles()
tester = ModelTester(model_path='keras_model.tflite',outputName='wav.xlsx')
# print(tester.model_inputDetails)
# for file in files:
#     if(file[-3:]=="wav"):
#         filePath = path+'\\'+file
#         tester.loadWavFile(filePath=filePath)
#         # tester.showMelSpectrogramArray()
#         tester.doLibrosa()
# tester.exportDeatils()
# tester.exportResult()
# tester.reset()


for file in files:
    if(file[-3:]=='bin'):
        filePath = path+'\\'+file
        tester.loadBinFile(filePath=filePath)
        # tester.showMelSpectrogramArray()
        tester.doLibrosa()
tester.exportResult()
tester.reset()