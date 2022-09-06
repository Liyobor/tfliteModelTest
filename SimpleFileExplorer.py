from genericpath import isdir
import os

class SimpleFileExplorer:

    def __init__(self,path) -> None:
        self.__path = path
    

    def getDirs(self) -> list:
        '''
        This function will return directory list
        '''
        dirs = []
        for item in os.scandir(self.__path):
            if item.is_dir():
                dirs.append(item.path)
        return dirs

    def getFiles(self) -> list:
        '''
        This function will return files list
        '''
        files = []
        for item in os.scandir(self.__path):
            if item.is_file():
                files.append(item.name)
        return files