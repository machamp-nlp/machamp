import myutils
import os

for name in os.listdir('logs'):
    model = myutils.getModel(name)
    numModels = len(os.listdir('logs/' + name))
    if model != '' and numModels > 1:
        for oldModel in os.listdir('logs/' + name):
            if oldModel not in model:
                cmd = 'rm -rf logs/' + name + '/' + oldModel
                print(cmd)
                os.system(cmd)
    

