
import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))

def readname():
    filepath = path + '\\outputs\\deploy'
    name = os.listdir(filepath)
    return name

if __name__=="__main__":
    name = readname()
    print(name)
    print(len(name))

