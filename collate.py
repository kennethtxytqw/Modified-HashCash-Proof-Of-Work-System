#!/usr/bin/python3

import sys
from os import listdir
from os.path import isfile, join
import re


dataFilePattern = re.compile("(\d+)-(\d+)\.err(\d+)")
genTimePattern = re.compile("Generation time: (\d+\.\d+) seconds\.")

data = {}

def main(argv):
    inputDirName = argv[0]
    filelist = [f for f in listdir(inputDirName) if isfile(join(inputDirName, f))]
    
    for filename in filelist:
        m = dataFilePattern.match(filename)
        if not m:
            continue
        
        expNum = int(m.group(1))
        threadNum = int(m.group(2))

        if expNum not in data:
            data[expNum] = {}
        if threadNum not in data[expNum]:
            data[expNum][threadNum] = {}
            data[expNum][threadNum]["Avg"] = 0
            data[expNum][threadNum]["Count"] = 0
            data[expNum][threadNum]["Min"] = None
            data[expNum][threadNum]["Max"] = None

        with open(join(inputDirName, filename)) as datafile:
            for line in datafile:
                m = genTimePattern.match(line)
                if not m:
                    continue
                
                genTime = float(m.group(1))

                if not data[expNum][threadNum]["Min"] or data[expNum][threadNum]["Min"] > genTime:
                    data[expNum][threadNum]["Min"] = genTime
                if not data[expNum][threadNum]["Max"] or data[expNum][threadNum]["Max"] < genTime:
                    data[expNum][threadNum]["Max"] = genTime

                data[expNum][threadNum]["Avg"] += genTime
                data[expNum][threadNum]["Count"] += 1
                break
    

    with open(join(inputDirName, "consolidate.out"), 'w') as resultFile:
        resultFile.write("Input, Num of Threads, Avg Time Taken(s), Max Time Taken(s), Min Time Taken(s)\n")
        for expNum in sorted(data.keys()):
            for threadNum in sorted(data[expNum].keys()):
                resultFile.write(str(expNum))
                resultFile.write(", ")
                resultFile.write(str(threadNum))
                resultFile.write(", ")
                resultFile.write(str(data[expNum][threadNum]["Avg"] / data[expNum][threadNum]["Count"]))
                resultFile.write(", ")
                resultFile.write(str(data[expNum][threadNum]["Max"]))
                resultFile.write(", ")
                resultFile.write(str(data[expNum][threadNum]["Min"]))
                resultFile.write("\n")

if __name__ == "__main__":
   main(sys.argv[1:])