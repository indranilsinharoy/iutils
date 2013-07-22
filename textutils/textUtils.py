#-------------------------------------------------------------------------------
# Name:        textUtils.py
# Purpose:
#
# Author:      Indranil
#
# Created:     06/10/2012
# Copyright:   (c) Indranil 2012
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import os, glob


# prompt the user to input the full path name of the file
#filename = input()

directory = "C:\\Users\\Indranil\\Downloads\\"
#pattern = "raytracing.txt"
pattern = "7 - 1 -*.txt"

filenames = glob.glob(directory+pattern)
#consolidated file (text)
fWriteCons = open(directory+"Consolidated.txt",'w')


for filename in filenames:
    # Open the file in read mode
    fRead = open(filename,'r')
    # Strip extention
    newfilename, fileExt = os.path.splitext(filename)
    #fWrite = open(newfilename+"_reformated"+fileExt,'w')
    fWriteCons.write("\n\n"+newfilename+"\n")
    for line in fRead:
        #fWrite.write(line.rstrip()+' ')
        fWriteCons.write(line.rstrip()+'",\n')
    # Close the files
    fRead.close()
    #fWrite.close()
    #Remove the old file
    os.remove(filename)

fWriteCons.close()

##def main():
##    pass
##
##if __name__ == '__main__':
##    main()
