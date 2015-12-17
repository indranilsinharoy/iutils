# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        textUtils.py
# Purpose:
#
# Author:      Indranil Sinharoy
#
# Created:     06/10/2012
# Copyright:   (c) Indranil 2012
# Licence:     MIT Licence
#-------------------------------------------------------------------------------
import os as _os, glob as _glob
import nltk as _nltk



def get_pos(text):
    """returns the POS tags for input text 

    Parameters
    ---------- 
    text : string 
        one or more sentences

    Returns
    ------- 
    taggedPOS : list 
        list of 2-tuples with word and POS tag 

    Notes
    ----- 
    Use `nltk.help.upenn_tagset()` to get help on the tags. 

    Examples
    -------- 
    >>>import iutils.text.general as tg
    >>>tg.get_pos(""A piece of cake"")
    [('A', 'DT'), ('piece', 'NN'), ('of', 'IN'), ('cake', 'NN')]
    """
    wTokens = _nltk.word_tokenize(text)
    wTagged = _nltk.pos_tag(wTokens)
    return wTagged



# # prompt the user to input the full path name of the file
# #filename = input()

# directory = "C:\\Users\\Indranil\\Downloads\\"
# #pattern = "raytracing.txt"
# pattern = "7 - 1 -*.txt"

# filenames = glob.glob(directory+pattern)
# #consolidated file (text)
# fWriteCons = open(directory+"Consolidated.txt",'w')


# for filename in filenames:
#     # Open the file in read mode
#     fRead = open(filename,'r')
#     # Strip extention
#     newfilename, fileExt = os.path.splitext(filename)
#     #fWrite = open(newfilename+"_reformated"+fileExt,'w')
#     fWriteCons.write("\n\n"+newfilename+"\n")
#     for line in fRead:
#         #fWrite.write(line.rstrip()+' ')
#         fWriteCons.write(line.rstrip()+'",\n')
#     # Close the files
#     fRead.close()
#     #fWrite.close()
#     #Remove the old file
#     os.remove(filename)

# fWriteCons.close()

##def main():
##    pass
##
##if __name__ == '__main__':
##    main()
