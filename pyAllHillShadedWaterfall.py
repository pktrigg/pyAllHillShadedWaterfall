import sys
sys.path.append("C:/development/Python/pyall")

import os.path
from datetime import datetime
import geodetic
import numpy as np
import time
from PIL import Image
from PIL import ImageChops
import math
import shadedRelief as sr
from matplotlib import pyplot as plt
import pyall
from glob import glob
import argparse
import matplotlib.cm as cm
import csv

def main():
    parser = argparse.ArgumentParser(description='Read Kongsberg ALL file and create a hill shaded color waterfall image.')
    parser.add_argument('-i', dest='inputFile', action='store', help='-i <ALLfilename> : input ALL filename to image. It can also be a wildcard, e.g. *.all')
    parser.add_argument('-s', dest='shadeScale', default = 1.0, action='store', help='-s <value> : Shade scale factor. a smaller number (0.1) provides less shade that a larger number (10) Range is anything.  [Default - 1.0]')
    parser.add_argument('-r', action='store_true', default=False, dest='rotate', help='-r : Rotate the resulting waterfall so the image reads from left to right instead of bottom to top.  [Default is bottom to top]')
    parser.add_argument('-gray', action='store_true', default=False, dest='gray', help='-gray : Apply a gray scale depth palette to the image instead of a color depth.  [Default is False]')

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    
    #load a nice color palette
    colors = loadPal(os.path.dirname(os.path.realpath(__file__)) + '/jeca.pal')
    args = parser.parse_args()

    print ("processing with settings: ", args)
    # print ("Files to Process:", glob(args.inputFile))
    for filename in glob(args.inputFile):
        # navigation = loadNavigation(filename)
        # print (navigation)
        xResolution, yResolution, beamCount, leftExtent, rightExtent = computeXYResolution(filename)
        print("XRes %.2f YRes %.2f beamCount %d leftExtent %.2f, rightExtent %.2f" % (xResolution, yResolution, beamCount, leftExtent, rightExtent)) 
        
        if beamCount == 0:
            print ("No data to process, skipping empty file")
            continue
        createWaterfall(filename, colors, beamCount, float(args.shadeScale), xResolution, yResolution, args.rotate, args.gray, leftExtent, rightExtent)

def computeXYResolution(fileName):    
    '''compute the approximate across and alongtrack resolution so we can make a nearly isometric Image'''
    '''we compute the across track by taking the average Dx value between beams'''
    '''we compute the alongtracks by computing the linear length between all nav updates and dividing this by the number of pings'''
    xResolution = 1
    YResolution = 1
    prevLong = 0 
    prevLat = 0
    r = pyall.ALLReader(fileName)
    recCount = 0
    acrossMeans = np.array([])
    alongIntervals = np.array([])
    leftExtents = np.array([])
    rightExtents = np.array([])
    beamCount = 0
    distanceTravelled = 0.0
    while r.moreData():
        TypeOfDatagram, datagram = r.readDatagram()
        if (TypeOfDatagram == 'P'):
            datagram.read()
            if prevLat == 0:
                prevLat =  datagram.Latitude
                prevLong =  datagram.Longitude
            range,bearing1, bearing2  = geodetic.calculateRangeBearingFromGeographicals(prevLong, prevLat, datagram.Longitude, datagram.Latitude)
            distanceTravelled += range
            # if range > 0.0:
            #     alongIntervals = np.append(alongIntervals, range)
            prevLat =  datagram.Latitude
            prevLong =  datagram.Longitude
        if (TypeOfDatagram == 'X') or (TypeOfDatagram == 'D'):
            datagram.read()
            acrossMeans = np.append(acrossMeans, np.average(np.diff(np.asarray(datagram.AcrossTrackDistance))))
            leftExtents = np.append(leftExtents, datagram.AcrossTrackDistance[0])
            rightExtents = np.append(rightExtents, datagram.AcrossTrackDistance[-1])
            recCount = recCount + 1
            beamCount = max(beamCount, len(datagram.Depth)) 
            
        #limit to a few records so it is fast
        if recCount == 100:
            break
    r.close()
    if recCount == 0:
        return 0,0,0 
    xResolution = np.average(acrossMeans)
    yResolution = distanceTravelled / recCount
    return xResolution, yResolution, beamCount, np.min(leftExtents), np.max(rightExtents)

def createWaterfall(filename, colors, beamCount, shadeScale=1, xResolution=1, yResolution=1, rotate=False, gray=False, leftExtent=-100, rightExtent=100):
    print ("Processing file: ", filename)

    r = pyall.ALLReader(filename)
    totalrecords = r.getRecordCount()
    start_time = time.time() # time the process
    recCount = 0
    waterfall = []

    isoStretchFactor = math.ceil(xResolution/yResolution)
    print ("xRes %.2f yRes %.2f AcrossStretch %.2f" % (xResolution, yResolution, isoStretchFactor))
    while r.moreData():
        TypeOfDatagram, datagram = r.readDatagram()
        if (TypeOfDatagram == 0):
            continue
        if (TypeOfDatagram == 'X') or (TypeOfDatagram == 'D'):
            datagram.read()
            # nadirBeam = int(datagram.NBeams / 2)

            # if datagram.SerialNumber == 275:                    
            for d in range(len(datagram.Depth)):
                datagram.Depth[d] = datagram.Depth[d] + datagram.TransducerDepth

            # we need to stretch the data to make it isometric, so lets use numpy interp routing to do that for Us
            xp = np.array(datagram.AcrossTrackDistance) #the x distance for the beams of a ping.  we could possibly use teh real values here instead todo
            fp = np.array(datagram.Depth) #the depth list as a numpy array
            x = np.linspace(leftExtent, rightExtent, beamCount) #the required samples needs to be about the same as the original number of samples, spread across the across track range
            # x = np.linspace(0, len(datagram.Depth), len(datagram.Depth) * isoStretchFactor) #the required samples
            newDepths = np.interp(x, xp, fp, left=-1.123, right=-1.123)
            waterfall.insert(0, np.asarray(newDepths))            


            # # we need to stretch the data to make it isometric, so lets use numpy interp routing to do that for Us
            # xp = np.arange(len(datagram.Depth)) #the x distance for the beams of a ping.  we could possibly use teh real values here instead todo
            # fp = np.array(datagram.Depth) #the depth list as a numpy array
            # x = np.linspace(0, len(datagram.Depth), beamCount * isoStretchFactor) #the required samples
            # # x = np.linspace(0, len(datagram.Depth), len(datagram.Depth) * isoStretchFactor) #the required samples
            # newDepths = np.interp(x, xp, fp)
            # waterfall.insert(0, np.asarray(newDepths))            
            
            # isoStretchFactor = 1
            # for repeat in range (isoStretchFactor):
            #     waterfall.insert(0, np.asarray(newDepths))            
        recCount += 1
        if r.currentRecordDateTime().timestamp() % 30 == 0:
            percentageRead = (recCount / totalrecords) 
            update_progress("Decoding .all file", percentageRead)
    update_progress("Decoding .all file", 1)

    # we now need to interpolate in the along track direction so we have apprximate isometry
    npGrid = np.array(waterfall) * shadeScale

    stretchedGrid = np.empty( shape=(len(waterfall), 0))    
    for column in npGrid.T:
        y = np.linspace(0, len(column), len(column) * isoStretchFactor) #the required samples
        yp = np.arange(len(column)) 
        w2 = np.interp(y, yp, column, left=0, right=0)
        stretchedGrid = np.append(stretchedGrid, w2,1)
    npGrid = stretchedGrid
    meanDepth = np.average(waterfall)
    print ("Mean Depth %.2f" % meanDepth)
    # npGrid = np.array(waterfall) * (200 / meanDepth)   
    if gray:
        # npGrid = np.array(waterfall) * shadeScale * -1.0   
        #Create hillshade a little brighter
        hs = sr.calcHillshade(npGrid, 1, 45, 30)
        hillShadeImage = Image.fromarray(hs).convert('RGBA')
        #rotate the image if the user requests this.  It is a little better for viewing in a browser
        if rotate:
            hillShadeImage = hillShadeImage.rotate(-90, expand=True)
        hillshadeFilename = os.path.splitext(filename)[0]+'.png'
        hillShadeImage.save(hillshadeFilename)
        print ("Saved to: ", os.path.splitext(filename)[0]+'.png')
    else:
        #Create hillshade a little darker as we are blending it
        # npGrid = np.array(waterfall) * shadeScale   
        hs = sr.calcHillshade(npGrid, 1, 45, 5)
        hillShadeImage = Image.fromarray(hs).convert('RGBA')
        # calculate color height map
        cmrgb = cm.colors.ListedColormap(colors, name='from_list', N=None)
        # norm = mpl.colors.Normalize(vmin=0, vmax=100)
        # m = cm.ScalarMappable(cmap=cm.Blues)
        m = cm.ScalarMappable(cmap=cmrgb)
        colorArray = m.to_rgba(npGrid, alpha=None, bytes=True)    
        colorImage = Image.frombuffer('RGBA', (colorArray.shape[1], colorArray.shape[0]), colorArray, 'raw', 'RGBA', 0,1)
        
        # now blend the two images
        blendedImage = ImageChops.subtract(colorImage, hillShadeImage).convert('RGB')
        #rotate the image if the user requests this.  It is a little better for viewing in a browser
        if rotate:
            blendedImage = blendedImage.rotate(-90, expand=True)
        # now save the file
        blendedFilename = os.path.splitext(filename)[0]+'.png'
        blendedImage.save(blendedFilename, "PNG")
        print ("Saved to: ", os.path.splitext(filename)[0]+'.png')

    r.rewind()
    print("Complete converting ALL file to waterfall :-)")
    r.close()    

def update_progress(job_title, progress):
    length = 20 # modify this to change the length
    block = int(round(length*progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 2))
    if progress >= 1: msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()

###################################
# zg_LL = lower limit of grey scale
# zg_UL = upper limit of grey scale
# zs_LL = lower limit of samples range
# zs_UL = upper limit of sample range
def samplesToGrayImageLogarithmic(samples, invert, clip):
    zg_LL = 0 # min and max grey scales
    zg_UL = 255
    zs_LL = 0 
    zs_UL = 0
    conv_01_99 = 1
    # channelMin = 0
    # channelMax = 0
    #create numpy arrays so we can compute stats
    channel = np.array(samples)   

    # compute the clips
    if clip > 0:
        channelMin, channelMax = findMinMaxClipValues(channel, clip)
    else:
        channelMin = channel.min()
        channelMax = channel.max()
    
    if channelMin > 0:
        zs_LL = math.log(channelMin)
    else:
        zs_LL = 0
    if channelMax > 0:
        zs_UL = math.log(channelMax)
    else:
        zs_UL = 0
    
    # this scales from the range of image values to the range of output grey levels
    if (zs_UL - zs_LL) is not 0:
        conv_01_99 = ( zg_UL - zg_LL ) / ( zs_UL - zs_LL )
   
    #we can expect some divide by zero errors, so suppress 
    np.seterr(divide='ignore')
    channel = np.log(samples)
    channel = np.subtract(channel, zs_LL)
    channel = np.multiply(channel, conv_01_99)
    if invert:
        channel = np.subtract(zg_UL, channel)
    else:
        channel = np.add(zg_LL, channel)
    # ch = channel.astype('uint8')
    image = Image.fromarray(channel).convert('L')
    
    return image

def loadPal(paletteFileName):
    '''this will load and return a .pal file so we can apply colors to depths.  It will strip off the headers from the file and return a list of n*RGB values'''
    colors = []
    with open(paletteFileName,'r') as f:
        next(f) # skip headings
        next(f) # skip headings
        next(f) # skip headings
        reader=csv.reader(f,delimiter='\t')
        for red,green,blue in reader:
            thiscolor = [float(red)/255.0, float(green) / 255.0, float(blue) / 255.0]
            colors.append(thiscolor)
    return colors

def loadNavigation(fileName):    
    '''loads all the navigation into lists'''
    navigation = []
    r = pyall.ALLReader(fileName)
    while r.moreData():
        TypeOfDatagram, datagram = r.readDatagram()
        if (TypeOfDatagram == 'P'):
            datagram.read()
            navigation.append([datagram.Time, datagram.Latitude, datagram.Longitude])
    r.close()
    return navigation

if __name__ == "__main__":
    main()

