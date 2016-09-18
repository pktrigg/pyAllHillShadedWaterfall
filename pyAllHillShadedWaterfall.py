import sys
sys.path.append("C:/development/Python/pyall")

import argparse
import csv
from datetime import datetime
import geodetic
from glob import glob
import math
# from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
from PIL import Image,ImageDraw,ImageFont, ImageOps, ImageChops
import pyall
import shadedRelief as sr
import time
import os.path
import warnings

# ignore numpy NaN warnings when applying a mask to the images.
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='Read Kongsberg ALL file and create a hill shaded color waterfall image.')
    parser.add_argument('-i', dest='inputFile', action='store', help='-i <ALLfilename> : input ALL filename to image. It can also be a wildcard, e.g. *.all')
    parser.add_argument('-s', dest='shadeScale', default = 0, action='store', help='-s <value> : Override Automatic Shade scale factor with this value. A smaller number (0.1) provides less shade that a larger number (10) Range is anything.  [Default: 0]')
    parser.add_argument('-z', dest='zoom', default = 1.0, action='store', help='-z <value> : Zoom scale factor. A larger number makes a larger image, and a smaller number (0.5) provides a smaller image, e.g -z 2 makes an image twice the native resolution. [Default: 1.0]')
    parser.add_argument('-a', action='store_true', default=False, dest='annotate', help='-a : Annotate the image with timestamps.  [Default: True]')
    parser.add_argument('-r', action='store_true', default=False, dest='rotate', help='-r : Rotate the resulting waterfall so the image reads from left to right instead of bottom to top.  [Default is bottom to top]')
    parser.add_argument('-gray', action='store_true', default=False, dest='gray', help='-gray : Apply a gray scale depth palette to the image instead of a color depth.  [Default is False]')

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    
    #load a nice color palette
    colors = loadPalette(os.path.dirname(os.path.realpath(__file__)) + '/jeca.pal')
    args = parser.parse_args()

    print ("processing with settings: ", args)
    for filename in glob(args.inputFile):
        xResolution, yResolution, beamCount, leftExtent, rightExtent, distanceTravelled, navigation = computeXYResolution(filename)
        print("xRes %.2f yRes %.2f  leftExtent %.2f, rightExtent %.2f, distanceTravelled %.2f" % (xResolution, yResolution, leftExtent, rightExtent, distanceTravelled)) 
        shadeScale = float(args.shadeScale)
        if (shadeScale==0): 
            shadeScale = 38 * math.pow(abs(leftExtent)+abs(rightExtent), -0.783)
            # args.shadeScale = 38 * math.pow(abs(leftExtent)+abs(rightExtent), -0.783)
        if beamCount == 0:
            print ("No data to process, skipping empty file")
            continue
        zoom = float(args.zoom)
        swathWidth = abs(leftExtent)+abs(rightExtent)
        while (swathWidth < 300):
            zoom *= 2
            swathWidth *= zoom 
        print("Shade %.2f Zoom %.2f beamCount %d swathWidth %.2f" % (shadeScale, zoom, beamCount, abs(leftExtent)+abs(rightExtent))) 
        createWaterfall(filename, colors, beamCount, shadeScale, zoom, args.annotate, xResolution, yResolution, args.rotate, args.gray, leftExtent, rightExtent, distanceTravelled, navigation)

def createWaterfall(filename, colors, beamCount, shadeScale=1, zoom=1.0, annotate=True, xResolution=1, yResolution=1, rotate=False, gray=False, leftExtent=-100, rightExtent=100, distanceTravelled=0, navigation=[]):
    print ("Processing file: ", filename)

    r = pyall.ALLReader(filename)
    totalrecords = r.getRecordCount()
    start_time = time.time() # time the process
    recCount = 0
    waterfall = []
    minDepth = 9999.0
    maxDepth = -minDepth
    outputResolution = beamCount * zoom
    isoStretchFactor = (yResolution/xResolution) * zoom
    # print ("xRes %.2f yRes %.2f AcrossStretch %.2f" % (xResolution, yResolution, isoStretchFactor))
    while r.moreData():
        TypeOfDatagram, datagram = r.readDatagram()
        if (TypeOfDatagram == 0):
            continue
        if (TypeOfDatagram == 'X') or (TypeOfDatagram == 'D'):
            datagram.read()
            if datagram.NBeams == 0:
                continue

            # if datagram.SerialNumber == 275:                    
            for d in range(len(datagram.Depth)):
                datagram.Depth[d] = datagram.Depth[d] + datagram.TransducerDepth

            # we need to remember the actual data extents so we can set the color palette mappings to the same limits. 
            minDepth = min(minDepth, min(datagram.Depth))
            maxDepth = max(maxDepth, max(datagram.Depth))

            # we need to stretch the data to make it isometric, so lets use numpy interp routing to do that for Us
            xp = np.array(datagram.AcrossTrackDistance) #the x distance for the beams of a ping.  we could possibly use the real values here instead todo
            fp = np.array(datagram.Depth) #the depth list as a numpy array
            # fp = geodetic.medfilt(fp,31)
            x = np.linspace(leftExtent, rightExtent, outputResolution) #the required samples needs to be about the same as the original number of samples, spread across the across track range
            newDepths = np.interp(x, xp, fp, left=0.0, right=0.0)

            # run a median filter to remove crazy noise
            # newDepths = geodetic.medfilt(newDepths,7)
            waterfall.insert(0, np.asarray(newDepths))            

        recCount += 1
        if r.currentRecordDateTime().timestamp() % 30 == 0:
            percentageRead = (recCount / totalrecords) 
            update_progress("Decoding .all file", percentageRead)
    update_progress("Decoding .all file", 1)
    r.close()    

    # we have all data loaded, so now lets make a waterfall image...
    #---------------------------------------------------------------    
    print ("Correcting for vessel speed...")
    # we now need to interpolate in the along track direction so we have apprximate isometry
    npGrid = np.array(waterfall)

    stretchedGrid = np.empty((0, int(len(npGrid) * isoStretchFactor)))    
    for column in npGrid.T:
        y = np.linspace(0, len(column), len(column) * isoStretchFactor) #the required samples
        yp = np.arange(len(column)) 
        w2 = np.interp(y, yp, column, left=0.0, right=0.0)
        # w2 = geodetic.medfilt(w2,7)
        
        stretchedGrid = np.append(stretchedGrid, [w2],axis=0)
    npGrid = stretchedGrid
    npGrid = np.ma.masked_values(npGrid, 0.0)
    
    if gray:
        print ("Hillshading...")
        #Create hillshade a little brighter and invert so hills look like hills
        colorMap = None
        npGrid = npGrid.T * shadeScale * -1.0
        hs = sr.calcHillshade(npGrid, 1, 45, 30)
        img = Image.fromarray(hs).convert('RGBA')
    else:
        print ("Color mapping...")
        npGrid = npGrid.T
        # calculate color height map
        cmrgb = cm.colors.ListedColormap(colors, name='from_list', N=None)
        colorMap = cm.ScalarMappable(cmap=cmrgb)
        colorMap.set_clim(vmin=minDepth, vmax=maxDepth)
        colorArray = colorMap.to_rgba(npGrid, alpha=None, bytes=True)    
        colorImage = Image.frombuffer('RGBA', (colorArray.shape[1], colorArray.shape[0]), colorArray, 'raw', 'RGBA', 0,1)
        #Create hillshade a little darker as we are blending it. we do not need to invert as we are subtracting the shade from the color image
        npGrid = npGrid * shadeScale 
        hs = sr.calcHillshade(npGrid, 1, 45, 5)
        img = Image.fromarray(hs).convert('RGBA')

        # now blend the two images
        img = ImageChops.subtract(colorImage, img).convert('RGB')

    if annotate:
        #rotate the image if the user requests this.  It is a little better for viewing in a browser
        annotateWaterfall(img, navigation, isoStretchFactor)
        meanDepth = np.average(waterfall)
        waterfallPixelSize = (abs(rightExtent) + abs(rightExtent)) /  img.width
        # print ("Mean Depth %.2f" % meanDepth)
        imgLegend = createLegend(filename, img.width, (abs(leftExtent)+abs(rightExtent)), distanceTravelled, waterfallPixelSize, minDepth, maxDepth, meanDepth, colorMap)
        img = spliceImages(img, imgLegend)

    if rotate:
        img = img.rotate(-90, expand=True)
    img.save(os.path.splitext(filename)[0]+'.png')
    print ("Saved to: ", os.path.splitext(filename)[0]+'.png')

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
    navigation = []

    while r.moreData():
        TypeOfDatagram, datagram = r.readDatagram()
        if (TypeOfDatagram == 'P'):
            datagram.read()
            if prevLat == 0:
                prevLat =  datagram.Latitude
                prevLong =  datagram.Longitude
            range,bearing1, bearing2  = geodetic.calculateRangeBearingFromGeographicals(prevLong, prevLat, datagram.Longitude, datagram.Latitude)
            distanceTravelled += range
            navigation.append([recCount, r.currentRecordDateTime(), datagram.Latitude, datagram.Longitude])
            prevLat =  datagram.Latitude
            prevLong =  datagram.Longitude
        if (TypeOfDatagram == 'X') or (TypeOfDatagram == 'D'):
            datagram.read()
            if datagram.NBeams > 1:
                acrossMeans = np.append(acrossMeans, np.average(np.diff(np.asarray(datagram.AcrossTrackDistance))))
                leftExtents = np.append(leftExtents, datagram.AcrossTrackDistance[0])
                rightExtents = np.append(rightExtents, datagram.AcrossTrackDistance[-1])
                recCount = recCount + 1
                beamCount = max(beamCount, len(datagram.Depth)) 
            
    r.close()
    if recCount == 0:
        return 0,0,0,0,0,[] 
    xResolution = np.average(acrossMeans)
    yResolution = distanceTravelled / recCount
    return xResolution, yResolution, beamCount, np.min(leftExtents), np.max(rightExtents), distanceTravelled, navigation

def annotateWaterfall(img, navigation, scaleFactor):
    '''loop through the navigation and annotate'''
    lastTime = 0.0 
    lastRecord = 0
    for record, date, lat, long in navigation:
        # if (record % 100 == 0) and (record != lastRecord):
        if (record - lastRecord >= 100):
            writeLabel(img, int(record * scaleFactor), str(date.strftime("%H:%M:%S")))
            lastRecord = record
    return img

def writeLabel(img, y, label):
    x = 0
    f = ImageFont.truetype("arial.ttf",size=16)
    txt=Image.new('RGBA', (500,16))
    d = ImageDraw.Draw(txt)
    d.text( (0, 0), label,  font=f, fill=(255,255,255))
    d.line((0, 0, 20, 0), fill=(0,0,255))
    # w=txt.rotate(-90,  expand=1)
    offset = (x, y)
    img.paste(txt, offset, txt)
    # img.paste( ImageOps.colorize(txt, (0,0,0), (0,0,255)), (x, y),  txt)
    return img

def update_progress(job_title, progress):
    length = 20 # modify this to change the length
    block = int(round(length*progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 2))
    if progress >= 1: msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()

def loadPalette(paletteFileName):
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
    # now interpolate the colors so we have a broader spectrum
    reds = [ seq[0] for seq in colors ]
    x = np.linspace(1, len(reds), 256) #the desied samples needs to be about the same as the original number of samples
    xp = np.linspace(1, len(reds), len(reds)) #the actual sample spacings
    newReds = np.interp(x, xp, reds, left=0.0, right=0.0)
    
    greens = [ seq[1] for seq in colors ]
    x = np.linspace(1, len(greens), 256) #the desied samples needs to be about the same as the original number of samples
    xp = np.linspace(1, len(greens), len(greens)) #the actual sample spacings
    newGreens = np.interp(x, xp, greens, left=0.0, right=0.0)
    
    blues = [ seq[2] for seq in colors ]
    x = np.linspace(1, len(blues), 256) #the desied samples needs to be about the same as the original number of samples, spread across the across track range
    xp = np.linspace(1, len(blues), len(blues)) #the actual sample spacings
    newBlues = np.interp(x, xp, blues, left=0.0, right=0.0)

    colors = []
    for i in range(0,len(newReds)):
        colors.append([newReds[i], newGreens[i], newBlues[i]])
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

def spliceImages(img1, img2):
    # images = map(Image.open, ['Test1.jpg', 'Test2.jpg', 'Test3.jpg'])
    images = [img1, img2]
    widths, heights = zip(*(i.size for i in images))

    width = max(widths)
    height = sum(heights)

    new_im = Image.new('RGB', (width, height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    return new_im

def createLegend(fileName, imageWidth=640, waterfallWidth=640, waterfallLength=640, waterfallPixelSize=1, minDepth=0, maxDepth=999, meanDepth=99, colorMap=None):
    '''make a legend specific for this waterfalls image'''
    # this legend will contain:
    # InputFileName: <filename>
    # Waterfall Width: xxx.xxm
    # Waterfall Length: xxx.xxxm
    # Waterfall Pixel Size: xx.xxm
    # Mean Depth: xx.xxm
    # Color Palette as a graphical representation

    x = 0
    y=0
    fontHeight = 18
    npGrid = np.array([])

    f = ImageFont.truetype("cour.ttf",size=fontHeight)
    img=Image.new('RGB', (imageWidth,256)) # the new image.  this needs to be the same width as the main waterfall image
    
    d = ImageDraw.Draw(img)

    label = "file:%s" % (fileName)
    white=(255,255,255)
    d.text( (x, y), label,  font=f, fill=white)

    y += fontHeight
    label = "Waterfall Width    : %.2fm" % (waterfallWidth)
    d.text( (x, y), label,  font=f, fill=white)

    y += fontHeight
    label = "Waterfall Length   : %.2fm" % (waterfallLength)
    d.text( (x, y), label,  font=f, fill=white)

    y += fontHeight
    label = "Pixel Size         : %.2fm" % (waterfallPixelSize)
    d.text( (x, y), label,  font=f, fill=white)

    y += fontHeight
    label = "Minimum Depth      : %.2fm" % (minDepth)
    d.text( (x, y), label,  font=f, fill=white)

    y += fontHeight
    label = "Maximum Depth      : %.2fm" % (maxDepth)
    d.text( (x, y), label,  font=f, fill=white)

    y += fontHeight
    label = "Mean Depth         : %.2fm" % (meanDepth)
    d.text( (x, y), label,  font=f, fill=white)

    if (colorMap==None):
        return img
    # Creates a list containing 5 lists, each of 8 items, all set to 0
    y += fontHeight
    npline = np.linspace(start=minDepth, stop=maxDepth, num=imageWidth - ( fontHeight)) # length of colorbar is almost same as image
    npGrid = np.hstack((npGrid, npline))
    for i in range(fontHeight*2): # height of colorbar
        npGrid = np.vstack((npGrid, npline))
    colorArray = colorMap.to_rgba(npGrid, alpha=None, bytes=True)    
    colorImage = Image.frombuffer('RGB', (colorArray.shape[1], colorArray.shape[0]), colorArray, 'raw', 'RGBA', 0,1)
    offset = x + int (fontHeight/2), y
    img.paste(colorImage,offset)

    # now make the depth labels alongside the colorbar
    y += 2 + fontHeight * 2
    labels = np.linspace(minDepth, maxDepth, 10)
    for l in labels:
        label= "%.2f" % (l)
        x = (l-minDepth) * ((imageWidth - fontHeight) / (maxDepth-minDepth))
        offset = int(x), int(y)
        txt=Image.new('RGB', (70,20))
        d = ImageDraw.Draw(txt)
        d.text( (0, 0), label,  font=f, fill=white)
        w=txt.rotate(90,  expand=1)
        img.paste( w, offset)
    return img

if __name__ == "__main__":
    main()

