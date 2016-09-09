![Alt text](sample.png?raw=true "Optional Title")
pyAllHillShadedWaterfall
=====
* created:       August 2016
* by:            p.kennedy@fugro.com
* description:   python module to convert a Kongsberg .all file into hillshaded waterfall PNG.
* notes:         See main at end of script for example how to use this
* based on ALL file version 
* version 1.00
* developed for Python version 3.4 

Done
====
* add option for -gray scale instead of color depth
* add option to rotate the image.  While we call this a waterfall, screens are usually wider than tall, so rotating through 90 makes a lot of sense
* add support for shade scale factor
* add a better palette than the off-the-shelf 'blues' from matplotlib so we see familiar colors
* Waterfall images of an EM2040 look very nice indeed.
* added color depth to the resulting imagery so we see both shaded relief and depth in a single image. 
* implemented generation of a shaded relief image with numpy by computing matrix of orthogonal and transverse slopes, rotating them into the desirder sun angle.
* greyscale images look very handy
* Tested X reader by making a waterfall image.  Results look good


2Do
===
* test to see if we need to smooth the data a little for visual niceness
* add support for iso stretching by computing the mean pixel spacing in along and across track axes.  We can then stretch the waterfall to approximate isometry
* check rotated image so we know the survey direction is L-->Read
* add timestamps and ticks burned into the image
* test with EM302 units.
* add support for dual heads
* we need to automate a decent shadescale factor.  Here are some numbers to estimate that
Depth GoodScale
20   2.0    20*2
2830 0.2    2830*0.2=566


Example Reader
-------
```
python pyallhillshadedwaterfall.py
python pyallhillshadedwaterfall.py -i <filename.all>  
python pyallhillshadedwaterfall.py -i *.all  

help
----

C:\development\python\pyAllWaterfall>python pyAllHillShadedWaterfall.py
usage: pyAllHillShadedWaterfall.py [-h] [-i INPUTFILE]

Read Kongsberg ALL file and create a hill shaded color waterfall image.

optional arguments:
  -h, --help    show this help message and exit
  -i INPUTFILE  -i <ALLfilename> : input ALL filename to image

```