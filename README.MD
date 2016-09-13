![Alt text](sample.png?raw=true "Color Shaded Waterfall direct from .all file.")
![Alt text](sample2.png?raw=true "Gray Shaded Waterfall direct from .all file.")
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
* improved color mapping by settting the colormap to the actual data extents rather than the interp extents.
* add timestamp annotation and ticks burned into the image
* some files are not as roll stabilised as they might be.  If the user uses equi-angular mode, the Dx values are not suitable for aligning on a per pixel basis.  It simply does nopt work well.  Try the Deakin EM2040 data ona small boat in 50m water as an example.  Therefore we need to take the ping and use numpy to interpolate across the ping so things line up 
* sort out crash with very small .all files.
* add support for iso stretching by computing the mean pixel spacing in along and across track axes.  We can then stretch the waterfall to approximate isometry
* handle empty all files elegantly
* improved D reader as we were missing 1 field (maxBeams)
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
* check rotated image so we know the survey direction is L-->Read
* test with EM302 units.
* add support for dual heads
* we need to automate a decent shadescale factor.  Here are some numbers to estimate that
    Depth GoodScale
    20   2.0    20*2
    35   1.5    Em2000 AUV
    2830 0.2    2830*0.2=566


Example Reader
-------
```
python pyallhillshadedwaterfall.py
python pyallhillshadedwaterfall.py -i <filename.all>  
python pyallhillshadedwaterfall.py -i *.all  

help
----


C:\projects\GP1533\VSM03>python c:\development\python\pyAllhillshadedWaterfall\pyAllHillShadedWaterfall.py
usage: pyAllHillShadedWaterfall.py [-h] [-i INPUTFILE] [-s SHADESCALE] [-r]
                                   [-gray]

Read Kongsberg ALL file and create a hill shaded color waterfall image.

optional arguments:
  -h, --help     show this help message and exit
  -i INPUTFILE   -i <ALLfilename> : input ALL filename to image
  -s SHADESCALE  -s <value> : Shade scale factor. a smaller number (0.1)
                 provides less shade that a larger number (10) Range is
                 anything. [Default - 1.0]
  -r             -r : Rotate the resulting waterfall so the image reads from
                 left to right instead of bottom to top. [Default is bottom to
                 top]
  -gray          -gray : Apply a gray scale depth palette to the image instead
                 of a color depth. [Default is False]


