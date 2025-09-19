import numpy as np
from astropy.io import fits
from pathlib import Path
    
class TESS_PRF:
    """TESS Pixel Response Function object
    
    """
    def __init__(self,cam,ccd,sector,colnum,rownum, prf_dir):
        """Get TESS PRF for detector location, sector.
        
        Downloads relevant PRF files from the MAST archive by default.
        
        ***To use pre-downloaded local files, give directory containing
        subdirectories of format "cam#_ccd#/" as localdatadir, appropriate
        for sector of interest (separate for Sectors 1-3, 4+)
        
        inputs:
         - cam (int): TESS camera number
         - ccd (int): TESS ccd number
         - sector (int): TESS sector number
         - colnum (float): column number near target
         - rownum (float): row number near target
         
        """
        self.cam,self.ccd,self.sector,self.colnum,self.rownum = cam,ccd,sector,colnum,rownum
        self.prfnsamp = 9 #samples/pixel for TESS PRFs
        
        # #Get PRF files
        # prf_dir = Path.cwd() / 'PRF files'
        # # PRFs for sec 1-3 are differerent than for sector 4 and onwards
        # if sector < 4:
        #     prf_dir = prf_dir / 'Sector 1'
        # else:
        #     prf_dir = prf_dir / 'Sector 4'
        
        # Choose the right directory for the camera and ccd combination
        prf_dir = prf_dir / f'cam{int(cam)}_ccd{int(ccd)}'
        
        files = list(prf_dir.glob('*.fits'))
        #One directory on MAST has some errant files with `phot` in filename
        files = [file for file in files if 'phot' not in file.stem]
        
        cols = np.array([int(file.stem.split('-')[-1][-4:]) for file in files])
        rows = np.array([int(file.stem.split('-')[-2][-4:]) for file in files])

        #Bilinear interpolation between four surrounding PRFs
        LL = np.where((rows < rownum) & (cols < colnum))[0] #lower left
        LR = np.where((rows > rownum) & (cols < colnum))[0] #lower right
        UL = np.where((rows < rownum) & (cols > colnum))[0] #upper left
        UR = np.where((rows > rownum) & (cols > colnum))[0] #uppper right
        dist = np.sqrt((rows-rownum)**2. + (cols-colnum)**2.)
        surroundinginds = [subset[np.argmin(dist[subset])] for subset in [LL,LR,UL,UR]]
        #Following https://stackoverflow.com/a/8662355
        points = []
        for ind in surroundinginds:
            hdulist = fits.open(files[ind])
            prf = hdulist[0].data
            points.append((cols[ind],rows[ind],prf))
            hdulist.close()
        points = sorted(points)
        (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

        self.prf = (q11 * (x2 - colnum) * (y2 - rownum) +
                    q21 * (colnum - x1) * (y2 -  rownum) +
                    q12 * (x2 - colnum) * ( rownum - y1) +
                    q22 * (colnum - x1) * ( rownum - y1)
                    ) / ((x2 - x1) * (y2 - y1) + 0.0)
        
        ## Need to reshape PRF file for interpolation
        ## Add models just beyond pixel edges
        
        ##Size: 11x11x13x13 
        #indices: subrow index (from bottom), subcol index (from left),
        #TPF row index (from bottom), TPF col index (from left),
        reshaped = np.zeros((11,11,13,13)) 
        
        #Un-interleve PRF samples
        for i in range(9): #col number
            for j in range(9): #row number
                reshaped[i+1,j+1,:,:] = self.prf[8-i::9, 8-j::9] #col, row, cols, rows
        
        #Add columns just beyond pixel edges
        for j in range(1,10): #loop over rows
            reshaped[j,0,:,:] = np.append(reshaped[j,-2,:,1:],np.zeros((13,1)),axis=1)
            reshaped[j,-1,:,:] = np.append(np.zeros((13,1)),reshaped[j,1,:,:-1],axis=1)
        
        #Add rows just beyond pixel edges
        for i in range(0,11):
            reshaped[0,i,:,:] = np.append(reshaped[-2,i,1:,:],np.zeros((1,13)),axis=0)
            reshaped[-1,i,:,:] = np.append(np.zeros((1,13)),reshaped[1,i,:-1,:],axis=0)
        
        #Store for later use
        self.reshaped = reshaped
        
    def locate(self, sourcecol, sourcerow, stampsize=(13,13)):
        """Interpolate TESS PRF at location within "interleaved" TPF
        
        sourcecol (float): col position of star (relative to TPF)
        sourcerow (float): row position of star (relative to TPF)
        stampsize (int,int): (height,width) of TPF
        
        Note: pixel positions follow the convention that integers refer to the 
        pixel center.
        """
        #Break into integer and fractional pixel
        #adding 0.5 to conform to convention
        colint = np.floor(sourcecol+0.5)
        colfract = (sourcecol+0.5) % 1
        rowint = np.floor(sourcerow+0.5)
        rowfract = (sourcerow+0.5) % 1
        
        if (colint < -6) or (colint > 16) or (rowint < -6) or (rowint > 16):
            return np.zeros(stampsize)
        
        #Sub-pixel sample locations (in each dirextion, w/ border added)
        pixelsamples = np.arange(-1/18,19.1/18,1/9)
        
        #Find four surrounding subpixel PRF models
        colbelow = np.max(np.where(pixelsamples < colfract)[0])
        colabove = np.min(np.where(pixelsamples >= colfract)[0])
        rowbelow = np.max(np.where(pixelsamples < rowfract)[0])
        rowabove = np.min(np.where(pixelsamples >= rowfract)[0])
        
        #interpolate
        points = []
        LL = self.reshaped[rowbelow,colbelow,:,:]
        points.append((pixelsamples[colbelow],pixelsamples[rowbelow],LL))
        UL = self.reshaped[rowabove,colbelow,:,:]
        points.append((pixelsamples[colbelow],pixelsamples[rowabove],UL))
        LR = self.reshaped[rowbelow,colabove,:,:]
        points.append((pixelsamples[colabove],pixelsamples[rowbelow],LR))
        UR = self.reshaped[rowabove,colabove,:,:]
        points.append((pixelsamples[colabove],pixelsamples[rowabove],UR))
    
        points = sorted(points)
        (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points
        
        subsampled = (q11 * (x2 - colfract) * (y2 - rowfract) +
                      q21 * (colfract - x1) * (y2 -  rowfract) +
                      q12 * (x2 - colfract) * ( rowfract - y1) +
                      q22 * (colfract - x1) * ( rowfract - y1)
                      ) / ((x2 - x1) * (y2 - y1) + 0.0)
        #re-normalize to 1
        subsampled /= np.sum(subsampled)
        
        #Now must place at correct location in TPF
        tpfmodel = np.zeros(stampsize)
        
        #PRF models are 13x13 pixels
        #center of PRF is pixel (6,6)
        midprf = 6
        
        #That pixel should be (colint,rowint) in TPF
        tpfmodel[int(np.max([0,rowint-midprf])):int(np.min([stampsize[0],rowint+midprf+1])),
                 int(np.max([0,colint-midprf])):int(np.min([stampsize[1],colint+midprf+1])),] = subsampled[
            int(np.max([0,midprf-rowint])):int(np.min([2*midprf+1,midprf-rowint+stampsize[0]])),
            int(np.max([0,midprf-colint])):int(np.min([2*midprf+1,midprf-colint+stampsize[1]])),
            ]
        
        return tpfmodel


'''
Adapted for use in this project from: https://github.com/keatonb/TESS_PRF, in accordance to the software's licence, which is included below:

keatonb/TESS_PRF is licensed under the MIT License

Copyright (c) 2021 Keaton Bell

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

