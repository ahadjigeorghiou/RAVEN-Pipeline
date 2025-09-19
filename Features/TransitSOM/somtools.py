# This file is part of the TransitSOM code accompanying the paper Armstrong et al 2016
# Copyright (C) 2016 David Armstrong
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

def KohonenSave(layer,outfile):  #basically a 3d >> 2d saver
    with open(outfile,'w') as f:
        f.write(str(layer.shape[0])+','+str(layer.shape[1])+','+str(layer.shape[2])+'\n')
        for i in range(layer.shape[0]):
            for j in range(layer.shape[1]):
                for k in range(layer.shape[2]):
                    f.write(str(layer[i,j,k]))
                    if k < layer.shape[2]-1:
                        f.write(',')
                f.write('\n')

def KohonenLoad(infile):
    with open(infile,'r') as f:
        lines = f.readlines()
    newshape = lines[0].strip('\n').split(',')
    out = np.zeros([int(newshape[0]),int(newshape[1]),int(newshape[2])])
    for i in range(int(newshape[0])):
        for j in range(int(newshape[1])):
            line = lines[1+(i*int(newshape[1]))+j].strip('\n').split(',')
            for k in range(int(newshape[2])):
                out[i,j,k] = float(line[k])
    return out

    
def Proportions_2group(mapped,groups, map_shape=(20,20)):
    bins_x = np.arange(0, map_shape[0]+1, 1, dtype=int)
    bins_y = np.arange(0, map_shape[1]+1, 1, dtype=int)
    
    map1 = (groups == 0)  # planets
    map2 = (groups == 1)   #false positive
    hist1 = np.histogram2d(mapped[map1][:,0],mapped[map1][:,1], bins=[bins_x, bins_y])[0]
    hist2 = np.histogram2d(mapped[map2][:,0],mapped[map2][:,1], bins=[bins_x, bins_y])[0]
    return hist1 / (hist1+hist2)


def MapErrors_MC(somobject,dataarray,errorarray,niter):
    map_all = np.zeros([dataarray.shape[0],len(somobject.K.shape)-1,niter])
    for iter in range(niter):
        #print iter
        resampledarray = dataarray + np.random.normal(0,errorarray,dataarray.shape)
        map_iter = somobject(resampledarray)
        map_all[:,:,iter] = map_iter
    return map_all

def Proportions(klayer,mapped,groups,ngroups,binsx,binsy):
    percentages = np.zeros([klayer.shape[0],klayer.shape[1]])
    toreturn = []
    hists = []
    for groupidx in range(ngroups):
        map = groups == groupidx
        hists.append(np.histogram2d(mapped[map][:,0],mapped[map][:,1],bins=[binsx,binsy])[0])
    sum = np.zeros([binsx,binsy])
    for h in hists:
        #print h.shape
        sum += h
    for h in hists:
        toreturn.append(h/sum)
    return np.array(toreturn),sum

 
def Classify(map_all,proportions,ngroups,proportion_weights):
    
    class_probs = np.zeros([map_all.shape[0],ngroups])
    localnormalise = np.zeros([map_all.shape[0],ngroups])
    for obj in range(map_all.shape[0]):
        #print obj
        for classidx in range(ngroups):
            for iteridx in range(map_all.shape[2]):
                localprop = proportions[classidx][map_all[obj,0,iteridx],map_all[obj,1,iteridx]]
                localweight = proportion_weights[map_all[obj,0,iteridx],map_all[obj,1,iteridx]]
                if localprop >= 0:
                    class_probs[obj,classidx] += localprop*localweight
                    localnormalise[obj,classidx] += localweight
    
    return class_probs/localnormalise

def Classify_P_FP(class_probs,class_weights_P,class_weights_FP):
    class_probs_P_FP = np.zeros([class_probs.shape[0],2])
    
    class_probs_P_FP[:,0] = np.sum(class_probs * class_weights_P,axis=1)
    class_probs_P_FP[:,1] = np.sum(class_probs * class_weights_FP,axis=1)

    class_probs_P_FP[:,0] = class_probs_P_FP[:,0] /np.sum(class_probs_P_FP,axis=1)
    class_probs_P_FP[:,1] = class_probs_P_FP[:,1] /np.sum(class_probs_P_FP,axis=1)
    return class_probs_P_FP

def Classify_Distances(map_all, avgdistances, pcols=(0,3), fpcols=(1,2,4)):
    if type(pcols)==int:
        planet_distances = avgdistances[:,:,pcols]
    else:
        planet_distances = np.mean(avgdistances[:,:,pcols],axis=2)
    if type(fpcols)==int:
        fp_distances = avgdistances[:,:,fpcols]
    else:
        fp_distances = np.mean(avgdistances[:,:,fpcols],axis=2)
    
    som_power_planet = planet_distances/(planet_distances+fp_distances)
    som_power_fp = fp_distances/(planet_distances+fp_distances)
    
    class_power = np.zeros([map_all.shape[0],2])
    for obj in range(map_all.shape[0]):
        
        for iteridx in range(map_all.shape[2]):
            class_power[obj,0] += som_power_fp[map_all[obj,0,iteridx],map_all[obj,1,iteridx]] 
            class_power[obj,1] += som_power_planet[map_all[obj,0,iteridx],map_all[obj,1,iteridx]]
    
    class_power /= map_all.shape[2]
    planet_prob = class_power[:,0] / np.sum(class_power,axis=1)
    return planet_prob,class_power
    
    
def ChiSqDist(template,base):
    return np.sum( np.power( template - base , 2 ) )

def EucDist(template,base):
    return np.sqrt(np.sum( np.power( template - base , 2 ) ))
    
def PixelClassifier(klayer,SOMarray,grouparray,ngroups,normalise=False,lowbound=0,highbound=1000):
    avgdistances = np.zeros([klayer.shape[0],klayer.shape[1],ngroups])
    for xpixel in range(klayer.shape[0]):
        #print xpixel
        for ypixel in range(klayer.shape[1]):
            distances = np.zeros(SOMarray.shape[0])
            for dataindex in range(SOMarray.shape[0]):
                distances[dataindex] = ChiSqDist(klayer[xpixel,ypixel,lowbound:highbound],SOMarray[dataindex,lowbound:highbound])
            for groupindex in range(ngroups):
                avgdistances[xpixel,ypixel,groupindex] = np.mean(distances[grouparray==groupindex])
    if normalise:
        for groupindex in range(ngroups):
            maxdist = np.sort(avgdistances[:,:,groupindex].flatten())[-normalise] #ignores the normalise largest distances. These can be anomalies - the key area we want is larger. 5 is a good value for K2
            mindist = np.min(avgdistances[:,:,groupindex])
            avgdistances[:,:,groupindex] = 1./(maxdist-mindist) * (avgdistances[:,:,groupindex]-mindist)
    return avgdistances
    