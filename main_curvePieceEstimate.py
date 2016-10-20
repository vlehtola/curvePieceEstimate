'''
Curve Piece Estimate

Attempt to construct a curved trajectory estimate from a point cloud of features. (This is SLAM.)
Different feature types may be used, but one is used at the moment.
- discontinuity features (dif)

If you use this code, or it helps you in any way, please cite:

Lehtola, V. V., Virtanen, J. P., Vaaja, M. T., Hyypp\"a, H., & N\"uchter, A. (2016). 
Localization of a mobile laser scanner via dimensional reduction. 
ISPRS Journal of Photogrammetry and Remote Sensing, 121, 48-59.

Also if you use intrinsic localization (data only from one laser scanner/profiler), please cite

Lehtola, V. V., Virtanen, J. P., Kukko, A., Kaartinen, H., & Hyypp\"a, H. (2015). 
Localization of mobile laser scanner using classical mechanics. 
ISPRS Journal of Photogrammetry and Remote Sensing, 99, 25-29. 

!!Run in processed/ directory!!

Uses 3DTK format, http://slam6d.sourceforge.net/

Reads .feature files.
Writes .pose and .frames files.

Amplifier in comments refers to the subsequent rounds of the curve piece estimate.
Angles: theta (along the trajectory), phi (horizontal), psi (vertical)

Created on Jul 1, 2015

@author: Ville Lehtola, ville.lehtola@iki.fi

'''

import glob
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import os, errno
import math
import copy
from timeit import default_timer as timer

###########
# AMPLIFIER parameters 
###########

#HEIGHT_SEGMENT_LENGTH = 15                       # in cycles, has to be an odd number. 7 cycles * 1.57 m circle ~= 11 meters    
HEIGHT_STEPS = 20                               # to discretize the below max angle. half for minus, half for plus sign.
MAX_PSI = 10.0/360*2*np.pi                      # angle from degrees to radians 
HEIGHT_PHASE_SHIFT = 1                          # how many phase shift attempts, min=1

###########
# Local correction parameters
###########

# if the input data is from left-handed coordinates
XD = 0
YD = 2      # 3dtk left handed coordinates
ZD = 1

#TURN_SEGMENT_LENGTH = 5                         # in cycles, has to be an odd number. 7 cycles * 1.57 m circle ~= 11 meters
TURN_STEPS = 10                                 # discretization for turning
STRAIGHT_PATH = False                           # if False, curved trajectory is computed
FLAT_FLOOR = False                              # perform height corrections or not. they are not recovered as is.

### Declare different methods
DIST_TO_LINE = 0
N_NEIGHBORS = 1
N_NEIGHBORS_HOR_VER = 2                         # amount of neighbors is used also in horizontal amplifier

### Assign the used method
#NNV = N_NEIGHBORS                               # choose method, either 0 or 1
NNV = N_NEIGHBORS_HOR_VER

COUNT_HORIZONTAL_LOOP = 0
COUNT_VERTICAL_LOOP = 0

PIVOT_ANGLE = 6.0                               # in degrees
RAD_PIVOT_ANGLE = PIVOT_ANGLE / 360 * 2*np.pi    

MIN_BINARY_STEP = 0.02 /360*2*np.pi     # 0.1
MIN_BINARY_STEP_PSI = 0.005 /360*2*np.pi

R_0 = 0.25
LENGTH_UNIT_MP = 100                                  # length unit multiplier, = 100 for cm for 3dtk, = 1 for meters
MAX_TURN_RADIUS = 2.0                           # in meters (/ rad)

PHIROT = (0,0,1)                                # rotate on 2D plane
PSIROT = (1,0,0)                                # height corrections
ROLLROT = (0,1,0)                               # correct FARO3D instrument artifact rolling error

PREFIX = "traji/"           # outputfile prefix

class TrajiFeature:
    def __init__(self, xyz, trajiSlice):
        self.xyz = xyz                          # the coordinates of this feature
        self.trajiSlice = trajiSlice            # onto which scan slice this feature belongs to            
    
    
def generateMatchScoreDistToLine(features):
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='kd_tree').fit(features)
    distances, indices = nbrs.kneighbors(features)
    
    sumDist = 0 #np.zeros(3, dtype='float')
    
    for i in range(1, len(indices), 1):                 # skip self-self distance
        # loop over all points
        sample = []
        for j in range(len(indices[i])):
            sample.append(features[indices[i,j]])
        pca = PCA(n_components=3)
        pca.fit(sample)
        V = pca.components_.T                                 # Problem: yksikkovektorit eri jarjestyksessa kuin explained score?

        #print pca.explained_variance_, pca.explained_variance_ratio_
        
        #
        # z_pca_axis is the plane normal, pca.mean_ is the centroid
        #
        distToPlane = np.zeros(3)
        # vahiten selittava suunta on aina vahan erilainen, ja sen ominaisarvo myos. siksi on eroja
        for k in range(3):
            distToPlane[k] = np.dot( (features[i] - pca.mean_), V[k] ) # length of z_pca_axis is unity   
        
        distToPlane = np.fabs( distToPlane )
        tmp = np.sum(distToPlane) - np.max(distToPlane)
        
        sumDist += tmp                    # minimoi kahta pieninta
                        
    return sumDist

def generateMatchScoreNNeighbors(features, N=10):
    nbrs = NearestNeighbors(n_neighbors=(N+1), algorithm='kd_tree').fit(features)
    distances, indices = nbrs.kneighbors(features)
    
    return sum(sum(distances))      # sum over all neighbors and all distances

def generateMatchScore(features):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(features)
    distances, indices = nbrs.kneighbors(features)
    
    return sum(distances)[1]    # skip the self-self distance

# nearest neighbor distance
def generatePairMatchScore(featuresA, featuresB):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(featuresA)
    distances, indices = nbrs.kneighbors(featuresB)
        
    return sum(distances)[0]    # do NOT skip the self-self distance
    
def generateInitialTrajectory(thetaList):
    stack = np.zeros(3, dtype='float32')               # x, y, z    
    traji = []  
    traji.append(np.array(stack))
    for i in range(1, len(thetaList), 1):
        stack[1] += LENGTH_UNIT_MP*R_0*(thetaList[i] - thetaList[i-1])          # advance in the right direction [1]
        traji.append(np.array(stack))
        
    return traji

# model for height differences, psi in radians!
# this function eats trajectories and rotates them 
def getHeightTrajectory(initialTraji, psiList):
    
    newTraji = [np.zeros(3, dtype='float') ]        
    newOri = [ np.zeros(3, dtype='float') ]
    stack = np.zeros(3, dtype='float32')         # x,y,z                
    
    for i in range(1, len(initialTraji), 1):
        rotMat = rotation_matrix(PSIROT, psiList[i])
        v = initialTraji[i] - initialTraji[i-1]                   
        rv = np.dot(rotMat, v)     
        # rotatoi eka pala, sitten jatka toinen pala siita mihin eka paattyy, ja rotatoi se, jne
        stack += rv        
        newTraji.append( np.array( stack ) )
        newOri.append( rv )        
        
    return np.array(newTraji), np.array(newOri)

 

# angle in radians! poses are relative poses
# curvatures sharpness is contained in ratio parameter: the curve is located in "ratio" part of the trajectory 
# ratio \in (0,1]
def getRotTrajectory(traji, phiList):
    
    newTraji = [np.zeros(3, dtype='float') ]
    newOri = [ np.zeros(3, dtype='float') ]
    stack= np.zeros(3, dtype='float')    
    
    for i in range(1, len(traji), 1):
        rotPhi = rotation_matrix(PHIROT, phiList[i])
        v = traji[i] - traji[i-1]
        rv = np.dot(rotPhi, v)
        stack += rv
        newTraji.append( np.array ( stack ))
        newOri.append( rv )
        
    return np.array(newTraji), np.array(newOri)
        

# param features is an array of array, while return value is an just an array
# modify point coordinates with trajectory
# returns right handed coordinates!!
def getRealCoordinates(features, traji, phiList, psiList, fn_index=-1):
    newFeatures = []    
    for i in range(len(features)):
        rotPhi = rotation_matrix(PHIROT, phiList[i])
        rotPsi = rotation_matrix(PSIROT, psiList[i]) 
        rotMat = np.dot(rotPhi, rotPsi)
        
        for j in range(len(features[i])):
            # rotate first, then translate            
            
            tmp = np.array( [ features[i][j][XD], features[i][j][YD], features[i][j][ZD] ], dtype='float32')
            
            ori = np.dot(rotMat, tmp)                           # rotate point coordinates

            xyz = np.array( ori + traji[i] )                                   # add transition
            newFeatures.append( xyz )
                        
    return newFeatures

def getRealCoordinatesPreservingSliceStructure(features, traji, phiList, psiList):
    newFeatures = []    
    for i in range(len(features)):
        tmp = getRealCoordinates([ features[i] ] , [ traji[i] ], [ phiList[i] ], [ psiList[i] ])
        newFeatures.append(tmp[:])
        
    return newFeatures

def main():
    #### OPtion 1.
    #HEIGHT_DISCRETIZATION = [6, 8,12,16,20,24,28]
    #ROTATION_DISCRETIZATION = [5]
    #AMOUNT_NNEIGHBORS_LIST = [1, 10, 20] #[1,3,10]                           # vertical parameter
    
    ### Option 2.
    #ROTATION_DISCRETIZATION = [2,3,4,5,6,7,8,9]
    #HEIGHT_DISCRETIZATION = [10]
    #AMOUNT_NNEIGHBORS_LIST = [1]                           # vertical parameter

    ### testing..
    ROTATION_DISCRETIZATION = [3,5,7] #[1,2,3,4,5,6,7,8,9]
    HEIGHT_DISCRETIZATION = [10]
    AMOUNT_NNEIGHBORS_LIST = [10]                           # vertical parameter
    
    timeFile = open("trajiFeatureMatcher_time_used.txt", 'a')
        
    for nn in AMOUNT_NNEIGHBORS_LIST:
        for it in HEIGHT_DISCRETIZATION:
            for TURN_SEGMENT_LENGTH in ROTATION_DISCRETIZATION:
                HEIGHT_SEGMENT_LENGTH = it
                start = timer()
                mainRoutine(HEIGHT_SEGMENT_LENGTH, TURN_SEGMENT_LENGTH, nn)
                end = timer()
                print >> timeFile, TURN_SEGMENT_LENGTH, HEIGHT_SEGMENT_LENGTH, nn, (end - start)      
    
    
def mainRoutine(HEIGHT_SEGMENT_LENGTH, TURN_SEGMENT_LENGTH, AMOUNT_NNEIGHBORS):
    mkdir_p(PREFIX)

    filenames = glob.glob("scan???.feature")
    filenames.sort()
    #filenames = []
    
    filenames2 = glob.glob("scan????.feature")
    filenames2.sort()
    
    filenames3 = glob.glob("scan?????.feature")
    filenames3.sort()
    
    filenames.extend(filenames2)
    filenames.extend(filenames3)
            
    oldRollCycles = 0
    
    discFeatures = []
    maxFeatures = []
    minFeatures = []
    poses = []
    outpfn = []
    lengths = []
    thetaList = []
    
    #lastPsi = 0 
    fn = filenames[0]
    koko = os.stat(fn).st_size
    print "size:", koko
    ki = 0
    while(koko == 0):
        ki += 1
        fn = filenames[ki]
        koko = os.stat(fn).st_size
                
    initLine = open(fn).readline().split()
    tmpvec = [ float( initLine[-3]), float( initLine[-2] ), float( initLine[-1] ) ]
    initialStartPoint = np.array( [ tmpvec[XD], tmpvec[YD], tmpvec[ZD] ] )
    globalStartPoint = [ tmpvec[XD], tmpvec[YD], tmpvec[ZD] ]           # change coordinate system from input
    globalRotMatrix = np.eye(3, dtype='float')           # keep track of the orientation between rollcycle loops         
    psiValues = []
    
    #
    # Values stored over the whole loop, used for the 'amplifier'.
    # traji below is the local trajectory that is monotonic.
    # We use this knowledge s.t. we can use only the end points of these local trajectories
    #
    ampPsi = []
    ampPhi = []
    ampTheta = []
    ampLocation = []
    ampRotation = []
    ampLocalTraji = []
    ampLocalDif = []
    ampFileName = []
    
    #for fn_index in range(ki, 2000, 1):                                 # TEMPORARY
    for fn_index in range(ki, len(filenames), 1):        
        fn = filenames[fn_index]
        f = open(fn)                                                    # I/O operaatio "tuplana" 1/2
        header1 = f.readline().split() 
        header2 = f.readline().split()

        f.close()

        dBegin = 0
        dEnd = int(header2[0])
        maxBegin = dEnd
        maxEnd = maxBegin + int(header2[1])
        minBegin = maxEnd
        minEnd = minBegin + int(header2[2])

        xyz = [ float(header1[XD]), float(header1[YD]), float(header1[ZD]) ]        
        poses.append(xyz)        
        outpfn.append(fn)

        theta = float(header2[3])
        thetaList.append(theta)
        
        data = np.genfromtxt(fn, skip_header=2, dtype='float32')        # I/O operaatio "tuplana" 2/2

        # esim. scan851.feature on korruptoitunut. miten hanskataan?
        if( len(data.shape) < 2):
            tmp = list([data])
            tmp.append([0,0,0])
            data = np.array(tmp)

        discFeatures.append( data[dBegin:dEnd] )                # each 2D slice is a separate entity 
        maxFeatures.append( data[maxBegin:maxEnd] )
        minFeatures.append( data[minBegin:minEnd] )
        
        
        # group by cycles, then do matching
        rollCycles = int(float(header1[1]))         
        if( (rollCycles > oldRollCycles) | (fn_index == len(filenames) -1 )):       # cleanup condition
                        
            oldRollCycles=rollCycles
            lengths = []            
            oldLen = np.linalg.norm(poses[0])
            # generate lengths array
            for i in range(len(poses)):
                l = np.linalg.norm(poses[i]) 
                dl = l - oldLen
                lengths.append(dl)
                if(dl > 10.0):
                    print 'Warning! Large slice-to-slice distance', dl, i, outpfn[i]                
                oldLen = l
            
            
            
            initialTraji = generateInitialTrajectory(thetaList)

            #
            # ORIENTATION CORRECTIONS, variable phi, orientations
            #            
            zeroPsi = np.zeros(len(thetaList))
            curvedTraji, bestPhi = horizontalCorrections(zeroPsi, thetaList, initialTraji, discFeatures, np.zeros(len(zeroPsi)), 2, fn_index)   # 2 means only the closest neighbor

            #
            # HEIGHT CORRECTIONS, variable psi, cannot be done reliably for one cycle only. => do this with AMPLIFIER
            #            

            bestTraji = curvedTraji

            #
            # WRITE OUTPUT
            #            
            
            writeTrajectory(bestTraji, bestPhi, zeroPsi, outpfn, globalStartPoint, globalRotMatrix)                                
                            
            # after advancement, update global rotmat and preceding start pos
            advancement = np.zeros(3, dtype='float')
            for k in range(1, len(bestTraji)):
                advancement += bestTraji[k] - bestTraji[k-1]       # telescope sum as path integral
            globalStartPoint = globalStartPoint + np.dot(globalRotMatrix, advancement) 
            
            # compute new global orientation matrices, traji includes height corr. already
                        
            #psiMat = rotation_matrix(PSIROT, zeroPsi[-1])
            phiMat = rotation_matrix(PHIROT, bestPhi[-1])
            #globalRotMatrix = np.dot ( np.dot( psiMat, phiMat) , globalRotMatrix)  
            globalRotMatrix = np.dot ( phiMat , globalRotMatrix)

            psiValues.append( zeroPsi[-1] )
            #print "koekoe:", zeroPsi[-1] - preDefPsi
            print "CYCLES:", rollCycles, "Phi:", bestPhi[-1], "translation:", globalStartPoint[0], globalStartPoint[1], globalStartPoint[2]

            # 'amplifier'
            ampPsi.append( zeroPsi[:] )
            ampPhi.append( bestPhi[:] )
            ampTheta.append( thetaList[:] )
            ampLocation.append( globalStartPoint[:] )
            ampRotation.append( globalRotMatrix[:] )
            ampLocalTraji.append( bestTraji[:] )
            ampLocalDif.append( discFeatures[:] )
            ampFileName.append( outpfn[:] )
            
            discFeatures = []
            maxFeatures = []
            minFeatures = []
            poses = []
            outpfn = []
            thetaList = []


    #
    # exit loop
    #
    # 'amplifier'
    # The idea is to use a longer correlation length than in the first phase (=one cycle). 
    # Especially for vertical corrections, this is important. Longer correlation = 4-10 cyles, for example.
    # (Representation in global coordinates.)
    # output in 3dtk format
    #

    # _cv: as in continue vertical loop
    outDir = "amplifier_t"+str(TURN_SEGMENT_LENGTH)+"_v"+str(HEIGHT_SEGMENT_LENGTH)+"_ff"+str(FLAT_FLOOR)+"_nn"+str(NNV)+"_"+str(AMOUNT_NNEIGHBORS)+"_ch"+str(COUNT_HORIZONTAL_LOOP)+"_cv"+str(COUNT_VERTICAL_LOOP)+"_pa"+str(PIVOT_ANGLE)+"/"
    mkdir_p(outDir)

    # change onto total angle representation
    totalPsi = cumulateAngleArray(ampPsi)    
    totalPhi = cumulateAngleArray(ampPhi)    
        
    # Generalize for the whole trajectory
    
    gTraji = []
    gTheta = []
    
    for j in range(len(ampLocalTraji[0])):
        gTraji.append( ampLocalTraji[0][j] + initialStartPoint )
        
        gTheta.append( ampTheta[0][j] )
           
    for i in range(1, len(ampLocalTraji), 1):

        tmpTraji = ampLocalTraji[i][:]
        for j in range(len(tmpTraji)):            
            tmpTraji[j] = np.dot( ampRotation[i-1], tmpTraji[j][:]) + ampLocation[i-1]
        
        gTraji.extend(tmpTraji)

        tmpTheta = ampTheta[i][:]
        gTheta.extend(tmpTheta)
        
    
    #
    # Attempt to re-correct the trajectory using a re-segmented version of the 'trajectory' and global features 
    #

    ampGlobalTraji, globalRot = globalTrajiFromAngles(ampTheta, ampPsi, ampPhi)
    
    arrLengthsHorizontal = segmentTrajectoryLinear( ampPhi, TURN_SEGMENT_LENGTH )    
    arrLengthsVertical = segmentTrajectoryLinear( ampPsi, HEIGHT_SEGMENT_LENGTH )

    
    print 'Trajectory re-segmented, horizontal parts: ', len(ampPhi), ' --> ', len(arrLengthsHorizontal), " vertical parts:", len(ampPsi), ' --> ', len(arrLengthsVertical)

    #
    # HORIZONTAL CORRECTIONS, variable phi, orientations
    #            
        
    continueHorizontalLoop = True
    countHorizontalLoop = COUNT_HORIZONTAL_LOOP
    while( (not STRAIGHT_PATH) & (continueHorizontalLoop) ):
    
        continueHorizontalLoop = False
        # totalPsi contains absolute psi values w.r.t. starting position
            
        sPsi = segmentArrayOfAngles( arrLengthsHorizontal, totalPsi )
        sPhi = segmentArrayOfAngles( arrLengthsHorizontal, totalPhi )
    
        sTheta = resegmentArray( arrLengthsHorizontal, ampTheta )
        sLocalDif = resegmentArray( arrLengthsHorizontal, ampLocalDif )
        sTraji = resegmentArray( arrLengthsHorizontal, ampGlobalTraji )    
    
        sTotalPhi = copy.deepcopy(sPhi)
    
        cumulativePhi = 0    
        for i in range(len(sTraji)-1):                  # omit the last segment to prevent finite size effects            
            curvedTraji, bestPhi = horizontalCorrections(sPsi[i], sTheta[i], sTraji[i], sLocalDif[i], sTotalPhi[i], AMOUNT_NNEIGHBORS, i)
            print "SEGMENT:", i, "Phi (corrected):", bestPhi[-1], "segment Length:", arrLengthsHorizontal[i]
            cPhi = np.linspace(cumulativePhi, cumulativePhi, len(bestPhi)) + bestPhi
            sPhi[i] = sPhi[i] + cPhi                     # update angles
            cumulativePhi += bestPhi[-1]
            if(bestPhi[-1] != 0):
                continueHorizontalLoop = True
    
    
        sPhi[-1] = sPhi[-1] + np.linspace(cumulativePhi, cumulativePhi, len(sPhi[-1]))           # last element
    
        # update trajectory with new phi
        sTraji, globalRot = globalTrajiFromAngles(sTheta, sPsi, sPhi, False)            # False: dont cumulate total angles
        
        if((continueHorizontalLoop == True) & (countHorizontalLoop > 0) ):
            countHorizontalLoop -= 1
            TURN_SEGMENT_LENGTH += 2
            arrLengthsHorizontal = segmentTrajectoryLinear( ampPhi, TURN_SEGMENT_LENGTH )
            print 'Trajectory iteration, horizontal parts: --> ', len(arrLengthsHorizontal)
            totalPhi = copy.deepcopy(sPhi)              # iterate Phi
        else:                        
            continueHorizontalLoop = False

                
        
    #
    # HEIGHT CORRECTIONS, variable psi, cannot be done reliably for one cycle only. Therefore AMPLIFIER.
    #            
        
    continueVerticalLoop = True
    countVerticalLoop = COUNT_VERTICAL_LOOP
    while( (not FLAT_FLOOR) & (continueVerticalLoop) ):
        
        continueVerticalLoop = False
        
        sPsi = segmentArrayOfAngles( arrLengthsVertical, totalPsi )
        sPhi = segmentArrayOfAngles( arrLengthsVertical, sPhi )                         # sPhi restructure
        
        sTheta = resegmentArray( arrLengthsVertical, ampTheta )
        sLocalDif = resegmentArray( arrLengthsVertical, ampLocalDif )
        sTraji = resegmentArray( arrLengthsVertical, sTraji )                           # sTraji restructure

        # after resegmentation, create overlap
        olPsi = createSegmentOverlap( arrLengthsVertical, sPsi )
        olPhi = createSegmentOverlap( arrLengthsVertical, sPhi )
        olTheta = createSegmentOverlap( arrLengthsVertical, sTheta )
        olLocalDif = createSegmentOverlap( arrLengthsVertical, sLocalDif )
        olTraji = createSegmentOverlap( arrLengthsVertical, sTraji )                           # sTraji restructure

        cycleIndices = [ [] for x in range(len(olPsi)) ]                                 # empty list of lists
        totalFrameIndices = [ [] for x in range(len(olPsi)) ]                                 # empty list of lists
        summa = 0
        localSumma = 0
        refSumma = 0
        refid = 1
        arrLengthsCycles = segmentTrajectoryLinear( ampPhi, 1 )
        
        # totalFrameIndices: full frame format with cycle indices 
        
        for it in arrLengthsCycles:
            summa += it
            localSumma += it
            # overlap is 50%
            if( localSumma > arrLengthsVertical[refid] ):
                if(refid >= len(arrLengthsVertical) - 2 ):
                    break   # end the loop, end adding pivot points, when last 25% of the last segment is left
                localSumma -= int(arrLengthsVertical[refid])
                refSumma += int(arrLengthsVertical[refid])
                refid += 1                      # remember overlap
                        
            cidx = refid*2 - 1
            
            if(localSumma < (arrLengthsVertical[refid] * 0.25) ):
                cycleIndices[cidx-1].append(localSumma + arrLengthsVertical[refid-1]/2 ) 
                totalFrameIndices[cidx-1].append(summa)               
            elif(localSumma > (arrLengthsVertical[refid] * 0.75) ):                
                cycleIndices[cidx+1].append( localSumma - int( arrLengthsVertical[refid]/2) )
                totalFrameIndices[cidx+1].append(summa)               
            else:
                cycleIndices[cidx].append(localSumma )
                totalFrameIndices[cidx].append(summa)               
                
        print "len cycles", len(cycleIndices), len(olTraji), len(arrLengthsVertical), len(arrLengthsCycles)
        print cycleIndices, arrLengthsVertical
        print totalFrameIndices
        
        pivotList = [ [0,0,0, len(olPsi[0])] ]        
        for i in range(1, len(olTraji)-1):                  # omit the first and the last segment to prevent finite size effects        

            curvedTraji, trialPsi, matchScore, sliceIndex, pivotIndex = verticalCorrectionsPivotAngle(olPhi[i], olPsi[i], olTheta[i], olTraji[i], olLocalDif[i], i, cycleIndices[i], AMOUNT_NNEIGHBORS)
            print "SEGMENT:", i, "Psi (corrected):", trialPsi[-1], "pivot:", pivotIndex, "score:", matchScore                
            pivotList.append( [ trialPsi[-1], sliceIndex, pivotIndex, len(olPsi[i])] )     # store only pivot points and angles            
            
        
        #
        # remove segment overlap. cumulate psi angle from pivots.
        #
        cumulativePsi = 0
        i=0
        j=0
        count = 0
        totalCount = 0
        
        
        print "pivotList:", pivotList
        for idx in range(len(pivotList)):
            it = pivotList[idx]
            
            count = 0                            
            while count < it[1]:                                # augment i and j up to the pivot point
                sPsi[i][j] = cumulativePsi
                j+= 1
                count += 1
                totalCount += 1
                if(j >= len(sPsi[i])):
                    i+= 1
                    j=0

            cumulativePsi += it[0]                              # add pivot point angle to the sum
            
            if(len(totalFrameIndices[idx]) > 0):
                while totalCount < totalFrameIndices[idx][-1]:          # augment i and j up to the beginning of next segment
                    sPsi[i][j] = cumulativePsi
                    j+= 1
                    totalCount += 1
                    if(j >= len(sPsi[i])):
                        i+= 1
                        j=0
                
                            
            if(it[0] != 0):
                if((continueVerticalLoop == False) & (countVerticalLoop > 0) ):
                    countVerticalLoop -= 1
                    continueVerticalLoop = True             # if any one pivot angle != 0 is found, continue
        
        # take care of time series end values
        while i < len(sPsi):
            sPsi[i][j] = cumulativePsi
            j+= 1
            if(j >= len(sPsi[i])):
                i+= 1
                j=0
                    
        sTraji, globalRot = globalTrajiFromAngles(sTheta, sPsi, sPhi, False)            # False: dont cumulate total angles

        if(continueVerticalLoop):
            # prepare for a second loop
            totalPsi = segmentArrayOfAngles( arrLengthsHorizontal, totalPsi )
            sPhi = segmentArrayOfAngles( arrLengthsHorizontal, sPhi )            
            sTraji = resegmentArray( arrLengthsHorizontal, sTraji )
        


    newTraji = flattenArray(sTraji)

    
    if(len(newTraji) != len(gTraji)):
        print "Ero leneissa:", len(newTraji), len(gTraji)
    
    ftmp = open(outDir+"newTraji.txt", 'w')
    for i in range(len(newTraji)):
        print >> ftmp, i, newTraji[i][0], newTraji[i][1], newTraji[i][2], gTraji[i][0], gTraji[i][1], gTraji[i][2], gTheta[i], (gTheta[i]*R_0*LENGTH_UNIT_MP)
       
    #
    # write output for amplifier
    #
    arrLengthsHorizontal = []
    for i in range(len(ampTheta)):
        arrLengthsHorizontal.append(len(ampTheta[i]))
        
    ampTraji = resegmentArray( arrLengthsHorizontal, sTraji )
    ampPsi = resegmentArray( arrLengthsHorizontal, sPsi )
    ampPhi = resegmentArray( arrLengthsHorizontal, sPhi )
    
    for i in range(len(ampTraji)):
        writeTrajectory(ampTraji[i], ampPhi[i], ampPsi[i], ampFileName[i], np.zeros(3), np.eye(3), outDir)        

# create 50% overlapping segments to deal with vertical corrections
def createSegmentOverlap( arrLengthsVertical, argArr ):
    arr = list( copy.deepcopy(argArr) )    
    N = len(arr)-2
    for k in range(N, -1, -1):
        bg = int( arrLengthsVertical[k]*0.5 )                       # overlap is 50%
        end = int( arrLengthsVertical[k+1]*0.5 )
        
        tmp = np.r_[ arr[k][bg:], arr[k+1][:end]]                   # concatenate using 1st axis
        arr.insert(k+1, tmp )
            
    return arr

### returns an array of ints that represents the new lengths of the local segments  of trajectory 
### step has to be an odd integer
def segmentTrajectoryLinear( ampAngle, segmentStep ):
    arrLengths = []
    
    totalCount = 0
    count = 0
    rangearr = range(0, len(ampAngle)- segmentStep, segmentStep)
    for i in rangearr:
        for j in range(segmentStep):
            count += len(ampAngle[i+j])
            
        arrLengths.append ( count )
        totalCount += count
        count = 0

    # add the remains
    for i in range(rangearr[-1]+segmentStep, len(ampAngle), 1):
        count += len(ampAngle[i])
        totalCount += len(ampAngle[i]) 
        
    if(count > 0):
        #print 'Remaining count in re-segmentation: ', count, totalCount
        arrLengths.append ( count )
    
    return arrLengths

# resegment arr into arrays of lengths that are defined in arrLengths
def resegmentArray( arrLengths, arr ):    
    # construct method output    
    newArr = []
    ki=0
    kj=0
    
    for i in range(len(arrLengths)):        
        newArr.append( [] )        
        for j in range(arrLengths[i]):
            newArr[-1].append( arr[ki][kj] )
            if(kj < len(arr[ki]) -1 ):
                kj += 1
            else:
                ki += 1
                kj = 0
                if(ki >= len(arr) ):
                    return newArr
                                
    return newArr

# segment arr into arrays of lengths that are defined in arrLengths
def segmentArrayOfAngles( arrLengths, totalAngle ):    
    # construct method output
    newArr = []
    ki=0
    kj=0
    count = 0        
    for i in range(len(arrLengths)):        
        newArr.append( [] )
        for j in range(arrLengths[i]):
            newArr[-1].append( totalAngle[ki][kj] )            
            count +=1            
            
            if(kj < len(totalAngle[ki]) -1 ):
                kj += 1
            else:
                ki += 1
                kj = 0
                if(ki >= len(totalAngle) ):
                    return newArr
                #    print "over:", ki, len(totalAngle), i,j, arrLengths[i], count 

                                
    return newArr


# compute global traji from angles
def globalTrajiFromAngles(ampTheta, ampPsi, ampPhi, useGlobalRot=True):
    tmpTraji = [ ]    
    globalRot = []
    startPos = np.zeros(3)     
    rotStack = np.eye(3, dtype='float32')
    newTraji = [  ]

    for i in range(len(ampTheta)):    
        initialTraji = generateInitialTrajectory(ampTheta[i] )
        
        tmpTraji = [ np.zeros(3, dtype='float32') ]
        stack = np.zeros(3, dtype='float32')                    # x,y,z
        for j in range(1, len(ampTheta[i]), 1):
            rotPhi = rotation_matrix(PHIROT, ampPhi[i][j])
            rotPsi = rotation_matrix(PSIROT, ampPsi[i][j])
            tmpRot = np.dot( rotPhi, rotPsi)
            tmpRot = np.dot( tmpRot, rotStack)                  # global rot
            v = initialTraji[j] - initialTraji[j-1]
            rv = np.dot(tmpRot, v)
            stack += rv
            tmpTraji.append( np.array ( stack[:] ))
            

        increment = tmpTraji[-1]
        tmpTraji += startPos
        
        # gather output
        newTraji.append(tmpTraji[:])
        globalRot.append( rotStack[:] )

        if(useGlobalRot):
            # update global rotation matrix
            rotPhi = rotation_matrix(PHIROT, ampPhi[i][-1])
            rotPsi = rotation_matrix(PSIROT, ampPsi[i][-1])
            rotStack = np.dot( rotStack, rotPhi) 
            rotStack = np.dot( rotStack, rotPsi)
            
        #update global start position
        startPos += increment
            
            
    return newTraji, globalRot  
    


def cumulateAngleArray(ampPhi):    
    # gather a cumulative total angle rather than differences
    # in order to avoid possible problems with finite precision.
    phiArr = []
    startAngle = 0
    for i in range(len(ampPhi)):        
        phiArr.append( [] )
        for j in range(len(ampPhi[i])):            
            phiArr[-1].append( (ampPhi[i][j] + startAngle) )
        
        startAngle += ampPhi[i][-1]

    return phiArr
        

def flattenArray( arr ):
    newArr = []
    for i in range( len(arr) ):
        newArr.extend(arr[i])

    return newArr


def verticalCorrectionsPivotAngle(phiList, origPsiList, thetaList, initialTraji, discFeatures, fn_index, cycleIndices, AMOUNT_NNEIGHBORS):
    N = len(phiList)    
            
    psiList = np.add( np.linspace(0, 0, N),  origPsiList)                              # := 0
    traji, newOri = getHeightTrajectory(initialTraji, psiList)             
    modDiscFeatures = getRealCoordinates(discFeatures, traji, phiList, psiList, fn_index)           # thetatilde

    # PLANAR OR LINE FIT MATCHSCORE
    if( (NNV == N_NEIGHBORS) | (NNV == N_NEIGHBORS_HOR_VER)):
        score = generateMatchScoreNNeighbors(modDiscFeatures, AMOUNT_NNEIGHBORS)
    elif(NNV == DIST_TO_LINE):
        score = generateMatchScoreDistToLine(modDiscFeatures)  

    score /= len(modDiscFeatures)
    bestPsiScore = (score, 0, 0, 0 )
    bestPsiTraji = np.array(traji)
    bestPsi = np.array(psiList)

    if( FLAT_FLOOR ):
        return bestPsiTraji, bestPsi, bestPsiScore[0], bestPsiScore[2], bestPsiScore[3]
    
    for psi in [ - RAD_PIVOT_ANGLE, RAD_PIVOT_ANGLE ]:
        for pId in range(len(cycleIndices)):
            pivot = cycleIndices[pId]
                
            psiList = np.add( np.r_[ np.linspace(0, 0, pivot), np.linspace(psi, psi, N-pivot) ],  origPsiList)   # 0-> 0 -> psi
            
            if(len(psiList) != len(initialTraji)):      # debug
                print "pivot, N:", pivot, N, cycleIndices

            traji, newOri = getHeightTrajectory(initialTraji, psiList)                   # trajectory for trial angle
    
            modDiscFeatures = getRealCoordinates(discFeatures, traji, phiList, psiList, fn_index)           # thetatilde                 
    
            # PLANAR OR LINE FIT MATCHSCORE
            if( (NNV == N_NEIGHBORS) | (NNV == N_NEIGHBORS_HOR_VER)):
                score = generateMatchScoreNNeighbors(modDiscFeatures, AMOUNT_NNEIGHBORS)
            elif(NNV == DIST_TO_LINE):
                score = generateMatchScoreDistToLine(modDiscFeatures)  
            
            score /= len(modDiscFeatures)                                               # norm the score
            print "pivotMatch w/ psi: ", psi, "pivot:", pivot, "score:", score, "best:", bestPsiScore
        
        
            if( (score < bestPsiScore[0]) | ( (score == bestPsiScore[0]) & (psi == 0) ) ):
                bestPsiScore = (score, psi, pivot, pId)
                bestPsiTraji = np.array(traji)
                bestPsi = np.array(psiList)

    print "bestPsi:", bestPsiScore[1], "Pivot slice, index:", bestPsiScore[2], bestPsiScore[3], "Score:", bestPsiScore[0], "(",bestPsi[-1], ")"
    return bestPsiTraji, bestPsi, bestPsiScore[0], bestPsiScore[2], bestPsiScore[3]


# Huom! initialTrajin pitaa olla suora, koska algoritmi ei kasittele "vanhoja" rotaatioita    
def horizontalCorrections(psiList, thetaList, initialTraji, discFeatures, oldPhi, AMOUNT_NNEIGHBORS, fn_index=-1):
        
    phiList = np.zeros(len(psiList))
    bestPhi = np.zeros(len(psiList))
    bestScore = (np.inf, 0, 0, 0, 0)                  
    traji, orientations = getRotTrajectory(initialTraji, phiList)                   # trajectory for trial angle
    bestTraji = copy.deepcopy(traji)
    
    if(STRAIGHT_PATH): 
        return bestTraji, phiList

    L = (thetaList[-1] - thetaList[0]) * R_0
    phiMax = L / MAX_TURN_RADIUS
    
    if(phiMax > np.pi*0.5):
        phiMax = np.pi*0.5                              # take into account previous turns, do not allow spiral trajis
    
    #print "In a distance of", L,"meters, you may turn ",phiMax, " radians"

    for i in range(TURN_STEPS+1):         #discretization                
        phi = phiMax * (- 1.0 + 2.0 * float(i)/TURN_STEPS )                             # trial angle        
        phiList = np.linspace(0, phi, len(psiList))
        totalPhiList = phiList + oldPhi
        traji, orientations = getRotTrajectory(initialTraji, phiList)                   # trajectory for trial angle             
        modDiscFeatures = getRealCoordinates(discFeatures, traji, totalPhiList, psiList, fn_index)        

        score = 0
        if( NNV == N_NEIGHBORS_HOR_VER ):
            score = generateMatchScoreNNeighbors(modDiscFeatures, AMOUNT_NNEIGHBORS)
        else:
            score = generateMatchScore(modDiscFeatures)
        
        score /= len(modDiscFeatures) 
        dif=0
        mif=0
        maf=0
        #print "match score: ", score, phi, "(",dif, mif, maf,")"
        if(score < bestScore[0]):
            bestScore = (score, phi, dif, mif, maf)
            bestTraji = copy.deepcopy(traji)
            bestPhi = copy.deepcopy( phiList )
    
    #
    #
    # Binary search a more accurate curvature
    #
    #
    
    step = phiMax / TURN_STEPS 
    trialPhi = []
    trialPhi.append( bestScore[1] - step )       # 1/2 step down
    trialPhi.append( bestScore[1] + step )       # 1/2 step up
    iterSign = +1
    iterContinue = True
    
    while( iterContinue & (step > MIN_BINARY_STEP) ):
        
        iterContinue= False
        
        for phi in trialPhi:
            phiList = np.linspace(0, phi, len(psiList))
            totalPhiList = phiList + oldPhi
            traji, orientations = getRotTrajectory(initialTraji, phiList)                   # trajectory for trial angle             
            modDiscFeatures = getRealCoordinates(discFeatures, traji, totalPhiList, psiList, fn_index)        
                                
            score = 0
            if( NNV == N_NEIGHBORS_HOR_VER ):
                score = generateMatchScoreNNeighbors(modDiscFeatures, AMOUNT_NNEIGHBORS)
            else:
                score = generateMatchScore(modDiscFeatures)
            
            score /= len(modDiscFeatures) 
        
            dif=0
            mif=0
            maf=0
            #print "match score: ", score, phi, "(",dif, mif, maf,")"
            if(score < bestScore[0]):
                iterSign = np.sign( bestScore[1] - phi )        # assign sign: minus if better phi is greater and vice versa

                bestScore = (score, phi, dif, mif, maf)
                bestTraji = copy.deepcopy(traji)
                bestPhi = copy.deepcopy( phiList )

                step *= 0.5
                trialPhi = [ phi + iterSign*step ]
                print "Binary search phi: ", phi
                iterContinue = True
                break   # break from trialPhi loop

    
    #print "bestScore, phi:", bestScore[0], bestScore[1] #, bestScore[2], bestScore[3], bestScore[4]    
    return bestTraji, bestPhi

    
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


# write .frames files (and zero .pose files)
def writeTrajectory(traji, phiList, psiList, outpfn, startPoint, globalRotMatrix, outdir=PREFIX):    

    for i in range(len(outpfn)):
        ori = np.dot(globalRotMatrix, traji[i][:])          # traji on jo local rotatoitu      
        xyz = ori + startPoint        
                
        #
        # Create pose file
        # NOTE: Keep all values at zero, because 3dtk 3D viewer cumulates these with .frames values
        #
        f = open(outdir+outpfn[i][:-8]+".pose", 'w')
        print >> f, 0, 0, 0     
        print >> f, 0, 0, 0                # rotation
        f.close()         

        phiMat = rotation_matrix(PHIROT, - phiList[i])
        psiMat = rotation_matrix(PSIROT, - psiList[i])

        rotMat = np.dot(phiMat, psiMat)
        m = np.dot( rotMat, globalRotMatrix.T )             # transpose to flip direction.  develop the orientation matrix

        lhMat = [ [ m[0,0], m[0,2], m[0,1] ], [ m[2,0], m[2,2], m[2,1] ], [ m[1,0], m[1,2], m[1,1] ]]
        
        # create frames file
              
        f = open(outdir+outpfn[i][:-8]+".frames", 'w')                
        st = ""
        for k in range(len(lhMat)):
            for l in range(len(lhMat[k])):
                st = st + str(lhMat[k][l]) + " "
                
            st = st + "0.0 "
        
        
        # write left-handed coordinates
        
        #st = "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0" # 0.0 0.0 6606.23 1.0 3.0
        
        print >> f, st, xyz[XD], xyz[YD], xyz[ZD], 1, 3            # orientation, translation, 1, 3
        f.close()         
           
# read .frames files
def readTrajectoryFrames(filenames):
    
    traji = []
    ori = []
    
    for i in range(filenames):
        arr = np.genfromtxt(filenames[i], dtype='float32')
        if( len( arr.shape ) == 2):
            arr = arr[-1]       # last line only
            
        a = arr[0:9]
        a = a.reshape(-1, 3)
        ori.append(  [ [ a[0,0], a[0,2], a[0,1] ], [ a[2,0], a[2,2], a[2,1] ], [ a[1,0], a[1,2], a[1,1] ]] )
        
        vec = [ arr[10+XD], arr[10+YD], arr[10+ZD] ]
        traji.append( vec[:] )
        
    return traji, ori
    

def rotation_matrix(axis, theta):
    """
    Euler-Rodrigues formula:
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2)
    b, c, d = -axis*math.sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])



    

if __name__ == '__main__':
    main()
