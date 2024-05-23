import os
import copy
import math
import numpy
import random
import shutil
import itertools
import multiprocessing
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.isomorphism as iso


from os.path import exists
import numpy as np
import copy, os
import matplotlib.pyplot as plt
import threading, multiprocessing


from RDAnalyzer.tools import load, dump, dump, load, \
    listToCSV, drawFigMulti, drawFigSig, printf, smooth, \
    dump, load, drawFigMulti, listToCSV, drawFigSig, printf, \
    removeCom

from RDAnalyzer.set import DEFAULT_BONDER_ORDER, ATOMTYPE, \
    ATOMTYPE_INVERSE, TIMESTEP, TIME_UNIT, DUMP_INTERVAL


from RDAnalyzer.species import Atom, Molecule, Mol, Reaction




class Analyzer:

    ################################################### split
    @staticmethod
    def split(interval, beginStep, endStep, trjName, bondName):
        trjOutFileName = trjName + '.pre' + str(interval)+'fra.' + str(beginStep) + '-'+str(endStep)
        bondOutFileName = bondName + '.pre' + str(interval)+'fra.'+str(beginStep) + '-'+str(endStep)

        if trjOutFileName in os.listdir() and bondOutFileName in os.listdir():
            printf("#[split              ] Exist", trjOutFileName, "and", bondOutFileName)
            return (trjOutFileName, bondOutFileName)
        else: pass

        aimStepList = [i for i in range(beginStep, endStep+1, interval*DUMP_INTERVAL)]

        with open(trjName) as trjFile:
            line = trjFile.readlines()[3]
            numOfAtoms = int(line.split()[0])
            printf("#[split              ] Number of atoms", numOfAtoms)
        trjFile.close()

        # read trj
        stepNumbuncount = copy.deepcopy(aimStepList)
        allFraTrj = []
        trjLineNumbOfOneFrame = numOfAtoms+9
        with open(trjName) as trjFile:
            linesNum = 0
            stepNum = 0
            thisFreTrj = []
            for lines in trjFile.readlines():
                linesNum = linesNum + 1
                thisFreTrj.append(lines)

                if linesNum%trjLineNumbOfOneFrame == 0:
                    if stepNum in aimStepList and stepNum in stepNumbuncount:
                        allFraTrj.append(thisFreTrj)
                        stepNumbuncount.remove(stepNum)
                    else: 
                        pass

                    thisFreTrj = []
                
                if linesNum%trjLineNumbOfOneFrame == 2:
                    stepNum = int(lines)
        trjFile.close()

        #write trj
        trjOutFile = open(trjOutFileName, 'w')
        for theList in allFraTrj:
            for lines in theList:
                trjOutFile.write(lines)
        trjOutFile.close()
        del allFraTrj
        printf("#[split              ] Split", trjName, "to", trjOutFileName, "done")

        # read bond
        allFraBond = []
        bondLineNumbOfOneFrame = numOfAtoms+8
        with open(bondName) as bondFile:
            linesNum = 0
            stepNum = 0
            thisFreBond = []
            for lines in bondFile.readlines():
                linesNum = linesNum + 1
                thisFreBond.append(lines)

                if linesNum % bondLineNumbOfOneFrame == 0:
                    if stepNum in aimStepList:
                        allFraBond.append(thisFreBond)
                    else:
                        pass

                    thisFreBond = []

                if linesNum % bondLineNumbOfOneFrame == 1:
                    stepNum = int(lines.split()[2])
        bondFile.close()

        #write bond
        bondOutFile = open(bondOutFileName, 'w')
        for theList in allFraBond:
            for lines in theList:
                bondOutFile.write(lines)
        bondOutFile.close()
        del allFraBond
        printf("#[split              ] Split", bondName, "to", bondOutFileName, "done")
        return (trjOutFileName, bondOutFileName)


    ################################################### to atoms
    @staticmethod
    def toAtoms(trjSp, bondSp):
        printf("#[toAtoms            ] Begin convert txt to Atom")

        thisFrame, allFrames = {}, {}
        readStep = False
        timeStep = 0
        box = []
        with open(trjSp) as trjFile:
            for lines in trjFile.readlines():
                line = lines.split()

                # atom information line
                if len(line) == 6 and line[0].isdigit():
                
                    x, y, z =  float(line[3]), float(line[4]), float(line[5])
                            
                    thisAtom = Atom(int(line[0]),  # id
                                          int(line[1]),  # type
                                          (x, y, z),  # position
                                          float(line[2]),  # charge
                                          [],  # bonds
                                          timeStep)  # timestep
                    thisFrame[int(line[0])] = thisAtom

                # step line, store previous frame
                elif line[-1] == 'TIMESTEP':
                    readStep = True
                    if len(thisFrame) != 0:
                        thisFrame['box'] = box
                        allFrames[timeStep] = thisFrame
                    else:
                        pass
                    box = []
                    thisFrame = {}

                elif readStep:
                    timeStep = int(line[0])
                    readStep = False

                # box information line
                elif len(line) == 2 and '.' in line[0]:
                    box.append((float(line[0]), float(line[1])))

                elif len(line) == 3 and '.' in line[0]:
                    #box.append((float(line[0]), float(line[1])))
                    box.append((float(line[0]), float(line[1]), float(line[2])))
                else:
                    pass

        thisFrame['box'] = box  # store the ending frame
        allFrames[timeStep] = thisFrame
        box = []
        thisFrame = {}
        trjFile.close()

        timeStep = 0
        with open(bondSp) as bondFile:
            for lines in bondFile.readlines():
                line = lines.split()

                # atom information line
                if line[0] != '#':
                    atomId = int(line[0])
                    bondNumb = int(line[2])
                    bondedAtomsID = line[3:3+bondNumb]
                    bondedAtomsID = [int(i) for i in bondedAtomsID]
                    bondsOrder = line[4+bondNumb: 2*bondNumb+4]
                    bondsOrder = [float(i) for i in bondsOrder]
                    for bondedID, bondedOrder in zip(bondedAtomsID, bondsOrder):
                        if bondedOrder >= DEFAULT_BONDER_ORDER:
                            allFrames[timeStep][atomId].bonds.append((bondedID, bondedOrder))  # add the bond info to the dir
                        else: pass              
                
                # time step line
                elif len(line) == 3 and line[1] == 'Timestep':
                    timeStep = int(line[2])
                else: pass

        bondFile.close()
        del thisFrame, box, timeStep, readStep
        printf("#[toAtoms            ] Convert txt to Atom done")
        return allFrames

    @staticmethod
    def getVdw_N_O_sigFrame(step, frame:dict, idList:list, length):
        fra=copy.deepcopy(frame)
        box = tuple(fra['box'])
        lx, ly, lz = box[0][1] - box[0][0], box[1][1] - box[1][0], box[2][1] - box[2][0],
        del fra['box']

        NList = [fra[i] for i in idList]

        for atom in fra.values():
            for NAtom in NList:
                L = atom.distanceWith_PBC(NAtom, lx, ly, lz)
                L = round(L, 2)

                if L <= length:
                    atom.vdw.append( (NAtom.id, L) )
                else: pass

        fra["box"] = frame['box']
        return (step, fra)



    @staticmethod
    def getVdw_N_O(allFrame_interval_begin_end, length = 6.0, type = 'N', ncores = 4):
        '''find atoms that shorter than the `length`'''


        allFrames, (interval, begin, end) = allFrame_interval_begin_end
        sigFrame = copy.deepcopy([i for i in allFrames.values()][0])
        del sigFrame['box']

        idList = []
        for atom in sigFrame.values():
            if atom.type == ATOMTYPE_INVERSE()[type]:
                idList.append(atom.id)
            else:
                pass

        pool = multiprocessing.Pool(ncores)
        asyResult = []

        for step, frame in allFrames.items():
            asyResult.append(
                pool.apply_async(
                    Analyzer.getVdw_N_O_sigFrame,
                    args=(step, frame, idList, length)
                    )
                )
        
        pool.close()
        pool.join()

        newFrames = {}
        for result in asyResult:
            step, frame = result.get()
            newFrames[step] = frame
        
        return newFrames, (interval, begin, end) 

    

    ########################################################## to mols
    @staticmethod
    def atomFrameToMols(frame, step):
        fra=copy.deepcopy(frame)
        box = tuple(fra['box'])
        del fra['box']

        g=nx.Graph()
        for atom in fra.values():
            g.add_node(atom, id=atom.id, type=atom.type)
            for id in atom.getRelatedAtomsID():
                g.add_edge(atom, fra[id])

        id=1
        molDir = {}
        for cc in nx.connected_components(g):
            subG=g.subgraph(cc)
            molDir[id]=Molecule()
            molDir[id].topology=subG
            molDir[id].step=step
            molDir[id].id=id
            molDir[id].box=box
            id=id+1
        return molDir

    @staticmethod
    def toMolecules(atomDict, nCores = None):
        printf("#[toMolecules        ] Begin convert Atom to Molecule")
        
        allFramesMols = {}
        if not nCores:
            for step, frame in atomDict.items():
                molDir=Analyzer.atomFrameToMols(frame, step)
                allFramesMols[step] = molDir
                pass
        else:
            pool = multiprocessing.Pool(processes=nCores)
            asyResult = {}
            for step, frame in atomDict.items():
                asyResult[step] = pool.apply_async(
                    func=Analyzer.atomFrameToMols,
                    args=(frame, step, True),
                    )
                pass
            pool.close()
            pool.join()

            for step, result in asyResult.items():
                allFramesMols[step] = result.get()
                pass
        
        printf("#[toMolecules        ] Convert Atom to Molecule done")
        return allFramesMols

    @staticmethod
    def classifyMol(molDict:dict):
        printf("#[classifyMol        ] begin to classify")
        
        # get the hash code set
        allFrameHash = []
        for frame in molDict.values():
            for mol in frame.values():
                allFrameHash.append(mol.getHash())
        allFrameHash = set(allFrameHash)

        # initialize dict
        classifyDict, uqMolDict = {}, {}
        for hashStr in allFrameHash:
            classifyDict[hashStr] = {}
            uqMolDict[hashStr] = {}
            uqMolDict[hashStr]['mol'] = None
            uqMolDict[hashStr]['num'] = 0

            for steps in molDict.keys():
                classifyDict[hashStr][steps] = {}
                

        # classify
        for steps, frame in molDict.items():
            for idx, mol in frame.items():
                if mol.getHash() in allFrameHash:
                    classifyDict[mol.getHash()][steps][idx] = mol
                    uqMolDict[mol.getHash()]['mol'] = mol
                    uqMolDict[mol.getHash()]['num'] = uqMolDict[mol.getHash()]['num'] + 1

                else:
                    printf("#[classifyRex   ] hash code wrong")

        # printf eg and number
        frameNumb = len([step for step in molDict.keys()])
        for hashCode, egMol in uqMolDict.items():
            avgN = egMol['num']/frameNumb
            avgN = round(avgN, 3)
            uqMolDict[hashCode]['num'] = avgN

            printf("#[classifyMol        ] hashCode:", hashCode, \
                    "|avgN:", "{:>12}".format(avgN), \
                    "|eg:", egMol['mol'].getSymbols())
        
        # write
        printf("#[classifyMol        ] classfied done")
        return (classifyDict, uqMolDict)

    @staticmethod
    def getAllNumbFig(molDictSig: dict = None, molDictDict: dict = None):
        '''
        ## Function
        draw the number evolution of rex

        ## Parameters 
        select one from them:
        `molDictSig`: single `molDict`, type dict, such as `{step:{id:mol, , , }, , ,  }`;
        `molDictDict`: `dict` of `molDict`, type `{hashCode: {step:{id:mol, , , }, , ,  }, , , }`;
       
        ## Return
        `None`

        '''

        def eg(molDict: dict):
            for frame in molDict.values():
                if frame:
                    for mol in frame.values():
                        return mol
                else:
                    pass

        def sigFra(molDict: dict):
            stepList = sorted([step for step in molDict.keys()])
            timeList = [step*TIMESTEP for step in stepList]
            numbList = [len(molDict[step]) for step in stepList]
            avgNumb = sum(numbList)/len(numbList)
            avgNumb = round(avgNumb, 3)

            egMol = eg(molDict=molDict)
            hs = egMol.getHash()
            syb = egMol.getSymbols()

            printf("#[getNumbFig         ] hashCode:", str(hs), \
                    "|avgN:", "{:>12}".format(avgNumb), \
                    "|eg:", syb)

            drawFigSig(
                xList=timeList,
                yList=numbList,
                labels=(
                    str(hs) + "\n" + \
                    syb + " | " + str(avgNumb),
                    "Time / ps",
                    "Number",
                ),
                nameForSave="molN-{}-{}".format(syb, hs[-5:])
            )

            listToCSV(
                nameTuple=("timeList"),
                listTuple=(timeList, numbList),
                csvName="raw-molN-{}-{}.csv".format(syb, hs[-5:])
            )

            with open("avgMolN.csv", 'a+') as f:
                strW = "{}, {}, {}, \n".format(syb, avgNumb, hs)
                f.write(strW)
            f.close()

        ##
        printf("#[getNumbFig         ]beginning:")
        if molDictSig:
            sigFra(molDict=molDictSig)

        elif molDictDict:
            for molDict in molDictDict.values():
                sigFra(molDict=molDict)

        else:
            printf("#[getNumbFig         ] parameter wrong !")
        printf("#[getNumbFig         ] finished")



    ######## RDF type-type
    @staticmethod
    def getAtomAndAtomRDF_list(thisFrame:dict, pair:tuple, r=10, n=100, step = 0):
        '''
        ## Function
        get the atom-atom RDF

        ## Parameters
        `thisFrame`: this mol's frame
        `pair`: atom type tuple, for example (3, 4)
        `r`: the max length of r
        `n`: resolution

        ## Return
        `None`
        '''
        # get atom list that will be considered
        atomTypeA, atomTypeB = pair
        listA, listB = [], []
        for mol in thisFrame.values():
            for atom in mol.getNodes():
                if atomTypeA==atom.type:
                    listA.append(atom)
                else: pass

                if atomTypeB==atom.type:
                    listB.append(atom)
                else: pass
        # get density
        box=thisFrame[1].box
        lx = box[0][1]-box[0][0]
        ly = box[1][1]-box[1][0]
        lz = box[2][1]-box[2][0]
        volume = lx*ly*lz
        density = len(listB)/volume
        # get r list and surface area (volume) list
        r=min(lx/2-1, ly/2-1, lz/2-1, r)
        dR=r/n
        rList, sList = [], []
        R=0
        for i in range(n):
            rList.append(R)
            sList.append(4*3.1415926*dR*R**2)
            R += dR
        rList.append(R)
        sList.append(4*3.1415926*dR*R**2)
        # get n 
        nList = [0 for i in rList]
        for atomA in listA:
            for atomB in listB:
                if atomA.id != atomB.id:
                    length = atomA.distanceWith_PBC(atomB, lx, ly, lz)
                    for i in range(len(rList)-1):
                        if length > rList[i] and length < rList[i+1]:
                            nList[i+1] +=1
                        else: pass
                else: pass
        # get g(r)
        gList, cnList = [0, ], [0,]
        for i in range(1, len(rList)):
            gList.append(nList[i]/sList[i]/density/len(listA))
            cnList.append(nList[i]/len(listA))

        CNList = []
        for  i in range(len(cnList)):
            CNList.append(sum(cnList[0:i+1]))

        listToCSV(
            nameTuple=('rList', "g(r)", "CN"),
            listTuple=(rList, gList, CNList),
            csvName="raw-RDF-CN-of-{}-{}-stp{}-r{}-n{}.csv".format(ATOMTYPE[atomTypeA], ATOMTYPE[atomTypeB], step, r, n)
        )

        return (rList, gList, CNList)


    @staticmethod
    def getAtomAndAtomRDF(molDict, pair, step, r=10, n=100, smth = False):
        '''
        ## Function
        get the atom-atom RDF

        ## Parameters
        `molDict`: Molecule dict
        `pair`: atom type tuple, for example (3, 4) 
        `step`: the step that being count, type `list` or `int`
        `r`: the max length of r
        `n`: resolution

        ## Return
        `None`
        '''
        
        atomTypeA, atomTypeB = pair
        # get list that will be considered

        if isinstance(step, list):
            glistDict, cnlistDict = {}, {}
            for stp in step:
                frame = molDict[stp]
                rList, gList_, CNList_ = Analyzer.getAtomAndAtomRDF_list(frame, pair, r, n, stp)
                glistDict[stp] = gList_
                cnlistDict[stp] = CNList_
            
            gList, CNList = [], []
            
            for i in range(len(glistDict[step[0]])):
                gL = [glistDict[stp][i] for stp in step]
                gList.append(sum(gL)/len(gL))

                cL = [cnlistDict[stp][i] for stp in step]
                CNList.append(sum(cL)/len(cL))

            stepStr = "{}-{}-{}".format(step[0], step[-1], len(step))
            
        else:
            thisFrame = molDict[step]
            rList, gList, CNList = Analyzer.getAtomAndAtomRDF_list(thisFrame, pair, r, n, step)
            stepStr = str(step)
        
        
        # draw
        drawFigSig(
            type='line',
            xList=rList,
            yList=gList,
            labels=(
                "Radial Distribution Function",
                "r(" + ATOMTYPE[atomTypeA] + "-" + ATOMTYPE[atomTypeB] + ") / Angstrom",
                "g(r)"
            ),
            nameForSave="RDF-of-{}-{}-step{}-r{}-n{}".format(ATOMTYPE[atomTypeA], ATOMTYPE[atomTypeB], stepStr, r, n)
        )

        if smth:
            smoothR, smoothG = smooth(x=rList, y=gList, size=5)
            drawFigSig(
                type='line',
                xList=smoothR,
                yList=smoothG,
                labels=(
                    "Smoothed Radial Distribution Function",
                    "r(" + ATOMTYPE[atomTypeA] + "-" + ATOMTYPE[atomTypeB] + ") / Angstrom",
                    "g(r)"
                ),
                nameForSave="smthRDF-{}-{}-step{}-r{}-n{}".format(ATOMTYPE[atomTypeA], ATOMTYPE[atomTypeB], stepStr, r, n)
            )

            listToCSV(
                nameTuple=('smth_rList', "smth_g(r)", "rList", "CN"),
                listTuple=(smoothR, smoothG, rList, CNList),
                csvName="raw-smth-RDF-CN-of-{}-{}-stp{}-r{}-n{}.csv".format(ATOMTYPE[atomTypeA], ATOMTYPE[atomTypeB], stepStr, r, n)
            )
        else: pass

        drawFigSig(
            type='line',
            xList=rList,
            yList=CNList,
            labels=(
                "Coordination Number",
                "r(" + ATOMTYPE[atomTypeA] + "-" + ATOMTYPE[atomTypeB] + ") / Angstrom",
                "Coordination Number"
            ),
            nameForSave="CN-of-{}-{}-step{}-r{}-n{}".format(ATOMTYPE[atomTypeA], ATOMTYPE[atomTypeB], stepStr, r, n)
        )

        if isinstance(step, list):
            listToCSV(
                nameTuple=('rList', "g(r)", "CN"),
                listTuple=(rList, gList, CNList),
                csvName="raw-avg-RDF-CN-of-{}-{}-stp{}-r{}-n{}.csv".format(ATOMTYPE[atomTypeA], ATOMTYPE[atomTypeB], stepStr, r, n)
            )
        else:
            pass


        gMAX = max(gList)
        rMAX = 0
        cnMAX = 0
        for i in range(len(gList)):
            if gMAX == gList[i]:
                rMAX = rList[i]
                cnMAX = CNList[i+1]
            else: pass
        printf("#[getAtomAndAtomRDF  ] peak RDF: (", rMAX, gMAX,")")
        printf("#[getAtomAndAtomRDF  ] in CN(r): (", rMAX, cnMAX, ")")
        


    ############### RDF N-O
    @staticmethod
    def _getAtomAndAtomRDF_list_NO(thisFrame:dict, r=10, n=100, step = 0):
        '''
        ## Function
        get the atom-atom RDF

        ## Parameters
        `thisFrame`: this mol's frame
        `pair`: atom type tuple, for example (3, 4)
        `r`: the max length of r
        `n`: resolution

        ## Return
        `None`
        '''
        # get atom list that will be considered
        atomTypeA, atomTypeB = 'N', 'O'
        listA, listB, listB_w, listB_OH = [], [], [], []

        for mol in thisFrame.values():
            mol:Molecule
            if Mol('OH').equal(mol) or Mol('H3O2').equal(mol) or Mol('H5O3').equal(mol):
                oxygen = mol.getAimOxygen()
                listB_OH.append(oxygen)
                listB.append(oxygen)

            elif Mol('H2O').equal(mol):
                for atom in mol.getNodes():

                    if atomTypeB==ATOMTYPE[atom.type]:
                        listB_w.append(atom)
                        listB.append(atom)
                    else: pass

            else: # N
                for atom in mol.getNodes():

                    if atomTypeA==ATOMTYPE[atom.type]:
                        listA.append(atom)
                    else: pass

        # get density
        box=thisFrame[1].box
        lx = box[0][1]-box[0][0]
        ly = box[1][1]-box[1][0]
        lz = box[2][1]-box[2][0]
        volume = lx*ly*lz
        density = len(listB)/volume
        density_OH = len(listB_OH)/volume
        density_w = len(listB_w)/volume

        # get r list and surface area (volume) list
        r=min(lx/2-1, ly/2-1, lz/2-1, r)
        dR=r/n
        rList, sList = [], []
        R=0
        for i in range(n):
            rList.append(R)
            sList.append(4*3.1415926*dR*R**2)
            R += dR
        rList.append(R)
        sList.append(4*3.1415926*dR*R**2)

        # get n 
        nList = [0 for i in rList]
        for atomA in listA: # N
            for atomB in listB: # O
                if atomA.id != atomB.id:
                    length = atomA.distanceWith_PBC(atomB, lx, ly, lz)
                    for i in range(len(rList)-1):
                        if length > rList[i] and length < rList[i+1]:
                            nList[i+1] +=1
                        else: pass
                else: pass

        nList_OH = [0 for i in rList]
        for atomA in listA: # N
            for atomB in listB_OH: # O_OH
                if atomA.id != atomB.id:
                    length = atomA.distanceWith_PBC(atomB, lx, ly, lz)
                    for i in range(len(rList)-1):
                        if length > rList[i] and length < rList[i+1]:
                            nList_OH[i+1] +=1
                        else: pass
                else: pass

        nList_w = [0 for i in rList]
        for atomA in listA: # N
            for atomB in listB_w: # O_w
                if atomA.id != atomB.id:
                    length = atomA.distanceWith_PBC(atomB, lx, ly, lz)
                    for i in range(len(rList)-1):
                        if length > rList[i] and length < rList[i+1]:
                            nList_w[i+1] +=1
                        else: pass
                else: pass
        # get g(r)
        gList, cnList = [0, ], [0,]
        for i in range(1, len(rList)):
            gList.append(nList[i]/sList[i]/density/len(listA))
            cnList.append(nList[i]/len(listA))

        gList_OH, cnList_OH = [0, ], [0,]
        for i in range(1, len(rList)):
            gList_OH.append(nList_OH[i]/sList[i]/density_OH/len(listA))
            cnList_OH.append(nList_OH[i]/len(listA))

        gList_w, cnList_w = [0, ], [0,]
        for i in range(1, len(rList)):
            gList_w.append(nList_w[i]/sList[i]/density_w/len(listA))
            cnList_w.append(nList_w[i]/len(listA))

        ##
        CNList = []
        for  i in range(len(cnList)):
            CNList.append(sum(cnList[0:i+1]))

        CNList_OH = []
        for  i in range(len(cnList_OH)):
            CNList_OH.append(sum(cnList_OH[0:i+1]))

        CNList_w = []
        for  i in range(len(cnList_w)):
            CNList_w.append(sum(cnList_w[0:i+1]))

        printf("#[getAtomAndAtomRDF  ] the Bias:  < (bigger than true value) ", dR)
        listToCSV(
            nameTuple=('rList', "g(r)", "CN"),
            listTuple=(rList, gList, CNList),
            csvName="raw-RDF-N_O-of-{}-{}-stp{}-r{}-n{}.csv".format('N', 'O', step, r, n)
        )

        listToCSV(
            nameTuple=('rList', "g(r)", "CN"),
            listTuple=(rList, gList_OH, CNList_OH),
            csvName="raw-RDF-N_O_OH-of-{}-{}-stp{}-r{}-n{}.csv".format('N', 'O_OH', step, r, n)
        )

        listToCSV(
            nameTuple=('rList', "g(r)", "CN"),
            listTuple=(rList, gList_w, CNList_w),
            csvName="raw-RDF-N_O_W-of-{}-{}-stp{}-r{}-n{}.csv".format('N', 'O_W', step, r, n)
        )

        return (rList, gList, CNList, gList_OH, CNList_OH, gList_w, CNList_w)

    @staticmethod
    def getAtomAndAtomRDF_NO(molDict, step, r=10, n=100, smth = False):
        '''
        ## Function
        get the atom-atom RDF

        ## Parameters
        `allMolFileName`: mol dict file's name, type `str`
        `pair`: atom type tuple, for example (3, 4)
        `step`: the step that being count, type `list` or `int`
        `r`: the max length of r
        `n`: resolution

        ## Return
        `None`
        '''
        
        # get list that will be considered

        if isinstance(step, list):
            glistDict, cnlistDict = {}, {}
            glistDict_OH, cnlistDict_OH = {}, {}
            glistDict_W, cnlistDict_W = {}, {}

            for stp in step:
                frame = molDict[stp]
                rList, gList_, CNList_, gList_OH_, CNList_OH_, gList_w_, CNList_w_ = Analyzer._getAtomAndAtomRDF_list_NO(frame, r, n, stp)

                glistDict[stp] = gList_
                cnlistDict[stp] = CNList_

                glistDict_OH[stp] = gList_OH_
                cnlistDict_OH[stp] = CNList_OH_

                glistDict_W[stp] = gList_w_
                cnlistDict_W[stp] = CNList_w_
            
            gList, CNList = [], []
            gList_OH, CNList_OH = [], []
            gList_W, CNList_W = [], []
            
            for i in range(len(glistDict[step[0]])):
                gL = [glistDict[stp][i] for stp in step]
                gList.append(sum(gL)/len(gL))

                cL = [cnlistDict[stp][i] for stp in step]
                CNList.append(sum(cL)/len(cL))

            for i in range(len(glistDict_OH[step[0]])):
                gL = [glistDict_OH[stp][i] for stp in step]
                gList_OH.append(sum(gL)/len(gL))

                cL = [cnlistDict_OH[stp][i] for stp in step]
                CNList_OH.append(sum(cL)/len(cL))

            for i in range(len(glistDict_W[step[0]])):
                gL = [glistDict_W[stp][i] for stp in step]
                gList_W.append(sum(gL)/len(gL))

                cL = [cnlistDict_W[stp][i] for stp in step]
                CNList_W.append(sum(cL)/len(cL))

            stepStr = "{}-{}-{}".format(step[0], step[-1], len(step))
            
        else:
            thisFrame = molDict[step]
            rList, gList, CNList, gList_OH, CNList_OH, gList_W, CNList_W = Analyzer._getAtomAndAtomRDF_list_NO(thisFrame, r, n, stp)
            stepStr = str(step)
        
        
        # draw
        drawFigSig(
            type='line',
            xList=rList,
            yList=gList,
            labels=(
                "Radial Distribution Function",
                "r(N-O) / Angstrom",
                "g(r)"
            ),
            nameForSave="RDF-of-{}-{}-step{}-r{}-n{}".format('N', 'O', stepStr, r, n)
        )

        drawFigSig(
            type='line',
            xList=rList,
            yList=gList_OH,
            labels=(
                "Radial Distribution Function",
                "r(N-O_OH) / Angstrom",
                "g(r)"
            ),
            nameForSave="RDF-of-{}-{}_OH-step{}-r{}-n{}".format('N', 'O', stepStr, r, n)
        )

        drawFigSig(
            type='line',
            xList=rList,
            yList=gList_W,
            labels=(
                "Radial Distribution Function",
                "r(N-O_W) / Angstrom",
                "g(r)"
            ),
            nameForSave="RDF-of-{}-{}_W-step{}-r{}-n{}".format('N', 'O', stepStr, r, n)
        )


        if smth:
            smoothR, smoothG = smooth(x=rList, y=gList, size=5)
            smoothR, smoothG_OH = smooth(x=rList, y=gList_OH, size=5)
            smoothR, smoothG_W = smooth(x=rList, y=gList_W, size=5)

            drawFigSig(
                type='line',
                xList=smoothR,
                yList=smoothG,
                labels=(
                    "Smoothed Radial Distribution Function",
                    "r(N-O) / Angstrom",
                    "g(r)"
                ),
                nameForSave="smthRDF-{}-{}-step{}-r{}-n{}".format('N', 'O', stepStr, r, n)
            )

            drawFigSig(
                type='line',
                xList=smoothR,
                yList=smoothG_OH,
                labels=(
                    "Smoothed Radial Distribution Function",
                    "r(N-O_OH) / Angstrom",
                    "g(r)"
                ),
                nameForSave="smthRDF-{}-{}-step{}-r{}-n{}".format('N', 'O_OH', stepStr, r, n)
            )

            drawFigSig(
                type='line',
                xList=smoothR,
                yList=smoothG_W,
                labels=(
                    "Smoothed Radial Distribution Function",
                    "r(N-O_W) / Angstrom",
                    "g(r)"
                ),
                nameForSave="smthRDF-{}-{}-step{}-r{}-n{}".format('N', 'O_W', stepStr, r, n)
            )


            listToCSV(
                nameTuple=('smth_rList', "smth_g(r)", "rList", "CN"),
                listTuple=(smoothR, smoothG, rList, CNList),
                csvName="raw-smth-RDF-CN-of-{}-{}-stp{}-r{}-n{}.csv".format('N', 'O', stepStr, r, n)
            )

            listToCSV(
                nameTuple=('smth_rList', "smth_g(r)", "rList", "CN"),
                listTuple=(smoothR, smoothG_OH, rList, CNList_OH),
                csvName="raw-smth-RDF-CN-of-{}-{}-stp{}-r{}-n{}.csv".format('N', 'O_OH', stepStr, r, n)
            )

            listToCSV(
                nameTuple=('smth_rList', "smth_g(r)", "rList", "CN"),
                listTuple=(smoothR, smoothG_W, rList, CNList_W),
                csvName="raw-smth-RDF-CN-of-{}-{}-stp{}-r{}-n{}.csv".format(
                    'N', 'O', stepStr, r, n)
            )
        else: pass

        

        drawFigSig(
            type='line',
            xList=rList,
            yList=CNList,
            labels=(
                "Coordination Number",
                "r(N-O) / Angstrom",
                "Coordination Number"
            ),
            nameForSave="CN-of-{}-{}-step{}-r{}-n{}".format('N', 'O', stepStr, r, n)
        )

        drawFigSig(
            type='line',
            xList=rList,
            yList=CNList_OH,
            labels=(
                "Coordination Number",
                "r(N-O_OH) / Angstrom",
                "Coordination Number"
            ),
            nameForSave="CN-of-{}-{}-step{}-r{}-n{}".format('N', 'O_OH', stepStr, r, n)
        )

        drawFigSig(
            type='line',
            xList=rList,
            yList=CNList_W,
            labels=(
                "Coordination Number",
                "r(N-O_W) / Angstrom",
                "Coordination Number"
            ),
            nameForSave="CN-of-{}-{}-step{}-r{}-n{}".format('N', 'O_W', stepStr, r, n)
        )


        if isinstance(step, list):
            listToCSV(
                nameTuple=('rList', "g(r)", "CN"),
                listTuple=(rList, gList, CNList),
                csvName="raw-avg-RDF-CN-of-{}-{}-stp{}-r{}-n{}.csv".format('N', 'O', stepStr, r, n)
            )

            listToCSV(
                nameTuple=('rList', "g(r)", "CN"),
                listTuple=(rList, gList_OH, CNList_OH),
                csvName="raw-avg-RDF-CN-of-{}-{}-stp{}-r{}-n{}.csv".format('N', 'O_OH', stepStr, r, n)
            )

            listToCSV(
                nameTuple=('rList', "g(r)", "CN"),
                listTuple=(rList, gList_W, CNList_W),
                csvName="raw-avg-RDF-CN-of-{}-{}-stp{}-r{}-n{}.csv".format('N', 'O_W', stepStr, r, n)
            )
        else:
            pass


        gMAX = max(gList)
        rMAX = 0
        cnMAX = 0
        for i in range(len(gList)):
            if gMAX == gList[i]:
                rMAX = rList[i]
                cnMAX = CNList[i+1]
            else: pass
        printf("#[getAtomAndAtomRDF  ] peak RDF: (", rMAX, gMAX,")")
        printf("#[getAtomAndAtomRDF  ] in CN(r): (", rMAX, cnMAX, ")")
        

    @staticmethod
    def getAngle(atomA, atomB:Atom, atomC, lx, ly, lz, r):
        if  atomB.id != atomA.id and \
            atomB.id != atomC.id and \
            atomA.id != atomC.id and \
            atomB.distanceWith_PBC(atomA, lx, ly, lz) <= r and \
            atomB.distanceWith_PBC(atomC, lx, ly, lz) <= r :
            
            A = atomB.adjust(atomA, lx, ly, lz)
            C = atomB.adjust(atomC, lx, ly, lz)
            return atomB.angle(atomA=A, atomC=C)

        else:
            return None

    @staticmethod
    def getAtomsADF(molFrame, pair, sample=1.0, r=3, n=100, nCores=10):
        '''
        ## Function
        get angle distribution function of atoms type in pair

        ## Parameters
        `molFrame`: single molecule dict frame, type {1:mol1, , , }
        `pair`: (typeA, typeB, typeC), typeB is the core atom
        `sample`: ratio of molecules for counting, which > 0.0 and <= 1.0
        `r`: max length of r
        `n`: resolution

        ## Return
        `None`
        '''

        printf("#[getAtomsADF        ] Begin to get ADF")

        # get frame, box and r
        box = molFrame[1].box
        lx = box[0][1]-box[0][0]
        ly = box[1][1]-box[1][0]
        lz = box[2][1]-box[2][0]
        r = min(lx/2-0.1, ly/2-0.1, lz/2-0.1, r)
        # get atom list that will be considered
        atomTypeA, atomTypeB, atomTypeC = pair
        listA, listB, listC = [], [], []
        molNum = len(molFrame)
        molNumSampled = int(molNum * sample)
        allMolList = [mol for mol in molFrame.values()]
        molList = random.sample(allMolList, molNumSampled)
        
        for mol in molList:
            for atom in mol.getNodes():
                if atomTypeA == atom.type:
                    listA.append(atom)
                else: pass

                if atomTypeB == atom.type:
                    listB.append(atom)
                else: pass

                if atomTypeC == atom.type:
                    listC.append(atom)
                else: pass
        
        # get atom pairs that will be considered
        pool = multiprocessing.Pool(nCores)
        asyncResultList = []

        for atomB in listB:
            for atomA in  listA:
                for atomC in listC:

                    asyncResultList.append(
                        pool.apply_async(
                            Analyzer.getAngle,
                            args=(atomA, atomB, atomC, lx, ly, lz, r),
                        )
                    )          
        pool.close()
        pool.join()

        angleList = []
        for result in asyncResultList:
            angle = result.get()
            if angle:
                angleList.append(angle)
            else: pass

        # get all angle
        dAngle = 180/n
        angle = 0
        xList = []
        for i in range(int(n)):
            xList.append(angle)
            angle = angle + dAngle
        xList.append(angle)
        # get n
        nList = [0 for i in xList]
        for ang in angleList:
            for i in range(len(xList)-1):
                if ang > xList[i] and ang <= xList[i+1]:
                    nList[i+1] = nList[i+1] + 1
        # get p
        pairNumb = len(angleList)
        pList = [i/pairNumb for i in nList]

        drawFigSig(
            type='line',
            xList=xList,
            yList=pList,
            labels=(
                "Angle distribution function",
                "angle(" + 
                ATOMTYPE[atomTypeA] + "-" +
                ATOMTYPE[atomTypeB] + "-" +
                ATOMTYPE[atomTypeC] + 
                ") / degree",
                "Probability"
            ),
            nameForSave="ADF-of-{}-{}-{}".format(
                ATOMTYPE[atomTypeA], ATOMTYPE[atomTypeB], ATOMTYPE[atomTypeC]
            )
        )

        listToCSV(
            nameTuple=("Angle", "P"),
            listTuple=(xList, pList),
            csvName = "ADF-of-{}-{}-{}.csv".format(
                ATOMTYPE[atomTypeA], ATOMTYPE[atomTypeB], ATOMTYPE[atomTypeC]
            )
        )

        pMAX = max(pList)
        angleMAX = 0
        for i in range(len(pList)):
            if pMAX == pList[i]:
                angleMAX = xList[i]
            else:
                pass
        printf("#[getAtomsADF        ] The peak is (", angleMAX, pMAX, ')')


        

    
    def splitToGroup(self, allMolFile:str, referenceFrame:dict, aimAtom='N', length=2): 
        '''
        Functions:
        ---
        split aim molecule to aim group according to aim atoms and the length

        Parameters:
        ---
        `allMolFile`: the mol dict's file name, type `str`
        `referenceFrame`: the reference frame before the groups react with OH-, type `dict` or `int`
        `aimAtom`: the aimed atom type, type `str`
        `length`: the length with `aimAtom`, type `int`

        Return:
        ---
        `None`, and dump the file to "`allMolFile[0:-3]`+`aimAtom`+str(`length`)+'.pkl"

        '''
        # get files
        printf('#[splitToGroup] Begin splitting molecule to groups')
        allFrames = load(allMolFile)
        allMolFrames = copy.deepcopy(allFrames)
        if isinstance(referenceFrame, int):
            referenceFrame=copy.deepcopy(allMolFrames[referenceFrame])
        else: pass

        # get all aim ID
        allAimAtomsID = []
        for mol in referenceFrame.values():
            if mol._isContainType(type=aimAtom):
                for atom in mol.getNodesAround(coreAtomType=aimAtom, length=length):
                    allAimAtomsID.append(atom.id)
            else:
                allAimAtomsID = allAimAtomsID + mol._getNodesID()

        # transfer the frames
        for frame in allMolFrames.values():
            for mol in frame.values():
                mol.unFreeze()
                for atom in mol.getNodes():
                    if atom.id in allAimAtomsID:
                        pass
                    else:
                        mol._removeNode(atom)

        # reget all the frames
        frames = {}
        for step, frame in allMolFrames.items():
            molDict = {}
            id=1
            for mol in frame.values():
                g = mol.topology
                for cc in nx.connected_components(g):
                    subG = g.subgraph(cc)
                    molDict[id] = Molecule()
                    molDict[id].topology = subG
                    molDict[id].step = step
                    molDict[id].id = id
                    id = id + 1
            frames[step] = molDict
        molFileName = allMolFile[0:-3]+aimAtom+'.'+str(length)+'.pkl'
        dump(frames, molFileName)
        self._molFiles.append(molFileName)
        self.__dumpFileInfo()
        printf('#[splitToGroup] dump to', molFileName)
                        
    def trackMol(self, mol:Molecule, allMolFile:str):
        '''
        ## Function
        tracking the `mol` in all frames that stored in `allMolFile`

        ## Parameters
        `mol`: the aimed molecule
        `allMolFile`: the file that stores the mol `dict`

        ## Return
        `{step{id:mol, , ,}, , , }`
        '''
        targetDict = {}
        allFrames = load(allMolFile)
        for step, frame in allFrames.items():
            targetThisFrame = {}

            for molID, molecule in frame.items():
                if molecule.isHasComAtomsWith(mol):
                    targetThisFrame[molID] = molecule
                else: pass

            targetDict[step] = targetThisFrame
        return targetDict
    
    def trackAtom(self, atomID, allMolFile):
        '''
        ## Function
        track mol containg the atom in all frames that stored in `allMolFile`

        ## Parameters
        `atomID`: the aimed atom's ID
        `allMolFile`: the file that stores the mol `dict`

        ## Return
        `{step{id:mol, , ,}, , , }`
        '''
        targetDict = {}
        allFrames = load(allMolFile)

        for step, frame in allFrames.items():
            targetThisFrame = {}

            for molID, molecule in frame.items():
                if molecule._isContainID(atomID):
                    targetThisFrame[molID] = molecule
                else: pass

            targetDict[step] = targetThisFrame
        return targetDict
    
    def trackType(self, atomType, allMolFile:str):
        '''
        ## Function
        track mol containg atoms type `atomType` in all frames that stored in `allMolFile`

        ## Parameters
        `atomType`: the aimed atom's type, `str`=C/H/O/N, or `list`=['C', 'H', , ,]
        `allMolFile`: the file that stores the mol `dict`

        ## Return
        `{step{id:mol, , ,}, , , }`
        '''
        targetDict = {}
        allFrames = load(allMolFile)

        if isinstance(atomType, str):
            for step, frame in allFrames.items():
                targetThisFrame = {}
                for molID, molecule in frame.items():
                    if molecule._isContainType(atomType):
                        targetThisFrame[molID] = molecule
                    else: pass

                targetDict[step] = targetThisFrame
            return targetDict
        
        elif isinstance(atomType, list):
            for step, frame in allFrames.items():
                targetThisFrame = {}
                for molID, molecule in frame.items():
                    for types in atomType:
                        if molecule._isContainType(types):
                            targetThisFrame[molID] = molecule
                            break
                        else: pass

                targetDict[step] = targetThisFrame
            return targetDict


    def getBondNumbEvolu(self, allMolFile:str):
        def sortBondType(x, y):
            typeX, typeY = x.type, y.type
            if typeX > typeY:
                return (typeY, typeX)
            else:
                return (typeX, typeY)
        ########
        
        # get all the number
        allFrames = load(allMolFile)
        allFrameBonds = {}
        allFrameUniqueBonds = []
        stepList = []
        for step, frame in allFrames.items():
            stepList.append(step)
            thisFrameBonds = {}
            thisFrameUniqueBonds = []
            for mol in frame.values():
                for atomA, atomB in mol.getEdges():
                    type1, type2 = sortBondType(atomA, atomB)
                    if (type1, type2) in thisFrameUniqueBonds:
                        thisFrameBonds[(type1, type2)] = thisFrameBonds[(type1, type2)] + 1
                    else:
                        thisFrameUniqueBonds.append((type1, type2))
                        thisFrameBonds[(type1, type2)] = 1
            allFrameBonds[step] = thisFrameBonds
            allFrameUniqueBonds = allFrameUniqueBonds + thisFrameUniqueBonds
        allFrameUniqueBonds = list(set(allFrameUniqueBonds))
        
        # initialize the bond number list's dict
        timeList = [step*self._timeStep for step in stepList]
        bondNumbListDict = {}
        for bond in allFrameUniqueBonds:
            bondNumbListDict[bond] = []

        # write number to the bond number list's dict
        for step, frame in allFrameBonds.items():
            for bond in allFrameUniqueBonds:
                if bond in frame.keys():
                    bondNumbListDict[bond].append(frame[bond])
                else:
                    bondNumbListDict[bond].append(0)
    
        # prepare list's tuple for drawing the figures
        yLists, legend = [], []
        for bond, numberList in bondNumbListDict.items():
            yLists.append(numberList)
            label = ATOMTYPE[bond[0]] + '-' + ATOMTYPE[bond[1]]
            legend.append(label)
        yLists = tuple(yLists)
        legend = tuple(legend)

        drawFigMulti(
            xList=timeList,
            yLists=yLists,
            labels=(
                "Number evolution of bonds",
                "Time / "+self._unit,
                "Number of bonds",
            ),
            legends=legend,
            type='line',
        )

    @staticmethod
    def getRelatedMolIDList(mol, frameDirt: dict):
        idList = []
        atomIdSet = set(mol._getNodesID())

        for id, mols in frameDirt.items():
            commonAtoms = set(atomIdSet) & set(mols._getNodesID())
            if commonAtoms:
                idList.append(id)
                atomIdSet = atomIdSet - commonAtoms

                if len(atomIdSet) == 0:
                    return idList
                else:
                    pass
            else:
                pass
        return idList

    @staticmethod
    def mpGetRex(thisFra:dict, thisStep:int, nextFra:dict, nextStep:int):
        topThisFrame = nx.DiGraph()
        for id, mols in thisFra.items():
            molsAfterID = Analyzer.getRelatedMolIDList(mols, nextFra)
            for idx in molsAfterID:
                topThisFrame.add_edge(thisFra[id], nextFra[idx])

        elemRexDictThisFrame = {}
        rexID = 1
        for wcc in nx.weakly_connected_components(topThisFrame):
            sub = topThisFrame.subgraph(wcc)
            ilist, jlist = [], []
            for m, n in sub.edges:
                ilist.append(m)
                jlist.append(n)
            r, p = set(ilist), set(jlist)
            elemRexDictThisFrame[rexID] = Reaction(
                R=list(r), P=list(p), stepR=thisStep, stepP=nextStep, id=rexID
            )
            rexID = rexID + 1
            pass

        del topThisFrame, thisFra, nextFra

        return (thisStep, nextStep, elemRexDictThisFrame)
    
    @staticmethod
    def toReactions(molDict, nCores):
        printf("#[toReactions        ] Begin convert Molecule to Reaction")
        # set the pool
        stepsList = [steps for steps in molDict.keys()]
        pool = multiprocessing.Pool(nCores)
        asyncResultList = []

        for i in range(len(stepsList)-1):
            thisFrame = molDict[stepsList[i]]
            thisStep = stepsList[i]
            nextFrame = molDict[stepsList[i+1]]
            nextStep = stepsList[i+1]

            asyncResultList.append(
                pool.apply_async(
                    Analyzer.mpGetRex,
                    args=(thisFrame, thisStep, nextFrame, nextStep),
                )
            )

        pool.close()
        pool.join()
        
        # store
        rexDict={}
        for result in asyncResultList:
            (thisStep, nextStep, thisResult) = result.get()
            rexDict[(thisStep, nextStep)] = thisResult
        
        printf("#[toReactions        ] Convert Molecule to Reaction done")
        return rexDict
    

    @staticmethod
    def spfRex(rex:Reaction) -> Reaction:
        '''
        ## Function
        simplify reactions

        ## Parameters

        '''


        hashR = [sps.getHash() for sps in rex.R]
        hashP = [sps.getHash() for sps in rex.P]
        removeCom(hashR, hashP)

        R, P = [], []
        for hashCode in hashR:
            for mol in rex.R:
                if mol.getHash() == hashCode:
                    R.append(mol)
                    break
                else: pass

        for hashCode in hashP:
            for mol in rex.P:
                if mol.getHash() == hashCode:
                    P.append(mol)
                    break
                else: pass

        newRex = Reaction(
            R=R,
            P=P,
            stepR=rex.stepR,
            stepP=rex.stepP,
            id=rex.id
        )
        return newRex
    

    @staticmethod
    def classifyRex(rexDict:dict, rexList:list=None, uqMols:list=None):
        '''
        ## Function 
        classify the elem reactions by their hash code

        ## Parameters
        `rexDict`: the total rex dict, type `str`, with `.pkl` in it
        `rexList`: the target rex's list, type `[rex, , , ]`. if `None`, classfify all the rex
        `uqMols`: the target mol's list, type `[mol, , , ]`. if `None`, classfify all the mol

        ## Return
        `{hashStr:[rex, , , ], , , }`

        '''

        # get the files
        printf("#[classifyRex        ] begin to classify the reactions")

        # get the hash code list of the rex list
        if rexList:
            hashList = [rex.getHash() for rex in rexList]
            hashSet = set(hashList)
        else:
            hashList = []
            for frame in rexDict.values():
                for rex in frame.values():
                    hashList.append(rex.getHash())
            hashSet = set(hashList)

        # initialize dict
        classifyDict, uqRexDict = {}, {}
        for hashStr in hashSet:
            classifyDict[hashStr] = {}
            uqRexDict[hashStr] = {}
            uqRexDict[hashStr]['rex'] = None
            uqRexDict[hashStr]['num'] = 0

            for steps in rexDict.keys():
                classifyDict[hashStr][steps] = {}
                
            
        # classify 
        for steps, frame in rexDict.items():
            for idx, rex in frame.items():
                if rex.getHash() in hashSet:
                    classifyDict[rex.getHash()][steps][idx] = rex
                    uqRexDict[rex.getHash()]['rex'] = rex
                    uqRexDict[rex.getHash()]['num'] = uqRexDict[rex.getHash()]['num'] + 1

                else:
                    printf("#[classifyRex        ] hash code wrong")


        # printf
        if uqMols: #
            n = 0
            for mol in uqMols:
                molHash = mol.getHash()
                printf(" _____________________________________________")
                printf("|idxInList:", n, "|molecule:", mol.getSymbols(), "|molHash:", molHash)

                avgNumbList = set([eg['num'] for eg in uqRexDict.values()])
                avgNumbList = sorted(avgNumbList, reverse=True)
                for avgN in avgNumbList:
                    for hashCode, eg in uqRexDict.items():
                        if eg['num'] == avgN:

                            if molHash in eg['rex'].getSpsHashR() and molHash in eg['rex'].getSpsHashP():
                                printf("inRP-->rexHash:", hashCode, "|num:", eg['num'], "|eg:", Analyzer.spfRex(eg['rex']), '|details:', str(eg['rex']).split(']')[-1])
                            elif molHash in eg['rex'].getSpsHashR() and molHash not in eg['rex'].getSpsHashP():
                                printf("inR--->rexHash:", hashCode, "|num:", eg['num'], "|eg:", Analyzer.spfRex(eg['rex']), '|details:', str(eg['rex']).split(']')[-1])
                            elif molHash not in eg['rex'].getSpsHashR() and molHash in eg['rex'].getSpsHashP():
                                printf("inP--->rexHash:", hashCode, "|num:", eg['num'], "|eg:", Analyzer.spfRex(eg['rex']), '|details:', str(eg['rex']).split(']')[-1])
                            else:
                                pass

                        else:
                            pass
                n = n + 1

        else:
            n=1
            avgNumbList = set([eg['num'] for eg in uqRexDict.values()])
            avgNumbList = sorted(avgNumbList, reverse=True)
            f= open("allRexNumb.csv", 'w')
            for avgN in avgNumbList:
                for hashCode, eg in uqRexDict.items():
                    if eg['num'] == avgN:
                        printf(n, "|rexHash:", hashCode, "|num:", eg['num'], "|eg:", Analyzer.spfRex(eg['rex']), '|details:', str(eg['rex']).split(']')[-1])
                        string = "rexHash, {}, numb, {}, eg, {}, {}, detail, {}, \n".format(
                            hashCode, 
                            eg['num'], 
                            str(Analyzer.spfRex(eg['rex'])).split(']')[0] + "]",
                            str(Analyzer.spfRex(eg['rex'])).split(']')[-1], 
                            str(eg['rex']).split(']')[-1]
                        )
                        
                        f.write(string)
                        n = n + 1
                    else:
                        pass
                pass
            pass
            f.close()
            


        printf("#[classifyRex        ] classify the reactions done")
        return (classifyDict, uqRexDict)
    
    @staticmethod
    def getRexNumb(rexDictDict:dict):
        '''
        ## Function
        draw the number evolution of rex

        ## Parameters 
        select one from them:
        `rexDictSig`: single `rexDict`, type dict, such as `{stepPair:{id:rex, , , }, , ,  }`;
        `rexDictDict`: `dict` of `rexDict`, type `{hashCode: {stepPair:{id:rex, , , }, , ,  }, , , }`;
       
        ## Return
        `None`

        '''

        def eg(rexDict:dict):
            for frame in rexDict.values():
                if frame:
                    for rex in frame.values():
                        return rex
                else:
                    pass

        def sigFra(rexDict:dict):
            stepPairList = sorted([stepPair for stepPair in rexDict.keys()])
            timeList = [sum(stepPair)*TIMESTEP/2 for stepPair in stepPairList]
            numbList = [len(rexDict[stepPair]) for stepPair in stepPairList]
            

            egRex = eg(rexDict=rexDict)
            hs = egRex.getHash()
            printf("#[getRexNumb         ] hs:", hs, "|N:", sum(numbList), "|eg:", Analyzer.spfRex(egRex), '|details:', str(egRex).split(']')[-1])

            drawFigSig(
                xList=timeList,
                yList=numbList,
                labels=(
                    "{}] \n {} \n {}".format(
                        str(egRex).split("]")[0], 
                        str(Analyzer.spfRex(egRex)).split("]")[-1],
                        str(egRex).split("]")[-1]
                        ),
                    "Time / ps",
                    "Number",
                    ),
                nameForSave="rexN-{}-{}-id{}".format(egRex.stepR, egRex.stepP, egRex.id))
            
            listToCSV(
                nameTuple=("timeList"),
                listTuple=(timeList, numbList),
                csvName="raw-rexN-{}-{}-id{}.csv".format(egRex.stepR, egRex.stepP, egRex.id))
            
        ##
        for hashCode, rexDict in rexDictDict.items():
            sigFra(rexDict=rexDict)
        printf("#[getRexNumb         ] finished")

    
    @staticmethod
    def drawRexNet(uqMolDict:dict, uqRexDict:dict, lowestN = 0):
        '''
        ## Function
        get and draw all unique elemtary reactions
        
        ## Parameters
        `uqSpecies`: the unique species's `list`, type `[mol, , , ]`
        `uqRexList`: the unique elemRex's `list`, type `[elemRex, , , ]`
        `rexDictName`: name of the rex `dict`, type `str`, with `.pkl` in it
        
        ## Return
        `idxDict`: type `{idx:mol, , ,}`
        '''
        printf("#[drawRexNet         ] begin to: draw the rexNet")
        

        # node is hash code
        idx = 1
        nodesDict = {}
        for hs, mol in uqMolDict.items():
            if mol['num'] >= lowestN:
                node = {}
                node['idx'] = idx
                node['mol'] = mol['mol']
                node['num'] = mol['num']
                node['hs'] = hs
                node['sb'] = mol['mol'].getSymbols()

                nodesDict[hs] = node
                idx = idx + 1

            else:
                pass

        # get graph
        G = nx.DiGraph()
        G.add_nodes_from([hs for hs in nodesDict.keys()])

        # get the node label dict and idxDict
        labels, idxDict = {}, {}
        for hs, node in nodesDict.items():
            labels[hs] = '[ID:{} |N:{:.2f}]\n{}'.format(
                node['idx'],
                node['num'],
                node['sb'],
                )
            idxDict[node['idx']] = node['mol']

        # get all the edges from the reactions
        for eg in uqRexDict.values():
            rex = eg['rex']
            for i,j in [(i, j) for i in rex.getSpsHashR() for j in rex.getSpsHashP()]:
                if i in nodesDict.keys() and j in nodesDict.keys() and i != j :
                    G.add_edge(i, j)
                else: pass

        # get edge label
        edgeLabels = {}
        for r, p in G.edges:
            rexNumb = 0
            for eg in uqRexDict.values():
                rex = eg['rex']

                if r in rex.getSpsHashR() and p in rex.getSpsHashP():
                    rexNumb = rexNumb + eg['num']
                else: pass

            string= "{}-{}|{}".format(
                nodesDict[r]['idx'], 
                nodesDict[p]['idx'], 
                rexNumb, 
                )

            edgeLabels[(r, p)] = string
            printf("#[drawRexNet         ] rexNumb :", string)

        
    
        
        if lowestN == 0:
            nodeN = 10
        else:
            nodeN = len(uqMolDict)
        # draw the figure
        fig, ax = plt.subplots(figsize=(2*nodeN, 2*nodeN))
        pos = nx.circular_layout(G)
        nx.draw_networkx(
            G=G,
            pos=pos,
            arrows=20,
            node_size=7000,
            node_color='w',
            labels=labels,
            font_color='red',
            font_size=12+nodeN,
        )
        nx.draw_networkx_edge_labels(
            G = G,
            pos = pos,
            edge_labels=edgeLabels,
            label_pos=0.3,
            font_size=9+nodeN,
            )

        # dump
        fig.savefig(fname="RexNet.png")
        plt.cla()
        plt.close()

        for idx, mols in idxDict.items():
            printf("#[drawRexNet         ] mol idx :", idx, "|symbols:", mols.getSymbols(), "|hashCode:", mols.getHash())
        printf("#[drawRexNet         ] store as: RexNet.png")

        return idxDict


    @staticmethod
    def getRexFrom(fromMol:Reaction, rexDict: dict, show=False):
        '''
        ## Function
        get the reactions from `fromID`, according to the `idxDict` and `rexDict`

        ## Parameters
        `fromID`: the reaction's common reactant's id, type `mol`
        `idxDict`: mol idx's `dict`, type `{idx:mol, , ,}`
        `rexDict`: all the elem rex's `dict`, type `{stepPair:{id:rex, , , }, , ,  }`

        ## Return
        the searched rexDict, type `{stepPair:{id:rex, , , }, , ,  }`

        '''
        printf("#[getRexFrom         ] get rex from", fromMol.getSymbols())
        allFrames = {}
        for steps, frame in rexDict.items():
            thisFrame = {}
            for idx, elemRex in frame.items():
                if elemRex.isRex() and elemRex.isContainR(fromMol):
                    thisFrame[idx] = elemRex
                else:
                    pass
            allFrames[steps] = thisFrame

        if show:
            for frame in allFrames.values():
                for elemRex in frame.values():
                    printf(elemRex)
        else:
            pass
        return allFrames
    

    @staticmethod
    def getRexTo(toMol:Reaction, rexDict: dict, show=False):
        '''
        ## Function
        get the reactions to the `toID`, according to the `idxDict` and `rexDict`

        ## Parameters
        `toID`: the reaction's common product's id, type `int`
        `idxDict`: mol idx's `dict`, type `{idx:mol, , ,}`
        `rexDict`: all the elem rex's `dict`, type `{stepPair:{id:rex, , , }, , ,  }`
        `printf`: whether print the searched reactions, type `bool`

        ## Return
        the searched rexDict, type `{stepPair:{id:rex, , , }, , ,  }`

        '''
    
        printf("#[getRexTo           ] get rex to", toMol.getSymbols())
        allFrames = {}
        for steps, frame in rexDict.items():
            thisFrame = {}
            for idx, elemRex in frame.items():
                if elemRex.isRex() and elemRex.isContainP(toMol):
                    thisFrame[idx] = elemRex
                else:
                    pass
            allFrames[steps] = thisFrame

        if show:
            for frame in allFrames.values():
                for elemRex in frame.values():
                    printf(elemRex)
        else:
            pass
        return allFrames



    @staticmethod
    def sigFraMovement(stepair, frame, targetMols):
        OH, H3O2, H5O3 = targetMols
        vehSol, groSol, allSol = np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])
        listVehSol, listGroSol, listAllSol= [], [], []

        vehSrf, groSrf, allSrf = np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])
        listVehSrf, listGroSrf, listAllSrf = [], [], []
        
        totalAll = np.array([0, 0, 0])
        listTotalAll, listUncount  = [], []


        for elemRex in frame.values():
            elemRex: Reaction

            (aimR, aimP) = elemRex.getTargetMols([OH, H3O2, H5O3])

            if len(aimR) == 0 and len(aimP) == 0:
                pass

            elif len(aimR) == 1 and len(aimP) == 1:
                R, P = aimR[0], aimP[0]
                box = R.box
                lx = box[0][1]-box[0][0]
                ly = box[1][1]-box[1][0]
                lz = box[2][1]-box[2][0]

                atmA, atmB = R.getAimOxygen(), P.getAimOxygen()
                vet = atmA.vectorWith_PBC(atmB, lx, ly, lz)
                L = np.array(vet)

                if atmA.id == atmB.id:
                    if R.isOxygenVdw() or P.isOxygenVdw():
                        listVehSrf.append(L)
                    else:
                        listVehSol.append(L)

                else:
                    if R.isOxygenVdw() or P.isOxygenVdw():
                        listGroSrf.append(L)
                    else:
                        listGroSol.append(L)

            elif len(aimR) == 2 and len(aimP) == 2:
                pass
            else:
                listUncount.append(1)

            listAllSol = listVehSol + listGroSol
            listAllSrf = listVehSrf + listGroSrf
            listTotalAll = listAllSol + listAllSrf

        nVehSol, nGroSol, nAllSol = len(listVehSol), len(listGroSol), len(listAllSol), 
        nVehSrf, nGroSrf, nAllSrf = len(listVehSrf), len(listGroSrf), len(listAllSrf)
        nTotalAll, nUncount = len(listTotalAll), len(listUncount)


        if nVehSol == 0: 
            pass
        else:
            vehSol = sum(listVehSol)/nVehSol

        if nGroSol == 0: 
            pass
        else:
            groSol = sum(listGroSol)/nGroSol

        if nAllSol == 0: 
            pass
        else:
            allSol = sum(listAllSol)/nAllSol


        if nVehSrf == 0: 
            pass
        else:
            vehSrf = sum(listVehSrf)/nVehSrf

        if nGroSrf == 0: 
            pass
        else:
            groSrf = sum(listGroSrf)/nGroSrf

        if nAllSrf == 0: 
            pass
        else:
            allSrf = sum(listAllSrf)/nAllSrf


        if nTotalAll == 0: 
            pass
        else:
            totalAll = sum(listTotalAll)/nTotalAll

        
        others =   (vehSol, groSol, allSol, nVehSol, nGroSol, nAllSol,
                    vehSrf, groSrf, allSrf, nVehSrf, nGroSrf, nAllSrf,
                    totalAll, nTotalAll, nUncount)

        return (stepair, others)


    @staticmethod
    def getDriftLength(rexDict, targetMols, nCores):
        printf("#[getDriftLength     ] begin to: get drift length by", nCores, "cores")
        
        pool = multiprocessing.Pool(nCores)
        asyncResultList = []

        for stepair, frame in rexDict.items():
            asyncResultList.append(
                pool.apply_async(
                    Analyzer.sigFraMovement,
                    args=(stepair, frame, targetMols),
                )
            )

        pool.close()
        pool.join()

        tempDict = {}
        for result in asyncResultList:
            (stepair, others) = result.get()
            tempDict[stepair] = others

        del asyncResultList, pool

        xVehSol,  yVehSol,  zVehSol  = [0, ], [0, ], [0, ]
        xGroSol,  yGroSol,  zGroSol  = [0, ], [0, ], [0, ]
        xAllSol,  yAllSol,  zAllSol  = [0, ], [0, ], [0, ]
        nVehSols, nGroSols, nAllSols = [   ], [   ], [   ]

        xVehSrf,  yVehSrf,  zVehSrf  = [0, ], [0, ], [0, ]
        xGroSrf,  yGroSrf,  zGroSrf  = [0, ], [0, ], [0, ]
        xAllSrf,  yAllSrf,  zAllSrf  = [0, ], [0, ], [0, ]
        nVehSrfs, nGroSrfs, nAllSrfs = [   ], [   ], [   ]

        xTotalAll, yTotalAll,   zTotalAll  = [0, ], [0, ], [0, ]
        timeList,  itnTotalAll, itnUncount = [0, ], [   ], [   ]

        sortedStepairs = sorted(tempDict.keys())
        for stepair in sortedStepairs:
            (vehSol, groSol, allSol, nVehSol, nGroSol, nAllSol,
            vehSrf, groSrf, allSrf, nVehSrf, nGroSrf, nAllSrf,
            totalAll, nTotalAll, nUncount) = tempDict[stepair]

            xVehSol.append(xVehSol[-1] + vehSol[0])
            yVehSol.append(yVehSol[-1] + vehSol[1])
            zVehSol.append(zVehSol[-1] + vehSol[2])

            xGroSol.append(xGroSol[-1] + groSol[0])
            yGroSol.append(yGroSol[-1] + groSol[1])
            zGroSol.append(zGroSol[-1] + groSol[2])

            xAllSol.append(xAllSol[-1] + allSol[0])
            yAllSol.append(yAllSol[-1] + allSol[1])
            zAllSol.append(zAllSol[-1] + allSol[2])

            nVehSols.append(nVehSol)
            nGroSols.append(nGroSol)
            nAllSols.append(nAllSol)


            xVehSrf.append(xVehSrf[-1] + vehSrf[0])
            yVehSrf.append(yVehSrf[-1] + vehSrf[1])
            zVehSrf.append(zVehSrf[-1] + vehSrf[2])

            xGroSrf.append(xGroSrf[-1] + groSrf[0])
            yGroSrf.append(yGroSrf[-1] + groSrf[1])
            zGroSrf.append(zGroSrf[-1] + groSrf[2])

            xAllSrf.append(xAllSrf[-1] + allSrf[0])
            yAllSrf.append(yAllSrf[-1] + allSrf[1])
            zAllSrf.append(zAllSrf[-1] + allSrf[2])

            nVehSrfs.append(nVehSrf)
            nGroSrfs.append(nGroSrf)
            nAllSrfs.append(nAllSrf)


            xTotalAll.append(xTotalAll[-1] + totalAll[0])
            yTotalAll.append(yTotalAll[-1] + totalAll[1])
            zTotalAll.append(zTotalAll[-1] + totalAll[2])


            step = (stepair[0] + stepair[1])/2
            timeList.append(step*TIMESTEP)
            itnTotalAll.append(nTotalAll)
            itnUncount.append(nUncount)
            
        timeList[0] = 2*timeList[1] - timeList[2]

        drawFigMulti(
            xList=timeList,
            yLists=(xGroSol, xVehSol, xAllSol, xGroSrf, xVehSrf, xAllSrf, xTotalAll),
            labels=(
                "The X-MassCenter of Target Species",
                "Time / ps",
                "MassCenter / Angstrom"),
            legends=('Gro sol', 'Veh sol', 'avg: G&V sol', 'Gro srf', 'Veh srf', 'avg: G&V srf', 'avg: all'),
            nameForSave="x-MassCenter-GroVehSrf-of-Target-Species",
            type='line'
        )

        drawFigMulti(
            xList=timeList,
            yLists=(yGroSol, yVehSol, yAllSol, yGroSrf, yVehSrf, yAllSrf, yTotalAll),
            labels=(
                "The Y-MassCenter of Target Species",
                "Time / ps",
                "MassCenter / Angstrom"),
            legends=('Gro sol', 'Veh sol', 'avg: G&V sol', 'Gro srf', 'Veh srf', 'avg: G&V srf', 'avg: all'),
            nameForSave="y-MassCenter-GroVehSrf-of-Target-Species",
            type='line'
        )

        drawFigMulti(
            xList=timeList,
            yLists=(zGroSol, zVehSol, zAllSol, zGroSrf, zVehSrf, zAllSrf, zTotalAll),
            labels=(
                "The Z-MassCenter of Target Species",
                "Time / ps",
                "MassCenter / Angstrom"),
            legends=('Gro sol', 'Veh sol', 'avg: G&V sol', 'Gro srf', 'Veh srf', 'avg: G&V srf', 'avg: all'),
            nameForSave="z-MassCenter-GroVehSrf-of-Target-Species",
            type='line'
        )


        drawFigMulti(
            xList=timeList[1:],
            yLists=(nGroSols, nVehSols, nAllSols,  
                    nGroSrfs, nVehSrfs, nAllSrfs, 
                    itnTotalAll, itnUncount),
            labels=(
                'Number of molecules',
                "Time / ps",
                'Number of molecules'),
            legends=('Gro sol', 'Veh sol', 'sum: G&V sol',
                     'Gro srf', 'Veh srf', 'sum: G&V srf',
                     'sum: all', 'unCount'),
            nameForSave='Number-of-used-molecules-GroVeh',
            type='line'
        )

        listToCSV(
            nameTuple=('time-1', 
                       'xGro sol', 'xVeh sol', 'avg: x G&V sol', 'xGro srf', 'xVeh srf', 'avg: x G&V srf', 'avg: x all', 
                       'yGro sol', 'yVeh sol', 'avg: y G&V sol', 'yGro srf', 'yVeh srf', 'avg: y G&V srf', 'avg: y all',
                       'zGro sol', 'zVeh sol', 'avg: z G&V sol', 'zGro srf', 'zVeh srf', 'avg: z G&V srf', 'avg: z all',
                        'time-2',
                        'nGro sol', 'nVeh sol', 'nSum: G&V sol',
                        'nGro srf', 'nVeh srf', 'nSum: G&V srf',
                        'nSum: all', 'nUnCount',
                       ),
            listTuple=( timeList,
                        xGroSol, xVehSol, xAllSol, xGroSrf, xVehSrf, xAllSrf, xTotalAll,
                        yGroSol, yVehSol, yAllSol, yGroSrf, yVehSrf, yAllSrf, yTotalAll,
                        zGroSol, zVehSol, zAllSol, zGroSrf, zVehSrf, zAllSrf, zTotalAll,
                        timeList[1:],
                        nGroSols, nVehSols, nAllSols,
                        nGroSrfs, nVehSrfs, nAllSrfs, 
                        itnTotalAll, itnUncount
                       ),
            csvName="getMassCenter_VehGroSrf_MP.csv",
            )

        printf("#[getDriftLength     ] finished: get MassCenter")


