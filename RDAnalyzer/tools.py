import os, re, shutil, copy, time, pickle, multiprocessing, numpy
import pandas as pd
import matplotlib
#matplotlib.use('Agg') # x display
import matplotlib.pyplot as plt

from itertools import combinations

from ase import Atoms
from ase.visualize import view

from RDAnalyzer.set import FIGURE_NUMBER, ATOMTYPE, TIMESTEP

##############################################################
def vectorBetween_PBC(lA:tuple, lB:tuple, lBox:tuple):
    '''
    ## Function
    get vector of A->B

    ## Parameters
    `lA`: (xa, ya, za)
    `lB`: (xb, yb, zb)
    `lBox`: (xbox, ybox, zbox)
    '''
        
    lx, ly, lz = lBox
    vx, vy, vz = tuple([b-a for a, b in zip(lA, lB)])
    nx = move(vx, lx)
    ny = move(vy, ly)
    nz = move(vz, lz)
    x, y, z = (lx*nx+vx, ly*ny+vy, lz*nz+vz)
    return (x, y, z)
    
def lengthBetween_PBC(lA:tuple, lB:tuple, lBox:tuple):
    '''
    `lA`: (xa, ya, za)
    `lB`: (xb, yb, zb)
    `lBox`: (xbox, ybox, zbox)
    '''
        
    lx, ly, lz = lBox
    vx, vy, vz = tuple([a-b for a, b in zip(lA, lB)])
    nx = move(vx, lx)
    ny = move(vy, ly)
    nz = move(vz, lz)
    x, y, z = (lx*nx+vx, ly*ny+vy, lz*nz+vz)

    return (x**2 + y**2 + z**2)**0.5


class timer():
    '''initialize the `timer` '''
    def __init__(self) -> None:
        self.begining = time.time()

        self.start = time.time()
        self.end = None

        self.fileList = [name for name in os.listdir(os.getcwd())]

    def timePoint(self, txt="", finish = False):
        '''print `time` now, and the `interval` between the ahead time point'''
        timeString = time.strftime("[%Y-%m-%d %H:%M:%S ]", time.localtime())
        printf(timeString, end=" ")

        self.end = time.time()
        interval = self.end - self.start
        if interval > 600:
            printf("Interval: {:.5f} min".format(interval/60),",",txt, end=" \n")
        else:
            printf("Interval: {:.5f} sec".format(interval),",",txt, end=" \n")

        fileNow = [name for name in os.listdir(os.getcwd())]
        fileDelete = set(self.fileList) - set(fileNow)
        fileAdded = set(fileNow) - set(self.fileList)
        printf(timeString, " deleted:", sorted(fileDelete))
        printf(timeString, "   added:", sorted(fileAdded))
        self.fileList = fileNow

        if finish:
            printf(timeString, end=" ")
            interval = self.end - self.begining
            if interval > 600:
                printf(" Totally: {:.5f} min".format(interval/60),",",txt, end=" \n")
            else:
                printf(" Totally: {:.5f} sec".format(interval),",",txt, end=" \n")
        else: pass
        self.start = self.end
        

def timeNow():
    '''get the time at this moment'''
    printf(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


def info(a):
    '''get information of variable `a`'''
    print('Type', type(a), '| Value', a, '| ID', id(a))


def kill():
    '''kill the code'''
    raise Exception("killed")


def printf_only(*txt, sep=' ', end='\n', fileName="all.log"):
    '''write the `txt` to the file `fileName` '''
    with open(fileName, 'a') as f:
        print(*txt, sep=sep, end=end, file=f)
    f.close()


def printf(*txt, sep=' ', end='\n', fileName="all.log"):
    '''write the `txt` to the file `fileName` '''
    print(*txt, sep=sep, end=end)

    with open(fileName, 'a') as f:
        print(*txt, sep=sep, end=end, file=f)
    f.close()

############################################################
def move(v, l, n=0) -> int:
    ''' the length need move / boundary length

    Parameter
    ---
    `v`: the length
    `l`: boundary length
    
    Return
    ---
    number need for move
    
    '''
    if v >= l/2.0:
        v = v-l
        n = n-1
        return move(v, l, n)
    elif v < -l/2.0:
        v = v+l
        n = n+1
        return move(v, l, n)
    else:
        return n

def moveToBox(x, y, z, box: list):
    lx = box[1]-box[0]
    ly = box[3]-box[2]
    lz = box[5]-box[4]
    if x < box[0]:
        x = x+lx
        x, y, z = moveToBox(x, y, z, box)
    elif x > box[1]:
        x = x-lx
        x, y, z = moveToBox(x, y, z, box)
    else:
        pass
    if y < box[2]:
        y = y+ly
        x, y, z = moveToBox(x, y, z, box)
    elif y > box[3]:
        y = y-ly
        x, y, z = moveToBox(x, y, z, box)
    else:
        pass
    if z < box[4]:
        z = z+lz
        x, y, z = moveToBox(x, y, z, box)
    elif z > box[5]:
        z = z-lz
        x, y, z = moveToBox(x, y, z, box)
    else:
        pass
    return (x, y, z)


def progressBar(i, N, txt=""):
    '''progress bar \n
    `i`: is the sequence number, begining from 1;\n
    `N`: is the total loop number;\n
    `txt`: information ahead the word "Progress"\n
    During the process, there should not have `print()` function  
     '''
    n = int(i/N*100)
    print("\r", end="")
    print(txt+" Progress: {}%: ".format(n), "▋" * (n//3), end="")


def dump(obj, file: str):
    '''dump `obj` to file `file`, and return `None`'''
    f = open(file, 'wb')
    pickle.dump(obj=obj, file=f)
    f.close()


def load(file: str):
    '''load and return the obj saved in file `file`'''
    f = open(file, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj

def smooth(x, y, size=5):
    '''
    ## Function
    smooth y list

    ## Parameter
    `x`: x list
    `y`: y list
    `size`: window size, bigger size means more smooth

    ## Return
    smoothed x and y list
    '''

    interval = x[1] - x[0]
    move = size/2*interval
    x = numpy.array(x)
    y = numpy.array(y)
    window = numpy.ones(int(size)) / float(size)
    yAvg = numpy.convolve(y, window, 'full')
    
    return (x-move, yAvg[0:len(x)])


def listToCSV(nameTuple, listTuple, csvName="new.csv"):
    '''
    ## Function
    save the list in `listTuple` to `csvName`, with the column name is `nameTuple`
    
    ## Parameter
    `nameTuple`: name's tuple
    `listTuple`: list's tuple
    `csvName`: csv's name, with '.csv' in it

    ## Return
    `None`
    '''
    maxLength = max([len(i) for i in listTuple])
    for thisList in listTuple:
        for i in range(maxLength-len(thisList)):
            thisList.append("-")
    dataDict = {}
    for (name, lists) in zip(nameTuple, listTuple):
        dataDict[name] = lists
    newName = naming(csvName)
    pd.DataFrame(dataDict).to_csv(newName, header=True, index=False)
    printf("#[listToCSV          ] store as:", newName)


def drawFigMulti(xList: list,
                 yLists: tuple,
                 legends: tuple = None,
                 labels: tuple = ("title", "x", "y"),
                 nameForSave=None,
                 type='line'):
    '''
    Function
    ---
    draw figure according to `list`s

    Parameters
    ---
    `xlist`: data of x axis, type `list`
    `ylist`: data of y axis, type `tuple(list, list, , ,)`
    `legends`: legend of xLists, type `tuple(str, str, str, , ,)`
    `labels`: label of figure/X/Y, type `tuple(str, str, str)`
    `nameForSave`: the file name for saving, type `str`, without `.png`
    `type`: `line`, `scatter`

    Return
    ---
    `None`
    '''

    title, xlabel, ylabel = str(labels[0]), str(labels[1]), str(labels[2])
    figNumb = FIGURE_NUMBER()
    plt.figure(figNumb)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if type == 'line':
        for yList in yLists:
            plt.plot(xList, yList, marker='o', markersize=3)
    elif type == 'scatter':
        for yList in yLists:
            plt.scatter(xList, yList)
    else:
        pass

    if legends:
        plt.legend(legends)
    else: 
        pass

    if nameForSave:
        name = "fig-"+nameForSave+".png"
        newName = naming(name)
        plt.savefig(newName)
        printf("#[drawFigMulti       ] store as:", newName)
    else:
        plt.show()

    plt.cla()
    plt.close()


def drawFigSig(
        xList: list,
        yList: list,
        labels: tuple,
        nameForSave:str,
        type:str = 'line'
        ):
    '''
    ## Function
    draw figure according to the `list`

    ## Parameters
    `xlist`: data of x axis, type `list`
    `ylist`: data of y axis, type `list`
    `labels`: label of figure/X/Y, type `tuple(str, str, str)`
    `nameForSave`: the file name for saving, type `str`, without `.png`
    `type`: `line`, `scatter`

    Return
    ---
    `None`
    '''
        
    drawFigMulti(
        xList=xList,
        yLists=(yList,),
        labels=labels,
        nameForSave=nameForSave,
        type=type, 
        )




def moveFiles(toDir: str, keyStr: tuple = (".png", ".csv"), fileNames:list = []):
    '''
    ## Function
    move files in dir now to the `toDir`

    ## Parameters
    `toDir`: the aim dir, not exist now
    `keyStr`: key string that files contain

    ## Return
    `None`
    '''
    path = os.getcwd()
    newPath = path + '/' + toDir

    if os.path.exists(newPath):
        printf("#[moveFiles          ] dir", toDir, 'exist, some files may be covered')
    else:
        os.mkdir(newPath)

    movedFiles = []
    for files in os.listdir():
        for nameStr in keyStr:
            if nameStr in files:
                movedFiles.append(files)
                shutil.move(files, newPath+'/' + files)

            else: pass

    for files in os.listdir():
        if files in fileNames:
            movedFiles.append(files)
            shutil.move(files, newPath+'/' + files)
        else: pass

    printf("#[moveFiles          ] moveFile:", sorted(movedFiles))



def naming(name):
    '''renaming the `name` to avoid repeat'''
    file_name = name
    while os.path.isfile(file_name):
        pattern = '(\d+)\)\.'
        if re.search(pattern, file_name) is None:
            file_name = file_name.replace('.', '(0).')
        else:
            current_number = int(re.findall(pattern, file_name)[-1])
            new_number = current_number + 1
            file_name = file_name.replace(
                f'({current_number}).', f'({new_number}).')

    return file_name

def inverseDict(theDict: dict) -> dict:
    newDict = {}
    for key, value in theDict.items():
        newDict[value] = key

    if len(theDict) == len(newDict):
        return newDict
    else:
        raise Exception("length wrong")
    

def removeCom(listA:list, listB:list):
    '''remove common element'''
    common = set(listA) & set(listB)
    for i in common:
        listA.remove(i)
        listB.remove(i)

    if common:
        return removeCom(listA, listB)
    else:
        return None


def mprun(func, inputDict: dict, ncores: int):
    '''
    ## Function:
    mp run functions

    ## Parameters
    `func`: the aim function
    `inputDict`: all the the input data, type `{name: [p1, , , ], , , }`
    `ncores`: number of  cores

    ## Ruturn
    type `{name: result-of-func, , , }`
    '''

    pool = multiprocessing.Pool(ncores)
    asyResult = {}
    printf("#[mpRun] begin build pool")
    for name, input in inputDict.items():
        asyResult[name] = pool.apply_async(func, args=tuple(input))

    printf("#[mpRun] pool built")
    pool.close()
    printf("#[mpRun] pool closed")
    pool.join()
    printf("#[mpRun] pool jointed")

    allResult = {}
    for name, result in asyResult.items():
        allResult[name] = result.get()
        pass

    return allResult


def jointFunc(funcList:list, inputList:list)->list:
    resultList = []
    for i in len(funcList):
        resultList.append(
            funcList[i](*inputList[i])
            )
    
    return resultList
































    
##########################################
def show(
    molFileName=None,
    stepAndID=None,
    rexNet=None,
    rex = None
):
    '''
    ## Function
    show molecule, frames, or reaction networks

    ## Parameters
    `molFileName`: searched molecule dict or full molecule dict, `str`
    `stepAndID`: show the molecule in the step and with the ID, `(step, id)`
    `rexNet`: the full reaction networks

    ## Return
    `None`
    
    ## Examples
    1,  `show(molFileName='water.pkl')`: to see the searched water molecules in each steps
    2,  `show(molFileName='mol.1-0-5000.pkl', stepAndID=(2000, 5))`: show the molecule with 
        it's ID is 5 at 2000 step from the file "mol.1-0-5000.pkl"
    3,  `show(stepAndID=(2000,2), rexNet=t)`: show the molecule with it's ID is 2 in 2000 step
        according the file t._molFileName
    '''

    def __viewAseAtomsList(atomsList):
        '''view ASE atoms list'''
        view(atomsList)

    def __molToAseAtoms(mol):
        '''transform `molecule` to ASE atoms'''
        typeList, positionList = [], []
        for node in mol.getNodes():
            typeList.append(ATOMTYPE[node.type])
            positionList.append(node.position)
        molAtoms = Atoms(symbols=typeList, positions=positionList)
        del typeList, positionList
        return molAtoms

    def __molDirtToAseAtomsList(fileName) -> list:
        '''pick up file `fileName` to obtain `molDict`, then transform it to ASE atoms List'''
        molsAseAtomsList = []
        with open(fileName, 'rb') as f:
            molDirt = pickle.load(f)
            for frame in molDirt.values():
                atomsThisFrame = Atoms()
                if isinstance(frame, list):
                    for mol in frame:
                        atomsThisFrame = atomsThisFrame + \
                            __molToAseAtoms(mol=mol)
                    molsAseAtomsList.append(atomsThisFrame)
                elif isinstance(frame, dict):
                    for mol in frame.values():
                        atomsThisFrame = atomsThisFrame + \
                            __molToAseAtoms(mol=mol)
                    molsAseAtomsList.append(atomsThisFrame)
        f.close()
        return molsAseAtomsList
    
    def __rex2AseList(rex):
        atomsR, atomsP = Atoms(), Atoms()
        for mol in rex.R:
            atomsR = atomsR + __molToAseAtoms(mol=mol)
        for mol in rex.P:
            atomsP = atomsP + __molToAseAtoms(mol=mol)
        
        return atomsR, atomsP


    def __viewSearched(fileName: str):
        if os.path.exists(fileName):
            print("#[__viewSearched] Try to show", fileName, '\n')
            molsAseAtomsList = __molDirtToAseAtomsList(fileName)
            __viewAseAtomsList(molsAseAtomsList)
        else:
            print("#[__viewSearched] File", fileName, "can not be found")

    def __viewSingleMol(step, id, molFileName):
        if os.path.exists(molFileName):
            print("#[__viewSingleMol] Try to show", molFileName, '\n')
        else:
            print("#[__viewSingleMol] File", molFileName, 'can not be found')
            return None
        with open(molFileName, 'rb') as f:
            allFramesMols = pickle.load(f)
            mol = allFramesMols[step][id]
        f.close()
        aseMol = __molToAseAtoms(mol)
        __viewAseAtomsList(aseMol)

    def __drawNetWork(rexNet):
        # initizlize list direct molAllFrame
        molAllFrame = {}
        molList = rexNet.getAllNodes()
        stepSet = set([mol.step for mol in molList])
        for step in stepSet:
            molAllFrame[step] = []
        # write mols to the list direct molAllFrame
        for mol in molList:
            molAllFrame[mol.step].append(mol)
        # trans mols to ase atoms
        molsAseAtomsList = []
        stepList = sorted(stepSet)
        for step in stepList:
            thisFrame = molAllFrame[step]
            atomsThisFrame = Atoms()
            for mols in thisFrame:
                atomsThisFrame = atomsThisFrame + __molToAseAtoms(mols)
            molsAseAtomsList.append(atomsThisFrame)
        # show
        __viewAseAtomsList(molsAseAtomsList)

    #####################################
    if molFileName and not stepAndID and not rexNet:
        __viewSearched(molFileName)
    elif molFileName and stepAndID and not rexNet:
        step, molId = stepAndID
        __viewSingleMol(step, molId, molFileName)
    elif not molFileName and stepAndID and rexNet:
        step, molId = stepAndID
        __viewSingleMol(step, molId, rexNet.molFileName())
    elif not molFileName and not stepAndID and rexNet:
        __drawNetWork(rexNet)
    elif rex:
        atomsR, atomsP = __rex2AseList(rex)
        __viewAseAtomsList(atomsList=atomsR)
        __viewAseAtomsList(atomsList=atomsP)

    else:
        print("#[__viewSingleMol] Parameters set wrong")
