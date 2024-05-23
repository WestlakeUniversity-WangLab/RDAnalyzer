import copy
import math
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.isomorphism as iso

from ase import Atoms, build
from ase.visualize import view
from ase.collections import g2



from RDAnalyzer.tools import move, printf, inverseDict
from RDAnalyzer.set import DEFAULT_BONDER_ORDER, ATOMTYPE, \
    ATOMTYPE_INVERSE, BOND_ORDER


########################################################## Atom
class Atom:
    def __init__(self,
                 atomID: int,
                 atomType: int,
                 position: tuple,
                 charge: float,
                 bonds: list,
                 steps=None) -> None:
        self.id = atomID
        self.type = atomType
        self.position = position
        self.charge = charge
        self.bonds = bonds
        self.vdw = []
        self.steps = steps

    def __eq__(self, other) -> bool:
        if isinstance(other, Atom):
            return hash(self) == hash(other)
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.id, self.steps))

    def __str__(self) -> str:
        thisID = "{:>5}".format(self.id)
        charge = "{:>6.3f}".format(self.charge)
        position = "({:>7.3f},{:>7.3f},{:>7.3f})".format(
            self.position[0], self.position[1], self.position[2])

        allStr = '| '
        allStr = allStr + "ID:" + thisID + " | "
        allStr = allStr + "Type: " + ATOMTYPE[self.type] + ' | '
        allStr = allStr + "Steps: " + str(self.steps) + " | "
        allStr = allStr + "Charge: " + charge + " | "
        allStr = allStr + "Position: " + position + " | "
        allStr = allStr + "Bonds: " + str(self.bonds) + " | "

        if len(self.vdw) != 0:
            allStr = allStr + "Vdw: " + str(self.vdw) + " | "
        else:
            pass

        return allStr

    def getRelatedAtomsID(self):
        atomsID = [bond[0] for bond in self.bonds]
        return atomsID


    def vectorWith(self, atom):
        '''distance vector: `self`->`atom`'''
        return tuple([b-a for a, b in zip(self.position, atom.position)])

    def vectorWith_PBC(self, atom, lx, ly, lz):
        '''get the vector of `self`->`atom`'''
        vx, vy, vz = tuple([b-a for a, b in zip(self.position, atom.position)])
        nx = move(vx, lx)
        ny = move(vy, ly)
        nz = move(vz, lz)
        x, y, z = (lx*nx+vx, ly*ny+vy, lz*nz+vz)
        return (x, y, z)

    def distanceWith(self, atom):
        '''get the direct distance with other `atom` without consider the PBC'''
        vect = self.vectorWith(atom)
        return (vect[0]**2 + vect[1]**2 + vect[2]**2)**0.5

    def distanceWith_PBC(self, atom, lx, ly, lz):
        '''
        Functions:
        ---
        get the distance with other `atom` with consider the PBC
        
        Parameters:
        ---
        `atom`: other atom
        `lx`: length of x boundary
        `ly`: length of y boundary
        `lz`: length of z boundary

        Return:
        ---
        the distance with other `atom`

        '''
        vx, vy, vz = tuple([b-a for a, b in zip(self.position, atom.position)])
        nx = move(vx, lx)
        ny = move(vy, ly)
        nz = move(vz, lz)
        x, y, z = (lx*nx+vx, ly*ny+vy, lz*nz+vz)
        return (x**2 + y**2 + z**2)**0.5
    
    def adjust(self, atom, lx, ly, lz):
        '''
        Functions:
        ---
        adjust other `atom`'s position with consideration of the PBC

        Parameters:
        ---
        `atom`: other atom
        `lx`: length of x boundary
        `ly`: length of y boundary
        `lz`: length of z boundary

        Return:
        ---
        adjusted `atom`

        '''
        thisAtom = copy.deepcopy(atom)
        px, py, pz = thisAtom.position
        vx, vy, vz = tuple(
            [b-a for a, b in zip(self.position, thisAtom.position)]
            )
        nx = move(vx, lx)
        ny = move(vy, ly)
        nz = move(vz, lz)
        x, y, z = (lx*nx+px, ly*ny+py, lz*nz+pz)
        thisAtom.position = (x, y, z)
        return thisAtom

    def angle(self, atomA, atomC):
        x1, y1, z1 = tuple(
            [b-a for a, b in zip(self.position, atomA.position)])
        x2, y2, z2 = tuple(
            [b-a for a, b in zip(self.position, atomC.position)])
        cos = (x1*x2 + y1*y2 + z1*z2) / (math.sqrt(x1**2 +
                                                   y1**2 + z1**2)*math.sqrt(x2**2 + y2**2 + z2**2))
        return math.degrees(math.acos(cos))

    def bondsNumber(self):
        '''get number of bonds'''
        return len(self.bonds)

    def sortedBond(self):
        '''sort all the bonds from smaller BO to bigger BO'''
        return sorted(self.bonds, key=lambda bond: bond[1])

    



########################################################## Molecule
class Molecule():

    def __init__(self):
        self.topology = None
        self.step = None
        self.id = None
        self.box = None

        self.hashType = None
        self.hashID = None

        self.aseAtoms = None


    def initialize(self, idList, allAtomThisFra: dict):
        atomsList = [allAtomThisFra[atomID] for atomID in idList]
        topology = nx.Graph()
        for atom in atomsList:
            topology.add_node(atom, id=atom.id, type=atom.type)
            for bond in atom.bonds:
                thisAtom = allAtomThisFra[bond[0]]
                topology.add_edge(atom, thisAtom, BO=bond[1])
        self.topology = topology

    def __eq__(self, mol) -> bool:
        '''id equal'''
        return hash(self) == hash(mol)

    def __hash__(self) -> int:
        '''id hash'''
        hashStr = nx.weisfeiler_lehman_graph_hash(
            G=self.topology, node_attr="type", iterations=5, digest_size=32
            )
        return hash((self.step, self.id, hashStr))

    def getHash(self, reGet=False) -> str:
        '''type hash'''
        if not reGet and self.hashType:
            return self.hashType  
        else:
            self.hashType = nx.weisfeiler_lehman_graph_hash(G=self.topology, node_attr="type", iterations=5, digest_size=16)
            return self.hashType
        
    def equal(self, mol) -> bool:
        '''type equal'''
        return self.getHash() == mol.getHash()

    def __str__(self) -> str:
        molStr = ''
        for node in self.topology.nodes:
            molStr = molStr + str(node) + '\n'
        return molStr[0:-1]

    #get
    def getNodes(self):
        '''get a `list` contains all the nodes (all the `atom`s)'''
        return list(self.topology.nodes)

    def _getNodesID(self):
        '''get a list contains all the node's ID (all the atom's ID)'''
        idList = []
        for node in self.getNodes():
            idList.append(node.id)
        return idList

    def _getNodeWithID(self, id):
        '''get the node according its ID (the atom's ID)'''
        if id not in self._getNodesID():
            return False
        else:
            for node in self.getNodes():
                if node.id == id:
                    return node
                else:
                    pass
        printf("#NOTE can not find atom with ID ",
              id, "in ", self._getNodesID())

    def _getNodesWithType(self, type: str):
        '''get nodes which type is `type`'''
        coreAtomList = []
        for node in self.getNodes():
            if ATOMTYPE[node.type] == type:
                coreAtomList.append(node)
            else:
                pass
        return coreAtomList

    def getNodesAroundNode(self, coreAtom, length):
        aimList = []
        for node in self.getNodes():
            if nx.shortest_path_length(self.topology, source=coreAtom, target=node) <= length:
                aimList.append(node)
            else:
                pass
        return aimList

    def getNodesAround(self, coreAtomType: str, length: int):
        '''get nodes that around atom `coreAtomType` and closer than `length`'''
        coreAtomList = self._getNodesWithType(coreAtomType)
        aimList = []
        for node in self.getNodes():
            for atom in coreAtomList:
                if nx.shortest_path_length(self.topology, source=atom, target=node) <= length:
                    aimList.append(node)
        return aimList

    def getAdjNodes(self, atom):
        return list(self.topology.neighbors(atom))

    def getSubMolNodes(self, mol, mapDictList:list = []):
        '''
        ## Function
        get all subGraph's node list that iso to the `mol`

        ## Parameter
        `mol`: the `mol` to be matched, type `mol`

        ## Return
        `[{atom:atom, , , }, , , ]`
        '''
        oldDictList = copy.deepcopy(mapDictList)
        nm = iso.numerical_node_match(attr="type", default=1)
        GM = iso.GraphMatcher(self.topology, mol.topology, node_match=nm)
        if GM.subgraph_is_isomorphic():
            newMol = copy.deepcopy(self)
            newMol.unFreeze()
            for atom in GM.mapping.keys():
                newMol._removeNode(atom=atom)
            
            oldDictList.append( inverseDict(theDict= GM.mapping) )
            return newMol.getSubMolNodes(mol=mol, mapDictList=oldDictList)
        
        else: 
            return oldDictList
        

    def getSubMol(self, mol, mapDictList:list = []):
        '''
        ## Function
        get all subGraph's node list that iso to the `mol`

        ## Parameter
        `mol`: the `mol` to be matched, type `mol`

        ## Return
        `[{atom:atom, , , }, , , ]`
        '''
        oldDictList = copy.deepcopy(mapDictList)
        nm = iso.numerical_node_match(attr="type", default=1)
        GM = iso.GraphMatcher(self.topology, mol.topology, node_match=nm)
        if GM.subgraph_is_isomorphic():
            newMol = copy.deepcopy(self)
            newMol.unFreeze()
            for atom in GM.mapping.keys():
                newMol._removeNode(atom=atom)
            
            oldDictList.append( inverseDict(theDict= GM.mapping) )
            return newMol.getSubMol(mol=mol, mapDictList=oldDictList)
        
        else: 
            subMolList = []
            
            for thisDict in mapDictList:
                frame = {}
                idList = []
                for node in thisDict.values():
                    frame[node.id] = node
                    idList.append(node.id)

                for atom in frame.values():
                    bonds = []
                    for b in atom.bonds:
                        if b[0] in idList:
                            bonds.append(b)
                        else: pass

                    atom.bonds = bonds
                
                thisMol = Molecule()
                thisMol.initialize(idList, frame)
                subMolList.append(thisMol)
                
            return subMolList

    def getEdges(self):
        return list(self.topology.edges)

    def getEdgesDFS(self, source=None):
        return nx.dfs_edges(self.topology, source=source)

    def getEdgesBFS(self, source=None):
        if not source:
            source = self.getNodes()[0]
        return nx.bfs_edges(self.topology, source=source)

    def getIso(self, molList):
        for mol in molList:
            if self.equal(mol):
                return mol
            else:
                pass
        return False

    #freeze
    def isFrozen(self):
        return nx.is_frozen(self.topology)

    def freeze(self):
        self.topology = nx.freeze(self.topology)

    def unFreeze(self):
        self.topology = nx.Graph(self.topology)

    #judge
    def isContain(self, atom: Atom) -> bool:
        return atom in self.getNodes()

    def _isContainID(self, atomID):
        return atomID in self._getNodesID()

    def _isContainIDs(self, idList):
        for id in idList:
            if self._isContainID(id):
                return True
            else:
                pass
        return False

    def _isContainType(self, type:str)->bool:
        for atom in self.getNodes():
            if ATOMTYPE[atom.type] == type:
                return True
            else:
                pass
        return False
    
    def isContainSubMol(self, mol) -> bool:
        nm = iso.numerical_node_match(attr="type", default=1)
        GM = iso.GraphMatcher(self.topology, mol.topology, node_match=nm)
        return GM.subgraph_is_isomorphic()
    

    def isHasComAtomsWith(self, mol):
        selfIdList = [atom.id for atom in self.getNodes()]
        molsIdList = [atom.id for atom in mol.getNodes()]
        commonID = set(selfIdList) & set(molsIdList)
        if len(commonID) == 0:
            return False
        else:
            return True

    def isIdEqual(self, mol) -> bool:
        nm = iso.numerical_node_match(attr="id", default=1)
        return nx.is_isomorphic(self.topology, mol.topology, node_match=nm)

    def isTypeEqual(self, mol) -> bool:
        nm = iso.numerical_node_match(attr="type", default=1)
        return nx.is_isomorphic(self.topology, mol.topology, node_match=nm)

    def isIn(self, molList, judgeBy='type') -> bool:
        if judgeBy == 'type':
            for mol in molList:
                if self.isTypeEqual(mol):
                    return True
                else:
                    pass
            return False
        elif judgeBy == 'id':
            for mol in molList:
                if self.isIdEqual(mol):
                    return True
                else:
                    pass
            return False
        else:
            printf("value of judgeBy is wrong")
    
    def isOxygenVdw(self):
        for atom in self.getNodes():
            if len(atom.vdw) > 0 and ATOMTYPE[atom.type] == 'O':
                return True
            else: pass
        
        return False

    # modify the molecule
    def _removeNode(self, atom:Atom):
        self.topology.remove_node(atom)

        for theAtom in self.topology.nodes:
            newBonds = []
            for bond in theAtom.bonds:
                if bond[0] == atom.id:
                    pass
                else:
                    newBonds.append(bond)
            theAtom.bonds = newBonds
        
    def _removeEdge(self, atom1, atom2):
        self.topology.remove_edge(atom1, atom2)

    def removeAtomWithID(self, atomID):
        if self._isContainID(atomID):
            self._removeNode(self._getNodeWithID(atomID))
        else:
            printf("# without atom with ID ", atomID)

    def addNode(self, atom:Atom):
        '''
        add atom by `atom`
        '''
        g = self.topology
        g.add_node(atom, id=atom.id, type=atom.type)
        for idx, bo in atom.bonds:
            if bo >= DEFAULT_BONDER_ORDER:
                node = self._getNodeWithID(idx)
                g.add_edge(atom, node)
            else:
                pass

    def addNodeType(self, type:str, *toID:int):
        '''
        add atom by `type` 
        '''
        maxID = max(self._getNodesID())
        bonds = [(idx, 1.0) for idx in toID]

        newAtom = Atom(
            atomID=maxID + 1,
            atomType= ATOMTYPE_INVERSE()[type],
            position=(0.0, 0.0, 0.0),
            charge=0.0,
            bonds=bonds,
            steps=1,
            )
        
        self.addNode(atom=newAtom)

    def changeNode(self, id:int, toType:str):
        '''change atom to other atom'''
        theNode = self._getNodeWithID(id=id)
        theNode.type = ATOMTYPE_INVERSE()[toType]
        self.topology.nodes[theNode]['type'] = theNode.type
    

    # ASE
    def toAseAtoms(self):
        '''transform this `molecule` to ASE `atoms`'''
        if self.aseAtoms:
            return self.aseAtoms
        else:
            self.adjustCoord()
            typeList, positionList = [], []
            for node in self.getNodes():
                typeList.append(ATOMTYPE[node.type])
                positionList.append(node.position)
                box = [self.box[0][1] - self.box[0][0], self.box[1][1] - self.box[1][0], self.box[2][1] - self.box[2][0]]
            self.aseAtoms = Atoms(symbols=typeList, positions=positionList, cell=box, pbc=True)
            return self.aseAtoms

    def getSymbols(self):
        '''get the symbol of the `molecule` by ASE'''
        molAtoms = self.toAseAtoms()
        return molAtoms.get_chemical_formula(mode='hill')

    def getMassCenter(self):
        '''get mass center by ASE'''
        molAtoms = self.toAseAtoms()
        return tuple(molAtoms.get_center_of_mass())

    def showGraph(self):
        labels = {}
        for node in self.getNodes():
            labels[node] = ATOMTYPE[node.type] + '\n' + str(node.id)
        
        length = max(0.5*len(self.getNodes()), 3)
        fig, ax = plt.subplots(figsize=(length, length))
        nx.draw_networkx(
            self.topology, 
            pos=nx.spring_layout(self.topology),
            node_size = 1000,
            labels=labels,
            font_size=15,
            )
        plt.show()
        plt.close()

    # distance and adjust
    def vectorWith(self, mol):
        '''
        get vector between `self` and another `mol`
        '''
        molA = copy.deepcopy(self)
        molB = copy.deepcopy(mol)

        box = molA.box
        lx = box[0][1] - box[0][0]
        ly = box[1][1] - box[1][0]
        lz = box[2][1] - box[2][0]
        mcA, mcB = molA.getMassCenter(), molB.getMassCenter()
        vx = mcA[0] - mcB[0]
        vy = mcA[1] - mcB[1]
        vz = mcA[2] - mcB[2]

        nx = move(vx, lx)
        ny = move(vy, ly)
        nz = move(vz, lz)
        return (lx*nx+vx, ly*ny+vy, lz*nz+vz)

    def distanceWith(self, mol):
        '''
        get distance between `self` and another `mol`
        '''
        lx, ly, lz = self.vectorWith(mol=mol)
        return (lx**2 + ly**2 + lz**2)**0.5

    def adjustCoord(self):
        '''adjust molecule's atom position'''
        box = self.box
        lx = box[0][1]-box[0][0]
        ly = box[1][1]-box[1][0]
        lz = box[2][1]-box[2][0]

        for a, b in self.getEdgesDFS():
            vx, vy, vz = a.vectorWith(b)
            n = move(vx, lx)
            b.position = (b.position[0]+lx*n, b.position[1], b.position[2])
            n = move(vy, ly)
            b.position = (b.position[0], b.position[1]+ly*n, b.position[2])
            n = move(vz, lz)
            b.position = (b.position[0], b.position[1], b.position[2]+lz*n)

    def getAimOxygen(self):
        '''
        Functions:
        ---
        get the `oxygen` of OH if self is OH, H3O2 or H5O3 

        Return:
        ---
        the aim oxygen, type `atom`
        '''
        if self.isTypeEqual(Mol('OH')):
            atomList = self._getNodesWithType(type='O')
            oxygen = atomList[0]
            return oxygen
            
        elif self.isTypeEqual(Mol('H3O2')):
            for hydrogen in self._getNodesWithType(type='H'):
                if hydrogen.bondsNumber()==2:
                    oxygenID = hydrogen.sortedBond()[0][0] #weaker bonded oxygen
                    oxygen = self._getNodeWithID(oxygenID)
                    return oxygen
                else: pass

        elif self.isTypeEqual(Mol('H5O3')):
            weakOxygens = []
            for hydrogen in self._getNodesWithType(type='H'):
                if hydrogen.bondsNumber() == 2:
                    oxygenID = hydrogen.sortedBond()[0][0] #weaker bonded oxygen
                    oxygen = self._getNodeWithID(oxygenID)
                    weakOxygens.append(oxygen)

                    if oxygen.bondsNumber()==2:
                        return oxygen
                    else: pass    
                else: pass

            return weakOxygens[0]
        
        elif self.isTypeEqual(Mol('H2O')):
            return self._getNodesWithType(type='O')[0]

        else: 
            printf('#[getAimOxygen] self is not OH, H3O2, H5O3, or H2O')



########################################################## Mol
def Mol(name: str) -> Molecule:
    '''
    Function:
    ----------
    name: 'water' or 'H2O', return a `Molecule` of water;\n
        \t'hydroxyl' or 'OH', return a `Molecule` of OH;\n
        \t'H3O2', return a `Molecule` of H3O2;\n
        \t'H5O3', return a `Molecule` of H5O3;\n
    '''
    def __getH2O():
        H1 = Atom(1, 2, (0.00, -0.763, -0.199), 0, [(2, 0.85)])
        O = Atom(2, 3, (0.00, 0.00, 0.398), 0, [(1, 0.85), (3, 0.85)])
        H2 = Atom(3, 2, (0, 0.763, -0.199), 0, [(2, 0.85)])
        atomsDir = {1: H1, 2: O, 3: H2}
        water = Molecule()
        water.initialize([1, 2, 3], atomsDir)
        water.id = 1
        water.box = ((-100, 100), (-100, 100), (-100, 100))
        return water

    def __getWater():
        H1 = Atom(1, 2, (10.00, -0.763, -0.199), 0, [(12, 0.85)])
        O = Atom(12, 3, (10.00, 0.00, 0.398), 0, [(1, 0.85), (3, 0.85)])
        H2 = Atom(3, 2, (10, 0.763, -0.199), 0, [(12, 0.85)])
        atomsDir = {1: H1, 12: O, 3: H2}
        water = Molecule()
        water.initialize([1, 12, 3], atomsDir)
        water.id = 2
        water.box = ((-100, 100), (-100, 100), (-100, 100))
        return water

    def __getOH():
        H = Atom(1, 2, (0, 0, -0.49), 0, [(2, 0.4)])
        O = Atom(2, 3, (0, 0, 0.490), 0, [(1, 0.4)])
        atomsDir = {1: H, 2: O}
        OH = Molecule()
        OH.initialize([1, 2], atomsDir)
        OH.box = ((-100, 100), (-100, 100), (-100, 100))
        return OH
    
    def __getNO():
        N = Atom(1, 4, (0, 0, -0.49), 0, [(2, 0.4)])
        O = Atom(2, 3, (0, 0, 0.490), 0, [(1, 0.4)])
        atomsDir = {1: N, 2: O}
        NO = Molecule()
        NO.initialize([1, 2], atomsDir)
        NO.box = ((-100, 100), (-100, 100), (-100, 100))
        return NO

    def __getH3O2():
        H1 = Atom(1, 2, (-0.358, 2.559, -0.788), 0, [(2, 0.4)])  # H
        O1 = Atom(2, 3, (-0.123, 1.519, -1.060),
                        0, [(1, 0.4), (3, 0.5)])  # O
        H2 = Atom(3, 2, (0, 0.763, -0.199), 0, [(2, 0.5), [4, 0.6]])  # H
        O2 = Atom(4, 3, (0.00, 0.00, 0.398),
                        0, [(3, 0.5), (5, 0.5)])  # O
        H3 = Atom(5, 2, (0.00, -0.763, -0.199), 0, [(4, 0.5)])  # O
        atomsDir = {1: H1, 2: O1, 3: H2, 4: O2, 5: H3}
        H3O2 = Molecule()
        H3O2.initialize([1, 2, 3, 4, 5], atomsDir)
        H3O2.box = ((-100, 100), (-100, 100), (-100, 100))
        return H3O2

    def __getH5O3():
        H1 = Atom(1, 2, (-0.358, 2.559, -0.788), 0, [(2, 0.4)])  # H
        O1 = Atom(2, 3, (-0.123, 1.519, -1.060), 0,
                        [(1, 0.4), (3, 0.5), (6, 0.5)])  # O
        w1_H1 = Atom(3, 2, (0, 0.763, -0.199), 0,
                           [(2, 0.5), [4, 0.6]])  # H
        w1_O = Atom(4, 3, (0.00, 0.00, 0.398),
                          0, [(3, 0.5), (5, 0.5)])  # O
        w1_H2 = Atom(5, 2, (0.00, -0.763, -0.199), 0, [(4, 0.5)])  # H
        w2_H1 = Atom(6, 2, (-0.167, 1.12, -2.200),
                           0, [(2, 0.5), [7, 0.6]])  # H
        w2_O = Atom(7, 3, (-0.286, 0.424, -3.2),
                          0, [(6, 0.5), (8, 0.5)])  # O
        w2_H2 = Atom(8, 2, (-0.307, 0.977, -4.293), 0, [(7, 0.5)])  # H
        atomsDir = {1: H1, 2: O1, 3: w1_H1, 4: w1_O,
                    5: w1_H2, 6: w2_H1, 7: w2_O, 8: w2_H2}
        H5O3 = Molecule()
        H5O3.initialize([1, 2, 3, 4, 5, 6, 7, 8], atomsDir)
        H5O3.box = ((-100, 100), (-100, 100), (-100, 100))
        return H5O3

    def __getH5O3_():
        H1 = Atom(1, 2, (-0.358, 2.559, -0.788), 0, [(2, 0.4)])  # H
        O1 = Atom(2, 3, (-0.123, 1.519, -1.060), 0,
                        [(1, 0.4), (3, 0.5), (6, 0.5)])  # O
        w1_H1 = Atom(3, 2, (0, 0.763, -0.199), 0,
                           [(2, 0.5), [4, 0.6]])  # H
        w1_O = Atom(4, 3, (0.00, 0.00, 0.398),
                          0, [(3, 0.5), (5, 0.5)])  # O
        w1_H2 = Atom(5, 2, (0.00, -0.763, -0.199), 0, [(4, 0.5)])  # H
        w2_H1 = Atom(6, 2, (-0.167, 1.12, -2.200),
                           0, [(2, 0.5), [7, 0.6]])  # H
        w2_O = Atom(7, 3, (-0.286, 0.424, -3.2),
                          0, [(6, 0.5), (8, 0.5)])  # O
        w2_H2 = Atom(8, 2, (-0.307, 0.977, -4.293), 0, [(7, 0.5)])  # H
        atomsDir = {1: H1, 2: O1, 3: w1_H1, 4: w1_O,
                    5: w1_H2, 6: w2_H1, 7: w2_O, 8: w2_H2}
        H5O3 = Molecule()
        H5O3.initialize([1, 2, 3, 4, 5, 6, 7, 8], atomsDir)
        H5O3.box = ((-100, 100), (-100, 100), (-100, 100))
        return H5O3

    def getG2(nameStr):
        # get atom frame
        atomFrame = {}
        atoms = g2[nameStr]
        idx = 1
        for atom in atoms:
            atomFrame[idx] = Atom(
                atomID=idx,
                atomType=ATOMTYPE_INVERSE()[atom.symbol],
                position=tuple( atom.position ),
                charge= 0.0,
                bonds=[],
                steps=1,
                )
            
            idx = idx + 1

        # get mol top
        newFrame = setBonds(atomFrame=atomFrame)
        g=nx.Graph()
        for atom in newFrame.values():
            g.add_node(atom, id=atom.id, type=atom.type)
            for id in atom.getRelatedAtomsID():
                g.add_edge(atom, newFrame[id])

        # set new mol
        newMol = Molecule()
        newMol.topology = g
        newMol.step = 1
        newMol.id = 1

        return newMol


    def setBonds(atomFrame:dict):
        '''
        set bond order
        '''
        newFrame = copy.deepcopy(atomFrame)

        for idxA, atomA in atomFrame.items():
            for idxB, atomB in atomFrame.items():
                if idxA == idxB:
                    pass
                else:
                    distance = atomA.distanceWith(atomB)
                    bondOrder = BOND_ORDER(ATOMTYPE[atomA.type], ATOMTYPE[atomB.type], distance)
                    if bondOrder >= DEFAULT_BONDER_ORDER:
                        newFrame[idxA].bonds.append( (atomB.id, bondOrder) )
                    else:
                        pass

        return newFrame
    

    if name == 'H2O':
        return __getH2O()
    elif name == 'water':
        return __getWater()
    elif name == 'OH' or name == 'hydroxyl' or name == 'HO':
        return __getOH()
    elif name == 'H3O2':
        return __getH3O2()
    elif name == 'H5O3':
        return __getH5O3()
    elif name == 'H5O3_':
        return __getH5O3_()
    elif name == 'NO' or name=='ON':
        return __getNO()
    else:
        return getG2(nameStr = name)


########################################################## Reaction
class Reaction():
    def __init__(self, R, P, stepR: int, stepP: int, id=None):
        self.R = R
        self.P = P
        self.stepR = stepR
        self.stepP = stepP
        self.id = id

        self.hashType = None
        self.hashID = None

        self.molHashSetR = None
        self.molHashSetP = None
        self.molHashSetAll = None

    def getHash(self, reGet = False):
        '''type hash'''
        if not reGet and self.hashType:
            return self.hashType
        else:
            hashR = sorted([mol.getHash() for mol in self.R])
            hashP = sorted([mol.getHash() for mol in self.P])
            self.hashType = hash((tuple(hashR), tuple(hashP)))
            return self.hashType

    def __eq__(self, rex) -> bool:
        '''id equal'''
        return hash((self.getHash(), self.stepR, self.stepP)) == hash((rex.getHash(), rex.stepR, rex.stepP))

    def __hash__(self) -> int:
        '''id hash'''
        return hash((self.getHash(), self.stepR, self.stepP))

    def __str__(self) -> str:
        RStr = ''
        for r in self.R:
            RStr = RStr + r.getSymbols() + " + "
        PStr = ''
        for p in self.P:
            PStr = PStr + p.getSymbols() + " + "

        stepStr = '[Step:' + str(self.stepR) + '-' + \
            str(self.stepP) + "|ID:" + str(self.id) + "] "
        return stepStr + RStr[0:-2] + "=> " + PStr[0:-2]

    def isRex(self):
        if len(self.R) == 1 and len(self.P) == 1 and self.R[0].getHash() == self.P[0].getHash():
            return False
        elif len(self.R) == 0 and len(self.P) == 0:
            return False
        else:
            return True

    def isContainR(self, mol):
        for i in self.R:
            if i.equal(mol):
                return True
            else:
                pass
        return False

    def isContainP(self, mol):
        for i in self.P:
            if i.equal(mol):
                return True
            else:
                pass
        return False

    def isContain(self, mol):
        if self.isContainR(mol) or self.isContainP(mol):
            return True
        else:
            return False

    def rEqual(self, rex):
        hashRac = sorted([mol.getHash() for mol in self.R])
        hashMol = sorted([mol.getHash() for mol in rex.R])
        if hashRac == hashMol:
            return True
        else:
            return False

    def pEqual(self, rex):
        hashPro = sorted([mol.getHash() for mol in self.P])
        hashMol = sorted([mol.getHash() for mol in rex.P])
        if hashPro == hashMol:
            return True
        else:
            return False

    def equal(self, rex):
        if self.getHash() == rex.getHash():
            return True
        else:
            return False

    def isIn(self, rexList):
        hashSelf = self.getHash()
        hashList = [rex.getHash() for rex in rexList]
        if hashSelf in hashList:
            return True
        else:
            return False

    def isHashIn(self, hashList):
        hashSelf = self.getHash()
        if hashSelf in hashList:
            return True
        else:
            return False

    def getIso(self, rexList):
        '''get iso of self in the rexList'''
        for rex in rexList:
            if self.equal(rex):
                return rex
            else:
                pass
        return False

    def getTargetMols(self, molList):
        '''get `mols` both in this `rex` and in the `molList`'''
        aimR, aimP = [], []
        for mol in self.R:
            if mol.isIn(molList):
                aimR.append(mol)
            else:
                pass

        for mol in self.P:
            if mol.isIn(molList):
                aimP.append(mol)
            else:
                pass

        return (aimR, aimP)

    def getSpsList(self):
        '''
        ## Function
        get all mol in R and P

        ## Return
        `[mol, , , ]`
        '''
        allSps = list(self.R + self.P)
        return allSps

    def getSpsHash(self, reGet = False):
        '''
        ## Function
        get all species hashCode set

        ## Return
        `{hashCode, , , }`
        '''
        if not reGet and self.molHashSetAll:
            return self.molHashSetAll
        else:
            hashList = [sps.getHash() for sps in self.getSpsList()]
            self.molHashSetAll = set(hashList)
            return self.molHashSetAll
    
    def getSpsHashR(self, reGet = False):
        '''
        ## Function
        get reactant's hashCode set

        ## Return
        `{hashCode, , , }`
        '''
        if not reGet and self.molHashSetR:
            return self.molHashSetR
        else:
            hashList = [sps.getHash() for sps in self.R]
            self.molHashSetR = set(hashList)
            return self.molHashSetR
    
    def getSpsHashP(self, reGet = False):
        '''
        ## Function
        get all product's hashCode set

        ## Return
        `{hashCode, , , }`
        '''
        if not reGet and self.molHashSetP:
            return self.molHashSetP
        else:
            hashList = [sps.getHash() for sps in self.P]
            self.molHashSetP = set(hashList)
            return self.molHashSetP


