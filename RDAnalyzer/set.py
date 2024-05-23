import os

DEFAULT_BONDER_ORDER = 0.5
'''
The `bond order` that determines whether a bond is formed
`default`: 0.3
'''

ORIGIN_FIGURE_NUMBER = 0
def FIGURE_NUMBER():
    '''
    generate unique global figure number
    '''
    global ORIGIN_FIGURE_NUMBER
    ORIGIN_FIGURE_NUMBER = ORIGIN_FIGURE_NUMBER + 1
    return ORIGIN_FIGURE_NUMBER


ATOMTYPE = {
    1: 'C',
    2: 'H',
    3: 'O',
    4: 'N',
    }
'''
atom type information that consistent with the `data.lmps` file
'''
def ATOMTYPE_INVERSE():
    '''
    the inverse dict of ATOMTYPE dict
    '''
    global ATOMTYPE

    newDict = {}
    for key, value in ATOMTYPE.items():
        newDict[value] = key

    return newDict



TIMESTEP = 0.0001 #ps
'''time step'''

DUMP_INTERVAL = 1000
'''dump step interval'''

TIME_UNIT = 'ps'
'''time unit'''



def BOND_ORDER(typeA:str, typeB:str, length):
    '''
    bond order
    '''
    types = {typeA, typeB}

    if types == {'C'}: # C
        if length <= 1.2:
            return 3.0
        elif length > 1.2 and length <= 1.34:
            return 2.0
        elif length > 1.34 and length <= 1.54:
            return 1.0
        else: 
            return 0.0

    elif types == {'C', 'H'}:
        if length <= 1.19:
            return 1.0
        else: 
            return 0.0
    
    elif types == {'C', 'N'}:
        if length <= 1.16:
            return 3.0
        elif length > 1.16 and length <= 1.35:
            return 2.0
        elif length > 1.35 and length <= 1.48:
            return 1.0
        else: 
            return 0.0
    
    elif types == {'C', 'O'}:
        if length <= 1.43:
            return 1.0
        else: 
            return 0.0
        
    
    elif types == {'H'}: # H
        if length <= 0.75:
            return 1.0
        else: 
            return 0.0
        
    elif types == {'H', 'O'}:
        if length <= 0.98:
            return 1.0
        else:
            return 0.0
    
    elif types == {'H', 'N'}:
        if length <= 1.01:
            return 1.0
        else:
            return 0.0
        
    elif types == {'O'}: # O
        if length <= 1.20:
            return 2.0
        elif length > 1.20 and length <= 1.48:
            return 1.0
        else:
            return 0.0
        
    elif types == {'O', 'N'}:
        if length <= 1.14:
            return 2.0
        elif length > 1.14 and length <= 1.46:
            return 1.0
        else:
            return 0.0
    
    elif types == {'N'}: # N
        if length <= 1.10:
            return 3.0
        elif length > 1.10 and length <= 1.25:
            return 2.0
        elif length > 1.25 and length <= 1.45:
            return 1.0
        else: 
            return 0.0

        
    else:
        raise Exception("unknown types: ", types)

