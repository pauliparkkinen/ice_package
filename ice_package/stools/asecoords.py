import coords
# from ase import *
# .. no good in never version of ASE..!
from ase import io, constraints, Atoms, Atom

import numpy as np
import copy


def geom2atoms(geom,atomsobj=None):
  fixinds=[]
  planefixinds=[]
  cc=0
  # from my geometry list format to ase Atom format
  myatoms=Atoms()
  syms=""
  for g in geom:
    syms=syms+g[0]
    newatom=Atom(g[0],(g[1],g[2],g[3]))
    myatoms.append(newatom)
    if (len(g)>4):
      
      if (g[4].has_key("fixed")):
	if (g[4]["fixed"]):
	  fixinds.append(cc)
	  
      if (g[4].has_key("planefixed")):
	if (g[4]["planefixed"]):
	  planefixinds.append(cc)
	  
    cc=cc+1
    
  c = constraints.FixAtoms(indices = fixinds) # these should come from ase.constraints
  d = constraints.FixedPlane(planefixinds, (0,0,1))
  
  if (len(fixinds)>0 and len(planefixinds)>0):
    myatoms.set_constraint([c, d])
  elif (len(fixinds)>0):
    myatoms.set_constraint(c)
  elif (len(planefixinds)>0):
    myatoms.set_constraint(d)
    
  if (atomsobj==None):
    return myatoms # pristine atoms object
  else:
    newatomsobj=copy.deepcopy(atomsobj)
    newatomsobj.set_chemical_symbols(syms)
    newatomsobj.set_positions(myatoms.get_positions())
    vels=myatoms.get_velocities()
    if (vels!=None):
      newatomsobj.set_velocities(myatoms.get_velocities())
    return newatomsobj


def changegeom(geom, atoms):
  # atoms: atoms object
  c=0
  for g in geom:
    atoms[c].set_position(np.array([g[1],g[2],g[3]]))
    c=c+1
    

def geoms2atoms(geoms):
  atl=[]
  for geom in geoms:
    atl.append(geom2atoms(geom))
    
  return atl
  

def atoms2geom(ats):
  geom=[]
  
  const=ats._get_constraints()
  inds=[]
  if (const!=[]):
    # inds=const[0].index
    # print const[0].indices # according to manual should be this..!
    pass # fcked!
  c=0
  for at in ats: # iterate over individual atoms
      spec=at.get_symbol()
      xyzs=at.get_position()
      # print ">",at.get_tag()
      # print ">",spec,xyzs
      g=[spec,xyzs[0],xyzs[1],xyzs[2]]
      if (inds!=[]):
	if (c in inds):
	  g.append({"fixed":True})
      # print g
      geom.append(g)
      c=c+1
  return geom


def atoms2geoms(li):
  # li = list of atomS objects
  # print "consts:",li[0].constraints
  # constraints _should_ be here.. tryout.constraints[0].index
  geoms=[]
  const=li[0]._get_constraints()
  inds=[]
  # print "const=",type(const)
  if (const!=[]):
    # print "const[0]"+str(type(const[0]))
    # if (type(const[0])==type(FixedPlane)):
    #if (type(const[0])==type(FixAtoms)):
    #  inds=const[0].index
      # print const[0].indices # according to manual should be this..!
    pass # fck!
  
  for ats in li: # iterate over differet geometries
    geom=[]
    c=0
    for at in ats: # iterate over individual atoms
      spec=at.get_symbol()
      xyzs=at.get_position()
      # print ">",at.get_tag()
      # print ">",spec,xyzs
      g=[spec,xyzs[0],xyzs[1],xyzs[2]]
      if (inds!=[]):
	if (c in inds):
	  g.append({"fixed":True})
      # print g
      geom.append(g)
      c=c+1
    geoms.append(geom)
      
  return geoms
      

def atoms_overwrite(ats,geom):
  # ats = atomS object
  # geom = my geom list
  c=0
  for at in ats: # iterate over individual atoms
    g=geom[c]
    at.set_position((g[1],g[2],g[3]))
    c=c+1
  
  

  