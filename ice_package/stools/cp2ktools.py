import coords
import copy
import os
import sys
import ase
import numpy as np
import numpy.linalg as lg
import math
from ase.io.trajectory import PickleTrajectory
import asecoords
from ase.constraints import FixBondLengths

deb=0.39343 # didive by this to go from from e*Bhr to debye
deb2=0.208194 # .. e*Ang to debye

# parameters given by Kari Laasonen:
def_kari={'MOTION': {'PRINT': {'RESTART_HISTORY OFF': {'inp': []}, 'inp': [], 'RESTART': {'inp': ['BACKUP_COPIES 1', 'ADD_LAST NUMERIC', 'COMMON_ITERATION_LEVELS 1'], 'EACH': {'inp': ['MD 5', 'QS_SCF 50']}}}, 'MD': {'inp': ['ENSEMBLE NVT', 'STEPS 10', 'TIMESTEP 1.0', 'TEMPERATURE 308.15'], 'THERMOSTAT': {'inp': ['TYPE NOSE'], 'NOSE': {'inp': ['LENGTH 3', 'YOSHIDA 3', 'TIMECON  300.0', 'MTS 2']}}}, 'inp': []}, 'FORCE_EVAL': {'DFT': {'MGRID': {'inp': ['CUTOFF 280']}, 'QS': {'inp': ['EPS_DEFAULT 1.0E-12', 'MAP_CONSISTENT', 'EXTRAPOLATION ASPC', 'EXTRAPOLATION_ORDER 3']}, 'SCF': {'MIXING': {'inp': ['ALPHA 0.45']}, 'PRINT': {'RESTART_HISTORY OFF': {'inp': []}, 'inp': [], 'RESTART ON': {'inp': []}}, 'OT': {'inp': ['MINIMIZER DIIS', 'PRECONDITIONER FULL_SINGLE_INVERSE', 'ENERGY_GAP 0.05']}, 'inp': ['SCF_GUESS RESTART', 'EPS_SCF 1.0E-6', 'MAX_SCF 35'], 'OUTER_SCF': {'inp': ['EPS_SCF 1.0E-6', 'MAX_SCF 20']}}, 'inp': ['BASIS_SET_FILE_NAME /wrk/sriikone/cp2k/libs/QS/BASIS_MOLOPT', 'POTENTIAL_FILE_NAME /wrk/sriikone/cp2k/libs/QS/GTH_POTENTIALS'], 'XC': {'XC_GRID': {'inp': ['XC_DERIV SPLINE2_smooth', 'XC_SMOOTH_RHO NN10']}, 'inp': []}}, 'SUBSYS': {'CELL': {'inp': ['ABC          5.0 5.0 5.0']}, 'KIND H': {'inp': ['BASIS_SET DZVP-MOLOPT-SR-GTH', 'POTENTIAL GTH-PBE-q1', 'MASS  2.0000']}, 'inp': [], 'COORD': {'inp': ['H 0.761894863075 1.31927995163 0.341141726776', 'O 0.798844667651 0.418195935151 0.0', 'H 0.0 0.0 0.341546423801']}, 'KIND O': {'inp': ['BASIS_SET DZVP-MOLOPT-SR-GTH', 'POTENTIAL GTH-PBE-q6']}}, 'inp': ['METHOD Quickstep']}, 'GLOBAL': {'inp': ['PREFERRED_FFT_LIBRARY FFTSG', 'PROJECT out-w256', 'RUN_TYPE MD', 'PRINT_LEVEL LOW']}, 'inp': []}


# spc water:
#def_spc={'MOTION': {'MD': {'inp': ['ENSEMBLE NVT', 'STEPS 10000', 'TIMESTEP 2.5', 'TEMPERATURE 300.0'], 'THERMOSTAT': {'inp': ['REGION MOLECULE'], 'NOSE': {'inp': ['LENGTH 3', 'YOSHIDA 3', 'TIMECON #1000', 'MTS 2']}}}, 'inp': [], 'CONSTRAINT': {'G3X3': {'inp': ['DISTANCES 1.8897268 1.8897268 3.0859239', 'MOLECULE 1', 'ATOMS 1 2 3']}, 'inp': []}}, 'FORCE_EVAL': {'MM': {'inp': [], 'FORCEFIELD': #{'NONBONDED': {'LENNARD-JONES': {'inp': ['atoms H H', 'EPSILON 0.0', 'SIGMA 3.30523', 'RCUT 11.4']}, 'inp': []}, 'CHARGE': {'inp': ['ATOM H', 'CHARGE 0.4238']}, 'inp': [], 'BEND': {'inp': ['ATOMS H O #H', 'K 0.', 'THETA0 1.8']}, 'BOND': {'inp': ['ATOMS O H', 'K 0.', 'R0 1.8']}}, 'POISSON': {'EWALD': {'inp': ['EWALD_TYPE ewald', 'ALPHA .5', 'GMAX 21']}, 'inp': []}}, 'inp': ['METHOD Fist']}, #'GLOBAL': {'inp': ['PROJECT cp2k', 'RUN_TYPE MD']}, 'inp': []}
#
# using ase.units:
# Bohr*1.8897268 ~ 1.0000
# Bohr*3.0859239 ~ 1.63300
#
# it seems that the charge values are for the polarizable spc model..?

# a more minimalistic..
def_spc={'MOTION': {'MD': {'inp': ['ENSEMBLE NVT', 'STEPS 10000', 'TIMESTEP 2.5', 'TEMPERATURE 300.0'], 'THERMOSTAT': {'inp': ['REGION MOLECULE'], 'NOSE': {'inp': ['LENGTH 3', 'YOSHIDA 3', 'TIMECON 1000', 'MTS 2']}}}, 'inp': [], 'CONSTRAINT': {'inp': []}}, 'FORCE_EVAL': {'MM': {'inp': [], 'FORCEFIELD': {'inp': []}, 'POISSON': {'EWALD': {'inp': ['EWALD_TYPE ewald', 'ALPHA .5', 'GMAX 21']}, 'inp': []}}, 'inp': ['METHOD Fist']}, 'GLOBAL': {'inp': ['PROJECT cp2k', 'RUN_TYPE MD']}, 'inp': []}


velocityfac=1.0
# 1 (ase time unit) = (tk) * 1 (cp2k time unit)
#
# [time in ase time unit] = velocityfac * [time in cp2k time unit]
#
# velocityfac = 1 / tk
#
#
# ase units: eV:=1 and Ang:=1
# ase units of time, lest call it "asetime"
#
# ase.units.fs *(time [fs]) = time [asetime]
#
# ase.units.AUT # atomic unit of time in ase units (i.e. when eV:=1 and Ang:=1)
# 
# ase.units.AUT * (time [atomic units]) = time [asetime]
#
# cp2k velocity unit: Bohr/AUT
#
# f = (ase.units.Bohr/ase.units.AUT)
#
# ****************************************************
# f * (velocity [bohr/AUT] ) = velocity [Ang/asetime]
# ****************************************************
#
# velocity [bohr/AUT] = (1/f) velocity [Ang/asetime]
#

# velocityfac=ase.units.Bohr/ase.units.AUT # multiply by this and you go from cp2k units to ase units
velocityfac=ase.units.Bohr/0.002375996007491888

# >>> import ase
# >>> ase.units.AUT
# 0.002375996007491888

spc={"o_charge":-0.8476,"h_charge":0.4238,"angle":109.47}
# .. those values are for the SPC/E model

def migprotos(fname,atnums):
  geom=coords.readxmol(fname)[-1]
  coords.migrate(geom,atnums)
  coords.writexmolmany("migrated.xmol",[geom])


def masses(traj,anum,mass):
  # change masses for a species in an atoms object
  # anum = atomic number
  # mass = new mass
  ar=traj.get_atomic_numbers()
  li=np.where(ar==anum)[0]
  # print "li=",li
  for i in range(len(li)):
    # print li[i],traj[li[i]]
    traj[li[i]].mass=copy.copy(mass)
  

def printvels(atomsobj,fac=10.0):
  scale=0.1
  color="black"
  # st="plot arrow "+str(x1)+" "+str(y1)+" "+str(z1)+" "+str(x2)+" "+str(y2)+" "+str(z2)+" "+str(scale)+" "+color
  append=""
  lines=[]
  for a in atomsobj:
    pos=a.get_position()
    mv=a.get_momentum()
    m=a.get_mass()
    v=mv/m
    
    fr=pos
    to=pos+v*fac
    
    if (lg.norm(v)>0.000001):
      st="plot arrow "+str(fr[0])+" "+str(fr[1])+" "+str(fr[2])+" "+str(to[0])+" "+str(to[1])+" "+str(to[2])+" "+str(scale)+" "+color+append
      lines.append(st)
      append=" append"
    
  return lines


def printcell(atomsobj, center=False):
  color="blue"
  scale=0.2
  lines=[]
  cell=atomsobj.get_cell()

  ofsx=0;ofsy=0;ofsz=0

  dx=cell[0][0]
  dy=cell[1][1]
  dz=cell[2][2]

  vec=np.array([dx,dy,dz])

  print "vec=",vec
  if (center):
    # set L/2 into center of mass
    # vec=vec-atomsobj.get_center_of_mass()
    ofsx=-vec[0]/2.0
    ofsy=-vec[1]/2.0
    ofsz=-vec[2]/2.0
    
  print ofsx,ofsy,ofsz
    
  # print "vec now=",vec
  dx=vec[0];dy=vec[1];dz=vec[2]

  # st="plot line "+str(-dx)+" "+str(0)+" "+str(0)+" "+str(dx)+" "+str(0)+" "+str(0)+" "+color; lines.append(st)
  # st="plot line "+str(0)+" "+str(-dy)+" "+str(0)+" "+str(0)+" "+str(dy)+" "+str(0)+" "+color+" append"; lines.append(st)
  # st="plot line "+str(0)+" "+str(0)+" "+str(-dz)+" "+str(0)+" "+str(0)+" "+str(dz)+" "+color+" append"; lines.append(st)
  
  append=""
  for i in range(0,2):
    for j in range(0,2):
      for k in range(0,2):
	x0=i*dx+ofsx
	y0=j*dy+ofsy
	z0=k*dz+ofsz
	
	x1=x0
	y1=y0
	z1=z0
	
	st="plot line "+str(x0)+" "+str(y0)+" "+str(z0)+" "+str(x1-i*dx)+" "+str(y1)+" "+str(z1)+" "+color+append; lines.append(st)
	append=" append"
	st="plot line "+str(x0)+" "+str(y0)+" "+str(z0)+" "+str(x1)+" "+str(y1-j*dy)+" "+str(z1)+" "+color+append; lines.append(st)
	st="plot line "+str(x0)+" "+str(y0)+" "+str(z0)+" "+str(x1)+" "+str(y1)+" "+str(z1-k*dz)+" "+color+append; lines.append(st)
  
  return lines


def chain(filename,n=0):
  # filename: eka.toka
  # changes eka-N => eka-N-1
  
  thisfile=copy.copy(filename)
  parts=filename.split(".")
  
  if (n==0):
    nst=""
  else:
    nst="-"+str(n)
    
  nnst="-"+str(n+1)
    
  tryfile=parts[0]+nst+"."+parts[1]
  
  if (os.path.exists(tryfile)):
    # if n already exists..
    chain(filename,n+1)
    nfile=parts[0]+nst+"."+parts[1]
    nnfile=parts[0]+nnst+"."+parts[1]
    comm="mv "+nfile+" "+nnfile
    print comm
    os.system(comm)
    comm="touch "+nnfile
    print comm
    os.system(comm)
  else:
    pass


def unchain(filename,n=1):
  # filename: eka.toka
  # changes eka-N-2 => eka-N-1
  # .. all the way to N-1
  
  if (n<=0):
    return
  
  # print "unchain",n
  
  thisfile=copy.copy(filename)
  parts=filename.split(".")
  
  nnst="-"+str(n+1)
  
  if (n==0):
    nst=""
  else:
    nst="-"+str(n)
    
  tryfile=parts[0]+nnst+"."+parts[1]
  
  if (os.path.exists(tryfile)):
    # if n exists, copy n to n-1 and remove n
    nfile=parts[0]+nst+"."+parts[1]
    nnfile=parts[0]+nnst+"."+parts[1]
    comm="mv "+nnfile+" "+nfile
    print comm
    os.system(comm)
    unchain(filename,n+1)
  else:
    pass


def fixtraj(initfile,posfile,velfile=None):
  # collect positions and velocities to a an initial atoms object
  #TODO: check the units! .. seem to be OK.
  traj=ase.io.read(initfile)
  print ">traj=",traj
  p=ase.io.read(posfile)
  
  traj.set_positions(p.get_positions())
  traj.set_chemical_symbols(p.get_chemical_symbols())
  # traj.set_masses(p.get_masses()) # this way you overwrite the (deuterium) masses!
  if (velfile!=None):
    # ok.. the velocities must be scaled to ase units..
    velos=ase.io.read(velfile).get_positions()
    velos=velos*velocityfac # from cp2k to ase units
    ### traj.set_velocities(ase.io.read(velfile).get_positions())
    traj.set_velocities(velos)
  return traj


def writefile(filename,crds):
    f=open(filename,'w')
    for l in crds:
	# print ">>>",l
        f.write(l+"\n")
    f.close()

def stuff(dic,subentries,final,value):
  if (len(subentries)==0):
    dic[final]=value
    # print "dic=",dic
    return
  else:
    entry=subentries[0]
    if (dic.has_key(entry)):
      pass
    else:
      dic[entry]={}
    
    # print "subentries=",subentries
    # newsubentries=subentries[1:]
    stuff(dic[entry],subentries[1:],final,value)
    # stuff(dic[entry],newsubentries,final,value)
  

def set_input(dic,subentries,final,value):
  # print "set_input:",dic
  # print "set_input:",subentries
  if (len(subentries)==0):
    # print "zero subentries"
    # print "dic=",dic
    # stop
    if (dic.has_key("inp")):
      # check if parameter is there..
      cc=0
      found=False
      for d in dic["inp"]:
	if (d.find(final)!=-1):
	  dic["inp"][cc]=final+" "+str(value)
	  found=True
	cc=cc+1
      if (found==False):
	dic["inp"].append(final+" "+str(value))
    else:
      dic["inp"]=[final+" "+str(value)]
    return
  else:
    entry=subentries[0]
    if (dic.has_key(entry)):
      pass
    else:
      dic[entry]={}
    
    # print "subentries=",subentries
    # newsubentries=subentries[1:]
    # print "calling setinput..",dic[entry],subentries[1:]
    set_input(dic[entry],subentries[1:],final,value)
    # stuff(dic[entry],newsubentries,final,value)


def del_input(dic,subentries,final):
  if (len(subentries)==0):
    # print "zero subentries"
    # print "dic=",dic
    # stop
    #if (dic.has_key(final)):
    #  del dic[final]
    #return
    for d in dic.keys():
      # print "d=",d
      if (d.split()[0]==final):
	# print ">>>>",d,final
	# stop
	del dic[d]
	return
  else:
    entry=subentries[0]
    # print "subentries=",subentries
    # newsubentries=subentries[1:]
    # print "calling setinput..",dic[entry],subentries[1:]
    del_input(dic[entry],subentries[1:],final)
    # stuff(dic[entry],newsubentries,final,value)
  
  
def get_input(dic,subentries,final):
  if (len(subentries)==0):
    # print "zero subentries"
    # print "dic=",dic
    # stop
    #if (dic.has_key(final)):
    #  del dic[final]
    #return
    for d in dic.keys():
      # print "d=",d
      if (d.split()[0]==final):
	# print ">>>>",d,final,dic[d]
	# stop
	return dic[d]
  else:
    entry=subentries[0]
    # print "subentries=",subentries
    # newsubentries=subentries[1:]
    # print "calling setinput..",dic[entry],subentries[1:]
    return get_input(dic[entry],subentries[1:],final)
    # stuff(dic[entry],newsubentries,final,value)
 
  
  
def parse(mydict,allines,start):
  cc=copy.copy(start)
  if (cc>len(allines)-1):
    closed=True
  else:
    closed=False
  while (not closed):
    l=allines[cc]
    if (len(l.lstrip().rstrip())==0):
      cc=cc+1
    elif (l.lstrip()[0]=="#"):
      cc=cc+1
    elif (l.find("&END")!=-1):
      # print "subsection closed"
      closed=True
      return cc+1
    elif (l.find("&")!=-1):
      # a new subsection
      st=l.split("#")[0]
      st=st[st.index("&")+1:].rstrip()
      secname=copy.copy(st)
      # print "new subsection",secname
      mydict[secname]={} # a new subdictionary
      mydict[secname]["inp"]=[]
      cc=parse(mydict[secname],allines,cc+1)
    else:
      mydict["inp"].append(l.lstrip().rstrip())
      cc=cc+1
    if (cc>len(allines)-1):
      closed=True
      
  return 0
  
  
def menusort(st1,st2):
  sts1=st1.split()
  sts2=st2.split()
  
  # print sts1
  # print sts2
  
  if ( len(sts1)<2 and len(sts2)<2 ):
    # print "both none"
    return 0
    
  if ( len(sts1)==2 and len(sts2)==1 ):
    # print "second none"
    return -1
    
  if ( len(sts1)==1 and len(sts2)==2 ):
    # print "first none"
    return 1
    
  try:
    i1=int(sts1[1])
  except ValueError:
    return 0
  
  try:
    i2=int(sts2[1])
  except ValueError:
    return 0
  
  # print ">",st1,st2,i1,i2
  
  if (i1>i2):
    return 1
  elif (i1<i2):
    return -1
  else:
    return 0
    
    
def menusort2(st1,st2):
  val=menusort(st1,st2)
  
  print ">",st1,st2
  print "val=",val
  return val
  
  
    
def testmenusort():  
  lis=['REPLICA 10', 'OPTIMIZE_BANDS', 'REPLICA 9', 'DIIS', 'REPLICA 5', 'inp', 'REPLICA 7', 'REPLICA 6', 'REPLICA 1', 'CONVERGENCE_CONTROL', 'REPLICA 2', 'REPLICA 3', 'REPLICA 4', 'REPLICA 8']
  lis=['COLVAR 3','COLVAR 2']
  lis.sort(menusort)
  print lis

    
def getlines(pars, inte=1, level=0):
  # printable lines from a dictionary
  lines=[]
  if (pars.has_key("inp")):
    for l in pars["inp"]:
      lines.append(" "*level+str(l))
    
  keylist=pars.keys()
  # print "keylist:",keylist
  keylist.sort()
  keylist.sort(menusort)
  # print "keylist sorted:",keylist
  # for key in pars.keys():
  for key in keylist:
    if (key!="inp"):
      lines.append(" "*level+"&"+key)
      
      lines=lines+getlines(pars[key],level=level+inte)
      # lines.append(" "*level+"&END "+key)
      lines.append(" "*level+"&END "+key.split(" ")[0])
      
      if (level==0):
	lines.append(" "); lines.append(" ");
      if (level==1):
	lines.append(" ");
      
  return lines
    

def geom2cpk(pars,geom):
  lines=[]
  for g in geom:
    lines.append(g[0]+" "+g[1]+" "+g[2]+" "+g[3])
    
  # return lines # this goes to ["FORCE_EVAL"]["SUBSYS"]["COORD"]["inp"]
  stuff(pars,["FORCE_EVAL","SUBSYS","COORD"],"inp",lines)
  

def get_extent(geom, verbose=False):
  # get the extent of the system ..
  # .. in the mt solver, 
  # http://groups.google.com/group/cp2k/tree/browse_frm/month/2008-3?hide_quotes=no
  """
  formally, the MT p-solver requires a box of twice the size of a
  sphere that encloses the total charge density of your molecule
  (i.e. atoms _and_ electron density). i can see small errors crop
  up on my (weakly interacting) test system starting from exactly
  the %50 mark and then the errors became intolerably large at
  having 40% vacuum. 
  ...
  ...
  The MT scheme requires therefore a simulation cell about twice as big
  as yous charge distribution ..
  """
  
  # 99% of the charge of the hydrogen atom is confined in a sphere
  # of radius of 4 au ~ 2.1 Ang
  # vdw radius: 120 pm = 1.2 Ang 
  #
  # Na: vdw radius: 227 pm = 2.27 Ang
  # O:  vdw radius: 152 pm = 1.52 Ang
  #
  # .. put a box of ~ 3 Ang around each atom?
  # .. calculate extent 
  # .. double that extent.. or multiply by 2.5 ..? .. or even with 3.0
  
  # for a very general case, set the "cutoff" for any atom
  
  # geom is an ase atoms object
  pos=geom.get_positions()
  cell=geom.get_cell()
  
  xma=pos[:,0].max()
  xmi=pos[:,0].min()
  
  yma=pos[:,1].max()
  ymi=pos[:,1].min()
  
  zma=pos[:,2].max()
  zmi=pos[:,2].min()
  
  dx=xma-xmi
  dy=yma-ymi
  dz=zma-zmi
  
  usemin=False
  
  if (usemin):
    # for some systems, say, molecules, atoms are in plane, creating unrealistically thin cells..
    dx=max(dx,2.5)
    dy=max(dy,2.5)
    dz=max(dz,2.5)
  
  atombox=3.0
  
  dx=dx+2.0*atombox
  dy=dy+2.0*atombox
  dz=dz+2.0*atombox
  
  dx=round(dx); dy=round(dy); dz=round(dz);
  
  if (verbose):
    print 
    print "dx,dy,dz=",dx,dy,dz
    print "mt solver would need a cellsize of > ",2*dx,2*dy,2*dz
    print "recommended cellsize (if you want to be extra carefull..): ",3*dx,3*dy,3*dz
    celli=[int(math.ceil(3*dx)),int(math.ceil(3*dy)),int(math.ceil(3*dz))]
    print "                 cell="+str(celli)
    # print "if you add a molecule at y 10 Ang.."
    # celli=[int(math.ceil(3*dx)),int(math.ceil(3*(dy+10))),int(math.ceil(3*dz))]
    # print "                 cell="+str(celli)
    
    #
    # length from X to X = L
    #
    # X        X        X       X
    #          |    |           |
    # atoms    |    |           | unit cell boundray
    # upto here     |
    #               | charge density
    #                has died out here
    #
    # .. i.e: there are atoms from 0 to L and charge density has (hopefully) died out before 3L/2
    #
    print "cellsize in this file is: ",cell[0][0],cell[1][1],cell[2][2]
    print
  
  vec=np.array([dx,dy,dz])
  
  return vec



def fixpars(pars,wrk,pseudo="PBE",basis="DZVP-MOLOPT-SR-GTH"):
  # where are the pseudos & basis sets..
  # wrkdir="/wrk/sriikone"
  # cp2ktools.stuff(pars,["FORCE_EVAL","DFT"],"inp",[
  #'BASIS_SET_FILE_NAME '+wrk+'/cp2k/libs/QS/BASIS_MOLOPT',
  #'POTENTIAL_FILE_NAME '+wrk+'/cp2k/libs/QS/GTH_POTENTIALS',
  #])
  
  set_input(pars,["FORCE_EVAL","DFT"],"BASIS_SET_FILE_NAME",wrk+'/cp2k/libs/QS/BASIS_MOLOPT')
  set_input(pars,["FORCE_EVAL","DFT"],"POTENTIAL_FILE_NAME",wrk+'/cp2k/libs/QS/GTH_POTENTIALS')
  
  bsline='BASIS_SET '+basis
  # some basis sets ..
  #O  SZV-MOLOPT-GTH SZV-MOLOPT-GTH-q6
  #O  DZVP-MOLOPT-GTH DZVP-MOLOPT-GTH-q6
  #O  TZVP-MOLOPT-GTH TZVP-MOLOPT-GTH-q6
  #O  TZV2P-MOLOPT-GTH TZV2P-MOLOPT-GTH-q6
  #O  TZV2PX-MOLOPT-GTH TZV2PX-MOLOPT-GTH-q6
  #O  SZV-MOLOPT-SR-GTH SZV-MOLOPT-SR-GTH-q6
  #O  DZVP-MOLOPT-SR-GTH DZVP-MOLOPT-SR-GTH-q6

  stuff(pars,["FORCE_EVAL","SUBSYS","KIND H"],"inp",[
    bsline,
    'POTENTIAL GTH-'+pseudo+'-q1',
    'MASS  2.0000'
    ])
    
  stuff(pars,["FORCE_EVAL","SUBSYS","KIND O"],"inp",[
    bsline,
    'POTENTIAL GTH-'+pseudo+'-q6'
  ])

  # return
  # some global parameters..
  stuff(pars,["GLOBAL"],"inp",[
    # 'PREFERRED_FFT_LIBRARY FFTACML',
    'PREFERRED_FFT_LIBRARY FFTW',
    'PROJECT cp2k',
    'RUN_TYPE MD',
    'PRINT_LEVEL LOW',
    'EXTENDED_FFT_LENGTHS'
  ])
  
  stuff(pars,["FORCE_EVAL","SUBSYS","KIND Cl"],"inp",[
   bsline,
   'POTENTIAL GTH-'+pseudo+'-q7'
  ])
  
  # O  DZVP-MOLOPT-SR-GTH DZVP-MOLOPT-SR-GTH-q6
  stuff(pars,["FORCE_EVAL","SUBSYS","KIND N"],"inp",[
   bsline,
   # 'BASIS_SET DZVP-MOLOPT-SR-GTH',
   # 'BASIS_SET TZVP-MOLOPT-SR-GTH', # not available
   # 'BASIS_SET TZVP-MOLOPT-GTH-q5',
   'POTENTIAL GTH-'+pseudo+'-q5'
  ])
  #print "fixpars:",pars

    
   # B  DZVP-MOLOPT-SR-GTH DZVP-MOLOPT-SR-GTH-q3
  stuff(pars,["FORCE_EVAL","SUBSYS","KIND B"],"inp",[
   bsline,
   # 'BASIS_SET DZVP-MOLOPT-SR-GTH',
   # 'BASIS_SET TZVP-MOLOPT-SR-GTH', # not available
   # 'BASIS_SET TZVP-MOLOPT-GTH-q3',
   'POTENTIAL GTH-'+pseudo+'-q3'
  ])


  stuff(pars,["FORCE_EVAL","SUBSYS","KIND Cu"],"inp",[
   bsline,
   # 'BASIS_SET DZVP-MOLOPT-SR-GTH',
   # 'BASIS_SET TZVP-MOLOPT-SR-GTH', # not available
   # 'BASIS_SET TZVP-MOLOPT-GTH-q5',
   'POTENTIAL GTH-'+pseudo+'-q11'
  ])

  stuff(pars,["FORCE_EVAL","SUBSYS","KIND Pt"],"inp",[
   bsline,
   # 'BASIS_SET DZVP-MOLOPT-SR-GTH',
   # 'BASIS_SET TZVP-MOLOPT-SR-GTH', # not available
   # 'BASIS_SET TZVP-MOLOPT-GTH-q5',
   'POTENTIAL GTH-'+pseudo+'-q18'
  ])

  stuff(pars,["FORCE_EVAL","SUBSYS","KIND Na"],"inp",[
   bsline,
   # 'BASIS_SET DZVP-MOLOPT-SR-GTH',
   # 'BASIS_SET TZVP-MOLOPT-SR-GTH', # not available
   # 'BASIS_SET TZVP-MOLOPT-GTH-q5',
   'POTENTIAL GTH-'+pseudo+'-q9'
  ])


def toggle_vdw(pars,wrk,xcf="PBE"):
  stuff(pars,["FORCE_EVAL","DFT","XC","VDW_POTENTIAL"],"inp",[
  # "DISPERSION_FUNCTIONAL PAIR_POTENTIAL"
    "POTENTIAL_TYPE PAIR_POTENTIAL"
    ])
  stuff(pars,["FORCE_EVAL","DFT","XC","VDW_POTENTIAL","PAIR_POTENTIAL"],"inp",[
    'TYPE DFTD3',
    'REFERENCE_FUNCTIONAL '+xcf, # or "pseudo" ?
    # 'CALCULATE_C9_TERM .TRUE.',
    'PARAMETER_FILE_NAME '+wrk+'/cp2k/libs/QS/dftd3.dat',
    'R_CUTOFF 15.0',
    'VERBOSE_OUTPUT'
    ])



def atoms2cpk(pars, ats, maspec=[], velocities=False, poisson="mt", mmpoisson=None,mols=None):
  # maspec: [['H',2.0],['O',..]]
  # mols: {"H2O":range(0,N)}
  lines=[]
  lines2=[]
  c=0
  
  # velocities available?
  
  vels=copy.copy(velocities)
  
  # print ats.get_momenta()
  # stop
  
  if (ats[0].get_momentum()==None):
    vels=False
    
  # mom=ats[0].get_momentum()
  # print "mom=",mom
  # stop
  
  # print "vels=",vels
  # stop
  
  # check if any FixAtoms constraints are present, and pass the info to cp2k accordingly..
  stt=""
  fx=[]
  counter = 0
  for constraint in ats.constraints:
    if (constraint.__class__.__name__=="FixAtoms"):
      c=0
      for i in range(constraint.index.shape[0]):
	if (constraint.index[i]==False):
	  pass
	elif (constraint.index[i]==True):
	  ind=c # so this is a mask..
	  stt += str(ind+1)+" "
	  fx.append(copy.copy(ind))
	  counter += 1
	else:
	  ind=constraint.index[i] # simply an index..
	  if counter % 10 == 0 and counter != 0:
	    stt += "\nLIST "
	  stt += str(ind+1)+" "
	  fx.append(copy.copy(ind))
	  counter += 1
	c=c+1
    
    
  if (stt!=""):
    set_input(pars,["MOTION","CONSTRAINT","FIXED_ATOMS"],"LIST",stt)
    set_input(pars,["MOTION","CONSTRAINT","FIXED_ATOMS"],"COMPONENTS_TO_FIX","XYZ")

    stuff(pars,["MOTION","GEO_OPT","BFGS"],"inp",[
        "USE_MODEL_HESSIAN FALSE",
        "USE_RAT_FUN_OPT TRUE",
        "TRUST_RADIUS 0.1"
    ])
  
  atcount=0
  for at in ats: # iterate over individual atoms
    spec=at.get_symbol()
    xyzs=at.get_position()
    mom=at.get_momentum()
    mass=at.get_mass()
    
    if (maspec!=[]):
      for mas in maspec:
	if (mas[0]==spec):
	  mass=mas[1]
    
    molst=""
    if (mols!=None):
      for key in mols.iterkeys():
	if ((atcount+1 in mols[key]) and (atcount not in fx)):
	  molst=" "+copy.copy(key)
	  
    lines.append(spec+" "+str(xyzs[0])+" "+str(xyzs[1])+" "+str(xyzs[2])+molst) # coordinates: ["FORCE_EVAL"]["SUBSYS"]["COORD"]["inp"]
    
    if (vels):
      vel=mom/mass
      vel=vel*(1.0/velocityfac) # from ase units to cp2k units
      lines2.append(str(vel[0])+" "+str(vel[1])+" "+str(vel[2])) # velocities: ["FORCE_EVAL"]["SUBSYS"]["VELOCITY"]["inp"]
  
    atcount=atcount+1
  
  cells=[]
  poissons=[]
  
  pbc=ats.get_pbc()
  # PERIODIC (X|Y|Z|XY|XZ|YZ|XYZ|NONE)
  # print "Atoms.pbc=",pbc # .. this is an array variable
  # stop
  
  if (list(pbc)==[False,False,False]):
    cells.append("PERIODIC NONE")
    poissons.append("PERIODIC NONE")
    # poissons.append("POISSON_SOLVER PERIODIC") # one iteration (with "20.1482036252 18.7296909273 13.9771498079" cell) 1.7 s. (16 pros)
    if (poisson=="wavelet"):
      poissons.append("POISSON_SOLVER WAVELET") # one iteration (with "ABC    20.1482036252 20.1482036252 20.1482036252" cell) 4.5 s (!)  (16 pros)
      # .. on the other hand, with 5 ang 32 prosessors, only 2.4 s.
      # .. with 7 and 64 pros. 2.2 s.
    if (poisson=="mt"):
      poissons.append("POISSON_SOLVER MT") # one iteration (with "20.1482036252 18.7296909273 13.9771498079" cell) 2.1 s.  needs 7-8 scf steps per geometry step  (16 pros)
      # .. mt solver with 9 ang vacuum.. 3.8 s per electronic step (and not yet even converged)
      # .. mt solver with 11 ang vacuum .. 4.0 s per electronic step .. wtf.. with 32 pros. !
      # poissons.append("POISSON_SOLVER ANALYTIC") # one iteration (with "20.1482036252 18.7296909273 13.9771498079" cell) 2.1 s. needs more than 20 scf steps per geometry step 
      # .. and ultimately does not even converge..
      #.. do not use!
      # (16 pros)
      # epp particle with gpaw: 32 pros., one electronic iteration 5-7 s..  one geometry step ~ 14 electronic iterations
      #
      # 64 pros., 11 ang vacuum, 2.3 s per electronic step .. scales ok..!
  elif (list(pbc)==[True,True,True]):
    cells.append("PERIODIC XYZ")
    poissons.append("PERIODIC XYZ")
    poissons.append("POISSON_SOLVER PERIODIC")
  elif (list(pbc)==[True,True,False]):
    cells.append("PERIODIC XY")
    poissons.append("PERIODIC XY")
    # poissons.append("POISSON_SOLVER PERIODIC")
    poissons.append("POISSON_SOLVER MT") # or here we should have wavelets?
  elif (list(pbc)==[True,False,False]):
    cells.append("PERIODIC X")
    poissons.append("PERIODIC X")
    # poissons.append("POISSON_SOLVER PERIODIC")
    poissons.append("POISSON_SOLVER MT") # or here we should have wavelets?
  elif (list(pbc)==[False,True,False]):
    cells.append("PERIODIC Y")
    poissons.append("PERIODIC Y")
    # poissons.append("POISSON_SOLVER PERIODIC")
    poissons.append("POISSON_SOLVER MT") # or here we should have wavelets?
  elif (list(pbc)==[False,False,True]):
    cells.append("PERIODIC Z")
    poissons.append("PERIODIC Z")
    # poissons.append("POISSON_SOLVER PERIODIC")
    poissons.append("POISSON_SOLVER MT") # or here we should have wavelets?
  else:
    print "no pbcs defined !"
    stop
     	
  cell=ats.get_cell()
  cells.append("ABC    "+str(cell[0][0])+" "+str(cell[1][1])+" "+str(cell[2][2])) # ["FORCE_EVAL"]["SUBSYS"]["CELL"]["inp"]
  
  # return lines # velocities: ["FORCE_EVAL"]["SUBSYS"]["VELOCITY"]["inp"]
  # http://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/SUBSYS/VELOCITY.html # Ang/(atomic unit time) ..?
  
  # print "pars=",pars
  
  stuff(pars,["FORCE_EVAL","SUBSYS","COORD"],"inp",lines)
  if (vels):
    stuff(pars,["FORCE_EVAL","SUBSYS","VELOCITY"],"inp",lines2)
    # if this section is written into the input file, velocity initialization by temperature
    # is ignored
    # UNITS!
  stuff(pars,["FORCE_EVAL","SUBSYS","CELL"],"inp",cells)
  # stuff(pars,["FORCE_EVAL","MM","POISSON"],"inp",poissons) # wrong place!
  stuff(pars,["FORCE_EVAL","DFT","POISSON"],"inp",poissons)
  
  if (mmpoisson!=None):
    if (mmpoisson=="ewald"):
      stuff(pars,["FORCE_EVAL","MM","POISSON","EWALD"],"inp",[
	"EWALD_TYPE ewald",
	"ALPHA .5",
        "GMAX 21"
        ])

def total_dipole_moment(pars, filename):
  section = ["FORCE_EVAL", "DFT", "PRINT", "MOMENTS"]
  #sec0s=[

  #  ["FORCE_EVAL","DFT","LOCALIZE","PRINT"],
  #  ["FORCE_EVAL","DFT","PRINT","LOCALIZATION"]
  #  ]
  stuff(pars, section, "inp", ["FILENAME ="+filename+"_tdip.dat", "REFERENCE COAC"])
  #for sec0 in sec0s:
  #  sec=sec0+["TOTAL_DIPOLE"]
  #  stuff(pars,sec,"inp",[
  #    "REFERENCE ZERO",
  #    "FILENAME ="+filename+"_tdip.dat"
  #    ])

def wannier_centers(pars, filename):
  section = ["FORCE_EVAL", "DFT", "LOCALIZE", "PRINT", "WANNIER_CENTERS"]
  stuff(pars, section, "inp", ["FILENAME ="+filename+"_wan.dat"])



def total_density(pars, filename):

  section = ["FORCE_EVAL","DFT","PRINT","TOT_DENSITY_CUBE"]
  stuff(pars, section, "inp",[
      "FILENAME ="+filename+".cube"
      ])
  # set restart off
    
  section = ["FORCE_EVAL","DFT","SCF", "PRINT"]
  #del_input(pars, section, "RESTART")
  section = ["FORCE_EVAL", "DFT", "SCF"]
  set_input(pars, section, "SCF_GUESS", "ATOMIC")
 
 
def electron_density(pars, filename):

  section = ["FORCE_EVAL","DFT","PRINT","E_DENSITY_CUBE"]
  stuff(pars, section, "inp",[
      "FILENAME ="+filename+".cube"
      ])
  # set restart off
    
  section = ["FORCE_EVAL","DFT","SCF", "PRINT"]
  #del_input(pars, section, "RESTART")
  section = ["FORCE_EVAL", "DFT", "SCF"]
  set_input(pars, section, "SCF_GUESS", "ATOMIC")
  

def electrostatic_potential(pars, filename):
  """
      Electrostatic potetial generated by the total density.
  """
  section = ["FORCE_EVAL","DFT","PRINT","V_HARTREE_CUBE"]
  stuff(pars, section, "inp",["FILENAME ="+filename+"_ht.cube"])
  

def molecular_dipole_moments(pars, filename):
  sec0s=[
    ["FORCE_EVAL","DFT","LOCALIZE","PRINT"],
    ["FORCE_EVAL","DFT","PRINT","LOCALIZATION"]
    ]

  for sec0 in sec0s:
    sec=sec0+["TOTAL_DIPOLE"]
    stuff(pars,sec,"inp",[
      "REFERENCE ZERO",
      "FILENAME ="+filename+"_tdip.dat"
      ])
    
    sec=sec0+["MOLECULAR_DIPOLES"]
    stuff(pars,sec,"inp",[
      "REFERENCE ZERO",
      "FILENAME ="+filename+"_mdip.dat"
      ])
    
  
def traj2cpk(ats,maspec=[],mols=None,names=True):
  atcount=0
  lines=[]
  for at in ats: # iterate over individual atoms
    spec=at.get_symbol()
    xyzs=at.get_position()
    mom=at.get_momentum()
    mass=at.get_mass()
    
    if (maspec!=[]):
      for mas in maspec:
	if (mas[0]==spec):
	  mass=mas[1]
    
    molst=""
    if (mols!=None):
      for key in mols.iterkeys():
	if (atcount in mols[key]):
	  molst=" "+copy.copy(key)
	  
    if (names):
      lines.append(spec+" "+str(xyzs[0])+" "+str(xyzs[1])+" "+str(xyzs[2])+molst) # coordinates: ["FORCE_EVAL"]["SUBSYS"]["COORD"]["inp"]
    else:
      lines.append(str(xyzs[0])+" "+str(xyzs[1])+" "+str(xyzs[2])+molst) # coordinates: ["FORCE_EVAL"]["SUBSYS"]["COORD"]["inp"]

  return lines


def atoms2neb(pars,imgs,pros=0,sw=0,fixed=[],opt="diis"):
  # imgs: a list of atomS object..
  
  # els=len(imgs)-2 # excluding start and end point
  els=len(imgs)
  
  # optimally, number of mpi tasks is: NUMBER_OF_REPLICA*NPROC_REP .. pros=els*nproc_rep
  # .. quite stupidly.. the start and end points also occupy mpi processes for nothing..
  # .. also the problem is that for the 20 water molecule system, 32 seems to be the
  # minimum number of processors when there is enough memory..
  # 32*10 = 320 (!)
  # (32/2)*10 = 160 ..
  # 64/4
  # images.. 8,16
  
  # nproc_rep=pros/els # optimal distribution of processors .. 
  # nproc_rep=nproc_rep/prosdiv # .. processes will cycle over the images..
  #if (nproc_rep=0):
  #   nproc_rep=1
    
  if (sw==0):
    nproc_rep=pros
  elif (sw==1):
    nproc_rep=1
  elif (sw==2):
    nproc_rep=pros/els
  
  set_input(pars,["GLOBAL"],"RUN_TYPE","BAND")
  
  if (fixed!=[]):
    stt=""
    for f in fixed:
      stt=stt+str(f)+" "
    set_input(pars,["MOTION","CONSTRAINT","FIXED_ATOMS"],"LIST",stt)
  
  sec=["MOTION","BAND"]
  
  stuff(pars,sec,"inp",[
    "NPROC_REP "+str(nproc_rep), # processors per replica
    # "BAND_TYPE CI-NEB",
    "BAND_TYPE IT-NEB",
    "K_SPRING 0.2",
    "ROTATE_FRAMES T",
    "NUMBER_OF_REPLICA "+str(els) # if this is bigger than the number of frames you have defined, more images will be created by interpolation..
    ])
      
  # stuff(pars,sec+["CI_NEB"],"inp",["NSTEPS_IT  5"])
   
  stuff(pars,sec+["CONVERGENCE_CONTROL"],"inp",[
    "MAX_FORCE 0.001",
    "RMS_FORCE 0.0005"
    ])
  
  stuff(pars,sec+["CONVERGENCE_INFO HIGH"],"inp",[
    "FILENAME =nebconv.out",
    "COMMON_ITERATION_LEVELS 1000000",
    ])
    
  if (opt=="diis"): # ********* DIIS **************
    stuff(pars,sec+["OPTIMIZE_BAND"],"inp",[
      "OPTIMIZE_END_POINTS F",
      "OPT_TYPE DIIS",
      ])
    
    stuff(pars,sec+["OPTIMIZE_BAND","DIIS"],"inp",[
      "MAX_STEPS 100",
      "N_DIIS 7",
      "NO_LS",
      "STEPSIZE 0.5",
      "MAX_STEPSIZE 1.0"
      ])
  else: # ************** MD *******************
    stuff(pars,sec+["OPTIMIZE_BAND"],"inp",[
      "OPTIMIZE_END_POINTS F",
      "OPT_TYPE MD"
      ])
    
    stuff(pars,sec+["OPTIMIZE_BAND","MD"],"inp",[
      "TIMESTEP 1.0",
      "TEMPERATURE 200"
      ])
    
    stuff(pars,sec+["OPTIMIZE_BAND","MD","TEMP_CONTROL"],"inp",[
      "TEMPERATURE 0" # target temp
      # TEMP_TOL
      # TEMP_TOL_STEPS
      ])
    
    # stuff(pars,sec+["OPTIMIZE_BAND","MD","VEL_CONTROL"],"inp",[
    #  ])
    
     
  imgn=1
  for ats in imgs:
    # each interation = image geometry
    lines=traj2cpk(ats,names=False)
    stuff(pars,sec+["REPLICA "+str(imgn),"COORD"],"inp",copy.deepcopy(lines))
    imgn=imgn+1



def dumpdens(pars):
  stuff(pars,["FORCE_EVAL","DFT","PRINT","E_DENSITY_CUBE"],"inp",[
    # "FILENAME ./cp2k.cube"
    "STRIDE 1"
    ])
  

def sic(pars,method=""): # method AD .. does not work
  # set_input(pars,["FORCE_EVAL","DFT"],"SPIN_POLARIZED","") # spin polarized
  
  # obviously, for the following you need a system with an unpaired electron ..
  set_input(pars,["FORCE_EVAL","DFT"],"ROKS","") # spin polarized, but resticted
  set_input(pars,["FORCE_EVAL","DFT","SCF","OT"],"ROTATION","") 
  
  stuff(pars,["FORCE_EVAL","DFT","SIC"],"inp",[
    "SIC_METHOD "+str(method),
    "ORBITAL_SET ALL"
    ])


def libxc(pars,xc="XC_MGGA_XC_M06_L",cc="XC_MGGA_C_M06_L"):
  sec=["FORCE_EVAL","DFT","XC"]
  del_input(pars,sec,"XC_FUNCTIONAL")
  stuff(pars,["FORCE_EVAL","DFT","XC","XC_FUNCTIONAL","LIBXC"],"inp",[
    "FUNCTIONAL "+xc+" "+cc
    ])


"""def metals(pars):
  sec=["FORCE_EVAL","DFT","SCF"]
  stuff(pars,sec,"inp",[
  "EPS_SCF 3.0E-8",
  # "MAX_SCF 250", # too much ..
  "MAX_SCF 150",
  "SCF_GUESS ATOMIC"
  ])
  
  del_input(pars,sec,"OUTER_SCF")
  
  stuff(pars,sec+["MIXING"],"inp",[
  "ALPHA 0.08",
  "METHOD PULAY_MIXING",
  "NBUFFER 9"
  ])
  
  del_input(pars,sec,"OT")
  
  stuff(pars,sec+["SMEAR"],"inp",[
  "METHOD FERMI_DIRAC"
      ])
  
  set_input(pars,sec,"ADDED_MOS","20")
"""

def metals(pars):
  sec=["FORCE_EVAL","DFT","SCF"]
  stuff(pars,sec,"inp",[
  "EPS_SCF 3.0E-8",
  # "MAX_SCF 250", # too much ..
  "MAX_SCF 150",
  "SCF_GUESS ATOMIC"
  ])
  
  del_input(pars,sec,"OUTER_SCF")
  
  stuff(pars,sec+["MIXING"],"inp",[
  "METHOD BROYDEN_MIXING",
  "ALPHA 0.1",
  "BETA 1.5",
  "NBROYDEN 8"
  #"NBUFFER 9"
  ])
  
  del_input(pars,sec,"OT")
  
  stuff(pars,sec+["SMEAR"],"inp",[
  "METHOD FERMI_DIRAC"
      ])
  
  set_input(pars,sec,"ADDED_MOS","50")

def mulliken(pars, filename = 'mulliken'):
  sec=["FORCE_EVAL","DFT","PRINT","MULLIKEN"]
  stuff(pars,sec,"inp",[
  "FILENAME = "+filename
  ]) 


def metals2(pars):
  # use metals(pars) .. it works!
  smear=False
  pulay=True
  
  # can we converge metallic systems with the orbitals transformation?
  sec=["FORCE_EVAL","DFT","SCF"]
  stuff(pars,sec,"inp",[
  "EPS_SCF 3.0E-8",
  # "MAX_SCF 250", # too much ..
  "MAX_SCF 60",
  "SCF_GUESS ATOMIC"
  ])
  
  if (pulay):
    stuff(pars,sec+["MIXING"],"inp",[
    # "ALPHA 0.1", # with OT, does not converge
    "ALPHA 0.05",
    "METHOD PULAY_MIXING",
    "NBUFFER 7"
    ])
  else:
    stuff(pars,sec+["MIXING"],"inp",[
    "ALPHA 0.1",
    ])
    # does not converge with OT
    
  stuff(pars,sec+["OT"],"inp",[
  "MINIMIZER CG",
  "PRECONDITIONER FULL_ALL",
  "LINE_SEARCH 3PNT",
  "ENERGY_GAP 0.5"
      ])
  
  # del_input(pars,sec,"OUTER_SCF")
  
  if (smear):
    set_input(pars,sec,"ADDED_MOS","20")
    stuff(pars,sec+["SMEAR"],"inp",[
    "METHOD FERMI_DIRAC"
	])
    # ERROR: CP2K| OT with ADDED_MOS/=0 not implemented

  
def pint(pars,pi_pars,pros=0,sw=0):
  # pi_pars: dictionary with
  # beads
  # steps
  # temp
  # dt
  # nrespa
  # pi_pars={"beads":4, "steps":10, "temp":50, "dt":0.5, "nrespa": 2}
  
  els=pi_pars["beads"]
  nproc_rep=1
  # see "atoms2beads"
  if (sw==0):
    nproc_rep=pros
  elif (sw==1):
    nproc_rep=1
  elif (sw==2):
    nproc_rep=pros/els
  
  set_input(pars,["GLOBAL"],"RUN_TYPE","PINT")
  stuff(pars,["MOTION","PINT"],"inp",[
    "P "+str(pi_pars["beads"]), # number of beads to use ..
    "PROC_PER_REPLICA "+str(nproc_rep),
    "NUM_STEPS "+str(pi_pars["steps"]), # number of ? steps in h2o example = 10
    "TEMP "+str(pi_pars["temp"]),
    "DT "+str(pi_pars["dt"]), # ok for hydrogen
    "NRESPA "+str(pi_pars["nrespa"]), # number of respa steps for each bead for each md .. default 5
    "TRANSFORMATION NORMAL" # transformation ? .. normal or stage
    ])
    
  stuff(pars,["MOTION","PINT","NOSE"],"inp",[
    "NNOS 3"
    ])

  
def atoms2beads(pars,imgs,pros=0,sw=0,fixed=[]):
  # TODO: not tested yet ..
  # imgs: a list of atomS object..
  
  # els=len(imgs)-2 # excluding start and end point
  els=len(imgs)
  
  # optimally, number of mpi tasks is: NUMBER_OF_REPLICA*NPROC_REP .. pros=els*nproc_rep
  # .. quite stupidly.. the start and end points also occupy mpi processes for nothing..
  # .. also the problem is that for the 20 water molecule system, 32 seems to be the
  # minimum number of processors when there is enough memory..
  # 32*10 = 320 (!)
  # (32/2)*10 = 160 ..
  # 64/4
  # images.. 8,16
  
  # nproc_rep=pros/els # optimal distribution of processors .. 
  # nproc_rep=nproc_rep/prosdiv # .. processes will cycle over the images..
  #if (nproc_rep=0):
  #   nproc_rep=1
    
  if (sw==0):
    nproc_rep=pros
  elif (sw==1):
    nproc_rep=1
  elif (sw==2):
    nproc_rep=pros/els
  
  if (fixed!=[]):
    stt=""
    for i, f in enumerate(fixed):
      stt=stt+str(f)+" "
      if i % 10 == 0:
        stt += "\n LIST"
    set_input(pars,["MOTION","CONSTRAINT","FIXED_ATOMS"],"LIST",stt)
    
    set_input(pars,["MOTION","CONSTRAINT","FIXED_ATOMS"],"COMPONENTS_TO_FIX","XYZ")

    stuff(pars,["MOTION","CONSTRAINT","BFGS"],"inp",[
        "USE_MODEL_HESSIAN FALSE",
        "USE_RAT_FUN_OPT TRUE",
        "TRUST_RADIUS 0.1",
        "@if ${HESSIAN} == 1",
        "RESTART_HESSIAN",
        "RESTART_FILE_NAME ${PROJECT}-BFGS.Hessian",
        "@endif"
    ])
  
  set_input(pars,["MOTION","PINT"],"P",str(els))
  set_input(pars,["MOTION","PINT"],"PROC_PER_REPLICA",str(nproc_rep))
     
  sec=["MOTION","PINT","BEADS"]
  imgn=1
  for ats in imgs:
    # each interation = image geometry
    lines=traj2cpk(ats,names=False)
    stuff(pars,sec+["COORD "+str(imgn)],"inp",copy.deepcopy(lines))
    imgn=imgn+1



def dftb(pars,di,adpars={}):
  # directory di must have..
  # scc/
  # scc/scc_parameter
  # uff_table
  # (nonscc/)
  #
  # /wrk/sriikone//cp2k/libs/dftb
  # = wrk+"/cp2k/libs/dftb"
  
  sec=["FORCE_EVAL","DFT","QS"]
  set_input(pars,sec,"METHOD","DFTB")
  
  stuff(pars,sec+["DFTB"],"inp",[
    "SELF_CONSISTENT T",
    "DO_EWALD T",
    "DISPERSION T"
    ])
  
  for key in adpars.keys():
    # whatever additional parameters..
    set_input(pars,sec+["DFTB"],key,str(adpars[key]))
  
  stuff(pars,sec+["DFTB","PARAMETER"],"inp",[
    "PARAM_FILE_PATH "+di+"/scc",
    "PARAM_FILE_NAME scc_parameter",
    "UFF_FORCE_FIELD uff_table"
    ])
   
  sec=["FORCE_EVAL","DFT","SCF"]
  
  # clean whatever other scf parameters..
  del_input(pars,sec,"OT")
  del_input(pars,sec,"OUTER_SCF")
  
  stuff(pars,sec,"inp",[
   "SCF_GUESS CORE",
   "MAX_SCF 20"
   ])
  
  sec=["FORCE_EVAL","DFT","SCF","MIXING"]
  stuff(pars,sec,"inp",[
   "METHOD DIRECT_P_MIXING",
   "ALPHA 0.2"
   ])
  
  sec=["FORCE_EVAL","DFT","POISSON","EWALD"]
  stuff(pars,sec,"inp",[
   "EWALD_TYPE SPME",
   "GMAX 25"
  ])
   


def smear(pars,temp):
  sec=["FORCE_EVAL","DFT","SCF","SMEAR ON"]
  stuff(pars,sec,"inp",[
  "METHOD FERMI_DIRAC"
  ])
  sec=["FORCE_EVAL","DFT","SCF"]
  set_input(pars,sec,"ADDED_MOS","20")
  # remove orbital transformation..
  del_input(pars,sec,"OT")



def ensemble(pars, etype, massivethermo):
  # .. a good account of the N(V/P)T parameters in general: schmidt99 (JPC B 113 11959)
  if (etype in ["NVT","NPT_F","NPT_I"]):
    stuff(pars,["MOTION","MD","THERMOSTAT"],"inp",[
    'TYPE NOSE'
    ])
    stuff(pars,["MOTION","MD","THERMOSTAT","NOSE"],"inp",[
    'LENGTH 3', # three thermostats in a chain..
    'YOSHIDA 3', 
    # 'TIMECON  300.0', # crap.. thanks Kari!
    'TIMECON  50.0', # value 50 tipped by Audrey .. depends .. schmidt99: 16.68 fs (i.e., 2000 1/cm)
    'MTS 2' 
    ])

    if (massivethermo):
      set_input(pars,["MOTION","MD","THERMOSTAT"],"REGION","MASSIVE")
      
  if (etype in ["NPT_F","NPT_I"]):
    stuff(pars,["MOTION","MD","BAROSTAT"],"inp",[
    "PRESSURE 1", # 1 bar
    "TIMECON 300"
    ])
    set_input(pars,["FORCE_EVAL","DFT","MGRID"],"CUTOFF","600")
    print "barostat not up to date..!"
    # .. cell_ref 
    stop
    
    
def bigcellref(pars,cellpar):
    abc = "ABC"
    if type(cellpar) == list or type(cellpar) == np.ndarray:
        if type(cellpar[0]) == list or type(cellpar[0]) == np.ndarray:
            out = []
            for i in range(3):
                out.append(abc[i]+" "+str(cellpar[i][0])+" "+str(cellpar[i][1])+" "+str(cellpar[i][2])) 
            stuff(pars,["FORCE_EVAL","SUBSYS","CELL","CELL_REF"],"inp", out)
        else:
            stuff(pars,["FORCE_EVAL","SUBSYS","CELL","CELL_REF"],"inp",["ABC "+str(cellpar[0])+" "+str(cellpar[1])+" "+str(cellpar[2])])

    set_input(pars,["FORCE_EVAL","DFT","MGRID"],"CUTOFF","600")
    # about reference cells, look for example:  https://groups.google.com/forum/#!topic/cp2k/RqPRsCJE-vI
      
  
def hfx(pars,di=""):
  # from li-hybrid-cam-b3lyp: (1)
  # H2O-hybrid-pbe0.inp (2)
  
  cutoff=2.0 # following guidon, hutter, vandevondele 2009
  
  # sec=["ATOM","METHOD","XC","HF"] # .. wrong place! (only for atomic, i.e. radial wfs calculations)
  sec=["FORCE_EVAL","DFT","XC","HF"]
  set_input(pars,sec,"FRACTION","0.25") # (2)
  # set_input(pars,sec+["MEMORY"],"MAX_MEMORY","5") # (2)
  # set_input(pars,sec+["MEMORY"],"MAX_MEMORY","100") # (1)
  stuff(pars,sec+["MEMORY"],"inp",["MAX_MEMORY 900","EPS_STORAGE_SCALING 0.1"]) # (3)
  # .. in louhi max memory should be 1024 MB
  
  stuff(pars,sec+["INTERACTION_POTENTIAL"],"inp",[
      # "POTENTIAL_TYPE MIX_CL","OMEGA 0.33","SCALE_LONGRANGE 0.94979","SCALE_COULOMB 0.18352" # (1)
      "CUTOFF_RADIUS "+str(cutoff),"POTENTIAl_TYPE TRUNCATED", # (3)
      "T_C_G_DATA "+di+"/t_c_g.dat" # (3)
    ])
  stuff(pars,sec+["SCREENING"],"inp",[
      "EPS_SCHWARZ 1.0E-10", # (1), (2), (3)
      # "SCREEN_ON_INITIAL_P FALSE" # (3)
      "SCREEN_ON_INITIAL_P TRUE", # otherwise first scf step takes ages!
      "SCREEN_P_FORCES TRUE" # faster, faster..!
    ])
    
  sec=["FORCE_EVAL","DFT","XC"]
  del_input(pars,sec,"XC_FUNCTIONAL")
  
  stuff(pars,sec+["XC_FUNCTIONAL PBE","PBE"],"inp",[
    "SCALE_X 0.75","SCALE_C 1.0" # (2),(3)
    ])
  
  stuff(pars,sec+["XC_FUNCTIONAL PBE","PBE_HOLE_T_C_LR"],"inp",[
    "CUTOFF_RADIUS 2.0","SCALE_X 0.25" # (3)
    ])  
  # CH3-PBE0_TC_LRC.inp: (3)
  
 

def fixbl(pars,traj,nn):
  # define collective variable
  # CP2K_INPUT / FORCE_EVAL / SUBSYS / COLVAR / DISTANCE
  n1=nn[0]; n2=nn[1];
  dist=np.linalg.norm(traj[n1].get_position()-traj[n2].get_position())
  # print "dist=",dist
  # stop
  dist=dist/ase.units.Bohr
  
  sec=["FORCE_EVAL","SUBSYS","COLVAR 1","DISTANCE"]
  stuff(pars,sec,"inp",[
    "ATOMS "+str(n1+1)+" "+str(n2+1)
    ])
  
  # CP2K_INPUT / MOTION / CONSTRAINT / COLLECTIVE
  sec=["MOTION","CONSTRAINT","COLLECTIVE"]
  stuff(pars,sec,"inp",[
    "COLVAR 1",
    "TARGET "+str(dist),
    "INTERMOLECULAR .TRUE."
    ])
  

def metadyn_prep(pars,hills=40):
  sec=["MOTION","FREE_ENERGY","METADYN"]
  stuff(pars,sec,"inp",[
    "DO_HILLS .TRUE.",
    # "NT_HILLS 40",
    "NT_HILLS "+str(hills), # every "hills" time step, add a gaussian
    "WW 1.40e-4"
    ])  
  sec=["MOTION","FREE_ENERGY","METADYN","PRINT"]
  set_input(pars,sec+["COLVAR"],"COMMON_ITERATION_LEVELS", "3")
  stuff(pars,sec+["COLVAR","EACH"],"inp",["MD 1"])
  set_input(pars,sec+["HILLS"],"COMMON_ITERATION_LEVELS","3")
  stuff(pars,sec+["HILLS","EACH"],"inp",["MD 1"])

# ehrenfest: check 
# cp2k/tests/QS/regtest-rtp/H2O_excit_emd.inp


def ehrenfest(pars):
  sec=["FORCE_EVAL","DFT","REAL_TIME_PROPAGATION"]
  stuff(pars,sec,"inp",[
  "MAX_ITER 8",
  "MAT_EXP PADE",
  "EXP_ACCURACY 1.0E-9",
  "EPS_ITER 1.0E-9",
  "PROPAGATOR ETRS",
  "INITIAL_WFN SCF_WFN"
  ])
  sec=["FORCE_EVAL","DFT","PRINT","WFN_MIX"]
  set_input(pars,sec,"OVERWRITE_MOS", "")
  stuff(pars,sec+["UPDATE"],"inp",[
  "RESULT_MO_INDEX 1", # RESULT = HOMO
  "ORIG_MO_INDEX 1",
  "RESULT_SPIN_INDEX ALPHA", # MAJ. SPIN
  "RESULT_SCALE 0.0", 
  # "ORIG_MARKED_STATE 1", # the state is defined in (**)
  "ORIG_SPIN_INDEX ALPHA",
  "ORIG_SCALE 1.000000000000000000000",
  "ORIG_IS_VIRTUAL .TRUE." # sets the meaning of ORIG_MO_INDEX: 1=LUMO and ascending
  # .. otherwise 1=HOMO and descending
  ])
  #
  # y=a*y+b*x
  # RESULT = RESULT_SCALE * RESULT + ORIG_SCALE * ORIG
  # (in this case result = 1.0 * orig
  # RESULT == HOMO
  # ORIG == LUMO, or as defined with MARK_STATES (**) and ORIG_MARKED_STATE
  #
  # (**)
  # LOCALIZE, PRINT, MOLECULAR_STATES, MARK_STATES 1 2
  # .. set 
  
  set_input(pars,["GLOBAL"],"RUN_TYPE","EHRENFEST_DYN")
 

def metadyn_rmds(pars,numcolvars,imgs,rmdslis=[]):
  # atom indices: start from 1
  #sec=["MOTION","FREE_ENERGY","METADYN"]
  #stuff(pars,sec,"inp",[
    #"DO_HILLS .TRUE.",
    #"NT_HILLS 40",
    #"WW 1.40e-4"
    #])
  
  sec=["MOTION","FREE_ENERGY","METADYN","METAVAR 1"]
  stuff(pars,sec,"inp",[
    "SCALE 0.15",
    "COLVAR "+str(numcolvars)
    ])
    
  #sec=["MOTION","FREE_ENERGY","METADYN","PRINT"]
  #set_input(pars,sec+["COLVAR"],"COMMON_ITERATION_LEVELS", "3")
  #stuff(pars,sec+["COLVAR","EACH"],"inp",["MD 1"])
  #set_input(pars,sec+["HILLS"],"COMMON_ITERATION_LEVELS","3")
  #stuff(pars,sec+["HILLS","EACH"],"inp",["MD 1"])

  sec=["FORCE_EVAL","SUBSYS","COLVAR 1","RMSD"]

  atlist=""
  for li in rmdslis:
    atlist=atlist+str(li)+" "

  stuff(pars,sec,"inp",[
    "SUBSET_TYPE LIST",
    "ATOMS "+atlist,
    "ALIGN_FRAMES  T"
    ])

  # define the start and end frames..
  imgn=1
  for ats in imgs:
    # each interation = image geometry
    lines=traj2cpk(ats,names=False)
    stuff(pars,sec+["FRAME "+str(imgn),"COORD"],"inp",copy.deepcopy(lines))
    imgn=imgn+1
  
  numcolvars=numcolvars+1
  return numcolvars
 
 
def metadyn_bl(pars,numcolvars,imgs,lis):
  # atom indices: start from 1
  # cc=2
  for li in lis:
    print "li=",li
    n1=li[0]
    n2=li[1]
  
    #if (imgs!=None):
    #  bl0=lg.norm(imgs[0][n1-1].get_position()-imgs[0][n2-1].get_position())
    #  bl1=lg.norm(imgs[1][n1-1].get_position()-imgs[1][n2-1].get_position())
    #  # print bl0,bl1
    #  # stop
    #  scale=str(bl1/bl0)
    # .. THAT IS NOT THE DEFINITION OF "SCALE"
    
    cc=numcolvars
    sec=["MOTION","FREE_ENERGY","METADYN","METAVAR "+str(cc)]
    stuff(pars,sec,"inp",[
      "SCALE 0.15",
      # "SCALE "+scale, # it seems i dont understand anything..
      "COLVAR "+str(cc)
      ])
    
    sec=["FORCE_EVAL","SUBSYS","COLVAR "+str(cc),"DISTANCE"]
    stuff(pars,sec,"inp",[
      "ATOMS "+str(n1)+" "+str(n2)
      ])
    numcolvars=numcolvars+1
  
  return numcolvars
  

def vibra(pars,nproc=32):
  set_input(pars,["GLOBAL"],"RUN_TYPE","VIBRATIONAL_ANALYSIS")
  stuff(pars,["VIBRATIONAL_ANALYSIS"],"inp",[
    # "DX","0.01", # step size in bohr
    "FULLY_PERIODIC .TRUE.",
    "NPROC_REP "+str(nproc),
    "INTENSITIES .TRUE."
    ])
  stuff(pars,["VIBRATIONAL_ANALYSIS","PRINT","PROGRAM_RUN_INFO HIGH"],"inp",[
    "FILENAME =freq.out",
    "COMMON_ITERATION_LEVELS 100000"
    ])
  stuff(pars,["VIBRATIONAL_ANALYSIS","PRINT","MOLDEN_VIB HIGH"],"inp",[
    "COMMON_ITERATION_LEVELS 100000"
    ])
  

def getwaterind(geom,ind):
    inds=[ind]
    if (geom[ind][0]=="O"):
      inds=inds+[ind+1,ind+2]
    else:
      if (geom[ind+1][0]=="O"):
	inds=inds+[ind-1,ind-2]
      else:
	inds=inds+[ind-1,ind+1]
    return inds
    


def makenebtraj(first,last,trajf,refat,num):
  # import asecoords
  
  # debugging ..
  # last="/home/sampsa/nanotubes/ice_cp2k/cp2k_vdw_hcl_steal_2/last.xmol"
  # first="/home/sampsa/nanotubes/ice_cp2k/cp2k_vdw_hcl_back/last.xmol"
  # trajf="/home/sampsa/nanotubes/ice_cp2k/cp2k_vdw_hcl_back/aserun.traj"
  # refat=49
  
  traj=ase.io.read(trajf)
  print "traj =",type(traj)
  
  li=[] # list of atomS objects
  
  # eka=tryout.copy()
  # vika=tryout.copy()
  # asecoords.atoms_overwrite(eka,res["startgeom"])
  # asecoords.atoms_overwrite(vika,res["endgeom"])
  
  geoms=[]
  geoms.append(coords.readxmol(first)[-1])
  geoms.append(coords.readxmol(last)[-1])
   
  # newgeoms=coords.nebpoints(geoms,ps=[0.5],atoms=[],upto=0,parabolic=False,amp=1.0)
  newgeoms=coords.morenebpoints(geoms,num)
  
  refvec0=np.array(newgeoms[0][refat][1:4])
  
  # keep a minimum distance for some water molecules..
  keeplist=[58-1,59-1]
  movelist=getwaterind(newgeoms[0],25-1)
  watercheck=[43]
  tol=1.45
  
  ci=0
  for geom in newgeoms:
    # subgeom=coords.subcoordinates(geom)["geom"] # not this way..
    print "ci=",ci
    
    # ------------ check that water molecules are not too close to HCl --------------
    movedlist=[] # indexes corresponding to water molecules already moved..
    for cc in keeplist:
      neis=coords.find_neighbours(geom[cc],geom)
      print "some neis:",neis[1:5]
      for nei in neis[1:5]: # check some of the nearest neighbours..
	if ((nei[1] not in keeplist) and (nei[1] in movelist) and (nei[1] not in movedlist)):
	  if (nei[0]<=tol):
	    # atom geom[nei[1]] is "on the way"
	    # print nei[1]
	    # print "ci=",ci
	    print "found atom too close..:",geom[nei[1]]
	    # find atoms in this water molecule
	    inds=getwaterind(geom,nei[1])
	    movedlist=movedlist+inds
	    print "inds=",inds
	    # move inds..
	    # vec=[0,8,0] # testing..
	    # newgeoms[ci]=coords.translate(geom,vec,li=inds)
	    print "translating.."
	    vec=coords.geomdiff([geom[cc],geom[nei[1]]])
	    d=coords.norm(vec) # this is the distance
	    dl=-(d-tol)
	    vec=coords.mul(coords.unitvec(vec),dl)    
	    geom=coords.translate(geom,vec,li=inds)
	    
    # -------- check that water molecules are ok.. ----------------
    for wc in watercheck:
      inds=getwaterind(geom,wc)
      for i in range(1,2):
	vec=coords.geomdiff([geom[inds[0]],geom[inds[i]]])
	d=coords.norm(vec)
	if (d<0.9):
	  dl=-(d-1.0)
	  vec=coords.mul(coords.unitvec(vec),dl)
	  geom=coords.translate(geom,vec,li=[inds[i]])
    
    ci=ci+1
    
    # hcl=[geom[-1],geom[-2]]
    # geom=coords.adjust_water2(geom)["newgeom"] # nope!
    # geom=geom+hcl
    
    tryout=copy.deepcopy(traj)
    
    refvec=np.array(geom[refat][1:4])
    tr=refvec-refvec0
    
    tryout=asecoords.geom2atoms(geom,tryout)
    tryout.translate(-tr)
    
    li.append(tryout)  
    
  
  coords.writexmolmany("nebgeoms.xmol",asecoords.atoms2geoms(li))
  ase.io.write('neb.traj',li)
  
  # should read this as follows..
  # from ase.io.trajectory import PickleTrajectory
  # traj = PickleTrajectory("neb.traj")
  # now: traj[0], etc.
  # (n, j, nebit)=nebimages(eka,vika,nebsteps,prog,pars,filename,nebfix)
  



def getatlist(ats,typ="H"):
  cc=1
  st=""
  for at in ats:
    if (at.get_symbol()==typ):
      st=st+" "+str(cc)
    cc=cc+1
  return st
  
  
def getypes(ats):
  lis=[]
  for at in ats:
    if (at.get_symbol() not in lis):
      lis.append(at.get_symbol())
  return lis
  

def makeqmmm(pars,ats,fac):
  # modifying names in the hierarchy..
  set_input(pars,["FORCE_EVAL"],"METHOD","QMMM")
  cells=[]
  
  lis0=getypes(ats)
  
  dummy="He"
  
  cell=ats.get_cell() # original cell
  
  cell0=copy.deepcopy(cell)
  cell=cell0*fac # bigger cell ..
  
  # print "cell",cell
  # print "cell0",cell0
  
  # "subsys cell" will be big while "qm cell" will be small ... qm cell = size of the quantum system
  # set_input(pars,["FORCE_EVAL","SUBSYS","CELL"],"ABC", str(cell[0][0])+" "+str(cell[1][1])+" "+str(cell[2][2]))
  
  #fac=1.0
  #stuff(pars,["FORCE_EVAL","SUBSYS","CELL"],"inp",[
  #  "ABC    "+str(cell[0][0]*fac)+" "+str(cell[1][1]*fac)+" "+str(cell[2][2]*fac),
  #  "PERIODIC NONE"
  #  ])
  
  stuff(pars,["FORCE_EVAL","QMMM","CELL"],"inp",[
    "ABC    "+str(cell0[0][0])+" "+str(cell0[1][1])+" "+str(cell0[2][2]),
    "PERIODIC NONE"
    ])
  
  # cells.append("ABC    "+str(cell0[0][0])+" "+str(cell0[1][1])+" "+str(cell0[2][2])) # ["FORCE_EVAL"]["SUBSYS"]["CELL"]["inp"]
  # cells.append("PERIODIC NONE")
  
  # biggercell=[cell[0][0]*1.5,cell[1][1]*1.5,cell[2][2]*1.5]
  faci=0.6 # skin thickness..
  skinvec=[cell0[0][0]*faci,cell0[1][1]*faci,cell0[2][2]*faci]
  
  # auxatomvec
  auxi=[cell[0][0]*0.91/2.0,cell[1][1]*0.91/2.0,cell[2][2]*0.91/2.0]
  
  auxatom=ase.Atom(dummy,(auxi[0],auxi[1],auxi[2]))
  ats.append(auxatom)
  # auxatom=ase.Atom('Es',(skinvec[0],skinvec[1],skinvec[2]))
  # ats.append(auxatom)
  
  # stuff(pars,["FORCE_EVAL","QMMM","CELL"],"inp",cells)
  stuff(pars,["FORCE_EVAL","QMMM","WALLS"],"inp",[
    'TYPE REFLECTIVE',
    'WALL_SKIN '+str(skinvec[0])+' '+str(skinvec[1])+' '+str(skinvec[2])
    ])
  stuff(pars,["FORCE_EVAL","MM","POISSON","EWALD"],"inp",[
    'EWALD_TYPE NONE'
    ])
  
  # stuff(pars,["FORCE_EVAL","QMMM","MM_KIND "+dummy],"inp",['RADIUS 1.0'])
  # stuff(pars,["FORCE_EVAL","QMMM","MM_KIND "+"O"],"inp",['RADIUS 1.0'])
  # stuff(pars,["FORCE_EVAL","QMMM","MM_KIND "+"H"],"inp",['RADIUS 1.0'])
  
  # cp2ktools.set_input(pars,["FORCE_EVAL","QMMM"],"METHOD","QMMM"
  
  # define which part of the system is DFT ..
  
  for l in lis0:
    stuff(pars,["FORCE_EVAL","QMMM","QM_KIND "+l],"inp",['MM_INDEX '+getatlist(ats,l)])
  
  # stuff(pars,["FORCE_EVAL","QMMM","QM_KIND Es"],"inp",['MM_INDEX '+getatlist(ats,"Es")]) # not needed!
  
  lis=getypes(ats)
  
  # classical potential definitions with dummy charges
  for l in lis:
    stuff(pars,["FORCE_EVAL","MM","FORCEFIELD","CHARGE "+str(l)],"inp",[
      "ATOM "+l,
      "CHARGE 0.0"
    ])
  # have to add LJ things as well ..?
  # .. indeed:
  #for c1 in range(0,len(lis)-1):
  #  for c2 in range(c1+1,len(lis)):
  for c1 in range(0,len(lis)):
    for c2 in range(0,len(lis)):
      a1=lis[c1]
      a2=lis[c2]
      stuff(pars,["FORCE_EVAL","MM","FORCEFIELD","NONBONDED","LENNARD-JONES "+str(a1)+" "+str(a2)],"inp",[
	"ATOMS "+a1+" "+a2,
	"EPSILON 0.0",
        "SIGMA   0.0",
        "RCUT    5.0",
        "RMAX    7.0"
      ])
 
  # dummy dft definitions
  stuff(pars,["FORCE_EVAL","SUBSYS","KIND "+dummy],"inp",[
    "ELEMENT "+dummy,
    "MASS 0.0",
    "GHOST",
    "BASIS_SET NONE"
    ])
    
  stuff(pars,["MOTION","CONSTRAINT","FIXED_ATOMS"],"inp",[
    "LIST "+str(len(ats))
    ])




def dummyQ(pars,sym):
  stuff(pars,["FORCE_EVAL","MM","FORCEFIELD","CHARGE "+sym],"inp",[
    "ATOM "+sym,
    "CHARGE 0.0"
    ])
    
    
def dummypair(pars,sym1,sym2):
  stuff(pars,["FORCE_EVAL","MM","FORCEFIELD","NONBONDED","LENNARD-JONES "+sym1+" "+sym2],"inp",[
    "ATOMS "+sym1+" "+sym2,
    "EPSILON 0.0",
    "SIGMA   0.0",
    "RCUT    5.0",
    "RMAX    7.0"
      ])
  
    

def makeqmmmslab(pars,traj,di=None,qmi=[],rundir="./",cell=None):
  # cell=None: automagically find "correct" cellsize for the QS
  # cell=[..]: cell dimensions
  # cell=0: same dimensions as the mm system
  #
  # slab with the same dimensions for the mm and qm systems: 
  # modifying names in the hierarchy..
  set_input(pars,["FORCE_EVAL"],"METHOD","QMMM")
  
  set_input(pars,["FORCE_EVAL","DFT","MGRID"],"COMMENSURATE"," ") # otherwise qmmm calculations will bust!
  
  skin=None
  
  if (cell==None):
    # find extent of the QS ..
    geom=asecoords.atoms2geom(traj)
    geom=coords.picknum(geom,qmi,adone=True)
    qtraj=asecoords.geom2atoms(geom)
    # q=get_extent(qtraj)*2
    q=get_extent(qtraj)*3.0 # to be extra sure..!
    skin=(q/3.0)/2.0
  elif (cell==0):
    pass
  else:
    q=cell
    
  if (cell!=0):
    qext=str(q[0])+" "+str(q[1])+" "+str(q[2])
    print "qext=",qext
  
  if (skin!=None):
    skinstr=str(skin[0])+" "+str(skin[1])+" "+str(skin[2])
    sec=["FORCE_EVAL","QMMM","WALLS"]
    stuff(pars,sec,"inp",[
      "WALL_SKIN "+skinstr
      ])
      
  
  # ************************** QM MM *****************************
  # force_eval -> qmmm
  # defines:
  # cell, coupling between classical and quantum systems, periodicity, mm_kind X, qm_kind X
  # cell defined here is the one for the quantum system 
  # cell defined in the normal force_eval -> subsys is for the whole system ..
 
  # copy here definitions from "subsys"
  p=get_input(pars,["FORCE_EVAL","SUBSYS","CELL"],"inp")
  # print "p=",p
  stuff(pars,["FORCE_EVAL","QMMM","CELL"],"inp",copy.deepcopy(p))
 
  if (cell!=0):
    set_input(pars,["FORCE_EVAL","QMMM","CELL"],"ABC",qext)
 
  sec=["FORCE_EVAL","QMMM"]
  
  # DEBUGGING ..:
  # stuff(pars,["FORCE_EVAL","QMMM","CELL"],"inp",[
  #  # "PERIODIC XY",
  #  "ABC    5 5 5"
  #  ])
  
  
  # qm-mm electrostatic coupling..
  stuff(pars,sec,"inp",[
    "NOCENTER .FALSE.",
    "NOCENTER0 .FALSE.",
    "ECOUPL GAUSS", # strange error about interpolation
    # "ECOUPL COULOMB", # segfault
    # "ECOUPL NONE", # work, but ..
    "NOCOMPATIBILITY",
    "USE_GEEP_LIB 6"
    ])
    
  # if (di!=None):
  #  set_input(pars,sec,"MM_POTENTIAL_FILENAME",di)
  # .. does not work
    
  stuff(pars,sec+["INTERPOLATOR"],"inp",[
    "EPS_R 1.0e-14",
    "EPS_X 1.0e-14",
    "MAXITER 100"
    ])
    
    
  # --- periodicity and electrostatics for QM and QM/MM coupling ----
  stuff(pars,sec+["PERIODIC"],"inp",[
    "GMAX 0.3"
    # "NGRIDS 17 29 23"
    ])
  stuff(pars,sec+["PERIODIC","MULTIPOLE"],"inp",[
    "EWALD_PRECISION 0.001",
    "RCUT 30.0"
    ])
      
  stuff(pars,sec+["PRINT","PERIODIC_INFO"],"inp",[
    ])
    
  # stuff(pars,sec+["PRINT","POTENTIAL"],"inp",[
  #   ])
  # .. no need for that ..
    
    
  # ***************************** MM electrostatics ******************************
  sec=["FORCE_EVAL","MM","POISSON"]
  
  # p=get_input(pars,["FORCE_EVAL","DFT","POISSON"],"inp")
  # stuff(pars,sec,"inp",p) 
  # .. commenting that one helped a bit ..
  
  stuff(pars,sec+["EWALD"],"inp",[
    "EWALD_TYPE ewald",
    "ALPHA .44",
    "GMAX 111"
    ])
     
  
  if (qmi!=None):
    # ************** QM MM ***************************
    sec=["FORCE_EVAL","QMMM"]
    # read a .py library somewhere (infodic), where the following is defined:
    # qm_index=[]
    # mm_kinds=[]
    # qm_kinds=[]
    
    qm_index=copy.copy(qmi)
    
    qms=""
    #for q in qm_index:
    #  qms=qms+str(q)+" "
    
    qmkinds={}
    mmkinds=[]
    checklis=[]
    
    allsyms=[]
    qmsyms=[]
    for i in range(0,len(traj)):
      at=traj[i]
      sym=at.get_symbol(); ind=i+1
      if (sym not in allsyms):
	allsyms.append(sym)
      
      if (ind in qm_index):
	qmsyms.append(sym)
	if (qmkinds.has_key(sym)):
	  qmkinds[sym]=qmkinds[sym]+str(ind)+" "
	  checklis.append(ind)
	else:
	  qmkinds[sym]=str(ind)+" "
	  checklis.append(ind)
	qm_index=list(set(qm_index)-set([ind])) # remove index from the list
      else:
	if (sym not in mmkinds):
	  mmkinds.append(sym)
      
      
    for k in qmkinds.keys():
      print "qk=",k
      stuff(pars,sec+["QM_KIND "+k],"inp",[
	"MM_INDEX "+qmkinds[k]
	])
    
    # double-check..
    
    # lines=coords.tcl_paintatoms(checklis,"black",adone=False)
    lines=coords.tcl_licorice(checklis,ads=0)
    
    # coords.writecoords(rundir+"/qms.tcl",lines) 
    coords.writecoords("qms.tcl",lines) 
    
    #&QM_KIND NA
      #MM_INDEX 3
    #&END QM_KIND
    #&QM_KIND CL
      #MM_INDEX 2
    #&END QM_KIND
    """
    for k in mmkinds:
      print "mk=",k
      stuff(pars,sec+["MM_KIND "+k],"inp",[
	"RADIUS 1.5875316249000"
      ])
    """
    
    #&MM_KIND NA
    #  RADIUS 1.5875316249000
    #&END MM_KIND
    #&MM_KIND CL
    #  RADIUS 1.5875316249000
    #&END MM_KIND
      
    #stuff(pars,["MOTION","CONSTRAINT","FIXED_ATOMS"],"inp",[
    #  "LIST "+str(len(ats))
    #  ])

    # add dummy charges and LJ parameters to 
    # qm-only elements:
    onlyqm=list(set(allsyms)-set(mmkinds))
    for sym in onlyqm:
      dummyQ(pars,sym)
      for sym2 in allsyms:
	dummypair(pars,sym,sym2)
    
    
  
def setposvel(pars,ats,groups=None):
  # groups: {"h2o":indices,"hcl":indices2}
  
  lines=[]
  lines2=[]
  c=0
  
  vels=False # this is only for ase usage.. do not use ever..
  # velocities available? 
  if (ats[0].get_momentum()==None):
    vels=False
    
  # mom=ats[0].get_momentum()
  # print "mom=",mom
  # stop
  
  atcount=0
  for at in ats: # iterate over individual atoms
    spec=at.get_symbol()
    xyzs=at.get_position()
    mom=at.get_momentum()
    mass=at.get_mass()
    
    molst=""
    if (groups!=None):
      for key in groups.iterkeys():
	# print "key",key
	# print atcount in groups[key]
	if (atcount in groups[key]):
	  molst=" "+copy.copy(key)
	  
    # print "molst",molst
    stt=spec+" "+str(xyzs[0])+" "+str(xyzs[1])+" "+str(xyzs[2])+molst
    # print ">",stt
    lines.append(stt) # coordinates: ["FORCE_EVAL"]["SUBSYS"]["COORD"]["inp"]
    
    
    if (vels):
      vel=mom/mass
      vel=vel*(1.0/velocityfac) # from ase units to cp2k units
      lines2.append(str(vel[0])+" "+str(vel[1])+" "+str(vel[2])) # velocities: ["FORCE_EVAL"]["SUBSYS"]["VELOCITY"]["inp"]
  
    atcount=atcount+1
    
  stuff(pars,["FORCE_EVAL","SUBSYS","COORD"],"inp",lines)
  if (vels):
    stuff(pars,["FORCE_EVAL","SUBSYS","VELOCITY"],"inp",lines2)
  
 
 
def watermodel(pars,di={},constraints=True):
  # SPC/E
  
  # o_charge=-0.8476
  # h_charge=0.4238
  # angle=109.47
  
  o_charge=di["o_charge"]
  h_charge=di["h_charge"]
  angle=di["angle"]
  
  oho=((math.pi*2.0)/(360.0))*angle
  
  oh_d=1.0/ase.units.Bohr
  
  hh_d=(oh_d*math.sin(oho/2.0))*2.0
  
  stuff(pars,["FORCE_EVAL","MM","FORCEFIELD","BEND"],"inp",[
    "ATOMS H O H",
    "K 0.",
    "THETA0 "+str(oho)
    ])
    
  stuff(pars,["FORCE_EVAL","MM","FORCEFIELD","BOND"],"inp",[
    "ATOMS O H",
    "K 0.",
    "R0 "+str(oh_d)
    ])
    
  stuff(pars,["FORCE_EVAL","MM","FORCEFIELD","CHARGE O"],"inp",[
    "ATOM O",
    "CHARGE "+str(o_charge)
    ])
  
  stuff(pars,["FORCE_EVAL","MM","FORCEFIELD","CHARGE H"],"inp",[
    "ATOM H",
    "CHARGE "+str(h_charge)
    ])
    

  stuff(pars,["FORCE_EVAL","MM","FORCEFIELD","NONBONDED","LENNARD-JONES OO"],"inp",[
    "atoms O O",
    "EPSILON 78.198",
    "SIGMA 3.166",
    "RCUT 11.4"
    ])

  stuff(pars,["FORCE_EVAL","MM","FORCEFIELD","NONBONDED","LENNARD-JONES OH"],"inp",[
    "atoms O H",
    "EPSILON 0.0",
    "SIGMA 3.6705",
    "RCUT 11.4"
    ])
    
  stuff(pars,["FORCE_EVAL","MM","FORCEFIELD","NONBONDED","LENNARD-JONES HH"],"inp",[
    "atoms H H",
    "EPSILON 0.0",
    "SIGMA 3.30523",
    "RCUT 11.4"
    ])


  ## adjust geometry to water model parameters..
  #geom=
  #res=coords.adjust_water(geom,oh=1.0,oho=angle) 
  #geom=res["newgeom"]
  
  #tryout=asecoords.geom2atoms(geom) # create ase atoms object
  
  ## put rigid monomer contsraints into ase atoms object
  ## tryout.set_constraint(constraints.FixInternals(tryout, bonds=res["bonds"]))
  #tryout.set_constraint(constraints.FixBondLengths(res["pairs"]))


  if (constraints):
    stuff(pars,["MOTION","CONSTRAINT","G3X3"],"inp",[
      "DISTANCES "+str(oh_d)+" "+str(oh_d)+" "+str(hh_d),
      # "MOLECULE 1", # THIS DOES NOT ALWAYS WORK ..!
      "MOLNAME H2O", # IDIOT PROOF
      "ATOMS 1 2 3"
      ])
 
  # constraints for the model as well..
  # stuff(pars,["MOTION","CONSTRAINT","FIXED_ATOMS"],"inp",[
  #   "LIST "+str(len(ats))
  #   ])
 
# ----------- postprocessing tools ----------------
 
 
def sparseprint(pars,n):
  #  MOTION / PRINT / TRAJECTORY / EACH
  stuff(pars,["MOTION","PRINT","TRAJECTORY","EACH"],"inp",[
    "MD "+str(n)
    ])
  stuff(pars,["MOTION","PRINT","VELOCITIES","EACH"],"inp",[
    "MD "+str(n)
    ])
  stuff(pars,["MOTION","MD","PRINT ON","ENERGY","EACH"],"inp",[
    "MD "+str(n)
    ])
  
  # stuff(pars,["MOTION","PRINT","MIXED_ENERGIES","EACH"],"inp",[
  #  "MD "+str(n)
  #  ])
  # .. that does not work
  
  
  
  

def ase_watermodel(traj,qms):
  #  qms.. atoms to be excluded from the treatment
  geom=asecoords.atoms2geom(traj)
  res=coords.subcoordinates(geom)
  finalist=[]
  # lines=[]; color="grey";
  for w in res["inwater"]:
    if ((w[0]+1 not in qms) and (w[1]+1 not in qms) and (w[2]+1 not in qms)):
      # this water molecule belongs to the MM system ..
      lis=copy.copy([ [w[0],w[1]], [w[0],w[2]], [w[1],w[2]] ])
      
      """
      if (color=="grey"):
	color="black"
      else:
	color="grey"
      lines=lines+coords.tcl_paintatoms(w,color,adone=True)
      """
      
      finalist=finalist+lis
    
  traj.set_constraint(FixBondLengths(finalist))
  # coords.writecoords("waters.tcl",lines)
  
  
   
def wanpoints(xmolfile,wanfile,step=-1):
  # read in wannier centers and coordinates..
  # .. both in xyz format
  debug=False
  
  wangeom=coords.readxmol(wanfile)
  print wanfile,":",len(wangeom)
  wangeom=wangeom[step]
  
  wanpoints=[]
  for w in wangeom:
    wanpoints.append([w[1],w[2],w[3]])
  
  lis=wanpoints
  lines=coords.tcl_spheres(lis,rad=0.1,scale=1.0,color="green")
  # coords.writecoords("centers.tcl",lines[0:8])
  coords.writecoords("allcenters.tcl",lines)

  # stop

  geom=coords.readxmol(xmolfile)
  print xmolfile,":",len(geom)
  geom=geom[step]
  
  print 'check with: grep -i "1_" cp2k-HOMO_centers_s1-1.data'
  
  subgeom=coords.subcoordinates(geom)["geom"]
  # renum=len(subgeom)*4
  # print str(renum)+" relevant wannier functions, right?"
  
  newsubgeom=copy.deepcopy(subgeom)
  for g in newsubgeom:
    g[4]["wfcs"]=[]
  
  # print newsubgeom
  # stop
  
  nlis=[]
  # associate wfcs with a certain water molecule
  for l in lis: # [0:renum]:
    neis=coords.find_neighbours(["H",l[0],l[1],l[2]],subgeom)
    num=neis[0][1]
    g=subgeom[num]
    
    origo=np.array([0,0,0]) # this could be at the center of all points?
    # origo=np.array([1,1,1])
    # origo=np.array([1000,1000,1000])

    newsubgeom[num][4]["origo"]=copy.deepcopy(origo)
    newsubgeom[num][4]["wfcs"].append([l[0],l[1],l[2]]) # absolute values
    nlis.append([l[0],l[1],l[2]])
    
  lines=coords.tcl_spheres(nlis,rad=0.1,scale=1.0,color="green")
  coords.writecoords("centers.tcl",lines)
    
  if (debug):
    # print newsubgeom
    # debugging.. now print the centers again
    
    lis2=[]
    # newsubgeom.pop(-1)
    newsubgeom.pop(0)
    
    for g in newsubgeom:
      # origo=g[4]["origo"]
      
      for w in g[4]["wfcs"]:
	lis2.append(w)
  
    stop
    
  c=0
  olist=[]
  hlist=[]
  wlist=[]
  tlines=[]; disp=[0.5, 0.2, 0.2]
  mean=0.0
  for g in newsubgeom:
      ionsum=0
      elsum=0
    
      # origo=g[4]["origo"]
      origo=np.array([0,0,0])
      vecsum=np.array([0,0,0])
      ovec=coords.g2vec(g)
      # ovec=np.array([0,0,0]) # this way the oxygen electron contribution will vanish..! (1)
      
      # print "origo=",origo
      
      vecsum=vecsum+(ovec-origo)*6.0 # the oxygen atom (two electrons in the paw core)
      ionsum=ionsum+6
      olist.append(list(ovec-origo))
      
      for p in g[4]["sub"][1:]:
	pvec=ovec+coords.g2vec(p)
	# print "pvec=",pvec
	vecsum=vecsum+(pvec-origo)*1.0 # a proton
	ionsum=ionsum+1
	hlist.append(list(pvec-origo))
	
      for w in g[4]["wfcs"]:
	# wavec=ovec+np.array(w) # (1)
	wavec=np.array(w) 
	vecsum=vecsum+(wavec-origo)*-2.0
	elsum=elsum-2
	wlist.append(list(wavec-origo))
      
      dipmom=lg.norm(vecsum)/deb2
      
      m=list(ovec-origo)
      tlines.append("plot text3 black "+str(m[0]+disp[0])+" "+str(m[1]+disp[1])+" "+str(m[2]+disp[2])+" "+str(round(dipmom,2))+" ")
      
      print c, ionsum, elsum
      print vecsum/deb2
      print lg.norm(vecsum)/deb2
      print
      mean=mean+dipmom
      
      c=c+1
      
  print "mean:",mean/c
  lines2=coords.tcl_spheres(hlist,rad=0.1,scale=1.0,color="white")
  lines2=lines2+coords.tcl_spheres(olist,rad=0.1,scale=1.0,color="red",append=True)
  lines2=lines2+coords.tcl_spheres(wlist,rad=0.1,scale=1.0,color="black",append=True)
  coords.writecoords("oh_centers.tcl",lines2)
  coords.writecoords("dipmoms.tcl",tlines)


def get_dipmom(filename,trajfile=None,fac=1.0,scale=0.2):
  # dipole moment from cp2k output file ..
  # .. the lines we are looking for look like this:
  #
  # iter_level                  dipole(x,y,z)[atomic units]                              dipole(x,y,z)[debye]                            delta_dipole(x,y,z)[atomic units]
  # 1_0                2.73985029E-01    1.55918678E-01   -9.17715812E-02    6.96400413E-01    3.96305712E-01   -2.33260071E-01    2.73985029E-01    1.55918678E-01   -9.17715812E-02
  #
  lines=coords.read_file(filename)
  c=0
  for l in lines:
    # li=l.split()
    # if (l.find("dipole(x,y,z)[atomic units]")!=-1):
    #  nums=lines[c+1].split()
    if (l.find("ENERGY| Total FORCE_EVAL")!=-1):
      nums=lines[c-2].split()
      dip=[np.float(nums[4]),np.float(nums[5]),np.float(nums[6])]
      print dip,lg.norm(dip)
    c=c+1
    
  if (trajfile!=None):
    # attach a vector to the center of mass of the particle
    traj=ase.read(trajfile)
    cent=traj.get_center_of_mass()
   
    x1=cent[0]-dip[0]*fac/2.0
    x2=cent[0]+dip[0]*fac/2.0
    y1=cent[1]-dip[1]*fac/2.0
    y2=cent[1]+dip[1]*fac/2.0
    z1=cent[2]-dip[2]*fac/2.0
    z2=cent[2]+dip[2]*fac/2.0
  
    color="black"
    print
    st="plot arrow "+str(x1)+" "+str(y1)+" "+str(z1)+" "+str(x2)+" "+str(y2)+" "+str(z2)+" "+str(scale)+" "+color
    print st
    print
    temp=[["H",x1,y1,z1],["H",x2,y2,z2]]
    print "dipole=",temp



def icelatcon(dirname,pars,tryout,latcon):
  # import asecoords
  debug=False
  # a utility to calculate ice lattice constant
  # needs:
  #		pars:   dictionary with the cp2k parametesr
  #		tryout: ase atoms object
  # uses:
  # 		runcp2k(dirname,"cp2k.inp","cp2k.out")
  
  os.system("mkdir "+str(dirname))
  infofile=dirname+"/run.info"
  coords.writefile(infofile,["# latcon run"])
  xyz=asecoords.atoms2geom(tryout)
  cell=tryout.get_cell()
 
  # span lattice constant values
  # lcs=list(numpy.linspace(0.95,1.05,10)*latcon)
  # lcs=list(np.linspace(0.95,1.15,6)*latcon)
  lcs=list(np.linspace(2.65,2.70,6))
  
  # lcs=list(numpy.linspace(0.95,1.15,6)*2.725)
  
  cc=1
  oldlc=copy.copy(latcon)
  for lc in lcs:
    print ">",cc,lc
    # scale to some other lattice constant..
    newlc=copy.copy(lc)
    # print
    # print "present xyz",xyz
    # print
    # print "latcons:",oldlc,newlc
    nw=coords.scalewaterto({"geom":xyz, "lattice":cell},oldlc,newlc) # scale from oldlc to newlc
    xyz=nw["geom"]
    # print
    # print "new scaled xyz",xyz
    # stop
    cell=nw["lattice"]
    
    # create ase Atoms object
    tryout=asecoords.geom2atoms(xyz)
    tryout.set_cell(cell)
    tryout.set_pbc((True, True, True))

    atoms2cpk(pars,tryout,velocities=False)
    
    tdirname=dirname+"/"+str(cc)
    os.system("mkdir "+tdirname)
    ase.io.write(tdirname+"/aserun_start.xmol",tryout,format="xyz")
    lines=getlines(pars)
    writefile(tdirname+"/cp2k.inp",lines)
    ase.io.write(tdirname+"/cp2k.traj",tryout)

    if (debug==False):
      runcp2k(tdirname,"cp2k.inp","cp2k.out")
      # newatoms=ase.io.read(tdirname+"/cp2k-pos-1.xyz")
      geom=coords.readxmol(tdirname+"/cp2k-pos-1.xyz")[-1]
      
    # coords.writefile(infofile,[str(cc)+" "+str(newlc)+" "+str(nrj)],typ="a")
    coords.writefile(infofile,[str(cc)+" "+str(newlc)],typ="a")
        
    # xyz=asecoords.atoms2geom(tryout)
    # print "got geom",xyz
    
    oldlc=copy.copy(newlc)
    cc=cc+1
    



def runcp2k(di, inputfile, outputfile):
  # di = directory
  exe=os.environ["cp2kexe"] 
  comm="cd "+di+"; "+exe+" "+di+"/"+inputfile+" &>"+di+"/"+outputfile
  print comm
  os.system(comm)
    

def main():
  
  # makenebtraj()
  # stop
  
  # testmenusort()
  # stop
  
  par=sys.argv[1]

  if (par=="makeneb"):
    print "makeneb first.xmol last.xmol file.traj refat num"
    first=sys.argv[2]
    last=sys.argv[3]
    trajf=sys.argv[4]
    refat=int(sys.argv[5])
    num=int(sys.argv[6])
    makenebtraj(first,last,trajf,refat,num)
    

  if (par=="nthstep"):
    print "cp2ktools nthstep xmolfile n"
    geoms=coords.readxmol(sys.argv[2])
    # sta=0
    # end=10000000000000
    #if (len(sys.argv)>4):
    #  sta=int(sys.argv[4])
    #  end=int(sys.argv[5])
    #  print "start, end:",sta,end
    m=int(sys.argv[3])
    n=0
    # rn=0
    newgeoms=[]
    for geom in geoms:
      if (n==0):
	# newgeoms.append(geom)
	geomi=copy.deepcopy(geom)
	geomi=coords.masscentrify(geomi)
	# print coords.find_masscenter(geomi)
	newgeoms.append(geomi)
	# rn=rn+1 # makes no sense
      n=n+1
      if (n>=m):
	n=0
    coords.writexmolmany('some.xmol',newgeoms)
    
    
  if (par=="mthstep"):
    # for BIIIIG files!
    print "cp2ktools mthstep xmolfile m"
    f=open(sys.argv[2],'r')
    target=open("some.xmol",'w')
    ok=True
    c=0
    while ok:
      geom=coords.readxmolstep(f)
      if (geom==None):
	ok=False
      else:
	print str(c)+"*"+sys.argv[3]
	coords.writexmolstep(target,geom)
	coords.jumpxmolsteps(f,int(sys.argv[3]))
	c=c+1
    f.close()
    target.close()
    
    
  if (par=="xmol2bin"):
    print "cp2ktools xmol2bin xmolfile"
    f=open(sys.argv[2],'r')
    target=open("out.xbin",'wb')
    ok=True
    c=0
    while ok:
      geom=coords.readxmolstep(f)
      if (geom==None):
	ok=False
      else:
	print c
	if (c==0):
	  coords.writexmolbinhead(target,geom)
	coords.writexmolbin(target,geom)
	# coords.jumpxmolsteps(f,int(sys.argv[3]))
	c=c+1
    f.close()
    target.close()
    
    
  if (par=="bin2xmol"):
    print "cp2ktools bin2xmol file.xbin (start stop (x))"
    print "x : how many steps to skip"
    f=open(sys.argv[2],'rb')
    target=open("xbin.xmol",'w')
    ok=True
    c=0
    count=1
    (g0,bsize)=coords.readxmolbinhead(f)
    sto=1000000000000
    if (len(sys.argv)>3):
      sta=int(sys.argv[3])
      sto=int(sys.argv[4])
      c=sta
      typ=1
    if (len(sys.argv)>5):
      count=int(sys.argv[5])
    while ok:
      geom=coords.readxmolbin(f,g0)
      if (geom==None or c>sto):
	ok=False
      else:
	print c
	coords.writexmolstep(target,geom)
	# coords.jumpxmolsteps(f,int(sys.argv[3]))
	coords.jumpbinsteps(f,bsize,count-1)
	c=c+count
    f.close()
    target.close()
    print "wrote xbin.xmol"


  if (par=="binsize"):
    print "cp2ktools binsize file.xbin"
    nsteps=coords.binsteps(sys.argv[2])
    print "File has",nsteps,"MD steps"
    
    
  if (par=="binlast"):
    print "cp2ktools binlast file.xbin m"
    print "last m steps into xmol"
    print "m<0: single step"
    nsteps=int(coords.binsteps(sys.argv[2]))
    m=int(sys.argv[3])
    if (m>=0):
      sta=nsteps-m
      sto=1000000000000
    else:
      sta=abs(m)
      sto=abs(m)
      print ">",sta,sto
    f=open(sys.argv[2],'rb')
    target=open("xbin.xmol",'w')
    ok=True
    c=0
    (g0,bsize)=coords.readxmolbinhead(f)
    c=sta
    typ=1
    #if (len(sys.argv)>5):
    #typ=2
    #print "counting from file end"
    coords.jumpbinsteps(f,bsize,sta,typ=typ)
    while ok:
      geom=coords.readxmolbin(f,g0)
      if (geom==None or c>sto):
	ok=False
      else:
	print c
	coords.writexmolstep(target,geom)
	# coords.jumpxmolsteps(f,int(sys.argv[3]))
	c=c+1
    f.close()
    target.close()
    
    
  if (par=="binconcat"):
    import array as ar
    print "cp2ktools binconcat file1.xbin file2.xbin"
    source1=open(sys.argv[2],'rb')
    source2=open(sys.argv[3],'rb')
    target=open("concat.xbin",'wb')
    
    (g1,bsize1)=coords.readxmolbinhead(source1)
    (g2,bsize2)=coords.readxmolbinhead(source2)
    
    if ((g1!=g2) or (bsize1!=bsize2)):
      print "something wrong!"
    else:
      geombsize=len(g1)*8*3 # bytes
      geomsize=len(g1)*3 # coordinates
      
      coords.writexmolbinhead(target,g1)
      # ********** first file ***********
      print sys.argv[2]
      ok=True
      cc=0
      while ok:
	try:
	  xsa=ar.array('d')
	  xsa.fromfile(source1,geomsize) # read step-by-step ..
	except EOFError:
	  print "EOF"
	  print "read/wrote",cc,"steps"
	  ok=False
	if (ok):
	  target.write(xsa)
	  cc=cc+1
	  
	  
      # ********* second file ***********
      sys.argv[3]
      ok=True
      cc=0
      while ok:
	try:
	  xsa=ar.array('d')
	  xsa.fromfile(source2,geomsize) # read step-by-step ..
	except EOFError:
	  print "EOF"
	  print "read/wrote",cc,"steps"
	  ok=False
	if (ok):
	  target.write(xsa)
	  cc=cc+1
      
      source1.close()
      source2.close()
      target.close()
    
  
  if (par=="nbinstep"):
    import array as ar
    print "cp2ktools nbinstep file.xbin sta (sto) n"
    print "from step sta to step sto, skip always n steps"
    print "(you will have every n+2:th step, your timestep will be n+1)"
    source=open(sys.argv[2],'rb')
    target=open("some.xbin",'wb')
    
    (g,bsize)=coords.readxmolbinhead(source)
    coords.writexmolbinhead(target,g)
    geombsize=len(g)*8*3 # bytes
    geomsize=len(g)*3 # coordinates
    cc=0
    sta=0
    sto=100000000000000000000
    ok=True
    
    if (len(sys.argv)>5):
      sta=int(sys.argv[3])
      sto=int(sys.argv[4])
      n=int(sys.argv[5])
    else:
      n=int(sys.argv[4])
    
    print "n=",n
    
    if (sta>0):
      coords.jumpbinsteps(source,bsize,sta)
    cc=sta
    
    while ok:
      try:
	xsa=ar.array('d')
	xsa.fromfile(source,geomsize) # read step-by-step ..
      except EOFError:
	print "EOF"
	ok=False
      if (ok):
	target.write(xsa)
	print cc
	coords.jumpbinsteps(source,bsize,n)
	cc=cc+n
      if (cc>=sto):
	ok=False
  
    source.close()
    target.close()
  
  
  if (par=="binfl"):
    import array as ar
    print "cp2ktools binfl file.xbin"
    print "gives the first and last steps into some.xbin"
    
    nsteps=coords.binsteps(sys.argv[2])
    
    jumpi=int(nsteps-2)
    
    
    source=open(sys.argv[2],'rb')
    target=open("some.xbin",'wb')
    
    (g,bsize)=coords.readxmolbinhead(source)
    coords.writexmolbinhead(target,g)
    geombsize=len(g)*8*3 # bytes
    geomsize=len(g)*3 # coordinates
    ok=True
    
    while ok:
      try:
	xsa=ar.array('d')
	xsa.fromfile(source,geomsize) # read step-by-step ..
      except EOFError:
	print "EOF"
	ok=False
      if (ok):
	target.write(xsa)
	coords.jumpbinsteps(source,bsize,jumpi)
  
    source.close()
    target.close()
  
  
  
  if (par=="comfix"):
    import array as ar
    print "cp2ktools comfix file.xbin"
    print "correcting the center of mass"
    source=open(sys.argv[2],'rb')
    target=open("com.xbin",'wb')
    
    (g,bsize)=coords.readxmolbinhead(source)
    coords.writexmolbinhead(target,g)
    geomsize=len(g)*3 # coordinates
    m=len(g)
    ok=True
    cc=0
    
    olis=coords.pickspeclist(g,["O"])
    hlis=coords.pickspeclist(g,["H"])
    n_o=len(olis)
    n_h=len(hlis)
    omass=16
    hmass=1
    ww=omass*n_o+hmass*n_h
    
    while ok:
      try:
	xsa=ar.array('d')
	xsa.fromfile(source,geomsize) # read step-by-step ..
      except EOFError:
	print "EOF"
	ok=False
      if (ok):
	newt=np.frombuffer(xsa)   
	newt=np.reshape(newt,(m,3))
	
	o_newt=newt[olis,:]
	h_newt=newt[hlis,:]
	
	xx=(np.sum(o_newt[:,0]*omass)+np.sum(h_newt[:,0]*hmass))/ww
	yy=(np.sum(o_newt[:,1]*omass)+np.sum(h_newt[:,1]*hmass))/ww
	zz=(np.sum(o_newt[:,2]*omass)+np.sum(h_newt[:,2]*hmass))/ww
	
	# com=np.array([cnewt[:,0].mean(), cnewt[:,1].mean(), cnewt[:,2].mean()]) #  crappy com correction
	com=np.array([xx, yy, zz])
	
	newt=newt-com
	xsa=ar.array('d',newt.flatten().tolist())
	target.write(xsa)
	print cc
	cc=cc+1
  
    source.close()
    target.close()
  
  
  if (par=="oxylayers"):
    print "cp2ktools oxylayers xmolfile/trajfile"
    t=ase.io.read(sys.argv[2])
    print t
    geom=asecoords.atoms2geom(t)
    lays=coords.to_layers2(geom,tol=0.5,spec=["O"])
    c=0
    for lay in lays:
      print c,lay
      c=c+1
    print "lays="+str(lays)
    
    
  if (par=="limitsteps"):
    print "cp2ktools limitsteps xmolfile start end"
    geoms=coords.readxmol(sys.argv[2])
    sta=int(sys.argv[3])
    end=int(sys.argv[4])
    n=0
    newgeoms=[]
    if (end==-1):
      end=len(geoms)+1
    for geom in geoms:
      if (n<=end and n>=sta):
	newgeoms.append(geom)
	# rn=rn+1 # makes no sense
      n=n+1
    coords.writexmolmany('newsome.xmol',newgeoms)
    
  if (par=="traj2xmol"):
    traj=ase.io.read(sys.argv[2])
    ase.io.write("traj.xmol",traj,format="xyz")

  if (par=="maketraj"):
    #if (len(sys.argv)>2):
    #  st="-"+str(sys.argv[2])
    #else:
    #  st="-1"
    if (len(sys.argv)<5):
      print "cp2ktools.py maketraj trajfile posfile newtrajfile (velfile)"
    else:
      trajfile=sys.argv[2]
      posfile=sys.argv[3]
      newtrajfile=sys.argv[4]
      
      velfile=None
      if (len(sys.argv)>=6):
	print "adding also the velocities"
	velfile=sys.argv[5]
      
      traj=fixtraj(trajfile,posfile,velfile)
      print "got traj>",traj
      ase.io.write(newtrajfile,traj)
      #
      # visualizing with python..
      # traj=ase.read('cp2k_all-1.traj')
      # ase.visualize.view(traj)
 
 
  if (par=="fixtraj"):
    print "cp2ktools.py fixtraj fname X Y Z"
    trajfile=sys.argv[2]
    traj=ase.io.read(trajfile)
    x=float(sys.argv[3]); y=float(sys.argv[4]); z=float(sys.argv[5]);
    fc=np.diag(np.array([x,y,z]))
    traj.set_cell(fc)
    ase.io.write("new.traj",traj)
    
 
  if (par=="printgeom"):
    # import asecoords
    traj=ase.io.read(sys.argv[2])
    geom=asecoords.atoms2geom(traj)
    print geom
    
 
  if (par=="printvel"):
    if (len(sys.argv)>2):
      st="-"+str(sys.argv[2])
    else:
      st="-1"
    traj=ase.io.read('cp2k_all'+st+'.traj')
    lines=printvels(traj, fac=20)
    lines=lines+printcell(traj,center=True)
    coords.writecoords("velo.tcl",lines)
    
  if (par=="wanpoints"):
    # python ~/python/cp2ktools.py wanpoints cp2k-pos-1.xyz cp2k-HOMO_centers_s1-1.data
    if (len(sys.argv)<4):
      print "cp2ktools wanpoints xmolfile wanfile"
      return
    else:
      step=-1
      if (len(sys.argv)>4):
	step=int(sys.argv[4])
      wanpoints(sys.argv[2],sys.argv[3],step=step)
      
  if (par=="printcell"):
    st=str(sys.argv[2])
    traj=ase.io.read(st)
    lines=printcell(traj,center=True)
    coords.writecoords("cell.tcl",lines)
    
  if (par=="dipmoms"):
    print "python cp2ktools.py dipmoms outputfile [trajfile/xmolfile]"
    print "or did you want python wantpoints..?"
    if (len(sys.argv)<=3):
      get_dipmom(sys.argv[2])
    else:
      get_dipmom(sys.argv[2],trajfile=sys.argv[3],fac=8.0,scale=0.2) # define xmolfile

  if (par=="getextent"):
    get_extent(ase.io.read(sys.argv[2]),verbose=True)
    
    
  if (par=="nebis"):
    print "nebis images frame"
    num=int(sys.argv[2])
    frame=int(sys.argv[3])
    geoms=[]
    for n in range(0,num):
      print n+1
      st=str(n+1)
      if (len(st)<2):
	st="0"+st
      fname="cp2k-pos-Replica_nr_"+st+"-1.xyz"
      geoms.append(coords.readxmol(fname)[frame])
      
    coords.writexmolmany("nebis.xmol",geoms)
  
  if (par=="collect"):
    print "python cp2ktools.py collect file.xmol start stop"
    base=ase.io.read("cp2k.traj")
    geoms=coords.readxmol(sys.argv[2])
    sta=int(sys.argv[3])
    sto=int(sys.argv[4])
    collect=[]
    c=0
    for geom in geoms[sta:sto+1]:
      print c
      collect.append(asecoords.geom2atoms(geom,base))
      c=c+1
    ase.io.write("collect.traj",collect)
    
    
  if (par=="multiply"):
    print "python cp2ktools.py multiply cp2k.traj (frame)"
    st=str(sys.argv[2])
    frame=-1
    if (len(sys.argv)>3):
      frame=int(sys.argv[3])
    # traj=ase.io.read(st)
    traj = PickleTrajectory(st)[frame]
    bigtraj=traj.repeat((3,2,1))
    ase.io.write('big.traj',bigtraj)
    ase.io.write('big.xmol',bigtraj,format="xyz")
    # lines=printcell(traj,center=True)
    # coords.writecoords("cell.tcl",lines)
    
    
  if (par=="brutemul"):
    print "python cp2ktools.py brutemul cp2k.traj traj/xmolfile"
    st=str(sys.argv[2])
    maintraj = PickleTrajectory(st)[0]
    
    tfile=sys.argv[3]
    #if (tfile.find(".xmol")>-1 or tfile.find(".xyz")>-1):
    #  trajs=coords.readxmol(tfile)
    #  xmol=True
    #else:
    #  trajs = PickleTrajectory(tfile)
	
    geoms=coords.readxmol(tfile)
	
    newgeoms=[]
    lattice=maintraj.get_cell()
    lattice=[lattice[0]] # choose here the lattice vector you want..
    # cells=[[-1,1]]
    cells=[[-2,0]]
    
    c=0
    # print "geom:",len(geoms[0])
    bigeoms=[]
    for geom in geoms:
      print c
      bigeom=coords.expand_geom3(geom,lattice,cells)
      bigeoms.append(bigeom)
      # print "bigeom:",len(bigeom)
      c=c+1
      
    coords.writexmolmany("big.xmol",bigeoms)
    
   
  if (par=="multicut" or par=="multicutfew"):
    print "python cp2ktools.py multicut cp2k.traj origo extent (traj/xmolfile)"
    print "python ~/python/cp2ktools.py multicut final.traj 146 10 cp2k-pos-1.xyz"
    st=str(sys.argv[2])
    frame=-1
    #if (len(sys.argv)>5):
    #  frame=int(sys.argv[5])
    # traj=ase.io.read(st)
    # traj = PickleTrajectory(st)[frame]
    # trajs = PickleTrajectory(st)
    xmol=False
    
    traj = PickleTrajectory(st)[0]
    if (len(sys.argv)>5):
      tfile=sys.argv[5]
      if (tfile.find(".xmol")>-1 or tfile.find(".xyz")>-1):
	trajs=coords.readxmol(tfile)
	xmol=True
      else:
	trajs = PickleTrajectory(tfile)
    else:
      trajs=[traj]
    
    
    origo=int(sys.argv[3])-1
    extent=float(sys.argv[4])
    
    geom=asecoords.atoms2geom(traj)
    lattice=traj.get_cell()
    lattice=[lattice[0],lattice[1]]
    
    cells=[[-2,2],[-2,2]]
    # cells=[[0,1],[0,1]] # copying around
    # print "****** no copying of the supercell *****"
    
    lims=[[-extent,extent],[-extent,extent],[-10000,10000]]
    
    # 1)
    # geom=coords.setorigo(geom,origo)
    # subgeom=coords.subcoordinates(geom)["geom"]
    # newsubgeom=coords.expand_geom3(subgeom,lattice,cells)
    
    # 2)
    print "manipulating frame one.."
    
    multi=False
    
    geom=coords.setorigo(geom,origo)
    bigeom=coords.expand_geom3(geom,lattice,cells)
    coords.addnums(bigeom)
    # cut convenient area from the slab
    newgeom=coords.cut_block(bigeom,lims)
    # use subcoordinates to filter out "orphaned" atoms
    newsubgeom=coords.subcoordinates(newgeom,noextra=False)["geom"]
    newgeom=coords.expandsub(newsubgeom)
    #for g in newgeom:
    #  print ">",g
    # stop
    # get a list of remaining atoms in the original numbering
    lis=coords.harvestnum(newgeom)
    # use this list to extract atoms from the expanded geometry
    bigeom=coords.picknum(bigeom,num=lis)
    
    print ".. done.  next manipulate all steps .."
    # coords.writexmolmany("bigeom.xmol",[bigeom])
    # ok, works.. now that list can be applied to each MD step.
    newgeoms=[]
    cc=1
    
    if (par=="multicutfew"):
      newtrajs=[]
      ls=[0,-5,-4,-3,-2,-1]
      for l in ls:
	newtrajs.append(trajs[l])
      trajs=newtrajs
      
    for tr in trajs:
      print "step",cc
      # expand geometry..
      if (xmol):
	bigeom=tr
      else:
	bigeom=asecoords.atoms2geom(tr)
      bigeom=coords.expand_geom3(bigeom,lattice,cells)
      bigeom=coords.picknum(bigeom,num=lis)
      # print "bigeom=",bigeom
      newgeoms.append(copy.copy(bigeom))
      cc=cc+1
      
    coords.writexmolmany("allnew.xmol",newgeoms)
      
    # ase.io.write('cut.traj',newtraj)
    # ase.io.write('cut.xmol',newtraj,format="xyz")
    # lines=printcell(traj,center=True)
    # coords.writecoords("cell.tcl",lines)
    
    
  if (par=="multibulk"):
    print "python cp2ktools.py multibulk cp2k.traj"
    st=str(sys.argv[2])
    traj=ase.io.read(st)
    bigtraj=traj.repeat((2,2,2))
    ase.io.write('big.traj',bigtraj)
    ase.io.write('big.xmol',bigtraj,format="xyz")
    # lines=printcell(traj,center=True)
    # coords.writecoords("cell.tcl",lines)
    
    
  if (par=="adwans"):
    print "python cp2ktools.py adwans file.xmol file.dat"
    geoms=coords.readxmol(sys.argv[2])
    wanps=coords.readxmol(sys.argv[3])
    newgeoms=[]
    c=0
    for geom in geoms:
      wangeom=wanps[c]
      newgeoms.append(geom+wangeom)
      print len(geom),len(wangeom),len(newgeoms[c])
      
      c=c+1
    coords.writexmolmany("geomwan.xmol",newgeoms)
    
  if (par=="plotwans"):
    print "python cp2ktools.py plotwans file.xmol stepnum"
    geoms=coords.readxmol(sys.argv[2])
    stepnum=int(sys.argv[3])-1
    
    # debugging ..
    # xgeom=coords.pickspec(geoms[stepnum],["X"])
    # wanpoints=[]
    # for w in xgeom:
    #  wanpoints.append([w[1],w[2],w[3]])
    # lis=wanpoints
    # lines=coords.tcl_spheres(lis,rad=0.1,scale=1.0,color="green")
    # coords.writecoords("testcenters.tcl",lines)
    
    subgeom=coords.subcoordinates(geoms[stepnum],noextra=False)["geom"]
    
    #xgeom=coords.pickspec(subgeom,["X"])
    #wanpoints=[]
    #for w in xgeom:
     #wanpoints.append([w[1],w[2],w[3]])
    #lis=wanpoints
    #lines=coords.tcl_spheres(lis,rad=0.1,scale=1.0,color="green")
    #coords.writecoords("testcenters.tcl",lines)
    
    
    subgeom=coords.subadwan(subgeom, cutoff=2.0)
    coords.dumpgeomwan("wans",subgeom)
    print "look for wans.*"
    print
    
  if (par=="migproto"):
    print "python cp2ktools.py migproto file.xmol at1 at2"
    ats=[int(sys.argv[3]),int(sys.argv[4])]
    fname=sys.argv[2]
    migprotos(fname,ats)
    
    
  if (par=="combwans"):
    # combine coordinates and .dat wanpoint files to a single .xmol file
    print "python cp2ktools.py combwans from to"
    fr=int(sys.argv[2])
    to=int(sys.argv[3])
    newgeoms=[]
    plaingeoms=[]
    for i in range(fr,to+1):
      print i
      geom=coords.readxmol(str(i)+"/aserun_start.xmol")[-1]
      wanps=coords.readxmol(str(i)+"/cp2k-HOMO_centers_s1.data")[-1]
      newgeoms.append(geom+wanps)
      plaingeoms.append(copy.copy(geom))
    
    coords.writexmolmany("plaingeoms.xmol",plaingeoms)
    coords.writexmolmany("wangeoms.xmol",newgeoms)
    
    
  if (par=="hidelayers"):
    print "python cp2ktools.py hidelayers file.xmol tol toplay"
    # python ~/python/cp2ktools.py hidelayers aserun_start.xmol 0.2 7
    geom=coords.readxmol(sys.argv[2])[-1]
    tol=float(sys.argv[3])
    upto=int(sys.argv[4])
    lines=coords.hidelayers2(geom,tol,upto)
    coords.writecoords("hidelays.tcl",lines)
    
  if (par=="scaled_systems"):
    # keywords: lattice constant, scaling, relative coordinates
    # create a single .traj file with geometries having different volume
    # print "python cp2ktools.py file.traj a,b,c,d,e,f"
    print "python cp2ktools.py scaled_systems file.traj 0.9,1.1,10"
    trajfile=sys.argv[2]
    # vals=sys.argv[3].split(',')
    nums=sys.argv[3].split(',')
    vals=list(np.linspace(float(nums[0]),float(nums[1]),int(nums[2])))
    trajs=[]
    traj=ase.io.read(trajfile)
    cell=copy.copy(traj.cell)
    print ">",traj.get_volume()
    for v in vals:
      newtraj=copy.deepcopy(traj)
      newtraj.cell=(float(v)**(1.0/3.0))*cell
      print v,newtraj.get_volume()
      trajs.append(newtraj)
      
    ase.io.write("scales.traj",trajs)
     
  if (par=="set_volumes"):
    # keywords: lattice constant, scaling, relative coordinates
    # create a single .traj file with geometries having different volume
    # print "python cp2ktools.py file.traj a,b,c,d,e,f"
    print "python cp2ktools.py set_volumes file.traj 100,120,10"
    trajfile=sys.argv[2]
    # vals=sys.argv[3].split(',')
    nums=sys.argv[3].split(',')
    if (len(nums)==1):
      vals=[float(nums[0])]
    else:
      vals=list(np.linspace(float(nums[0]),float(nums[1]),int(nums[2])))
    trajs=[]
    traj=ase.io.read(trajfile)
    cell=copy.copy(traj.cell)
    basevol=traj.get_volume()
    print ">",traj.get_volume()
    for v in vals:
      f=v/basevol # how much bigger is the new cell
      newtraj=copy.deepcopy(traj)
      newtraj.cell=(f**(1.0/3.0))*cell # scale lattice vectors
      print v,newtraj.get_volume()
      trajs.append(newtraj)
      
    ase.io.write("scales.traj",trajs)
    
    
  if (par=="scale"):
    # for scaling bulk materials (metals, etc.)
    print "python cp2ktools.py scale file.traj lc_start,lc_stop,steps initscale element"
    trajfile=sys.argv[2]
    # vals=sys.argv[3].split(',')
    nums=sys.argv[3].split(',')
    iscale=1.0
    
    if (len(sys.argv)>4):
      iscale=float(sys.argv[4])
    
    if (len(nums)==1):
      vals=[float(nums[0])]
    else:
      vals=list(np.linspace(float(nums[0]),float(nums[1]),int(nums[2])))
    trajs=[]
    traj=ase.io.read(trajfile)
    
    if (len(sys.argv)>5):
      el=sys.argv[5]
      print "setting element",el
      traj.set_chemical_symbols([el]*len(traj))
    
    print "dividing by",iscale
    traj.cell=traj.cell/iscale
    traj.positions=traj.positions/iscale
    print "  >",traj
    
    trajs=[]
    cc=0
    for v in vals:
      newtraj=copy.deepcopy(traj)
      newtraj.cell=traj.cell*v
      newtraj.positions=traj.positions*v
      print cc,v
      print newtraj
      print newtraj.positions
      print
      trajs.append(newtraj)
      cc=cc+1
      
    ase.io.write("scales.traj",trajs)
    
    
  if (par=="arrange_water"):
    # fix traj files.. group water molecules ..
    print "python cp2ktools.py arrange_water file.traj (x)"
    print "(if x present, then sort in z)"
    trajfile=sys.argv[2]
    traj=ase.io.read(trajfile)
    geom=asecoords.atoms2geom(traj)
    if (len(sys.argv)>3):
      print "z sort"
      coords.sortgeom4(geom)
    geom=coords.groupwater(geom)
    # coords.writexmolmany("test.xmol",[geom])
    traj=asecoords.geom2atoms(geom,atomsobj=traj)
    ase.io.write(trajfile,traj)
    # ase.io.write("new.traj",traj)
    
  if (par=="paintsome"):
    print "python cp2ktools.py paintsome n m"
    sta=int(sys.argv[2])
    sto=int(sys.argv[3])
    lis=range(sta,sto+1)
    print "atoms from ",sta," to ",sto,"(python indices)"
    lines=coords.tcl_paintatoms(lis,color="blue",adone=True)
    coords.writecoords("paint.tcl",lines)
    
  
  if (par=="subsome"):
    print "python cp2ktools.py from.traj to.traj"
    print "certain positions in from.traj will be substituted"
    print "** list of substituted atoms defined inside script **"
    frfile=sys.argv[2]
    tofile=sys.argv[3]
    fr=ase.io.read(frfile)
    to=ase.io.read(tofile)
    indices=range(288,383+1); print "list> 128 atom layer slab"
    for i in indices:
      fr[i].set_position(to[i].get_position())
    ase.io.write("subs.traj",fr)
    
    
if (__name__ == "__main__"):
  main()

# pars={"inp":[]}
# lines=["eka","toka"]
# stuff(pars,["FORCE_EVAL","SUBSYS","COORD"],"inp",lines)
# print "pars=",pars

#pars={"inp":[]}
#parse(pars,coords.read_file("/home/sampsa/cp2k/h2o.inp"),0)
  
#print "pars="+str(pars)
  
#lines=getlines(pars,inte=1)
#for l in lines:
  #print l


#defpars=copy.deepcopy(def_kari)

## where are the pseudos & basis sets..
#defpars["FORCE_EVAL"]["DFT"]["inp"]=[
  #'BASIS_SET_FILE_NAME /wrk/sriikone/cp2k/libs/QS/BASIS_MOLOPT',
  #'POTENTIAL_FILE_NAME /wrk/sriikone/cp2k/libs/QS/GTH_POTENTIALS',
  #]

## cell shape & dimensions:
#defpars["FORCE_EVAL"]["SUBSYS"]["CELL"]["inp"]=[
  #'ABC          5.0 5.0 5.0'
  #]

#defpars["FORCE_EVAL"]["SUBSYS"]["KIND H"]["inp"]=[
  #'BASIS_SET DZVP-MOLOPT-SR-GTH',
   #'POTENTIAL GTH-PBE-q1',
   #'MASS  2.0000'
  #]
  


#defpars["FORCE_EVAL"]["SUBSYS"]["KIND O"]["inp"]=[
  #'BASIS_SET DZVP-MOLOPT-SR-GTH',
  #'POTENTIAL GTH-PBE-q6'
#]

#defpars["FORCE_EVAL"]["SUBSYS"]["COORD"]["inp"]=[
  ## here coordinates in xyz format:
  #]

## some global parameters..
#defpars["GLOBAL"]["inp"]=[
  #'PREFERRED_FFT_LIBRARY FFTSG',
  #'PROJECT out-w256',
  #'RUN_TYPE MD',
  #'PRINT_LEVEL LOW'
 #]
 
 
# paska={}
# stuff(paska,["eka","toka","kolkki"],"inp",[1,2,3])
# print paska

# todo: traj / xyz => cp2k format .. read traj / xmol .. ok => convert to list format
# .xmol file & cp2k restart file => .traj (velocitis, forces, atomic positions)


# - output always dipole moment values
# - possibility to output wannier function centers







