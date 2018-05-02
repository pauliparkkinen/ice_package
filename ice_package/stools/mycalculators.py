import os
import sys
import string
import copy
from os.path import join, isfile, islink, getmtime
from cmath import exp
import array
import subprocess

import numpy as np

from ase import *

from ase.data import chemical_symbols
from ase.units import Rydberg, fs, Bohr, Hartree
# from ase.io.cube import read_cube_data


# from ase.io.turbomole import write_turbomole

def writelines(fh,lines):
      for l in lines:
	fh.write(l+'\n')


def dumplines(f,lines):
  fh=open(f,'w')
  writelines(fh,lines)
  fh.close()


def subsline(lines,line,newline):
  c=0
  for l in lines:
    if (line in l):
      # print "line,l",line,l
      lines[c]=copy.deepcopy(newline)
    c=c+1 


def deline(lines,line):
  c=0
  newlines=[]
  for l in lines:
    if (line in l):
      # print "line,l",line,l
      pass
    else:
      newlines.append(l)
    c=c+1
  return newlines
      
      
def read_file(fil):
    lin=[]
    f=open(fil,'r')
    lin=f.readlines()
    f.close

    c=0
    lin2=[]
    for s in lin:
        ss=s[:-1]
        lin2.append(ss)
        c=c+1
        # print ">>>",c,ss

    return lin2


def stderr(line):
  sys.stderr.write(line+"\n")


def runseq(exe,seq,cwd=None):
  # for example.. lev00 utility for vasp:
  # seq=["V","OUTCAR","POSCAR","SD","1","10"] # spin difference
  # exe="lev00"
  pros=subprocess.Popen("",executable=exe,stdin=subprocess.PIPE,stdout=subprocess.PIPE,cwd=cwd)
  # pros=subprocess.Popen("",executable=bin,stdin=inp,stdout=out)
  
  stderr("runseq>exe"+str(exe))
  ss=""
  for s in seq:
    ss=ss+s+"\n"
	  
  print ss
  
  com=pros.communicate(ss)
  #for l in com:
  #	  print l
  
  return com
  
# turbomole io module has a stupid bug .. it constraints present, it only writes out fixed atoms (!)
# here the modified turbomole writer
def write_turbomole(filename, atoms):                                                                                                                                                                                                        
    """Method to write turbomole coord file                                                                                                                                                                                                  
    """
    import numpy as np
    from ase.constraints import FixAtoms

    if isinstance(filename, str):
        f = open(filename, 'w')
    else: # Assume it's a 'file-like object'
        f = filename

    coord = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    printfixed = False

    if atoms.constraints:
        for constr in atoms.constraints:
            if isinstance(constr, FixAtoms):
                fix_index=constr.index
                # printfixed=True # wft?
                
    #print sflags
        
    if (printfixed):
        fix_str=[]
        for i in fix_index:
            if i == 1:
                fix_str.append("f")
            else:
                fix_str.append(" ")


    f.write("$coord\n")
    if (printfixed):
        for (x, y, z), s, fix in zip(coord,symbols,fix_str):
            f.write('%20.14f  %20.14f  %20.14f      %2s  %2s \n' 
                    % (x/Bohr, y/Bohr, z/Bohr, s.lower(), fix))

    else:
        for (x, y, z), s in zip(coord,symbols):
            f.write('%20.14f  %20.14f  %20.14f      %2s \n' 
                    % (x/Bohr, y/Bohr, z/Bohr, s.lower()))
    f.write("$end\n")


class molpro:
    """Class for doing molpro calculations.

    S.R.

    """
    def __init__(self, label='mol', dire="", calcs=["hf","mp2","force"], memory=0, kpts=None, nbands=None, width=None, meshcutoff=None, charge=None, pulay=5, mix=0.1, maxiter=120,basis=None, debug=False):
      self.converged = False
      self.label=label
      self.calcs=calcs
      self.dire=dire
      self.debug=debug
      self.memory=memory
      os.system("mkdir "+self.dire)
      # calcs=["hf","mp2","force"] # hartree-fock, followed by mp2
      # calcs=["hf","lmp2","force"]
      # linear scaling methods: LMP2, DF-LMP2, DF-SCS-LMP2 (df: density fitting, scs: spin component scaled)
      
      	 
    def initialize(self, atoms):
        self.converged = False
	self.numbers = atoms.get_atomic_numbers().copy() # list of atomic numbers
        self.species = []
        for a, Z in enumerate(self.numbers):
            if Z not in self.species:
                self.species.append(Z)


    def write_input(self, atoms):
        """Write input parameters"""
        fh = open(self.dire+"/"+self.label + '.com', 'w')

	lines=[]
	
	if (self.memory==0):
	  memfield=""
	else:
	  memfield="memory,"+str(int(self.memory/8))+",m"
	  # molpro takes the amount of memory in mega_words_
	  # for example: ask 2 GB ~ 2000 Mb in your batch script
	  # then in the calculator, set memory to 1600 Mb (200 MW)
	
	lines.append("***,mol")
	if (memfield!=""):
	  lines.append(memfield)
	lines=lines+[
	"file,2,mol.wfu,new",
	"file,3,mol.aux,new",
	"geomtyp=xyz",
	"geometry={",
	# number of atoms
	str(len(atoms)),
	"kokkelis" # just some useless comment line
	]
	
	for atom in atoms:
	  spec=atom.get_symbol()
	  xyzs=atom.get_position()
	  lines.append(str(spec)+","+str(xyzs[0])+","+str(xyzs[1])+","+str(xyzs[2]))
	
	lines.append("}")
	# comment
	# O,0.595274161601,1.11022302463e-16,0.767796478392
	# H,-0.0,0.0,0.0
	# ...
	for calc in self.calcs:
	  lines.append(calc)
	
	lines=lines+[
	"put,XYZ,mol.xyz",
	"put,MOLDEN,mol.vmd",
	""
	]
	
	writelines(fh,lines)

        fh.close()


    def calculate(self, atoms):
        self.positions = atoms.get_positions().copy()
        self.cell = atoms.get_cell().copy()
        self.pbc = atoms.get_pbc().copy()

        self.write_input(atoms)

	# vuori: $MOLPROP, $MOLPROS = parallel and serial executables
        exe = os.environ['MOLPROP'] # molpro executable
        myexe = os.environ['MYEXE'] # srun $MOLPROP
        
        comm=myexe+" -d"+self.dire+" -I"+self.dire+" -W"+self.dire+" "+self.dire+"/"+self.label+".com"
        comm="cd "+self.dire+"; "+comm
        stderr("")
        stderr("*** running molpro with command.. ***")
        stderr(comm)
        stderr("")
        # stop
        
        if (self.debug):
	  stop
	else:
	  os.system(comm)
        
        # comm = "/usr/bin/dir" # test
        # locals = {'label': self.label}
        
        # execfile(comm, {}, locals) # execfile executes a python command
        
        # the command in vuori should be something like this..
        #
        # /v/linux26_x86_64/appl/chem/molpro/molpro2010.1/serial/bin/molpro -d/wrk/sriikone/runs//test3 -I/wrk/sriikone/runs//test3 -W/wrk/sriikone/runs/test3/wfu all.com
        # .. just define it in the calculator
        
        # locals["exitcode"]=0
        
        # exitcode = locals['exitcode']
        # if exitcode != 0:
        #    raise RuntimeError(('molpro exited with exit code: %d.  ' +
        #                        'Check %s.out and log for more information.') %
        #                       (exitcode, self.label))

        # self.dipole = self.read_dipole()
        self.read()

        self.converged = True
        
        
    def read(self):
        """ energy from the .xyz file 
	    forces from the .out file
        """
        text = open(self.dire+"/"+self.label + '.out', 'r').read().lower()
        assert 'error' not in text
        # lines = iter(text.split('\n'))
        lines=text.split('\n')
        # print "lines=",lines
        
        # ** find forces ***
        # format:
        #
        # MP2 GRADIENT FOR STATE 1.1
	#
	# Atom          dE/dx               dE/dy               dE/dz
	#
	# 1        -0.090030302         0.000000000         0.000000000
	# 2         0.090030302         0.000000000         0.000000000
	#
        
        # print ">>>",self.numbers
        self.forces = np.zeros((len(self.numbers),3))
        # self.forces[i, 0] = float(lines[i][6:18].strip())
        
        # text = open(self.label + '.log', 'r').read().lower()
        # lines = iter(text.split('\n'))
        
        c=0
	for line in lines:
	    # print ">"+line+"<"
	    if (line.find("gradient for state")>-1):
	      # print ">>"+line+"<<"
	      # stop
	      sta=c+4
	      sto=sta+len(self.numbers)
	      i=0
	      for l in lines[sta:sto]:
		# print "force>",l
		vals=l.split()
		self.forces[i,0]=float(vals[1])
		self.forces[i,1]=float(vals[2])
		self.forces[i,2]=float(vals[3])
		i=i+1
		# forces are in rydbergs?
		# ASE: eV and Ang => [F] = eV/Ang
		
	      # stop
	      # print "forces=",self.forces
	      # .. molpro reports actually the derivatives
	      # .. and forces are -1 * derivatives   :)
	      self.forces=-self.forces*(Hartree/Bohr) 
	      # as in https://wiki.fysik.dtu.dk/gpaw/devel/overview.html: gpaw
	    c=c+1
	    
	  
            # if line.startswith('siesta: etot    =') and counter == 0:
            #    counter += 1
            # pass	
	
        text = open(self.dire+"/"+self.label + '.xyz', 'r').read().lower()
        # lines = iter(text.split('\n'))
        lines = text.split('\n')
        self.etotal  = float(lines[1].split()[-1])
        self.efree  = float(lines[1].split()[-1])
        
        self.etotal=self.etotal*Hartree
        self.efree = self.efree*Hartree
        # print "e=",self.etotal
        # stop
        stderr("nrj:"+str(self.etotal))
        stderr("forces:")
        stderr(str(self.forces))
        stderr("-----")

    

    def get_potential_energy(self, atoms, force_consistent=False):
        self.update(atoms)
        #if force_consistent:
        #    return self.efree
        #else:
        #    # Energy extrapolated to zero Kelvin:
        #    return  (self.etotal + self.efree) / 2
	return self.efree


    def get_forces(self, atoms):
        self.update(atoms)
        return self.forces.copy()


    def get_stress(self, atoms):
	return None


    def update(self, atoms):
	# print "this is update, converged=",self.converged
        if (not self.converged or
            len(self.numbers) != len(atoms) or
            (self.numbers != atoms.get_atomic_numbers()).any()):
            self.initialize(atoms)
            self.calculate(atoms)
        elif ((self.positions != atoms.get_positions()).any() or
              (self.pbc != atoms.get_pbc()).any() or
              (self.cell != atoms.get_cell()).any()):
            self.calculate(atoms)


# call chain:
# get_potential_energy(atoms) => update(atoms) .. if atoms have been moved, then => calculate(atoms)



class turbomole:
    """Class for doing Turbomole calculations.
    """
    def __init__(self, dirname=".", tasks=[], basis="def2-TZVPD", auxbasis="aug-cc-pVTZ", jkauxbasis="aug-cc-pVTZ", library="", clean=True, pros=None):
        """Construct TURBOMOLE-calculator object.

	adequate aux basis: def2-TZVPD
	if not available, use: aug-cc-pVTZ

	# dscf, rimp2 (rimp2prep only once)

        Parameters
        ==========
        label: str
            Prefix to use for filenames (label.txt, ...).
            Default is 'turbomole'.

        Examples
        ========
        This is poor man's version of ASE-Turbomole:

        First you do a normal turbomole preparation using turbomole's
        define-program for the initial and final states.

        (for instance in subdirectories Initial and Final)

        Then relax the initial and final coordinates with desired constraints
        using standard turbomole.

        Copy the relaxed initial and final coordinates 
        (initial.coord and final.coord)
        and the turbomole related files (from the subdirectory Initial) 
        control, coord, alpha, beta, mos, basis, auxbasis etc to the directory
        you do the diffusion run.
        
        For instance:
        cd $My_Turbomole_diffusion_directory
        cd Initial; cp control alpha beta mos basis auxbasis ../;
        cp coord ../initial.coord;
        cd ../;
        cp Final/coord ./final.coord;
        mkdir Gz ; cp * Gz ; gzip -r Gz
        
        from ase.io import read
        from ase.calculators.turbomole import Turbomole
        a = read('coord', index=-1, format='turbomole')
        calc = Turbomole()
        a.set_calculator(calc)
        e = a.get_potential_energy()
        
        """
        
        self.dire=dirname
        self.tasks=tasks # which routines are used to calculate energy and forces
        
        # get names of executables for turbomole energy and forces
        # get the path

	self.head="cd "+self.dire+"; "
	self.headi=self.dire+"/"
	
	comm="mkdir "+self.dire
	os.system(comm)
	
	preamp=False
	if (preamp):
	  # is this needed if we have loaded the correct module in csc machines?
	  os.system(self.head+'rm -f sysname.file; sysname > sysname.file')
	  f = open(self.headi+'sysname.file')
	  architechture = f.readline()[:-1]
	  f.close()
	  tmpath = os.environ['TURBODIR']
	  pre = tmpath + '/bin/' + architechture + '/'
	  self.pre=pre
	else:
	  self.pre=""

	# ------ old tm interface ---------
        # if os.path.isfile('control'):
        #    f = open('control')
        # else:
        #    print 'File control is missing'
        #    raise RuntimeError, \
        #        'Please run Turbomole define and come thereafter back'
        # lines = f.readlines()
        # f.close()
        #
        #
        #
        #     
	#
        # self.tm_program_energy=pre+'dscf'
        # self.tm_program_forces=pre+'grad'        
        #
        # --------- old tm interface -------
        # for line in lines:
        #    if line.startswith('$ricore'):
        #        self.tm_program_energy=pre+'ridft'
        #        self.tm_program_forces=pre+'rdgrad'
	# -----------------------------------

        # self.label = label
        self.converged = False
        
        # file cleansing
        #clean up turbomole energy file
        # os.system(self.head+'rm -f energy; touch energy')
	# to-do: clean also gradient file

	self.library=library

	self.basis=basis
	self.auxbasis=auxbasis
	self.jkauxbasis=jkauxbasis
	
	if (clean): # set this to false when doing code testing..
	  print "cleaning!"
	  self.cleanfiles()
	  self.cleanerror()
	  # comm=self.head+"rm -f *.log"
	  # stderr(comm)
	  # os.system(comm)
	
	self.countereset()

        #turbomole has no stress
        self.stress = np.empty((3, 3))
        
        self.pros=pros
        
        
    def countereset(self):
      self.count=0
    
    def cleanfiles(self):
      comm=self.head+"rm -f energy gradient"
      stderr(comm)
      os.system(comm)
    
      
    def cleaninput(self):
      comm=self.head+"rm -f basis control"
      stderr(comm)
      os.system(comm)
    
    
    def cleanerror(self):
      comm=self.head+"rm -f *_problem"
      stderr(comm)
      os.system(comm)
      
    
    def write_coords(self, atoms):
      write_turbomole(self.headi+'coord', atoms)
    
    
    def copy_control(self,filename):
      comm="cp -f "+self.library+"/"+filename+" "+self.headi+"/control"
      stderr(comm)
      os.system(comm)
 
    
    
    def tune_control(self, atoms, model, pars={}):
      import asecoords
      import coords
      # so, run the EXTREMELY ANNOYING "define" program once for your
      # system & bang your head into the wall with missing jkbas sets and the like.
      # .. save the resulting "control" file into python library (a string)
      # .. then give that string to this routine (variable "model").
      
      geom=asecoords.atoms2geom(atoms)
      debug=False
      # model=water_tzvpd # debugging
      newlines=[]

      # geom=[["H",0,0,0],["H",0,0,0],["O",0,0,0],["O",0,0,0]] # debugging
      # for i in range(0,50):
      #  geom.append(["H",0,0,0])


      if (pars.has_key("rij")):
	if (pars["rij"]==False):
	  model["control"]=deline(model["control"],"$rij")
	  # TODO: in ricc2 run should add rij again to the control file..!

      c=0
      for l in model["control"]:
	if (debug): print ">",l
	if (l.find("SPECIES")!=-1):
	  ll=l.split(":")
	  spec=ll[1]
	  if (debug): print "SPEC>",spec
	  lis=coords.pickspeclist(geom,spec)
	  if (debug): print "LIS>",lis
	  # column 81: "\"
	  # st=spec.lower()+"  "*80+"\\"
	  st=spec.lower()
	  st=st+"  "
	  for l in lis:
	    st=st+str(l+1)
	    if (l==lis[-1]):
	      pass
	    else:
	      st=st+","
	    l=len(st)
	    if (l>75):
	      st=st+(79-l)*" "+"\\"
	      newlines.append(copy.copy(st))
	      if (debug): print ">>",st
	      st="   "
	  if (st.lstrip().rstrip()!=""):
	    # st=st[0:-1]
	    l=len(st)
	    st=st+(79-l)*" "+"\\"
	    newlines.append(copy.copy(st))
	    if (debug): print ">>",st
	  
	else:
	  newlines.append(l)
	  # print st
	c=c+1


      # tune control file by hand ..
      if (pars.has_key("D3")):
	if (pars["D3"]):
	  newlines.insert(-1,"$disp3")
	  
      if (pars.has_key("turbofunc")):
	newlines.pop(-1) # remove "$end"
	newlines=newlines+[
	  "$dft",
	  "  functional "+pars["turbofunc"],
	  "  gridsize   m3",
	  "$end"
	  ]
	

      dumplines(self.headi+"control",newlines)
      dumplines(self.headi+"basis",model["basis"])
      dumplines(self.headi+"auxbasis",model["auxbasis"])
      
      write_turbomole(self.headi+'coord', atoms) # needs coordinates..
      # stop
      
      if (pars.has_key("charge")):
	charge=str(int(pars["charge"]))
      else:
	charge=""
	
      # keys should be pressed as follows:
      seq=[
	"",
	"",
	"",
	"eht",
	"y",
	charge,
	"", 
	"",
	"",
	"*"
	]
      
      # start the celebrated "define" program
      tx=self.pre+"define"
      stderr("running "+tx)
      output=runseq(tx,seq,self.headi) # running the turbomole input generator define
      
        
      
    
    def create_control(self, atoms, sections=[1],pars={}):
      """
      a hint: if you want to visualize densities, add the following
      lines before the $end tag in your control file:
      
      $pointval fmt=cub
      $anadens
      gsdens
      
      and then run
      
      ricc2 -fanal
      ricc2 -proper
      
      $pointval mo 99-101
      
      should create .plt files for mo's 99-101 but at the moment this does not work
      (numeration can be found with the "eiger" program)
      
      more info here:
      http://www.turbomole-gmbh.com/manuals/version_6_3/Documentation_html/node318.html#31047
      http://www.teokem.lu.se/~ulf/Methods/turbomole.html
      """
      
      write_turbomole(self.headi+'coord', atoms) # needs coordinates..
      
      if (pars.has_key("charge")):
	charge=str(int(pars["charge"]))
      else:
	charge=""
      
      if (0 in sections): # prepare turbomole for dscf calculations.. commands: dscf, grad
	if (pars.has_key("rimem")):
	  mem=str(pars["rimem"])
	else:
	  mem="200"
	# **** prepare turbomole for DFT calculations ***
	self.cleaninput()
	# define run with cc-pVDZ basis etc.
	# creates a turbomole control file
	seq=[
	"","kokkelis", # input title, etc.
	"a coord", # coordinates in atomic units
	# "aa coord", # coordinates in angstroms
	"*", 
	"no", # dont use internal coords
	
	# "b","all cc-pVDZ","*", # basis set definition (1)
	# "lib","1","*", # basis set from library (2) .. this should be ok for water systems?
	# "b","all def2-TZVPD","*", # (3) .. nope. to get the dipole moments right, must use diffuse basis functions (PD) .. this does not have
	# auxiliary basis yet
	
	# "b","all def2-TZVPD","*", # (4) def2-TZVPD & aug-cc-pVTZ
	# "b","all def2-TZVPD","*", # (5) def2-TZVPD & def2-TZVP  .. diffuusi kantafunktio, ei-niin-diffuusi apufunktio?
	"b","all "+self.basis,"*",
	
	
	"eht", "",charge,"",# occupation numbers from guess
	# ........C.... charge here..
	# after this it takes a while..
	# put ri definitions here!
	# now we are at general menu..
	
	"dft","on","func",pars["turbofunc"],"", # functionals: pbe, b-lyp, b3-lyp, tpss, pbe0 # then must run dscf, grad
	
	"*" # exit
	]

	tx=self.pre+"define"
	stderr("running "+tx)
	
	output=runseq(tx,seq,self.headi) # running the turbomole input generator define
	# print ">"+str(output)+"<"
	# stop
	lines=[]
	
	for se in seq:
	  lines.append(">>"+se)
	
	for o in output:
	  lines.append(str(o))
	
	dumplines(self.headi+"define.output",lines)

	lines=read_file(self.headi+"control")
	subsline(lines,"$scfiterlimit","$scfiterlimit 150") # change some def pars
	# lines.insert(-1,"$denconv 1.d-7")
	if (pars.has_key("D3")):
	  lines.insert(-1,"$disp3 ") # Grimme dispersion corrections
	dumplines(self.headi+"control",lines)
      
      
      if (1 in sections): # prepare turbomole for ridft calculations.  commands: ridft, rdgrad
	if (pars.has_key("rimem")):
	  mem=str(pars["rimem"])
	else:
	  mem="200"
	# **** prepare turbomole for DFT calculations ***
	self.cleaninput()
	# define run with cc-pVDZ basis etc.
	# creates a turbomole control file
	seq=[
	"","kokkelis", # input title, etc.
	"a coord", # coordinates in atomic units
	# "aa coord", # coordinates in angstroms
	"*", 
	"no", # dont use internal coords
	
	# "b","all cc-pVDZ","*", # basis set definition (1)
	# "lib","1","*", # basis set from library (2) .. this should be ok for water systems?
	# "b","all def2-TZVPD","*", # (3) .. nope. to get the dipole moments right, must use diffuse basis functions (PD) .. this does not have
	# auxiliary basis yet
	
	# "b","all def2-TZVPD","*", # (4) def2-TZVPD & aug-cc-pVTZ
	# "b","all def2-TZVPD","*", # (5) def2-TZVPD & def2-TZVP  .. diffuusi kantafunktio, ei-niin-diffuusi apufunktio?
	"b","all "+self.basis,"*",
	
	
	"eht", "",charge,"",# occupation numbers from guess
	# after this it takes a while..
	# put ri definitions here!
	# now we are at general menu..
	
	"dft","on","func",pars["turbofunc"],"", # functionals: pbe, b-lyp, b3-lyp # then must run dscf, grad
	
	# --------- this is for MP2 calculations -------
	# "cc","cbas","","","b","all aug-cc-pVTZ","*","*", # (4) 
	# "cc","cbas","","","b","all def2-TZVP","*","*", # (5) 
	# "cc","cbas","","","b","all "+self.auxbasis,"*","*",
	
	# --------- this goes for dft calculations -------
	# ri on jbas b all.. * m mem *
	# rijk . . on jkbas all.. * m mem *
	
	# .. definition of ri aux basis for dft calculations
	
	"ri", # hartree integrals
	"on","jbas","b","all "+self.auxbasis,"*","m",mem,"*"
	]
	
	if (pars.has_key("shit")):
	  # if the basis is not found, you need those stupid extra enters..
	  seq=seq+[
	  "rijk", # exchange integrals
	  "","","", # extra enters .. this depends on number of elements?  horror.
	  "jkbas","b","all "+self.jkauxbasis,"*", # "m",mem,
	  "*"
	  ]
	else:
	  # if the basis is automagically found, dont need them.  thanks turbomole devs!
	  seq=seq+[
	  "rijk", # exchange integrals
	  "jkbas","b","all "+self.jkauxbasis,"*", # "m",mem,
	  "*"
	  ]
	
	seq=seq+[
	"marij","", # multipole accelerated
	
	# old version, only for rijk..
	# "rijk",
	# "jkbas","b","all "+self.auxbasis,"*", # .. nope. not that easy
	# "on","","",
	# "jkbas","b","all "+self.auxbasis,"*",
	# "m",mem,
	# "*",
	
	"*" # exit
	]

	# if jkbas and marij get screwed in this generation, then they are not simply used
	# = no harm done (maybe just some speed lost)

	tx=self.pre+"define"
	stderr("running "+tx)
	
	output=runseq(tx,seq,self.headi) # running the turbomole input generator define
	# print ">"+str(output)+"<"
	# stop
	lines=[]
	
	for se in seq:
	  lines.append(">>"+se)
	
	for o in output:
	  lines.append(str(o))
	
	dumplines(self.headi+"define.output",lines)

	lines=read_file(self.headi+"control")
	subsline(lines,"$scfiterlimit","$scfiterlimit 150") # change some def pars
	# lines.insert(-1,"$denconv 1.d-7")
	if (pars.has_key("D3")):
	  lines.insert(-1,"$disp3 ") # Grimme dispersion corrections
	dumplines(self.headi+"control",lines)
      
      
      if (2 in sections):
	# **** prepare turbomole for MP2 calculations ***
	# .. I have been seeming to use this lately.. why not section 1 ..?
	self.cleaninput()
	# define run with cc-pVDZ basis etc.
	# creates a turbomole control file
	seq=[
	"","kokkelis", # input title, etc.
	"a coord", # coordinates in atomic units
	# "aa coord", # coordinates in angstroms
	"*", 
	"no", # dont use internal coords
	
	# "b","all cc-pVDZ","*", # basis set definition (1)
	# "lib","1","*", # basis set from library (2) .. this should be ok for water systems?
	# "b","all def2-TZVPD","*", # (3) .. nope. to get the dipole moments right, must use diffuse basis functions (PD) .. this does not have
	# auxiliary basis yet
	
	# "b","all def2-TZVPD","*", # (4) def2-TZVPD & aug-cc-pVTZ
	# "b","all def2-TZVPD","*", # (5) def2-TZVPD & def2-TZVP  .. diffuusi kantafunktio, ei-niin-diffuusi apufunktio?
	"b","all "+self.basis,"*",
	
	"eht", "",charge,"",# occupation numbers from guess
	# after this it takes a while..
	# put ri definitions here!
	# now we are at general menu..  # dft,on,func,b3-lyp,"" # dscf,grad # b-lyp, pbe
	
	# "cc","cbas","","","b","all aug-cc-pVTZ","*","*", # (4) 
	# "cc","cbas","","","b","all def2-TZVP","*","*", # (5) 
	"cc","cbas",
	"","","", # depends on number of elements..?
	"b","all "+self.auxbasis,"*","*", # "OPTIONS FOR RICC2"
	
	# .. definition of ri aux basis for dft calculations
	# "ri", # hartree integrals
	# "on","jbas","b","all "+self.auxbasis,"*","m",mem,"*",
	# "rijk", # exchange integrals
	# "","","on","jkbas","b","all "+self.auxbasis,"*","m",mem,"*",
	# "marij","", # multipole accelerated
	
	"cc","jkbas",
	"","","",
	"b","all "+self.jkauxbasis,"*","*",
	
	
	"*" # (1), (2), (3), (4), exit
	]

	tx=self.pre+"define"
	stderr("running "+tx)
	
	output=runseq(tx,seq,self.headi) # running the turbomole input generator define
	# print ">"+str(output)+"<"
	# stop
	lines=[]
	
	for se in seq:
	  lines.append(">>"+se)
	
	for o in output:
	  lines.append(str(o))
	
	dumplines(self.headi+"define.output",lines)

	lines=read_file(self.headi+"control")

	subsline(lines,"$scfiterlimit","$scfiterlimit 150") # change some def pars

	# lines.insert(-1,"$denconv 1.d-7")
	
	more=["$denconv 1.d-7","$ricc2","   mp2","   geoopt model=mp2"]
	
	endline=copy.deepcopy(lines[-1]) # take last element
	lines.pop(-1) # remove last element
	lines=lines+more # add more attributes
	lines.append(endline) # close file with last element
	
	dumplines(self.headi+"control",lines)
	
	
      if (3 in sections): # DONT USE THIS ANYMORE UNDER ANY CIRCUMSTANCES!
	seq=[ # simply default values..
	  "",
	  "*",
	  "","",
	  "*"
	  ]
	
	# tx=self.pre+"rimp2prep"
	# stderr("running "+tx)
	# output=runseq(tx,seq,self.headi) # running the turbomole input generator rimp2prep
	# .. this is needed in the case auxbasis was not defined?
	
	lines=[]
	for o in output:
	  lines.append(str(o))
	
	dumplines(self.headi+"rimp2prep.output",lines)
	
	more=["$ricc2","   mp2","   geoopt model=mp2"]

	lines=read_file(self.headi+"control")
	
	endline=copy.deepcopy(lines[-1]) # take last element
	lines.pop(-1) # remove last element
	lines=lines+more # add more attributes
	lines.append(endline) # close file with last element
	
	dumplines(self.headi+"control",lines)
	
      write_turbomole(self.headi+'coord', atoms) # write coordinates again just in case..
      
    def update(self, atoms):
        """Energy and forces are calculated when atoms have moved
        by calling self.calculate
        """
        if (not self.converged or
            len(self.numbers) != len(atoms) or
            (self.numbers != atoms.get_atomic_numbers()).any()):
            self.initialize(atoms)
            self.calculate(atoms)
        elif ((self.positions != atoms.get_positions()).any() or
              (self.pbc != atoms.get_pbc()).any() or
              (self.cell != atoms.get_cell()).any()):
            self.calculate(atoms)

    def initialize(self, atoms):
        self.numbers = atoms.get_atomic_numbers().copy()
        self.species = []
        for a, Z in enumerate(self.numbers):
            self.species.append(Z)
        self.converged = False
        
    def get_potential_energy(self, atoms):
        self.update(atoms)
        return self.etotal

    def get_forces(self, atoms):
        self.update(atoms)
        return self.forces.copy()
    
    def get_stress(self, atoms):
        self.update(atoms)
        return self.stress.copy()

    def get_dipole_moment(self):
	self.update(atoms)
        return self.dipole_moment.copy()
  

    def calculate(self, atoms):
        """Total Turbomole energy is calculated (to file 'energy'
        also forces are calculated (to file 'gradient')
        """
        self.positions = atoms.get_positions().copy()
        self.cell = atoms.get_cell().copy()
        self.pbc = atoms.get_pbc().copy()

	# clean up the directory
	self.cleanfiles()
	self.cleanerror()

        #write current coordinates to file 'coord' for Turbomole
        write_turbomole(self.headi+'coord', atoms)

	tn=0
	# ------------ task loop ------
	self.outputs=[]
	for task in self.tasks:
	  if (self.pros==None):
	    prosdef=""
	  else:
	    prosdef=" export PARNODES="+str(self.pros)+"; "
	  
	  tx=self.head+self.pre+prosdef+task # this is the final command to run
	  
	  # fname="turbomole.out."+str(self.count)+"."+str(tn)
	  fname="turbomole.out."+str(tn)
	  self.outputs.append(fname) # save the output filenames
	  
	  # efile="energy."+str(self.count)+"."+str(tn)
	  # gfile="gradient."+str(self.count)+"."+str(tn)
	  
	  stepi=str(self.count)+"."+str(tn)
	  
	  os.system(self.head+"rm -f "+fname)
	  stderr(tx)
	  os.system(tx+"> "+fname) # *** parallelization ? *** .. ok  RUNNING TURBOMOLE!
	  os.system(self.head+'echo "ASE TURBOMOLE STEP '+stepi+' " >> turbomole.log')
	  os.system(self.head+"cat "+fname+">> turbomole.log")
	  
	  # os.system(self.head+"rm -f "+efile)
	  # os.system(self.head+"rm -f "+gfile)
	  # os.system(self.head+"cp -f energy "+efile)
	  # os.system(self.head+"cp -f gradient "+gfile)
	  # .. that creates thousand files.. lets use instead:
	  
	  tn=tn+1
	  
	  # -- old interface
	  #Turbomole energy calculation
	  # os.system(self.head+'rm -f output.energy.dummy; \
	  #              '+ self.tm_program_energy +'> output.energy.dummy')
	  # --

	  #check that the energy run converged
	  if os.path.isfile(self.headi+'dscf_problem'):
	      print 'Turbomole scf energy calculation did not converge!'
	      stderr('Turbomole scf energy calculation did not converge!')
	      # print 'issue command t2x -c > last.xyz'
	      # print 'and check geometry last.xyz and job.xxx or statistics'
	      raise RuntimeError, \
		  'Turbomole scf energy calculation did not converge!'
	# ------------- task loop ------

	os.system(self.head+'echo "ASE TURBOMOLE STEP '+str(self.count)+' " >> energy.log')
	os.system(self.head+'echo "ASE TURBOMOLE STEP '+str(self.count)+' " >> gradient.log')
	os.system(self.head+"cat energy >> energy.log")
	os.system(self.head+"cat gradient >> gradient.log")

        self.read_energy()
        
        # self.read_dipole_moment() # .. include this one to the "read_energy" method

	# -- old interface
        #Turbomole atomic forces calculation
        #killing the previous gradient file because 
        #turbomole gradients are affected by the previous values
        #os.system(self.head+'rm -f gradient; rm -f output.forces.dummy; \
        #              '+ self.tm_program_forces +'> output.forces.dummy')
	# --

        self.read_forces(atoms)

	# remove files: energy, gradient
	
        self.converged = True

	self.count=self.count+1
	
        
    def read_energy(self):
      
	sw=2
	
	if (sw==1):
	  """Read Energy from Turbomole energy file."""
	  text = open(self.headi+'energy', 'r').read().lower() # actually this would be easier from the gradients file..
	  lines = iter(text.split('\n'))

	  # Energy:
	  for line in lines:
	      if line.startswith('$end'):
		  break
	      elif line.startswith('$'):
		  pass
	      else:
		  #print 'LINE',line
		  splitted=line.split()
		  # print "splitted=",splitted
		  energy_tmp=float(splitted[1])
		  if (len(splitted)>4): # in this case we have something "extra", i.e. MP2 energies in the file
		    mp2=float(splitted[4])
		    # print "mp2=",mp2
		    energy_tmp=energy_tmp+mp2
	  # print 'energy_tmp',energy_tmp
	  
	if (sw==2):
	  # read energy from turbomole gradient file
	  ffile = open(self.headi+'gradient','r')
	  line=ffile.readline()
	  line=ffile.readline()
	  ind=line.find("energy =")
	  if (ind<0):
	    print "cant read energy from gradient file!"
	    stop
	  else:
	    val=line[ind:-1].split()[2]
	    energy_tmp=float(val)
	    # print "gradient: energy_tmp=",energy_tmp
	 
        self.etotal = energy_tmp * Hartree

	# read dipole moment, but from the current output file
	# print "opening file",self.headi+self.outputs[0] # dipole is read from the first output file .. this might depend..
	text = open(self.headi+self.outputs[0],'r').read().lower()
	lines = iter(text.split('\n'))
	# the line we are looking for:
	#   | dipole moment | =     6.8855 a.u. =    17.5014 debye 
	self.dipole_moment=None
	for line in lines:
	  # print "line=",line
	  if (line.startswith("   | dipole moment | = ")):
	    spl=line.split()
	    self.dipole_moment=float(spl[8])



    def read_forces(self,atoms):
        """Read Forces from Turbomole gradient file."""

        ffile = open(self.headi+'gradient','r')
        line=ffile.readline()
        line=ffile.readline()
        tmpforces = np.array([[0, 0, 0]])
        while line:
            if 'cycle' in line:
                for i, dummy in enumerate(atoms):
                            line=ffile.readline()
                forces_tmp=[]
                for i, dummy in enumerate(atoms):
                            line=ffile.readline()
                            line2=string.replace(line,'D','E')
                            #tmp=np.append(forces_tmp,np.array\
                            #      ([[float(f) for f in line2.split()[0:3]]]))
                            tmp=np.array\
                                ([[float(f) for f in line2.split()[0:3]]])
                            tmpforces=np.concatenate((tmpforces,tmp))  
            line=ffile.readline()
            

        #note the '-' sign for turbomole, to get forces
        self.forces = (-np.delete(tmpforces, np.s_[0:1], axis=0))*Hartree/Bohr
        # print 'forces', self.forces


    def read(self):
        """Dummy stress for turbomole"""
        self.stress = np.empty((3, 3))


class dummyturbomole(turbomole):
    def __init__(self):
      self.headi=os.environ["PWD"]+"/"
      self.outputs=["turbomole.out"]
      
      # nrj=self.read_energy()
      # frs=self.read_forces()


class cp2k:
    """Class for doing SIESTA calculations.
    The default parameters are very close to those that the SIESTA
    Fortran code would use.  These are the exceptions::
      calc = Siesta(label='siesta', xc='LDA', pulay=5, mix=0.1)
    Use the set_fdf method to set extra FDF parameters::
      calc.set_fdf('PAO.EnergyShift', 0.01 * Rydberg)
    """
    def __init__(self, dire, execom=None, qms=None, relax=False):
        self.converged=False
        self.execom=execom
	self.dire=dire
	self.energy=None
	self.forces=None
	self.count=0
	self.etotal=None
	self.pars={}
	self.groups=None
	self.stress=None
	
	self.relax=relax
	
	self.inputfile="cp2k.inp"
	self.outputfile="cp2k.out"
	
	if (qms!=None):
	  self.qms=[]
	  for q in qms:
	    self.qms.append(q-1)
	else:
	  self.qms=None
	
	comm="mkdir "+str(self.dire)
	os.system(comm)
       
 
    def setdir(self,dire):
      self.dire=dire
      comm="mkdir "+str(self.dire)
      os.system(comm)
      
 
    def setpars(self,pars):
	self.pars=pars # pars is a hierarchical dictionary object..
	
 
    def setgroups(self,groups):
	self.groups=groups # [{"h2o":indices},{"hcl":indices}] .. indices=list
 
 
    def update(self, atoms):
	if (not self.converged or len(self.numbers) != len(atoms) or (self.numbers != atoms.get_atomic_numbers()).any()):
	  self.initialize(atoms)
	  self.calculate(atoms)
        elif ((self.positions != atoms.get_positions()).any() or (self.pbc != atoms.get_pbc()).any() or (self.cell != atoms.get_cell()).any()):
	  self.calculate(atoms)
            
            
    def initialize(self, atoms):
        self.converged = False
	self.numbers = atoms.get_atomic_numbers().copy() # list of atomic numbers
        self.species = []
        for a, Z in enumerate(self.numbers):
            if Z not in self.species:
                self.species.append(Z)
                
        
    def get_potential_energy(self, atoms, force_consistent=False):
        self.update(atoms)
        return self.etotal
        #if force_consistent:
	#return self.efree
        #else:
	## Energy extrapolated to zero Kelvin:
	#return  (self.etotal + self.efree) / 2
            
            
    def get_forces(self, atoms):
        self.update(atoms)
        return self.forces.copy()
        
        
    def get_stress(self, atoms):
        self.update(atoms)
        return self.stress.copy()
        # return None
        
        
    def get_dipole_moment(self, atoms):
        """Returns total dipole moment of the system."""
        # self.update(atoms)
        # return self.dipole
        return None
        
        
    def read_dipole(self):
	pass
        #dipolemoment = np.zeros([1, 3])
        #for line in open(self.label + '.txt', 'r'):
            #if line.rfind('Electric dipole (Debye)') > -1:
                #dipolemoment = np.array([float(f) for f in line.split()[5:8]])
        ##debye to e*Ang (the units of VASP)
        #dipolemoment = dipolemoment*0.2081943482534
        #return dipolemoment
        
        
    def cleanfiles(self):
	files=[
	  "forces*",
	  "cp2k*.out",
	  "cp2k*.inp",
	  "*.restart*",
	  "*.ener*",
	  "*.xyz*"
	  ]
	  
	for f in files:
	  comm="rm -f "+self.dire+"/"+f
	  print ">",comm
	  os.system(comm)
	  
	  
    def collectfiles(self):
	files=[
	  ["cp2k.out","cp2k.log"], # tail the cp2k outputfile into cp2k.log
	  ["cp2k-HOMO_centers_s1.data","homo_centers.log"]
	  ]
	  
	for f in files:
	  comm='echo "** ASE STEP '+str(self.count)+' **" >> '+self.dire+"/"+f[1]
	  print ">",comm
	  os.system(comm)
	  
	  comm="cat "+self.dire+"/"+f[0]+" >>"+self.dire+"/"+f[1]
	  print ">",comm
	  os.system(comm)
	  
	  
    def write_input(self, atoms):
      import cp2ktools
      # manipulate cp2k parameters so that cp2k becomes a mere energy/force calculator:
      if (self.relax):
	cp2ktools.set_input(self.pars,["GLOBAL"],"RUN_TYPE","GEO_OPT")
	cp2ktools.set_input(self.pars,["FORCE_EVAL","PRINT","FORCES"],"FILENAME"," =forces")
      else:
	cp2ktools.set_input(self.pars,["GLOBAL"],"RUN_TYPE","ENERGY_FORCE")
	cp2ktools.set_input(self.pars,["FORCE_EVAL","PRINT","FORCES"],"FILENAME"," =forces")
      #if (self.count>1):
      #	cp2ktools.set_input(self.pars,["FORCE_EVAL","DFT","SCF"],"SCF_GUESS","HISTORY_RESTART")
      #	cp2ktools.set_input(self.pars,["FORCE_EVAL","DFT","SCF"],"MAX_SCF_HISTORY","10")
      # .. for some reason cp2k makes only one scf step when these pars are used..?
      
      # manipulate self.pars to have correct atomic positions and velocities..
      cp2ktools.setposvel(self.pars,atoms,groups=self.groups)
      # set the cell..
      cell=atoms.get_cell()
      st=str(cell[0][0])+" "+str(cell[1][1])+" "+str(cell[2][2])
      cp2ktools.set_input(self.pars,["FORCE_EVAL","SUBSYS","CELL"],"ABC",st)
      # check if velocities available..
      lines=cp2ktools.getlines(self.pars)
      # for l in lines:
      #  print l
      ff=self.dire+"/"+self.inputfile
      # print "ff",ff
      cp2ktools.writefile(ff,lines)
      
      
    def run(self):
      # di = directory
      inputfile=self.inputfile
      outputfile=self.outputfile
      exe=os.environ["cp2kexe"] 
      comm="cd "+self.dire+"; "+exe+" "+self.dire+"/"+inputfile+" &>"+self.dire+"/"+outputfile
      print ">",comm
      os.system(comm)
    
    
    def setpos(self, atoms):
	self.positions = atoms.get_positions().copy()
    
    def calculate(self, atoms):
        self.positions = atoms.get_positions().copy()
        self.cell = atoms.get_cell().copy()
        self.pbc = atoms.get_pbc().copy()
        
        self.cleanfiles()
        
        self.forces=np.empty((len(atoms), 3))
        
        self.write_input(atoms)
        
        # exefile = os.environ['cp2kexe']
        # locals = {'label': self.label}
        # execfile(exefile, {}, locals)
        # exitcode = locals['exitcode']
        
        self.run()
        
        # stop
        # self.debugsimulate()
        
        # print "atoms:",len(atoms)
        self.read()
        
        self.collectfiles()
        
        # atoms_structout = read_struct('%s.STRUCT_OUT' % self.label)
        # atoms.cell = atoms_structout.cell
        # atoms.positions = atoms_structout.positions
        self.converged = True
        
        self.count=self.count+1
        
        
        
    def read_force_sec(self,sec,atns):
      s=0
      # --------- FORCES --------
      # self.forces=np.array([0,0,0])
      text = open(self.dire+'forces', 'r').read().lower()
      lines = iter(text.split('\n'))
      # self.forces = np.zeros((len(atoms), 3))
      stc=0
      cc=0
      atn=0
      inside=False
      for line in lines:
	if (line.startswith(' sum of')):
	  inside=False 
	  # print "OUT"
	if (inside):
	  lis=line.split()
	  # print "lis=",lis
	  forcetmp=np.array([float(lis[-3]),float(lis[-2]),float(lis[-1])])
	  # print "forcetmp=",forcetmp
	  # self.forces=np.vstack((self.forces,forcetmp))
	  # print ">"+line
	  # print "atn",atn
	  # print "force=",forcetmp
	  ind=atns[atn]
	  self.forces[ind]=self.forces[ind]+copy.deepcopy(forcetmp)
	  atn=atn+1
	# Atom   Kind   Element          X              Y              Z
	if (line.startswith(' # atom   kind')):
	  s=s+1
	  if (s==sec):
	    inside=True
	cc=cc+1

  
        
	
	
    def readtraj(self):
      fname=self.dire+'cp2k-pos-1.xyz'
      tmptraj=io.read(fname)
      self.positions=tmptraj.positions.copy()
      
      return self.positions
      
      
    def debugsimulate(self):
      st="cp -f /home/sampsa/nanotubes/ice_cp2k/cp2k_spc_epp1/* /home/sampsa/nanotubes/ice_cp2k/cp2k_spc_epp1_save/"
      print ">",st
      os.system(st)

def read_cp2k_energy(output_file):
    text = open(output_file, 'r').read().lower()
    lines = iter(text.split('\n'))
    nrj=None
    scf = None
    for line in lines:
        if (line.startswith(' energy|')):
            nrj = float(line.split()[-1]) * Hartree
        if (line.startswith('  *** scf run converged in')):
            scf = True
            print "mycalculators, cp2k> scf run converged"
        if (line.startswith('  *** scf run not converged')):
            scf = False
            print "mycalculators, cp2k> scf run NOT converged"
        if (scf == None):
            pass
        elif (scf == False):
            nrj = None
            print "mycalculators, cp2k> setting energy to None"
    return nrj
	

      


def testmolpro():
  import asecoords
  
  xyz=[
    ["H",0,0,0],
    ["H",0,0,1]
    ]
  
  calc=molpro()
  
  tryout=asecoords.geom2atoms(xyz)
  cell=[1,1,1]
  tryout.set_cell(cell) 
  tryout.set_pbc((False, False, False))
  
  tryout.set_calculator(calc)
  
  # tryout.get_total_energy()
  
  # tryout.calc.initialize(tryout)
  # tryout.calc.read()
  
  filename="dummy"
  dyn = QuasiNewton(tryout, logfile=filename+'.log', trajectory=filename+'.traj')
  dyn.run(fmax=0.03)


def tmtests():
  # 1. x2t .xyz > coord

  # 2.
  seq=[
    "","kokkelis","aa coord","*", # input title, etc.
    "no", # dont use internal coords
    "b","all cc-pVDZ","*", # basis set definition
    "eht", "","","",# occupation numbers from guess
    # after this it takes a while..
    "*" # exit
    ]

  output=runseq("define",seq) # testing the turbomole input generator..
  print output

  lines=read_file("control")

  subsline(lines,"$scfiterlimit","$scfiterlimit 150") # change some def pars

  lines.insert(-1,"$denconv 1.d-7")
  dumplines("control",lines)

  # 3. insert line "$denconv 1.d-7" into file

  # 4. dscf

  # 5. mp2pred -g

  # 6. mpgrad

  # 7. goto 4

def tmclasstest():
  import ice04structs
  import asecoords
  
  clean=False
  # di="/wrk/sriikone/runs/turbomoletests/molpro_h2o"
  di="/home/sampsa/nanotubes/ice04/turbotmp"
  
  tm=turbomole(dirname=di,library="/wrk/sriikone/turbomole",clean=clean)
  xyz=ice04structs.h2o
  print "xyz=",xyz
  g=asecoords.geom2atoms(xyz)
  print "g=",g
  print "pos=",g.positions
  
  # tm.cleanfiles()
  # tm.cleaninput()
  
  # tm.write_coords(g)
 
  # tm.create_control(g,sections=[1,2])
 
  # tm.calculate(g)
  # tm.copy_control("h2o")
  tm.outputs=["turbomole.out.24.0","turbomole.out.24.1"]
  tm.read_energy(); 
  print "energy=", tm.etotal
  print "dipole moment=", tm.dipole_moment
  
  tm.read_forces(g)


def examples():
  # here we assume that ..
  # - your ase geometry object is called "tryout"
  # - you have said "import mycalculators as me"
  # - "rank" has the number of your MPI process
  #
  
  charge=0 # if your system is charged..
  dirname="kokkelis" # where the calculation is to be run
  
  prog="turbomole" # choose either "turbomole" (mp2) or "turbodft" (pbe, blyp, etc.)
  debug=False
  
  # *** default basis sets..***
  bas="tzvpd" # def2-tzvpd # this is the "state of the art" basis set
  # bas="szvpd" # this is cheaper basis set .. use with care
  
  # *** "hacked" basis sets .. ***
  # bas="tzvpd+" # no diffuse basis functions in aux basis 
  # bas="tzvpd++" # very simple (svp) aux basis
  
  # *** if you use "turbodft", choose correct functional .. ***
  # turbofunc="b3-lyp"
  # turbofunc="pbe"
  
  if (prog=="turbomole" and debug==False):
    tasks=["dscf","ricc2"]
    if (bas=="tzvpd"):
      basis="def2-TZVPD"; auxbasis="def2-TZVPD"; jkauxbasis="aug-cc-pVTZ";
    elif (bas=="szvpd"):
      basis="def2-SVPD"; auxbasis="def2-SVP"; jkauxbasis="def2-SVP"; # no auxbasis for def2-SVPD ..! use def2-SVP instead? # "svpd"
    elif (bas=="tzvpd+"):
      basis="def2-TZVPD"; auxbasis="def2-TZVP"; jkauxbasis="def2-TZVP"; # "tzvpd+"
    elif (bas=="tzvpd++"):
      basis="def2-TZVPD"; auxbasis="def2-SVP"; jkauxbasis="def2-SVP"; # "tzvpd++"
    
    calc=me.turbomole(dirname=dirname,tasks=tasks,basis=basis, auxbasis=auxbasis, jkauxbasis=jkauxbasis) 
    
    if (rank==0):
      calc.create_control(tryout,sections=[2],pars={"charge":charge})
	
  elif (prog=="turbodft" and debug==False):
    # ******************
    tasks=["dscf","grad"] # dscf, use section 0
    sections=[0]
    # ****************
    # tasks=["ridft","rdgrad"] # ridft, use section 1
    # sections=[1]
    if (bas=="szvpd"):
      turboshit=False; basis="def2-SVPD"; auxbasis="def2-SVPD"; jkauxbasis="def2-SVPD";
    elif (bas=="tzvpd"):
      turboshit=True; basis="def2-TZVPD"; auxbasis="aug-cc-pVTZ"; jkauxbasis="aug-cc-pVTZ"; # should probably always use this.. # def2 not in murska .. now it is!
    # turboshit=False; basis="aug-cc-pVTZ"; auxbasis="aug-cc-pVTZ"; jkauxbasis="aug-cc-pVTZ"; # this is *very* slow
    # turboshit=False; basis="def2-SVP"; auxbasis="def2-SVP"; jkauxbasis="def2-SVP"; # only good for testing..? for nothing else!
    # ******************
    
    morepars={"turbofunc":turbofunc,"rimem":800,"charge":charge} # some turbomole dft parameters
    if (turboshit):
      morepars["shit"]=True
    # pars={"dirname":dirname,"tasks":tasks,"sections":sections,"morepars":morepars} # encapsulate parameteres for a neb run..
    calc=me.turbomole(dirname=dirname,tasks=tasks, basis=basis, auxbasis=auxbasis, jkauxbasis=jkauxbasis)
    
    if (rank==0):
      calc.create_control(tryout,sections=sections,pars=morepars)
    # .. 340 MB for ri integrals.. (2/3 of 512MB as instructed in vuori..)
    # (2/3) of 2000 .. 
  
  
  """
  and finally, an example batch job for vuori:
  remember to modify lines that end with "!!!"
  ---------------------------------
  #!/bin/bash -l
  #SBATCH -p parallel
  #SBATCH -J kokkelis
  #SBATCH -n 24
  #SBATCH --ntasks-per-node=12
  #SBATCH --mem-per-cpu=1500
  #SBATCH -t 72:00:00
  #SBATCH -e vuorijob.err%J
  #SBATCH -o vuorijob.out%J
  module load gpaw
  export PYTHONPATH=$PYTHONPATH:/wrk/sriikone/python  !!! .. your python library
  export PARA_ARCH=MPI
  export MPI_REMSH=/usr/bin/ssh
  export HOSTS_FILE=$PWD/hostfile.0.35696697803 !!! .. your host file
  export PARNODES=$SLURM_NPROCS
  rm -f $HOSTS_FILE
  srun hostname > $HOSTS_FILE
  cat $HOSTS_FILE
  module load turbomole/6.31
  cd /wrk/sriikone/python/run  !!! .. where your python script is
  python your_python_script.py !!! .. your own python/ase script
  ---------------------------------
  """
  
  
def test_cp2k():
  # di="/home/sampsa/nanotubes/ice_cp2k/cp2k_spc_epp1_save/"
  # di2="/home/sampsa/nanotubes/ice_cp2k/cp2k_spc_epp1/"
  # di3="/wrk/sriikone/runs/cp2k_vdw_qmmm_test_qmmm/"
  di4="/wrk/sriikone/runs/temppi/"
  
  # calcu=cp2k(di3,qms=[145,146])
  calcu=cp2k(di4)
  
  # ats=io.read(di3+"/cp2k.traj")
  ats=io.read(di4+"/aserun.traj")
  
  calcu.setpos(ats)
  # inds=range(0,len(ats))
  # print "inds",inds
  # calcu.setgroups({'H2O':inds})
  calcu.read()
  # print "nrj=",calcu.energy
  print "nrj=",calcu.etotal
  print "forces=",calcu.forces
  # calcu.get_potential_energy(ats)
  
# tests()
# tmtests()
# tmclasstest()

# test_cp2k()


