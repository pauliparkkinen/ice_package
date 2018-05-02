import math
import copy
import string
import random
import numpy
from scipy import spatial
import transformations
import numpy.linalg as lg
import array as ar

# **** 553 / automatic generation ****
y_step=0.353553390593273
x_step=0.6127/3
z_step=0
# **** graphene ******
graphene_basis=[
['C',0,0],
['C',1.0/(2.0*math.sqrt(3.0)),0.5]
]

graphene_basis2=[
['C', 0.0, 0.0, 0.0], 
['C', -0.5, 0.28867513459481298, 0.0]
]

bn_basis=[
['B',0,0],
['N',1.0/(2.0*math.sqrt(3.0)),0.5]
]
# different vectors for the triangular lattice..
trired0=[[0,1],[math.sqrt(3.0)/2.0,0.5]]
trired_neg=[[-math.sqrt(3.0)/2.0,0.5],[-math.sqrt(3.0)/2.0,-0.5]]
trired_pos=[[math.sqrt(3.0)/2.0,0.5],[math.sqrt(3.0)/2.0,-0.5]]
trired_pos2=[[math.sqrt(3.0)/2.0,-0.5],[math.sqrt(3.0)/2.0,0.5]]
trired_sym=[[0.5,math.sqrt(3.0)/2.0],[-0.5,math.sqrt(3.0)/2.0]] # graphene_basis2
trired_sym2=[[-0.5,math.sqrt(3.0)/2.0],[0.5,math.sqrt(3.0)/2.0]] # graphene_basis2
# *****************
# Silicon surface 1x1 lattice vectors..
Si_1x1=[
[0.0,1.0/math.sqrt(2)],
[math.sqrt(3.0/8.0),math.sqrt(1.0/8.0)]
]


def runfile(filename):
	f=open(filename,'r')
	lines=f.readlines()
	f.close()
	f=[]
	
	for l in lines:
		# print ">",l
		exec(l)
	lines=[]
	# print "locals=",locals()
	dic=locals()
	# remove confusing extra variables..
	dic.pop("f")
	dic.pop("lines")
	dic.pop("l")
	dic.pop("filename")
	# stop
	# return locals() # stupid extra variables
	# print "dic=",dic
	# stop
	return dic


def writepy(filename, dic):
  f=open(filename,'w')
  lines=[]
  for key in dic.iterkeys():
    if (type(dic[key])==type("ss")):
      f.write(key+"='"+str(dic[key])+"'\n")
    else:
      f.write(key+"="+str(dic[key])+"\n")
  f.close()


def printlist(li):
  c=0
  for l in li:
    print c,l
    c=c+1


def mul(vec,fac):
    vec2=copy.deepcopy(vec)
    c=0
    for m in vec:
        vec2[c]=vec[c]*fac
        c=c+1

    return vec2
    
def muls(vecs,fac):
	vecs2=copy.deepcopy(vecs)
	c=0
	for vec in vecs:
		vecs2[c]=mul(vec,fac)
		c=c+1

	return vecs2
	

def geomcmpz(g1,g2):
	c=3
	if ((g1[c]>g2[c])):
		return 1
	if ((g1[c]==g2[c])):
		return 0
	if ((g1[c]<g2[c])):
		return -1
	
	
def geomcmpzinv(g1,g2):
	c=3
	if ((g1[c]<g2[c])):
		return 1
	if ((g1[c]==g2[c])):
		return 0
	if ((g1[c]>g2[c])):
		return -1


def geomcmpy(g1,g2):
	c=2
	if ((g1[c]>g2[c])):
		return 1
	if ((g1[c]==g2[c])):
		return 0
	if ((g1[c]<g2[c])):
		return -1
		
def geomcmpx(g1,g2):
	c=1
	if ((g1[c]>g2[c])):
		return 1
	if ((g1[c]==g2[c])):
		return 0
	if ((g1[c]<g2[c])):
		return -1
	
def geomcmpn(g1,g2):
	tag="d"
	if ((g1[4][tag]>g2[4][tag])):
		return 1
	if ((g1[4][tag]==g2[4][tag])):
		return 0
	if ((g1[4][tag]<g2[4][tag])):
		return -1
	
	
def geomcmpspec1(g1,g2):
	if ((g1[0]!="Fe") and (g2[0]=="Fe")):
		return -1
	if ((g1[0]=="Fe") and (g2[0]=="Fe")):
		return 0
	if ((g1[0]=="Fe") and (g2[0]!="Fe")):
		return 1
	else:
		return 0


def geomcmpspec2(g1,g2):
	if ((g1[0]!="Fe") and (g2[0]=="Fe")):
		return 1
	if ((g1[0]=="Fe") and (g2[0]=="Fe")):
		return 0
	if ((g1[0]=="Fe") and (g2[0]!="Fe")):
		return -1
	else:
		return 0

def geomcmpspecCu(g1,g2):
	if ((g1[0]!="Cu") and (g2[0]=="Cu")):
		return 1
	if ((g1[0]=="Cu") and (g2[0]=="Cu")):
		return 0
	if ((g1[0]=="Cu") and (g2[0]!="Cu")):
		return -1
	else:
		return 0

def geomcmpnum(g1,g2):
	if ((g1[4]["num"]>g2[4]["num"])):
		return 1
	if ((g1[4]["num"]==g2[4]["num"])):
		return 0
	if ((g1[4]["num"]<g2[4]["num"])):
		return -1


def sortgeom(geom):
	geom.sort(geomcmpx)
	geom.sort(geomcmpy)
	geom.sort(geomcmpz)

def sortgeom2(geom):
	geom.sort(geomcmpx)
	geom.sort(geomcmpy)
	geom.sort(geomcmpzinv)

def sortgeom3(geom):
	geom.sort(geomcmpy)
	geom.sort(geomcmpx)
	geom.sort(geomcmpzinv)

def sortgeom4(geom):
	geom.sort(geomcmpzinv)

def sortgeomn(geom):
	geom.sort(geomcmpn)
	
def sortgeomspec(geom):
	geom.sort(geomcmpspec1)

def sortgeomspec2(geom):
	geom.sort(geomcmpspec2)

def sortgeomspecCu(geom):
	geom.sort(geomcmpspecCu)

def sortgeomnum(geom):
	geom.sort(geomcmpnum)



def centersort(geom,mycent=[]):
	if (mycent==[]):
		cent=find_center(geom)
		cent.insert(0,"H")
	else:
		cent=mycent
	for g in geom:
		if (len(g)>4):
			pass
		else:
			g.append(dict())
		# print cent
		# print g[0:4]
		# stop
		g[4]["d"]=distance(cent,g[0:4])
	
	#for g in geom:
	#	print ">",g
	#print
	sortgeomn(geom)	
	#for g in geom:
	#	print ">>",g
	c=0
	for g in geom:
		# print ">>",g[4]
		# g[4].pop("d")
		# geom[c][4].pop("d") # does not work..
		c=c+1


def multiply(geom,vectors):
    # vectors of the form: [[x,y,z],[x,y,z],..]
    newgeom=[]

    # print "vectors=",vectors

    for i in vectors:
        # print "i=",i

        gc=0
        for k in geom:
            gc=gc+1
            # print "gc=",gc
            g=[]
            g=copy.deepcopy(k)

            c=0
            for j in i: # vector components
                c=c+1 # coordinate count
                # print "c=",c,j
                
                g[c]=g[c]+j 

            newgeom.append(g)


    return newgeom


def reorder(geom,li):
	newgeom=[]
	c=0
	for g in geom:
		newgeom.append(geom[li[c]])
		c=c+1
	return newgeom


def switchat(geom,n1,n2):
	tmp=copy.deepcopy(geom[n2])
	geom[n2]=copy.deepcopy(geom[n1])
	geom[n1]=tmp


def rel2abs(geom,scale):
    geom2=[]
    c=0
    for g in geom:	    
        g2=[]
        g2.append(g[0])
	# for i in range(1,len(g)-1):
        g2.append(g[1]*scale)
        g2.append(g[2]*scale)
        g2.append(g[3]*scale)
	if (len(g)>4):
		g2.append(g[4])
        geom2.append(g2)

    return geom2
    
 
def scale(geom,scales):
	geom2=[]
    	c=0
    	for g in geom:
        	g2=[]
        	g2.append(g[0])
		# for i in range(1,len(g)-1):
        	g2.append(g[1]*scales[0])
        	g2.append(g[2]*scales[1])
        	g2.append(g[3]*scales[2])
        	geom2.append(g2)

	return geom2
	
        
def writexmol(file,crds):
    	f=open(file,'w')
    	f.write(str(len(crds))+"\n\n")
    	for l in crds:
        	f.write(l+"\n")
    	f.close()

def writefile(file,crds,typ="w"):
	f=open(file,typ)
    	for l in crds:
        	f.write(str(l)+"\n") # added str()
    	f.close()
   
   
def writedict(file,di):
  lines=[]
  for d in di.iterkeys():
    lines.append(d+"="+str(di[d]))
  writefile(file,lines)
  
   
def genxmol(crds):
	lines=[]
	lines.append(str(len(crds)))
	lines.append(" ")
	for l in crds:
       		lines.append(l)
	return lines
	
    
def writexmolmany(file,geoms,scale=1.0):
	f=open(file,'w')
	for geom in geoms:
		f.write(str(len(geom))+"\n\n")
		for g in geom:
			ss=g[0]+" "+str(g[1]*scale)+" "+str(g[2]*scale)+" "+str(g[3]*scale)
			f.write(ss+"\n")
    	f.close()


def appendxmol(file,geom,scale=1.0):
    f=open(file,'a')
    f.write(str(len(geom))+"\n\n")
    for g in geom:
	    ss=g[0]+" "+str(g[1]*scale)+" "+str(g[2]*scale)+" "+str(g[3]*scale)
	    f.write(ss+"\n")
    f.close()

def writexmolstep(f,geom,scale=1.0):
    # append to already opened xmol file
    f.write(str(len(geom))+"\n\n")
    for g in geom:
	    ss=g[0]+" "+str(g[1]*scale)+" "+str(g[2]*scale)+" "+str(g[3]*scale)
	    f.write(ss+"\n")
    


def readxmol(file, maxsteps=0):
	# read an xmol animation..
	
	steps=[]
	if (type(file).__name__=="str"):
	  f=open(file,'r')
	  lines=f.readlines()
	  f.close()
	else:
	  # its a list
	  lines=file
	  
	newlines=[]
	for l in lines:
	        # print "l>",l
		# newlines.append(l[0:-1]) # wtf?
		newlines.append(l)
		
	# print ">>>>",newlines
	c=0
	step=1
	for l in newlines:
		# print "l>>",l
		c=c+1
		if (c==1):
			n=int(l.strip())
		if (c==2):
			m=0
			crd=[]
		if (c>2):
			m=m+1
			if (m>n):
				c=1
				# move coordinates to the buffer..
				# print "*** STEP ***",step
				# print crd
				steps.append(crd)
				# print 
				step=step+1
			else:
				cr=l.split()
				crd.append([cr[0],float(cr[1]),float(cr[2]),float(cr[3])])
	
		if (c==1):
			n=int(l)
		
	steps.append(crd)
	
	# for s in steps:
	#	print s
	return steps


def readxmolstep(fil):
	# read a step from xmol animation..
	# file has been opened first..
	c=0
	for l in fil:
		if (l==None):
		  return None # end of file reached ..
		# print "l>>",l
		c=c+1
		if (c==1):
			n=int(l.strip())
		if (c==2):
			m=0
			crd=[]
		if (c>2):
			m=m+1
			if (m>n):
				print "something went wrong!"
				return -1
			else:
				cr=l.split()
				crd.append([cr[0],float(cr[1]),float(cr[2]),float(cr[3])])
				if (m==n):
				  return crd
		if (c==1):
			n=int(l)


def jumpxmolsteps(fil,m):
  # we are at step N and we go to step N+M
  for i in range(0,m):
    ste=readxmolstep(fil)
  

def writexmolbin(f,geom):
  # import array as ar
  # writes xmol binary file
  xs=[] # a long list of coordinates..
  for g in geom:
    xs=xs+g[1:]
  xsa=ar.array('d',xs)
  # print ">>",xsa
  f.write(xsa)


def readxmolbin(f,geom):
  # import array as ar
  # reads a step from xmol binary file
  # xs=[] # a long list of coordinates..
  
  # xsa=ar.array('d')
  # xsa.fromfile(f,ns)
  
  # print "geom>",geom
  newgeom=copy.copy(geom)
  
  for g in newgeom:
    try:
      xsa=ar.array('d')
      xsa.fromfile(f,3)
      # print "xsa>",xsa
      g[1:]=list(xsa)
      # print ">>",g
    except EOFError:
      print "EOF"
      return None
    
  return newgeom


def readbin(f,m):
  # f = binfile
  # m = number of atoms
  xsa=ar.array('d')
  xsa.fromfile(f,m*3) # natoms*3  
  newt=numpy.frombuffer(xsa)   
  newt=numpy.reshape(newt,(m,3))
    
  return newt


def binmultinum(newt,v1,v2):
  # multiplies unit cell, adds numbering
  nums=numpy.array(range(len(newt)))
  nn0=numpy.atleast_2d(nums).transpose()
  
  cc=numpy.hstack((newt,nn0)) # add column with numbering
  # cc=crd.copy()
  cc=numpy.vstack((cc,cc+v1,cc+2*v1,cc-v1,cc-2*v1))
  cc=numpy.vstack((cc,cc+v2,cc+2*v2,cc-v2,cc-2*v2))

  nums=numpy.array(range(len(cc)))
  nn=numpy.atleast_2d(nums).transpose()
  cc=numpy.hstack((cc,nn))
  
  # columns: x,y,z,unit cell numbering,supercell numbering
  return cc
  

def binmulti(newt,v1,v2):
  # multiplies unit cell
  cc=newt
  cc=numpy.vstack((cc,cc+v1,cc+2*v1,cc-v1,cc-2*v1))
  cc=numpy.vstack((cc,cc+v2,cc+2*v2,cc-v2,cc-2*v2))

  # columns: x,y,z,unit cell numbering,supercell numbering
  return cc


def bintable(newt,newtt):
  nntab=numpy.zeros((newt.shape[0],newtt.shape[0]))
  for i in range(newt.shape[0]):
    for j in range(i+1,newtt.shape[0]):
      dis=numpy.sqrt(numpy.sum((newt[i,0:3]-newtt[j,0:3])**2))
      nntab[i,j]=dis.copy()
      # nntab[j,i]=dis.copy()
  return nntab
  
  
def binearest(newt,newtt,nn):
  nntab=numpy.zeros((newt.shape[0],nn))
  ditab=numpy.zeros((newt.shape[0],nn))
  
  nntab[:,:]=-1
  ditab[:,:]=1000.0
  
  print "calculating distance table.."
  # netab=bintable(newt,newtt)
  netab=spatial.distance.cdist(newt,newtt)
  
  for i in range(newt.shape[0]):
    # for i in [(220-1)]:
    # print ">i",i*3+1
    print i
    for j in range(newtt.shape[0]):
      # dis=numpy.sqrt(numpy.sum((newt[i,0:3]-newtt[j,0:3])**2))
      dis=netab[i,j]
      if (dis>0.0 and dis<=20.0):
	# print ">>j",j*3+1
	# print ">>d",dis
	# ind=numpy.where(dis<ditab[i,:])[0]
	tem=ditab[i,:]-dis
	# print ">>tem",tem
	v=numpy.max(tem)
	# print ">>v",v
	if (v>0):
	  ind=numpy.where(tem==v)[0][0]
	  # print ">>ind",ind
	  ditab[i,ind]=dis.copy()
	  nntab[i,ind]=copy.copy(j)
	  # print nntab[i,:]*3++1,ditab[i,:]
	# print
    # stop
  
  # for i in range(nntab.shape[0]):
  #  print (i*3)+1,ditab[i,:],nntab[i,:]*3+1
  # stop
  
  return nntab, ditab


def writexmolbinhead(f,geom):
  # import array as ar
  # writes xmol binary file header
  ns=[] # number of atoms
  xs=[] # list of atomic types
  
  # write number of atoms
  ns=[len(geom)]
  nsa=ar.array('i',ns)
  f.write(nsa)
  
  #st=""
  #for g in geom:
  #  spec=g[0].lstrip().rstrip()
  #  if (len(spec)<2):
  #    st=st+spec+' '
  #  else:
  #    st=st+spec
  # print st
  # xsa=ar.array('c',st)
  
  xsa=ar.array('c')
  cc=0
  for g in geom:
    spec=g[0].lstrip().rstrip()
    xsa.append(spec[0])
    if (len(spec)>1):
      xsa.append(spec[1])
    else:
      xsa.append(' ')
      
  f.write(xsa)
  

def readxmolbinhead(f):
  nsa=ar.array('i')
  nsa.fromfile(f,1)
  ars=ar.array('c')
  ars.fromfile(f,nsa[0]*2)
  
  # print nsa,ars
  # print ars.itemsize
  
  # so, what is the length of 
  # one xmol step in the file in bytes?
  # print "ars.itemsize",ars.itemsize
  # print "nsa[0]",nsa[0]
  blocksize=(nsa[0]*3*8) # +ars.itemsize*nsa[0]*2)
  
  
  specs=list(ars)
  geom=[]
  
  for i in range(0,len(specs),2):
    sp=(specs[i]+specs[i+1]).rstrip()
    geom.append([sp,None,None,None])
    
  return (geom, blocksize)


def jumpbinsteps(f,blocksize,n,typ=1):
  # so.. we were between steps N and N+1 and now between N+n and N+n+1
  # if you want every m:th step, take n=m-2
  try:
    f.seek(blocksize*n, typ)
  except EOFError:
    print "EOF"
    return False
  return True
  # typ = 2 = relative to file's end
  # manual: f.seek(offset, from_what)
  # "The position is computed from adding offset to a reference point; the reference point is selected by the from_what argument."
  # 1=current pos
  # i.e. new position = current position + ofs
  #.. new step = current step + n (i.e. skips n steps)

def binsteps(fname):
  import os
  siz=os.path.getsize(fname)
  f=open(fname,'rb')
  (geom, bsize)=readxmolbinhead(f)
  f.close()
  n=len(geom)
  
  geomsize=n*8*3 # atoms * 64-bit d * xyz
  charsize=n*1*2 # atoms * byte * 2 (each element name occupies two bytes)
  
  # print "siz-charsize",siz-charsize
  # print "geomsize",geomsize
  # .. of course.. the integer in the beginning..
  charsize=charsize+4 
  # print "mod",(siz-charsize)%geomsize
  nsteps=float(siz-charsize)/float(geomsize)
  
  return nsteps
  
  
def read_xmol_final(file):
  # reads the very last xmol step in laaarge .xmol files
  pass



def writecoords(file,crds):

    f=open(file,'w')
    for l in crds:
	# print ">>>",l
        f.write(l+"\n")
    f.close()

def appendfile(file,crds):

    f=open(file,'a')
    line=""
    for l in crds:
	line=line+" "+str(l)
    f.write(line+"\n")
    f.close()


def coords(arg,arg2):
    global y_step, x_step, z_step

    n=arg[0]
    m=arg[1]
    l=arg[2]

    ofsx=arg2[0]
    ofsy=arg2[1]
    ofsz=arg2[2]

    dx=x_step
    dy=y_step
    dz=z_step

    x=n*dx
    y=m*dy
    z=l*dz


def coord(arg):
    global y_step, x_step
    # first layer coordinates for Si553/Au
    # n (down=1, or up=2 row)
    # m, column (1..n)
    n=arg[0]
    m=arg[1]
    dy=y_step
    dx=x_step
    y=(n-1)*dy
    x=(m-1)*dx

    return [y,x]
 

def fixgeomat(arg,atnum):
	x=arg[atnum][1]
	y=arg[atnum][2]
	z=arg[atnum][3]
	for l in arg:
		l[1]=l[1]-x
		l[2]=l[2]-y
		l[3]=l[3]-z


def fixgeom(arg,x=True,y=True,z=True):
    # set the minimum coordinate to zero
    xmin=100000
    ymin=100000
    zmin=100000
    c=0
    for l in arg:
        # c=c+1
        # print c
        xmin=min(l[1],xmin)
        ymin=min(l[2],ymin)
        zmin=min(l[3],zmin)
    
    # print xmin,ymin,zmin
    c=0
    for l in arg:
        # c=c+1
        # print ">",c
        # l[1]=l[1]-xmin
        # l[2]=l[2]-ymin
        # l[3]=l[3]-zmin
        
	if (x):
		l[1]=l[1]-xmin
	if (y):
		l[2]=l[2]-ymin
	if (z):
		l[3]=l[3]-zmin


def findzmax(geom):
  zmax=-1000000
  for g in geom:
    zmax=max(g[3],zmax)
    
  return zmax
  
	
def obj2geom(arg,ofss):
    # binary chain to first layer coordinates mapping for Si553/Au
    # 0=empty, 1=si down, 2=si up, 3=Au down, 4=Au up
    line=[]
    g=[]

    ofsx=1.020620
    ofsy=0.660660
    z=2.581020

    ofsx=ofss[0]
    ofsy=ofss[1]
    z=ofss[2]
    
    ix=0
    arg2=[]
    for i in arg:
        if ((type(i)==type([]))==False):
            # print i,[i]
            arg2.append([i])
        # print arg2
        else:
            arg2.append(i)
    
    for i in arg2:
        ix=ix+1
        # print ix,i
        # print ">",i
        # if (len(i)==1):
        c=0
        for j in i:
            ## print j
##             dxx=0.0
##             if ((c==1),(ix<len(arg2)-1)==True,True):
##                 if (j==arg2[ix][0]):
##                     dxx=-x_step/4
##                     print "dxx=",dxx
            if (j>0):
                spec="Si"
                if (j>2):
                    spec="Au"
                    j=j-2

                a=coord([j,ix])
                # print spec,a[1]+ofsx,a[0]+ofsy,z
                
                g=[]
                g.append(spec)
                dx=0
                if (len(i)>1):
                    dx=x_step/4.0
                    if (c==0):
                        dx=-dx
                
                g.append(float(a[1]+ofsx+dx))
                g.append(float(a[0]+ofsy))
                g.append(float(z))
                line.append(g)
            c=c+1
            
    return line


def distance(geom,geom2):
    	import math
	
	#print "g",geom
	#print "g2",geom2
	su=0
	for i in range(1,min(len(geom),4)):
		#print i
		su=su+(geom[i]-geom2[i])**2
	d=math.sqrt(su)

    	return d
        

def correct(geom):
    import math
    global x_step
    # correct the bonds that have unphysically long lengths

    dx=0.05
    l_old=geom[0]
    for l in geom:
        if (distance(l,l_old)>2.0*x_step):

            x=l[1]-l_old[1]
            dy=l[2]-l_old[2]
            dz=l[3]-l_old[3]

            dx=math.sqrt((2.0*x_step)**2-dy**2-dz**2)
            dx=(x-dx)/2.0
            
            l[1]=l[1]-dx
            l_old[1]=l_old[1]+dx
        l_old=l


# *** for differents coordinate formats see examples() ***
def crd2siesta(line,species):
    li=line.rsplit(" ")
    # li=string.splitfields(line," ")
    # print "li=",li
    sp="?"
    for s in species:
        if (s[0]==li[0]):
            sp=s[1]

    out=str(li[1])+" "+str(li[2])+" "+str(li[3])+" "+str(sp)
    return out


def siesta2crd(line,species):
    li=line.split()
    # line=string.splitfields(li, " ")
    # print "li=",li
    sp="?"
    for s in species:
	# print s[1],li[3]
        if (str(s[1])==li[3]):
            sp=s[0]
	else:
		# print s[1],li[3]
		pass

    out=str(sp)+" "+str(li[0])+" "+str(li[1])+" "+str(li[2])
    return out


def crds2siesta(lines,species):
    siesta_lines=[]
    for s in lines:
        lin=crd2siesta(s,species)
        siesta_lines.append(lin)

    return siesta_lines


def siestas2crd(lines,species):
    coord_lines=[]
    # print "lines=",lines
    for s in lines:
	# print "s=",s
        lin=siesta2crd(s,species)
        coord_lines.append(lin)

    return coord_lines


def crd2geom(lines):
    geom=[]
    ss2=[]
    for s in lines:
        # ss=s.rsplit()
        ss=string.splitfields(s)
        # print ss
        # print ss[1]
        ss2=[]
        ss2.append(ss[0])
        ss2.append(float(ss[1]))
        ss2.append(float(ss[2]))
        ss2.append(float(ss[3]))
        geom.append(ss2)
    
    return(geom)


def geom2crd(geom):
    lines=[]
    # print ">>",geom
    for g in geom:
        ss=g[0]+" "+str(g[1])+" "+str(g[2])+" "+str(g[3])
        lines.append(ss)
    
    return(lines)
    

def get_geom(file):
    print ">>>>>> file=",file
    geom=[]
    
    f=open(file,'r')
    ok=1
    ins=0
    while (ok==1):
        st=f.readline()
        if (len(st)<1):
            ok=0
        if (ok==1):
            if (st=="\n"):
                ins=0

            st=st[:-1]
            # print ">",st
                
            if (ins==1):
                # print ">>",st
                sta=[]
                g=[]
                # sta=st.split()
                sta=string.splitfields(st)
                g.append(sta[4])
                g.append(float(sta[0]))
                g.append(float(sta[1]))
                g.append(float(sta[2]))
                geom.append(g)
                
            if (ins==0):
                if (st.find("outcoor: Relaxed atomic coordinates (scaled):")>-1):
                    # print "found!"
                    ins=1
    f.close()
    return(geom)


def get_harris(file):
    eharris=float(0.0)
    
    f=open(file,'r')
    ok=1
    while (ok==1):
        st=f.readline()
        if (len(st)<1):
            ok=0
        if (ok==1):
            # sp=st.rsplit()
            sp=string.splitfields(st)
            if (len(sp)==4):
#                print ">>>>",sp
                if ((sp[0],sp[1],sp[2])==('siesta:','Eharris','=')):
                    # print "!!!"
                    eharris=sp[3]

    return eharris


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


def readxyz(fil):
	return crd2geom(read_file(fil))

def get_block(line,blockname):
    lin=[]
    t1=0
    for s in line:
        # l=s.rsplit()
        l=string.splitfields(s)
        l.append(" ")
        l.append(" ")

        if ((l[0],l[1]) == ("%endblock",blockname)):
            t1=0

        if (t1==1):
            # print "** append"
            lin.append(s)

        if (t1==0):
            if ((l[0],l[1]) == ("%block",blockname)):
                # print "found"
                t1=1
    return lin


def find_neighbours(geom1,geom2,specs=[]):
    # find nearest neighbours of geom1 in geom2
    # geom1: just one coordinate, geom2: a list

    g=[]
    g2n=0
    for g2 in geom2:
        su=0
	for i in range(1,min(len(g2),4)):
        # for i in range(1,4):
            # print "i=",i
	    # print geom1[i],",",g2[i]
            su=su+(geom1[i]-g2[i])**2
        
	su=math.sqrt(su)
	if ((specs==[]) or (g2[0] in specs)):
        	g.append([su,g2n]) # list with: [distance, number]
        g2n=g2n+1

    g.sort()
    return g

    # for gg in g:
    #    print gg


def checkgeom(geom,tol=0.05):
  badlist=[]
  c=0
  for g in geom:
    d=c+1
    for g2 in geom[c+1:]:
      if (abs(distance(g,g2))<=tol):
	badlist.append([c,d])
      d=d+1
    c=c+1
    
  return badlist
  

def H_neighbours(geom,spec=["H"]):
    # find the nearest neighbours of the H-atoms

    li2=[]
    c=0
    for g in geom:
        # print c,g
        if (g[0] in spec):
            li=[]
            li=find_neighbours(g,geom)
            dummy=li.pop(0)
            # print "neighbour=",li[0],li[1],li[2]
            # print li[0][1]
            while (li[0] in spec):
                dummy=li.pop(0)
            
            li2.append(li[0][1])
        c=c+1

    return li2


def adinfo(g,nametag,val):
	# if an atom does not have the info block (fifth element)
	# then add one, otherwise just expand it
	if (len(g)<=4):
		g.append({})
	g[4][nametag]=val


def invlist(geom,li):
  # create list of all other atoms than in the list
  # = invert the list
  c=0
  lis=[]
  for g in geom:
    if (c in li):
      pass
    else:
      lis.append(c)
    c=c+1
  
  return lis



def fixsome(xyz,li):
	c=0
	for g in xyz:
		if (c in li):
			adinfo(g,"fixed",True)
		c=c+1

def fixall(xyz):
  for g in xyz:
    adinfo(g,"fixed",True)

def freeall(xyz):
  reminfos(xyz,"fixed")


def atrib2list(geom):
  li=[]
  for g in geom:
    li.append(g[4]["num"])
  return li


def addnums(geom):
  c=0
  for g in geom:
    # print "g",g
    adinfo(g,"num",c)
    c=c+1


def harvestnum(geom):
  lis=[]
  for g in geom:
    lis.append(g[4]["num"])
  return lis
  

def adinfo(g,nametag,val):
	# if an atom does not have the info block (fifth element)
	# then add one, otherwise just expand it
	if (len(g)<=4):
		g.append({})
	g[4][nametag]=val


def reminfo(g,tag):
	if (len(g)<=4):
		pass
	if (tag=="*"):
	  g.pop(4)
	else:
		if (g[4].has_key(tag)):
			# print ">>>",g[4][tag]
			g[4].pop(tag)
			# print ">",g[4]
			# stop



def adinfos(geom,nametag,val):
  for g in geom:
    adinfo(g,nametag,val)


def reminfos(geom,tag):
  for g in geom:
    reminfo(g,tag)




def tcl_paintatoms(li,color,adone=True,ofs=0):
	lines=[]
	for i in li:
		ii=i
		if (adone):
			ii=ii+1
		lines.append("atom color xmol xmol "+str(ii+ofs)+" "+color)
		
	return lines

def tcl_no_h_bonds(geom):
	lines=[]
	c1=0
	for g1 in geom:
		c2=0
		for g2 in geom:
			if (g1[0]=="H"):
				lines.append("edit bond break xmol xmol "+str(c2+1)+" xmol xmol "+str(c1+1))
			c2=c2+1
		c1=c1+1
	return lines
			
def tcl_no_bonds(geom,spec1,spec2):
	lines=[]
	c1=0
	for g1 in geom:
		c2=0
		for g2 in geom:
			if ((g1[0]==spec1) and (g2[0]==spec2)):
				# stop
				lines.append("edit bond break xmol xmol "+str(c2+1)+" xmol xmol "+str(c1+1))
			c2=c2+1
		c1=c1+1
	return lines	


def atoms2spheres(geom,scale=1.0,rad=0.2,color="green",spec=["H"]):
	lines=[]
	app=""
	for g in geom:
		if (g[0] in spec):
			lines.append("plot sphere "+str(g[1]*scale)+" "+str(g[2]*scale)+" "+str(g[3]*scale)+" "+str(rad)+" "+color+" "+app)
			app="append"
	return lines


def tcl_spheres(lis,rad=0.1,scale=1.0,color="green",append=False):
  lines=[]
  app=""
  if (append):
    app="append"
  for l in lis:
    lines.append("plot sphere "+str(l[0]*scale)+" "+str(l[1]*scale)+" "+str(l[2]*scale)+" "+str(rad)+" "+color+" "+app)
    app="append"
  return lines
  

def tcl_hideatoms(li,adone=True):
	lines=[]
	for i in li:
		ii=i
		if (adone):
			ii=ii+1
		lines.append("atom -display xmol xmol "+str(ii))
		
	return lines


def tcl_showatoms(li,adone=True):
	lines=[]
	for i in li:
		ii=i
		if (adone):
			ii=ii+1
		lines.append("atom display xmol xmol "+str(ii))
		
	return lines


def tcl_setscale(li,scale):
	lines=[]
	# atom  scale cpk Float Seg Res Atm
	for i in li:
		lines.append("atom -licorice xmol xmol "+str(i+1))
		lines.append("atom cpk "+" xmol xmol "+str(i+1))
		lines.append("atom scale cpk "+str(scale)+" xmol xmol "+str(i+1))
		
	return lines


def tcl_licorice(li,ads=1):
	lines=[]
	# atom  scale cpk Float Seg Res Atm
	for i in li:
		lines.append("atom licorice xmol xmol "+str(i+ads))
	
	return lines
	

def tcl_no_licorice(li):
	lines=[]
	# atom  scale cpk Float Seg Res Atm
	for i in li:
		lines.append("atom -licorice xmol xmol "+str(i+1))
	
	return lines


def tcl_bondbreak(geom,li):
	# break bonds between atom numbers in "li" at geometry "geom"
	c=0
	lines=[]
	for g in geom:
		c=c+1
		for l in li:
			lines.append("edit bond break xmol xmol "+str(c)+" xmol xmol "+str(l+1))
	
	return lines

def tcl_breakallbonds(geom):
	c=0
	lines=[]
	for g in geom:
		c=c+1
		d=0
		for g2 in geom:
			d=d+1
			lines.append("edit bond break xmol xmol "+str(c)+" xmol xmol "+str(d))
	return lines

def tcl_paintspec(geom,specs,color,scale):
	c=0
	li=[]
	for g in geom:
		if (g[0] in specs):
			li.append(c)
		c=c+1
	
	lines=tcl_paintatoms(li,color)
	lines=lines+tcl_setscale(li,scale)
	
	return lines
		

def tcl_createbonds(geom,specs,tol,color="white",scale=1.0):
	lines=[]
	c=0
	for g in geom:
		cc=c+1
		for gg in geom[c+1:]:
			if ((specs==[]) or ((g[0] in specs) and (gg[0] in specs))):
				# print "haloo!"; stop
				if (distance(g,gg)<=tol):
					# print "haloo!";stop
					# create a bond!
					lines.append("edit bond create xmol xmol "+str(c+1)+" xmol xmol "+str(cc+1))
			cc=cc+1
		
		c=c+1	
	
	# print lines
	return lines


def tcl_plotlines(geom,a,b,color="black",scale=0.2,append=True):
  lines=[]
  x1=geom[a][1];y1=geom[a][2];z1=geom[a][3]
  x2=geom[b][1];y2=geom[b][2];z2=geom[b][3]
  st="plot cylinder "+str(x1)+" "+str(y1)+" "+str(z1)+" "+str(x2)+" "+str(y2)+" "+str(z2)+" "+str(scale)+" "+color
  if (append):
    st=st+" append"
  lines.append(st)
  return lines


def tcl_plotarrows(geom,a,b,color="black",scale=0.2,geomscale=1.0,append=True):
  lines=[]
  x1=geom[a][1];y1=geom[a][2];z1=geom[a][3]
  x2=geom[b][1];y2=geom[b][2];z2=geom[b][3]
  
  x2=x1+geomscale*(-x1+x2)
  y2=y1+geomscale*(-y1+y2)
  z2=z1+geomscale*(-z1+z2)
  
  st="plot arrow "+str(x1)+" "+str(y1)+" "+str(z1)+" "+str(x2)+" "+str(y2)+" "+str(z2)+" "+str(scale)+" "+color
  if (append):
    st=st+" append"
  lines.append(st)
  return lines


def tcl_plotarrows2(g,b,color="black",scale=0.2,geomscale=1.0,append=True):
  lines=[]
  x1=g[1];y1=g[2];z1=g[3]
  
  x2=x1+geomscale*(b[0])
  y2=y1+geomscale*(b[1])
  z2=z1+geomscale*(b[2])
  
  st="plot arrow "+str(x1)+" "+str(y1)+" "+str(z1)+" "+str(x2)+" "+str(y2)+" "+str(z2)+" "+str(scale)+" "+color
  if (append):
    st=st+" append"
  lines.append(st)
  return lines



def vizhbond(geom,scale):
  # find hydrogen bonds
  # tol=3.0 # in angstroms
  # .. we should find this automagically..
  
  lines=[]
  app=False
  c=0
  li=pickspec(geom,"O")
  # print li
  # stop
  tol=distance(li[0],li[1])*1.10
  
  for g in geom:
    if (g[0]=="O"):
      hn=0
      
      nei=find_neighbours(g,geom) # 0=the O atom, 1=H, 2=H, 3,4=hbond H
      
      # first find out the direction of covalent H bonds...
      h1=nei[1][1]
      h2=nei[2][1]
      
      h1=geomdiff([g,geom[h1]])
      h2=geomdiff([g,geom[h2]])
      uv=unitvec(add(h1,mul(add(mul(h1,-1.0),h2),0.5))) # symmetry axis
      
      n1=nei[3][1]
      n2=nei[4][1]
      dv1=geomdiff([g,geom[n1]])
      dv2=geomdiff([g,geom[n2]])
      # .. those are supposedly O that donate hydrogen bonds to us
      
      n1v=dot(dv1,uv)
      n2v=dot(dv2,uv)
      # .. project (neighboring hb donor - to us) vector to our symmetry axis
      # .. should be positive (negative?) and above some treshold
      
      sw=2 # use sw=2, it is more sophisticated
      
      if (sw==1):
	if (geom[n1][0]=="H" and n1v<0):
	  hn=hn+1
	if (geom[n2][0]=="H" and n2v<0):
	  hn=hn+1
	
	if (hn==2):
	  lines=lines+tcl_plotlines(geom,c,n1,color="black",scale=scale,append=app)
	  app=True
	  lines=lines+tcl_plotlines(geom,c,n2,color="black",scale=scale,append=app)
	  
      elif (sw==2):
	if ((geom[n1][0]=="H") and (n1v<=0) and (norm(dv1)<=tol)):
	  lines=lines+tcl_plotlines(geom,c,n1,color="black",scale=scale,append=app)
	  app=True
	if ((geom[n2][0]=="H") and (n2v<=0) and (norm(dv2)<=tol)):
	  lines=lines+tcl_plotlines(geom,c,n2,color="black",scale=scale,append=app)
	  app=True
	
    c=c+1 
  return lines


def get_thickness(geom):

    mi=10000
    ma=-10000
    
    for l in geom:
        mi=min(mi,l[3])
        ma=max(ma,l[3])

    t=ma-mi

    # print mi,ma,t

    return t


def symslab(spec,geom,th):
	# takes adsorbant molecules and copies them to
	# the surface below
	adgeom=[]
	for g in geom:
		if (g[0]==spec):
			gg=copy.deepcopy(g)
			gg[3]=gg[3]-th
			adgeom.append(gg)
			
	return adgeom


def symslabli(geom,th,li):
	# takes adsorbant molecules and copies them to
	# the surface below
	adgeom=[]
	c=0
	for g in geom:
		if (c in li):
			gg=copy.deepcopy(g)
			gg[3]=2.0*th-gg[3]
			adgeom.append(gg)
		c=c+1
			
	return adgeom


def dropatoms(geom,geom2,lattice,maxdist):
	# drop all atoms from geom, that are not nearest neighbours of the atoms in geom2
	# nearest neighbours are all the atoms that are within distance "maxdist" from
	# any of the atoms in geom2.  Takes into account the periodicity through the variable lattice
	#
	# lattice=[[5.94928862097, 0, 0], [0, 6.7357072752910767, 11.666587225715471], [0, -6.7357072752910767, 11.666587225715471]]
	# remove all atoms but nearest neighbours
	# geom=coords.crd2geom(coords.read_file("/home/sampsa/coordinates/arktube.xyz"))
	# latcon=2.5147
	# maxdist=coords.distance(geom[96],geom[31])
	
	geomtrans=[]
	geomtrans.append(geom)
	for v in lattice:
		geomtrans.append(translate(geom,v))
		# pass
	for v in lattice:
		w=mul(v,-1.0)
		geomtrans.append(translate(geom,w))
		
		
	# geom,geom+x,..,..,geom-x,..,..
	# print "geomtrans=",geomtrans
	
	atomlist=[]
	
	# geom2=[]
	# c=0
	# for g in geom:
	#	if (g[0]=="Fe"):
	#		geom2.append(g)
	#		atomlist.append(c)
	#	c=c+1
	
	for g in geom2:
		# print "g=",g
		li=find_neighbours(g,geom)
		# print "first=",li[1][0]
		fl=li[1][0] # first nearest neighbour distance
		
		c=0
		for tgeom in geomtrans:
			# print "c=",c
			c=c+1
			
			li=find_neighbours(g,tgeom)
			
			for l in li[0:]:
				d=l[0] # distance of this particular atom...
				frac=d/fl
				# if (l[1]==64):
				#	print l, frac,"            ", d
				if (d<=maxdist):
					# print l, frac,"            ", d
					atomlist.append(l[1])
	# print "atomlist=",atomlist
	# lines=coords.tcl_paintatoms(atomlist,"blue")
	# coords.writecoords("test.tcl",lines)
	
	# remove unwanted atoms..
	c=0
	newgeom=[]
	for g in geom:
		save=False
		for i in atomlist:
			if (i==c):
				save=True
		if (save):
			newgeom.append(g)
		c=c+1
		
	return newgeom


def find_center(geom):
	x=0.0;y=0.0;z=0.0;
	for g in geom:
		x=x+g[1]
		y=y+g[2]
		z=z+g[3]
		
	n=float(len(geom))
	x=x/n
	y=y/n
	z=z/n
	
	res=[x,y,z]
	
	return res
	
def find_masscenter(geom):
	x=0.0;y=0.0;z=0.0;
	mass=1.0
	masu=0.0
	for g in geom:
		if (g[0]=="H"):
		  mass=2.0
		if (g[0]=="O"):
		  mass=16.0
		if (g[0]=="Cl"):
		  mass=35.45
		
		masu=masu+mass
		x=x+g[1]*mass
		y=y+g[2]*mass
		z=z+g[3]*mass
		mass=1.0
		
	n=float(len(geom))
	x=(x/masu)
	y=(y/masu)
	z=(z/masu)
	
	res=[x,y,z]
	
	return res


def centrify(geom):
	cent=find_center(geom)
	# print "cent=",cent
	# stop
	geom=translate(geom,mul(cent,-1.0))
	
	
def masscentrify(geom):
	cent=find_masscenter(geom)
	# print "cent=",cent
	# stop
	geom=translate(geom,mul(cent,-1.0))
	return geom
	

def centrifymany(geoms):
	cent=find_center(geoms[0].xyz)
	for g in geoms:
		# print g
		translate(g.xyz,cent)

def setorigo(geom,atnum):
	origo=geom[atnum][1:4]
	newgeom=translate(geom,mul(origo,-1.0))
	return newgeom


def flipx(geom,atnum):
  newgeom=setorigo(geom,atnum)
  c=0
  for g in newgeom:
    newgeom[c][1]=-g[1]
    c=c+1
  return newgeom

def setzaxis(geom,atnums):
	z=unitvec(geomdiff([geom[atnums[0]],geom[atnums[1]]]))
	y=[0,1,0]
	x=crossprod(z,y)
	crd1=[[1,0,0],[0,1,0],[0,0,1]] # how things looked in the old coord system..
	crd2=[x,y,z] # how they should look in the new one..
	geomnew=rotate(transf(crd2,crd1),geom)

	return geomnew
	
	

def find_radius(geom):
	r=find_center(geom)
	cent=['H',r[0],r[1],r[2]]
	
	neigh=find_neighbours(cent,geom)
	
	ind=neigh[-1][1] # number of the last atom in the list
	su=neigh[-1][0]

	res=dict()
	res["radius"]=su
	res["center"]=cent
	
	return res
	

def shuffle_input(fdf_lines,subs):
    lnew=''
    newlines=[]
    for l in fdf_lines:
        # print l
        for s in subs:
            ss=copy.deepcopy(s)
            tag='@'+ss[0] # tag
            sub=ss[1] # substitution

            # print ">",tag,sub
            # lnew=l.replace(tag,str(sub))
            lnew=string.replace(l,tag,str(sub))
            # print ">>",lnew
        newlines.append(lnew)

    return newlines


def H_atoms(geom):
    
    li=[]
    c=0
    for g in geom:
        if (g[0]=="H"):
            li.append(c)
        c=c+1

    return li


def list2txt(front,li,end):
    # list to boundary conditions

    lis=[]
    for l in li:
        lis.append(front+str(l+1)+end)

    return lis


def to_layers(geom,tol):
    c=0
    li=[]
    for g in geom:
        c=c+1
        s=[]
        s.append(g[3])
        s.append(c)
        li.append(s)

    # print "atoms=",c

    li.sort()

    laynum=0
    lch=copy.deepcopy(li[0][0])
    for l in li:
        if (abs(l[0]-lch)<tol):
            l[0]=laynum
        else:
            lch=copy.deepcopy(l[0])
            laynum=laynum+1
            l[0]=laynum

    li2=[]
    for l in li:
        el0=copy.deepcopy(l[0])
        el1=copy.deepcopy(l[1])

        li2.append([el1,el0])

    return li2
        

def to_layers2(geom,tol,spec=[]):
    c=0
    li=[]
    li3=[]
    for g in geom:
        # c=c+1 # numbering starting from 1..
	if ((spec==[]) or (g[0] in spec)):
		s=[]
		s.append(g[3])
		s.append(c)
		li.append(s)
	c=c+1 # numbering starting from 0..
	
    li.sort()

    # for l in li:
    #    print ">",l

    laynum=0
    lch=copy.deepcopy(li[0][0])
    for l in li:
        if ((l[0]-lch)<tol):
            l[0]=laynum
        else:
            lch=copy.deepcopy(l[0])
            laynum=laynum+1
            l[0]=laynum
            li3.append([])

    li3.append([])
    li2=[]
    for l in li:
        el0=copy.deepcopy(l[0])
        el1=copy.deepcopy(l[1])

        li2.append([el1,el0])

    # print "li3:",li3

    for l in li2:
        laynum=l[1]
        atnum=l[0]
        # print "lay,at",laynum,atnum
        li3[laynum].append(atnum)

    return li3


def to_layers3(geom,tol,spec=[]):
    c=0
    li=[]
    li3=[]
    for g in geom:
        # c=c+1 # numbering starting from 1..
	if ((spec==[]) or (g[0] in spec)):
		s=[]
		s.append(g[3])
		s.append(c)
		li.append(s)
	c=c+1 # numbering starting from 0..
	
    # li: [[z,atnum],..]
    li.sort()

    # for l in li:
    #    print ">",l

    laynum=0
    lch=copy.deepcopy(li[0][0]) # take the first z value..
    for l in li: # [[z,atnum],..]
        if (abs(l[0]-lch)<tol):
            l[0]=laynum # [[z,atnum],..] => [[laynum,atnum],..]
        else:
            lch=copy.deepcopy(l[0])
            laynum=laynum+1
            l[0]=laynum
            li3.append([])

    li3.append([])
    li2=[]
    for l in li:
        el0=copy.deepcopy(l[0])
        el1=copy.deepcopy(l[1])

        li2.append([el1,el0])

    # print "li3:",li3

    for l in li2:
        laynum=l[1]
        atnum=l[0]
        # print "lay,at",laynum,atnum
        li3[laynum].append(atnum)

    return li3



def to_layersx(geom,tol,ind,spec=[]):
    # along coordinates "ind"
    c=0
    li=[]
    li3=[]
    for g in geom:
        # c=c+1 # numbering starting from 1..
	if ((spec==[]) or (g[0] in spec)):
		s=[]
		s.append(g[ind])
		s.append(c)
		li.append(s)
	c=c+1 # numbering starting from 0..
	
    li.sort()

    # for l in li:
    #    print ">",l

    laynum=0
    lch=copy.deepcopy(li[0][0])
    for l in li:
        if ((l[0]-lch)<tol):
            l[0]=laynum
        else:
            lch=copy.deepcopy(l[0])
            laynum=laynum+1
            l[0]=laynum
            li3.append([])

    li3.append([])
    li2=[]
    for l in li:
        el0=copy.deepcopy(l[0])
        el1=copy.deepcopy(l[1])

        li2.append([el1,el0])

    # print "li3:",li3

    for l in li2:
        laynum=l[1]
        atnum=l[0]
        # print "lay,at",laynum,atnum
        li3[laynum].append(atnum)

    return li3


def line2vecs(line,scale):
    li=string.splitfields(line,"/")
    vecs=[]
    for l in li:
        # print "l",l
        v=string.splitfields(l,",")
        g=[]
        for vc in v:
            g.append(float(vc)*scale)
        vecs.append(g)

    return vecs

# there was some older routine like this...
#def dimerize(geom,ats,dir,frac):
    ## frac=wanted fractional distance in direction dir
    #geomnew=copy.deepcopy(geom)

    #at1=geomnew[ats[0]-1]
    #at2=geomnew[ats[1]-1]

    ## print "at1",at1
    ## print "at2",at2

    #d1=at1[dir]
    #d2=at2[dir]

    ## print "d1,d2",d1,d2

    #d=math.sqrt((d1-d2)**2)
    #delta=(d*(1-frac))/2

    #d1=d1+delta
    #d2=d2-delta

    ## print ">>d1,d2",d1,d2

    #at1[dir]=d1
    #at2[dir]=d2

    ## print ">>",at1
    ## print ">>",at2

    #return geomnew


def dimerize(geom,ats,frac):
    	# frac=wanted fractional distance in direction dir
	geomnew=copy.deepcopy(geom)

	g=copy.deepcopy(geom[ats[0]])
	h=copy.deepcopy(geom[ats[1]])
	
	# print "h",h,h[1:4] # why here 4?
	# print "g",g,g[1:4]

	# create unit vector from g to h
	vec=[]
	c=0
	for i in range(1,len(h)):
		# print i,h[i],g[i]
		if (c<3):
			vec.append(h[i]-g[i])
		c=c+1
		# print i,h[i]
	vec=unitvec(vec)
	# print "unitvec=",vec
	d=distance(h,g)
	# print "d=",d
	# move g to direction vec and h to direction -vec
	dvec=mul(vec,(1.0-frac)*d*0.5)
	
	# print ">",g[1:4],dvec
	
	g[1:4]=add(g[1:4],dvec)
	h[1:4]=add(h[1:4],mul(dvec,-1))
	
	geomnew[ats[0]]=g
	geomnew[ats[1]]=h

    	return geomnew





def deltadist(geom,ats,delta,sym=True,uvec=[]):
    	# frac=wanted fractional distance in direction dir
	geomnew=copy.deepcopy(geom)

	g=copy.deepcopy(geom[ats[0]])
	h=copy.deepcopy(geom[ats[1]])
	
	# print "h",h,h[1:4] # why here 4?
	# print "g",g,g[1:4]

	# create unit vector from g to h
	vec=[]
	for i in range(1,min(len(h),4)):
		# print i,h[i],g[i]
		vec.append(h[i]-g[i])
		# print i,h[i]
	vec=unitvec(vec)
	if (uvec!=[]):
		vec=uvec
	# print "unitvec=",vec
	d=distance(h,g)
	# print "d=",d
	# move g to direction vec and h to direction -vec
	dvec=mul(vec,0.5*delta)
	
	# print ">",g[1:4],dvec
	
	if (sym):
		g[1:4]=add(g[1:4],dvec)
		h[1:4]=add(h[1:4],mul(dvec,-1))
	else:
		g[1:4]=add(g[1:4],dvec)
		g[1:4]=add(g[1:4],dvec)
	
	geomnew[ats[0]]=g
	geomnew[ats[1]]=h

    	return geomnew



def ohosym(geom,ats,bl):
  # coords.ohosym(geom,[37,15,13],1.0)
  o1=geom[ats[0]-1]
  o2=geom[ats[2]-1]
  h=geom[ats[1]-1]
  
  #print "o1",o1
  #print "o2",o2
  #print "h",h
  #print distance(o1,h)
  #print distance(h,o2)
  
  o1v=numpy.array(o1[1:4])
  o2v=numpy.array(o2[1:4])
  hv=numpy.array(h[1:4])
  
  dv=(-o1v+o2v)
  # .. length of that should be bl*2
  # print "bl=",bl
  tl=2*bl # "target" length
  l=lg.norm(dv) # length
  u=dv/l
  
  d=(tl-l)
  
  o1v=o1v-u*d/2.0
  o2v=o2v+u*d/2.0
  hv=o1v+bl*u
 
  o1[1:4]=list(o1v)
  o2[1:4]=list(o2v)
  h[1:4]=list(hv)
 
  #print
  #print "new values"
  #print "o1",o1
  #print "o2",o2
  #print "h",h
  #print distance(o1,h)
  #print distance(h,o2)
  

def ohosym2(geom,ats,ool,hl):
  # coords.ohosym(geom,[37,15,13],1.0)
  o1=geom[ats[0]-1]
  o2=geom[ats[2]-1]
  h=geom[ats[1]-1]
  
  #print "o1",o1
  #print "o2",o2
  #print "h",h
  #print distance(o1,h)
  #print distance(h,o2)
  
  o1v=numpy.array(o1[1:4])
  o2v=numpy.array(o2[1:4])
  hv=numpy.array(h[1:4])
  
  dv=(-o1v+o2v)
  # .. length of that should be ool
  # print "bl=",bl
  tl=ool # "target" length
  l=lg.norm(dv) # length
  u=dv/l
  
  d=(tl-l)
  
  o1v=o1v-u*d/2.0
  o2v=o2v+u*d/2.0
  hv=o1v+0.5*(-o1v+o2v)+hl*u
 
  o1[1:4]=list(o1v)
  o2[1:4]=list(o2v)
  h[1:4]=list(hv)
 
  #print
  #print "new values"
  #print "o1",o1
  #print "o2",o2
  #print "h",h
  #print distance(o1,h)
  #print distance(h,o2)
  #print distance(o1,o2)


def expand_geom(geom,vecs,n):
    # print vecs
    # print vecs[1]
    # n=4
    points=[]
    # geom=[['C',0,0,0],['O',0.1,0.1,0.1]]
    for i in range(-n,n+1):
        #    print i
        for j in range(-n,n+1):
            for k in range(-n,n+1):
                x=vecs[0][0]*i + vecs[1][0]*j + vecs[2][0]*k
                y=vecs[0][1]*i + vecs[1][1]*j + vecs[2][1]*k
                z=vecs[0][2]*i + vecs[1][2]*j + vecs[2][2]*k
                for ng in geom:
                    g=[]
                    g.append(ng[0])
                    g.append(x+ng[1])
                    g.append(y+ng[2])
                    g.append(z+ng[3])
		    # print ">>",ng
		    if (len(ng)>4):
			    g.append(ng[4])
                    points.append(g)

    return points


def expand_geom_first(geom,vecs,n):
  # expand in such a way that atoms in the primal unit cell are the first ones in the list
  points=copy.deepcopy(geom)
  for i in range(-n,n+1):
        #    print i
        for j in range(-n,n+1):
            for k in range(-n,n+1):
		if ((i==0) and (j==0) and (k==0)):
		  pass
		else:
		  x=vecs[0][0]*i + vecs[1][0]*j + vecs[2][0]*k
		  y=vecs[0][1]*i + vecs[1][1]*j + vecs[2][1]*k
		  z=vecs[0][2]*i + vecs[1][2]*j + vecs[2][2]*k
		  for ng in geom:
		      g=[]
		      g.append(ng[0])
		      g.append(x+ng[1])
		      g.append(y+ng[2])
		      g.append(z+ng[3])
		      # print ">>",ng
		      if (len(ng)>4):
			      g.append(ng[4])
		      points.append(g)

  return points
  

def expand_geom_first2(geom,vecs,latcopy):
  # expand in such a way that atoms in the primal unit cell are the first ones in the list
  points=copy.deepcopy(geom)
  n1=latcopy[0]
  n2=latcopy[1]
  n3=latcopy[2]
  for i in range(-n1,n1+1):
        #    print i
        for j in range(-n2,n2+1):
            for k in range(-n3,n3+1):
		if ((i==0) and (j==0) and (k==0)):
		  pass
		else:
		  x=vecs[0][0]*i + vecs[1][0]*j + vecs[2][0]*k
		  y=vecs[0][1]*i + vecs[1][1]*j + vecs[2][1]*k
		  z=vecs[0][2]*i + vecs[1][2]*j + vecs[2][2]*k
		  for ng in geom:
		      g=[]
		      g.append(ng[0])
		      g.append(x+ng[1])
		      g.append(y+ng[2])
		      g.append(z+ng[3])
		      # print ">>",ng
		      if (len(ng)>4):
			      g.append(ng[4])
		      points.append(g)

  return points



def expand_add(geom,lis):
  newgeom=copy.deepcopy(geom)
  # [[vec,fac1,fac2..],[],..]
  for l in lis:
    vec=l[0]
    # print "geom=",geom
    for f in l[1:]:
      # print "f=",f
      newgeom=newgeom+translate(geom,mul(vec,f))
   
  return newgeom
  


def expand_geom2(geom,vecs,n):
    # print vecs
    # print vecs[1]
    # n=4
    points=[]
    # geom=[['C',0,0,0],['O',0.1,0.1,0.1]]
    for i in range(0,n):
        #    print i
        for j in range(0,n):
            for k in range(0,n):
                x=vecs[0][0]*i + vecs[1][0]*j + vecs[2][0]*k
                y=vecs[0][1]*i + vecs[1][1]*j + vecs[2][1]*k
                z=vecs[0][2]*i + vecs[1][2]*j + vecs[2][2]*k
                for ng in geom:
                    g=[]
                    g.append(ng[0])
                    g.append(x+ng[1])
                    g.append(y+ng[2])
                    g.append(z+ng[3])
		    # print ">>",ng
		    if (len(ng)>4):
			    g.append(ng[4])
                    points.append(g)

    return points

# for p in points:
#    print p
# coords.writexmol("test.xyz",coords.geom2crd(coords.rel2abs(points,4.1)))
    

def expand_geom3(smallgeom,lattice,cells):
	# cells=[[-1,1],[-1,1],[-1,1]]
	
	debug=False
	if (debug):
	  print "smallgeom"
	  print smallgeom
	  print "lattice"
	  print lattice
	  print "cells"
	  print cells
	  stop
	
	count=0
	
	geom=[]
	cc=0
	for i in range(cells[0][0],cells[0][1]):
		geom=geom+translate(smallgeom,mul(lattice[0],i))
		cc=cc+1
		count=count+1
	# print "isum>",cc
	if (len(lattice)<2):
		return geom
	
	geom2=[]
	cc=0
	if (len(lattice)>1):
		for i in range(cells[1][0],cells[1][1]):
			geom2=geom2+translate(geom,mul(lattice[1],i))
			cc=cc+1
			count=count+1
	# print "isum>",cc
	if (len(lattice)<3):
		return geom2
	
	geom3=[]
	if (len(lattice)>2):
		for i in range(cells[2][0],cells[2][1]):
			print i
			geom3=geom3+translate(geom2,mul(lattice[2],i))
			count=count+1
		return geom3


def find_vecs(points):
    xval=1000
    yval=1000
    zval=1000
    for p in points:
        sp=p[0]
        x=p[1]
        y=p[2]
        z=p[3]
        if not ((y==0) and (x==0) and (z==0)):
            if ((y==0) and (z==0)):
                # print p
                xval=min(abs(x),xval)           
            if ((x==0) and (z==0)):
                # print p
                yval=min(abs(y),yval)
            if ((x==0) and (y==0)):
                # print p
                zval=min(abs(z),zval)

    v1=[xval,0,0]
    v2=[0,yval,0]
    v3=[0,0,zval]
    return([v1,v2,v3])
        
# print "x,y,z",xval,yval,zval

#def cut_block(geom,lims):
    #tol=0.0
    #newgeom=[]
    ## print "geom:",geom
    #for g in geom:
        ## print "g>",g
        #x=g[1]
        #y=g[2]
        #z=g[3]

        #dx=lims[0]-x
        #dy=lims[1]-y
        #dz=lims[2]-z

        #if ((x>0) and (y>0) and (z>0)):
            #if ((dx>=tol) and (dy>=tol) and (dz>=tol)):
                #print g
                #newgeom.append(g)
            
    #return(newgeom)


def cut_block(geom,lims,spec=[]):
	newgeom=[]
	for g in geom:
		x=g[1];y=g[2];z=g[3];
		ap=False
		if ( (lims[0][0]<=x<=lims[0][1]) and (lims[1][0]<=y<=lims[1][1]) and (lims[2][0]<=z<=lims[2][1]) ):
			ap=True
			newgeom.append(g)
		if (spec!=[] and (g[0] not in spec) and ap==False):
			newgeom.append(g)
	return newgeom
	
	
def cut_block_lis(geom,lims,spec=[]):
	li=[]
	c=0
	for g in geom:
		x=g[1];y=g[2];z=g[3];
		ap=False
		if ( (lims[0][0]<=x<=lims[0][1]) and (lims[1][0]<=y<=lims[1][1]) and (lims[2][0]<=z<=lims[2][1]) ):
			ap=True
			li.append(c)
		if (spec!=[] and (g[0] not in spec) and ap==False):
			li.append(c)
		c=c+1
	return li


def find_block(geom,lims):
	# gives a list of atoms within some limits..
	li=[]
	c=0
	for g in geom:
		x=g[1];y=g[2];z=g[3];
		if ( (lims[0][0]<=x<=lims[0][1]) and (lims[1][0]<=y<=lims[1][1]) and (lims[2][0]<=z<=lims[2][1]) ):
			li.append(c)
		c=c+1
			
	return li


#def find_block_water(geom,lims):
  #li=[]
  #c=0
  #for g in geom:
    #spec=g[0];x=g[1];y=g[2];z=g[3];
    #if ( (lims[0][0]<=x<=lims[0][1]) and (lims[1][0]<=y<=lims[1][1]) and (lims[2][0]<=z<=lims[2][1]) ):
      #if (spec=="O"):
      
	    #li.append(c)
    #c=c+1
  #return li
  
 
def cut_radial(geom,rad,cent=[0,0,0]):
	# cuts radial block..
	dummy=['H']
	dummy=dummy+cent
	li=[]
	for g in geom:
		d=distance(dummy,g)
		# print "d=",d
		if (d<=rad):
			li.append(g)
	
	return li


def cut_radial2(geom,rad,cent=["H",0,0,0]):
	# cuts radial block..
	# the center atom is not included
	dummy=cent
	li=[]
	for g in geom:
		d=distance(dummy,g)
		# print "d=",d
		if ((d<=rad) and (d>=0.00001)):
			li.append(g)
	
	return li


def cut_2drad(geom,rad,cent=[0,0]):
	newgeom=[]
	for g in geom:
		dr=math.sqrt((g[1]-cent[0])**2+(g[2]-cent[1])**2)
		if (dr<=rad):
			newgeom.append(g)
	return newgeom
	
def findcol(geom,ind,tol):
	c=0
	lis=[]
	x=geom[ind][1]
	y=geom[ind][2]
	for g in geom:	
		if (math.sqrt((g[1]-x)**2+(g[2]-y)**2)<tol):
			lis.append(c)
		c=c+1
		
	return lis
		
def findcolperiod(geom,lattice,ind,tol):
	# lattice: 2 vectors
	x=geom[ind][1]
	y=geom[ind][2]
	tmpgeom=copy.deepcopy(geom)
	# add enumeration to tmpgeom
	#c=0
	#for g in tmpgeom:
	#	g.append(c)
	#	c=c+1
	# ********* we assume a dictionary element g[4]["num"]
	
	inds=[[1,0],[-1,0],[0,1],[0,-1]]
	for ind in inds:
		i1=ind[0]
		i2=ind[1]
		l1=mul(lattice[0],i1)
		l2=mul(lattice[1],i2)
		
		tmpgeom=tmpgeom+add(translate(geom,l1),translate(geom,l2))
	
	#for g in tmpgeom:
	#	print ">",g
	#stop
	
	lis=[]
	for g in tmpgeom:
		if (math.sqrt((g[1]-x)**2+(g[2]-y)**2)<tol):
			num=g[4]["num"]
			if (num not in lis):
				lis.append(num)
	return lis
	

def find_species(geom,spe):
	li=[]
	c=0
	for g in geom:
		if (g[0]==spe):
			li.append(c)
		c=c+1
			
	return li


def find_species2(geom,spe):
	# now spe is a list..
	li=[]
	c=0
	for g in geom:
		if (g[0] in spe):
			li.append(c)
		c=c+1
			
	return li
		
		
def set_bondlength(geom,at,l):
	n=at[0]
	m=at[1]
	# sets the bond length between atoms n and m
	# to l by moving n
	vec=atoms2vec(geom[m],geom[n]) # m->n
	# print "vec:",vec
	vu=unitvec(vec)
	# print "vu=",vu
	# print "geom=",geom[n]
	# print "geom[m]=",geom[m]
	geom[n][1]=l*vu[0]+geom[m][1]
	geom[n][2]=l*vu[1]+geom[m][2]
	geom[n][3]=l*vu[2]+geom[m][3]
	# print "geom now=",geom[n]


def C_layer(copies):
	# *** these are the "wrong" lattice vectors, etc., taken from
	# v=[]
	# FCC (111) plane, where lattice constant = a = side of the FCC unit cell
	# v.append([0,1.0/math.sqrt(2.0)]) 
	# v.append([math.sqrt(3.0/8.0),1/math.sqrt(8.0)])
	# v.append([math.sqrt(3.0/8.0),-1/math.sqrt(8.0)])
	# print "v="+str(v)
	# copies=[9,4]
	
	# basis=[['C',0,0],
	# ['B',0.1,0.1]]
	# basis=[
	# ['C',0,0],
	# ['C',0.5/math.sqrt(2.0*3.0),0.5/math.sqrt(2.0)]
	# ]
	# *******************************************************
	v=copy.copy(trired0)
	basis=copy.copy(graphene_basis)
	
	geom=[]
	for i in range(-copies[0],copies[0]+1):
		# print str(i)
		for j in range(-copies[1],copies[1]+1):
			# print "      "+str(j)
			x=i*v[0][0]+j*v[1][0]
			y=i*v[0][1]+j*v[1][1]
			for k in basis:
				geom.append([k[0],x+k[1],y+k[2]])
	return geom
			
			
def some_layer(red,basis,copies):
	v=copy.copy(red)
	basis=copy.copy(basis)
	
	geom=[]
	for i in range(-copies[0],copies[0]+1):
		# print str(i)
		for j in range(-copies[1],copies[1]+1):
			# print "      "+str(j)
			x=i*v[0][0]+j*v[1][0]
			y=i*v[0][1]+j*v[1][1]
			for k in basis:
				geom.append([k[0],x+k[1],y+k[2]])
	return geom
	
	
def some_layer2(red,basis,copies):
	v=copy.copy(red)
	basis=copy.copy(basis)
	
	geom=[]
	for i in range(0,copies[0]):
		# print str(i)
		for j in range(0,copies[1]):
			# print "      "+str(j)
			x=i*v[0][0]+j*v[1][0]
			y=i*v[0][1]+j*v[1][1]
			for k in basis:
				geom.append([k[0],x+k[1],y+k[2]])
	return geom
	

def some_layer3(red,basis,lis):
	# lis=[[0,0],[0,1],[1,1],..]
	v=copy.copy(red)
	basis=copy.copy(basis)
	
	# print "v=",v
	# stop
	
	geom=[]
	for l in lis:
		print l
		i=l[0]
		j=l[1]
		# print "      "+str(j)
		x=i*v[0][0]+j*v[1][0]
		y=i*v[0][1]+j*v[1][1]
		
		#print "x=",x
		#print "y=",y
		#print
		
		for k in basis:
			geom.append([k[0],x+k[1],y+k[2]])
			
	#stop
	return geom


def g2vec(g):
  return numpy.array([g[1],g[2],g[3]])
  
def v2num(g):
  return numpy.array([g[0],g[1],g[2]])

def vec2g(spec,vec):
  return [spec,vec[0],vec[1],vec[2]]

def num2v(vec):
  return [vec[0],vec[1],vec[2]]


def atoms2vec(at1,at2):
	v=[]
	
	for i in range(1,4):
		v.append(at2[i]-at1[i])
		
	return v

def dot(vec,vec2):
	c=0
	s=0.0
	for v in vec:
		s=s+v*vec2[c]
		c=c+1
	return s
		
		
def cross(vec,vec2):
	# print vec[0]*vec2[1]
	# print -vec2[0]*vec[1]
	return vec[0]*vec2[1]-vec2[0]*vec[1]

def crossprod(v,w):
	z=[
		v[1]*w[2]-v[2]*w[1],
		v[2]*w[0]-v[0]*w[2],
		v[0]*w[1]-v[1]*w[0]
	]
	
	return z
	
	
def geomnormal(geom):
	# finds a normal to a plane,
	# defined by three atoms in geom
	v=add(mul(geom[0][1:],-1.0),geom[1][1:])
	w=add(mul(geom[0][1:],-1.0),geom[2][1:])

	return crossprod(unitvec(v),unitvec(w))


def geomdiff(geom):
	if (len(geom[0])>4):
		v=add(mul(geom[0][1:4],-1.0),geom[1][1:4])
		# print v
		# stop
	else:
		v=add(mul(geom[0][1:],-1.0),geom[1][1:])

	return v


def find_normal(v):
	# finds a normal for vector v automagically
	#
	# (1) see if there are components=0
	i=-1
	if (v.count(0.0)>0):
		i=v.index(0.0)
	
	if (len(v)==2):
		w=[0,0]
		if (i>=0):
			w[i]=1.0
		else:
			w[0]=-v[1]
			w[1]=v[0]
		
	if (len(v)==3):
		w=[0,0,0]
		if (i>=0):
			w[i]=1.0
		else:
			w[0]=-v[1]
			w[1]=v[0]
			w[2]=0.0
	
	return w
	
		
def norm(vec):
	return math.sqrt(dot(vec,vec))
	
	
def unitvec(vec):
	# print "vec",vec
	if (norm(vec)>0.0):
		return mul(vec,(1/norm(vec)))
	else:
		return vec

def subs(vec1,vec2):
	c=0
	nev=[]
	for v in vec1:
		nev.append(v-vec2[c])
		c=c+1
	return nev
	
def add(vec1,vec2):
	c=0
	nev=[]
	for v in vec1:
		nev.append(v+vec2[c])
		c=c+1
	return nev

def find_int(frac):
	test=1000.0
	tol=0.000000000001
	j=0
	while (test>tol):
		j=j+1
		# i=(i/j)*j
		i=frac*j
		test=math.sqrt((i-int(i))**2)
		# print "j:",j,"num:",num,"test:",test,num2/j,j/num2
	return [i,j]
	
		
def transf(oldvecs,newvecs):
	mat=[]
	for v in newvecs:
		row=[]
		for v2 in oldvecs:
			row.append(dot(v,v2))
		mat.append(row)
	return(mat)
		

def fixcell(refgeom, geom, lattice2):
  mol=[refgeom,geom]
  startgeom=refgeom

  morevecs=False
  if (morevecs):
	  # create also linear combinations of lattice vectors..
	  alat=[]
	  for l in lattice2:
		  for l2 in lattice2:
			  #for l3 in lattice2:
			  #  alat.append(add(add(l,l2),l3))
			  #  alat.append(add(add(mul(l,-1),l2),l3))
			  #  alat.append(add(add(l,mul(l2,-1)),l3))
			  #
			  #  alat.append(add(add(l,l2),mul(l3,-1)))
			  
			  alat.append(add(l,l2))
			  alat.append(add(mul(l,-1),l2))
			  # alat.append(add(mul(l2,-1),l)) # this is already included!
	  alat=alat+lattice2
	  
	  # remove identical vectors..
	  lac=0
	  alat2=[]
	  for l in alat:
		  if (l in alat[lac+1:]):
			  pass
		  else:
			  if (l!=[0,0,0]):
				  alat2.append(l)
		  lac=lac+1	
	  
	  for a in alat2:
		  print "alat2..",a
	  # stop	
	  
	  #for g in info.startgeom:
	  #	print g
	  # stop
  else:
	  alat2=copy.deepcopy(lattice2)
	  #for l in lattice2: # the negative shifts are already included!
	  #	alat2.append(mul(l,-1)) # already included!

  #print alat2
  #stop

  specvecs=False
  if (specvecs): # check some specific vectors..
	  alat2=[]
	  l=lattice2[0]
	  l2=lattice2[1]
	  # alat2.append(add(mul(l,-1.0),l2))
	  alat2.append(add(mul(l,1.0),l2))

  # trace atom movements.. if they move more than a unit cell, move them back
  c=1
  for now in mol[1:]: # now, prev = one geometry
	  prev=mol[0]
	  d=0
	  news=[]
	  for nowcoord in now:
		  prevcoord=prev[d]
		  dv=geomdiff([nowcoord,prevcoord])
		  # dv=geomdiff([prevcoord,nowcoord])
		  #print "> atom", d+1
		  #print "> prev",prevcoord
		  #print "> now", nowcoord
		  #print "> dif",dv
		  #print ">"
		  
		  # toll=0.98 # molecules, fcc
		  toll=0.8 # surface stuff
		  # toll=0.7
		  
		  for l in alat2:
			  dott=dot(dv,unitvec(l))
			  dr=dot(unitvec(dv),unitvec(l)) # this is = 1.0 if movement is in direction of this vector..
			  #if (math.sqrt(dot(dv,dv))>10):
				  #print "--------"
				  #print dot,math.sqrt(dot(dv,dv))
				  #print "nowcoord",nowcoord
				  #print "prevcoord",prevcoord
				  #print l
				  #print "abs(dot)",abs(dot)
				  #print "dot(l,l)",dot(l,l)
			  #if ((d+1)==9):
				  #print "---------"
				  #print "we are at step",c
				  #print "atom number",d+1
				  #print "dv",dv
				  #print "dot",dot
				  #print "l",l
				  #print "dv.l",dot(dv,unitvec(l))
				  #print "nowcoord",nowcoord
				  #print "prevcoord",prevcoord
				  #print "measure",(abs(dot)/math.sqrt(dot(l,l)))
			  if (  ((abs(dott)/math.sqrt(dot(l,l)))>0.8) and (abs(dr)>=toll)):
				  # stop
				  #print "---------"
				  #print "we are at step",c
				  #print "atom number",d+1
				  #print "dv",dv
				  #print "dot",dot
				  #print "l",l
				  #print "nowcoord",nowcoord
				  #print "prevcoord",prevcoord
				  # we have moved amount of lattice vector l.. must correct
				  
				  ll=mul(l,(dott/abs(dott))) # correct by +l or -l ?
				  now[d]=[nowcoord[0],nowcoord[1]+ll[0],nowcoord[2]+ll[1],nowcoord[3]+ll[2],startgeom[d][4]]
				  
				  # ll=mul(l,dot)
				  # now[d]=[nowcoord[0],nowcoord[1]-ll[0],nowcoord[2]-ll[1],nowcoord[3]-ll[2],info.startgeom[d][4]]
				  
				  # print "now",now[d]
				  # print "now[d][4]=",now[d][4]
				  
				  #if (len(nowcoord)>4):
				  #	# now[d].append(prevcoord[4]) # WAS LEFT HERE!
				  # now[d].append(info.startgeom[4])
				  # print "new",now[d]
				  nowcoord=copy.deepcopy(now[d]) # important!
				  
		  if (len(now[d])<=4):
			  now[d].append(startgeom[d][4])
		  else:
			  now[d][4]=startgeom[d][4]
			  # print "2) now[d][4]=",now[d][4]
				  
				  
		  # new.append(info.startgeom[d][4])
		  # print "new=",new
		  # stop
		  # news.append(new)
		  d=d+1
	  # stop
	  mol[c]=copy.deepcopy(now)
	  c=c+1

  return mol[-1]


def scroll(geom,atoms):
	# scrolls a sheet into a tube..
	n=atoms[0]
	m=atoms[1]
	# .. take atom n to atom m
	 
	vec=atoms2vec(geom[m],geom[n])
	vu=unitvec(vec)
	# print "vec="+str(vec)
	d=norm(vec)
	r=d/(2.0*math.pi)
	
	origo=geom[m]
	
	c=0
	gg=[]
	for g in geom:
		c=c+1
		v=atoms2vec(origo,g)
		# print "dot=",dot(v,vu)
		dist=dot(v,vu)
		# print c,dist 
		angle=dist/d
		z=(1-math.cos(2*math.pi*angle))*r
		# print c,dist,angle,2.0*math.pi*angle,math.cos(2.0*math.pi*angle)
		# z=g[3]+angle
		k=math.sin(2*math.pi*angle)*r
		dv=mul(vu,(k-dot(g[1:3],vu)))
		# print "z="+str(z)
		gg.append([g[0],g[1]+dv[0],g[2]+dv[1],z])  
		
	return gg
	
	
def bend(geom,atoms,rad):
	# scrolls a sheet into a tube..
	# the idea:
	# normally folded nanotube, has its circumference
	# = dist = the distance between the equivalent
	# atoms connected by the chirality vector..
	# Now we just modify this by setting
	# dist:=dist*rad
	# now rad=2 gives a half nanotube, etc..
	
	n=atoms[0]
	m=atoms[1]
	# .. take atom n to atom m
	 
	vec=atoms2vec(geom[m],geom[n])
	vu=unitvec(vec)
	# print "vec="+str(vec)
	d=norm(vec)*rad
	r=d/(2.0*math.pi)
	
	origo=geom[m]
	
	c=0
	gg=[]
	for g in geom:
		c=c+1
		v=atoms2vec(origo,g)
		# print "dot=",dot(v,vu)
		dist=dot(v,vu)
		# print c,dist 
		angle=dist/d
		z=(1-math.cos(2*math.pi*angle))*r
		# print c,dist,angle,2.0*math.pi*angle,math.cos(2.0*math.pi*angle)
		# z=g[3]+angle
		k=math.sin(2*math.pi*angle)*r
		dv=mul(vu,(k-dot(g[1:3],vu)))
		# print "z="+str(z)
		gg.append([g[0],g[1]+dv[0],g[2]+dv[1],z])  
		
	return gg


def find_point(nv,v):
	# *************************************************
	# **** finds a Bravais-lattice (defined by v) point
	# **** in direction nv
	# *************************************************
	d0=cross(nv,v[0])
	d1=cross(nv,v[1])
	fac=1
	if (d0/abs(d0)==d1/abs(d1)):
		fac=-1
	#i=20
	#j=1
	#vec=add(mul(v[0],i),mul(v[1],j))
	# print "nv=",nv
	# print "v=",v
	# print "vec=",vec
	#d=cross(nv,vec)
	
	#print "d=",d
	
	iofs=0
	jofs=0
	isave=0
	jsave=0
	
	dcheck=1
	dmin=1000.0
	i=0
	j=0
	while (dcheck==1):
		j=0
		i=i+1
		# print "i=",i
		
		# i=1, j=0  x  nv  => dn
		vec=add(mul(v[0],i),mul(v[1],j))
		dn=cross(nv,vec)
		# print "dn=",dn
		
		# i=1, j=0...n  x  nv => d
		vec=add(mul(v[0],i),mul(v[1],j))
		d=cross(nv,vec)
		# print "d=",d
		
		sdn=dn/abs(dn)
		sd=d/abs(d)
		
		j=0
		while (sdn==sd):
			j=j+fac
			# print "       j=",j
			vec=add(mul(v[0],i),mul(v[1],j))
			d=cross(nv,vec)
			
			# print "       d=",d
			
			sdn=dn/abs(dn)
			if (abs(d)>0.00001):
				sd=d/abs(d)
			else:
				dcheck=0
				sd=1000
			# print "       dn,d=",sdn,sd
		# now we "stepped over" the line..
		# print "stepped over!"
	return [i,j]
	

def remove_duplicates(orig,cop,n,tol=0.0000001):
	# removes from orig all the copy
	# atoms that are present in cop
	#
	# n=1: leave one copy in orig
	# n=0: leave none
	#
	c=0
	dups=1
	gg3=copy.deepcopy(orig)
	for g in cop:
		li=find_neighbours(g,gg3)
		#print "atom:",c
		#print g
		#print "list:"
		#for l in li:
			#print l
		# print
		for l in li[n:]:
		#if (c<=len(gg3)):
			if (l[0]<=tol and gg3[c][0]!='null'): # wtf this las part included?
				gg3[l[1]][0]='null'
		c=c+1
		
	c=0
	while(c<=len(gg3)-1):
		# print "c=",c
		if (gg3[c][0]=="null"):
			gg3.pop(c)
			print "removed duplicate!",dups
			dups=dups+1
			c=0
		else:
			c=c+1
	return gg3


def remove_duplicates2(orig,cop,n,tol=0.0000001):
	# removes from orig all the copy
	# atoms that are present in cop
	#
	# n=1: leave one copy in orig
	# n=0: leave none
	#
	c=0
	dups=1
	gg3=copy.deepcopy(orig)
	for g in cop:
		li=find_neighbours(g,gg3)
		#print "atom:",c
		#print g
		#print "list:"
		#for l in li:
			#print l
		# print
		for l in li[n:]:
		#if (c<=len(gg3)):
			if (l[0]<=tol): # and gg3[c][0]!='null'): # wtf this las part included?
				gg3[l[1]][0]='null'
		c=c+1
		
	c=0
	while(c<=len(gg3)-1):
		# print "c=",c
		if (gg3[c][0]=="null"):
			gg3.pop(c)
			print "removed duplicate!",dups
			dups=dups+1
			c=0
		else:
			c=c+1
	return gg3


def find_duplicates(orig,cop,tol=0.000001):
      c=0
      n=1
      dups=False
      gg3=copy.deepcopy(orig)
      for g in cop:
	      li=find_neighbours(g,gg3)
	      #print "atom:",c
	      #print g
	      #print "list:"
	      #for l in li:
		      #print l
	      # print
	      for l in li[n:]:
	      #if (c<=len(gg3)):
		      if (l[0]<=tol): # and gg3[c][0]!='null'): # wtf this las part included?
			      # gg3[l[1]][0]='null'
			      print "duplicate at orig:",l[1]," cop:",c
			      print "  orig=",gg3[l[1]][0:3+1]
			      print "  copy=",g[0:3+1]
			      dups=True
	      c=c+1
	      
      return dups
  


def translate(geom,vec,spec=[],li=[]):
	newgeom=[]
	c=0
	# print "geom",geom
	# print "vec",vec
	for g in geom:
		if ( ((spec==[]) or (g[0] in spec)) and ((li==[]) or (c in li)) ):
			# print "g>",g
			# print "vec>",vec
			v=[g[0],g[1]+vec[0],g[2]+vec[1],g[3]+vec[2]]
		else:
			v=[g[0],g[1],g[2],g[3]]
		if (len(g)>4):
			v.append(copy.deepcopy(g[4]))
		newgeom.append(v)
		c=c+1
	return newgeom


def pickspec(geom,spec):
	newgeom=[]
	for g in geom:
		if (g[0] in spec):
			newgeom.append(g)
			
	return newgeom


def pickotherspec(geom,spec):
	newgeom=[]
	for g in geom:
		if (g[0] not in spec):
			newgeom.append(g)
			
	return newgeom


def pickspeclist(geom,spec):
	nums=[]
	c=0
	for g in geom:
		if (g[0] in spec):
			nums.append(c)
		c=c+1
			
	return nums


def seg2li(geom,segnum):
	    li=[]
	    
	    c=0
	    for g in geom:
	      iseg=False
	      if (len(g)>4):
		if (g[4].has_key("seg")):
		  # print "seg=",g[4]["seg"]
		  if (g[4]["seg"]==segnum):
		    iseg=True
	      if (iseg==False):
		# print "passing!"
		pass
	      else:
		li.append(c)
	      c=c+1
	    
	    return li


def clearspec(geom,spec):
	newgeom=[]
	for g in geom:
		if (g[0] not in spec):
			newgeom.append(g)
			
	return newgeom
		
		
def clearspecrad(geom,ind,spec,rad):
	newgeom=[]
	for g in geom:
		if ((g[0] not in spec) or (distance(g,geom[ind])<rad)):
			newgeom.append(g)
			
	return newgeom
	

def numspec(geom,spec):
	count=0
	for g in geom:
		if (g[0]==spec):
			count=count+1
	return count


def glue(basegeom,adgeom,source,target,trans,remtarget=False,remtargetlist=[]):
	geom=copy.deepcopy(basegeom)
	geom2=copy.deepcopy(adgeom)
	
	sourceat=basegeom[source]
	targetat=adgeom[target]
	
	origo=add(sourceat[1:4],trans)
	origo2=targetat[1:4]
	o=add(origo,mul(origo2,-1.0))
	
	if (remtarget):
		geom2.pop(target)
	if (remtargetlist!=[]):
		geom2=removesome(geom2,remtargetlist)
	
	geom=geom+translate(geom2,o)
	
	return geom
	

def gluesome(basegeom,adgeom,source,target,trans,spec="Fe"):
	# glue only certain species from adgeom to basegeom
	# species to be ignored must be defined..
	
	geom=copy.deepcopy(basegeom)
	geom2=copy.deepcopy(adgeom)
	
	sourceat=basegeom[source]
	targetat=adgeom[target]
	
	origo=add(sourceat[1:4],trans)
	origo2=targetat[1:4]
	o=add(origo,mul(origo2,-1.0))
	
	# if (remtarget):
	#	geom2.pop(target)
	
	# before glueing things, remove unwanted species..
	geom3=[]
	for g in geom2:
		if (g[0]!=spec):
			geom3.append(g)
	
	geom=geom+translate(geom3,o)
	
	return geom


def gluemore(basegeom,adgeom,source,target,trans,spec="Fe"):
	# glue only certain species from adgeom to basegeom
	# species to be ignored must be defined..
	
	geom=copy.deepcopy(basegeom)
	geom2=copy.deepcopy(adgeom)
	
	sourceat=basegeom[source]
	targetat=adgeom[target]
	
	origo=add(sourceat[1:4],trans)
	origo2=targetat[1:4]
	o=add(origo,mul(origo2,-1.0))
	
	# if (remtarget):
	#	geom2.pop(target)
	
	# glue everything and then remove atoms that are too close each other..
	geom3=copy.deepcopy(geom2)
	geom3=translate(geom3,o)
	# for g in geom2:
	#	if (g[0]!=spec):
	#		geom3.append(g)
	
	# geom=geom+translate(geom3,o)
	# now remove atoms that are too close..
	
	geom=remove_duplicates(geom,geom3,0,tol=0.1)
	
	return geom


def relate(basegeom,adgeom,source,target):
	# translate "adgeom" to conform with "basegeom"
	
	geom=copy.deepcopy(basegeom)
	geom2=copy.deepcopy(adgeom)
	
	#print "basegeom=",basegeom
	#print "adgeom=",adgeom
	#print "source=",source
	#print basegeom[source]
	#stop
	
	sourceat=basegeom[source]
	targetat=adgeom[target]
	
	origo=add(sourceat[1:4],[0,0,0])
	origo2=targetat[1:4]
	o=add(origo,mul(origo2,-1.0))
	
	# if (remtarget):
	#	geom2.pop(target)
	
	# glue everything and then remove atoms that are too close each other..
	geom3=copy.deepcopy(geom2)
	geom3=translate(geom3,o)
	
	return geom3

def relate2(geom1,geom2,at):
	# move geom2 so, that geom1[at] and geom2[at] are at the same point
	v=geomdiff([geom2[at],geom1[at]])
	geom2=translate(geom2,v)
	
	return geom2


def mapgeom(geom1,geom2,treshold=0.01):
	# geom1 ~ big slab
	# geom2 ~ small slab
	mapp=[]
	for g in geom1:
		mapp.append(None)
		
	c1=0
	c2=0
	for g1 in geom1:
		c2=0
		for g2 in geom2:
			d=distance(g1,g2)
			if (d<treshold):
				mapp[c1]=c2
			c2=c2+1
		c1=c1+1
						
	return mapp
			

def gluemap(smallgeom,biggeom,mapp,maxnum=10000):
	# replace things in biggeom by things in smallgeom
	newgeom=copy.deepcopy(biggeom)
	# newgeom=[]
	c=0
	for g in biggeom:
		if (mapp[c]!=None):
			if (mapp[c]<=maxnum):
				newgeom[c][1:4]=smallgeom[mapp[c]][1:4]
			# newgeom.append(g)
			# pass
		c=c+1
	
	c=0
	for g in smallgeom:
		if (c>=maxnum):
			newgeom.append(g)
		c=c+1
	
	
	return newgeom



def gluemap2(smallgeom,biggeom,mapp,maxnum=10000):
	# replace things in biggeom by things in smallgeom
	newgeom=copy.deepcopy(biggeom)
	# newgeom=[]
	c=0
	for g in biggeom:
		if (mapp[c]!=None):
			if (mapp[c]<=maxnum):
				newgeom[c][1:4]=smallgeom[mapp[c]][1:4]
			# newgeom.append(g)
			# pass
		c=c+1
	
	c=0
	for g in smallgeom:
		if (c>=maxnum):
			newgeom.append(g)
		c=c+1
	
	
	return newgeom

def rotationmatrix(angle=0,axis="z"):
	# creates the rotation matrix..
	# alpha=x_angle*2*math.pi
	# beta=y_angle*2*math.pi
	# gamma=z_angle*2*math.pi
	
	alpha=angle*2*math.pi
	
	# rot=[[0,0,0],[0,0,0],[0,0,0]]
	
	
	Qx=[
	[1,	0,				0],
	[0,	math.cos(alpha),	-math.sin(alpha)],
	[0,	math.sin(alpha),	math.cos(alpha)]
	]
	
	Qy=[
	[math.cos(alpha),	0,	math.sin(alpha)],
	[0,			1,			0],
	[-math.sin(alpha),	0,	math.cos(alpha)]
	]
	
	Qz=[
	[math.cos(alpha),	-math.sin(alpha),	0],
	[math.sin(alpha),	math.cos(alpha),	0],
	[0		,		0,		1]
	]
	
	if (axis=="x"):
		Q=Qx
	elif(axis=="y"):
		Q=Qy
	else:
		Q=Qz
	
	
	#rot[0][0]=math.cos(beta)*math.cos(gamma)
	#rot[0][1]=-math.sin(beta)
	#rot[0][2]=-math.cos(beta)*math.sin(gamma)

	#rot[1][0]=math.sin(beta)*math.cos(gamma)
	#rot[1][1]=math.cos(beta)
	#rot[1][2]=-math.sin(beta)*math.sin(gamma)

	#rot[2][0]=math.sin(gamma)
	#rot[2][1]=0.0
	#rot[2][2]=math.cos(gamma)
	
	return Q


def rotations(mol,rota,spec=[]):
	newmol=copy.deepcopy(mol)
	for r in rota:
		print r
		mat=rotationmatrix(angle=r["angle"],axis=r["axis"])
		if (r.has_key("cent")):
			cent=r["cent"]
		else:
			cent=[0,0,0]
		newmol=rotate(mat,newmol,spec=spec,cent=cent)
		
	return newmol


def quickz(geom,li,angle):
  if (li==[]):
    li=range(0,len(geom))
  for l in li:
    geom[l]=rotate(rotationmatrix(angle),[geom[l]],cent=[0,0,0])[0]


def quicky(geom,li,angle):
  if (li!=[]):
    for l in li:
      geom[l]=rotate(rotationmatrix(angle,axis="y"),[geom[l]],cent=[0,0,0])[0]
  else:
    # print "geom=",geom
    geom=rotate(rotationmatrix(angle,axis="y"),geom,cent=[0,0,0])
    # print "geom now=",geom
    #stop
  return geom
    

def quickx(geom,li,angle):
  newgeom=copy.deepcopy(geom)
  if (li!=[]):
    for l in li:
      newgeom[l]=rotate(rotationmatrix(angle,axis="x"),[newgeom[l]],cent=[0,0,0])[0]
  else:
    # print "geom=",geom
    newgeom=rotate(rotationmatrix(angle,axis="x"),newgeom,cent=[0,0,0])
    # print "geom now=",geom
    #stop
  return newgeom




def expand_rotations(ammonia,pos,fac):
	c=0
	geom=[]
	for k in pos.iterkeys():
		print k
		mol=rotations(ammonia,pos[k]["rotations"])
		d=c*fac
		geom=geom+translate(mol,[d,0,0])
		c=c+1
	
	# coords.writexmol("test.xyz",coords.geom2crd(geom))
	return geom


def rotate(mat,gg,spec=[],cent=[0,0,0],li=[]):
	# coordinate transformation using matrix mat
	# cent=rotation center
	c=0
	newgeom=[]
	
	ggg=copy.deepcopy(gg)
	ggg=translate(ggg,mul(cent,-1.0))
	
	if (len(mat)==2):
		for g in ggg:
			x=g[1]*mat[0][0]+g[2]*mat[0][1]
			y=g[1]*mat[1][0]+g[2]*mat[1][1]
			newgeom.append([g[0],x,y,g[3]])
	else:
		num=0
		for g in ggg:
			x=[0,0,0]
			for i in range(0,3):
				for j in range(0,3):
					x[i]=x[i]+g[j+1]*mat[i][j]
			dv=[g[0],x[0],x[1],x[2]]
			if (len(g)>4):
				dv.append(g[4])
			
			# print "x",x
			# print "dv",dv
			
			if (((spec==[]) or (g[0] in spec)) and ((li==[]) or (num in li))):
				newgeom.append(dv)
			else:
				newgeom.append(g)
			num=num+1
	
	# print "newgeom=",newgeom
	
	newgeom=translate(newgeom,cent)
	
	return newgeom


def flipz(geom):
	for g in geom:
		g[3]=-g[3]

def flipx(geom):
	for g in geom:
		g[1]=-g[1]

def inversion(geom):
	newgeom=[]
	for g in geom:
		newgeom.append([g[0],-g[1],-g[2],-g[3]])
	
	return newgeom
	

def rotatevec(mat,gg):
	# transform vecs to geom
	ggg=copy.deepcopy(gg)
	for g in ggg:
		g.insert(0,"H")
	
	newg=rotate(mat,ggg)
	
	# .. geom to vecs..
	for g in newg:
		g.pop(0)
		
	return newg
	
	
def randomize(geom,fac):
	li=[]
	for g in geom:
		dx=random.random()*fac
		dy=random.random()*fac
		dz=random.random()*fac
			
		print "dx=",dx,dy,dz
		li.append([g[0],g[1]+dx,g[2]+dy,g[3]+dz])
	
	return li


def spher2cart(inp,degrees=False,frac=False):
	r=copy.copy(inp[0])
	teta=copy.copy(inp[1])
	phi=copy.copy(inp[2])
	
	if (degrees):
		teta=teta*(2.0*math.pi/360.0)
		phi=phi*(2.0*math.pi/360.0)
		
	if (frac):
		teta=teta*(2.0)*math.pi
		phi=phi*(2.0)*math.pi
	
	x=r*math.sin(teta)*math.cos(phi)
	y=r*math.sin(teta)*math.sin(phi)
	z=r*math.cos(teta)
	
	res=[x,y,z]
	
	return res
	
def mol2cart(at,inp):
	atom=[]
	atom=spher2cart(inp,degrees=True)
	atom.insert(0,at)
	
	return atom


def gluespher(geom,inp):
	newgeom=[]
	# where to glue..
	point=spher2cart(inp,degrees=True)
	for g in geom:
		# print point
		# print g
		newgeom.append(translate([g],point)[0])
	return newgeom


def ar2mat(name,data):
	import numpy
	if (len(data.shape)==2):
		lines=[]
		n=data.shape[0]
		m=data.shape[1]
		lines.append(name+"=[")
		for i in range(0,n):
			col=data[i,:]
			# print "col>",col
			s=""
			for el in col:
				if (abs(el)<0.000001):
					el=0.0
				s=s+str(el)+" "
			s=s+";"
			lines.append(s)
		lines.append("];")
	else:
		lines=[]
		n=data.shape[0]
		m=data.shape[1]
		l=data.shape[2]
		
		cc=1
		lines.append(name+"=zeros("+str(n)+","+str(m)+","+str(l)+");")
		
		for k in range(0,l):
			lines.append(name+"(:,:,"+str(k+1)+")=[")
			for j in range(0,m):
				col=data[:,j,k]
				s=""
				for el in col:
					if (el<0.000001):
						el=0.0
					s=s+str(el)+" "
				s=s+";"
				lines.append(s)
			lines.append("];")
		
		#for row in data[:,:,1]: # data[:,:,1] = 2D => row = 1D
			#print "row=",row
			#lines.append("A(:,:,"+str(cc)+")=[")
			#s=""
			#for col in row: # single element
				#print "col=",col
				#s=s+str(col)+" "
			#s=s+";"
			#lines.append(s)
			#cc=cc+1
			#lines.append("];")
	
	return lines


def nanotube_length(v,nv):
	# find nanotube periodicity
	# v=the chirality or "scrolling" vector (as I call it..)
	# nv=the unit (direction) vector of the translation vector (T)
	[i,j]=find_point(nv,v)
	k=dot(nv,v[0])*i+dot(nv,v[1])*j
	return k

def set_HC_bonds(geom,lns):
	li=find_species(geom,'H')
	for l in li:
		lis=find_neighbours(geom[l],geom)
		# find nearest C atom..
		el=''
		c=-1
		while (el!='C'):
			c=c+1
			el=geom[lis[c][1]][0]
			# print "el=",el
		n=lis[c][1]
		set_bondlength(geom,[l,n],lns)


def setdir(dirtag,mol,at1,at2):
  # set line from at1 to at2 into z direction
  newmol=setorigo(mol,at1)
  
  p1=unitvec(geomdiff([newmol[at1],newmol[at2]]))
  p2=find_normal(p1)
  p3=crossprod(p1,p2)
  
  if (dirtag=="x"):
    old=[p1,p2,p3]
  elif (dirtag=="y"):
    old=[p3,p1,p2]
  elif (dirtag=="z"):
    old=[p2,p3,p1]
  
  new=[[1,0,0],[0,1,0],[0,0,1]]
  mat=transf(new,old)
  newmol=rotate(mat,newmol)
  return newmol


def set2dir(dirtag,dirtag2,mol,at1,at2,at3):
  # set dirtag into direction corresponding from at1 to at2
  # set dirtag2 into normal of plane at1,at2,at3
  newmol=setorigo(mol,at1)
  
  if (dirtag2!=""):
    p1=unitvec(geomdiff([newmol[at1],newmol[at2]]))
    aux=unitvec(geomdiff([newmol[at2],newmol[at3]]))
    p2=unitvec(crossprod(p1,aux))
    p3=crossprod(p1,p2)
  else:
    p1=unitvec(geomdiff([newmol[at1],newmol[at2]]))
    p2=unitvec(find_normal(p1))
    p3=crossprod(p1,p2)
    
  if (dirtag=="x" and dirtag2=="y"):
    old=[p1,p2,p3]
  elif (dirtag=="x" and dirtag2=="z"):
    old=[p1,p3,p2]
  elif (dirtag=="y" and dirtag2=="x"):
    old=[p2,p1,p3]
  elif (dirtag=="y" and dirtag2=="z"):
    old=[p3,p1,p2]
  elif (dirtag=="z" and dirtag2=="x"):
    old=[p2,p3,p1]
  elif (dirtag=="z" and dirtag2=="y"):
    old=[p3,p2,p1]
  elif (dirtag2==""):
    # specify only one dir
    old=[p1,p2,p3]
    
  else:
    print "weird directions.."
    stop
    
  new=[[1,0,0],[0,1,0],[0,0,1]]
  mat=transf(new,old)
  newmol=rotate(mat,newmol)
  return newmol

def find_minmax(geom,cn):
	c=0
	mi=100000
	ma=-100000
	for g in geom:
		if (g[cn]<mi):
			mi=g[cn]
			nmi=c
		if (g[cn]>ma):
			ma=g[cn]
			nma=c
		c=c+1
			
	return [nmi,nma]
		
		
def cut_geometric(geom,atm):
	# atm is an array of atom numbers..
	lineset=[]
	for n in range(0,len(atm)):
		
		if (n<len(atm)-1):
			fn=n
			tn=n+1
			# print ":)"
		else:
			fn=n
			tn=0
			# print ":("
		
		fr=atm[fn]
		to=atm[tn]
		v=geom[fr][1:3] # this is just in the xy-plane..
		w=geom[to][1:3]
		
		# v -> w
		# print "v,w",v,w
		
		vec=add(mul(v,-1.0),w)
		vec=unitvec(vec) # now we have a unit vector along the line..
		point=copy.copy(v) # .. and a point on it
		# create a "right-handed" normal
		# print "vec=",vec
		nor=[vec[1],-vec[0]]
		# print "nor=",nor
		
		lineset.append([point,nor])
		
	# print "lineset=",lineset
		
	gnew=[]
	c=0
	for g in geom:
		c=c+1
		v=g[1:3]
		# v -> point
		ok=True
		for l in lineset:
			# v -> point
			point=l[0]
			vec=add(mul(point,-1.0),v)
			
			d=dot(vec,l[1])
			
			# if (c==253):
			#	print "l=",l
			#	print d
			
			if (d<-0.001):
				ok=False
		if (ok):
			gnew.append(g)
		else:
			# gnew.append(['B',g[1],g[2],g[3]])
			pass
						
	return gnew
		

def makeper(geom,vecs,sw=0,n=2,tol=0.5,debug=False):
	debug=True
	newgeom=copy.deepcopy(geom)
	u=vecs[0]
	v=vecs[1]
	w=vecs[2]
	for i in range(-n,1): # these used to be -n,n .. ?  -n,1  .. must be like this.. otherwise you remove the same atom twice!
		if (debug):
			print "-------------------"
			print "i=",i
		for j in range(-n,1):
			if (debug):
				print "  j=",j
			for k in range(-n,1):
				if (debug):
					print "    k=",k
					# print "       i,j=",i,j
				vec=add(mul(u,1.0*i),mul(v,1.0*j))
				vec=add(vec,mul(w,1.0*k))
				# print vec
				# stop
				if (((i==0) and (j==0) and (k==0)) or (norm(vec)<0.01)):
					pass
				else:
					if (debug):
						print "              >",i,j,vec
					newgeom=remove_duplicates2(newgeom,translate(geom,vec),sw,tol=tol)
					if (debug):
						print "              len>",len(newgeom)
	# stop
	return newgeom


def findshells(geom,cent,tolerance=0.01,movecent=True):
	c=0
	li=[]
	res=dict()
	
	nl=find_neighbours(cent,geom)
	if (movecent):
		center=geom[nl[0][1]]
	else:
		center=cent
	
	# print "center=",center
	
	for g in geom:
		d=distance(center,g)
		li.append([d,c])
		c=c+1
	li.sort()
	
	# res["d"]=copy.copy(li)
	for l in li:
		# print ">",l
		pass
	
	poin=[]
	lis=[]
	lis.append(poin)
	lold=li[0]
	dists=[0.0]
	for l in li:
		# print "old",lold
		# print "l",l
		if (math.sqrt((lold[0]-l[0])**2)<tolerance):
			poin.append(l[1])
			# print "lis=",lis
		else:
			# print "new shell"
			poin=[]
			lis.append(poin)
			poin.append(l[1])
			lold=l
			dists.append(l[0])
	# print lis
		 
		
	# for l in li:
		# print l
		
	res["shells"]=lis
	res["dist"]=dists
	return res
		
	
def removesome(geom,atnums):
	c=0
	newgeom=[]
	for g in geom:
		remove=False
		for a in atnums:
			if (c==a):
				remove=True
		if (remove):
			pass
		else:
			newgeom.append(copy.deepcopy(g))
			
		c=c+1
	return newgeom
		
		
def removebottom(geom,at,tol=0.0):
	  # remove all atoms with z less or equal than in at..
	  newgeom=[]
	  c=0
	  for g in geom:
	    if (g[3]<=(geom[at][3]+tol)):
	      pass
	    else:
	      newgeom.append(copy.deepcopy(g))
	      
	  return newgeom
	  

def removetop(geom,at,tol=0.0):
	  # remove all atoms with z more or equal than in at..
	  newgeom=[]
	  c=0
	  for g in geom:
	    if (g[3]>=(geom[at][3]+tol)):
	      pass
	    else:
	      newgeom.append(copy.deepcopy(g))
	      
	  return newgeom


def keepsome(geom,atnums):
	c=0
	newgeom=[]
	for g in geom:
		remove=True
		for a in atnums:
			if (c==a):
				remove=False
		if (remove):
			pass
		else:
			newgeom.append(copy.deepcopy(g))
			
		c=c+1
	return newgeom
		

def pickspec(geom,spec=[]):
	newgeom=[]
	for g in geom:
		if ((spec==[]) or (g[0] in spec)):
			newgeom.append(g)
	
	return newgeom

def picknum(geom,num=[],adone=False):
	newgeom=[]
	c=0
	if (adone):
	  c=1
	for g in geom:
		if ((num==[]) or (c in num)):
			newgeom.append(g)
		c=c+1
	return newgeom


def picknumany(geoms,num=[]):
	geoms2=[]
	for geom in geoms:
		geoms2.append(picknum(geom,num))	
	return geoms2
	

def changespec(geom,fr,to,li=[]):
	c=0
	for g in geom:
		# print c
		if ((g[0]==fr) and ((c in li) or (li==[]))):
			# print g[0],c
			g[0]=to
		c=c+1
		

def copystruct(s,lattice,counts):
	i=counts[0]
	j=counts[1]
	geom=[]
	# WTF??
	for i in range(0,3):
		for j in range(0,3):
			tvec=add(mul(lattice[0],i),mul(lattice[1],j))
			# print i,j,tvec
			geom=geom+translate(s,tvec)
	return geom


def copystruct2(s,lattice,counts):
	geom=[]
	
	for i in range(-counts[0],counts[0]):
		for j in range(-counts[1],counts[1]):
			tvec=add(mul(lattice[0],i),mul(lattice[1],j))
			# print i,j,tvec
			geom=geom+translate(s,tvec)
	return geom

def copystruct3(s,lattice,counts):
	geom=[]
	# WTF??
	fac=[1,1]
	fac[0]=abs(counts[0])/counts[0]
	fac[1]=abs(counts[1])/counts[1]
	for i in range(0,abs(counts[0])):
		for j in range(0,abs(counts[1])):
			print "i,j",i*fac[0],j*fac[1]
			tvec=add(mul(lattice[0],i*fac[0]),mul(lattice[1],j*fac[1]))
			# print i,j,tvec
			geom=geom+translate(s,tvec)
	return geom


def hidelayers(s,tol,lic=-1,dis=-2):
	lay=to_layers2(s,1.0)
	# print lay
	# stop
	lines=[]
	for l in lay[0:-1]:
		for c in l:
			lines.append("atom -licorice xmol xmol "+str(c+1))
	for l in lay[0:-2]:
		for c in l:
			lines.append("atom -display xmol xmol "+str(c+1))
	return lines
	
	
def hidelayers2(s,tol,upto):
	lay=to_layers2(s,tol)
	# print lay
	# stop
	lines=[]
	for l in lay[0:upto+1]:
		for c in l:
			lines.append("atom -display xmol xmol "+str(c+1))
	return lines
	

def shownumbers(s,li=[],disp=[0,0,1.0],fromzero=False):
	lines=[]
	c=0
	for m in s:
		c=c+1
		val=c
		if (fromzero):
		  val=c-1
		if ( (li==[]) or (c-1 in li) ):
			lines.append("plot text3 black "+str(m[1]+disp[0])+" "+str(m[2]+disp[1])+" "+str(m[3]+disp[2])+" "+str(val)+" ")
	return lines


def shownumbers2(s,li=[],disp=[0,0,1.0]):
	lines=[]
	c=0
	for m in s:
		if (len(m)>=5):
		  c=m[4]["num"]
		  lines.append("plot text3 black "+str(m[1]+disp[0])+" "+str(m[2]+disp[1])+" "+str(m[3]+disp[2])+" "+str(c)+" ")
	return lines


def tcl_print(geom,li,txs,disp=[0.5,0,0.5],ofs=1,color="black"):
  c=0
  lines=[]
  for l in li:
    m=geom[l-ofs]
    tx=str(txs[c])
    c=c+1
    lines.append("plot text3 "+color+" "+str(m[1]+disp[0])+" "+str(m[2]+disp[1])+" "+str(m[3]+disp[2])+" "+tx+" ")
    
  return lines
  
  
def tcl_print2(geom,n,tx,disp=[0.5,0,0.5],color="black"):
  lines=[]
  m=geom[n]
  lines.append("plot text3 "+color+" "+str(m[1]+disp[0])+" "+str(m[2]+disp[1])+" "+str(m[3]+disp[2])+" "+tx+" ")
    
  return lines
  

  
    
def subcoordinates(geom,typ="h2o",only=[], noextra=True):
	inwater=[]
	newgeom=[]
	newind=[]
	mainlist=["O"] # the "root" atoms .. i.e. the "anchor" atoms for molecules
	for g in geom:
		newind.append(-1)
	
	if (typ=="h2o"):
		#first find O-O length
		if (only==[]):
		  li=pickspeclist(geom,["O"])
		else:
		  li=only
		# print "subcoordinates> li=",li
		firsto=geom[li[0]]
		nei=find_neighbours(firsto,geom,["O"])
		ood=nei[1][0] # O-O distance
		ood=ood*0.95
	
	if (only==[]):
	  geomg=geom
	else:
	  geomg=picknum(geom,only)
	
	c=0
	geomli=[] 
	for g in geomg:
		if (g[0] in mainlist):
		  nei=find_neighbours(g,geom,specs=["O","H","H"])
		  #[[d,number],..]
		  d0=nei[0][0]
		  g0=geom[nei[0][1]]
		  d1=nei[1][0]
		  g1=geom[nei[1][1]]
		  d2=nei[2][0]
		  g2=geom[nei[2][1]]
		  
		  # in the case we have H30 species..
		  d3=nei[3][0]
		  g3=geom[nei[3][1]]
		  
		  if (typ=="h2o"):
			  if ((g0[0]=="O") and (g1[0]=="H") and (g2[0]=="H") and (d1<ood*0.5) and (d2<ood*0.5)):
				  # geomli.append([nei[1][1],nei[2][1]]) # old version.. does some program need this?
				  geomli.append([nei[0][1],nei[1][1],nei[2][1]]) # include also the oxygen atom
				  # number of this guy is..
				  mynum=nei[0][1]
				  
				  # oldind=oldind[newind] # oldind = atom number for the oxygen atom, newind = the main coordinates
				  # newind=newind[oldind]
				  
				  # water molecule ok..
				  subgeom=[]
				  
				  ag=["O",0,0,0]
				  if (len(g0)>4):
				    ag.append(g0[4])
				  
				  subgeom.append(ag)
				  
				  h1=geomdiff([g0,g1])
				  h1.insert(0,"H")
				  if (len(g1)>4):
				    h1.append(g1[4])
		  
				  h2=geomdiff([g0,g2])
				  h2.insert(0,"H")
				  if (len(g2)>4):
				    h2.append(g2[4])
				    
				  h3=geomdiff([g0,g3])
				  h3.insert(0,"H")
				  if (len(g3)>4):
				    h3.append(g3[4])
				  
				  subgeom.append(h1)
				  subgeom.append(h2)
				  # H3O species..
				  if ( (g3[0]=="H") and abs((d3-d1))<0.08):
				    subgeom.append(h3)
				  
				  newgeom.append(["H",g0[1],g0[2],g0[3],{"sub":subgeom}])
				  newind[mynum]=c
				  c=c+1
				  
				  # inwater=inwater+[nei[0][1],nei[1][1],nei[2][1]]
				  inwater.append([nei[0][1],nei[1][1],nei[2][1]])
	
	c=0
	for g in geom:
	  if ((c not in inwater) and noextra==False):
	    #if (g[0]!="H"): # dont include orphaned H's
	    #if (g[0]=="N"):
	    # print "N>",norm(geomdiff([g,["H",0,0,0]]))
	    # if ((norm(geomdiff([g,["H",0,0,0]]))<5.0) or (g[0]!="O") or (g[0]!="H")):
	    # dont include border O or H atoms
	    newgeom.append(g)
	    # pass
	  c=c+1
	
	# print "atoms grouped into molecules:",len(inwater)
	
	res={"geom":newgeom,"ind":newind,"inwater":inwater,"geomli":geomli}
	return res



def groupwater(geom):
  # simpler wa to group atoms into water molecules
  os=pickspeclist(geom,["O"])
  newlis=[]
  for o in os:
    newlis.append(o)
    neis=find_neighbours(geom[o],geom)
    for nei in neis[1:2+1]:
      num=nei[1]
      if (geom[num][0]=="H" and (num not in newlis)):
	newlis.append(num)
  
  # add remaining whatever..
  for i in range(0,len(geom)):
    if (i not in newlis):
      newlis.append(i)
  
  newgeom=[]
  for n in newlis:
    newgeom.append(geom[n])
  
  return newgeom

 
def subadwan(subgeom, cutoff=2.0):
  debug=False
  # for g in subgeom:
  #   print g
  # print 
  
  if (debug): print "subgeom:",len(subgeom)
  
  xgeom=pickspec(subgeom,["X"])
  newsubgeom=pickotherspec(subgeom,["X"])
  # newsubgeom=pickspec(subgeom,["H"])
  
  if (debug):
    # print "xgeom=",xgeom
    print
    # print "newsubgeom=",newsubgeom
    print
    print "xgeom:",len(xgeom)
    print "newsubgeom:",len(newsubgeom)
  
  # print "xgeom=",xgeom
  # print
  # print "newsubgeom=",newsubgeom
  
  if (debug):
    # **************** debugging ***********
    wanpoints=[]
    for w in xgeom:
      wanpoints.append([w[1],w[2],w[3]])
    lis=wanpoints
    lines=tcl_spheres(lis,rad=0.1,scale=1.0,color="green")
    # coords.writecoords("centers.tcl",lines[0:8])
    writecoords("paska.tcl",lines)
    # ***************************************
  
  for g in newsubgeom:
    #if (len(g)<5):
    #  g.append({})
    g[4]["wanpoints"]=[]
    for w in xgeom:
      gd=geomdiff([g,w]) # vector from g to w
      if ( norm(gd) <= cutoff ):
	# attach this point to this molecule..
	v=copy.copy(gd)
	g[4]["wanpoints"].append(['X',v[0],v[1],v[2]])

  if (debug):
    # **************** debugging ***********
    wanpoints=[]
    for g in newsubgeom:
      for ww in g[4]["wanpoints"]:
	w=copy.copy(g)
	w[1]=w[1]+ww[1]
	w[2]=w[2]+ww[2]
	w[3]=w[3]+ww[3]
	wanpoints.append([w[1],w[2],w[3]])
    lis=wanpoints
    lines=tcl_spheres(lis,rad=0.1,scale=1.0,color="green")
    writecoords("paska2.tcl",lines)
    # ***************************************

  if (debug): print "newsubgeom now:",len(newsubgeom)
  return newsubgeom
  
  
def dumpgeomwan(filename,subgeom):
  writexmolmany(filename+".xmol",[expandsub(subgeom)])
  wanpoints=[]
  for g in subgeom:
    for ww in g[4]["wanpoints"]:
      w=copy.copy(g)
      w[1]=w[1]+ww[1]
      w[2]=w[2]+ww[2]
      w[3]=w[3]+ww[3]
      wanpoints.append([w[1],w[2],w[3]])
  lis=wanpoints
  lines=tcl_spheres(lis,rad=0.1,scale=1.0,color="green")
  writecoords(filename+".tcl",lines)
  
  
def expandsub(geom,forget=False):
	# forget: forget subgeom attribs or not?
	newgeom=[]
	for g in geom:
		if (len(g)>4):
		  if (g[4].has_key("sub")):
		    for subg in g[4]["sub"]:
			    xyz=add(g[1:4],subg[1:4])
			    xyz.insert(0,subg[0]) # add element name
			    
			    if (g[4].has_key("fixed")): # add attributes from the main geom
			      xyz.append({})
			      xyz[4]["fixed"]=True
			      
			    if (g[4].has_key("seg")): # add attributes from the main geom
			      xyz.append({})
			      xyz[4]["seg"]=copy.deepcopy(g[4]["seg"])
			      
			    if (len(subg)>4 and forget==False): # .. subgeom attributes override main geom attributes. by def
			      xyz.append(subg[4])
			      
			    newgeom.append(xyz)
		  else:
		    newgeom.append(g)
		else:
		  newgeom.append(g)
			
	return newgeom


def adjust_water(ge,oh=1.0,oho=109):
  geom=subcoordinates(ge)["geom"]
  cc=0
  bonds=[]
  angles=[]
  pairs=[]
  for g in geom:
    print "g=",g
    sub=g[4]["sub"]
    o=numpy.array(sub[0][1:])
    h1=numpy.array(sub[1][1:])
    h2=numpy.array(sub[2][1:])
    # print "o,h1,h2",o,h1,h2
    # print "h1",h1
    # print "h2",h2
    
    
    oxynum=(cc-1)*3
    bonds.append([oh,[oxynum+0,oxynum+1]])
    bonds.append([oh,[oxynum+0,oxynum+2]])
    
    pairs.append([oxynum+0,oxynum+1])
    pairs.append([oxynum+0,oxynum+2])
    
    rota=numpy.cross(h1,h2)
    
    diag=h1+(-h1+h2)/2.0
    
    angle=numpy.pi*oho/360.0 # have been divided by two..
    R0=transformations.rotation_matrix(angle, -rota)
    d1=numpy.inner(R0,diag)
    R0=transformations.rotation_matrix(angle, rota)
    d2=numpy.inner(R0,diag)
    
    d1=(d1/numpy.linalg.norm(d1))*oh
    d2=(d2/numpy.linalg.norm(d2))*oh
  
    el1=['H',d1[0],d1[1],d1[2]]
    el2=['H',d2[0],d2[1],d2[2]]
  
    # print "el1=",el1
    # print "el2=",el2
    # print
  
    geom[cc][4]["sub"][1]=copy.deepcopy(el1)
    geom[cc][4]["sub"][2]=copy.deepcopy(el2)
    
    cc=cc+1
    
  newgeom=expandsub(geom)
  
  res={"newgeom":newgeom,"bonds":bonds,"angles":angles,"pairs":pairs}
  
  return res
  
  
def adjust_water2(ge,oh=1.0,oho=109):
  geom=subcoordinates(ge)["geom"]
  cc=0
  bonds=[]
  angles=[]
  pairs=[]
  for g in geom:
    print "g=",g
    sub=g[4]["sub"]
    o=numpy.array(sub[0][1:])
    h1=numpy.array(sub[1][1:])
    h2=numpy.array(sub[2][1:])
    # print "o,h1,h2",o,h1,h2
    # print "h1",h1
    # print "h2",h2
    
    if ( (lg.norm(h1)<=0.8) or (lg.norm(h1)<=0.8) ):
    
      oxynum=(cc-1)*3
      bonds.append([oh,[oxynum+0,oxynum+1]])
      bonds.append([oh,[oxynum+0,oxynum+2]])
      
      pairs.append([oxynum+0,oxynum+1])
      pairs.append([oxynum+0,oxynum+2])
      
      rota=numpy.cross(h1,h2)
      
      diag=h1+(-h1+h2)/2.0
      
      angle=numpy.pi*oho/360.0 # have been divided by two..
      R0=transformations.rotation_matrix(angle, -rota)
      d1=numpy.inner(R0,diag)
      R0=transformations.rotation_matrix(angle, rota)
      d2=numpy.inner(R0,diag)
      
      d1=(d1/numpy.linalg.norm(d1))*oh
      d2=(d2/numpy.linalg.norm(d2))*oh
    
      el1=['H',d1[0],d1[1],d1[2]]
      el2=['H',d2[0],d2[1],d2[2]]
    
      # print "el1=",el1
      # print "el2=",el2
      # print
    
      geom[cc][4]["sub"][1]=copy.deepcopy(el1)
      geom[cc][4]["sub"][2]=copy.deepcopy(el2)
      
      cc=cc+1
    
  newgeom=expandsub(geom)
  
  res={"newgeom":newgeom,"bonds":bonds,"angles":angles,"pairs":pairs}
  
  return res


def scalewaterto(res,fr,to):
  # scale geometry and lattice coordinates..
  # fr original latcon
  # to new latcon
  geom=res["geom"]
  lattice=res["lattice"]
  subcoords=subcoordinates(geom)["geom"] # reduce
  
  subcoords=rel2abs(subcoords,to/fr) # scale coords
  lattice=muls(lattice,to/fr) # scale lattice
  
  geom=expandsub(subcoords) # expand back
  
  newres={
    "geom" : geom,
    "lattice" : lattice
    }
  
  return newres


def scalewatertoase(geom,fac):
  subcoords=subcoordinates(geom)["geom"] # reduce  
  subcoords=rel2abs(subcoords,fac) # scale coords
  newgeom=expandsub(subcoords) # expand back
  xyz=[]
  for g in newgeom:
    xyz.append([g[1],g[2],g[3]])
  xyz=numpy.array(xyz)
  return xyz


def migrate(geom,atinds):
  # migrate a proton..
  atc=1
  for fr in atinds[0:-1]: # master step
    print "protonmigrator> master step="+str(atc)
    print "protonmigrator> at atom index "+str(fr)
    print "protonmigrator> .. to atom index "+str(atinds[atc])
    geomfr=geom[fr] # source H3O oxygen atom
    to=[atinds[atc]][0] 
    geomto=geom[atinds[atc]] # target H3O oxygen atom
    # lets find the target oxygen's hydrogen neighbours
    neisto=find_neighbours(geomto,geom)
    nh1=neisto[1][1]
    nh2=neisto[2][1]
    
    df=geomdiff([geomfr,geomto])
    dist=norm(df)
    dv=unitvec(df) # directional vector between two oxygen atoms
    
    neis=find_neighbours(geomfr,geom)
    
    pronum=-1 # the proton number to be migrated..
    for nei in neis[1:4]: # 1,2,3
      proto=geom[nei[1]]
      dpro=unitvec(geomdiff([geomfr,proto])) # vector from oxygen to one of its protons
      dire=dot(dv,dpro) # test of proton is in the wanter direction..
      if ((proto[0]=="H") and (dire>0.85)):
	# this is it!
	pronum=copy.deepcopy(nei[1])
    
    # now we have the atom number and the direction we must move it..
    print "protonmigrate> pronum="+str(pronum+1)
    
    # resolve proton to oxygen direction and distance
    p2ov=unitvec(geomdiff([geom[pronum],geomto]))
    p2od=norm(geomdiff([geom[pronum],geomto]))
    
    print "protonmigrate> distance between "+str(atinds[atc]+1)+" "+str(pronum+1)+" "+str(p2od)
    
    # if (debug):
    # double checking..
    # d1=np.linalg.norm(self.atoms[fr].get_position()-self.atoms[pronum].get_position())
    # stderr("to="+str(to)+ "fr="+str(fr))
    # d2=np.linalg.norm(self.atoms[to].get_position()-self.atoms[pronum].get_position())
    # print "protonmigrate> initial O-H-O: "+str(d1)+" "+str(d2)
    
    oh=norm(geomdiff([geomfr,geom[pronum]]))
    
    mainsteps=1
    
    dl=float(p2od-oh)/float(mainsteps)
    dvv=numpy.array(mul(p2ov,dl))
    
    # resolve the finite step length
    #
    # dl=float(dist-2*oh)/float(mainsteps-1)
    # dvv=np.array(coords.mul(dv,dl))
    
    # stderr("protonmigrate> fixing bondlength of "+str(fr)+" "+str(pronum))
    # constraints=FixBondLength(fr,pronum) # wrong!
    # stderr("protonmigrate> fixing bondlength of "+str(to)+" "+str(pronum))
    # constraints2=FixBondLength(to,pronum) # wrong!
    
    
    # ok, now starting to migrate the proton
    
    #stages=range(0,mainsteps) # 0=>N-1 # mainstep = stage of the master step
    
    #if (extrastep):
      #stages=range(0,mainsteps+1) # 0=>N # .. the last step is unconstrained relaxation
    
    #for mainstep in stages: # 0=>N-1 # mainstep = stage of the master step
    
    mainstep=mainsteps
   
    print "   protonmigrate> modifying bond lengths.."
    cr=numpy.array(geom[pronum][1:4])
    # self.atoms[pronum].set_position(self.atoms[pronum].get_position()+dvv)
    cr=cr+dvv
    geom[pronum][1:4]=list(cr)
    
    # if (debug):
    # double checking..
    # d1=np.linalg.norm(self.atoms[fr].get_position()-self.atoms[pronum].get_position())
    # stderr("to="+str(to)+ "fr="+str(fr))
    # d2=np.linalg.norm(self.atoms[to].get_position()-self.atoms[pronum].get_position())
    # stderr("   protonmigrate> O-H-O: "+str(d1)+" "+str(d2))
    atc=atc+1
    

def coordsys(geom,inds):
  # inds: [a,b,c,d]
  # origo: a
  # vecs: (a->b, a->c, a->d)
  a=g2vec(geom[inds[0]])
  b=g2vec(geom[inds[1]])
  c=g2vec(geom[inds[2]])
  d=g2vec(geom[inds[3]])
  
  return (a, b-a, c-a, d-a)
  
  
def adsorbit(geom,site,dire,mol,atind,flip=False):
  # adsorb to geometry "geom" at site "site" (plain coordinates)
  # molecule in such a way that direction defined by two atoms "atind" in molecule
  # point into dire
  
  # adsorbit(geom,origo+0.5*vec1+0.5*vec2,cross(ve2,vec3),KOH,[0,0,1],flip=False)
  # site and dire numpy arrays
  
  mol=setorigo(mol,atind[0])
  print "center atom:",atind[0]
  
  geoms=[]
  geoms.append(copy.deepcopy(mol))
  
  if (flip):
    # flip in the z direction
    # flipz(mol)
    pass
  
  suvex=num2v(dire)
  suvex=unitvec(suvex)
  # print "suvex=",suvex
  # stop
  mole=[mol[atind[1]],mol[atind[2]]]
  molex=unitvec(geomdiff(mole))
  
  # writexmolmany("paskamol0.xmol",[mol])
  
  # first turn molecule x into molex
  x=unitvec(molex)
  z=unitvec(find_normal(x)) # forgot fucking unitvec..!
  y=unitvec(crossprod(x,z))
  crd2=[x,y,z]
  crd1=[[1,0,0],[0,1,0],[0,0,1]]
  
  print "crd2 norms (1):",norm(x),norm(y),norm(z)
  
  mol=rotate(transf(crd1,crd2),mol)
  
  geoms.append(copy.deepcopy(mol))
  # print crd1
  # print crd2
  # writexmolmany("paskamol1.xmol",[mol])
  # stop
  
  # .. then molex into suvex
  x=suvex
  z=unitvec(find_normal(x)) # forgot fucking unitvec...!
  y=unitvec(crossprod(x,z))
  crd2=[x,y,z]
  crd1=[[1,0,0],[0,1,0],[0,0,1]]
  
  print "crd2 norms (2):",norm(x),norm(y),norm(z)
  # print "crd1,crd2>",crd1,crd2
  
  mol=rotate(transf(crd2,crd1),mol)
  
  geoms.append(copy.deepcopy(mol))
  # writexmolmany("rotations.xmol",geoms)
  
  # print crd2
  # writexmolmany("paskamol2.xmol",[mol])

  # stop

  mol=setorigo(mol,atind[0])
  newgeom=geom+translate(mol,v2num(site))
  
  return newgeom


def cover_np(xyz,sequence,num,dist):
	import random
	# gluing stuff to nanoparticle
	# .. into random places
	# .. in sequence, defined in sequence=["B","N"]
	centrify(xyz)
	rad=find_radius(xyz)
	rad=rad["radius"]
	
	pl=dist
	
	random.seed()

	c=0	
	for n in range(0,num):
		rotas=[]
		
		angle=random.random()
		rotas.append({'axis':'x','angle':angle})
		
		angle=random.random()
		rotas.append({'axis':'y','angle':angle})
		
		angle=random.random()
		rotas.append({'axis':'z','angle':angle})
		
		xyz=rotations(xyz,rotas)
		
		if (c>len(sequence)-1):
			c=0
		
		spec=sequence[c]
		
		xyz.append([spec,0,0,rad+pl])
		c=c+1
		
	return xyz
	
def nebpoints(geoms,ps=[0.5],atoms=[],upto=0,parabolic=False,amp=1.0):
	# take two points (their distance is "1")
	# and interpolate a new geometries to points ps
	# between the two points
	
	f=geoms[0] # first points
	l=geoms[1] # last points
	
	if (atoms==[]):
		if (upto>0):
			ll=upto
		else:
			ll=len(f)
		atoms=range(0,ll)
		
	# print atoms
	# stop
	
	v={}
	for a in atoms:
		# print "l[a]=",l[a]
		vec=geomdiff([f[a][0:4],l[a][0:4]])
		ln=distance(l[a],f[a])
		# print "vec=",vec
		# print "ln=",ln
		uv=unitvec(vec)
		v[str(a)]={}
		v[str(a)]["unitvec"]=uv
		v[str(a)]["ln"]=ln
	# stop
	
	newgeoms=[]
	newgeoms.append(f)
	cc=0
	for p in ps:
		newgeom=copy.deepcopy(f)
		# create intermediate geometry..
		for a in atoms:
			uv=v[str(a)]["unitvec"]
			ln=v[str(a)]["ln"]
			dv=mul(uv,ln*p)
			
			if (parabolic):
				xx=float(cc+1)/float(len(ps)+1)
				zmod=((-xx**2+xx)/0.25)*amp
				# print " ******* ZMOD ********"
				# print cc,xx,zmod
				dv[2]=dv[2]+zmod
			
			newgeom[a]=translate([newgeom[a]],dv)[0]		
		newgeoms.append(newgeom)
		cc=cc+1

	newgeoms.append(l)
	
	return newgeoms


def morenebpoints(geoms,np,atoms=[]):
	# we can go from 5 (2 fixed + 3 images) to, say 7 neb points
	f=geoms[0] # first points
	l=geoms[1] # last points
	
	if (atoms==[]):
		atoms=range(0,len(f))
		
	c=0
	geomnp=len(geoms)
	dl=1.0/(geomnp-1)
	xgeom=[]
	for n in range(0,len(geoms)):
		xgeom.append(c*dl)
		c=c+1
		
	c=0
	xnewgeom=[]
	dl=1.0/(np-1)
	for n in range(0,np):
		xnewgeom.append(c*dl)
		c=c+1
	
	# print "xgeom=",xgeom
	# print "xnewgeom=",xnewgeom
	
	newgeoms=[]
	neighs=[]
	for x in xnewgeom:
		# first find nearest points in the old images..
		left=100000
		right=100000
		lp=0
		rp=1000
		c=0
		for xs in xgeom:
			dist=x-xs
			d=abs(dist)
			if (dist>=0):
				if (abs(left)>=d):
					lp=c
					left=d
			if (dist<=0):
				# print "x,xs",x,xs
				if (abs(right)>=d):
					rp=c
					# print "rp===",rp
					right=d
				
			c=c+1
		
		# print "----------"
		
		ne={"l":left,"lp":lp,"rp":rp}
		neighs.append(ne)
		
	c=0
	for x in xnewgeom:
		# print "x=",x,neighs[c]
		c=c+1
	
	# checking the points with matlab..
	# print "-----"
	# for x in xgeom:
	#	print x
	#print "-----"
	#for x in xnewgeom:
	#	print x
				
	# create the new images
	c=0
	newgeoms=[]
	for x in xnewgeom:
		nei=neighs[c]	
		d=nei["l"]
		il=nei["lp"]
		ir=nei["rp"]
		ll=xgeom[ir]-xgeom[il]
		
		# print "d,il,ir",d,il,ir
		
		if (il==ir):
			newgeom=geoms[il]
		else:
			d=d/ll
			auxgeoms=[geoms[il],geoms[ir]]
			nn=nebpoints(auxgeoms,ps=[d],atoms=atoms)
			# print ">",d,nn
			# print "  ==>",nn[1]
			newgeom=nn[1]
	
		newgeoms.append(newgeom)
		
		c=c+1
		
	return newgeoms


def glue111sites(geom,latcon,baseat,typ="111",nums=[],el="H",debug=False):
	import structs3
	# ok, it is named as "111" but works for all cases..
	# baseat = into which atom
	# nums = number of desired sites. [] = all
	# el = which element
	if (typ=="111"):
		sites=copy.deepcopy(structs3.octa111)
	elif (typ=="110"):
		sites=copy.deepcopy(structs3.octa110)
	elif (typ=="100"):
		sites=copy.deepcopy(structs3.octa100)
		
	v=mul(geom[baseat][1:4],1)
	# v=[0,0,0]
	sites=translate(sites,v) # move origo of octahedral site geometries to baseat
	
	if (debug):
		lines=shownumbers(rel2abs(sites,latcon),li=[],disp=[-0.7,0,0]) # write numbers 
		writecoords("nums.tcl",lines)
	
	finalsites=[]
	c=1
	lines=[]
	for s in sites:
		if ((c in nums) or (nums==[])):
			g=copy.deepcopy(s)
			g[0]=el
			finalsites.append(g)
		c=c+1
	
	newgeom=geom+finalsites
	
	return newgeom


def glueanysites(geom,latcon,baseat,sites,nums=[],el="H",debug=False):
	# baseat = into which atom
	# sites = sites typically from structs3
	# nums = number of desired sites. [] = all
	# el = which element
		
	v=mul(geom[baseat][1:4],1)
	# v=[0,0,0]
	sites=translate(sites,v) # move origo of octahedral site geometries to baseat
	
	if (debug):
		lines=shownumbers(rel2abs(sites,latcon),li=[],disp=[-0.7,0,0]) # write numbers 
		writecoords("nums.tcl",lines)
	
	finalsites=[]
	c=1
	lines=[]
	for s in sites:
		if ((c in nums) or (nums==[])):
			g=copy.deepcopy(s)
			g[0]=el
			finalsites.append(g)
		c=c+1
	
	newgeom=geom+finalsites
	
	return newgeom


def examples():

    # some examples:
    obj=[1,2,0,2,1,0,1,2,0,2,1] # bulk termination
    obj=[1,0,2,0,2,0,1,4,0,2,1] # n2s
    obj=[1,0,2,0,2,0,1,0,3,2,1] # n4s
    
    # adds the coordinates in "obj" to the "lines" list
    # lines and new lines are _geometry lists_ in which every elements
    # of the list contains: species, x,y,z
    newlines=[]
    newlines=obj2geom(obj)
    lines=crd2geom(read_file("553.xyz"))
    
    # print "**"
    # print newlines
    # print "********"
    # print lines
    
    all_lines=newlines+lines 
    
    # defining correspondence between species and tags:
    species=[["Si",1],["H",2],["Au",3]]
    
    # converting geometry list into _coordinate list_
    # in which each element (=line) is just a normal string variable
    # .. and converting these into siesta format
    
    # print all_lines
    # print geom2crd(all_lines)
    
    coord_lines=crds2siesta(geom2crd(all_lines),species)
    coord_lines.insert(0,"%block AtomicCoordinatesAndSpecies")
    coord_lines.append("%endblock AtomicCoordinatesAndSpecies")
    
    fdf_lines=read_file("INPUT")
    
    # get coordinates from an fdf file
    # block=get_block(fdf_lines,"AtomicCoordinatesAndAtomicSpecies")
    
    # .. convert them..
    # block2=siestas2crd(block,species)
    
    input_lines=fdf_lines+coord_lines
    
    # print input lines..
    for s in input_lines:
        print s
    
    # get a geometry from a siesta output file..
    geom=get_geom("logi.test_1")
    print(geom)
    
    # requirements for objects:
    # a) 12 slots
    # b) first one is (Si/Au) down = (1,3)
    # c) last one is (Si/Au) up = (2,4)
    # c) consecutive slots can't have the same value
    # d) only 6-8 slots can have values > 0 
    # e) only one Au atom (value 4) is available


def examples2():

    geom=[['Si',0,0,0],['Si',1,0,0],['Si',1,0,1],['Si',1,1,1]]

    find_neighbours(geom[1],geom)
    
    
def examples3():
	
	lc=2.45
	
	# create a periodic nanotube..
	geom=C_layer([11,7])
	for g in geom:
		g.append(0.0)
	writexmol("sheet.xyz",geom2crd(rel2abs(geom,lc)))
	
	crd1=[]
	crd1.append([1.0,0.0])
	crd1.append([0.0,1.0])
	
	# the sheet vectors..
	# v=[]
	# v.append([math.sqrt(3.0/8.0),1/math.sqrt(8.0)]) # n ~ v1
	# v.append([0,1.0/math.sqrt(2.0)]) # m ~ v2
	
	# better to use the standard vectors..
	v=copy.copy(trired_pos)
	
	# some test nanotubes:
	atoms=[17,73] 
	atoms=[17,91]
	
	# some nanotubes with the standard notation..
	atoms=[211,81] 
	n=5
	m=5
	
	# check this!
	atoms=[243,81]
	
	gg=scroll(geom,atoms)
	# find out the scrolling (="chirality") vector..
	
	writexmol("scrollraw.xyz",geom2crd(rel2abs(gg,lc)))
	
	vec=atoms2vec(geom[atoms[0]],geom[atoms[1]])
	# create a unit vector with the direction of the translation T vector
	vu=unitvec(vec)
	nv=[-vu[1],vu[0]]
	# create a coordinate system..
	crd2=[nv,vu]
	
	# create a transform matrix..
	mat=transf(crd2,crd1) # (oldsystem,newsystem)
	
	# rotate nanotube axis into x-axis
	gg=rotate(transf(crd2,crd1),gg)
		
	# cut nanotube from some point..
	g=geom[265]
	gnew=cut_block(gg,[[g[1],100],[-1000,1000],[-1000,1000]])
	
	writexmol("scrollraw1.xyz",geom2crd(rel2abs(gnew,lc)))
	
	# find nanotube periodicity..
	k=nanotube_length(v,nv)
	
	print "k=",k
	
	# cut a periodic bit..
	[t,x,y,z]=gg[222]
	gg2=cut_block(gg,[[x,x+k],[-1000,1000],[-1000,1000]])
			
	# remove duplicates within the unit cell
	gg3=remove_duplicates(gg2,gg2,1)
	# remove duplicates that result from the periodicity
	gg2=remove_duplicates(gg3,translate(gg3,[k,0,0]),0)
	gg3=remove_duplicates(gg2,translate(gg2,[-k,0,0]),0)
	# now we have a periodic unit cell! :D
	
	writexmol("scroll.xyz",geom2crd(rel2abs(gg2,lc)))
	writexmol("scrollmany.xyz",geom2crd(rel2abs(gg2+translate(gg2,[k,0,0])+translate(gg2,[-k,0,0]),lc)))
	
	
def examples4():
	
	lc=2.45
	
	# create a bended stripe of graphene..
	v=copy.copy(trired_pos)

	geom=C_layer([11,7])
	for g in geom:
		g.append(0.0)
	writexmol("sheet.xyz",geom2crd(rel2abs(geom,lc)))
	
	g=geom[240]
	# print "g=",g
	
	print add(v[0],v[1])
	xl=dot(add(v[0],v[1]),[1,0])
	yl=dot(add(v[0],mul(v[1],-1.0)),[0,1])
	
	height=yl*10
	width=xl*6
	geom=cut_block(geom,[[g[1],g[1]+width],[g[2],g[2]+height],[-1000,1000]])
	
	geom=remove_duplicates(geom,translate(geom,[width,0,0]),0)
	geom=remove_duplicates(geom,translate(geom,[-width,0,0]),0)
	
	# ***** hydrogen saturation of the dangling bonds *****
	g=geom[244]
	li=find_block(geom,[[-1000,1000],[g[2]-0.00001,g[2]+0.0001],[-1000,1000]])
	for l in li:
		geom[l][0]='H'
	g=geom[28]
	li=find_block(geom,[[-1000,1000],[g[2]-0.00001,g[2]+0.0001],[-1000,1000]])
	for l in li:
		geom[l][0]='H'
	
	geomsave=copy.deepcopy(geom)
	li=find_species(geom,'H')
	
	# set_bondlength(geom,[251,247],0.3)
	# reset all H-C bond lengths
	for l in li:
		lis=find_neighbours(geom[l],geom)
		# find nearest C atom..
		el=''
		c=-1
		while (el!='C'):
			c=c+1
			el=geom[lis[c][1]][0]
			# print "el=",el
		n=lis[c][1]
		set_bondlength(geom,[l,n],0.3)
	
	geom=bend(geom,[28,244],8.0)
	# lets check the periodicity..
	# geombig=geom+translate(geom,[width+0.2,0,0])+translate(geom,[-width-0.2,0,0])
	# geom=geombig
	writexmol("sheet.xyz",geom2crd(rel2abs(geom,lc)))
	
	
def make_strip(par,dn,hc):
	# create a bended stripe of graphene..
	
	lc=2.45
	
	pey=par[0]
	pex=par[1]
	
	v=copy.copy(trired_pos)
	v2=copy.copy(graphene_basis)
	
	geom=C_layer([11,7])
	for g in geom:
		g.append(0.0)
	# writexmol("sheet.xyz",geom2crd(rel2abs(geom,lc)))
	
	g=geom[240]
	# print "g=",g
	
	# print add(v[0],v[1])
	xl=dot(add(v[0],v[1]),[1,0])
	yl=dot(v[0],[0,1])
	
	# print "yl",yl
	height=yl*pey+0.001
	width=xl*pex
	geom=cut_block(geom,[[g[1],g[1]+width],[g[2],g[2]+height],[-1000,1000]])
	
	geom=remove_duplicates(geom,translate(geom,[width,0,0]),0)
	geom=remove_duplicates(geom,translate(geom,[-width,0,0]),0)
	
	# writexmol("sheet.xyz",geom2crd(rel2abs(geom,lc)))
	# return
	
	# ***** hydrogen saturation of the dangling bonds *****
	[mi,ma]=find_minmax(geom,2)
	g=geom[mi]
	li=find_block(geom,[[-1000,1000],[g[2]-0.00001,g[2]+0.0001],[-1000,1000]])
	for l in li:
		geom[l][0]='H'
	g=geom[ma]
	li=find_block(geom,[[-1000,1000],[g[2]-0.00001,g[2]+0.0001],[-1000,1000]])
	for l in li:
		geom[l][0]='H'
	
	geomsave=copy.deepcopy(geom)
	#li=find_species(geom,'H')
	
	# writexmol("sheet.xyz",geom2crd(rel2abs(geom,lc)))
	# return
	
	# set_HC_bonds(geom,0.3)
	
	# writexmol("sheet.xyz",geom2crd(rel2abs(geom,lc)))
	# return
	
	n=mi
	c=0
	for g in geom:
		if (abs(g[2]==geom[ma][2])<0.001 and (abs(g[1]-geom[mi][1])<0.001)):
			m=c
		c=c+1
	
	# print "n,m",n,m
	
	geom=bend(geom,[n,m],dn)
	set_HC_bonds(geom,hc)
	# lets check the periodicity..
	# geombig=geom+translate(geom,[width+0.2,0,0])+translate(geom,[-width-0.2,0,0])
	# geom=geombig
	# writexmol("sheet.xyz",geom2crd(rel2abs(geom,lc)))
	return [width,geom]
		
def test_5():
	
	lc=2.45
	
	# [xl,geom]=make_strip([14,1],4.0,0.733)
	[xl,geom]=make_strip([14,1],8.0,0.433)
	writexmol("sheet.xyz",geom2crd(rel2abs(geom,lc)))
	
	
def expand(n,v,unitcell): 
	# "n=1" should correspond to a single unit cell..
	# in this routine, "n=0" corresponds to a single unit cell
	n=n-1
	# .. now it is ok
	geom=[]
	
	i=n
	for j in range(0,n+1):
		# print "i,j",i,j
		vec=add(mul(v[0],i),mul(v[1],j))
		# print ">",unitcell,vec
		geom=geom+multiply(unitcell,[vec])
		# print "geom>",geom
		
	# print "><"
	j=n
	for i in range(0,n):
		# print "i,j",i,j
		vec=add(mul(v[0],i),mul(v[1],j))
		geom=geom+multiply(unitcell,[vec])

	return geom
	

def test_6():
	
	lc=2.45
	
	v=copy.copy(trired_pos)
	basis=copy.copy(graphene_basis)
	
	geom=copy.deepcopy(basis)
	unitcell=copy.deepcopy(basis)
	lat=copy.deepcopy(v)
	
	# add more graphene
	
	for i in range(2,10): # 2-> 9.. i corresponds to the new unit cell size (i=1 "the" unit cell)
		geom=geom+expand(i,v,unitcell)
		geom2=copy.deepcopy(geom)
		for g in geom2:
			g.append(0.0)
		# print "geom=",geom
		writexmol("sheet"+str(i)+".xyz",geom2crd(rel2abs(geom2,lc)))
		for j in range(0,2):
			lat[j]=add(lat[j],v[j])
			
		# periodicity check..
		geombig=copy.deepcopy(geom)
		vec=lat
		vec.append(add(lat[0],lat[1]))
		# print "vec=",vec
		geombig=geombig+multiply(geombig,vec)
		geombig2=copy.deepcopy(geombig)
		for g in geombig2:
			# print ">",g
			g.append(0.0)
		writexmol("sheetbig"+str(i)+".xyz",geom2crd(rel2abs(geombig2,lc)))
		
def test_7():
	lc=2.45
	# create a vacancy in graphene and then add more graphene..
	v=copy.copy(trired_pos)
	basis=copy.copy(graphene_basis)
	
	# create 2x2 unit cell.. "vacancy"
	vacancy=copy.copy(basis) # vacancy=basis
	vec=copy.copy(v) # vec=v
	vec.append(add(v[0],v[1])) # add v[0]+v[1] to the vector list, now, v[0],v[1],v[0]+v[1]
	vacancy=basis+multiply(basis,vec) # vacancy=basis+(basis copied to direction of vectors in "vec")
	# now we double the lattice vectors:
	lat=copy.copy(v) # lat=v
	for j in range(0,2): # and double them..
		lat[j]=add(lat[j],v[j])
	print "lat=",lat
	# remove one atom..
	vacancy.pop(2)
	print "vacancy=",vacancy
	# vacancy=dimerize(vacancy,[2,4],0.0)
	
	unitcell=copy.copy(basis) # unitcell=basis
	
	# output of the 2x2 geometry..
	geom2=copy.deepcopy(vacancy)
	for g in geom2: # add the z-coordinate
		g.append(0.0)
	writexmol("sheet"+str(2)+".xyz",geom2crd(rel2abs(geom2,lc)))
	writexmol("sheetbig"+str(2)+".xyz",geom2crd(rel2abs(geom2+multiply(geom2,lat+[add(lat[0],lat[1])]),lc)))
	
	geom=copy.copy(vacancy)
	
	# add more graphene
	for i in range(3,11):
		geom=geom+expand(i,v,unitcell) # this corresponds now to a i x i unit cell
		geom2=copy.deepcopy(geom)
		for g in geom2:
			g.append(0.0)
		# print "geom=",geom
		writexmol("sheet"+str(i)+".xyz",geom2crd(rel2abs(geom2,lc)))
		for j in range(0,2): # create i x i lattice vectors..
			lat[j]=add(lat[j],v[j])
		# periodicity check..
		writexmol("sheetbig"+str(i)+".xyz",geom2crd(rel2abs(geom2+multiply(geom2,lat+[add(lat[0],lat[1])]),lc)))


def sheet(latvec,basis,n=1):
	v=latvec # primitive lattice vectors
	lat=copy.copy(latvec) # this are the lattice vectors to be expanded..
	unitcell=copy.copy(basis) # primitive cell
	geom=copy.copy(basis) # this is the (primitive) cell to be expanded..
	
	# add more unitcells..
	for i in range(1,n):
		if (i>0):
			geom=geom+expand(i+1,v,unitcell) # this corresp,onds now to a i x i unit cell
			for j in range(0,2): # create i x i lattice vectors..
				lat[j]=add(lat[j],v[j])
		
	geom2=copy.deepcopy(geom)
	for g in geom2:
		g.append(0.0)
		
	res=dict()
	res["lattice"]=lat
	res["coords"]=geom2	
	
	return res


def grapheneasy(n,bn=False):
	
	if (bn):
		unitcell=copy.copy(bn_basis)
	else:
		unitcell=copy.copy(graphene_basis)

	# 2D lattice vectors..
	v=copy.copy(trired_pos)
	# print "v=",v
	
	xyz=some_layer2(v,unitcell,[n,n])
	
	xyznew=[]
	for x in xyz:
		xyznew.append([x[0],x[1],x[2],0.0])
	
	# make unit vectors bigger..
	v[0].append(0.0) # add the z-coordinate..
	v[1].append(0.0)
	
	v[0]=mul(v[0],n)
	v[1]=mul(v[1],n)
	
	res={}
	res["lattice"]=v
	res["xyz"]=xyznew
		
	return res


def writeinfopy(infomod,file="info.py"):
	# infomod is a dict
	f=open(file,'w')
	for ff in infomod.iterkeys():
		print ff+"=",infomod[ff]
		f.write(ff+"="+str(infomod[ff])+"\n")
    	f.close()


def readinfopy(filename):
	f=open(filename,'r')
	lines=f.readlines()
	f.close()
	f=[]
	
	for l in lines:
		# print ">",l
		exec(l)
	lines=[]
	# print "locals=",locals()
	dic=locals()
	# remove confusing extra variables..
	dic.pop("f")
	dic.pop("lines")
	dic.pop("l")
	dic.pop("filename")
	# stop
	# return locals() # stupid extra variables
	# print "dic=",dic
	# stop
	return dic


def copyhexas(lis,saturated=False):
	#if (saturated):
		#hexa=[['C', 0.86599999999999999, 1.0, 0.0], ['C', 1.1546666666666667, 0.5, 0.0], ['C', 0.86599999999999999, 0.0, 0.0], ['C', 0.28866666666666674, 0.0, 0.0], ['C', 0.0, 0.5, 0.0], ['C', 0.28866666666666674, 1.0, 0.0], ['H', 1.7320254037844385, 0.5, 0.0], ['H', 1.1546920704511052, -0.5, 0.0], ['H', 1.1546920704511052, 1.5, 0.0], ['H', -2.5403784438604582e-05, 1.5, 0.0], ['H', -0.57735873711777186, 0.5, 0.0], ['H', -2.5403784438604582e-05, -0.5, 0.0]]
	# else: # this crap does not work..
	hexa=[['C', 0.86599999999999999, 1.0, 0.0], ['C', 1.1546666666666667, 0.5, 0.0], ['C', 0.86599999999999999, 0.0, 0.0], ['C', 0.28866666666666674, 0.0, 0.0], ['C', 0.0, 0.5, 0.0], ['C', 0.28866666666666674, 1.0, 0.0]]
	
	# here C-C distance is 0.5773
	
	unitcell=hexa
	v=copy.copy(trired_pos)
	
	xyz=some_layer3(v,unitcell,lis)
	
	xyznew=[]
	for x in xyz:
		xyznew.append([x[0],x[1],x[2],0.0])

	xyz=remove_duplicates(xyznew,xyznew,1,tol=0.05)
	
	if (saturated):
		lis=[]
		cc=0
		for at in xyz:
			nei=find_neighbours(at,xyz)
			neis=[nei[1][0],nei[2][0],nei[3][0]]
			if (abs(abs(neis[0]-neis[1])-abs(neis[1]-neis[2]))>0.001):
				lis.append(cc)
			cc=cc+1
			# print nei
			# print
		# list of undercoordinated atoms in "lis"
		for l in lis:
			nei=find_neighbours(xyz[l],xyz)
			n1=xyz[nei[1][1]]
			n2=xyz[nei[2][1]]
			
			g1=geomdiff([n1,xyz[l]])
			g2=geomdiff([n2,xyz[l]])
			
			v=add(xyz[l][1:4],add(g1,g2))
			xyz.append(["H",v[0],v[1],v[2]])
	# stop
	res={}
	res["xyz"]=xyz
		
	return res


def test_8():
	geom=[
		['Si',1,1,1],
		['Si',0,1,0],
		['Si',0,0,1],
		['Si',0,0,0]
	]
	
	print "dist=",distance(geom[3],geom[0])
	newgeom=dimerize(geom,[3,0],0.5)
	for n in newgeom:
		print n
	print "dist=",distance(newgeom[3],newgeom[0])
	
		
def make_nanotube(atoms):
	
	lc=2.45
	
	# create a periodic nanotube..
	geom=C_layer([11,7])
	for g in geom:
		g.append(0.0)
	writexmol("sheet.xyz",geom2crd(rel2abs(geom,lc)))
	
	crd1=[]
	crd1.append([1.0,0.0])
	crd1.append([0.0,1.0])
	
	v=copy.copy(trired_pos)
	
	# some test nanotubes:
	# atoms=[17,73] 
	# atoms=[17,91]
	
	# some nanotubes with the standard notation..
	# atoms=[211,81] 
	# n=5
	# m=5
	
	# check this!
	# atoms=[243,81]
	
	gg=scroll(geom,atoms)
	# find out the scrolling (="chirality") vector..
	
	writexmol("scrollraw.xyz",geom2crd(rel2abs(gg,lc)))
	
	vec=atoms2vec(geom[atoms[0]],geom[atoms[1]])
	# create a unit vector with the direction of the translation T vector
	vu=unitvec(vec)
	nv=[-vu[1],vu[0]]
	# create a coordinate system..
	crd2=[nv,vu]
	
	# create a transform matrix..
	mat=transf(crd2,crd1) # (oldsystem,newsystem)
	
	# rotate nanotube axis into x-axis
	gg=rotate(transf(crd2,crd1),gg)
		
	# cut nanotube from some point..
	g=geom[265]
	gnew=cut_block(gg,[[g[1],100],[-1000,1000],[-1000,1000]])
	
	writexmol("scrollraw1.xyz",geom2crd(rel2abs(gnew,lc)))
	
	# find nanotube periodicity..
	
	# testing the routines..
	#print "cross:",cross([1,1],[1,1])
	# [i,j]=find_point(unitvec([42,1]),[[1,0],[0,1]])
	# print "i,j=",i,j
	
	[i,j]=find_point(nv,v)
	# print "> i,j=",i,j
	
	k=dot(nv,add(mul(v[0],i),mul(v[1],j)))
	
	# print "k=",k
	
	# cut a periodic bit..
	[t,x,y,z]=gg[222]
	gg2=cut_block(gg,[[x,x+k],[-1000,1000],[-1000,1000]])
			
	# remove duplicates within the unit cell
	gg3=remove_duplicates(gg2,gg2,1)
	# remove duplicates that result from the periodicity
	gg2=remove_duplicates(gg3,translate(gg3,[k,0,0]),0)
	gg3=remove_duplicates(gg2,translate(gg2,[-k,0,0]),0)
	# now we have a periodic unit cell! :D
	
	writexmol("scroll.xyz",geom2crd(rel2abs(gg3,lc)))
	writexmol("scrollmany.xyz",geom2crd(rel2abs(gg3+translate(gg3,[k,0,0])+translate(gg3,[-k,0,0]),lc)))
	
	return [k, gg3]


def make_nanotube2(atn,n,m):
	# create a periodic nanotube..
	# in this version, create a BIIIG sheet of graphene..
	# atn=atom to connecto to an atom that is (n,m) unit vectors apart
	
	lc=2.45
	
	geom=C_layer([15,15])
	for g in geom:
		g.append(0.0)
	writexmol("sheet.xyz",geom2crd(rel2abs(geom,lc)))
	
	crd1=[]
	crd1.append([1.0,0.0])
	crd1.append([0.0,1.0])
	
	# these are nice for denoting nanotubes with our graphene sheet..
	vv=copy.copy(trired_neg)
	
	# for periodicity purposes..
	v=copy.copy(trired_pos)
	
	# some test nanotubes:
	# atoms=[17,73] 
	# atoms=[17,91]
	
	# some nanotubes with the standard notation..
	# atoms=[211,81] 
	# n=5
	# m=5
	
	# check this!
	# atoms=[243,81]
	
	# for g in geom:
	#	g.append(0.0)
		
	# print "geom=",geom
	# n=5
	vec=geom[atn][1:4]
	vec=add(vec,add(mul(vv[0],n),mul(vv[1],m)))
	at=[geom[atn][0],vec[0],vec[1],vec[2]]
	# print "at=",at
	# return
	# atoms=[61]
	li=find_neighbours(at,geom)
	# print "li=",li[0][1]
	# return
	#atoms.append(find_atom(geom,add(vec,add(mul(v[0],n),mul(v[1],n)))))
	# print "atoms=",atoms
	
	atoms=[atn,li[0][1]]
	
	gg=scroll(geom,atoms)
	# find out the scrolling (="chirality") vector..
	
	writexmol("scrollraw.xyz",geom2crd(rel2abs(gg,lc)))
	
	vec=atoms2vec(geom[atoms[0]],geom[atoms[1]])
	# create a unit vector with the direction of the translation T vector
	vu=unitvec(vec)
	nv=[-vu[1],vu[0]]
	# create a coordinate system..
	crd2=[nv,vu]
	
	# create a transform matrix..
	mat=transf(crd2,crd1) # (oldsystem,newsystem)
	
	# rotate nanotube axis into x-axis
	gg=rotate(transf(crd2,crd1),gg)
		
	# cut nanotube from some point..
	g=geom[265]
	gnew=cut_block(gg,[[g[1],100],[-1000,1000],[-1000,1000]])
	
	writexmol("scrollraw1.xyz",geom2crd(rel2abs(gnew,lc)))
	
	# find nanotube periodicity..
	
	# testing the routines..
	#print "cross:",cross([1,1],[1,1])
	# [i,j]=find_point(unitvec([42,1]),[[1,0],[0,1]])
	# print "i,j=",i,j
	
	[i,j]=find_point(nv,v)
	# print "> i,j=",i,j
	
	k=abs(dot(nv,add(mul(v[0],i),mul(v[1],j))))
	
	# print "k=",k
	
	# cut a periodic bit..
	[t,x,y,z]=gg[222]
	gg2=cut_block(gg,[[x,x+k],[-1000,1000],[-1000,1000]])
			
	# remove duplicates within the unit cell
	gg3=remove_duplicates(gg2,gg2,1)
	# remove duplicates that result from the periodicity
	gg2=remove_duplicates(gg3,translate(gg3,[k,0,0]),0)
	gg3=remove_duplicates(gg2,translate(gg2,[-k,0,0]),0)
	# now we have a periodic unit cell! :D
	
	writexmol("scroll.xyz",geom2crd(rel2abs(gg3,lc)))
	writexmol("scrollmany.xyz",geom2crd(rel2abs(gg3+translate(gg3,[k,0,0])+translate(gg3,[-k,0,0]),lc)))
	
	return [k, gg3]
	
	
def make_nanotube3(atn,basis,n,m,debug=False):
	# create a periodic nanotube..
	# in this version, create a BIIIG sheet of graphene..
	# atn=atom to connecto to an atom that is (n,m) unit vectors apart
	# basis should be a basis for system in the triangular lattice.
	
	lc=2.45
	
	#geom=C_layer([15,15])
	
	# getgeom=sheet(trired_pos,basis,15)
	# geom=getgeom["coords"]
	
	geom=some_layer(trired0,basis,[16,13])
	
	# print geom
	
	for g in geom:
		g.append(0.0)
	
	if (debug):
		writexmol("sheet.xyz",geom2crd(rel2abs(geom,lc)))
	
	# return
	
	crd1=[]
	crd1.append([1.0,0.0])
	crd1.append([0.0,1.0])
	
	# these are nice for denoting nanotubes with our graphene sheet..
	vv=copy.copy(trired_neg)
	vv[0].append(0.0)
	vv[1].append(0.0)
	
	# for periodicity purposes..
	v=copy.copy(trired_pos)
	v[0].append(0.0)
	v[1].append(0.0)
	
	# some test nanotubes:
	# atoms=[17,73] 
	# atoms=[17,91]
	
	# some nanotubes with the standard notation..
	# atoms=[211,81] 
	# n=5
	# m=5
	
	# check this!
	# atoms=[243,81]
	
	# for g in geom:
	#	g.append(0.0)
		
	# print "geom=",geom
	# n=5
	vec=geom[atn][1:4]
	
	# print "vec=",vec,"vv=",vv
	
	vec=add(vec,add(mul(vv[0],n),mul(vv[1],m)))
	at=[geom[atn][0],vec[0],vec[1],vec[2]]
	# print "at=",at
	# return
	# atoms=[61]
	li=find_neighbours(at,geom)
	# print "li=",li[0][1]
	# return
	#atoms.append(find_atom(geom,add(vec,add(mul(v[0],n),mul(v[1],n)))))
	# print "atoms=",atoms
	
	atoms=[atn,li[0][1]]
	
	gg=scroll(geom,atoms)
	# find out the scrolling (="chirality") vector..
	
	# writexmol("scrollraw.xyz",geom2crd(rel2abs(gg,lc)))
	
	vec=atoms2vec(geom[atoms[0]],geom[atoms[1]])
	# create a unit vector with the direction of the translation T vector
	vu=unitvec(vec)
	nv=[-vu[1],vu[0]]
	# create a coordinate system..
	crd2=[nv,vu]
	
	# create a transform matrix..
	mat=transf(crd2,crd1) # (oldsystem,newsystem)
	
	# rotate nanotube axis into x-axis
	gg=rotate(transf(crd2,crd1),gg)
		
	# cut nanotube from some point..
	# g=geom[265]
	# gnew=cut_block(gg,[[g[1],100],[-1000,1000],[-1000,1000]])
	
	if (debug):
		writexmol("scrollraw1.xyz",geom2crd(rel2abs(gg,lc)))
	
	# find nanotube periodicity..
	
	# testing the routines..
	#print "cross:",cross([1,1],[1,1])
	# [i,j]=find_point(unitvec([42,1]),[[1,0],[0,1]])
	# print "i,j=",i,j
	
	[i,j]=find_point(nv,v)
	# print "> i,j=",i,j
	
	k=abs(dot(nv,add(mul(v[0],i),mul(v[1],j))))
	
	# print "k=",k
	
	# cut a periodic bit..
	[t,x,y,z]=gg[atn]
	gg2=cut_block(gg,[[x,x+k],[-1000,1000],[-1000,1000]])
			
	# remove duplicates within the unit cell
	gg3=remove_duplicates(gg2,gg2,1)
	# remove duplicates that result from the periodicity
	gg2=remove_duplicates(gg3,translate(gg3,[k,0,0]),0)
	gg3=remove_duplicates(gg2,translate(gg2,[-k,0,0]),0)
	# now we have a periodic unit cell! :D
	
	# writexmol("scroll.xyz",geom2crd(rel2abs(gg3,lc)))
	# writexmol("scrollmany.xyz",geom2crd(rel2abs(gg3+translate(gg3,[k,0,0])+translate(gg3,[-k,0,0]),lc)))
	
	fixgeom(gg3)
	
	res=dict()
	res["len"]=k
	res["coords"]=gg3
	return res
	
	
def test_9():
	
	lc=2.45
	
	# some test nanotubes:
	atoms=[17,73] 
	atoms=[17,91]
	atoms=[211,81] 
	
	[k, gg3]=make_nanotube(atoms)
	
	# create x2 unit cell..
	geom=gg3+translate(gg3,[k,0,0])
	
	unitcell=copy.copy(gg3)
	
	writexmol("scrollx2.xyz",geom2crd(rel2abs(geom,lc)))
	
	geom.pop(19)
	# print "geom=",geom
	
	# geom=dimerize(geom,[2,34],0.6)
	
	writexmol("scrollx2.xyz",geom2crd(rel2abs(geom,lc)))
	# return
	
	# print "geom=",geom
	
	# add more graphene
	for i in range(2,11):
		print "i=",i
		writexmol("sheet"+str(i)+".xyz",geom2crd(rel2abs(geom,lc)))
		
		geom=geom+translate(unitcell,[k*i,0,0])


def test_10(n):
	
	lc=2.45
	
	[k, gg3]=make_nanotube2(61,n,n)
	
	# create x2 unit cell..
	geom=gg3+translate(gg3,[k,0,0])
	
	unitcell=copy.copy(gg3)
	
	writexmol("scrollx2.xyz",geom2crd(rel2abs(geom,lc)))
	
	geom.pop(1)
	# print "geom=",geom
	
	# geom=dimerize(geom,[2,34],0.6)
	
	writexmol("scrollx2.xyz",geom2crd(rel2abs(geom,lc)))
	# return
	
	# print "geom=",geom
	
	# add more graphene
	for i in range(2,11):
		print "i=",i
		writexmol("sheet"+str(i)+".xyz",geom2crd(rel2abs(geom,lc)))
		
		geom=geom+translate(unitcell,[k*i,0,0])
			
def test_11():
	lc=2.8
	geom=create_Fe_nanoparticle(lc,4.0)
	writexmol("nanoparticle.xyz",geom2crd(rel2abs(geom,lc)))
	
def bn_sheet_test():
	res=sheet(trired_pos,bn_basis,10)
	lat=res["lattice"];geom2=res["coords"]
	print lat
	writexmol("sheet.xyz",geom2crd(rel2abs(geom2,2.45)))
	
def bn_nanotube_test():
	res=make_nanotube3(323,bn_basis,10,10,debug=True)
	k=res["len"]
	geom=res["coords"]
	writexmol("tube.xyz",geom2crd(rel2abs(geom,2.45)))
	for i in range(1,4):
		print i
		geom=geom+translate(geom,[k*i,0,0])
	writexmol("tubemany.xyz",geom2crd(rel2abs(geom,2.45)))
	

def test_12():
	geoms=[
	[["Si",0,0,0]],
	[["Si",3,3,3]]
	];	
	newgeoms=nebpoints(geoms,ps=[0.5],atoms=[])
	
	print "--- SOME POINTS -------"
	for n in newgeoms:
		print n
	
	for n in newgeoms:
		print n[0][3]
	
	# stop
	
	# print newgeoms
	newgeoms=morenebpoints(newgeoms,7,atoms=[])
	
	print "--- SOME MORE POINTS -------"
	for n in newgeoms:
		print n
		
	for n in newgeoms:
		print n[0][3]

def cutstring(st,maxlen):
	c=0
	newst=""
	for s in st:
		c=c+1
		if (c>=maxlen and (s=="," or s=="]")):
			# newst=newst+s+"\\\n"
			newst=newst+s+"\n"
			c=0
			# print "--------------"
			# stop
		else:
			newst=newst+s
	# newst="VITTUUUUUUUUUUUUUU"
	return newst

# test_5()
# examples3()
# test_7()
# test_8()
# test_9()
# test_10(10)
# test_11()

# res=sheet(trired_pos,bn_basis,10)
# lat=res["lattice"];geom2=res["coords"]
# print lat
# writexmol("sheet.xyz",geom2crd(rel2abs(geom2,2.45)))

# bn_nanotube_test()

# readxmol("/home/sampsa/nanotubes/surf01/THICK/Fe_slab_test_111_5/Fe_slab.xmol")

# test_12()

# testst="aaaaa bbbbbbbbb ccccccccc"
# newst=cutstring(testst,4)
# print newst

# t1=[["H",0,0,0],["H",1,1,0],["H",2,2,2]]
# t2=[["H",1,0,0],["H",0,0,0],["H",2,2,2]]

# t2=relate2(t1,t2,1)

# print "t1=",t1
# print "t2=",t2

