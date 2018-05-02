import coords
import sys

def vizhbond(geom,scale):
  # find hydrogen bonds
  # tol=3.0 # in angstroms
  # .. we should find this automagically..
  
  lines=[]
  app=False
  c=0
  li=coords.pickspec(geom,"O")
  # print li
  # stop
  tol=coords.distance(li[0],li[1])*1.20
  
  for g in geom:
    if (g[0]=="O"):
      hn=0
      
      nei=coords.find_neighbours(g,geom) # 0=the O atom, 1=H, 2=H, 3,4=hbond H
      
      # first find out the direction of covalent H bonds...
      h1=nei[1][1]
      h2=nei[2][1]
      
      h1=coords.geomdiff([g,geom[h1]])
      h2=coords.geomdiff([g,geom[h2]])
      uv=coords.unitvec(coords.add(h1,coords.mul(coords.add(coords.mul(h1,-1.0),h2),0.5))) # symmetry axis
      
      n1=nei[3][1]
      n2=nei[4][1]
      dv1=coords.geomdiff([g,geom[n1]])
      dv2=coords.geomdiff([g,geom[n2]])
      # .. those are supposedly O that donate hydrogen bonds to us
      
      n1v=coords.dot(dv1,uv)
      n2v=coords.dot(dv2,uv)
      # .. project (neighboring hb donor - to us) vector to our symmetry axis
      # .. should be positive (negative?) and above some treshold
      
      sw=2 # use sw=2, it is more sophisticated
      
      if (sw==1):
	if (geom[n1][0]=="H" and n1v<0):
	  hn=hn+1
	if (geom[n2][0]=="H" and n2v<0):
	  hn=hn+1
	
	if (hn==2):
	  lines=lines+coords.tcl_plotlines(geom,c,n1,color="black",scale=scale,append=app)
	  app=True
	  lines=lines+coords.tcl_plotlines(geom,c,n2,color="black",scale=scale,append=app)
	  
      elif (sw==2):
	if ((geom[n1][0]=="H") and (n1v<=0.01) and (coords.norm(dv1)<=tol)):
	  lines=lines+coords.tcl_plotlines(geom,c,n1,color="black",scale=scale,append=app)
	  app=True
	if ((geom[n2][0]=="H") and (n2v<=0.01) and (coords.norm(dv2)<=tol)):
	  lines=lines+coords.tcl_plotlines(geom,c,n2,color="black",scale=scale,append=app)
	  app=True
	
    c=c+1 
  return lines
  

filename=sys.argv[1]
geom=coords.readxmol(filename)[-1]
folder = filename[:filename.rindex("/")+1]
lines=vizhbond(geom,0.06) # 0.06 for af structure. 0.02 ok
coords.writecoords(folder + "hbonds.tcl",lines)

