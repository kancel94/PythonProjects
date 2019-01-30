import stdio
import sys
import stddraw
import math
import numpy as np
import picture

#reading 2 command line arguments
T=float(sys.argv[1])
deltaT=float(sys.argv[2])

#read from txt file 
N=stdio.readInt() #number of particles
R=stdio.readFloat() #radius of universe

#creating empty arrays for each field
xPosArr = []
yPosArr = []
VelXarr = []
VelYarr = []
massarr = []
imgFilearr = []
for i in range(N): #number of particles determines how many times itll loop
	initX=stdio.readFloat() #initial x coordinate
	initY=stdio.readFloat() #initial y coordinate
	iVelX=stdio.readFloat() #initial velocity x
	iVelY=stdio.readFloat() #initial velocity y
	m=stdio.readFloat() #mass of sun
	imgFile=stdio.readString() #imagefile of particle
	#the values will be appended to arrays
	xPosArr.append(initX)
	yPosArr.append(initY)
	VelXarr.append(iVelX)
	VelYarr.append(iVelY)
	massarr.append(m)
	imgFilearr.append(imgFile)

t=0.0
while t<T:
	#arrays are now populated for calculations to take place
	G=6.67e-11
	ArrCounter=0
	xnewPosArr = []
	ynewPosArr = []
	xForceArr = []
	yForceArr = []
	for n in range(N): 
		indexCounter=0 #counter for including every index but i
		xnetForceArr = []
		ynetForceArr = []
		while indexCounter <= (N-1): #while counter is less than N (it will continuously increment until N and ignore if it matches i)
			if indexCounter != n: #if they dont match
				m1=massarr[n]
				m2=massarr[indexCounter]
				deltaX=xPosArr[indexCounter]-xPosArr[n]
				deltaY=yPosArr[indexCounter]-yPosArr[n]
				r=math.sqrt(deltaX**2+deltaY**2)
				FORCE=(G*m1*m2)/(r**2)
				xForce=FORCE*deltaX/r
				yForce=FORCE*deltaY/r
				xForceArr += [xForce]
				yForceArr += [yForce]
				xForceArr[ArrCounter]=xForce
				yForceArr[ArrCounter]=yForce
				indexCounter+=1
				ArrCounter+=1
			else: #skip that index if it matches
				indexCounter+=1
	xnetForceArr=np.add.reduceat(xForceArr, np.arange(0, len(xForceArr), 4))
	ynetForceArr=np.add.reduceat(yForceArr, np.arange(0, len(yForceArr), 4))
	for i in range(N):
		xAccel=xnetForceArr[i]/massarr[i]
		yAccel=ynetForceArr[i]/massarr[i]
		xnewVel=VelXarr[i]+deltaT*xAccel
		ynewVel=VelYarr[i]+deltaT*yAccel
		xnewPos=xPosArr[i]+deltaT*xnewVel
		ynewPos=yPosArr[i]+deltaT*ynewVel
		VelXarr[i]=xnewVel
		VelYarr[i]=ynewVel
		xPosArr[i]=xnewPos
		yPosArr[i]=ynewPos

	stddraw.clear()
	#canvas scale
	stddraw.setXscale(-R, +R)
	stddraw.setYscale(-R, +R)
	background = picture.Picture('starfield.jpg')
	stddraw.picture(background,0,0)
	for i in range(N):
		planet=picture.Picture(imgFilearr[i])
		stddraw.picture(planet,xPosArr[i],yPosArr[i])
	t+=deltaT
	stddraw.show(1)
stdio.writeln(N)
stdio.writeln(R)
for i in range(N):
	stdio.writef('%7.5e   %7.5e   %7.5e   %7.5e   %7.5e   %15s\n',xPosArr[i], yPosArr[i], VelXarr[i], VelYarr[i], massarr[i], imgFilearr[i])