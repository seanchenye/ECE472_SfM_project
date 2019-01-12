import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
import glob
from numpy import linalg as LA
from numpy.linalg import inv

# import modules used here -- sys is a very standard one
# Reference solution, credits to Ajayan, Akshitha
import sys

def drawmyobject(x,y):
	plt.plot((x[0], x[1]), (y[0], y[1]), 'r')
	plt.plot((x[1], x[2]), (y[1], y[2]), 'r')
	plt.plot((x[2], x[3]), (y[2], y[3]), 'r')
	plt.plot((x[3], x[0]), (y[3], y[0]), 'r')
	plt.plot((x[0], x[4]), (y[0], y[4]), 'r')
	plt.plot((x[1], x[5]), (y[1], y[5]), 'r')
	plt.plot((x[2], x[6]), (y[2], y[6]), 'r')
	plt.plot((x[3], x[7]), (y[3], y[7]), 'r')
	plt.plot((x[4], x[8]), (y[4], y[8]), 'r')
	plt.plot((x[5], x[8]), (y[5], y[8]), 'r')
	plt.plot((x[6], x[8]), (y[6], y[8]), 'r')
	plt.plot((x[7], x[8]), (y[7], y[8]), 'r')
	plt.plot((x[4], x[5]), (y[4], y[5]), 'r')
	plt.plot((x[5], x[6]), (y[5], y[6]), 'r')
	plt.plot((x[6], x[7]), (y[6], y[7]), 'r')
	plt.plot((x[7], x[4]), (y[7], y[4]), 'r')

def drawmy3dobject(x,y,z):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot((x[0], x[1]), (y[0], y[1]), (z[0], z[1]), 'r')
	ax.plot((x[1], x[2]), (y[1], y[2]), (z[1], z[2]), 'r')
	ax.plot((x[2], x[3]), (y[2], y[3]), (z[2], z[3]), 'r')
	ax.plot((x[3], x[0]), (y[3], y[0]), (z[3], z[0]), 'r')
	ax.plot((x[0], x[4]), (y[0], y[4]), (z[0], z[4]), 'r')
	ax.plot((x[1], x[5]), (y[1], y[5]), (z[1], z[5]), 'r')
	ax.plot((x[2], x[6]), (y[2], y[6]), (z[2], z[6]), 'r')
	ax.plot((x[3], x[7]), (y[3], y[7]), (z[3], z[7]), 'r')
	ax.plot((x[4], x[8]), (y[4], y[8]), (z[4], z[8]), 'r')
	ax.plot((x[5], x[8]), (y[5], y[8]), (z[5], z[8]), 'r')
	ax.plot((x[6], x[8]), (y[6], y[8]), (z[6], z[8]), 'r')
	ax.plot((x[7], x[8]), (y[7], y[8]), (z[7], z[8]), 'r')
	ax.plot((x[4], x[5]), (y[4], y[5]), (z[4], z[5]), 'r')
	ax.plot((x[5], x[6]), (y[5], y[6]), (z[5], z[6]), 'r')
	ax.plot((x[6], x[7]), (y[6], y[7]), (z[6], z[7]), 'r')
	ax.plot((x[7], x[4]), (y[7], y[4]), (z[7], z[4]), 'r')
	ax.view_init(elev=-66, azim=90)

def triangulatePoints(pl, pr, Rlr, tlr):
	plt = pl
	prt = pr
	#print "This is plt: " + str(np.matmul(Rlr, prt))
	q = np.cross(plt, np.matmul(Rlr, prt))
	q = q/LA.norm(q)
	A = [plt, np.matmul(-Rlr,prt), q]
	A = np.asarray(A)
	eq = np.matmul(inv(A), tlr)
	return eq[0]*plt+0.5*eq[2]*q

def triangulatePoints2(leftray, rightray, Rlr, tlr, Twl):
	recon = []
	for i in range(0,9):
		wrt_left = triangulatePoints(leftray[:, i], rightray[:, i], Rlr, tlr)
		temp = [wrt_left[0], wrt_left[1], wrt_left[2], 1]
		pts = np.matmul(Twl, np.transpose(np.asarray(temp)))
		recon.append(np.transpose(pts))
	return recon

def main():

	print "\n\nPROBLEM 1:\n"

	r = [-1, 1]
	
	points = [[0, 0, 0],
			[0, 0, 1],
			[1, 0, 1],
			[1, 0, 0],
			[0, 1, 0],
			[0, 1, 1],
			[1, 1, 1],
			[1, 1, 0],
			[0.5, 2, 0.5]]

	points = np.array(points)

	R = [[-1.7, 0, 0],
		[0, 0.707, 0],
		[0, 0, 1]]

	t = [3,
		0.5,
		3]

	Ml = [[-1.7, 0, 0, 3],
		[0, 0.707, 0, 0.5],
		[0, 0, 1, 3]]

	K = [[200, 0, 100],
		[-0, 200, 100],
		[0, 0, 1]]
	
	#print "The original house coordinates are: \n"+str(points)

	temp = []
	for j in range(0, 9):
		X = []
		for i in range(0, 3):
			X.append(points[j, i])
		X.append(1)
		#print X
		#print m
		temp.append(np.matmul(np.asarray(K), (np.matmul(np.asarray(Ml), np.asarray(X)))))
	
	#print temp

	temp = np.asarray(temp)
	xl = []
	yl = []
	for i in range(0, 9):
		 xl.append(temp[i, 0]/temp[i, 2])
		 yl.append(temp[i, 1]/temp[i, 2])

	R = [[0.707, 0, 0],
		[0, 0.707, 0],
		[0, 0, 1]]

	t = [3,
		0.5,
		3]

	Mr = [[0.707, 0, 0, 3],
		[0, 0.707, 0, 0.5],
		[0, 0, 1, 3]]

	K = [[200, 0, 100],
		[-0, 200, 100],
		[0, 0, 1]]

	temp = []
	for j in range(0, 9):
		X = []
		for i in range(0, 3):
			X.append(points[j, i])
		X.append(1)
		#print X
		#print m
		temp.append(np.matmul(np.asarray(K), (np.matmul(np.asarray(Mr), np.asarray(X)))))
	
	#print temp

	temp = np.asarray(temp)
	xr = []
	yr = []
	for i in range(0, 9):
		 xr.append(temp[i, 0]/temp[i, 2])
		 yr.append(temp[i, 1]/temp[i, 2])

	A = []
	for i in range(0,9):
		A.append([xl[i]*xr[i], xl[i]*yr[i], xl[i], yl[i]*xr[i], yl[i]*yr[i], yl[i], xr[i], yr[i], 1])

	A = np.asarray(A)
	U, S, V = np.linalg.svd(A)
	V = np.transpose(V)
	F = [[V[0, -1], V[1, -1], V[2, -1]],
		[V[3, -1], V[4, -1], V[5, -1]],
		[V[6, -1], V[7, -1], V[8, -1]]]
	F = np.asarray(F)
	print("The fundamental matrix F is: \n" + str(F))

	print "\n\nPROBLEM 2:\n"

	E = np.matmul(np.matmul(np.transpose(K), F), K)
	print("The essential matrix E is: \n" + str(E))

	print "\n\nPROBLEM 3:\n"

	drawmyobject(xl,yl)
	plt.title("Left Camera Image")
	plt.draw()
	plt.waitforbuttonpress(0) # this will wait for indefinite time
	plt.close()

	drawmyobject(xr,yr)
	plt.title("Right Camera Image")
	plt.draw()
	plt.waitforbuttonpress(0) # this will wait for indefinite time
	plt.close()

	W = [[0, -1, 0],
		[1, 0, 0],
		[0, 0, 1]]

	W = np.asarray(W)

	Z = [[0, 1, 0],
		[-1, 0, 0],
		[0, 0, 0]]

	Z = np.asarray(Z)

	U, S, V = np.linalg.svd(E)
	V = np.transpose(V)

	Rlr = np.matmul(np.matmul(U, W), np.transpose(V))
	tlr = U[:, 2]
	Mlr = np.array([[Rlr[0,0], Rlr[0,1], Rlr[0,2], tlr[0]],
		[Rlr[1,0], Rlr[1,1], Rlr[1,2], tlr[1]],
		[Rlr[2,0], Rlr[2,1], Rlr[2,2], tlr[2]]])
	
	xrecon = []
	yrecon = []
	zrecon = []
	
	for i in range(0,9):
		pl = [xl[i], yl[i], 1]
		pl = np.transpose(np.asarray(pl))
		pr = [xr[i], yr[i], 1]
		pr = np.transpose(np.asarray(pr))

		pts = triangulatePoints(pl, pr, Rlr, tlr)
		xrecon.append(pts[0])
		yrecon.append(pts[1])
		zrecon.append(pts[2])

	drawmy3dobject(xrecon, yrecon, zrecon)
	plt.title("Reconstruction Using Essential Matrix")
	plt.show()

	print "\n\nPROBLEM 4:\n"

	Tlw = [[-1.7, 0, 0, 3],
		[0, 0.707, 0, 0.5],
		[0, 0, 1, 3],
		[0, 0, 0, 1]]

	Trw = [[0.707, 0, 0, 3],
		[0, 0.707, 0, 0.5],
		[0, 0, 1, 3],
		[0, 0, 0, 1]]
	Trw = np.asarray(Trw)
	Tlw = np.asarray(Tlw)
	Twr = inv(Trw)
	Twl = inv(Tlw)
	Tlr = np.matmul(Tlw, Twr)
	Rlr = Tlr[0:3,0:3]
	tlr = Tlr[0:3,3]
	rpts = []
	lpts = []
	for i in range(0,9):
		rpts.append([xr[i], yr[i], 1])
		lpts.append([xl[i], yl[i], 1])

	rightray = np.matmul(inv(K), np.transpose(np.asarray(rpts)))
	leftray = np.matmul(inv(K), np.transpose(np.asarray(lpts)))
	#print(rightray)
	#print(leftray)
	xrecon1 = []
	yrecon1 = []
	zrecon1 = []
	#for i in range(0,9):
	#	pl = [xl[i], yl[i], 1]
	#	pl = np.transpose(np.asarray(pl))
	#	pr = [xr[i], yr[i], 1]
	#	pr = np.transpose(np.asarray(pr))

	#	pts = triangulatePoints(pl, pr, Rlr, tlr)
	#	xrecon1.append(pts[0])
	#	yrecon1.append(pts[1])
	#	zrecon1.append(pts[2])
	pts = triangulatePoints2(leftray, rightray, Rlr, tlr, Twl)
	pts = np.asarray(pts)
	#print pts
	for i in range(0,9):
		xrecon1.append(pts[i,0])
		yrecon1.append(pts[i,1])
		zrecon1.append(pts[i,2])

	drawmy3dobject(xrecon1, yrecon1, zrecon1)
	plt.title("Reconstruction Using Camera Matrix")
	plt.show()

	print "\n\nPROBLEM 5:\n"

	img1 = cv.imread('test.jpg',0)
	img2 = cv.imread('test2.jpg',0)
	# Initiate ORB detector
	orb = cv.ORB_create()
	# find the keypoints with ORB
	kp1 = orb.detect(img1,None)
	# compute the descriptors with ORB
	kp1, des1 = orb.compute(img1, kp1)

	orb = cv.ORB_create()
	# find the keypoints with ORB
	kp2 = orb.detect(img2,None)
	# compute the descriptors with ORB
	kp2, des2 = orb.compute(img2, kp2)

	# create BFMatcher object
	bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

	# Match descriptors.
	matches = bf.match(des1,des2)

	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)
	img3 = cv.drawMatches(img1,kp1,img2,kp2,matches, None, flags=2)

	plt.imshow(img3),plt.show()

	print "\n\nPROBLEM 6:\n"

	xl = []
	xr = []
	yl = []
	yr = []
	pts1 = []
	pts2 = []

	# For each match...
	for mat in matches:

	    # Get the matching keypoints for each of the images
	    img1_idx = mat.queryIdx
	    img2_idx = mat.trainIdx

	    # x - columns
	    # y - rows
	    # Get the coordinates
	    (x1,y1) = kp1[img1_idx].pt
	    pts1.append(kp1[img1_idx].pt) 
	    (x2,y2) = kp2[img2_idx].pt
	    pts2.append(kp2[img2_idx].pt) 

	    # Append to each list
	    xl.append(x1)
	    yl.append(y1)
	    xr.append(x2)
	    yr.append(y2)

	#A = []
	#or i in range(len(xl)):
#		A.append([xl[i]*xr[i], xl[i]*yr[i], xl[i], yl[i]*xr[i], yl[i]*yr[i], yl[i], xr[i], yr[i], 1])

	#A = np.asarray(A)
	#U, S, V = np.linalg.svd(A)
	#V = np.transpose(V)
	#F = [[V[0, -1], V[1, -1], V[2, -1]],
	#	[V[3, -1], V[4, -1], V[5, -1]],
#		[V[6, -1], V[7, -1], V[8, -1]]]
#	F = np.asarray(F)
	pts1 = np.asarray(pts1, dtype=np.float)
	pts2 = np.asarray(pts2, dtype=np.float)
	F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)

	print("The fundamental matrix F is: \n" + str(F))
	#print("other F is: \n"+str(F1))

	K = [[1.04019837e+03, 0.00000000e+00, 5.65176422e+02],
 		[0.00000000e+00, 1.04465379e+03, 3.45023779e+02],
 		[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
 	K = np.asarray(K)

	E = np.matmul(np.matmul(np.transpose(K), F), K)
	print("\nThe essential matrix E is: \n" + str(E))

	print "\n\nPROBLEM 7:\n"

	U, S, V = np.linalg.svd(E)
	V = np.transpose(V)

	print("The essential matrix E is: \n" + str(E))

	Rlr = np.matmul(np.matmul(U, np.transpose(W)), V)
	S = np.matmul(np.matmul(-U, Z), np.transpose(U))
	tlr = np.transpose(np.asarray([S[2,1], S[0, 2], -S[0,1]]))
	Mlr = np.array([[Rlr[0,0], Rlr[0,1], Rlr[0,2], tlr[0]],
		[Rlr[1,0], Rlr[1,1], Rlr[1,2], tlr[1]],
		[Rlr[2,0], Rlr[2,1], Rlr[2,2], tlr[2]]])
	Mr = np.matmul(K, np.identity(3))
	Mr = np.array([[Mr[0,0], Mr[0,1], Mr[0,2], 0],
		[Mr[1,0], Mr[1,1], Mr[1,2], 0],
		[Mr[2,0], Mr[2,1], Mr[2,2], 0]])
	Ml = np.matmul(K, Mlr)
	print("\nThe right camera matrix derived from E is: \n" + str(Mr))
	print("\nThe left camera matrix derived from E is: \n" + str(Ml))
	print("\nThe Mlr matrix derived from E is: \n" + str(Mlr))

	print "\n\nPROBLEM 8:\n"

	xrecon = []
	yrecon = []
	zrecon = []
	
	#for i in range(len(xl)):
	#	pl = [xl[i], yl[i], 1]
	#	pl = np.transpose(np.asarray(pl))
	#	pr = [xr[i], yr[i], 1]
	#	pr = np.transpose(np.asarray(pr))

	#	pts = triangulatePoints(pl, pr, Rlr, tlr)
	for i in range(len(pts1)):
		#print(pts1[i])
		X = (cv.triangulatePoints(Ml, Mr, pts1[i], pts2[i]))
		#print(X)
		xrecon.append(X[0]/X[3])
		yrecon.append(X[1]/X[3])
		zrecon.append(X[2]/X[3])
	xrecon = np.asarray(xrecon)
	yrecon = np.asarray(yrecon)
	zrecon = np.asarray(zrecon)
	xrecon.flatten()
	yrecon.flatten()
	zrecon.flatten()
	#print(xrecon)
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.scatter(xrecon, yrecon, zrecon)
	ax.view_init(elev=-44, azim=-46)
	#plt.title("3D Reconstruction Point Cloud")
	plt.show()


# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
	main()