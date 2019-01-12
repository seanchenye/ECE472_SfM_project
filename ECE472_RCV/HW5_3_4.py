# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 16:43:56 2018

@author: Shuyu
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print('problem 3')

"""
img1 = np.array([[173.33333333, 160,         160,         173.33333333, 206.66666667,
  210,        210,        206.66666667, 228,        228,       ],
 [183.33333333, 175,         225,         216.66666667, 183.33333333,
  175,        225,         216.66666667, 188,        212,       ]])

img2 = np.array([[217.14995857, 208.18668707, 197.04873027, 208.18668707, 235.91797501,
  227.73261066, 217.14995857, 227.73261066, 242.82556079, 237.79676128],
 [200,         184.69778118, 200,         215.30221882, 200.,
  181.93315266, 200.,         218.06684734, 194.28255608, 206.22032387]])

"""

def takePicture(mExt):
    House_Vert_3d_2 = np.concatenate((House_Vert_3d, np.asarray([np.ones(10)]).T), axis=1)
    House_Vert_2d_homo = M_int.dot(mExt).dot(House_Vert_3d_2.T)
    House_Vert_2d = []
    for vert in House_Vert_2d_homo.T:
        House_Vert_2d.append([vert[0]/vert[2], vert[1]/vert[2]])
    return np.asarray(House_Vert_2d)

def drawmyobject(house2d, ax2d):
    for edge in House_Edge:
        ax2d.plot([house2d[edge[0]-1][0],house2d[edge[1]-1][0]],\
                  [house2d[edge[0]-1][1], house2d[edge[1]-1][1]])

House_Vert_3d = [[0,0,0],
                 [4,0,0],
                 [4,4,0],
                 [0,4,0],
                 [0,0,2],
                 [4,0,2],
                 [4,4,2],
                 [0,4,2],
                 [2,1,3],
                 [2,3,3]]
House_Edge = [(1, 2), (1, 4), (3, 2), (3, 4), (1, 5), (2, 6), (4, 8), (3, 7), \
                 (5, 6), (5, 8), (7, 8), (7, 6), (9, 10), (5, 9), (6, 9), (7, 10), (8, 10)]

M_ext_1 = np.asarray([[-0.707, -0.707, 0, 3], [0.707, -0.707, 0, 0.5], [0, 0, 1, 3]])
M_int = np.asarray([[-100, 0, 200], [-0, -100, 200], [0, 0, 1]])
House_Vert_2d_1 = takePicture(M_ext_1)

theta1 = 5/8*np.pi;  theta2 = 1/8*np.pi
#theta2 and theta1 are set for T_cam_world, But the result we need is T_world_cam
tmp = np.array([[1,0,0],[0, np.cos(theta2), np.sin(theta2)], [0, -np.sin(theta2), np.cos(theta2)]]).T.dot(
        (np.asarray([[np.cos(theta1), -np.sin(theta1), 0], [np.sin(theta1), np.cos(theta1), 0], [0, 0, 1]])))
    
M_ext_2 = np.concatenate((tmp, -tmp.dot(np.asarray([[-3, 6, -4]]).T)), axis=1)
print(tmp)

House_Vert_2d_2 = takePicture(M_ext_2)

fig, ax = plt.subplots()
ax.plot(House_Vert_2d_1[:, 0], House_Vert_2d_1[:,1])
drawmyobject(House_Vert_2d_1, ax)

fig, ax = plt.subplots()
ax.plot(House_Vert_2d_2[:, 0], House_Vert_2d_2[:,1])
drawmyobject(House_Vert_2d_2, ax)

plt.show()
#plt.waitforbuttonpress(0) # this will wait for indefinite time
#plt.close()


A = []

for i in range(0, 8):
    x_l = House_Vert_2d_1[i, 0]; x_r = House_Vert_2d_2[i,0];
    y_l = House_Vert_2d_1[i, 1]; y_r = House_Vert_2d_2[i,1];
    
    
    #x_l = img1[i, 0]; x_r = img2[i,0];
    #y_l = img1[i, 1]; y_r = img2[i,1];
    A.append([x_l*x_r, x_l*y_r, x_l, y_l*x_r, y_l*y_r, y_l, x_r, y_r, 1])
    
A = np.asarray(A)
#print(A)
U, S, V = np.linalg.svd(A)
V=V.T
q = V[:, -1]
#print(q)
F=np.asarray([q[0:3], q[3:6], q[6:]]).T
#print('F: '+str(F))
krt=M_int.T
kl=M_int
E=krt.dot(F).dot(kl)
print(E)
U, S, V = np.linalg.svd(E)
V=V.T
W=np.asarray([[0,-1,0],
  [1,0,0],
  [0,0,1]])
Z=np.asarray([[0,1,0],
  [-1,0,0],
  [0,0,0]])

#S=U.dot(Z).dot(U.T)
R1=U.dot(W).dot(V)
R2=U.dot(W.T).dot(V)
t = U[:,2]

K = np.asarray([[-100, 0, 200],
    [0, -100, 200],
    [0, 0, 1]])

House_recon_3d = []



"""

"""

#Triangulation manually
b_A1_t1 = True; b_A1_t2 = True; b_A2_t1 = True; b_A2_t2 = True; 

for i in range(0, len(House_Vert_3d)):
    p_tilta_l = np.linalg.inv(K).dot(np.asarray([House_Vert_2d_1[i, 0], House_Vert_2d_1[i, 1], 1]))
    p_tilta_r = np.linalg.inv(K).dot(np.asarray([House_Vert_2d_2[i, 0], House_Vert_2d_2[i, 1], 1]))
    #ptim_1 = House_Vert_2d_1[i]; p_tilta_l = np.asarray([(ptim_1[0]-200)*(-1), (ptim_1[1]-200)*(-1), 100])
    #ptim_2 = House_Vert_2d_2[i]; p_tilta_r = np.asarray([(ptim_2[0]-200)*(-1), (ptim_2[1]-200)*(-1), 100])
    q1 = np.cross(p_tilta_l, R1.dot(p_tilta_r.T))
    q2 = np.cross(p_tilta_l, R2.dot(p_tilta_r.T))
    A1 = []; A2 = []
    A1.append(p_tilta_l); A1.append(-R1.dot(p_tilta_r.T)); A1.append(q1); A1 = np.asarray(A1).T
    A2.append(p_tilta_l); A2.append(-R2.dot(p_tilta_r.T)); A2.append(q2); A2 = np.asarray(A2).T
    #start constructing points
    #Dealing with the 4 fold ambiguity
    r_A1_t1 = np.linalg.inv(A1).dot(t).T
    r_A1_t2 = np.linalg.inv(A1).dot(-t).T
    r_A2_t1 = np.linalg.inv(A2).dot(t).T
    r_A2_t2 = np.linalg.inv(A2).dot(-t).T
    
    p_l = r_A1_t1[0]*p_tilta_l + r_A1_t1[2]/2*q1
    """if(p_l[2]>0):
        print(str(i)+":  r_A1_t1")
        House_recon_3d.append(p_l)
        
        continue
    p_l = r_A1_t2[0]*p_tilta_l + r_A1_t2[2]/2*q1
    if(p_l[2]>0):
        print(str(i)+":  r_A1_t2")
        House_recon_3d.append(p_l)
        continue
    p_l = r_A2_t1[0]*p_tilta_l + r_A2_t1[2]/2*q2
    if(p_l[2]>0):
        print(str(i)+":  r_A2_t1")
        House_recon_3d.append(p_l)
        continue
    p_l = r_A2_t2[0]*p_tilta_l + r_A2_t2[2]/2*q2
    House_recon_3d.append(p_l)
    if(~p_l[2]>0):
        print("Error: all four cases less than zerp")
    """
    if(p_l[2]<0):
        print(str(i)+":  r_A1_t1")
        b_A1_t1 = False
    p_l = r_A1_t2[0]*p_tilta_l + r_A1_t2[2]/2*q1
    if(p_l[2]<0):
        print(str(i)+":  r_A1_t2")
        b_A1_t2 = False
    p_l = r_A2_t1[0]*p_tilta_l + r_A2_t1[2]/2*q1
    if(p_l[2]<0):
        print(str(i)+":  r_A2_t1")
        b_A2_t1 = False
    p_l = r_A2_t2[0]*p_tilta_l + r_A2_t2[2]/2*q1
    if(p_l[2]<0):
        print(str(i)+":  r_A2_t1")
        b_A2_t2 = False
        
R_E=[]
t_E=[]
if(b_A1_t1):
    R_E = R1
    t_E = t
if(b_A1_t2):
    R_E = R1
    t_E = -t
if(b_A2_t1):
    R_E = R2
    t_E = t
if(b_A2_t2):
    R_E = R2
    t_E = -t

M1_E = np.identity(4)
M2_E = np.concatenate((np.concatenate((M_int.dot(R_E), M_int.dot(np.asarray([t_E]).T)), axis = 1),
                                  [[0,0,0,1]]), axis=0)
#NN = len(House_Vert_2d_1)
#House_Vert_Homo_1 = np.concatenate((House_Vert_2d_1.T, [np.ones(NN)]), axis = 0)
#House_Vert_Homo_2 = np.concatenate((House_Vert_2d_2.T, [np.ones(NN)]), axis = 0)

tmp1 = House_Vert_2d_1.T; tmp2 = House_Vert_2d_2.T
X = cv2.triangulatePoints(M1_E[:3], M2_E[:3], tmp1, tmp2)

X /= X[3]

House_recon_3d_E = X[:3].T

"""
print('problem 4')

C_l = K.dot(M_ext_1)
C_r = K.dot(M_ext_2)

#credit to HW4_sol

House_recon_3d = []
for i in range(len(House_Vert_2d_1)):
    A=[]
    A.append(House_Vert_2d_1[i, 0]*C_l[2,:]-C_l[0,:])
    A.append(House_Vert_2d_1[i, 1]*C_l[2,:]-C_l[1,:])
    A.append(House_Vert_2d_2[i, 0]*C_r[2,:]-C_r[0,:])
    A.append(House_Vert_2d_2[i, 1]*C_r[2,:]-C_r[1,:])
    
    U, S, V = np.linalg.svd(A)
    V=V.T
    House_recon_3d.append(np.asarray([V[0,-1], V[1,-1], V[2,-1]])/V[3,-1])

House_recon_3d = np.asarray(House_recon_3d)
"""
print('problem 4')

C_l = M_int.dot(M_ext_1)
C_r = M_int.dot(M_ext_2)

#credit to HW4_sol

House_recon_3d = []
for i in range(len(House_Vert_2d_1)):
    A=[]
    A.append(House_Vert_2d_1[i, 0]*C_l[2,:]-C_l[0,:])
    A.append(House_Vert_2d_1[i, 1]*C_l[2,:]-C_l[1,:])
    A.append(House_Vert_2d_2[i, 0]*C_r[2,:]-C_r[0,:])
    A.append(House_Vert_2d_2[i, 1]*C_r[2,:]-C_r[1,:])
    
    U, S, V = np.linalg.svd(A)
    V=V.T
    House_recon_3d.append(np.asarray([V[0,-1], V[1,-1], V[2,-1]])/V[3,-1])

House_recon_3d = np.asarray(House_recon_3d)

def drawmyobject3d(house3d, ax3d):
    for edge in House_Edge:
        ax3d.plot([house3d[edge[0]-1][0],house3d[edge[1]-1][0]],\
                  [house3d[edge[0]-1][1], house3d[edge[1]-1][1]],\
                 [house3d[edge[0]-1][2], house3d[edge[1]-1][2]])
        
fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')
drawmyobject3d(House_recon_3d_E, ax)

ax = fig.add_subplot(212, projection='3d')
drawmyobject3d(House_Vert_3d, ax)
plt.show()

print('problem 4')

C_l = M_int.dot(M_ext_1)
C_r = M_int.dot(M_ext_2)

X = cv2.triangulatePoints(C_l, C_r, tmp1, tmp2)
X /= X[3]

House_recon_3d_C = X[:3].T
fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')
drawmyobject3d(House_recon_3d_C, ax)

ax = fig.add_subplot(212, projection='3d')
drawmyobject3d(House_Vert_3d, ax)
plt.show()
    
import cv2 
path = "O:\\Photographs\\20181106\\"
pic1 = cv2.imread(path+"Bottle_label_1.jpg")
pic2 = cv2.imread(path+"Bottle_label_2.jpg")
gray1= cv2.cvtColor(pic1,cv2.COLOR_BGR2GRAY)
gray2= cv2.cvtColor(pic2,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1,None)
kp2, des2 = sift.detectAndCompute(gray2,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

#img=cv2.drawKeypoints(gray,kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imwrite('sift_keypoints.jpg',img)
#print(img)
good = []
counter = 0
for m,n in matches:
    counter+=1
    if(counter<100):
        print(str(m.distance) + "  " +str(n.distance))
    if m.distance < 0.7*n.distance:
        good.append([m])
        
img3 = cv2.drawMatchesKnn(pic1,kp1,pic2,kp2,good,None,flags=2)
img3 = cv2.resize(img3, (1600,800), interpolation=cv2.INTER_CUBIC)
cv2.imshow('img', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
