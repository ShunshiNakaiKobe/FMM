import numpy as np
import math
import random
import sys
import time

#solid harmonics　を使用


# scipy.specialから球面調和関数、ルジャンドル陪関数をインポート
from math import gamma
from scipy.special import sph_harm
from scipy.special import lpmv

#可視化のためのライブラリえをインポート
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

#Pythonの方がライブラリが充実しているので一旦Pythonで作成してみる!
#「球面調和関数を用いた多重極展開」と「局所展開」を用いて粒子間の作用を効率的に計算!

#極座標系を直交座標系に変換する関数
def orth_convert(r, t, p):
    x = r*np.sin(t)*np.cos(p)
    y = r*np.sin(t)*np.sin(p)
    z = r*np.cos(t)
    return [x, y, z]

#直交座標系を極座標系に変換する関数
def polar_convert(x, y, z):
    r = math.sqrt(x**2+y**2+z**2)
    t = math.acos(z/r)
    p = math.atan2(y,x)
    if p < 0:
        p = 2*math.pi + p
    return [r, t, p]

#階乗の関数の改造版
def my_factorial(n):
    f = math.factorial(abs(n))
    return f

#使う
def re_solid_harm(m,n,p,t,r):
    s = ((r**n)/math.factorial(n+abs(m))) * ((-1)**abs(m)) * lpmv(abs(m), n, math.cos(t))

    if m>=0:
        return complex(s*math.cos(m*p), s*math.sin(m*p))
    else:
        return complex(((-1)**m)*s*math.cos(abs(m)*p), ((-1)**(m+1))*s*math.sin(abs(m)*p))

def ire_solid_harm(m,n,p,t,r):
    s = ((-1)**(n+abs(m))) * (math.factorial(n-abs(m))/(r**(n+1))) * ((-1)**abs(m)) * lpmv(abs(m), n, math.cos(t))
    
    if m>=0:
        return complex(s*math.cos(m*p), s*math.sin(m*p))
    else:
        return complex(((-1)**m)*s*math.cos(abs(m)*p), ((-1)**(m+1))*s*math.sin(abs(m)*p))

#局所展開の係数を計算するのに必要なA(n,m)を計算する関数関数
def a_cal(n,m):
    a = ((-1)**n) / math.sqrt(math.factorial(n-m)*math.factorial(n+m))
    return a

#===========================変数設定============================
"""
number_par:粒子の数(ここだけ変える)
flag:粒子の潜在するセル(レベル4)
orth:粒子の場所(x,y,z)
polar:粒子の場所(r,θ,φ) (距離,0<=θ<=π,0<=φ<2π)
distance_par:粒子同士の距離
quantity_par:粒子の電荷量
potential:注目する粒子のポテンシャル
"""
number_par = 100 #粒子の数(ここだけ変える)
orth_f = [[0] * 3 for i in range(number_par)]
re_orth = [[0] * 3 for i in range(number_par)]
polar_f = [[0] * 3 for i in range(number_par)]
distance_par_f = [[0] * number_par for i in range(number_par)]
quantity_par_f = [0] * number_par
mass_par_f = [0] * number_par

orth_d = [[0] * 3 for i in range(number_par)]
polar_d = [[0] * 3 for i in range(number_par)]
distance_par_d = [[0] * number_par for i in range(number_par)]
quantity_par_d = [0] * number_par
mass_par_d = [0] * number_par

potential_f = [[0] * 100 for i in range(number_par)]
power_f = [[0] * 3 for i in range(number_par)]
re_power_f = 0.0
acceleration_f = [[0] * 3 for i in range(number_par)]
velocity_f = [[0] * 3 for i in range(number_par)]

potential_d = [[0] * 100 for i in range(number_par)]
power_d = [[0] * 3 for i in range(number_par)]
re_power_d = 0.0
acceleration_d = [[0] * 3 for i in range(number_par)]
velocity_d = [[0] * 3 for i in range(number_par)]

#可視化のために変数を追加
X = [0] * number_par
Y = [0] * number_par
Z = [0] * number_par

dif_values = []
t_values = []

frames = []

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

"""
near_par:近くの粒子の直接計算したポテンシャル

cellsize:セルのサイズ[-a,a]×[-a,a]

point_point:ある点からみたある点の場所
multipole_point:多重極展開を行う点(x,y,z)
local_point:局所展開を行う点(x,y,z)
n_max:nの最大値
multipole:多極子モーメント(n,m) n=0,1,2,3 m=-3,-2,-1,0,1,2,3
localcoefficient:局所展開の係数n=0,1,2,3 m=-3,-2,-1,0,1,2,3
"""
near_par_f=[0] * number_par
distant_par_f = [0] * number_par
temp_power_par_f = [[0] * 3 for i in range(number_par)]
power_par_f = [[0] * 3 for i in range(number_par)]

near_par_d=[0] * number_par
distant_par_d = [0] * number_par
temp_power_par_d = [[0] * 3 for i in range(number_par)]
power_par_d = [[0] * 3 for i in range(number_par)]

simu_size = 10000.0
max_level = 2
line = 2
cell_size = simu_size/(2**max_level)

point_point_f = [0] * 3
temp_point1_f = [0] * 3
temp_point2_f = [0] * 3

point_point_d = [0] * 3
temp_point1_d = [0] * 3
temp_point2_d = [0] * 3

re_point_point = [[0] * 3 for i in range(number_par)]
re_distance = [0] * number_par

n_max = 3

ex_point0 = [0] * 3
ex_point1 = [[[[0] * (3) for z in range(2)] for y in range(2)] for x in range(2)]
ex_point2 = [[[[0] * (3) for z in range(4)] for y in range(4)] for x in range(4)]
ex_point3 = [[[[0] * (3) for z in range(8)] for y in range(8)] for x in range(8)]
ex_point4 = [[[[0] * (3) for z in range(16)] for y in range(16)] for x in range(16)]

local_point = [0] * 3

flag = [[[0] * 3 for i in range(number_par)] for a in range(max_level+1)]
check = [[[[0] * (2**max_level) for y in range(2**max_level)] for x in range(2**max_level)] for a in range(max_level+1)]

ex_multipole0 = [[0] * (100) for i in range(n_max+1)]

multipole0 = [[0] * (100) for i in range(n_max+1)]
multipole1 = [[[[[0] * (100) for i in range(n_max+1)] for z in range(2)] for y in range(2)] for x in range(2)]
multipole2 = [[[[[0] * (100) for i in range(n_max+1)] for z in range(4)] for y in range(4)] for x in range(4)]
multipole3 = [[[[[0] * (100) for i in range(n_max+1)] for z in range(8)] for y in range(8)] for x in range(8)]
multipole4 = [[[[[0] * (100) for i in range(n_max+1)] for z in range(16)] for y in range(16)] for x in range(16)]

localcoefficient0 = [[0] * (100) for i in range(n_max+1)]
localcoefficient1 = [[[[[0] * (100) for i in range(n_max+1)] for z in range(2)] for y in range(2)] for x in range(2)]
localcoefficient2 = [[[[[0] * (100) for i in range(n_max+1)] for z in range(4)] for y in range(4)] for x in range(4)]
localcoefficient3 = [[[[[0] * (100) for i in range(n_max+1)] for z in range(8)] for y in range(8)] for x in range(8)]
localcoefficient4 = [[[[[0] * (100) for i in range(n_max+1)] for z in range(16)] for y in range(16)] for x in range(16)]

"""
epsilon:真空の誘電率
coulomb:クーロン定数
"""
epsilon = 1.00000
#epsilon = 8.85419e-12
coulomb = 1.0000/(4.0000*math.pi*epsilon)

"""
attempt_number:試行回数
error:誤差
total_error:誤差合計
parameter:精度パラメータ
tottal_parameter:精度パラメータ合計

"""
number_step = 100
t_step = 0.05
error = [[0] * 100 for i in range(number_par)]
total_error = [0] * 100
parameter = [[0] * 100 for i in range(number_par)]
total_parameter = [0] * 100
total_power_f = [0] * 2
total_power_d = [0] * 2
parameter_D = [0] * number_step
dif_power = [0] * number_par
ave_distance = [0] * number_step

time_start1 = 0.0
time_end1 = 0.0
time_start2 = 0.0
time_end2 = 0.0

P = [0] * number_step
T = [0] * number_step

random.seed(2000)

#===========================初期設定(繰り返しで変わらない)============================
#各粒子について
#テキストファイルから初期位置を決定
"""
with open('orth.txt', 'r') as f:
    lines = f.readlines()
    for i in range(number_par):
        orth_f[i][0] = float(lines[3*i].strip())
        orth_f[i][1] = float(lines[3*i+1].strip())
        orth_f[i][2] = float(lines[3*i+2].strip())
"""

for i in range(number_par):
    
    #電荷量、質量設定
    quantity_par_f[i] = 100.0
    quantity_par_d[i] = 100.0
    #quantity_par[i] = -1.60218e-19
    mass_par_f[i] = 0.00001
    mass_par_d[i] = 0.00001
    
    #粒子の初期位置設定 
    orth_f[i][0] = random.uniform(-simu_size/3.0, simu_size/3.0)
    orth_f[i][1] = random.uniform(-simu_size/3.0, simu_size/3.0)
    orth_f[i][2] = random.uniform(-simu_size/3.0, simu_size/3.0)

    orth_d[i][0] = orth_f[i][0]
    orth_d[i][1] = orth_f[i][1]
    orth_d[i][2] = orth_f[i][2]
    

    #実験のために粒子の初期位置を固定
    """
    if i%3 == 0:
        orth_f[i][0] = 3124.406443379831
        orth_f[i][1] = 4740.565479250263
        orth_f[i][2] = -3422.3647450202393
    elif i%3 == 1:
        orth_f[i][0] = 2545.001707281201
        orth_f[i][1] = 278.43275712714956
        orth_f[i][2] = 3059.384189565783
    elif i%3 == 2:
        orth_f[i][0] = -5742.538829254972
        orth_f[i][1] = -2434.031131437611
        orth_f[i][2] = 1570.9576126362317

    orth_d[i][0] = orth_f[i][0]
    orth_d[i][1] = orth_f[i][1]
    orth_d[i][2] = orth_f[i][2]
    """

    
    #実験用の粒子の位置を固定する場合
    """
    if i == 0:
        orth_f[i][0] = -100.0
        orth_f[i][1] = -100.0
        orth_f[i][2] = -100.0

        orth_d[i][0] = -100.0
        orth_d[i][1] = -100.0
        orth_d[i][2] = -100.0
    else :
        orth_f[i][0] = 100.0
        orth_f[i][1] = 100.0
        orth_f[i][2] = 100.0

        orth_d[i][0] = 100.0
        orth_d[i][1] = 100.0
        orth_d[i][2] = 100.0
    """
    

    #どのセルにいるか
    for a in range(max_level+1):
        flag[a][i][0] = int((orth_f[i][0]+(simu_size/2.0))/(cell_size*(2**(max_level-a))))
        flag[a][i][1] = int((orth_f[i][1]+(simu_size/2.0))/(cell_size*(2**(max_level-a))))
        flag[a][i][2] = int((orth_f[i][2]+(simu_size/2.0))/(cell_size*(2**(max_level-a))))
        
    #極座標表示
    polar_f[i][0],polar_f[i][1],polar_f[i][2] = polar_convert(orth_f[i][0],orth_f[i][1],orth_f[i][2])
    polar_d[i][0],polar_d[i][1],polar_d[i][2] = polar_convert(orth_d[i][0],orth_d[i][1],orth_d[i][2])



#展開点
for l in range(max_level+1):
    for x in range(2**l):
        for y in range(2**l):
            for z in range(2**l):
                
                if l == 0:
                    ex_point0[0] = 0.0
                    ex_point0[1] = 0.0
                    ex_point0[2] = 0.0
                elif l == 1:
                    ex_point1[x][y][z][0] = (x - 2**(l-1)) * (simu_size/(2**l)) + simu_size/(2**(l+1))
                    ex_point1[x][y][z][1] = (y - 2**(l-1)) * (simu_size/(2**l)) + simu_size/(2**(l+1))
                    ex_point1[x][y][z][2] = (z - 2**(l-1)) * (simu_size/(2**l)) + simu_size/(2**(l+1))
                elif l == 2:
                    ex_point2[x][y][z][0] = (x - 2**(l-1)) * (simu_size/(2**l)) + simu_size/(2**(l+1))
                    ex_point2[x][y][z][1] = (y - 2**(l-1)) * (simu_size/(2**l)) + simu_size/(2**(l+1))
                    ex_point2[x][y][z][2] = (z - 2**(l-1)) * (simu_size/(2**l)) + simu_size/(2**(l+1))
                elif l == 3:
                    ex_point3[x][y][z][0] = (x - 2**(l-1)) * (simu_size/(2**l)) + simu_size/(2**(l+1))
                    ex_point3[x][y][z][1] = (y - 2**(l-1)) * (simu_size/(2**l)) + simu_size/(2**(l+1))
                    ex_point3[x][y][z][2] = (z - 2**(l-1)) * (simu_size/(2**l)) + simu_size/(2**(l+1))
                elif l == 4:
                    ex_point4[x][y][z][0] = (x - 2**(l-1)) * (simu_size/(2**l)) + simu_size/(2**(l+1))
                    ex_point4[x][y][z][1] = (y - 2**(l-1)) * (simu_size/(2**l)) + simu_size/(2**(l+1))
                    ex_point4[x][y][z][2] = (z - 2**(l-1)) * (simu_size/(2**l)) + simu_size/(2**(l+1))
                    
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[繰り返し部分]>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
for t in range(number_step):
    
    #リセット

    #受ける力
    for i in range(number_par):
        temp_power_par_f[i][0] = 0.0
        temp_power_par_f[i][1] = 0.0
        temp_power_par_f[i][2] = 0.0

        temp_power_par_d[i][0] = 0.0
        temp_power_par_d[i][1] = 0.0
        temp_power_par_d[i][2] = 0.0
        
        power_par_f[i][0] = 0.0
        power_par_f[i][1] = 0.0
        power_par_f[i][2] = 0.0

        power_par_d[i][0] = 0.0
        power_par_d[i][1] = 0.0
        power_par_d[i][2] = 0.0

        power_d[i][0] = 0.0
        power_d[i][1] = 0.0
        power_d[i][2] = 0.0


    total_power_f[1] = 0.0
    total_power_d[0] = 0.0

    #ポテンシャル
    for i in range(number_par):
        for j in range(50):
            potential_f[i][j] = 0.0
            potential_d[i][j] = 0.0
    
    error = [[0] * 100 for i in range(number_par)]
    total_error = [0] * 100
    parameter = [[0] * 100 for i in range(number_par)]
    total_parameter = [0] * 100

    for i in range(number_par):
        for a in range(max_level+1):
            flag[a][i][0] = int((orth_f[i][0]+(simu_size/2.0))/(cell_size*(2**(max_level-a))))
            flag[a][i][1] = int((orth_f[i][1]+(simu_size/2.0))/(cell_size*(2**(max_level-a))))
            flag[a][i][2] = int((orth_f[i][2]+(simu_size/2.0))/(cell_size*(2**(max_level-a))))
        
    #極座標表示
    polar_f[i][0],polar_f[i][1],polar_f[i][2] = polar_convert(orth_f[i][0],orth_f[i][1],orth_f[i][2])
    polar_d[i][0],polar_d[i][1],polar_d[i][2] = polar_convert(orth_d[i][0],orth_d[i][1],orth_d[i][2])
    
    #距離計算
    for i in range(number_par):
        for j in range(i+1,number_par):
            
            point_point_f[0] = orth_f[i][0] - orth_f[j][0]
            point_point_f[1] = orth_f[i][1] - orth_f[j][1]
            point_point_f[2] = orth_f[i][2] - orth_f[j][2]

            point_point_d[0] = orth_d[i][0] - orth_d[j][0]
            point_point_d[1] = orth_d[i][1] - orth_d[j][1]
            point_point_d[2] = orth_d[i][2] - orth_d[j][2]

            #距離
            distance_par_f[i][j] = math.sqrt(point_point_f[0]**2+point_point_f[1]**2+point_point_f[2]**2)
            distance_par_f[j][i] = distance_par_f[i][j]

            distance_par_d[i][j] = math.sqrt(point_point_d[0]**2+point_point_d[1]**2+point_point_d[2]**2)
            distance_par_d[j][i] = distance_par_d[i][j]

    dif_distance = 0.0

    ex_multipole0 = [[0] * (100) for i in range(n_max+1)]

    multipole0 = [[0] * (100) for i in range(n_max+1)]
    multipole1 = [[[[[0] * (100) for i in range(n_max+1)] for z in range(2)] for y in range(2)] for x in range(2)]
    multipole2 = [[[[[0] * (100) for i in range(n_max+1)] for z in range(4)] for y in range(4)] for x in range(4)]
    multipole3 = [[[[[0] * (100) for i in range(n_max+1)] for z in range(8)] for y in range(8)] for x in range(8)]
    multipole4 = [[[[[0] * (100) for i in range(n_max+1)] for z in range(16)] for y in range(16)] for x in range(16)]

    localcoefficient0 = [[0] * (100) for i in range(n_max+1)]
    localcoefficient1 = [[[[[0] * (100) for i in range(n_max+1)] for z in range(2)] for y in range(2)] for x in range(2)]
    localcoefficient2 = [[[[[0] * (100) for i in range(n_max+1)] for z in range(4)] for y in range(4)] for x in range(4)]
    localcoefficient3 = [[[[[0] * (100) for i in range(n_max+1)] for z in range(8)] for y in range(8)] for x in range(8)]
    localcoefficient4 = [[[[[0] * (100) for i in range(n_max+1)] for z in range(16)] for y in range(16)] for x in range(16)]

    #===============================多重極展開+局所展開+空間分割[準備]================================
    print("~~~~~~~~~~~~~~~~~~[Step"+str(t+1)+"]~~~~~~~~~~~~~~~~~~")
    #計測開始
    time_start2 = time.perf_counter()
    
    #===P2M===
    #基本セルの全てのMAXレベルセルの多極子モーメント計算
    for i in range(number_par):

        if max_level == 1:
            temp_point1_f[0] = ex_point1[flag[1][i][0]][flag[1][i][1]][flag[1][i][2]][0]
            temp_point1_f[1] = ex_point1[flag[1][i][0]][flag[1][i][1]][flag[1][i][2]][1]
            temp_point1_f[2] = ex_point1[flag[1][i][0]][flag[1][i][1]][flag[1][i][2]][2]
        elif max_level == 2:
            temp_point1_f[0] = ex_point2[flag[2][i][0]][flag[2][i][1]][flag[2][i][2]][0]
            temp_point1_f[1] = ex_point2[flag[2][i][0]][flag[2][i][1]][flag[2][i][2]][1]
            temp_point1_f[2] = ex_point2[flag[2][i][0]][flag[2][i][1]][flag[2][i][2]][2]
        elif max_level == 3:
            temp_point1_f[0] = ex_point3[flag[3][i][0]][flag[3][i][1]][flag[3][i][2]][0]
            temp_point1_f[1] = ex_point3[flag[3][i][0]][flag[3][i][1]][flag[3][i][2]][1]
            temp_point1_f[2] = ex_point3[flag[3][i][0]][flag[3][i][1]][flag[3][i][2]][2]
        elif max_level == 4:
            temp_point1_f[0] = ex_point4[flag[4][i][0]][flag[4][i][1]][flag[4][i][2]][0]
            temp_point1_f[1] = ex_point4[flag[4][i][0]][flag[4][i][1]][flag[4][i][2]][1]
            temp_point1_f[2] = ex_point4[flag[4][i][0]][flag[4][i][1]][flag[4][i][2]][2]
        
        point_point_f[0] = temp_point1_f[0]-orth_f[i][0]
        point_point_f[1] = temp_point1_f[1]-orth_f[i][1]
        point_point_f[2] = temp_point1_f[2]-orth_f[i][2]
        
        #極座標表示
        temp_point2_f = polar_convert(point_point_f[0],point_point_f[1],point_point_f[2])


        #多極子モーメントの計算(式1.1)
        for n in range(n_max+1):
            for m in range(-n,n+1):
                
                if max_level == 1:
                    multipole1[flag[1][i][0]][flag[1][i][1]][flag[1][i][2]][n][m+n_max] = multipole1[flag[1][i][0]][flag[1][i][1]][flag[1][i][2]][n][m+n_max]\
                        + quantity_par_f[i] * re_solid_harm(m, n, temp_point2_f[2], temp_point2_f[1], temp_point2_f[0])
                elif max_level == 2:
                    multipole2[flag[2][i][0]][flag[2][i][1]][flag[2][i][2]][n][m+n_max] = multipole2[flag[2][i][0]][flag[2][i][1]][flag[2][i][2]][n][m+n_max]\
                        + quantity_par_f[i] * re_solid_harm(m, n, temp_point2_f[2], temp_point2_f[1], temp_point2_f[0])
                elif max_level == 3:
                    multipole3[flag[3][i][0]][flag[3][i][1]][flag[3][i][2]][n][m+n_max] = multipole3[flag[3][i][0]][flag[3][i][1]][flag[3][i][2]][n][m+n_max]\
                        + quantity_par_f[i] * re_solid_harm(m, n, temp_point2_f[2], temp_point2_f[1], temp_point2_f[0])
                elif max_level == 4:
                    multipole4[flag[4][i][0]][flag[4][i][1]][flag[4][i][2]][n][m+n_max] = multipole4[flag[4][i][0]][flag[4][i][1]][flag[4][i][2]][n][m+n_max]\
                        + quantity_par_f[i] * re_solid_harm(m, n, temp_point2_f[2], temp_point2_f[1], temp_point2_f[0])
                    
    #sys.exit()
    
    #===M2M===
    for l in range(max_level):
        
        for x in range(int(2**(max_level-l-1))):
            for y in range(int(2**(max_level-l-1))):
                for z in range(int(2**(max_level-l-1))):
                        
                    for x2 in range(2):
                        for y2 in range(2):
                            for z2 in range(2):

                                if (max_level-l) == 4:
                                    point_point_f[0] = ex_point3[x][y][z][0]-ex_point4[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][0]
                                    point_point_f[1] = ex_point3[x][y][z][1]-ex_point4[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][1] 
                                    point_point_f[2] = ex_point3[x][y][z][2]-ex_point4[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][2]
                                elif (max_level-l) == 3:
                                    point_point_f[0] = ex_point2[x][y][z][0]-ex_point3[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][0]
                                    point_point_f[1] = ex_point2[x][y][z][1]-ex_point3[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][1]
                                    point_point_f[2] = ex_point2[x][y][z][2]-ex_point3[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][2]
                                elif (max_level-l) == 2:
                                    point_point_f[0] = ex_point1[x][y][z][0]-ex_point2[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][0]
                                    point_point_f[1] = ex_point1[x][y][z][1]-ex_point2[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][1]
                                    point_point_f[2] = ex_point1[x][y][z][2]-ex_point2[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][2]
                                elif (max_level-l) == 1:
                                    point_point_f[0] = ex_point0[0]-ex_point1[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][0]
                                    point_point_f[1] = ex_point0[1]-ex_point1[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][1]
                                    point_point_f[2] = ex_point0[2]-ex_point1[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][2]
                                    
                                #極座標表示
                                temp_point1_f = polar_convert(point_point_f[0],point_point_f[1],point_point_f[2])

                                #多極子モーメントの中心点シフト(式1.2)？
                                for j in range(n_max+1):
                                    for k in range(-j,j+1):
                                        for n in range(n_max+1):
                                            for m in range(-n,n+1):
                                                
                                                if (j-n)>=0:
                                                    if (max_level-l) == 4:
                                                        multipole3[x][y][z][j][k+n_max] = multipole3[x][y][z][j][k+n_max] \
                                                            + multipole4[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][n][m+n_max] * re_solid_harm(k-m, j-n, temp_point1_f[2], temp_point1_f[1], temp_point1_f[0])
                                                    elif (max_level-l) == 3:
                                                        multipole2[x][y][z][j][k+n_max] = multipole2[x][y][z][j][k+n_max] \
                                                            + multipole3[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][n][m+n_max] * re_solid_harm(k-m, j-n, temp_point1_f[2], temp_point1_f[1], temp_point1_f[0])
                                                    elif (max_level-l) == 2:
                                                        multipole1[x][y][z][j][k+n_max] = multipole1[x][y][z][j][k+n_max] \
                                                            + multipole2[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][n][m+n_max] * re_solid_harm(k-m, j-n, temp_point1_f[2], temp_point1_f[1], temp_point1_f[0])
                                                    elif (max_level-l) == 1:
                                                        multipole0[j][k+n_max] = multipole0[j][k+n_max] \
                                                            + multipole1[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][n][m+n_max] * re_solid_harm(k-m, j-n, temp_point1_f[2], temp_point1_f[1], temp_point1_f[0])
    
    #sys.exit()
    
    #===M2L===
    for l in range(max_level+1):
        
        for x in range(2**l):
            for y in range(2**l):
                for z in range(2**l):

                    for x2 in range(2*(line+3)+1):

                        if l!=0:
                            if (x%2)==0:
                                if x2 == 0:
                                    continue
                            else:
                                if x2 == 2*(line+3):
                                    continue

                        x2 = x + (x2 - (line+3))

                        if x2 < 0 or x2 > 2**l-1:
                            continue
                        
                        point_point_f[0] = -(x2 - x) *  (cell_size * (2.0 ** (max_level-l)))
            
                        for y2 in range(2*(line+3)+1):

                            if l!=0:
                                if (y%2)==0:
                                    if y2 == 0:
                                        continue
                                else:
                                    if y2 == 2*(line+3):
                                        continue

                            y2 = y + (y2 - (line+3))

                            if y2 < 0 or y2 > 2**l-1:
                                continue

                            point_point_f[1] = -(y2 - y) *  (cell_size * (2.0 ** (max_level-l)))
                
                            for z2 in range(2*(line+3)+1):

                                if l!=0:
                                    if (z%2)==0:
                                        if z2 == 0:
                                            continue
                                    else:
                                        if z2 == 2*(line+3):
                                            continue

                                z2 = z + (z2 - (line+3))
                                
                                if z2 < 0 or z2 > 2**l-1:
                                    continue
                    
                                point_point_f[2] = -(z2 - z) *  (cell_size * (2.0 ** (max_level-l)))

                                if (abs(x2 - x) > line) or (abs(y2 - y) >  line) or (abs(z2 - z) >  line):

                                    #極座標表示
                                    temp_point1_f = polar_convert(point_point_f[0],point_point_f[1],point_point_f[2])


                                    #多極子モーメントから局所展開係数を求める(式1.3)？
                                    for j in range(n_max+1):
                                        for k in range(-j,j+1):
                                            for n in range(n_max+1):
                                                for m in range(-n,n+1):
                                                    
                                                    if l == 0:
                                                        localcoefficient0[j][k+n_max] = localcoefficient0[j][k+n_max] \
                                                            + multipole0[n][m+n_max] * ire_solid_harm(-(m+k), n+j, temp_point1_f[2], temp_point1_f[1], temp_point1_f[0])
                                                    elif l == 1:
                                                        localcoefficient1[x][y][z][j][k+n_max] = localcoefficient1[x][y][z][j][k+n_max] \
                                                            + multipole1[x2][y2][z2][n][m+n_max] * ire_solid_harm(-(m+k), n+j, temp_point1_f[2], temp_point1_f[1], temp_point1_f[0])
                                                    elif l == 2:
                                                        localcoefficient2[x][y][z][j][k+n_max] = localcoefficient2[x][y][z][j][k+n_max] \
                                                            + multipole2[x2][y2][z2][n][m+n_max] * ire_solid_harm(-(m+k), n+j, temp_point1_f[2], temp_point1_f[1], temp_point1_f[0])
                                                    elif l == 3:
                                                        localcoefficient3[x][y][z][j][k+n_max] = localcoefficient3[x][y][z][j][k+n_max] \
                                                            + multipole3[x2][y2][z2][n][m+n_max] * ire_solid_harm(-(m+k), n+j, temp_point1_f[2], temp_point1_f[1], temp_point1_f[0])
                                                    elif l == 4:
                                                        localcoefficient4[x][y][z][j][k+n_max] = localcoefficient4[x][y][z][j][k+n_max] \
                                                            + multipole4[x2][y2][z2][n][m+n_max] * ire_solid_harm(-(m+k), n+j, temp_point1_f[2], temp_point1_f[1], temp_point1_f[0])
          
    #sys.exit()
        
    #===L2L===
    #基本セルの全てのレベル3,2,1,0セルの多極子モーメント計算
    for l in range(max_level+1):

        for x in range(2**l):
            for y in range(2**l):
                for z in range(2**l):

                    for x2 in range(2):
                        for y2 in range(2):
                            for z2 in range(2):
                                
                                if l == 0:
                                    point_point_f[0] = ex_point1[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][0]-ex_point0[0]
                                    point_point_f[1] = ex_point1[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][1]-ex_point0[1]
                                    point_point_f[2] = ex_point1[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][2]-ex_point0[2]
                                elif l == 1:
                                    point_point_f[0] = ex_point2[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][0]-ex_point1[x][y][z][0]
                                    point_point_f[1] = ex_point2[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][1]-ex_point1[x][y][z][1]
                                    point_point_f[2] = ex_point2[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][2]-ex_point1[x][y][z][2]
                                elif l == 2:
                                    point_point_f[0] = ex_point3[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][0]-ex_point2[x][y][z][0]
                                    point_point_f[1] = ex_point3[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][1]-ex_point2[x][y][z][1]
                                    point_point_f[2] = ex_point3[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][2]-ex_point2[x][y][z][2]
                                elif l == 3:
                                    point_point_f[0] = ex_point4[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][0]-ex_point3[x][y][z][0]
                                    point_point_f[1] = ex_point4[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][1]-ex_point3[x][y][z][1]
                                    point_point_f[2] = ex_point4[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][2]-ex_point3[x][y][z][2]
                        
                                #極座標表示
                                temp_point1_f = polar_convert(point_point_f[0],point_point_f[1],point_point_f[2])

                                #局所展開の中心点のシフト(式1.4)
                                for j in range(n_max+1):
                                    for k in range(-j,j+1):
                                        for n in range(n_max+1):
                                            for m in range(-n,n+1):
                                                
                                                if (n-j)>=0:  
                                                    if l == 0:
                                                        localcoefficient1[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][j][k+n_max] = localcoefficient1[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][j][k+n_max] \
                                                            + localcoefficient0[n][m+n_max] * re_solid_harm(m-k, n-j, temp_point1_f[2], temp_point1_f[1], temp_point1_f[0])
                                                    elif l == 1:
                                                        localcoefficient2[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][j][k+n_max] = localcoefficient2[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][j][k+n_max] \
                                                            + localcoefficient1[x][y][z][n][m+n_max] * re_solid_harm(m-k, n-j, temp_point1_f[2], temp_point1_f[1], temp_point1_f[0])
                                                    elif l == 2:
                                                        localcoefficient3[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][j][k+n_max] = localcoefficient3[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][j][k+n_max] \
                                                            + localcoefficient2[x][y][z][n][m+n_max] * re_solid_harm(m-k, n-j, temp_point1_f[2], temp_point1_f[1], temp_point1_f[0])
                                                    elif l == 3:
                                                        localcoefficient4[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][j][k+n_max] = localcoefficient4[int(x*2)+x2][int(y*2)+y2][int(z*2)+z2][j][k+n_max] \
                                                            + localcoefficient3[x][y][z][n][m+n_max] * re_solid_harm(m-k, n-j, temp_point1_f[2], temp_point1_f[1], temp_point1_f[0])
    
    #sys.exit()

    #===L2P===
    #近くのポテンシャル計算
    for i in range(number_par):
        
        #近くの粒子は直接計算

        x = flag[max_level][i][0]
        y = flag[max_level][i][1]
        z = flag[max_level][i][2]

        for j in range(i+1,number_par):
            
            x2 = flag[max_level][j][0]
            y2 = flag[max_level][j][1]
            z2 = flag[max_level][j][2]

            if (abs(x-x2) <= line) and (abs(y-y2) <= line) and (abs(z-z2) <= line):
                point_point_f[0] = orth_f[i][0] - orth_f[j][0]
                point_point_f[1] = orth_f[i][1] - orth_f[j][1]
                point_point_f[2] = orth_f[i][2] - orth_f[j][2]

                temp_power_par_f[i][0] = temp_power_par_f[i][0] + coulomb * quantity_par_f[i] * quantity_par_f[j] * (point_point_f[0]/distance_par_f[i][j]) / (distance_par_f[i][j]**2)
                temp_power_par_f[i][1] = temp_power_par_f[i][1] + coulomb * quantity_par_f[i] * quantity_par_f[j] * (point_point_f[1]/distance_par_f[i][j]) / (distance_par_f[i][j]**2)
                temp_power_par_f[i][2] = temp_power_par_f[i][2] + coulomb * quantity_par_f[i] * quantity_par_f[j] * (point_point_f[2]/distance_par_f[i][j]) / (distance_par_f[i][j]**2)
            
                temp_power_par_f[j][0] = temp_power_par_f[j][0] - coulomb * quantity_par_f[i] * quantity_par_f[j] * (point_point_f[0]/distance_par_f[i][j]) / (distance_par_f[i][j]**2)
                temp_power_par_f[j][1] = temp_power_par_f[j][1] - coulomb * quantity_par_f[i] * quantity_par_f[j] * (point_point_f[1]/distance_par_f[i][j]) / (distance_par_f[i][j]**2)
                temp_power_par_f[j][2] = temp_power_par_f[j][2] - coulomb * quantity_par_f[i] * quantity_par_f[j] * (point_point_f[2]/distance_par_f[i][j]) / (distance_par_f[i][j]**2)
                
                #極座標表示
                temp_point1_f = polar_convert(point_point_f[0],point_point_f[1],point_point_f[2])
                
                #ポテンシャル計算
                near_par_f[i] = near_par_f[i] + coulomb * quantity_par_f[i]*quantity_par_f[j]/temp_point1_f[0]
                near_par_f[j] = near_par_f[j] + coulomb * quantity_par_f[j]*quantity_par_f[i]/temp_point1_f[0]
        
        #局所展開を行う点から見た粒子の場所
        if max_level == 1:
            temp_point1_f[0] = ex_point1[x][y][z][0]
            temp_point1_f[1] = ex_point1[x][y][z][1]
            temp_point1_f[2] = ex_point1[x][y][z][2]
        elif max_level == 2:
            temp_point1_f[0] = ex_point2[x][y][z][0]
            temp_point1_f[1] = ex_point2[x][y][z][1]
            temp_point1_f[2] = ex_point2[x][y][z][2]
        elif max_level == 3:
            temp_point1_f[0] = ex_point3[x][y][z][0]
            temp_point1_f[1] = ex_point3[x][y][z][1]
            temp_point1_f[2] = ex_point3[x][y][z][2]
        elif max_level == 4:
            temp_point1_f[0] = ex_point4[x][y][z][0]
            temp_point1_f[1] = ex_point4[x][y][z][1]
            temp_point1_f[2] = ex_point4[x][y][z][2]
            
        point_point_f[0] = orth_f[i][0] - temp_point1_f[0]
        point_point_f[1] = orth_f[i][1] - temp_point1_f[1]
        point_point_f[2] = orth_f[i][2] - temp_point1_f[2]
        
        temp_point2_f = polar_convert(point_point_f[0],point_point_f[1],point_point_f[2])

        #局所展開を用いたポテンシャル計算
        #該当箇所(式1.5)？
        element=0.0
        for j in range(n_max+1):
            for k in range(-j,j+1):
                if max_level == 1:
                    distant_par_f[i] = distant_par_f[i] + coulomb * quantity_par_f[i] * localcoefficient1[flag[max_level][i][0]][flag[max_level][i][1]][flag[max_level][i][2]][j][k+n_max] \
                        * re_solid_harm(k, j, temp_point2_f[2], temp_point2_f[1], temp_point2_f[0])
                elif max_level == 2:
                    distant_par_f[i] = distant_par_f[i] + coulomb * quantity_par_f[i] * localcoefficient2[flag[max_level][i][0]][flag[max_level][i][1]][flag[max_level][i][2]][j][k+n_max] \
                        * re_solid_harm(k, j, temp_point2_f[2], temp_point2_f[1], temp_point2_f[0])

                    if j-1>=0:
                        power_par_f[i][0] = power_par_f[i][0] + coulomb * quantity_par_f[i] * localcoefficient2[flag[max_level][i][0]][flag[max_level][i][1]][flag[max_level][i][2]][j][k+n_max] \
                            * (re_solid_harm(k+1, j-1, temp_point2_f[2], temp_point2_f[1], temp_point2_f[0]) - re_solid_harm(k-1, j-1, temp_point2_f[2], temp_point2_f[1], temp_point2_f[0]))/2.0
                        
                        element = (re_solid_harm(k+1, j-1, temp_point2_f[2], temp_point2_f[1], temp_point2_f[0]) + re_solid_harm(k-1, j-1, temp_point2_f[2], temp_point2_f[1], temp_point2_f[0]))
                        
                        power_par_f[i][1] = power_par_f[i][1] + coulomb * quantity_par_f[i] * localcoefficient2[flag[max_level][i][0]][flag[max_level][i][1]][flag[max_level][i][2]][j][k+n_max] \
                            * complex(element.imag, -element.real)/2.0
                        
                        power_par_f[i][2] = power_par_f[i][2] + coulomb * quantity_par_f[i] * localcoefficient2[flag[max_level][i][0]][flag[max_level][i][1]][flag[max_level][i][2]][j][k+n_max] \
                            * re_solid_harm(k, j-1, temp_point2_f[2], temp_point2_f[1], temp_point2_f[0])
                        
                elif max_level == 3:
                    distant_par_f[i] = distant_par_f[i] + coulomb * quantity_par_f[i] * localcoefficient3[flag[max_level][i][0]][flag[max_level][i][1]][flag[max_level][i][2]][j][k+n_max] \
                        * re_solid_harm(k, j, temp_point2_f[2], temp_point2_f[1], temp_point2_f[0])

                    if j-1>=0:
                        power_par_f[i][0] = power_par_f[i][0] + coulomb * quantity_par_f[i] * localcoefficient3[flag[max_level][i][0]][flag[max_level][i][1]][flag[max_level][i][2]][j][k+n_max] \
                            * (re_solid_harm(k+1, j-1, temp_point2_f[2], temp_point2_f[1], temp_point2_f[0]) - re_solid_harm(k-1, j-1, temp_point2_f[2], temp_point2_f[1], temp_point2_f[0]))/2.0
                        
                        element = (re_solid_harm(k+1, j-1, temp_point2_f[2], temp_point2_f[1], temp_point2_f[0]) + re_solid_harm(k-1, j-1, temp_point2_f[2], temp_point2_f[1], temp_point2_f[0]))
                        
                        power_par_f[i][1] = power_par_f[i][1] + coulomb * quantity_par_f[i] * localcoefficient3[flag[max_level][i][0]][flag[max_level][i][1]][flag[max_level][i][2]][j][k+n_max] \
                            * complex(element.imag, -element.real)/2.0
                        
                        power_par_f[i][2] = power_par_f[i][2] + coulomb * quantity_par_f[i] * localcoefficient3[flag[max_level][i][0]][flag[max_level][i][1]][flag[max_level][i][2]][j][k+n_max] \
                            * re_solid_harm(k, j-1, temp_point2_f[2], temp_point2_f[1], temp_point2_f[0])
                        
                elif max_level == 4:
                    distant_par_f[i] = distant_par_f[i] + coulomb * quantity_par_f[i] * localcoefficient4[flag[max_level][i][0]][flag[max_level][i][1]][flag[max_level][i][2]][j][k+n_max] \
                        * re_solid_harm(k, j, temp_point2_f[2], temp_point2_f[1], temp_point2_f[0])
                    
               
            potential_f[i][j+1] = (distant_par_f[i].real) + near_par_f[i]

        temp_power_par_f[i][0] = temp_power_par_f[i][0] + power_par_f[i][0]
        temp_power_par_f[i][1] = temp_power_par_f[i][1] + power_par_f[i][1]
        temp_power_par_f[i][2] = temp_power_par_f[i][2] - power_par_f[i][2]

        total_power_f[1] = total_power_f[1] + math.sqrt(abs(temp_power_par_f[i][0])**2+abs(temp_power_par_f[i][1])**2+abs(temp_power_par_f[i][2])**2)

    #計測終了
    time_end2 = time.perf_counter()

    #sys.exit()

    #直接計算によるポテンシャル計算
    
    for i in range(number_par):
        for j in range(i+1,number_par):
            
            point_point_d[0] = orth_d[i][0] - orth_d[j][0]
            point_point_d[1] = orth_d[i][1] - orth_d[j][1]
            point_point_d[2] = orth_d[i][2] - orth_d[j][2]

            #距離
            distance_par_d[i][j] = math.sqrt(point_point_d[0]**2+point_point_d[1]**2+point_point_d[2]**2)
            distance_par_d[j][i] = distance_par_d[i][j]

            power_d[i][0] = power_d[i][0] + coulomb * quantity_par_d[i] * quantity_par_d[j] * (point_point_d[0]/distance_par_d[i][j]) / (distance_par_d[i][j]**2)
            power_d[i][1] = power_d[i][1] + coulomb * quantity_par_d[i] * quantity_par_d[j] * (point_point_d[1]/distance_par_d[i][j]) / (distance_par_d[i][j]**2)
            power_d[i][2] = power_d[i][2] + coulomb * quantity_par_d[i] * quantity_par_d[j] * (point_point_d[2]/distance_par_d[i][j]) / (distance_par_d[i][j]**2)
            
            power_d[j][0] = power_d[j][0] - coulomb * quantity_par_d[i] * quantity_par_d[j] * (point_point_d[0]/distance_par_d[i][j]) / (distance_par_d[i][j]**2)
            power_d[j][1] = power_d[j][1] - coulomb * quantity_par_d[i] * quantity_par_d[j] * (point_point_d[1]/distance_par_d[i][j]) / (distance_par_d[i][j]**2)
            power_d[j][2] = power_d[j][2] - coulomb * quantity_par_d[i] * quantity_par_d[j] * (point_point_d[2]/distance_par_d[i][j]) / (distance_par_d[i][j]**2)
            
            #極座標表示
            temp_point2_d = polar_convert(point_point_d[0],point_point_d[1],point_point_d[2])
            
            #直接ポテンシャル計算
            potential_d[i][0] = potential_d[i][0] + coulomb * quantity_par_d[i] * quantity_par_d[j] / temp_point2_d[0]
            potential_d[j][0] = potential_d[j][0] + coulomb * quantity_par_d[j] * quantity_par_d[i] / temp_point2_d[0]

        total_power_d[0] = total_power_d[0] + math.sqrt(power_d[i][0]**2+power_d[i][1]**2+power_d[i][2]**2)
        

    #===========================更新============================
    for i in range(number_par):
        #加速度
        acceleration_f[i][0] = (temp_power_par_f[i][0] / mass_par_f[i]).real
        acceleration_f[i][1] = (temp_power_par_f[i][1] / mass_par_f[i]).real
        acceleration_f[i][2] = (temp_power_par_f[i][2] / mass_par_f[i]).real

        acceleration_d[i][0] = (power_d[i][0] / mass_par_d[i]).real
        acceleration_d[i][1] = (power_d[i][1] / mass_par_d[i]).real
        acceleration_d[i][2] = (power_d[i][2] / mass_par_d[i]).real

        #位置
        orth_f[i][0] = orth_f[i][0] + velocity_f[i][0] * t_step #+ (acceleration_f[i][0] * (t_step**2)) / 2.0
        orth_f[i][1] = orth_f[i][1] + velocity_f[i][1] * t_step #+ (acceleration_f[i][1] * (t_step**2)) / 2.0
        orth_f[i][2] = orth_f[i][2] + velocity_f[i][2] * t_step #+ (acceleration_f[i][2] * (t_step**2)) / 2.0

        orth_d[i][0] = orth_d[i][0] + velocity_d[i][0] * t_step #+ (acceleration_d[i][0] * (t_step**2)) / 2.0
        orth_d[i][1] = orth_d[i][1] + velocity_d[i][1] * t_step #+ (acceleration_d[i][1] * (t_step**2)) / 2.0
        orth_d[i][2] = orth_d[i][2] + velocity_d[i][2] * t_step #+ (acceleration_d[i][2] * (t_step**2)) / 2.0
                                 
        #速度
        velocity_f[i][0] = velocity_f[i][0] + acceleration_f[i][0] * t_step
        velocity_f[i][1] = velocity_f[i][1] + acceleration_f[i][1] * t_step
        velocity_f[i][2] = velocity_f[i][2] + acceleration_f[i][2] * t_step

        velocity_d[i][0] = velocity_d[i][0] + acceleration_d[i][0] * t_step
        velocity_d[i][1] = velocity_d[i][1] + acceleration_d[i][1] * t_step
        velocity_d[i][2] = velocity_d[i][2] + acceleration_d[i][2] * t_step


        polar_f[i][0],polar_f[i][1],polar_f[i][2] = polar_convert(orth_f[i][0],orth_f[i][1],orth_f[i][2])

        polar_d[i][0],polar_d[i][1],polar_d[i][2] = polar_convert(orth_d[i][0],orth_d[i][1],orth_d[i][2])

        # 反射境界条件の適用
        if orth_f[i][0] < -simu_size/2.0:
            orth_f[i][0] = -simu_size/2.0 - (orth_f[i][0] + simu_size/2.0)
            velocity_f[i][0] = -velocity_f[i][0]
        elif orth_f[i][0] > simu_size/2.0:
            orth_f[i][0] = simu_size/2.0 - (orth_f[i][0] - simu_size/2.0)
            velocity_f[i][0] = -velocity_f[i][0]

        if orth_f[i][1] < -simu_size/2.0:
            orth_f[i][1] = -simu_size/2.0 - (orth_f[i][1] + simu_size/2.0)
            velocity_f[i][1] = -velocity_f[i][1]
        elif orth_f[i][1] > simu_size/2.0:
            orth_f[i][1] = simu_size/2.0 - (orth_f[i][1] - simu_size/2.0)
            velocity_f[i][1] = -velocity_f[i][1]

        if orth_f[i][2] < -simu_size/2.0:
            orth_f[i][2] = -simu_size/2.0 - (orth_f[i][2] + simu_size/2.0)
            velocity_f[i][2] = -velocity_f[i][2]
        elif orth_f[i][2] > simu_size/2.0:
            orth_f[i][2] = simu_size/2.0 - (orth_f[i][2] - simu_size/2.0)
            velocity_f[i][2] = -velocity_f[i][2]

        # 反射境界条件の適用
        if orth_d[i][0] < -simu_size/2.0:
            orth_d[i][0] = -simu_size/2.0 - (orth_d[i][0] + simu_size/2.0)
            velocity_d[i][0] = -velocity_d[i][0]
        elif orth_d[i][0] > simu_size/2.0:
            orth_d[i][0] = simu_size/2.0 - (orth_d[i][0] - simu_size/2.0)
            velocity_d[i][0] = -velocity_d[i][0]

        if orth_d[i][1] < -simu_size/2.0:
            orth_d[i][1] = -simu_size/2.0 - (orth_d[i][1] + simu_size/2.0)
            velocity_d[i][1] = -velocity_d[i][1]
        elif orth_d[i][1] > simu_size/2.0:
            orth_d[i][1] = simu_size/2.0 - (orth_d[i][1] - simu_size/2.0)
            velocity_d[i][1] = -velocity_d[i][1]

        if orth_d[i][2] < -simu_size/2.0:
            orth_d[i][2] = -simu_size/2.0 - (orth_d[i][2] + simu_size/2.0)
            velocity_d[i][2] = -velocity_d[i][2]
        elif orth_d[i][2] > simu_size/2.0:
            orth_d[i][2] = simu_size/2.0 - (orth_D[i][2] - simu_size/2.0)
            velocity_d[i][2] = -velocity_d[i][2]


    
    for i in range(number_par):
        re_point_point[i][0] = orth_d[i][0] - orth_f[i][0]
        re_point_point[i][1] = orth_d[i][1] - orth_f[i][1]
        re_point_point[i][2] = orth_d[i][2] - orth_f[i][2]

        #距離
        ave_distance[t] = ave_distance[t] + math.sqrt(re_point_point[i][0]**2+re_point_point[i][1]**2+re_point_point[i][2]**2)

    ave_distance[t] = ave_distance[t]/number_par

    print("~~~~~~~~~~~~~~~~~~[Step"+str(t+1)+"]~~~~~~~~~~~~~~~~~~")
    """
    print("fmm")
    print(orth_f[0][0])
    print(acceleration_f[0][0])
    print(velocity_f[0][0])
    print(temp_power_par_f[0][0].real)
    print(total_power_f[1])
    print("direct")
    print(orth_d[0][0])
    print(acceleration_d[0][0])
    print(velocity_d[0][0])
    print(power_d[0][0])
    print(total_power_d[0])
    print("dif")
    print(abs(total_power_f[1]-total_power_d[0])/total_power_d[0])
    print(dif_distance)
    """
    print("distance")
    print(ave_distance[t])



print("~~~~~~~~~~~~~~~~~~[fin]~~~~~~~~~~~~~~~~~~")
print("result_orth_distance")
for t in range(number_step):
    print(ave_distance[t])


    
        
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[繰り返し終了]>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
if number_step == 1:
    
    print("")
    #===========================1つの粒子に注目し平均とる============================
    """
    print("==================================================")
    print("==                    誤差平均                  ==")
    print("==================================================")
    print("[多重極展開を用いて計算(1項目,2項目,3項目まで)]")
    print("誤差の平均(1項目まで) = "+str(total_error[1]/attempt_number))
    print("誤差の平均(2項目まで) = "+str(total_error[2]/attempt_number))
    print("誤差の平均(3項目まで) = "+str(total_error[3]/attempt_number))
    print("--------------------------------------------------")
    print("[球面調和関数を用いた多重極展開を用いて計算(n=0,1,2,3の時)]")
    print("誤差の平均(n=0) = "+str(total_error[4]/attempt_number))
    print("誤差の平均(n=1) = "+str(total_error[5]/attempt_number))
    print("誤差の平均(n=2) = "+str(total_error[6]/attempt_number))
    print("誤差の平均(n=3) = "+str(total_error[7]/attempt_number))
    print("--------------------------------------------------")
    print("[多重極展開(n=3)+局所展開を用いて計算]")
    print("誤差の平均 = "+str(total_error[8]/attempt_number))
    """
    
    print("=========================[結果]=========================")
    print("[ポテンシャル]")
    print("精度パラメータ = "+str(total_parameter[n_max]/(number_par*number_step)))
    
    print("")
    
    print("[受ける力]")
    """
    print("==================================================")
    print("power1 = "+str(power[0]))
    print("power2 = "+str(temp_power_par[0]))
    print("--------------------------------------------------")
    print("power1 = "+str(power[1]))
    print("power2 = "+str(temp_power_par[1]))
    print("==================================================")
    """
    print("直接計算した力の平均  = "+str(abs(total_power[0])/(number_par*number_step)))
    print("FMMで計算した力の平均 = "+str(abs(total_power[1])/(number_par*number_step)))
    print("精度パラメータ = "+str(abs(total_power[0]-total_power[1])/total_power[0]))
    
    print("")
    
    print("[処理時間]")
    print("直接計算の処理時間 = "+str(time_end1-time_start1))
    print("FMMの処理時間 = "+str(time_end2-time_start2))
    print("FMM/直接計算 = "+str((time_end2-time_start2)/(time_end1-time_start1)))
    print("========================================================")
    
    print("")


