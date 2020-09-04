import math 
from PIL import Image
from pylab import *
import matplotlib.cm as cm
import scipy as sp
import random
from collections import defaultdict
import operator
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


math_path = r'/content/sub7.jpg'
math = cv2.imread(math_path)
plt.figure();plt.axis('off');plt.imshow(math);
Carregamos a imagem
img = cv2.imread('/content/sub7.jpg')

#Criamos uma imagem de máscara semelhante
mask = np.zeros(img.shape[:2],np.uint8)

#Fornecer parâmetros para a retirada do fundo
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (50,50,500,500)

#Número de iterações do algoritmo = 5
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

#Modificação da máscara e forma que todos os pixels de 0 e 2 pixels sejam colocados em 0 (ou seja, fundo) 
# e todos os pixels de 1 e 3 pixels sejam colocados em 1 (ou seja, pixels de primeiro plano). 

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

#Basta multiplicar pela imagem de entrada para obter a imagem segmentada.

img = img*mask2[:,:,np.newaxis]

cv2.waitKey(0)
cv2.imwrite('roi1.jpg', img)
plt.imshow(img),plt.colorbar(),plt.show()

image = cv2.imread("/content/roi1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# remodelar a imagem para uma matriz 2D de pixels e 3 valores de cor (RGB)
pixel_values = image.reshape((-1, 3))

# Converter para ponto flutuante
pixel_values = np.float32(pixel_values)
print(pixel_values.shape)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Definir o número de clusters
k = 3
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# converter de volta para valores de 8 bits
centers = np.uint8(centers)

# nivelar a matriz de rótulos
labels = labels.flatten()

# converter todos os pixels para a cor dos centróides
imagem_classificada = centers[labels.flatten()]

# remodelar de volta para a dimensão da imagem original
imagem_classificada = imagem_classificada.reshape(image.shape)
cv2.imwrite('imagem_classificada1.jpg', img)

# Mostrar imagem
plt.imshow(imagem_classificada)
plt.show()

im = Image.open('/content/roi1.jpg')
arr = np.asarray(im)

out = Image.open('/content/imagem_classificada1.jpg').convert('L')
arr_out = np.asarray(out)

rows,columns = np.shape(arr_out)
print ('background pixel level',arr_out[0][0])

#print '\nrows',rows,'columns',columns
ground_out = np.zeros((rows,columns))

#Convertendo imagem de entrada em imagem binária para avaliação

for i in range(rows):
  for j in range(columns):
    if arr_out[i][j] > 120:
      ground_out[i][j] = 0

    else:
      ground_out[i][j] = 1

plt.figure()
plt.imshow(ground_out, cmap="Greys_r")
plt.show()

shape = np.shape(arr)

rows = shape[0]
columns = shape[1]

#obtendo 6 pontos centróides aleatórios

r_points = [ random.randint(0, 255) for i in range(6) ]
g_points = [ random.randint(0, 255) for i in range(6) ]
b_points = [ random.randint(0, 255) for i in range(6) ]

grey_l = defaultdict(list)

# Níveis de escala de cinza correspondendo a 6 clusters

grey1 = 40
grey2 = 80
grey3 = 120
grey4 = 160
grey5 = 200
grey6 = 240

grey_l[0] = 40
grey_l[1] = 80
grey_l[2] = 120
grey_l[3] = 160
grey_l[4] = 200
grey_l[5] = 240


g = defaultdict(list)

g2 = []
g3 = []
g4 = []
g5 = []
g6 = []

end = np.zeros((rows,columns))
zavg = [0,0,0]  

#calculando centroides médios após cada iteração

def find_centroids(g):
  red_cent_list = []
  blue_cent_list = []
  green_cent_list = []
  #print '0 shape',np.shape(g)
  for i in range(0,6):
    array = np.matrix(g[i])
    avg = np.mean(array,0)
    #print '\naverage values',avg
    pavg = np.ravel(avg)
    #print '2 shape', np.shape(pavg)
    #print 'pavg', pavg
    if not len(pavg):
      red_cent_list.append(zavg[0]) 
      blue_cent_list.append(zavg[1]) 
      green_cent_list.append(zavg[2])
    else:
      red_cent_list.append(pavg[0]) 
      blue_cent_list.append(pavg[1]) 
      green_cent_list.append(pavg[2])
  return[red_cent_list,blue_cent_list,green_cent_list] 
    
      
#Calculando 10 iterações para obter centróides convergentes de seis clusters' 
  
for it in range(0,10):
  print ('\niteration',it)
  g= defaultdict(list)
  for r in range(rows):
    for c in range(columns):
      img = arr[r][c]
      #print '\nimg', img
      red = img[0]
      green = img[1]
      blue =  img[2]
      #print '\n red',red,'blue',blue,'green',green

      distance_list = []
      for k in range(0,6):
        #computando a distância absoluta de cada cluster e atribuindo-o a um determinado cluster com base na distância        
      

        distance = math.sqrt(((int(r_points[k])- red)**2)+((int(g_points[k]) - green)**2)+((int(b_points[k])-blue)**2))
      
        distance_list.append(distance)
    
      index, value = min(enumerate(distance_list), key=operator.itemgetter(1))
      end[r][c] = grey_l[index]
      
      g[index].append([red,blue,green])
  centroids= find_centroids(g)

  r_points = []
  b_points = []
  g_points = []
  r_points = centroids[0]
  b_points = centroids[1]
  g_points = centroids[2]


#Por observação, sabemos que o pixel da imagem [0,0] é parte do fundo. Isso é usado para avaliação 

result = np.zeros((rows,columns))
ref_val = end[0][0]
#print '\nref_val',ref_val
for i in range(rows):
  for j in range(columns):
    if end[i][j] ==  ref_val:
      result[i][j] = 1

    else:
      result[i][j] = 0


''' **********************************Cálculo de Tpr, Fpr, F-Score  ***************************************************'''

tp = 0
tn = 0
fn = 0
fp = 0

for i in range(rows):
  for j in range(columns):
    if ground_out[i][j] == 1 and result[i][j] == 1:
      tp = tp + 1
    if ground_out[i][j] == 0 and result[i][j] == 0:
      tn = tn + 1
    if ground_out[i][j] == 1 and result[i][j] == 0:
      fn = fn + 1
    if ground_out[i][j] == 0 and result[i][j] == 1:
      fp = fp + 1


import math 
from PIL import Image
from pylab import *
import matplotlib.cm as cm
import scipy as sp
import random
from collections import defaultdict
import operator
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


math_path = r'/content/sub7.jpg'
math = cv2.imread(math_path)
plt.figure();plt.axis('off');plt.imshow(math);
Carregamos a imagem
img = cv2.imread('/content/sub7.jpg')

#Criamos uma imagem de máscara semelhante
mask = np.zeros(img.shape[:2],np.uint8)

#Fornecer parâmetros para a retirada do fundo
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (50,50,500,500)

#Número de iterações do algoritmo = 5
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

#Modificação da máscara e forma que todos os pixels de 0 e 2 pixels sejam colocados em 0 (ou seja, fundo) 
# e todos os pixels de 1 e 3 pixels sejam colocados em 1 (ou seja, pixels de primeiro plano). 

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

#Basta multiplicar pela imagem de entrada para obter a imagem segmentada.

img = img*mask2[:,:,np.newaxis]

cv2.waitKey(0)
cv2.imwrite('roi1.jpg', img)
plt.imshow(img),plt.colorbar(),plt.show()

image = cv2.imread("/content/roi1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# remodelar a imagem para uma matriz 2D de pixels e 3 valores de cor (RGB)
pixel_values = image.reshape((-1, 3))

# Converter para ponto flutuante
pixel_values = np.float32(pixel_values)
print(pixel_values.shape)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Definir o número de clusters
k = 3
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# converter de volta para valores de 8 bits
centers = np.uint8(centers)

# nivelar a matriz de rótulos
labels = labels.flatten()

# converter todos os pixels para a cor dos centróides
imagem_classificada = centers[labels.flatten()]

# remodelar de volta para a dimensão da imagem original
imagem_classificada = imagem_classificada.reshape(image.shape)
cv2.imwrite('imagem_classificada1.jpg', img)

# Mostrar imagem
plt.imshow(imagem_classificada)
plt.show()

im = Image.open('/content/roi1.jpg')
arr = np.asarray(im)

out = Image.open('/content/imagem_classificada1.jpg').convert('L')
arr_out = np.asarray(out)

rows,columns = np.shape(arr_out)
print ('background pixel level',arr_out[0][0])

#print '\nrows',rows,'columns',columns
ground_out = np.zeros((rows,columns))

#Convertendo imagem de entrada em imagem binária para avaliação

for i in range(rows):
  for j in range(columns):
    if arr_out[i][j] > 120:
      ground_out[i][j] = 0

    else:
      ground_out[i][j] = 1

plt.figure()
plt.imshow(ground_out, cmap="Greys_r")
plt.show()

shape = np.shape(arr)

rows = shape[0]
columns = shape[1]

#obtendo 6 pontos centróides aleatórios

r_points = [ random.randint(0, 255) for i in range(6) ]
g_points = [ random.randint(0, 255) for i in range(6) ]
b_points = [ random.randint(0, 255) for i in range(6) ]

grey_l = defaultdict(list)

# Níveis de escala de cinza correspondendo a 6 clusters

grey1 = 40
grey2 = 80
grey3 = 120
grey4 = 160
grey5 = 200
grey6 = 240

grey_l[0] = 40
grey_l[1] = 80
grey_l[2] = 120
grey_l[3] = 160
grey_l[4] = 200
grey_l[5] = 240


g = defaultdict(list)

g2 = []
g3 = []
g4 = []
g5 = []
g6 = []

end = np.zeros((rows,columns))
zavg = [0,0,0]  

#calculando centroides médios após cada iteração

def find_centroids(g):
  red_cent_list = []
  blue_cent_list = []
  green_cent_list = []
  #print '0 shape',np.shape(g)
  for i in range(0,6):
    array = np.matrix(g[i])
    avg = np.mean(array,0)
    #print '\naverage values',avg
    pavg = np.ravel(avg)
    #print '2 shape', np.shape(pavg)
    #print 'pavg', pavg
    if not len(pavg):
      red_cent_list.append(zavg[0]) 
      blue_cent_list.append(zavg[1]) 
      green_cent_list.append(zavg[2])
    else:
      red_cent_list.append(pavg[0]) 
      blue_cent_list.append(pavg[1]) 
      green_cent_list.append(pavg[2])
  return[red_cent_list,blue_cent_list,green_cent_list] 
    
      
#Calculando 10 iterações para obter centróides convergentes de seis clusters' 
  
for it in range(0,10):
  print ('\niteration',it)
  g= defaultdict(list)
  for r in range(rows):
    for c in range(columns):
      img = arr[r][c]
      #print '\nimg', img
      red = img[0]
      green = img[1]
      blue =  img[2]
      #print '\n red',red,'blue',blue,'green',green

      distance_list = []
      for k in range(0,6):
        #computando a distância absoluta de cada cluster e atribuindo-o a um determinado cluster com base na distância        
      

        distance = math.sqrt(((int(r_points[k])- red)**2)+((int(g_points[k]) - green)**2)+((int(b_points[k])-blue)**2))
      
        distance_list.append(distance)
    
      index, value = min(enumerate(distance_list), key=operator.itemgetter(1))
      end[r][c] = grey_l[index]
      
      g[index].append([red,blue,green])
  centroids= find_centroids(g)

  r_points = []
  b_points = []
  g_points = []
  r_points = centroids[0]
  b_points = centroids[1]
  g_points = centroids[2]


#Por observação, sabemos que o pixel da imagem [0,0] é parte do fundo. Isso é usado para avaliação 

result = np.zeros((rows,columns))
ref_val = end[0][0]
#print '\nref_val',ref_val
for i in range(rows):
  for j in range(columns):
    if end[i][j] ==  ref_val:
      result[i][j] = 1

    else:
      result[i][j] = 0


''' **********************************Cálculo de Tpr, Fpr, F-Score  ***************************************************'''

tp = 0
tn = 0
fn = 0
fp = 0

for i in range(rows):
  for j in range(columns):
    if ground_out[i][j] == 1 and result[i][j] == 1:
      tp = tp + 1
    if ground_out[i][j] == 0 and result[i][j] == 0:
      tn = tn + 1
    if ground_out[i][j] == 1 and result[i][j] == 0:
      fn = fn + 1
    if ground_out[i][j] == 0 and result[i][j] == 1:
      fp = fp + 1

import math 
from PIL import Image
from pylab import *
import matplotlib.cm as cm
import scipy as sp
import random
from collections import defaultdict
import operator
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


math_path = r'/content/sub7.jpg'
math = cv2.imread(math_path)
plt.figure();plt.axis('off');plt.imshow(math);
Carregamos a imagem
img = cv2.imread('/content/sub7.jpg')

#Criamos uma imagem de máscara semelhante
mask = np.zeros(img.shape[:2],np.uint8)

#Fornecer parâmetros para a retirada do fundo
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (50,50,500,500)

#Número de iterações do algoritmo = 5
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

#Modificação da máscara e forma que todos os pixels de 0 e 2 pixels sejam colocados em 0 (ou seja, fundo) 
# e todos os pixels de 1 e 3 pixels sejam colocados em 1 (ou seja, pixels de primeiro plano). 

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

#Basta multiplicar pela imagem de entrada para obter a imagem segmentada.

img = img*mask2[:,:,np.newaxis]

cv2.waitKey(0)
cv2.imwrite('roi1.jpg', img)
plt.imshow(img),plt.colorbar(),plt.show()

image = cv2.imread("/content/roi1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# remodelar a imagem para uma matriz 2D de pixels e 3 valores de cor (RGB)
pixel_values = image.reshape((-1, 3))

# Converter para ponto flutuante
pixel_values = np.float32(pixel_values)
print(pixel_values.shape)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Definir o número de clusters
k = 3
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# converter de volta para valores de 8 bits
centers = np.uint8(centers)

# nivelar a matriz de rótulos
labels = labels.flatten()

# converter todos os pixels para a cor dos centróides
imagem_classificada = centers[labels.flatten()]

# remodelar de volta para a dimensão da imagem original
imagem_classificada = imagem_classificada.reshape(image.shape)
cv2.imwrite('imagem_classificada1.jpg', img)

# Mostrar imagem
plt.imshow(imagem_classificada)
plt.show()

im = Image.open('/content/roi1.jpg')
arr = np.asarray(im)

out = Image.open('/content/imagem_classificada1.jpg').convert('L')
arr_out = np.asarray(out)

rows,columns = np.shape(arr_out)
print ('background pixel level',arr_out[0][0])

#print '\nrows',rows,'columns',columns
ground_out = np.zeros((rows,columns))

#Convertendo imagem de entrada em imagem binária para avaliação

for i in range(rows):
  for j in range(columns):
    if arr_out[i][j] > 120:
      ground_out[i][j] = 0

    else:
      ground_out[i][j] = 1

plt.figure()
plt.imshow(ground_out, cmap="Greys_r")
plt.show()

shape = np.shape(arr)

rows = shape[0]
columns = shape[1]

#obtendo 6 pontos centróides aleatórios

r_points = [ random.randint(0, 255) for i in range(6) ]
g_points = [ random.randint(0, 255) for i in range(6) ]
b_points = [ random.randint(0, 255) for i in range(6) ]

grey_l = defaultdict(list)

# Níveis de escala de cinza correspondendo a 6 clusters

grey1 = 50
grey2 = 80
grey3 = 120
grey4 = 170
grey5 = 200
grey6 = 240

grey_l[0] = 50
grey_l[1] = 80
grey_l[2] = 120
grey_l[3] = 170
grey_l[4] = 200
grey_l[5] = 240


g = defaultdict(list)

g2 = []
g3 = []
g4 = []
g5 = []
g6 = []

end = np.zeros((rows,columns))
zavg = [0,0,0]  

#calculando centroides médios após cada iteração

def find_centroids(g):
  red_cent_list = []
  blue_cent_list = []
  green_cent_list = []
  #print '0 shape',np.shape(g)
  for i in range(0,6):
    array = np.matrix(g[i])
    avg = np.mean(array,0)
    #print '\naverage values',avg
    pavg = np.ravel(avg)
    #print '2 shape', np.shape(pavg)
    #print 'pavg', pavg
    if not len(pavg):
      red_cent_list.append(zavg[0]) 
      blue_cent_list.append(zavg[1]) 
      green_cent_list.append(zavg[2])
    else:
      red_cent_list.append(pavg[0]) 
      blue_cent_list.append(pavg[1]) 
      green_cent_list.append(pavg[2])
  return[red_cent_list,blue_cent_list,green_cent_list] 
    
      
#Calculando 10 iterações para obter centróides convergentes de seis clusters' 
  
for it in range(0,10):
  print ('\niteration',it)
  g= defaultdict(list)
  for r in range(rows):
    for c in range(columns):
      img = arr[r][c]
      #print '\nimg', img
      red = img[0]
      green = img[1]
      blue =  img[2]
      #print '\n red',red,'blue',blue,'green',green

      distance_list = []
      for k in range(0,6):
        #computando a distância absoluta de cada cluster e atribuindo-o a um determinado cluster com base na distância        
      

        distance = math.sqrt(((int(r_points[k])- red)**2)+((int(g_points[k]) - green)**2)+((int(b_points[k])-blue)**2))
      
        distance_list.append(distance)
    
      index, value = min(enumerate(distance_list), key=operator.itemgetter(1))
      end[r][c] = grey_l[index]
      
      g[index].append([red,blue,green])
  centroids= find_centroids(g)

  r_points = []
  b_points = []
  g_points = []
  r_points = centroids[0]
  b_points = centroids[1]
  g_points = centroids[2]


#Por observação, sabemos que o pixel da imagem [0,0] é parte do fundo. Isso é usado para avaliação 

result = np.zeros((rows,columns))
ref_val = end[0][0]
#print '\nref_val',ref_val
for i in range(rows):
  for j in range(columns):
    if end[i][j] ==  ref_val:
      result[i][j] = 1

    else:
      result[i][j] = 0


''' **********************************Cálculo de Tpr, Fpr, F-Score  ***************************************************'''

tp = 0
tn = 0
fn = 0
fp = 0

for i in range(rows):
  for j in range(columns):
    if ground_out[i][j] == 1 and result[i][j] == 1:
      tp = tp + 1
    if ground_out[i][j] == 0 and result[i][j] == 0:
      tn = tn + 1
    if ground_out[i][j] == 1 and result[i][j] == 0:
      fn = fn + 1
    if ground_out[i][j] == 0 and result[i][j] == 1:
      fp = fp + 1


print ('\n************Calculo de Tpr, Fpr, F-Score********************')

#TP rate = TP/TP+FN
tpr= float(tp)/(tp+fn)
print ("\nTPR is:",tpr)


