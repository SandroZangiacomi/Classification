# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 20:57:03 2020

@author: Zangiacomi Sandro
"""
import random
from PIL import Image
from numpy import asarray
import numpy as np
import time 

#PREPROCESSING#

##Mean of coordinate of a list of point##
def mean(points):
    a=np.mean(points, axis=0)
    return a.tolist()      

##Distance of two point##
    
def minkowski_distance(a,b):  #p=5
    summ=0
    for i in range (0,len(a)):
        summ+=(abs(a[i]-b[i]))**5
    return (summ)**(1/5)
        
    
def Manhattan_distance(a,b) :
    
    summ=0
    for i in range (0,len(a)):
        summ+=abs(a[i]-b[i])
    return summ
    
def Euclidian_distance(a,b) :  #a and b are two points with their coordinates as a list
    a=asarray(a)
    b=asarray(b)
    c=np.linalg.norm(a-b)
    return c

###Creation of random centroids###

def centroids(k):
    Centroids=[]
    for i in range (0,k):
        Centroids.append([random.randint(0,255),random.randint(0,255),random.randint(0,255)])
    return Centroids    

def centroids_medoids(k,data_set):
    Centroids=[]
    for i in range (0,k):
        Centroids.append(data_set[random.randint(0,len(data_set))])
    return Centroids  
    
##Creation of lists for clusters##

def cluster(k):
    Clusters=[]
    for i in range (0,k):
        Clusters.append([])
        
    return Clusters

def association_to_cluster(data_set,Centroids):
    New_asso=[]
    Clusters=cluster(len(Centroids))
    for pixels in data_set:
        Distance_List=[]
        for centroid in Centroids:
            
            Distance_List.append(Euclidian_distance(pixels,centroid))
        
        index=Distance_List.index(min(Distance_List))
        
        Clusters[index].append(pixels)
        
        New_asso.append(pixels+[index])
    
    Clear_cluster=[] 
    for clusters in   Clusters:
        if clusters!=[]:
            Clear_cluster.append(clusters)
            
            
    return Clear_cluster,New_asso 
  
def centroids_adjustment(Clear_cluster):
    New_centroid=[]
    for i in Clear_cluster:
        New_centroid.append(mean(i))
      
    return New_centroid

        
def k_means(data_set,k):
    
    Centroids0=centroids(k)
    Clear_cluster, New_asso =association_to_cluster(data_set,Centroids0)
   
    New_centroid= centroids_adjustment(Clear_cluster)
    Centroids0=centroids(len(New_centroid))                #needed to help the while loop to start
    T1=time.time()
    while np.linalg.norm(np.asarray(New_centroid)-np.asarray(Centroids0))> 10:
        print('loop')
        Centroids0=New_centroid
        Clear_cluster, New_asso= association_to_cluster(data_set,Centroids0)
        New_centroid=centroids_adjustment(Clear_cluster)
    T2=time.time()    
        
    Clustered_pixels=[]
    
       
    for pixels in New_asso :
        
        Clustered_pixels.append(New_centroid[pixels[-1]])
        
    arr = asarray(Clustered_pixels).astype('uint8')
     
    
  
    print(T2-T1)
    return [arr, Clear_cluster, New_centroid]

def medoid(New_centroid,Clear_cluster):
     medoids=[]
     for i in range (0,len(New_centroid)) :
        distance=[]
        for points in Clear_cluster[i]:
            
            distance.append(Euclidian_distance(points,New_centroid[i]))
        
        index=distance.index(min(distance))
       
        
        medoids.append(Clear_cluster[i][index])
     return medoids
        
    

def k_medoids(data_set,k):
    Centroids0=centroids_medoids(k,data_set)
    Clear_cluster, New_asso =association_to_cluster(data_set,Centroids0)
   
    New_centroid= centroids_adjustment(Clear_cluster)
    medoids=medoid(New_centroid,Clear_cluster)
    Centroids0=centroids(len(medoids))                #needed to help the while loop to start
    T1=time.time()
    while np.linalg.norm(np.asarray(medoids)-np.asarray(Centroids0))> 10:
        print('loop')
        Centroids0=medoids
        Clear_cluster, New_asso= association_to_cluster(data_set,Centroids0)
        New_centroid=centroids_adjustment(Clear_cluster)
        medoids=medoid(New_centroid,Clear_cluster)
    
    T2=time.time() 
    print(T2-T1) 
    
    medoids=medoid(New_centroid,Clear_cluster)
    Clustered_pixels=[]   
    for pixels in New_asso :
        
        Clustered_pixels.append(medoids[pixels[-1]])
        
    arr = asarray(Clustered_pixels).astype('uint8')
    
    return [arr, Clear_cluster, New_centroid]


   
###########             MAIN                   ######################  
    
#######K-MEDOIDS #######
#BEACH  
  
   
image=Image.open('beach.bmp')
image.show()
data=np.array(image)
data_set=np.reshape(data,(len(data)*len(data[0]),3))
data_set=data_set.tolist()        
                                   
Clustered_matrix=np.reshape( k_medoids(data_set,20)[0],(len(data),len(data[0]),3))
pillow_image = Image.fromarray(Clustered_matrix)
pillow_image.show()       
#pillow_image=pillow_image.save('(1)beachclustered_20k_medoids.bmp')



Clustered_matrix0=np.reshape( k_medoids(data_set,4)[0],(len(data),len(data[0]),3))
pillow_image0 = Image.fromarray(Clustered_matrix0)
pillow_image0.show()       
#pillow_image0=pillow_image0.save('(2)beachclustered_4k_medoids.bmp')

Clustered_matrix0=np.reshape( k_medoids(data_set,15)[0],(len(data),len(data[0]),3))
pillow_image0 = Image.fromarray(Clustered_matrix0)
pillow_image0.show()       
pillow_image0=pillow_image0.save('(4)beachclustered_20k_medoids.bmp')

Clustered_matrix0=np.reshape( k_medoids(data_set,40)[0],(len(data),len(data[0]),3))
pillow_image0 = Image.fromarray(Clustered_matrix0)
pillow_image0.show()       
pillow_image0=pillow_image0.save('(5)beachclustered_40k_medoids.bmp')

#FOOTBALL    #################################################   


image1=Image.open('football.bmp')
image1.show()
data1=np.array(image1)
data_set1=np.reshape(data1,(len(data1)*len(data1[0]),3))
data_set1=data_set1.tolist()        

                                
Clustered_matrix1=np.reshape( k_medoids(data_set1,4)[0],(len(data1),len(data1[0]),3))
pillow_image1 = Image.fromarray(Clustered_matrix1)
pillow_image1.show()
#pillow_image1=pillow_image1.save('footballclustered_4k_medoids.bmp')

#ME   ########################################################


image2=Image.open('Sandro.jpg')
image2.show()
data2=np.array(image2)
data_set2=np.reshape(data2,(len(data2)*len(data2[0]),3))
data_set2=data_set2.tolist()        
                                   
Clustered_matrix2=np.reshape( k_medoids(data_set2,4)[0],(len(data2),len(data2[0]),3))
pillow_image2 = Image.fromarray(Clustered_matrix2)
pillow_image2.show()
#pillow_image2=pillow_image2.save('Sandroclustered_4k_medoids.jpg')

 
#####K-means#####
#BEACH 

      
image3=Image.open('beach.bmp')
image3.show()
data3=np.array(image3)
data_set3=np.reshape(data3,(len(data3)*len(data3[0]),3))
data_set3=data_set3.tolist()        
                                   
Clustered_matrix3=np.reshape( k_means(data_set3,20)[0],(len(data3),len(data3[0]),3))
pillow_image3 = Image.fromarray(Clustered_matrix3)
pillow_image3.show()       
#pillow_image3=pillow_image3.save('beachclustered_20k_means.bmp')



Clustered_matrix03=np.reshape( k_means(data_set3,4)[0],(len(data3),len(data3[0]),3))
pillow_image03 = Image.fromarray(Clustered_matrix03)
pillow_image03.show()       
#pillow_image03=pillow_image03.save('beachclustered_4k_means.bmp')

 

#FOOTBALL    #################################################

   
image13=Image.open('football.bmp')
image13.show()
data13=np.array(image13)
data_set13=np.reshape(data13,(len(data13)*len(data13[0]),3))
data_set13=data_set13.tolist()        
                                   
Clustered_matrix13=np.reshape( k_means(data_set13,4)[0],(len(data13),len(data13[0]),3))
pillow_image13 = Image.fromarray(Clustered_matrix13)
pillow_image13.show()
#pillow_image13=pillow_image13.save('footballclustered_4k_means.bmp')

 

#ME   ########################################################


image23=Image.open('Sandro.jpg')
image23.show()
data23=np.array(image23)
data_set23=np.reshape(data23,(len(data23)*len(data23[0]),3))
data_set23=data_set23.tolist()        
                                   
Clustered_matrix23=np.reshape( k_means(data_set23,4)[0],(len(data23),len(data23[0]),3))
pillow_image23 = Image.fromarray(Clustered_matrix23)
pillow_image23.show()
#pillow_image23=pillow_image23.save('Sandroclustered_4k_means.jpg')

 


