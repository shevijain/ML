
# coding: utf-8

# In[1]:
import cv2
import os
import numpy as np
from tqdm import tqdm
#from sklearn.decomposition import PCA
import math
#import sklearn


# In[2]:


DIR='/Users/shevijain/Desktop/UTA_Classes/Sem2/ML/LastProject/Training'
NEW='/Users/shevijain/Desktop/UTA_Classes/Sem2/ML/LastProject/Testing'
items=['Apple','Banana','Orange','Pineapple','Rasberry']
training_data=[]


# In[3]:


def process_images(folder,directory,training_data):
    
    Dir = os.path.join(directory,folder)
    print(Dir)
    for filename in tqdm(os.listdir(Dir)):
        #print(filename)
        path = os.path.join(Dir, filename)
        
        #print(filename,path)
        if filename.endswith('.DS_Store'):
            continue
        else:
            img = cv2.imread(path,0)
            img = cv2.resize(img, (32, 32))
            #print(img)
            training_data.append(img)
    return img,training_data


# In[4]:


for i in items:
    img,training_data=process_images(i,DIR,training_data)



# In[17]:


d1=[]
for i in range(len(training_data)):
    d1.append(training_data[i].flatten())
print('D1',np.array(d1).shape)

'''
pca = PCA(15)
pca.fit(d1)
print('inbuilt components',pca.components_.shape)
print(pca.explained_variance_)
# transform data
B = pca.transform(d1)
print('after transform',B.shape)
'''
# In[18]:


classes=[]
np.array(d1).shape
for i in range(len(training_data)):
    if(i<100):
        classes.append(0)
    elif(i>=100 and i<200):
        classes.append(1)
    elif(i>=200 and i<300):
        classes.append(2)
    elif(i>=300 and i<400):
        classes.append(3)
    else:
        classes.append(4)


# In[19]:





# In[96]:


d1=np.array(d1)
#print(d1.shape[0])
centered=[]


#centering the input data
for i in range(d1.shape[0]):
    mean_col=d1[i].mean()
    centered.append(d1[i]-mean_col)

centered=np.array(centered)
#print('Centered',centered)
#print(d1)
print('centered',centered.shape)
#mean_train = d1.mean()
#d1=d1-mean_train
#print(d1)


# In[40]:

#calculating the covariance
cov=np.cov(centered.T)
print('inbuilt cov',cov.shape)
#cov=np.dot(centered.T, centered)
#print('cov',cov.shape)

#caluculating eigen values and vectors
eigenvalues, eigenvectors = np.linalg.eigh(cov)
print('eigenvector',eigenvectors.shape)
'''
tot=sum(eigenvalues)
var_exp = [(i/tot)*100 for i in sorted(eigenvalues,reverse=True)]
print(tot)
#print(sorted(eigenvalues,reverse=True) ) 
tot1=0
ss=sorted(eigenvalues,reverse=True)
#print(ss)
for i in range(25):
    tot1+=ss[i]
print(tot1/tot)
cum_var_exp = np.cumsum(var_exp)
exp_var_percentage = 0.86
'''


# In[5]:

#soritng in reverse
eigen_pairs=[(eigenvalues[i],eigenvectors[:,i])for i in range(len(eigenvalues))]
print(len(eigen_pairs))

eigen_pairs.sort()
eigen_pairs.reverse()

#getting principla components
features=[]
for i in range(20):
    features.append(eigen_pairs[i][1])
features=np.array(features)
print('feature',features.T.shape)

# In[53]:projecton matrix from feature matrix
projection_train=np.dot(centered,features.T)
print(projection_train.shape)


# In[16]:
testing_data=[]
for i in items:
    img1,testing_data=process_images(i,NEW,testing_data)


# In[17]:

d2=[]
for i in range(len(testing_data)):
    d2.append(testing_data[i].flatten())
print('D2',np.array(d2).shape)
d2=np.array(d2)

classes_test=[]
for i in range(len(testing_data)):
    if(i<20):
        classes_test.append(0)
    elif(i>=20 and i<30):
        classes_test.append(1)
    elif(i>=30 and i<40):
        classes_test.append(2)
    elif(i>=40 and i<50):
        classes_test.append(3)
    else:
        classes_test.append(4)



centered_test=[]
for i in range(d2.shape[0]):
    mean_col=d2[i].mean()
    centered_test.append(d2[i]-mean_col)
    
centered_test=np.array(centered_test)
#print('Centered',centered)
#print(d1)
print('centered test',centered_test.shape)

projection_test=np.dot(centered_test,features.T)
print('test',projection_test.shape)
#dist=sklearn.metrics.pairwise.euclidean_distances(projection_train,projection_test)
#dist = np.linalg.norm(projection_train-projection_test)
#print(dist

value=[]
classes_set=set()


# In[6]:


def calculate_euclidean(v1,v2):
    for i in range(projection_test.shape[1]):
        #distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
        dist=0
        dist+=(v1[i]-v2[i])**2
        dist=math.sqrt(dist)
    #dist=dist.sort()
    return dist


# In[ ]:KNN algorithm
final_class=[]
for i in range(projection_test.shape[0]):
    for j in range(projection_train.shape[0]):
        dist=calculate_euclidean(projection_test[i],projection_train[j])
        #dist = math.sqrt(sum([(a - b) ** 2 for a, b in zip(projection_test[i], projection_train[j])]))
        value.append((dist,classes[j]))
        value.sort()
        classes_set.add(classes[j])
    #print(value)
    class2=[]
    
    counts={}
    #choosing K nearest neighbor
    for k in range(7):
        class2.append(value[k][1])
    #print(class2)
    for word in class2:
        if word in counts:
            counts[word]+=1
        else:
            counts[word]=1
    
        final_class.append((counts[word],word))     #counting the no. of occurences of particular class

    final_class.sort(reverse=True)



# In[ ]:projection test matrix
for i in range(len(final_class)):
    print((final_class[i][1]))
print(projection_test.shape[0])


# In[ ]:accuracy
cout=0
for i in range(len(classes_test)):
    if(classes_test[i]==final_class[i][1]):
        cout+=1
print('Accuracy',(cout/len(classes_test))*100)

