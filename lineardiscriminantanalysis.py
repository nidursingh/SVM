import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math
data=pd.read_csv('lda_data.csv')
X_train,X_test,Y_train,Y_test=train_test_split(data,data['Y'],test_size=0.35,random_state=0)

print(X_train.shape)
print(Y_train.shape)

print(X_test.shape)
print(Y_test.shape)


datamean=X_train.groupby(['Y'])['X'].mean()
print(datamean)

count=X_train.groupby(['Y'])['X'].count()
print(count)

Prob_0= count[0]/(count[0]+count[1])
Prob_1= count[1]/(count[0]+count[1])

print(Prob_0)
print(Prob_1)

variance= X_train.groupby(['Y'])['X'].var()
print(variance)
print('******************************************************************')  
X_val=X_test['X'].to_numpy()
Y_val=X_test['Y'].to_numpy()

fid1=open('X_test.txt','w')
for i in range(len(X_val)):
    #string = str(X_val[i] +'  ' + Y_val[i])
    fid1.write(str(X_val[i]))
    fid1.write('\t')
    fid1.write(str(Y_val[i]))
    fid1.write('\n')

fid1.close()
Y_predicted=np.zeros([len(Y_val)]) 
counter=0   
for val in (X_test['X']):
    #val=X_test[i][0]
#     print(val)
#     
#     print(np.log(Prob_0))
#     print(np.log(Prob_1))    
#     print('\n')
    
#     print( val*(datamean[0]/variance[0]))
#     print(((datamean[0])**2)/(2*variance[0]))
#     
#     print('\n')
#     
#     print( val*(datamean[1]/variance[1]))
#     print(((datamean[1])**2)/(2*variance[1]))
#     
#     print('\n')
#     
    discriminant_0=val*(datamean[0]/variance[0]) -  ((datamean[0])**2)/(2*variance[0]) + np.log(Prob_0)
    discriminant_1=val*(datamean[1]/variance[1]) -  ((datamean[1])**2)/(2*variance[1]) + np.log(Prob_1)
    
#    print(discriminant_0)
#    print(discriminant_1)
    
    if(discriminant_0 > discriminant_1 ):
        class_val=0
    else:
        class_val=1
    Y_predicted[counter]= class_val
    counter=counter+1  
   # print(class_val)      
print('accuracy score is')
print(accuracy_score(Y_val,Y_predicted))      
    