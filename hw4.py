from __future__ import division
import numpy as np
import sys
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Jae Chang 2019 April 16th
#For ColumbiaX edX HW4
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

#this data is fed automatically by edX checker
#a csv file. each row contains 3 values: user index, object index and rating
train_data = np.genfromtxt(sys.argv[1], delimiter = ",")

#what edX wants initial paramters to be
lam = 2
sigma2 = 0.1
d = 5
iterations = 50

# Implement function here
def PMF(train_data):
    #Function for Probabilistic Matrix Factorization
    nrow = train_data.shape[0] #number of rows in data
    ncol = train_data.shape[1] #number of columns
    L = np.zeros((iterations,1))
    
    #unique list
    user_list = [] #must fill from the data
    object_list = [] #ditto..
    
    #calculate number of unique users and unique objects from the data
    maxUserIndex = 0
    maxObjectIndex = 0
    for row in range(nrow):
        temp1 = int(train_data[row,0])
        temp2 = int(train_data[row,1])
        
        isUserInList = int(temp1) in user_list
        isObjectInList = int(temp2) in object_list
        
        if temp1 > maxUserIndex:
            maxUserIndex = temp1
        if temp2 > maxObjectIndex:
            maxObjectIndex = temp2
        
        #if this user was not in the list before add user to list
        if isUserInList == False:
            user_list.extend([temp1])
        #if this object was not in the list before then add to list
        if isObjectInList == False:
            object_list.extend([temp2])
            
    #sort this list
    user_list.sort() #from smallest to biggest
    object_list.sort() #from smallest to biggest
    
    #nusers = len(user_list) #number of users
    #nobjects = len(object_list) #number of objects
    nusers = int(maxUserIndex)
    nobjects = int(maxObjectIndex)
    
    #the matrix factorization
    Omega = np.full((nusers,nobjects),0.00)
    matrix_M = np.full((nusers,nobjects),0) #be like a boolean matrix 1 or 0 if [i,j] was rated
    
    #fill omega with ratings. there should be many missing data
    for row in range(nrow):
        usr = int(train_data[row,0]) #user #
        obj = int(train_data[row,1]) #object #
        rating = train_data[row,2] #ratings
        Omega[usr-1,obj-1] = rating
        matrix_M[usr-1,obj-1] = 1 #indicate that i,j was rated
    
    #probabilistic user locations and object locations
    Nj = np.zeros((nusers,d)) #like the U with dimension: N1*d
    Vj = np.zeros((nobjects,d)) #like the V with dimension: N2*d
    
    #generate N(j) ~ N(0,I/lambda) where N(j) is in R^d. Ditto for V(j)
    #the assumptions of the parameters of the distribution are specified in the script above
    list_mu = [0]*d #since d = 5, [0,0,0,0,0] this is the 0 vector
    matrix_sigma = np.diag([(1/lam)]*d) # this is I/lambda covariance matrix
    for row in range(nusers):
        #user locations being randomly generated
        temp = np.random.multivariate_normal(list_mu,matrix_sigma) #remind myself how to draw random vectors from gaussian multivariate later
        temp2 = np.matrix(temp) #1xd
        Nj[row,0:d] = temp2
    for col in range(nobjects):
        #object locations being randomly generated
        temp = np.random.multivariate_normal(list_mu,matrix_sigma) #remind myself how to draw random vectors from gaussian multivariate later
        temp2 = np.matrix(temp) #1xd
        Vj[col,0:d] = temp2
    
    #for tracking purposes
    testU = [np.zeros((nusers,d)) for b in range(iterations)] #we should have 50 matrices of nuser x d matrix
    testV = [np.zeros((nobjects,d)) for b in range(iterations)]

    #start the iteration
    for i in range(iterations):
        
        #for each user, update the user location
        for N1 in range(nusers):
            sigma_M = np.diag([(lam*sigma2)]*d)
            temp = np.zeros((d,d))
            temp4 = np.zeros((d,1))
            for jj in range(nobjects):
                if matrix_M[N1,jj] == 1:
                    #for this particular value of user, this user has rated the (jj)th object
                    vtemp = Vj[jj,0:d]
                    vtemp = np.matrix(vtemp) #1xd matrix
                    temp += np.matmul(vtemp.transpose(),vtemp)
                    temp4 += Omega[N1,jj]*vtemp.transpose() #M[i,j] * v[j] vector.
            temp2 = temp + sigma_M
            temp3 = np.linalg.inv(temp2)
            temp5 = np.matmul(temp3,temp4) #dx1 matrix
            Nj[N1,0:d] = temp5.transpose() #1xd should match 1:(d+1) which is also 1xd
               
        #for each object, update the object location
        for N2 in range(nobjects):
            sigma_M = np.diag([(lam*sigma2)]*d)
            temp = np.zeros((d,d))
            temp4 = np.zeros((d,1))
            for ii in range(nusers):
                if matrix_M[ii,N2] == 1:
                    #for this particular object, this object has been rated by the (ii)th user.
                    utemp = Nj[ii,0:d]
                    utemp = np.matrix(utemp) #1xd matrix
                    temp += np.matmul(utemp.transpose(),utemp) #dx1 1xd so dxd
                    temp4 += Omega[ii,N2]*utemp.transpose() #dx1
            temp2 = temp + sigma_M
            temp3 = np.linalg.inv(temp2)
            temp5 = np.matmul(temp3,temp4) #dx1 matrix
            Vj[N2,0:d] = temp5.transpose() #now 1xd
        
        #some accounting
        testU[i] = Nj #Nj[0:(nusers),0:(d)]
        testV[i] = Vj #Vj[0:(nobjects),0:(d)]
        
        #likelihood
        #for (i,j) in omega, sum (1/2sigma2*(Mij - t(u)*v))^2 
        summer = 0
        for rows in range(nusers):
            for cols in range(nobjects):
                temp = matrix_M[rows,cols] 
                if temp == 1:
                    #i,j exist in this set
                    Mm = Omega[rows,cols]
                    vN = Nj[rows,0:(d)]
                    vV = Vj[cols,0:(d)]
                    vN = np.matrix(vN) #1xd
                    vV = np.matrix(vV)
                    tempVec = np.matmul(vN,vV.transpose()) #should be in R1
                    summer += (1/(2*sigma2))*((Mm - tempVec)**(2))
        summer2 = 0
        for ii in range(nusers):
            vecTemp = Nj[ii,0:(d)]
            vecTemp = np.matrix(vecTemp) #1xD
            Temp2 = np.matmul(vecTemp,vecTemp.transpose())
            summer2 +=  (lam/2)*Temp2
        summer3 = 0
        for ii in range(nobjects):
            vecTemp = Vj[ii,0:(d)]
            vecTemp = np.matrix(vecTemp)
            Temp2 = np.matmul(vecTemp,vecTemp.transpose())
            summer3 += (lam/2)*Temp2
        L[i,0] = -1*summer + -1*summer2 + -1*summer3
    #end of iteration
    
    return(L,testU,testV)
#end of PMF

# Assuming the PMF function returns Loss L, U_matrices and V_matrices (refer to lecture)
L, U_matrices, V_matrices = PMF(train_data)

#edX checks for iterations x 1 matrix = L for likelihood
np.savetxt("objective.csv", L, delimiter=",")

#the ith row should contain the ith user's vector (5 values)
#only need for the following iterations: 10,25,50
np.savetxt("U-10.csv", U_matrices[9], delimiter=",")
np.savetxt("U-25.csv", U_matrices[24], delimiter=",")
np.savetxt("U-50.csv", U_matrices[49], delimiter=",")

#the jth row should contain the jth object's vector (5 values)
#only need for the following iterations: 10, 25, 50
np.savetxt("V-10.csv", V_matrices[9], delimiter=",")
np.savetxt("V-25.csv", V_matrices[24], delimiter=",")
np.savetxt("V-50.csv", V_matrices[49], delimiter=",")
