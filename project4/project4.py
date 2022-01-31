import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from numpy.linalg import norm
import os
from random import normalvariate
from math import sqrt
FOLDER = "./Dataset/"
FILES = os.listdir(FOLDER)
TEST_DIR = "./Testset/"

def load_images_train_and_test(TEST):
    test=np.asarray(Image.open(TEST)).flatten()
    train=[]
    for name in FILES:
        train.append(np.asarray(Image.open(FOLDER + name)).flatten())
    train= np.array(train)
    return test,train
   
def normalize(test,train):
    """
    TODO : Normalize test and train and return them properly
    Hint : To calculate mean properly use two arguments version of mean numpy method (https://www.javatpoint.com/numpy-mean)
    Hint : normalize test with train mean
    """
    arr = np.mean(train, axis=0)
    
    normalized_test = test - arr
    
    normalized_train = np.empty((0,train.shape[1]))
    for i in range(train.shape[0]):
        normalized_train = np.vstack([normalized_train, (np.array(train[i, :]) - arr)])

    return normalized_test,normalized_train
def svd_function(images):
    """
    TODO : implement SVD (use np.linalg.svd) and return u,s,v 
    Additional(Emtiazi) todo : implement svd without using np.linalg.svd
    """
    if iteration == 0:
        return svd(images)
    else:
        return u,s,v
#   return np.linalg.svd(images, full_matrices=False)
    
    
def project_and_calculate_weights(img,u):
    """
    TODO : calculate element wise multiplication of img and u . (you can use numpy methods)
    """
    return np.multiply(img, u)

def predict(test,train):
    """
    TODO : Find the most similar face to test among train set by calculating errors and finding the face that has minimum error
    return : index of the data that has minimum error in train dataset
    Hint : error(i) = norm(train[:,i] - test)       (you can use np.linalg.norm)
    
    """
    min_error = 1_000_000_000
    min_index = 0
    for i in range(train.shape[1]):
        error = np.linalg.norm(train[:,i] - test)
        if error < min_error:
            min_index = i
            min_error = error
    
    return min_index
def plot_face(tested,predicted):
    """
    TODO : Plot tested image and predicted image . It would be great if you show them next to each other 
    with subplot and figures that you learned in matplotlib video in the channel.
    But you are allowed to show them one by one
    """
    
    f, plt_arr = plt.subplots(1, 2 ,figsize=(7, 3))
    f.suptitle('Result Plots')

    plt_arr[0].imshow(tested, cmap = "gray")
    plt_arr[0].set_title('tested')

    plt_arr[1].imshow(predicted, cmap = "gray")
    plt_arr[1].set_title('predicted')

###################################### SVD part ######################################

def randomUnitVector(n):
    unnormalized = [normalvariate(0, 1) for _ in range(n)]
    theNorm = sqrt(sum(x * x for x in unnormalized))
    return [x / theNorm for x in unnormalized]


def svd_1d(A):
    ''' The one-dimensional SVD 
        we use  Power iteration method to calculate svd :
        In fact this algorithm will produce the greatest (in absolute value) eigenvalue of A,
        so with help of this algo we can find eigen values and eigen vectors
        (eigen vectors will be orthogonal) one by one and then construct svd from them.
        
        In Power iteration method we start with v_0 which might be a random vector.
        At every iteration this vector is updated using following rule:
                v_k+1 = Bv_k / ||Bv_k||
        We’ll continue until result has converged.
        Power method has few assumptions:
         - v_0 has a nonzero component in the direction of an eigenvector associated with the dominant eigenvalue.
           (it means v_0 is NOT orthogonal to the eigenvector)
           Initializing v_0 randomly minimizes possibility that this assumption is not fulfilled.
         - matrix A has dominant eigenvalue which has strictly greater magnitude than other eigenvalues.
        These assumptions guarantee that algorithm converges to a reasonable result.
        
        So at the end when v_i converges enough this method found dominant singular value/eigenvector and return the eigen vector.
    '''

    n, m = A.shape
    
    # v_0 = x = random unit vector
    x = randomUnitVector(min(n,m))
    currentV = x
    
    #v_1 = ?
    lastV = None
    
    
    # calculate B according to shape of A so that we smaller size computation
    if n > m:
        B = np.dot(A.T, A)
    else:
        B = np.dot(A, A.T)
        
         
    # v_k+1 = Bv_k / ||Bv_k||
    iterations = 0
    epsilon=1e-10
    while True:
        iterations += 1
        lastV = currentV
        currentV = np.dot(B, lastV)
        currentV = currentV / norm(currentV)
        
        # continue until result has converged (updates are less than threshold).
        # if two normal vector become same then inner product of them will be 1
        if abs(np.dot(currentV, lastV)) > 1 - epsilon:
            return currentV


def svd(A):
    '''
        Compute the singular value decomposition of a matrix A
        using the power method. A is the input matrix, and k
        is the number of singular values you wish to compute.
        If k is None, this computes the full-rank decomposition.
    '''
    A = np.array(A, dtype=float)
    n, m = A.shape
    # save (singular value, u, v) as each element
    svdSoFar = []
    k = min(n, m)

    for i in range(k):
        matrixFor1D = A.copy()
        
        # remove all previous eigen values and vectors (dominant one) from matrix 
        # so the next dominant eigen value and vector won't be repetitive
        # A_next = A-(singular_value)(u)(v.T)
        for singularValue, u, v in svdSoFar[:i]:
            matrixFor1D -= singularValue * np.outer(u, v)

        # 1. find v_i which is the next eigen vector for B = A.T @ A
        # 2. find sigma_i = ||Av_i|| (reason is in the next line)
        # ||Av_i||^2 = (Av_i).T A (Av_i) = (v_i).T @ A.T @ A @ v_i = (v_i).T @ (landa_i * v_i) ==v_i is orthonormal== landa_i = sigma_i ^ 2
        # 3. find u_i = 1/sigma_i * Av_i
        if n > m:
            # 1
            v = svd_1d(matrixFor1D)  
            u_unnormalized = np.dot(A, v)
            # 2
            sigma = norm(u_unnormalized)  # next singular value
            # 3
            u = u_unnormalized / sigma
        else:
            u = svd_1d(matrixFor1D)  # next singular vector
            v_unnormalized = np.dot(A.T, u)
            sigma = norm(v_unnormalized)  # next singular value
            v = v_unnormalized / sigma

        # add new (sigma, u, v) we have found
        svdSoFar.append((sigma, u, v))

    singularValues, us, vs = [np.array(x) for x in zip(*svdSoFar)]
    return us.T, singularValues, vs
###################################################################################

true_predicts=0
all_predicts=0
iteration = 0
for TEST_FILE in os.listdir(TEST_DIR):
    # Loading train and test
    test,train=load_images_train_and_test(TEST_DIR+TEST_FILE)
    test,train=normalize(test,train)

    test=test.T
    train=train.T
    test = np.reshape(test, (test.size, 1))
    
    # Singular value decomposition
    u,s,v=svd_function(train)

    # Weigth for test
    w_test=project_and_calculate_weights(test,u)
    w_test=np.array(w_test, dtype='int8').flatten()

    # Weights for train set
    w_train=[]
    for i in range(train.shape[1]):
        w_i=project_and_calculate_weights(np.reshape(train[:, i], (train[:, i].size, 1)),u)
        w_i=np.array(w_i, dtype='int8').flatten()
        w_train.append(w_i)
    w_train=np.array(w_train).T
    
    # Predict 
    index_of_most_similar_face=predict(w_test,w_train)
    # Showing results
    print("Test : "+TEST_FILE)
    print(f"The predicted face is: {FILES[index_of_most_similar_face]}")
    print("\n***************************\n")
    
    # Calculating Accuracy
    all_predicts+=1
    if FILES[index_of_most_similar_face].split("-")[0]==TEST_FILE.split("-")[0]:
        true_predicts+=1
        # Plotting correct predictions 
        plot_face(Image.open(TEST_DIR+TEST_FILE),Image.open(FOLDER+FILES[index_of_most_similar_face]))
    else:
        # Plotting wrong predictions
        plot_face(Image.open(TEST_DIR+TEST_FILE),Image.open(FOLDER+FILES[index_of_most_similar_face]))
    iteration += 1
# Showing Accuracy
accuracy=true_predicts/all_predicts
print(f'Accuracy : {"{:.2f}".format(accuracy*100)} %')
