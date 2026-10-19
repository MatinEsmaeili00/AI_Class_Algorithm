
#mathematical operations library
import numpy as np

# each row is representing as a image for our example here,
x_train_img = np.array([

[1,1],
[2,2],
[8,8],
[9,9]

], dtype = np.float32) # I'm turning int64 to float32 bc I wanna check the distance with the other image that is fload and in the checking distance we weill subtrack

y_train = np.array([0,0,1,1]) # 0 for class 0 and 1 for class  1

test_img = np.array([[1.5,1.5]], dtype = np.float32) # this is the image that we want to classify


def predict_knn(x_train, y_train, test, k=3): # it is ganna choose 3 neighbors !


    # Calculate Euclidean distance from test to every training point
    # 1. (X_train - test): Subtract test from each training point
    # 2. ** 2: Square each difference
    # 3. np.sum(..., axis=1): Sum squared differences row by row
    # 4. np.sqrt(): Take square root to get the final distances
    distances = np.sqrt(
        np.sum((x_train - test) ** 2, axis =1) 
        
        #axis=1 adds across columns, producing one number per training example.
    )
