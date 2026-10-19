
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


distances = np.sqrt(
        np.sum((x_train_img - test_img) ** 2, axis =1))

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
    

    # Indix of the k closest examples
    #nearest = np.argsort(distances)[:k]

    #indices of the k closest example long version


# ----------- Start ----------------- trying different algorithm for finding min distance and index ---- 
#  
# def min_index_distance(distance_arr,k):
#     min_index = 0 ;
#     min_distance = 0;

#     for i in range(k):
#         temp_distance_arr = distance_arr;
#         for c in range(len(distance_arr)):
#             for j in range(len(distance_arr)):
#                 index = j; 
#                 if c == index:
#                     index+1;
#                 if temp_distance_arr[index] == -1:
#                     index+1;
#                 if index > j:
#                     break;
#                 if distance_arr [c] > temp_distance_arr[index]:



# very first try

# def min_index_distance(distance_arr, k):

#     nearest = []
#     temp_distance_arr = distance_arr.copy()

#     for i in range(k):

#         min_index = -1
#         min_distance = float('inf')

#         for c in range(len(distance_arr)):

#             if temp_distance_arr[c] == -1:
#                 continue

#             for j in range(len(distance_arr)):

#                 index = j

#                 # If comparing with itself, move to next index
#                 if c == index:
#                     index += 1

#                 # Make sure index is inside the array
#                 if index >= len(temp_distance_arr):
#                     break

#                 # If already selected, move to next index
#                 while index < len(temp_distance_arr) and temp_distance_arr[index] == -1:
#                     index += 1

#                 # Check again to prevent array out of bounds
#                 if index >= len(temp_distance_arr):
#                     break

#                 # If another distance is smaller, c is not minimum
#                 if temp_distance_arr[c] > temp_distance_arr[index]:
#                     break

#             # If no smaller distance was found, save c
#             else:
#                 min_index = c
#                 min_distance = temp_distance_arr[c]
#                 break

#         nearest.append(min_index)
#         temp_distance_arr[min_index] = -1

#     return nearest




# First try algorithm
# def min_index_distance(distance_arr, k):

#     nearest = []
#     temp_distance_arr = distance_arr.copy()

#     for i in range(k):

#         min_index = -1
#         min_distance = float('inf')

#         for c in range(len(distance_arr)):

#             if temp_distance_arr[c] == -1:
#                 continue

#             for j in range(len(distance_arr)):

#                 index = j

#                 if c == index:
#                     continue

#                 if temp_distance_arr[index] == -1:
#                     continue

#                 if temp_distance_arr[c] > temp_distance_arr[index]:
#                     break

#             else:
#                 min_index = c
#                 min_distance = temp_distance_arr[c]
#                 break

#         nearest.append(min_index)
#         temp_distance_arr[min_index] = -1

#     return nearest



# more Optimize version algorithm
def min_index_distance(distance_arr, k):

    nearest = []

    # Copy distances so we don't change the original array
    temp_distance_arr = distance_arr.copy()

    for i in range(k):

        min_index = -1
        min_distance = float('inf')

        for j in range(len(temp_distance_arr)):

            # Skip distances that we already selected
            if temp_distance_arr[j] == -1:
                continue

            # Is this distance smaller than our current minimum?
            if temp_distance_arr[j] < min_distance:

                min_distance = temp_distance_arr[j]
                min_index = j

        # Save the index of the closest image
        nearest.append(min_index)

        # Mark this image as already selected
        temp_distance_arr[min_index] = -1

    return nearest

# ----------- End ---------------- trying different algorithm for finding min distance and index ----


# nearest = min_index_distance(distances, 3)
# class_arrange = 0;
# class_name = 0;

# # identifying class and their index
# def class_index_connection(y_train):
#     for i in range(len(y_train)):
#         index = i;
#          while index < len(y_train):
#             index+1;
#             if y_train[i] == y_train[index]:
#                 class_arrange +1;
                 
#             else:
#                 class_name.append( y_train[i])




# nearest = min_index_distance(distances, 3)

# class_arrange = {}

# # identifying class and their index
# def class_index_connection(y_train):

#     class_arrange = {}

#     for i in range(len(y_train)):

#         index = i + 1

#         # If class doesn't exist, add it
#         if y_train[i] not in class_arrange:
#             class_arrange[y_train[i]] = [i]

#         while index < len(y_train):

#             if y_train[i] == y_train[index]:

#                 # Add matching index to this class
#                 if index not in class_arrange[y_train[i]]:
#                     class_arrange[y_train[i]].append(index)

#             index += 1

#     return class_arrange


# class_arrange = class_index_connection(y_train)

# print(class_arrange)






# # identifying classes and their indices

# def class_index_connection(y_train):

#     class_arrange = {}

#     for i in range(len(y_train)):

#         found = False

#         for class_name in class_arrange:

#             if y_train[i] == class_name:

#                 class_arrange[class_name].append(i)
#                 found = True
#                 break

#         if found == False:

#             class_arrange[y_train[i]] = [i]

#     return class_arrange


# y_train = np.array([3, 0, 7, 3, 2, 7, 0, 3])

# class_arrange = class_index_connection(y_train)

# print(class_arrange)




#                     
