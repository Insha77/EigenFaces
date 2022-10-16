
# This function reads image from the given image path 
def read_im(path):
  im = PIL.Image.open(path)
  im = np.array(im)
  return im

# As working with images, this function plots any binnary image
def show_im(image):
  plt.imshow(image,cmap='gray')
  plt.show()


SUB = 40        # Number of subjects
IPS = 10        # Images per subject
N = 256         # Width and Height of images
m = SUB*IPS     # Total Images

face_mat = np.zeros((m,N,N))   # To store images

# Getting all images into the face matrix
count = 0
data=np.load("../input/olivetti_faces.npy")
X=data.reshape((data.shape[0],data.shape[1]*data.shape[2]))
target=np.load("../input/olivetti_faces_target.npy")


def train_test_split(face_mat,Train_per_sub = 8):
    # Train test Split:
    Train_ims = Train_per_sub*SUB           # Total Train Ims
    Test_per_sub = IPS - Train_per_sub      # Test Images per subject
    Test_ims = Test_per_sub*SUB             # Total Test Ims

    train_mat = np.zeros((Train_ims,N,N))   # Train Matrix
    test_mat = np.zeros((Test_ims,N,N))     # Test Matrix
    train_count = 0                         # Train counter
    test_count = 0                          # Test counter

    for count,face in enumerate(face_mat):
        if count%IPS <Test_per_sub:
            # print("TEST",count,test_count)
            test_mat[test_count] = face_mat[count]
            test_count += 1
        
        else:
            # print("TRAIN",count,train_count)
            train_mat[train_count] = face_mat[count]
            train_count += 1
    return train_mat,test_mat
train_per_sub = 8              # Train images per subject (max 22)
m = train_per_sub*SUB           # Total ims
face_mat /= 255                 # Making images out of 1
# Train Test split
train_mat, test_mat = train_test_split(face_mat,train_per_sub)

# We need A of shape (N*N,m)
A = np.reshape(train_mat,(m,N*N)).T
print("Shape of A is ",A.shape)

# Calculating Average Face
A_avg = np.mean(A,axis=1)
print("Shape of A_avg is ",A_avg.shape)
print("This is the average face-")
show_im(np.reshape(A_avg,(N,N)))

# Calculating Normalized Faces
Phi_1 = (A.T - A_avg.T).T
print("Phi_1 shape is ",Phi_1.shape)
# Calculating m*m covarience matrices and their eigen values
Cov_mbym = np.dot(Phi_1.T,Phi_1)
print("Cov_mxm shape is ",Cov_mbym.shape)

# Finding the Eigen vectors of the Covariance Matrix
eig,eig_vec = np.linalg.eig(Cov_mbym)
eig,eig_vec = np.real(eig),np.real(eig_vec)

# Selecting K values out of m and dotting N.Nxm with mxk 
k = len(eig_vec)
eig_k,eig_k_vec = eig[:k],np.dot(Phi_1,eig_vec[:,:k])
print(eig_k.shape,eig_k_vec.shape)

# Projecting the Normalized images on to the K vector Space
embedded = np.dot(Phi_1.T,eig_k_vec)
print("Embedding for all faces", embedded.shape)
# Defining the distance function
def cos_sim(a,b):
  return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

# Defining a function to return a face for given image embedding that has most number of minimum distances
def check_face_all(face,faces,thres=0.6):
    check = np.zeros(SUB)
    
    for i in range(SUB):
        for j in range(train_per_sub):
            if cos_sim(face,faces[i*train_per_sub +j])>thres:
                check[i] += 1
    return np.argmax(check),check

thres = 0.9
# We check the train accuracy here of our eigen faces method

correct = 0
mat = np.zeros((2,2))

for i in range(SUB):
    for j in range(train_per_sub):
        
        id,t = check_face_all(embedded[i*train_per_sub+ j],embedded,thres)
       
        if id == i: #id==i: #
            correct += 1
print("Correct = {}, Wrong = {}".format(correct,300-correct))
print("Accuracy = ",correct/300)
# We use the this block to check the testing accuracy

# Setting the threshold
thres = 0.8

test_count = SUB*IPS - m
test_per_sub = IPS - train_per_sub
print("test_mat shape is ",test_mat.shape)

# We change the test mat to (m,N*N)
test_A = np.reshape(test_mat,(test_count,N*N)).T
print('Test_A shape is ',test_A.shape)

# We normalize the test faces
test_phi_1 = (test_A.T - A_avg.T).T
print("test_phi_1 shape is",test_phi_1.shape)

# We project the test images into our k space
test_embed = np.dot(test_phi_1.T,eig_k_vec)
print("test_embed shape is ",test_embed.shape)

# We check the accuracy
correct = 0
for i in range(SUB):
    for j in range(test_per_sub):
        id,t = check_face_all(test_embed[i*test_per_sub+ j],embedded,thres)
        print("For test image of subject-{}, {} images found for {} face".format(str(i+1).zfill(2),str(int(max(t))).zfill(2),str(np.argmax(t)+1).zfill(2)))
        if id == i:
            correct += 1
print("Correct = {}, Wrong = {}".format(correct,test_count-correct))
print("Accuracy = ",correct/test_count)
