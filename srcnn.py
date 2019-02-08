import time
import os
import matplotlib.pyplot as plt
import skimage
import numpy as np
import tensorflow as tf
import scipy
#we need to add ndimage here
import scipy.ndimage
import pdb #breakpoints: line 139

#========================Task 1 Understanding IR===============================
    #Ques 1: read
path = 'C:/Users/surface/Desktop/CW1_for_students/CW1_Handout_Template_code/tf-SRCNN/image/butterfly_GT.bmp'
butterfly = scipy.misc.imread(path)#256x256x3 RGB

    #Ques 3: change RGB into gray—scale by using formula Y' = 0.299 R + 0.587 G + 0.114 B 
butterfly_gray = np.dot(butterfly,[0.299,0.587,0.114])
plt.figure(1)
plt.imshow(butterfly_gray, cmap=plt.get_cmap('gray'))
plt.title('Original (no-bicubic)')
    #Ques 4: use bicubic to shrink the current image
h,w = butterfly_gray.shape[:2]
#prepare to shrink 3 times
h_,w_ = int(h/3),int(w/3)
#Ques4: shrinked
bicubic_butterfly = scipy.misc.imresize(butterfly, (h_,w_),interp='bicubic',mode='F')#shrink 3 times
plt.figure(2)
plt.subplot(1,2,1)
plt.imshow(bicubic_butterfly, cmap=plt.get_cmap('gray'))
plt.title('LR-input (bicubic)')

#Hints: bicubic_butterfly_re != butterfly_gray. because the data structure are floar and uint8
    #Ques4: by using tensorflow.image.resize_images. method=2 represent bicubic interpolation
with tf.Session() as sess:
    resized = tf.image.resize_images(butterfly, [h_,w_],method=2)  #第一个参数为原始图像，第二个参数为图像大小，第三个参数给出了指定的算法
    resized = np.asarray(resized.eval(),dtype='uint8')
    plt.figure(2)
    plt.subplot(1,2,2)
    plt.imshow(resized)
    plt.title('LR-input(bicubic_tf)')
    sess.close()
    
#Ques5: enlarged (recover by using the current img)
bicubic_butterfly_re = scipy.misc.imresize(bicubic_butterfly, (h,w), interp='bicubic',mode='F')#enlarge the shrinked_img 3times to recover
#==============================================================================


def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def preprocess(path, scale=3):
  """
  Preprocess single image file 
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation
  Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
  """
  image = imread(path, is_grayscale=True)
  label_ = modcrop(image, scale)

  # Must be normalized
  image = image / 255.
  label_ = label_ / 255.

  input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
  input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)

  return input_, label_

"""Set the image hyper parameters
"""
c_dim = 1
input_size = 255

"""Define the model weights and biases 
"""

# define the placeholders for inputs and outputs
inputs = tf.placeholder(tf.float32, [None, input_size, input_size, c_dim], name='inputs')

## ------ Add your code here: set the weight of three conv layers
# replace '0' with your hyper parameter numbers 
# conv1 layer with biases: 64 filters with size 9 x 9
# conv2 layer with biases and relu: 32 filters with size 1 x 1
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5
weights = {
    'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1'),#which means there are 64 numbers of 9x9x1 filters
    'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),#32 numbers of 1x1x64 filters
    'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')#just 1 of 5x5x32 filters
    }

biases = {
      'b1': tf.Variable(tf.zeros([64]), name='b1'),#<64>the number of bias depends on the number of filters in this layer
      'b2': tf.Variable(tf.zeros([32]), name='b2'),#<32> filters
      'b3': tf.Variable(tf.zeros([1]), name='b3')#<1> filter
    }


"""Define the model layers with three convolutional layers
"""
#======================= Task 3 Image SR using DNN ============================
## ------ Add your code here: to compute feature maps of input low-resolution images
# replace 'None' with your layers: use the tf.nn.conv2d() and tf.nn.relu()
# conv1 layer with biases and relu : 64 filters with size 9 x 9
    #Ques3.1
#path = 'C:/Users/surface/Desktop/CW1_for_students/CW1_Handout_Template_code/tf-SRCNN/image/butterfly_GT.bmp'
#butterfly = scipy.misc.imread(path)#256x256x3 RGB
#plt.imshow(butterfly)
#    #Ques3.2
#[LR_input, HR_original] = preprocess(path, scale=3)

#change the 'inputs' here to 'LR_input' to finish ques3.2
conv1 = tf.nn.relu(tf.nn.conv2d(inputs, weights['w1'], strides=[1,1,1,1], padding='VALID') + biases['b1'])
##------ Add your code here: to compute non-linear mapping
# conv2 layer with biases and relu: 32 filters with size 1 x 1

conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['w2'], strides=[1,1,1,1], padding='VALID') + biases['b2'])
##------ Add your code here: compute the reconstruction of high-resolution image
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5
conv3 = tf.nn.conv2d(conv2, weights['w3'], strides=[1,1,1,1], padding='VALID') + biases['b3']


"""Load the pre-trained model file
"""
model_path='C:/Users/surface/Desktop/CW1_for_students/CW1_Handout_Template_code/tf-SRCNN/model/model.npy'
model = np.load(model_path, encoding='latin1').item()
#b1 = model['b1']
#b2 = model['b2']
#b3 = model['b3']
#w1 = model['w1']
#w2 = model['w2']
#w3 = model['w3']
##------ Add your code here: show the weights of model and try to visualisa
## variabiles (w1, w2, w3)
##====================== Task 2 Understanding DNNs by CNNs======================
#que4_input = tf.Variable(tf.random_normal([1,19,19,1]))#define 1 random img with 9x9, 1 channel as the test of Ques4.5
#init_task2 = tf.initialize_all_variables()
#with tf.Session() as sess:
#    sess.run(init_task2)
#    #Ques2.3 Show the No.1 filters of layer_1. <Hint: index started from 0>
#    w1_1 = sess.run(weights['w1'])[:,:,:,0]
#        #Ques2.4 show the No.10 of layer_1_bias
#    b1_10 = sess.run(biases['b1'])[9]
#        #Ques3.3 the No.5 filters of layer_2
#    w2_5 = sess.run(weights['w2'])[:,:,:,4]
#        #Ques3.4 the No.6 of layer_2_bias
#    b2_6 = sess.run(biases['b2'])[5]
#    #Ques3.5 the channel number of the input: ANSWER is 64, which depends on the number of filters in previous layer
#        #Ques4.3 Show the No.1 filters of layer_3.
#    w3_1 = sess.run(weights['w3'])[:,:,:,0]
#        #Ques4.4 Show the No.1 of layer_3_bias
#    b3_1 = sess.run(biases['b3'])[0]
#        #Ques4.5 do convolution by using a designed 2-D filter and a 2-D matrix
#    que4_w = weights['w1']
#        #the middle of one is the strides.another two must be one.Valid==0 padding.using relu here with layer1_bias
#    ques4_5_conv2d = tf.nn.relu(tf.nn.conv2d(que4_input, que4_w, strides=[1, 1, 1, 1], padding='VALID')+biases['b1'])
#    ans4_5 = sess.run(ques4_5_conv2d)
#sess.close()
##Set the breakpoint
#pdb.set_trace() 
#==============================================================================


"""Initialize the model variabiles (w1, w2, w3, b1, b2, b3) with the pre-trained model file
"""
# launch a session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for key in weights.keys():
  sess.run(weights[key].assign(model[key]))

for key in biases.keys():
  sess.run(biases[key].assign(model[key]))

"""Read the test image
"""
blurred_image, groudtruth_image = preprocess('./image/butterfly_GT.bmp')

"""Run the model and get the SR image
"""
# transform the input to 4-D tensor
input_ = np.expand_dims(np.expand_dims(blurred_image, axis =0), axis=-1)

# run the session
# here you can also run to get feature map like 'conv1' and 'conv2'
ouput_ = sess.run(conv3, feed_dict={inputs: input_})
output = ouput_[0,:,:,0]
    #Ques3.5 Show the output SR-img of SRCNN 
plt.figure(1)
plt.subplot(1,3,3)
plt.title('High-Resolution')
plt.imshow(output,cmap=plt.get_cmap('gray'))
##------ Add your code here: save the blurred and SR images and compute the psnr
# hints: use the 'scipy.misc.imsave()'  and ' skimage.meause.compare_psnr()'
    #Ques3.6 Evaluate the performance
    #HINT!here we must use padding ='SAME', otherwise we eill loss 12 pixels per col and row 
#=============TO CALCULATE THE PSNR OF GroundTRUTH and SRCNN-OUTPUT============
#diff = output - groudtruth_image
#rmse = np.sqrt(np.sum(diff**2))/len(output)
#psnr = 20*np.log10(255/rmse)     #max/rmse
    #Ques3.7 Use bicubic to enlarge the output pic
diff_baseline = bicubic_butterfly_re - butterfly_gray
psnr_baseline =  20*np.log10(255/(np.sqrt(np.sum(diff_baseline**2))/len(bicubic_butterfly_re)))
    #Ques3.8 Compare psnr
psnr_scikit = skimage.measure.compare_psnr(bicubic_butterfly_re,butterfly_gray)
plt.figure(2)
plt.title('GT')
plt.imshow(groudtruth_image,cmap=plt.get_cmap('gray'))

plt.figure(3)
plt.title('HR-BI')
plt.imshow(bicubic_butterfly_re,cmap=plt.get_cmap('gray'))

plt.figure(4)
plt.title('HR-SRCNN')
plt.imshow(output,cmap=plt.get_cmap('gray'))
    #Ques9: ANS: Here the result of 'psnr_baseline' = ' psnr_scikit', which represent the 
            #result of baseline_bicubic. The SRCNN-PSNR result is saved as 'psnr'
            #psnr ~=68.88  and psnr_baseline ~=22.04. So the result of SRCNN is better than Baseline