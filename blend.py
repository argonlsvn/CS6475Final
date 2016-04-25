__author__ = 'Anh Nguyen'


import numpy as np
import scipy as sp
import scipy.signal
import cv2
import math

""" Assignment 6 - Blending

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, writes the image that
    you are being passed in. Do that on your own if you need to save the images
    but the functions should NOT save the image to file. (This is a problem
    for us when grading because running 200 files results a lot of images being
    saved to file and opened in dialogs, which is not ideal). Thanks.

    2. DO NOT import any other libraries aside from the three libraries that we
    provide. You may not import anything else, you should be able to complete
    the assignment with the given libraries (and in most cases without them).

    3. DO NOT change the format of this file. Do not put functions into classes,
    or your own infrastructure. This makes grading very difficult for us. Please
    only write code in the allotted region.
"""

def generatingKernel(parameter):
  """ Return a 5x5 generating kernel based on an input parameter.

  Note: This function is provided for you, do not change it.

  Args:
    parameter (float): Range of value: [0, 1].

  Returns:
    numpy.ndarray: A 5x5 kernel.

  """
  kernel = np.array([0.25 - parameter / 2.0, 0.25, parameter,
                     0.25, 0.25 - parameter /2.0])
  return np.outer(kernel, kernel)

def reduce(image):
    """ Convolve the input image with a generating kernel of parameter of 0.4 and
    then reduce its width and height by two.

    Please consult the lectures and readme for a more in-depth discussion of how
    to tackle the reduce function.

    You can use any / all functions to convolve and reduce the image, although
    the lectures have recommended methods that we advise since there are a lot
    of pieces to this assignment that need to work 'just right'.

    Args:
    image (numpy.ndarray): a grayscale image of shape (r, c)

    Returns:
    output (numpy.ndarray): an image of shape (ceil(r/2), ceil(c/2))
      For instance, if the input is 5x7, the output will be 3x4.

    """
    # WRITE YOUR CODE HERE.
    kernel = generatingKernel(0.4)

    convolvedImage = scipy.signal.convolve2d(image,kernel,'same')
    reducedImage = convolvedImage[::2,::2].copy()
    return reducedImage


  # END OF FUNCTION.

def expand(image):
    """ Expand the image to double the size and then convolve it with a generating
    kernel with a parameter of 0.4.

    You should upsample the image, and then convolve it with a generating kernel
    of a = 0.4.

    Finally, multiply your output image by a factor of 4 in order to scale it
    back up. If you do not do this (and I recommend you try it out without that)
    you will see that your images darken as you apply the convolution. Please
    explain why this happens in your submission PDF.

    Please consult the lectures and readme for a more in-depth discussion of how
    to tackle the expand function.

    You can use any / all functions to convolve and reduce the image, although
    the lectures have recommended methods that we advise since there are a lot
    of pieces to this assignment that need to work 'just right'.

    Args:
    image (numpy.ndarray): a grayscale image of shape (r, c)

    Returns:
    output (numpy.ndarray): an image of shape (2*r, 2*c)
    """
    # WRITE YOUR CODE HERE.

    kernel = generatingKernel(0.4)
    height, width = image.shape[:2]
    upscaledImage = np.zeros((height*2., width*2.))
    upscaledImage[::2,::2] = image
    expandedImage = scipy.signal.convolve2d(upscaledImage,kernel,'same')
    expandedImage[:] = [x * 4 for x in expandedImage]
    return expandedImage
    # END OF FUNCTION.

def gaussPyramid(image, levels):
    """ Construct a pyramid from the image by reducing it by the number of levels
    passed in by the input.

    Note: You need to use your reduce function in this function to generate the
    output.

    Args:
    image (numpy.ndarray): A grayscale image of dimension (r,c) and dtype float.
    levels (uint8): A positive integer that specifies the number of reductions
                    you should do. So, if levels = 0, you should return a list
                    containing just the input image. If levels = 1, you should
                    do one reduction. len(output) = levels + 1

    Returns:
    output (list): A list of arrays of dtype np.float. The first element of the
                   list (output[0]) is layer 0 of the pyramid (the image
                   itself). output[1] is layer 1 of the pyramid (image reduced
                   once), etc. We have already included the original image in
                   the output array for you. The arrays are of type
                   numpy.ndarray.

    Consult the lecture and README for more details about Gaussian Pyramids.
    """
    output = [image]
    # WRITE YOUR CODE HERE.
    for i in range(0,levels):
        levelImage = reduce(output[i])
        output.append(levelImage)
    return output
    # END OF FUNCTION.

def laplPyramid(gaussPyr):
    """ Construct a Laplacian pyramid from the Gaussian pyramid, of height levels.

    Note: You must use your expand function in this function to generate the
    output. The Gaussian Pyramid that is passed in is the output of your
    gaussPyramid function.

    Args:
    gaussPyr (list): A Gaussian Pyramid as returned by your gaussPyramid
                     function. It is a list of numpy.ndarray items.

    Returns:
    output (list): A Laplacian pyramid of the same size as gaussPyr. This
                   pyramid should be represented in the same way as guassPyr,
                   as a list of arrays. Every element of the list now
                   corresponds to a layer of the Laplacian pyramid, containing
                   the difference between two layers of the Gaussian pyramid.

           output[k] = gauss_pyr[k] - expand(gauss_pyr[k + 1])

           Note: The last element of output should be identical to the last
           layer of the input pyramid since it cannot be subtracted anymore.

    Note: Sometimes the size of the expanded image will be larger than the given
    layer. You should crop the expanded image to match in shape with the given
    layer.

    For example, if my layer is of size 5x7, reducing and expanding will result
    in an image of size 6x8. In this case, crop the expanded layer to 5x7.
    """
    output = []
    # WRITE YOUR CODE HERE.
    for i in range(0,len(gaussPyr)-1):
        expandedImg = expand(gaussPyr[i+1].copy())
        output.append(expandedImg)
    output.append(gaussPyr[len(gaussPyr)-1].copy())


    for i in range(0,len(gaussPyr)-1):
        height, width = output[i].shape[:2]
        if gaussPyr[i].shape[0] < output[i].shape[0]:
            height = height - 1
        if gaussPyr[i].shape[1] < output[i].shape[1]:
            width = width - 1
        output[i] = output[i][0:height, 0:width]
        output[i] = gaussPyr[i] - output[i]
    return output
    # END OF FUNCTION.

def blend(laplPyrWhite, laplPyrBlack, gaussPyrMask):
    """ Blend the two Laplacian pyramids by weighting them according to the
    Gaussian mask.

    Args:
    laplPyrWhite (list): A Laplacian pyramid of one image, as constructed by
                         your laplPyramid function.

    laplPyrBlack (list): A Laplacian pyramid of another image, as constructed by
                         your laplPyramid function.

    gaussPyrMask (list): A Gaussian pyramid of the mask. Each value is in the
                         range of [0, 1].

    The pyramids will have the same number of levels. Furthermore, each layer
    is guaranteed to have the same shape as previous levels.

    You should return a Laplacian pyramid that is of the same dimensions as the
    input pyramids. Every layer should be an alpha blend of the corresponding
    layers of the input pyramids, weighted by the Gaussian mask. This means the
    following computation for each layer of the pyramid:
    output[i, j] = current_mask[i, j] * white_image[i, j] +
                   (1 - current_mask[i, j]) * black_image[i, j]
    Therefore:
    Pixels where current_mask == 1 should be taken completely from the white
    image.
    Pixels where current_mask == 0 should be taken completely from the black
    image.

    Note: current_mask, white_image, and black_image are variables that refer to
    the image in the current layer we are looking at. You do this computation for
    every layer of the pyramid.
    """

    blended_pyr = []
    # WRITE YOUR CODE HERE.
    if len(laplPyrWhite) == len(laplPyrBlack) == len(gaussPyrMask):
        for level in range(0,len(laplPyrWhite)):
            whiteImage = laplPyrWhite[level]
            blackImage = laplPyrBlack[level]
            currentMask = gaussPyrMask[level]
            height, width = whiteImage.shape[:2]
            output = currentMask.copy()
            for i in range(0,height):
                for j in range(0,width):
                    output[i][j] = whiteImage[i][j] * currentMask[i][j] + (1 - currentMask[i][j]) * blackImage[i][j]
            blended_pyr.append(output)
    return blended_pyr
    # END OF FUNCTION.

def collapse(pyramid):
    """ Collapse an input pyramid.

    Args:
    pyramid (list): A list of numpy.ndarray images. You can assume the input is
                  taken from blend() or laplPyramid().

    Returns:
    output(numpy.ndarray): An image of the same shape as the base layer of the
                           pyramid and dtype float.

    Approach this problem as follows, start at the smallest layer of the pyramid.
    Expand the smallest layer, and add it to the second to smallest layer. Then,
    expand the second to smallest layer, and continue the process until you are
    at the largest image. This is your result.

    Note: sometimes expand will return an image that is larger than the next
    layer. In this case, you should crop the expanded image down to the size of
    the next layer. Look into numpy slicing / read our README to do this easily.

    For example, expanding a layer of size 3x4 will result in an image of size
    6x8. If the next layer is of size 5x7, crop the expanded image to size 5x7.
    """
    # WRITE YOUR CODE HERE.
    output = pyramid[len(pyramid)-1].copy()
    for i in range(len(pyramid)-1,0,-1):
        expanded = expand(output)
        height, width = expanded.shape[:2]
        if pyramid[i-1].shape[0] < height:
            height = height - 1
        if pyramid[i-1].shape[1] < width:
            width = width - 1
        expanded = expanded[0:height, 0:width]
        output = pyramid[i-1] + expanded
    #output = pyramid[0].copy()
    return output
    # END OF FUNCTION.

def viz_gauss_pyramid(pyramid):
  """ This function creates a single image out of the given pyramid.
  """
  height = pyramid[0].shape[0]
  width = pyramid[0].shape[1]

  out = np.zeros((height*len(pyramid), width), dtype = float)

  for idx, layer in enumerate(pyramid):
    if layer.max() <= 1:
      layer = layer.copy() * 255

    out[(idx*height):((idx+1)*height),:] = cv2.resize(layer, (width, height),
        interpolation = 3)

  return out.astype(np.uint8)

def viz_lapl_pyramid(pyramid):
  """ This function creates a single image out of the given pyramid.
  """
  height = pyramid[0].shape[0]
  width = pyramid[0].shape[1]

  out = np.zeros((height*len(pyramid), width), dtype = np.uint8)

  for idx, layer in enumerate(pyramid[:-1]):
     # We use 3 for interpolation which is cv2.INTER_AREA. Using a value is
     # safer for compatibility issues in different versions of OpenCV.
     patch = cv2.resize(layer, (width, height),
         interpolation = 3).astype(float)
     # scale patch to 0:256 range.
     patch = 128 + 127*patch/(np.abs(patch).max())

     out[(idx*height):((idx+1)*height),:] = patch.astype(np.uint8)

  #special case for the last layer, which is simply the remaining image.
  patch = cv2.resize(pyramid[-1], (width, height),
      interpolation = 3)
  out[((len(pyramid)-1)*height):(len(pyramid)*height),:] = patch

  return out

def run_blend(black_image, white_image, mask):
  """ This function administrates the blending of the two images according to
  mask.

  Assume all images are float dtype, and return a float dtype.
  """

  # Automatically figure out the size
  min_size = min(black_image.shape)
  depth = int(math.floor(math.log(min_size, 2))) - 4 # at least 16x16 at the highest level.

  gauss_pyr_mask = gaussPyramid(mask, depth)
  gauss_pyr_black = gaussPyramid(black_image, depth)
  gauss_pyr_white = gaussPyramid(white_image, depth)


  lapl_pyr_black  = laplPyramid(gauss_pyr_black)
  lapl_pyr_white = laplPyramid(gauss_pyr_white)

  outpyr = blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask)
  outimg = collapse(outpyr)

  outimg[outimg < 0] = 0 # blending sometimes results in slightly out of bound numbers.
  outimg[outimg > 255] = 255
  outimg = outimg.astype(np.uint8)

  return lapl_pyr_black, lapl_pyr_white, gauss_pyr_black, gauss_pyr_white, \
      gauss_pyr_mask, outpyr, outimg


white_img = cv2.imread("images/crop.jpg")
black_img = cv2.imread("images/portrait.jpg")
mask_img = cv2.imread("images/mask.jpg")

black_img = black_img.astype(float)
white_img = white_img.astype(float)
mask_img = mask_img.astype(float) / 255

print "Applying blending."
lapl_pyr_black_layers = []
lapl_pyr_white_layers = []
gauss_pyr_black_layers = []
gauss_pyr_white_layers = []
gauss_pyr_mask_layers = []
out_pyr_layers = []
out_layers = []

for channel in range(3):
  lapl_pyr_black, lapl_pyr_white, gauss_pyr_black, gauss_pyr_white, gauss_pyr_mask,\
      outpyr, outimg = run_blend(black_img[:,:,channel], white_img[:,:,channel], \
                       mask_img[:,:,channel])

  lapl_pyr_black_layers.append(viz_lapl_pyramid(lapl_pyr_black))
  lapl_pyr_white_layers.append(viz_lapl_pyramid(lapl_pyr_white))
  gauss_pyr_black_layers.append(viz_gauss_pyramid(gauss_pyr_black))
  gauss_pyr_white_layers.append(viz_gauss_pyramid(gauss_pyr_white))
  gauss_pyr_mask_layers.append(viz_gauss_pyramid(gauss_pyr_mask))
  out_pyr_layers.append(viz_lapl_pyramid(outpyr))
  out_layers.append(outimg)

lapl_pyr_black_img = cv2.merge(lapl_pyr_black_layers)
lapl_pyr_white_img = cv2.merge(lapl_pyr_white_layers)
gauss_pyr_black_img = cv2.merge(gauss_pyr_black_layers)
gauss_pyr_white_img = cv2.merge(gauss_pyr_white_layers)
gauss_pyr_mask_img = cv2.merge(gauss_pyr_mask_layers)
outpyr = cv2.merge(out_pyr_layers)
outimg = cv2.merge(out_layers)

print "Writing images"
cv2.imwrite("images/blend.jpg",outimg)