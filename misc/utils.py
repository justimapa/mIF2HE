import matplotlib.pyplot as plt
import SimpleITK as sitk # (modified with elastix)
from skimage.transform import resize
import numpy as np
from PIL import Image
from scipy import ndimage
import cv2


import inspect
import logging
import os
import shutil

class bcolors:

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def display_images_with_alpha(alpha, fixed, moving):
    """
    Displays the superimposition of two images.
    Alpha allows one image or the other to be shaded.
    """

    image = (1.0 - alpha)*fixed[:,:] + alpha*moving[:,:]

    plt.figure(figsize=(11,11))
    plt.imshow(sitk.GetArrayViewFromImage(image), cmap='Greys');
    plt.axis('off')
    plt.show()

def display_image(image, size = (5,5), cmap = 'Greys'):
    """
    Displays a single image
    """

    plt.figure(figsize=size)
    plt.imshow(sitk.GetArrayViewFromImage(image), cmap = cmap);
    plt.axis('off')

def display_difference(image1, image2, cmap1 = 'Greys', save = False, location = 'difference.tif'):
    """
    Displays the subtraction of one image by the other.
    """

    image = image1-image2

    if save:
        write_image(image = image,location=location, bitformat=sitk.sitkInt16)

    plt.figure(figsize=(11,11))
    plt.imshow(sitk.GetArrayViewFromImage(image), cmap = cmap1);
    plt.axis('off')
    plt.show()

def checkerboard(image1, image2, size=(11,11), save = False, location = 'checkerboard.tiff'):
    """
    Displays the checkerboard of two images
    """

    checker = sitk.CheckerBoard(image1, image2, [4,4,4])

    if save:
        writer = sitk.ImageFileWriter()
        writer.SetFileName(location)
        writer.Execute(checker)

    display_image(checker, size = size)
    
    return checker

    

def display_normalized_difference(image1, image2, cmap1 = 'Greys', save = False, location = 'difference.tif'):
    """
    Displays the subtraction of one normalized image by the other.
    """

    image = sitk.Normalize(image1)-sitk.Normalize(image2)

    if save:
        write_image(image = image,location=location, bitformat=sitk.sitkInt16)

    plt.figure(figsize=(11,11))
    plt.imshow(sitk.GetArrayViewFromImage(image), cmap = cmap1);
    plt.axis('off')
    plt.show()


def write_image(image, location, bitformat = sitk.sitkUInt8):
    """
    save image in original format (Unsigned 8 bit int)
    """

    sitk.WriteImage(sitk.Cast(image,bitformat), location)   


def get_diviseur_commun(big_num,small_num):
    """
    Gets the diviseur commun
    big_num: a big integer
    small_num: a smaller integer
    """

    assert big_num >= small_num, 'ERROR, your big number is smaller than your small number'

    value = 0
    increment_sup = small_num
    increment_inf = small_num

    while value == 0:

        if big_num%increment_inf == 0:

            value = increment_inf

        increment_inf -= 1

        if value !=0:
            break

        if big_num%increment_sup == 0:

            value = increment_sup

        increment_sup += 1

    return value

def downscale_pair_image(image1, image2, scaling_factor = 32):
    """
    Downscales a pair of images by a certain factor
    """

    x_scaling_factor_image1 = get_diviseur_commun(image1.shape[0], 32)
    y_scaling_factor_image1 = get_diviseur_commun(image1.shape[1], 32)

    x_scaling_factor_image2 = get_diviseur_commun(image2.shape[0], 32)
    y_scaling_factor_image2 = get_diviseur_commun(image2.shape[1], 32)

    resized_image1 = resize(image1, (image1.shape[0] // x_scaling_factor_image1, image1.shape[1] // y_scaling_factor_image1), anti_aliasing=True)
    resized_image2 = resize(image2, (image2.shape[0] // x_scaling_factor_image2, image2.shape[1] // y_scaling_factor_image2), anti_aliasing=True)

    return sitk.GetImageFromArray(resized_image1), sitk.GetImageFromArray(resized_image2), x_scaling_factor_image2, y_scaling_factor_image2

def pre_register(HE, IF):
    """
    Pre-register a pair of images
    """

    print(bcolors.OKBLUE + "Downscaling Images..." + bcolors.ENDC)
    resized_HE, resized_IF, x_scaling_factor_IF, y_scaling_factor_IF = downscale_pair_image(HE[:,:,0], IF[:,:,0])
    print(bcolors.OKGREEN + "done" + bcolors.ENDC)

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(resized_HE)
    elastixImageFilter.SetMovingImage(resized_IF)

    parameterMapVector = sitk.VectorOfParameterMap()

    parameterMapVector.append(sitk.GetDefaultParameterMap("translation"))
    
    elastixImageFilter.SetParameterMap(parameterMapVector)

    print(bcolors.OKBLUE + "Registering resized image" + bcolors.ENDC)
    elastixImageFilter.Execute()
    print(bcolors.OKGREEN + "done" + bcolors.ENDC)

    return elastixImageFilter, x_scaling_factor_IF, y_scaling_factor_IF

def image_rescale(IF_ITK, elastixImageFilter, x_scaling_factor, y_scaling_factor):
    """
    Pre-registers the original IF image with rescaled parameters
    """

    print("Rescaling original IF Image")

    parameterMap = elastixImageFilter.GetTransformParameterMap()

    transform_parameters = parameterMap[0]["TransformParameters"]
    size_parameters = parameterMap[0]["Size"]

    rescaled_parameters = (str(float(transform_parameters[0])*y_scaling_factor), str(float(transform_parameters[1])*x_scaling_factor))
    rescaled_size = (str(int(float(size_parameters[0])*y_scaling_factor)), str((int(float(size_parameters[1])*x_scaling_factor))))
    
    print(rescaled_size)
    print(IF_ITK.GetSize())
    
    parameterMap[0]["TransformParameters"] = rescaled_parameters
    parameterMap[0]["Size"] = rescaled_size

    transformix = sitk.TransformixImageFilter()
    transformix.SetTransformParameterMap(parameterMap)
    
    transformix.SetMovingImage(IF_ITK)
    transformix.Execute()

    return transformix.GetResultImage()

def tiling(image, x_pixels, y_pixels):
    """
    Tiles a given image with the given dimensions.
    image: image
    x_pixels: wanted tile size along x axis
    y_pixels: wanted tile size along y axis
    """

    print(bcolors.OKBLUE + "Performing multichannel tiling..." + bcolors.ENDC)

    frames = []

    size_x, size_y = image.GetSize()

    x_window = get_diviseur_commun(size_x, x_pixels)
    y_window = get_diviseur_commun(size_y, y_pixels)

    print("x window size: {a}\nywindow size: {b}".format(a=x_window, b=y_window))

    num_x_tiles = int(size_x/x_window)
    num_y_tiles = int(size_y/y_window)

    print("number of tiles along x axis: {a}\nnumber of tiles along y axis: {b}".format(a=num_x_tiles, b=num_y_tiles))

    x_coordinates = [0]
    y_coordinates = [0]

    #if save : mkdir ("./tiles")

    for index in np.arange(0, num_x_tiles, 1):

        x_coordinates.append(x_window*(index+1))

    
    for index in np.arange(0, num_y_tiles ,1):

        y_coordinates.append(y_window*(index+1))
        

    for x_index in np.arange(0, num_x_tiles, 1):

        for y_index in np.arange(0, num_y_tiles, 1):
            
            im = image[x_coordinates[x_index]:x_coordinates[x_index+1],y_coordinates[y_index]:y_coordinates[y_index+1]]
            #if save : cv2.imwrite(f"./tiles/{x_index}-{y_index}.png",cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            frames.append(im)

    print(bcolors.OKGREEN + "done" + bcolors.ENDC)

    return frames

def translate_arrays(array_image_fixed, array_image_moving, method = ['translation','affine', 'bspline'], weight=1, regis = False, num_iter = 256, number_resolution = 4):
    """
    Translate arrays of image
    array_image_fixed: array of fixed image
    array_image_moving: array of moving images
    method: array of wanted methods. Default contains translation
    """

    print(bcolors.OKBLUE + "performing whole slide image registration..." + bcolors.ENDC )

    elastixImageFilter = sitk.ElastixImageFilter()
    parameterMapVector = sitk.VectorOfParameterMap()

        
    if 'translation' in method:
        
        # sets all the paramters for a translation transform 
        
        p = sitk.GetDefaultParameterMap('translation')
        p["MaximumNumberOfIterations"] = [str(float(num_iter))]
        p["MovingImagePyramidSchedule"] = [str(8), str(8), str(4), str(4), str(2), str(2), str(1), str(1)]
        p["NumberOfResolutions"]=[str(float(4))]
        parameterMapVector.append(p) 
        
    if 'rigid':
        
        # sets all the paramters for a rigid transform
        
        p = sitk.GetDefaultParameterMap('rigid')
        p["MaximumNumberOfIterations"] = [str(float(num_iter))]
        p["MovingImagePyramidSchedule"] = [str(8), str(8), str(4), str(4), str(2), str(2), str(1), str(1)]
        p["NumberOfResolutions"]=[str(float(4))]
        parameterMapVector.append(p)
        
    if 'affine' in method:
        
        # sets all the paramters for an affine transform 
        
        p = sitk.GetDefaultParameterMap('affine')
        p["MaximumNumberOfIterations"] = [str(float(num_iter))]
        p["MovingImagePyramidSchedule"] = [str(8), str(8), str(4), str(4), str(2), str(2), str(1), str(1)]
        p["NumberOfResolutions"]=[str(float(4))]
        parameterMapVector.append(p)
        
        
    if 'bspline' in method:
        
        # sets all the paramters for a bspline transform 
        
        p = sitk.GetDefaultParameterMap('bspline')
        p["GridSpacingSchedule"] = [str(float(5)), str(float(4)), str(float(3)),str(float(2)),str(float(1))]
        p["Metric1Weight"] = [str(float(weight))]
        p["MaximumNumberOfIterations"] = [str(float(num_iter))]
        p["MovingImagePyramidSchedule"] = [str(16), str(16), str(8), str(8), str(4),str(4),str(2),str(2),str(1),str(1)]
        p["NumberOfResolutions"]=[str(float(number_resolution))]
        parameterMapVector.append(p)

    elastixImageFilter.SetParameterMap(parameterMapVector)
    
    regis_array = []
    param_maps = []
    grids = []


    for index, (image_fixed, image_moving) in enumerate(zip(array_image_fixed, array_image_moving)):

        print('registering frame {a}/{b}'.format(a=index+1,b=len(array_image_fixed)))

        elastixImageFilter.SetFixedImage(image_fixed)
        elastixImageFilter.SetMovingImage(image_moving)

        try:
            elastixImageFilter.Execute();

        except:
            print("Registration {a} failed".format(a=index+1))

        # saves each parameter map for each tile
        param_maps.append(elastixImageFilter.GetTransformParameterMap())
        
        if regis:
            
            #saves each registered image for each tile
            
            regis_array.append(elastixImageFilter.GetResultImage())
        

    print(bcolors.OKGREEN + "done" + bcolors.ENDC)
    
    if regis:
        return regis_array, param_maps
    
    else:
        return param_maps


def transform_array(array_image, array_parameter):
    """
    Transforms an array of images given an array of parameter maps
    """
    
    result = []
    
    transform = sitk.TransformixImageFilter()
    
    for image, parameterMap in zip(array_image,array_parameter):
    
        transform.SetTransformParameterMap(parameterMap)
        transform.SetMovingImage(image)
        transform.Execute()
        
        result.append(transform.GetResultImage())
    
    return result

def get_concat_h(im1, im2):
    """
    concatenates horizontally two images
    """
    
    dst = Image.new('F', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    """
    concatenates vertically two images
    """

    im2_copy = Image.fromarray(im2)
    dst = Image.new('F', (im1.width, im1.height + im2_copy.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2_copy, (0, im1.height))
    return dst

def reconstruct_image(size, frames):
    """
    Reconstructs the image from the tiles with it's original size
    """

    frames_copy = frames.copy()

    x_patch_size = frames[0].GetSize()[0]
    y_patch_size = frames[0].GetSize()[1]

    x_size = size[0]
    y_size = size[1]

    num_x_intervals = int(x_size / x_patch_size)
    num_y_intervals = int(y_size / y_patch_size)

    print("Number of x intervals: {}".format(num_x_intervals))
    print("Number of y intervals: {}".format(num_y_intervals))
    

    assert (num_x_intervals * num_y_intervals) == len(frames), 'ERROR : not the same number between frames and iterations'

    for x in np.arange(0,num_x_intervals,1):

        column = Image.fromarray(sitk.GetArrayViewFromImage(frames_copy.pop(0)))

        for y in np.arange(0,num_y_intervals-1,1):

            im1 = sitk.GetArrayViewFromImage(frames_copy.pop(0))
            column = get_concat_v(column,im1)

        if x == 0:
            image = column
        else:
            image = get_concat_h(image,column)


    return np.asarray(image)

def draw_point(image,x,y, size=1): 
    """
    draws a square of a given size on an empty image
    """
    
    for element in np.arange(-size+1,size-1,1):
        
        for element2 in np.arange(-size+1, size-1,1):
            
            image.SetPixel(int(x+element),int(y+element2),255)     



def extract_patch(image, x, y, patch_size):
    """
    Extract a patch at a certain position (x,y) of an image, with a certain patch size
    """
    
    patch = np.asarray(image.read_region((x-int(patch_size/2),y-int(patch_size/2)),0, (patch_size, patch_size)))[:,:,0:3]
    
    return patch

def draw_circle(image, x, y, pixel_value = (0,255,0), radius = 10, thickness = 2):
    """
    Draws a circle around a certain position (x,y) of an image
    """
    
    cv2.circle(image, (x,y), radius, pixel_value, thickness)

def is_nucleus(image, x, y, threshold, radius = 10):
    """
    Assess if a nucleaus is present
    """
    
    patch = extract_patch(image,x,y,patch_size=radius)
    pixel_values = patch.flatten()
        
    mean = np.mean(pixel_values)
    
    if mean > threshold: 
        return False
    
    else:
        return True



####
def normalize(mask, dtype=np.uint8):
    return (255 * mask / np.amax(mask)).astype(dtype)


####
def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


####
def cropping_center(x, crop_shape, batch=False):
    """Crop an input image at the centre.

    Args:
        x: input array
        crop_shape: dimensions of cropped array
    
    Returns:
        x: cropped array
    
    """
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    return x


####
def rm_n_mkdir(dir_path):
    """Remove and make directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


####
def mkdir(dir_path):
    """Make directory."""
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


####
def get_inst_centroid(inst_map):
    """Get instance centroids given an input instance map.

    Args:
        inst_map: input instance map
    
    Returns:
        array of centroids
    
    """
    inst_centroid_list = []
    inst_id_list = list(np.unique(inst_map))
    for inst_id in inst_id_list[1:]:  # avoid 0 i.e background
        mask = np.array(inst_map == inst_id, np.uint8)
        inst_moment = cv2.moments(mask)
        inst_centroid = [
            (inst_moment["m10"] / inst_moment["m00"]),
            (inst_moment["m01"] / inst_moment["m00"]),
        ]
        inst_centroid_list.append(inst_centroid)
    return np.array(inst_centroid_list)


####
def center_pad_to_shape(img, size, cval=255):
    """Pad input image."""
    # rounding down, add 1
    pad_h = size[0] - img.shape[0]
    pad_w = size[1] - img.shape[1]
    pad_h = (pad_h // 2, pad_h - pad_h // 2)
    pad_w = (pad_w // 2, pad_w - pad_w // 2)
    if len(img.shape) == 2:
        pad_shape = (pad_h, pad_w)
    else:
        pad_shape = (pad_h, pad_w, (0, 0))
    img = np.pad(img, pad_shape, "constant", constant_values=cval)
    return img


####
def color_deconvolution(rgb, stain_mat):
    """Apply colour deconvolution."""
    log255 = np.log(255)  # to base 10, not base e
    rgb_float = rgb.astype(np.float64)
    log_rgb = -((255.0 * np.log((rgb_float + 1) / 255.0)) / log255)
    output = np.exp(-(log_rgb @ stain_mat - 255.0) * log255 / 255.0)
    output[output > 255] = 255
    output = np.floor(output + 0.5).astype("uint8")
    return output


####
def log_debug(msg):
    frame, filename, line_number, function_name, lines, index = inspect.getouterframes(
        inspect.currentframe()
    )[1]
    line = lines[0]
    indentation_level = line.find(line.lstrip())
    logging.debug("{i} {m}".format(i="." * indentation_level, m=msg))


####
def log_info(msg):
    frame, filename, line_number, function_name, lines, index = inspect.getouterframes(
        inspect.currentframe()
    )[1]
    line = lines[0]
    indentation_level = line.find(line.lstrip())
    logging.info("{i} {m}".format(i="." * indentation_level, m=msg))


def remove_small_objects(pred, min_size=64, connectivity=1):
    """Remove connected components smaller than the specified size.

    This function is taken from skimage.morphology.remove_small_objects, but the warning
    is removed when a single label is provided. 

    Args:
        pred: input labelled array
        min_size: minimum size of instance in output array
        connectivity: The connectivity defining the neighborhood of a pixel. 
    
    Returns:
        out: output array with instances removed under min_size

    """
    out = pred

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndimage.generate_binary_structure(pred.ndim, connectivity)
        ccs = np.zeros_like(pred, dtype=np.int32)
        ndimage.label(pred, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out
