import numpy as np
import pydicom as dicom
import os
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
import pylibjpeg
import PIL
import nrrd
import cv2

def load_scan(path):
    """Takes a path with DICOMs and returns slices
    """
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.SliceLocation), reverse=True) #Z increases in value towards the head/cranially
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    #if 'reverse' in path:
        #slices = np.flip(slices)
    for s in slices:
        s.SliceThickness = slice_thickness

        return slices

def load_nrrd(path):
    data, header = nrrd.read(path,index_order='C') #z,y,x?
    return data

def get_pixels_hu(scans):
    """Takes slices and converts to numpy pixel values, returns houndsfield units (numpy array int16)
    """
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def sample_stack(stack, rows=10, cols=10, start_with=1, show_every=2):
    """Displays a stack of images (output of get_pixels_hu)
    """
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()

def center_body_fov(stack,crop_size=256,slice_to_seg=1,show_mask=False):
    """Takes slice stack, finds body with connectedcomponent (largest cluster), crops 256x256 around body centroid.
    crop_size: must be divisible by 2. Default 256
    slice_to_seg: slice to use to find body centroid. Usually 1st (top) slice works great.
    """
    binary_image = np.array((stack>-320), dtype=np.int8)
    index_seg = binary_image[slice_to_seg,:,:]
    vectorized = np.reshape(index_seg,[np.prod(index_seg.shape),1])
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    K=6
    attempts=10
    ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_RANDOM_CENTERS)
    label = label.flatten()
    center = np.uint8(center)
    res = center[label.flatten()]
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(index_seg)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    max_centroid = centroids[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
            max_centroid = centroids[i]
    img2 = np.zeros(output.shape)
    img2[output == max_label] = 1
    if show_mask:
        plt.imshow(img2)
    UL = (round(max(0,max_centroid[0]-crop_size/2)),round(max(0,max_centroid[1]-crop_size/2)))
    UR = (round(UL[0]+crop_size),round(UL[1]))
    LL = (round(UL[0]),round(UL[1]+crop_size))
    LR = (round(UL[0]+crop_size),round(UL[1]+crop_size))
    x1 = UL[0]
    x2 = UR[0]
    y1 = UL[1]
    y2 = LL[1]
    yx = [y1,y2,x1,x2]
    cropped_stack = stack[:,y1:y2,x1:x2] #apparently [z,y,x]
    return cropped_stack, x1, x2, y1, y2

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + [scan[0].PixelSpacing[0]] + [scan[0].PixelSpacing[1]]))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing

def do_preprocessing(parent_dir,save_dirs):
    parent_dirs = os.listdir(parent_dir)
    img_dirs = [i for i in parent_dirs if "UCSF" in i]
    for img_dir in img_dirs:
        print("Processing " + img_dir)
        cur_img_path = os.path.join(parent_dir, img_dir)
        _, dirs, _ = os.walk(cur_img_path)
        cur_dcm_path = dirs[0]+'/'+dirs[1][0]
        dicom_stack = load_scan(cur_dcm_path)
        pixels_hu_stack = get_pixels_hu(dicom_stack)
        resampled_pixels, new_spacing = resample(pixels_hu_stack, dicom_stack)
        print("Shape before resampling\t", pixels_hu_stack.shape)
        print("Shape after resampling\t", resampled_pixels.shape)
        resampled_cropped_pixels, x1, x2, y1, y2 = center_body_fov(resampled_pixels,show_mask=True)
        nrrdpath = glob(cur_img_path+'/*.nrrd',recursive=True)
        target = load_nrrd(nrrdpath[0])
        target = np.flip(target,0) #flip masks in z axis -- needs to be head to toe
        resampled_target,_ = resample(target, dicom_stack)
        resampled_cropped_target = resampled_target[:,y1:y2,x1:x2]
        print("Saving stacks")
        np.save(save_dirs['fullimage_dir'] + "/fullimages_%s.npy" % (img_dir), pixels_hu_stack)
        np.save(save_dirs['croppedimage_dir'] + "/resampled_crop_%s.npy" % (img_dir), resampled_cropped_pixels)
        np.save(save_dirs['fulltarget_dir'] + "/target_%s.npy" % (img_dir), target)
        np.save(save_dirs['croppedtarget_dir'] + "/resampled_crop_target_%s.npy" % (img_dir), resampled_cropped_target)
    return

parent_dir = "E:/CT_scans/"
save_dirs = {'fullimage_dir':"E:/CT_scans/Out/Full_images/", \
'croppedimage_dir':"E:/CT_scans/Out/Cropped_images/", \
'fulltarget_dir':"E:/CT_scans/Out/Full_targets/", \
'croppedtarget_dir':"E:/CT_scans/Out/Cropped_targets/"}
do_preprocessing(parent_dir,save_dirs)
