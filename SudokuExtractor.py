import numpy as np
import cv2
from imutils import contours
import matplotlib.pyplot as plt
import operator
def show_image(img):
    '''Shows an image until any key is pressed'''
    #print(type(img))
    #print(img.shape)
    #cv2.imshow('image',img)
    #cv2.imwrite('images/gau_sudoku3.jpg',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return img

def show_digits(digits,color=255):
    '''Show list of 81 extracted digits in a grid format'''
    rows=[]
    with_border = [cv2.copyMakeBorder(img.copy(),1,1,1,1,cv2.BORDER_CONSTANT,None,color)for img in digits]
    for i in range(9):
        row= np.concatenate(with_border[i*9:((i+1)*9)],axis=1)
        rows.append(row)
    img=show_image(np.concatenate(rows))
    return img

def pre_process_img(image,skip_dilate=False):
    '''
    using a blur function,adaptive thresholding, dilation to expose the main features of the image
    :param image:
    :return the preprocessed image:
    '''
    proc = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    proc = cv2.GaussianBlur(proc,(9,9),0)
    proc = cv2.adaptiveThreshold(proc,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    proc=cv2.bitwise_not(proc,proc)
    if not skip_dilate:
        kernel = np.array([[0.,1.,0.],[1.,1.,1.],[0.,1.,0.]],np.uint8)
        proc = cv2.dilate(proc,kernel)
    return proc

def find_corners_of_largest_polygon(img):
    '''
    find the extreme corners of the largest polygon in the image.
    the largest polgon is nothing but the sudoku grid

    :param processed:
    :return list containing 4 corners of the sudoku:
    '''
    opencv_version = cv2.__version__.split('.')[0]
    if opencv_version == '3':
        _,contours,h= cv2.findContours(img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, h = cv2.findContours(img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # sorting the contours on the basis of area
    contours = sorted(contours, key = cv2.contourArea, reverse=True)
    polygon = contours[0]
    # use of operator.itemgetter with min and max function helps us to get the index of the point
    # bottom-right = largest value of(x+y)
    # bottom-left = smallest value of(x-y)
    # top-left = smallest value of (x+y)
    # top-right = largest value of (x-y)
    top_left,_ = min(enumerate(pt[0][0]+pt[0][1] for pt in polygon),key = operator.itemgetter(1))
    top_right,_ = max(enumerate(pt[0][0]- pt[0][1] for pt in polygon),key= operator.itemgetter(1))
    bottom_left,_ = min(enumerate(pt[0][0] - pt[0][1] for pt in polygon), key=operator.itemgetter(1))
    bottom_right,_ = max(enumerate(pt[0][0] + pt[0][1] for pt in polygon), key=operator.itemgetter(1))

    return [polygon[top_left][0],polygon[top_right][0],polygon[bottom_right][0],polygon[bottom_left][0]]

def distance_between(p1,p2):
    '''
    this calulates the distance between 2 points
    :param a:
    :param b:
    :return: distance
    '''
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a**2) + (b**2))

def crop_and_wrap(img,crop_rect):
    '''
    crops and wraps the section of image from an image into square of smaller size
    :param img:
    :param crop_rect:
    :return croppedimage, original image and points:
    '''
    top_left,top_right,bottom_right,bottom_left = crop_rect[0],crop_rect[1],crop_rect[2],crop_rect[3]
    src = np.array([top_left,top_right,bottom_right,bottom_left],dtype='float32')
    side = max([
        distance_between(top_left, top_right),
        distance_between(top_left,bottom_left),
        distance_between(bottom_left,bottom_right),
        distance_between(bottom_right,top_right)
    ])

    # describing a new square that we want to extract from the image
    dst= np.array([[0,0],[side-1,0],[side-1,side-1],[0,side-1]],dtype = 'float32')
    # getting the transformation matrix
    m = cv2.getPerspectiveTransform(src,dst)
    return cv2.warpPerspective(img, m,(int(side),int(side)))


def cut_from_rect(img,rect):
    return img[int(rect[0][1]):int(rect[1][1]),int(rect[0][0]):int(rect[1][0])]

def find_largest_features(inp_img, scan_tl=None,scan_br= None):
    '''
    Uses the fact that floodfill function returns a bounding box of the area, it fills to find the biggest
    connected pixel structure in the image......fill this structure in white and reducing the remaining to black
    :param inp_img:
    :param scan_tl:
    :param scan_b2:
    :return:
    '''

    img = inp_img.copy()
    height,width = img.shape[:2]
    maxarea =0
    seedPoint =(None,None)
    if scan_tl is None:
        scan_tl=[0,0]
    if scan_br is None:
        scan_br=[width,height]

    # looping through the image
    for x in range(scan_tl[0],scan_br[0]):
        for y in range(scan_tl[1],scan_br[1]):
            if img.item(y,x)==255 and x<width and y<height:
                # only operate on ligth and white spaces
                area = cv2.floodFill(img,None,(x,y),64)
                if( area[0]> maxarea):
                    maxarea= area[0]
                    seedPoint=(x,y)
    # coloring everything grey
    for x in range(width):
        for y in range(height):
            if(img.item(y,x)==255 and x<width and y<height):
                cv2.floodFill(img,None,(x,y),64)

    # masks that are 2 pixels bigger than the image
    mask = np.zeros((height+2,width+2),np.uint8)

    # highlighting the main feature
    if all([p is not None for p in seedPoint]):
        cv2.floodFill(img,mask,seedPoint,255)

    top,bottom,left,right = height, 0, width,0
    for x in range(width):
        for y in range(height):
            if img.item(y,x)==64:  # hide anything that is not important
                cv2.floodFill(img,mask,(x,y),0)
            # finding the bounding parameters
            if img.item(y,x) == 255:
                top = y if y < top else top
                bottom = y if y>bottom else bottom
                left = x if x<left else left
                right = x if x>right else right

    bbox = [[left,top],[right,bottom]]
    return img,np.array(bbox,dtype= 'float32'),seedPoint

def scale_and_centre(img, size, margin=0,background=0):
    """
    Scale and centre of an image onto a new background image
    :param margin:
    :param background:
    :return:
    """
    h,w = img.shape[:2]

    def centre_pad(length):
        '''handles centreing for a given length that may be odd or even'''
        if length%2==0:
            side1 = int((size-length)/2)
            side2= side1
        else:
            side1 = int((size-length)/2)
            side2= side1+1
        return side1, side2

    def scale(r,x):
        return int(r*x)
    if h>w:
        t_pad= int (margin/2)
        b_pad = t_pad
        ratio = (size-margin)/h
        w,h = scale(ratio,w),scale(ratio,h)
        l_pad, r_pad =centre_pad(w)
    else:
        l_pad = int(margin/2)
        r_pad = l_pad
        ratio = (size-margin)/w
        w,h = scale(ratio,w),scale(ratio,h)
        t_pad,b_pad= centre_pad(h)

    img = cv2.resize(img,(w,h))
    img = cv2.copyMakeBorder(img,t_pad,b_pad,l_pad,r_pad,cv2.BORDER_CONSTANT,None,background)
    return cv2.resize(img,(size,size))

def extract_digits(img,rect,size):
    '''
    Extract a digit from a sudoku square if any exist
    :param img:
    :param rect:
    :param size:
    :return:
    '''
    digit = cut_from_rect(img,rect)
    h,w = digit.shape[:2]
    margin = int(np.mean([h,w])/2.5)
    _,bbox,seed = find_largest_features(digit,[margin,margin],[w-margin,h-margin])
    digit = cut_from_rect(digit,bbox)
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]
    # Ignore any small bounding boxes
    if w>0 and h>0 and (w*h)>100 and len(digit) > 0:
        return scale_and_centre(digit,size,4)
    else:
        return np.zeros((size,size),np.uint8)

def infer_grid(img):
    squares=[]
    side = img.shape[:1]
    side = side[0]/9
    for j in range(9):
        for i in range(9):
            p1=(i*side,j*side)
            p2= ((i+1)*side,(j+1)*side)
            squares.append((p1,p2))
    return squares

def get_digits(img,squares,size):
    digits = []
    img = pre_process_img(img.copy(),skip_dilate=True)
    for square in squares:
        digits.append(extract_digits(img,square,size))
    return digits


def parse_grid(image):
    original = image.copy()
    processed = pre_process_img(image)
    corners = find_corners_of_largest_polygon(processed)
    cropped = crop_and_wrap(original,corners)
    squares = infer_grid(cropped)
    digits =get_digits(cropped,squares,28)
    final_image = show_digits(digits)
    return final_image

def extract_sudoku(image):
    final_image = parse_grid(image)
    return final_image
