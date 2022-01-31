from utils import *
import numpy as np


def warpPerspective(img, transform_matrix, output_width, output_height):
    """
    TODO : find warp perspective of image_matrix and return it
    :return a (width x height) warped image
    """
    transformed_coordinate = np.array((3,))
    init_coordinate = np.array((3,))
    result = np.zeros_like(img)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            init_coordinate = [x, y, 1]
            transformed_coordinate = np.dot(transform_matrix, init_coordinate)
            transformed_coordinate_2d = [int(transformed_coordinate[0] /transformed_coordinate[2]), int(transformed_coordinate[1] /transformed_coordinate[2])]
            if(transformed_coordinate_2d[0] < output_width and transformed_coordinate_2d[1] < output_height):
                result[transformed_coordinate_2d[0], transformed_coordinate_2d[1], :] = img[x, y, :]
    return result[:output_width, :output_height, :]


def grayScaledFilter(img):
    """
    TODO : Complete this part based on the description in the manual!
    """
    transfer_matrix = np.array([[0.299, 0.587, 0.114], [0.299, 0.587, 0.114], [0.299, 0.587, 0.114]])
    return Filter(img, transfer_matrix)


def crazyFilter(img):
    """
    TODO : Complete this part based on the description in the manual!
    """
    transfer_matrix = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 0]])
    return Filter(img, transfer_matrix)


def customFilter(img):
    """
    TODO : Complete this part based on the description in the manual!
    """
    transfer_matrix = np.array([[1, 3, 0], [0, 1, 1], [0.5 , 0, 2]])
    filtered = Filter(img, transfer_matrix)
    showImage(filtered, title="filtered image")
    
    inverese_matrix = np.linalg.inv(transfer_matrix)
    
    undo_filtered = Filter(filtered, inverese_matrix)
    showImage(undo_filtered, title="undo filter image")


def scaleImg(img, scale_width, scale_height):
    """
    TODO : Complete this part based on the description in the manual!
    """
    new_width = img.shape[0] * scale_width
    new_hieght = img.shape[1] * scale_height
    result = np.zeros((new_width, new_hieght, 3))
    for x in range(new_width):
        for y in range(new_hieght):
            new_x = int(x * img.shape[0] / new_width)
            new_y = int(y * img.shape[1] / new_hieght)
            result[x, y, :] = img[new_x, new_y, :]
    
    return result


def cropImg(img, start_row, end_row, start_column, end_column):
    """
    TODO : Complete this part based on the description in the manual!
    """
    return img[start_column:end_column, start_row:end_row, :]


if __name__ == "__main__":
    image_matrix = get_input('pic.jpg')

    # You can change width and height if you want
    width, height = 300, 400

    # TODO : Find coordinates of four corners of your inner Image ( X,Y format)
    #  Order of coordinates: Upper Left, Upper Right, Down Left, Down Right
    pts1 = np.float32([[105, 217], [378, 180], [160, 644], [492, 572]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    m = getPerspectiveTransform(pts1, pts2)

    warpedImage = warpPerspective(image_matrix, m, width, height)
    showWarpPerspective(warpedImage)

    grayScalePic = grayScaledFilter(warpedImage)
    showImage(grayScalePic, title="Gray Scaled")

    crazyImage = crazyFilter(warpedImage)
    showImage(crazyImage, title="Crazy Filter")

    customFilter(warpedImage)

    croppedImage = cropImg(warpedImage, 50, 300, 50, 225)
    showImage(croppedImage, title="Cropped Image")

    scaledImage = scaleImg(warpedImage, 2, 3)
    showImage(scaledImage, title="Scaled Image")
