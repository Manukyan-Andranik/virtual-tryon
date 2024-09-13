import pdb

import numpy as np
import cv2
from PIL import Image, ImageDraw

label_map = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "head": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
    "scarf": 17,
}

def extend_arm_mask(wrist, elbow, scale):
  wrist = elbow + scale * (wrist - elbow)
  return wrist

def hole_fill(img):
    img = np.pad(img[1:-1, 1:-1], pad_width = 1, mode = 'constant', constant_values=0)
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)

    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst

def refine_mask(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area = []
    for j in range(len(contours)):
        a_d = cv2.contourArea(contours[j], True)
        area.append(abs(a_d))
    refine_mask = np.zeros_like(mask).astype(np.uint8)
    if len(area) != 0:
        i = area.index(max(area))
        cv2.drawContours(refine_mask, contours, i, color=255, thickness=-1)

    return refine_mask


def far_left_right(image:Image.Image, top_bottom):
    # Convert the PIL Image to a NumPy array
    image_np = np.array(image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Invert the grayscale image to have black contours on a white background
    inverted_gray = cv2.bitwise_not(gray)

    # Find contours in the inverted image
    contours, _ = cv2.findContours(inverted_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store the far-left and far-right points
    far_left_point = None
    far_right_point = None

    print("top_bottom:", top_bottom)
    # Iterate through contours to find the far-left and far-right points among points with Y0 location
    for contour in contours:
        # Filter points with Y-coordinate of 0
        y0_points = [point[0] for point in contour if top_bottom - 70 <= point[0][1] <= top_bottom + 70]
        
        # If there are no points with Y-coordinate of 0 in the current contour, continue to the next contour
        if not y0_points:
            continue
        
        # Find the far-left and far-right points among the Y0 points
        min_x = min(y0_points, key=lambda p: p[0])
        max_x = max(y0_points, key=lambda p: p[0])
        
        # Update the far-left and far-right points if needed
        if far_left_point is None or min_x[0] < far_left_point[0]:
            far_left_point = min_x
        if far_right_point is None or max_x[0] > far_right_point[0]:
            far_right_point = max_x

    # Draw circles at the far-left and far-right points on the image
    image_with_points = cv2.circle(image_np.copy(), tuple(far_left_point), 5, (0, 0, 255), -1)  # Red circle at far-left point
    image_with_points = cv2.circle(image_with_points, tuple(far_right_point), 5, (0, 255, 0), -1)  # Green circle at far-right point

    img_save = Image.fromarray(image_with_points)
    return img_save, far_left_point, far_right_point


def get_highestPoint(image:Image.Image):
    # Read the image
    input_image = image

    # Convert the PIL Image to a NumPy array
    image_np = np.array(input_image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Invert the grayscale image
    inverted_gray = cv2.bitwise_not(gray)

    # Find contours in the inverted image
    contours, _ = cv2.findContours(inverted_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store the highest y-coordinate and its corresponding contour
    max_y = 0
    highest_contour = None

    # Iterate through contours to find the one with the highest y-coordinate
    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Update max_y and highest_contour if the current contour's y-coordinate is higher
        if y > max_y:
            max_y = y
            highest_contour = contour

    # If a contour with the highest y-coordinate is found, get its topmost point
    if highest_contour is not None:
        # Find the topmost point of the contour
        topmost = tuple(highest_contour[highest_contour[:, :, 1].argmin()][0])
        #print("Topmost point location on the black contour:", topmost)

        threshold_y = topmost

        return threshold_y

def get_lowestPoint(image: Image.Image):
    # Convert the PIL Image to a NumPy array
    image_np = np.array(image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Invert the grayscale image
    inverted_gray = cv2.bitwise_not(gray)

    # Find contours in the inverted image
    contours, _ = cv2.findContours(inverted_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store the lowest y-coordinate and its corresponding contour
    min_y = float('inf')
    lowest_contour = None

    # Iterate through contours to find the one with the lowest y-coordinate
    for contour in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Update min_y and lowest_contour if the current contour's y-coordinate is lower
        if y < min_y:
            min_y = y
            lowest_contour = contour

    # If a contour with the lowest y-coordinate is found, get its lowest point
    if lowest_contour is not None:
        # Find the lowest point of the contour
        lowest_point = tuple(lowest_contour[lowest_contour[:, :, 1].argmax()][0])

        return lowest_point

# Example usage:
# lowest_point = get_lowestPoint(image)
# print("Lowest point location on the black contour:", lowest_point)

def draw_polygon(image, points):
    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Draw lines between consecutive points to form the polygon
    for i in range(len(points)):
        start_point = points[i]
        end_point = points[(i + 1) % len(points)]  # Wrap around to connect the last point with the first point
        draw.line([start_point, end_point], fill='black', width=2)

    # Fill the polygon with black color
    draw.polygon(points, fill='black')

    return image

def fill_above_y_with_white(image: Image.Image, threshold_y: int):
    # Convert the image to numpy array
    img_array = np.array(image)

    # Create a mask to fill the area above the threshold y-value with white color
    mask = np.ones_like(img_array[:, :, 0]) * 255
    mask[:threshold_y, :] = 0  # Set pixels above threshold_y to 0 (black)

    # Apply the mask to the image
    img_array[mask == 0] = [255, 255, 255]  # Fill pixels with black mask with white color

    # Convert the result back to a PIL Image
    result_image = Image.fromarray(img_array)

    return result_image

def fill_below_y_with_white(image: Image.Image, threshold_y: int):
    # Convert the image to numpy array
    img_array = np.array(image)

    # Ensure threshold_y is an integer
    if not isinstance(threshold_y, int):
        raise ValueError("threshold_y must be an integer")

    # Create a mask to fill the area below the threshold y-value with black color
    mask = np.ones_like(img_array[:, :, 0]) * 255
    mask[threshold_y:, :] = 0  # Set pixels below threshold_y to 0 (black)

    # Apply the mask to the image
    img_array[mask == 0] = [255, 255, 255]  # Fill pixels with black mask with white color

    # Convert the result back to a PIL Image
    result_image = Image.fromarray(img_array)

    return result_image

def get_mask_location(model_type, category, model_parse: Image.Image, keypoint: dict, width=384,height=512):
    # Load pose points
    pose_data = keypoint["pose_keypoints_2d"]
    pose_data = np.array(pose_data)
    pose_data = pose_data.reshape((-1, 2))

    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)

    '''
    parse_left_shoe = (parse_array == label_map["left_shoe"]).astype(np.float32)            
    inpaint_mask = 1 - parse_left_shoe
    img = np.where(inpaint_mask, 255, 0)

    # Convert img to uint8 array
    img_uint8 = img.astype(np.uint8)

    # Create the PIL Image
    img_save = Image.fromarray(img_uint8, mode='L')  # mode='L' for grayscale image

    # Convert to RGB if needed
    img_save = img_save.convert('RGB')

    img_save.save('./images_output/parse_left_shoe.jpg')
    top_left_shoe = get_highestPoint(img_save)
    top_left_shoe = (top_left_shoe[0], top_left_shoe[1]-10)
    
    print("top_left_shoe:", top_left_shoe)


    parse_right_shoe = (parse_array == label_map["right_shoe"]).astype(np.float32)            
    inpaint_mask = 1 - parse_right_shoe
    img = np.where(inpaint_mask, 255, 0)

    # Convert img to uint8 array
    img_uint8 = img.astype(np.uint8)

    # Create the PIL Image
    img_save = Image.fromarray(img_uint8, mode='L')  # mode='L' for grayscale image

    # Convert to RGB if needed
    img_save = img_save.convert('RGB')

    img_save.save('./images_output/parse_right_shoe.jpg')
    top_right_shoe = get_highestPoint(img_save)
    top_right_shoe = (top_right_shoe[0], top_right_shoe[1]-10)
    '''

    top_left_shoe = tuple(pose_data[13][:2])
    top_right_shoe = tuple(pose_data[10][:2])
    
    print("top_right_shoe:", top_right_shoe)

    lower_mask = np.zeros_like(parse_array)
    dresses_mask = np.zeros_like(parse_array)
    dress_lower_mask = np.zeros_like(parse_array)

    parse_dresses = (parse_array == label_map["dress"]).astype(np.float32)

    if not np.all(parse_dresses == 0):
                         
        inpaint_mask = 1 - parse_dresses
        img = np.where(inpaint_mask, 255, 0)

        # Convert img to uint8 array
        img_uint8 = img.astype(np.uint8)

        # Create the PIL Image
        img_save = Image.fromarray(img_uint8, mode='L')  # mode='L' for grayscale image

        # Convert to RGB if needed
        img_save = img_save.convert('RGB')

        fixed_img_dress_lower = fill_above_y_with_white(img_save, int(tuple(pose_data[7][:2])[1]))
        #parse_mask_total = fill_below_y_with_white(parse_mask_total, tuple(pose_data[7][:2])[1])
        save_img_dress_lower = fixed_img_dress_lower.convert('RGB')

        # Convert the image to a grayscale NumPy array
        img_array = np.array(fixed_img_dress_lower.convert('L'))

        # Threshold the grayscale array to convert it back to a binary array
        dress_lower_mask = (img_array == 255).astype(np.float32)

        # Invert the binary array if needed (optional)
        dress_lower_mask = 1 - dress_lower_mask


        lowest_dresses = get_lowestPoint(img_save)[1]
        img_save_dresses, bottom_left_dresses, bottom_right_dresses = far_left_right(img_save, lowest_dresses)

        bottom_left = (tuple(bottom_left_dresses)[0], top_right_shoe[1])
        bottom_right = (tuple(bottom_right_dresses)[0], top_left_shoe[1])

        # Example usage:
        # Assuming you have a white image 'white_image' and a list of points 'polygon_points'
        white_image = Image.new('RGB', (384, 512), color='white')  # Example white image

        polygon_points = [tuple(bottom_left_dresses)]
        if bottom_left[0] < top_right_shoe[0]:
            polygon_points.append(bottom_left)
            polygon_points.append(top_right_shoe)
        else:
            polygon_points.append(top_right_shoe)
            polygon_points.append(bottom_left)

        if bottom_right[0] > top_left_shoe[0]:
            polygon_points.append(top_left_shoe)
            polygon_points.append(bottom_right)
        else:
            polygon_points.append(bottom_right)
            polygon_points.append(top_left_shoe)
        
        polygon_points.append(tuple(bottom_right_dresses))
        polygon_points.append(tuple(bottom_left_dresses))
    
        #polygon_points = [tuple(top_left), tuple(bottom_left_pants), top_right_shoe, 
            #top_left_shoe, bottom_right, tuple(bottom_right_pants), tuple(top_right), tuple(top_left)]  # Example polygon points
        # Create a drawing object
        draw = ImageDraw.Draw(img_save_dresses)

        # Draw the points on the image
        for point in polygon_points:
            draw.ellipse((point[0]-2, point[1]-2, point[0]+2, point[1]+2), fill='red')  # Draw a small red circle around each point

        # Save the image with the points drawn

        # Draw the polygon on the white image
        img_polygon = draw_polygon(white_image, polygon_points)

        # Convert the image to a grayscale NumPy array
        img_array = np.array(img_polygon.convert('L'))

        # Threshold the grayscale array to convert it back to a binary array
        dresses_mask = (img_array == 255).astype(np.float32)

        # Invert the binary array if needed (optional)
        dresses_mask = 1 - dresses_mask

        # Display the recovered parse_right_shoe array
        print('dresses_mask:', dresses_mask)

    if (model_type == 'dc' and np.all(parse_dresses == 0)):
        parse_pants_skirts = (parse_array == label_map["skirt"]).astype(np.float32) + \
                            (parse_array == label_map["pants"]).astype(np.float32)                  
        inpaint_mask = 1 - parse_pants_skirts
        img = np.where(inpaint_mask, 255, 0)
        # Convert img to uint8 array
        img_uint8 = img.astype(np.uint8)

        # Create the PIL Image
        img_save = Image.fromarray(img_uint8, mode='L')  # mode='L' for grayscale image

        # Convert to RGB if needed
        img_save = img_save.convert('RGB')

        top_pants = get_highestPoint(img_save)[1]
        img_save_top_pants, top_left, top_right = far_left_right(img_save, top_pants)
        print("top letf:", top_left)
        print("top right:", top_right)

        lowest_pants = get_lowestPoint(img_save)[1]
        img_save_lowest_pants, bottom_left_pants, bottom_right_pants = far_left_right(img_save, lowest_pants)

        bottom_left = (tuple(bottom_left_pants)[0], top_right_shoe[1])
        bottom_right = (tuple(bottom_right_pants)[0], top_left_shoe[1])

        # Example usage:
        # Assuming you have a white image 'white_image' and a list of points 'polygon_points'
        white_image = Image.new('RGB', (384, 512), color='white')  # Example white image

        polygon_points = [tuple(top_left), tuple(bottom_left_pants)]
        if bottom_left[0] < top_right_shoe[0]:
            polygon_points.append(bottom_left)
            polygon_points.append(top_right_shoe)
        else:
            polygon_points.append(top_right_shoe)
            polygon_points.append(bottom_left)
        

        if bottom_right[0] > top_left_shoe[0]:
            polygon_points.append(top_left_shoe)
            polygon_points.append(bottom_right)
        else:
            polygon_points.append(bottom_right)
            polygon_points.append(top_left_shoe)

        polygon_points.append(tuple(bottom_right_pants))
        polygon_points.append(tuple(top_right))
        polygon_points.append(tuple(top_left))
        
    
        #polygon_points = [tuple(top_left), tuple(bottom_left_pants), top_right_shoe, 
            #top_left_shoe, bottom_right, tuple(bottom_right_pants), tuple(top_right), tuple(top_left)]  # Example polygon points
        # Create a drawing object
        draw = ImageDraw.Draw(img_save_top_pants)

        # Draw the points on the image
        for point in polygon_points:
            draw.ellipse((point[0]-2, point[1]-2, point[0]+2, point[1]+2), fill='red')  # Draw a small red circle around each point


        # Draw the polygon on the white image
        img_polygon = draw_polygon(white_image, polygon_points)

        # Convert the image to a grayscale NumPy array
        img_array = np.array(img_polygon.convert('L'))

        # Threshold the grayscale array to convert it back to a binary array
        lower_mask = (img_array == 255).astype(np.float32)

        # Invert the binary array if needed (optional)
        lower_mask = 1 - lower_mask

        # Display the recovered parse_right_shoe array
        print('lower_mask:', lower_mask)

        

    parse_left_leg_shoe = (parse_array == label_map["left_leg"]).astype(np.float32) + \
                        (parse_array == label_map["left_shoe"]).astype(np.float32)                  
    inpaint_mask = 1 - parse_left_leg_shoe
    img = np.where(inpaint_mask, 255, 0)

    # Convert img to uint8 array
    img_uint8 = img.astype(np.uint8)

    # Create the PIL Image
    img_left_leg_shoe = Image.fromarray(img_uint8, mode='L')  # mode='L' for grayscale image

    # Convert to RGB if needed
    img_left_leg_shoe = img_left_leg_shoe.convert('RGB')

    #fixed_img_left_leg_shoe = fill_above_y_with_white(img_left_leg_shoe, top_left_shoe[1]-50)
    fixed_img_left_leg_shoe = fill_above_y_with_white(img_left_leg_shoe, int(tuple(pose_data[13][:2])[1]))
    save_img_left_leg_shoe = fixed_img_left_leg_shoe.convert('RGB')

    print('pose data1:', tuple(pose_data[1][:2])[1])
    print('pose data2:', tuple(pose_data[2][:2])[1])
    print('pose data3:', tuple(pose_data[3][:2])[1])
    print('pose data4:', tuple(pose_data[4][:2])[1])
    print('pose data5:', tuple(pose_data[5][:2])[1])
    print('pose data6:', tuple(pose_data[6][:2])[1])
    print('pose data7:', tuple(pose_data[7][:2])[1])
    print('pose data8:', tuple(pose_data[8][:2])[1])
    print('pose data9:', tuple(pose_data[9][:2])[1])
    print('pose data10:', tuple(pose_data[10][:2])[1])
    print('pose data11:', tuple(pose_data[11][:2])[1])
    print('pose data12:', tuple(pose_data[12][:2])[1])
    print('pose data13:', tuple(pose_data[13][:2])[1])
    print('pose data14:', tuple(pose_data[14][:2])[1])
    # Convert the image to a grayscale NumPy array
    img_array = np.array(fixed_img_left_leg_shoe.convert('L'))

    # Threshold the grayscale array to convert it back to a binary array
    left_leg_shoe_mask = (img_array == 255).astype(np.float32)

    # Invert the binary array if needed (optional)
    left_leg_shoe_mask = 1 - left_leg_shoe_mask

    # Display the recovered parse_right_shoe array
    print('left_leg_shoe_mask:', left_leg_shoe_mask)


    parse_right_leg_shoe = (parse_array == label_map["right_leg"]).astype(np.float32) + \
                        (parse_array == label_map["right_shoe"]).astype(np.float32)                  
    inpaint_mask = 1 - parse_right_leg_shoe
    img = np.where(inpaint_mask, 255, 0)

     # Convert img to uint8 array
    img_uint8 = img.astype(np.uint8)

    # Create the PIL Image
    img_right_leg_shoe = Image.fromarray(img_uint8, mode='L')  # mode='L' for grayscale image

    # Convert to RGB if needed
    img_right_leg_shoe = img_right_leg_shoe.convert('RGB')

    #fixed_img_right_leg_shoe = fill_above_y_with_white(img_right_leg_shoe, top_right_shoe[1]-50)
    fixed_img_right_leg_shoe = fill_above_y_with_white(img_right_leg_shoe, int(tuple(pose_data[10][:2])[1]))
    save_img_right_leg_shoe = fixed_img_right_leg_shoe.convert('RGB')

    # Convert the image to a grayscale NumPy array
    img_array = np.array(fixed_img_right_leg_shoe.convert('L'))

    # Threshold the grayscale array to convert it back to a binary array
    right_leg_shoe_mask = (img_array == 255).astype(np.float32)

    # Invert the binary array if needed (optional)
    right_leg_shoe_mask = 1 - right_leg_shoe_mask

    # Display the recovered parse_right_shoe array
    print('right_leg_shoe_mask:', right_leg_shoe_mask)


    if model_type == 'hd':
        arm_width = 60
    elif model_type == 'dc':
        arm_width = 45
    else:
        raise ValueError("model_type must be \'hd\' or \'dc\'!")

    parse_head = (parse_array == 1).astype(np.float32) + \
                 (parse_array == 3).astype(np.float32) + \
                 (parse_array == 11).astype(np.float32)
    belt_mask = np.zeros_like(parse_array)

    parser_mask_fixed = (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["hat"]).astype(np.float32) + \
                        (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                        (parse_array == label_map["bag"]).astype(np.float32) + \
                        left_leg_shoe_mask + right_leg_shoe_mask + belt_mask

    parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)
    #parser_mask_changeable = np.zeros_like(parser_mask_changeable)

    arms_left = (parse_array == 14).astype(np.float32)
    arms_right = (parse_array == 15).astype(np.float32)
    arms = arms_left + arms_right

    if category == 'dresses':
        parse_mask = (parse_array == 7).astype(np.float32) + \
                     (parse_array == 4).astype(np.float32) + \
                     (parse_array == 5).astype(np.float32) + \
                     (parse_array == 6).astype(np.float32) + \
                     lower_mask + dresses_mask

        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

    else:
        raise NotImplementedError

    im_arms_left = Image.new('L', (width, height))
    im_arms_right = Image.new('L', (width, height))
    arms_draw_left = ImageDraw.Draw(im_arms_left)
    arms_draw_right = ImageDraw.Draw(im_arms_right)
    if category == 'dresses':
        shoulder_right = np.multiply(tuple(pose_data[2][:2]), height / 512.0)
        shoulder_left = np.multiply(tuple(pose_data[5][:2]), height / 512.0)
        elbow_right = np.multiply(tuple(pose_data[3][:2]), height / 512.0)
        elbow_left = np.multiply(tuple(pose_data[6][:2]), height / 512.0)
        wrist_right = np.multiply(tuple(pose_data[4][:2]), height / 512.0)
        wrist_left = np.multiply(tuple(pose_data[7][:2]), height / 512.0)
        ARM_LINE_WIDTH = int(arm_width / 512 * height)
        size_left = [shoulder_left[0] - ARM_LINE_WIDTH // 2, shoulder_left[1] - ARM_LINE_WIDTH // 2, shoulder_left[0] + ARM_LINE_WIDTH // 2, shoulder_left[1] + ARM_LINE_WIDTH // 2]
        size_right = [shoulder_right[0] - ARM_LINE_WIDTH // 2, shoulder_right[1] - ARM_LINE_WIDTH // 2, shoulder_right[0] + ARM_LINE_WIDTH // 2,
                      shoulder_right[1] + ARM_LINE_WIDTH // 2]
        
        print('pose point:', tuple(pose_data[5][:2]))

        if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
            im_arms_right = arms_right
        else:
            wrist_right = extend_arm_mask(wrist_right, elbow_right, 1.2)
            arms_draw_right.line(np.concatenate((shoulder_right, elbow_right, wrist_right)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            arms_draw_right.arc(size_right, 0, 360, 'white', ARM_LINE_WIDTH // 2)

        if wrist_left[0] <= 1. and wrist_left[1] <= 1.:
            im_arms_left = arms_left
        else:
            wrist_left = extend_arm_mask(wrist_left, elbow_left, 1.2)
            arms_draw_left.line(np.concatenate((wrist_left, elbow_left, shoulder_left)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            arms_draw_left.arc(size_left, 0, 360, 'white', ARM_LINE_WIDTH // 2)

        hands_left = np.logical_and(np.logical_not(im_arms_left), arms_left)
        hands_right = np.logical_and(np.logical_not(im_arms_right), arms_right)
        parser_mask_fixed += hands_left + hands_right

    parser_mask_fixed = np.logical_or(parser_mask_fixed, parse_head)
    parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)
    if category == 'dresses':
        neck_mask = (parse_array == 18).astype(np.float32)
        #neck_mask = cv2.dilate(neck_mask, np.ones((5, 5), np.uint16), iterations=1)
        neck_mask = np.logical_and(neck_mask, np.logical_not(parse_head))
        parse_mask = np.logical_or(parse_mask, neck_mask)
        arm_mask = cv2.dilate(np.logical_or(im_arms_left, im_arms_right).astype('float32'), np.ones((5, 5), np.uint16), iterations=4)
        parse_mask += np.logical_or(parse_mask, arm_mask)

    parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))

    parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
    inpaint_mask = 1 - parse_mask_total
    img = np.where(inpaint_mask, 255, 0)

    # Convert img to uint8 array
    img_uint8 = img.astype(np.uint8)

    # Create the PIL Image
    img_save = Image.fromarray(img_uint8, mode='L')  # mode='L' for grayscale image

    # Convert to RGB if needed
    img_save = img_save.convert('RGB')



    dst = hole_fill(img.astype(np.uint8))
    dst = refine_mask(dst)
    inpaint_mask = dst / 255 * 1
    mask = Image.fromarray(inpaint_mask.astype(np.uint8) * 255)
    mask_gray = Image.fromarray(inpaint_mask.astype(np.uint8) * 127)

    return mask, mask_gray