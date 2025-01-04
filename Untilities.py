## Image segmentation and analysis from breast cancer cells for Mitochondria
#
#
# Tzu-Hsi Dec 2024
import os
import cv2
import sys
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from scipy.spatial.distance import cdist
from scipy import ndimage
from skimage.segmentation import watershed

def check_file_format(data_path):
    img_format = os.listdir(data_path)
    check_img = img_format[0]
    if check_img.endswith('.png'):
        imgformat = '.png'
    elif check_img.endswith('.jpg'):
        imgformat = '.jpg'
    elif check_img.endswith('.jpeg'):
        imgformat = '.jpeg'
    elif check_img.endswith('.tiff'):
        imgformat = '.tiff'
    elif check_img.endswith('.tif'):
        imgformat = '.tif'
    else:
        imgformat = ''

    return imgformat

def Get_Image_Data(data_path, Sample_index, type):
    imgformat = check_file_format(data_path)
    if imgformat:
        search_pattern = os.path.join(data_path, Sample_index + '*_c{}_ORG'.format(type) + imgformat)
        matching_files_org = glob.glob(search_pattern, recursive=True)
        search_pattern = os.path.join(data_path, Sample_index + '*_c{}'.format(type) + imgformat)
        matching_files_color = glob.glob(search_pattern, recursive=True)

        return matching_files_org, matching_files_color
    else:
        print('Please ensure the file is in one of the following formats: .png, .jpg, .jpeg, .tif, or .tiff')
        sys.exit()

def Brightness_Enhance(Img):
    hsv = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v += 255
    final_hsv = cv2.merge((h, s, v))
    BE_Img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return BE_Img

def imfill_function(mask):
    im_floodfill = mask.copy()
    mask_new = np.zeros((mask.shape[0]+2, mask.shape[1]+2), dtype=np.uint8)
    cv2.floodFill(im_floodfill, mask_new, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = mask | im_floodfill_inv

    return im_out

def Cell_Mask_Generate(cell_mask, cell_distance_set, nuclei_mask, Suitable_Nuclei_distance_set, element_region_size, cell_intensity_thresh, cell_mask_ORG, cell_mask_contour,advanced_seg, save_sign, save_path, Sample_name):
    # Morphological operations for initial noise remove

    _, th_nuclei_mask = cv2.threshold(nuclei_mask, 0, 255, cv2.THRESH_BINARY)
    th_nuclei_mask = imfill_function(th_nuclei_mask)
    th_nuclei_mask = cv2.morphologyEx(th_nuclei_mask, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8), iterations=5)

    cell_mask = 255 - cell_mask
    _, th_cell_mask = cv2.threshold(cell_mask, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    th_cell_mask = imfill_function(th_cell_mask)
    th_cell_mask = cv2.morphologyEx(th_cell_mask, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8), iterations=5)

    # Separate connected cells =========================================================================================
    # Use Marker-Controlled Watershed method
    # Use Nuclei centroids as the points
    # It was used by Local Maximum points being depended on the cell segmentation results
    Cell_dist_transform = cv2.distanceTransform(th_cell_mask, cv2.DIST_L2, 5)
    Cell_localMax = peak_local_max(Cell_dist_transform, min_distance=cell_distance_set, labels=th_cell_mask)

    Nuclei_dist_transform = cv2.distanceTransform(th_nuclei_mask, cv2.DIST_L2,5)
    Nuclei_localMax = peak_local_max(Nuclei_dist_transform, min_distance=Suitable_Nuclei_distance_set, labels=th_nuclei_mask)

    Nuclei_extra_point = []
    for c_l_ind in Cell_localMax:
        distances = cdist([c_l_ind], Nuclei_localMax, metric='euclidean')
        Check_distances = distances < (element_region_size/2)
        has_false = True in Check_distances
        if not has_false:
            if th_nuclei_mask[c_l_ind[1], c_l_ind[0]] > 0:
                Nuclei_extra_point.append(c_l_ind)
    Nuclei_extra_point = np.array(Nuclei_extra_point)

    if len(Nuclei_extra_point) != 0:
        Nuclei_localMax = np.vstack((Nuclei_localMax, Nuclei_extra_point))

    Nuclei_localMax_mask = np.zeros(Nuclei_dist_transform.shape, dtype=bool)
    Nuclei_localMax_mask[tuple(Nuclei_localMax.T)] = True
    Nuclei_markers = ndimage.label(Nuclei_localMax_mask, structure=np.ones((3, 3)))[0]
    Nuclei_labels = watershed(-Nuclei_dist_transform, Nuclei_markers, mask=th_nuclei_mask)

    new_Nuclei_mask = np.zeros(Nuclei_labels.shape, dtype='uint8')
    nuclei_max_label = np.unique(Nuclei_labels)
    for n_i in nuclei_max_label:
        if n_i > 0:
            tmp_nuclear_mask = np.zeros(Nuclei_labels.shape, dtype='uint8')
            tmp_nuclear_mask[Nuclei_labels==n_i] = 255
            tmp_nuclear_mask = cv2.morphologyEx(tmp_nuclear_mask, cv2.MORPH_ERODE, kernel=np.ones((5, 5), np.uint8), iterations=1)
            tmp_nuclear_mask_label = label(tmp_nuclear_mask)
            tmp_prop = regionprops(tmp_nuclear_mask_label)
            if tmp_prop != []:
                tmp_center = tmp_prop[0].centroid
                tmp_center = np.round(np.array(tmp_center))
                tmp_center = tmp_center.astype('int')
                if th_cell_mask[tmp_center[0], tmp_center[1]] != 0:
                    new_Nuclei_mask = new_Nuclei_mask+tmp_nuclear_mask

    Cell_markers = ndimage.label(new_Nuclei_mask, structure=np.ones((3, 3)))[0]
    Cell_labels = watershed(-Cell_dist_transform, Cell_markers, mask=th_cell_mask)

    # Extract the separate lines
    new_cell_mask = np.zeros(th_cell_mask.shape, dtype=np.uint8)
    cell_unique = np.unique(Cell_labels)
    tmp_mask = np.zeros(Cell_labels.shape, dtype='uint8')
    for c_i in cell_unique:
        if c_i > 0:
            tmp_mask[Cell_labels == cell_unique[c_i]] = 255
            tmp_mask_D = cv2.dilate(tmp_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
            tmp_edge = tmp_mask_D - tmp_mask
            new_cell_mask = new_cell_mask + tmp_edge

    separate_line = np.multiply(th_cell_mask, new_cell_mask / 255)
    separate_line[separate_line > 0] = 255
    separate_line = (255 - separate_line) / 255

    # Generate the intermediate cell mask
    th_cell_mask = np.multiply(th_cell_mask, separate_line)
    th_cell_mask = np.uint8(th_cell_mask)

    # Remove small area and non-target cells based on intensity
    regions = regionprops(Cell_labels)
    new_th_cell_mask = np.zeros(th_cell_mask.shape, dtype=np.uint8)

    for i, element in enumerate(regions):
        if element.area > element_region_size:
            check_edge = np.array(element.bbox)
            check_start = np.where(check_edge <= 12, 1, 0)
            check_end_x = np.where(check_edge >= th_cell_mask.shape[0] - 12, 1, 0)
            check_end_y = np.where(check_edge >= th_cell_mask.shape[1] - 12, 1, 0)
            tmp_label_intensity = cell_mask_ORG[Cell_labels == (i + 1)]
            check_intensity = np.max(tmp_label_intensity)
            if check_intensity > cell_intensity_thresh:
                if np.sum(check_start) + np.sum(check_end_x) + np.sum(check_end_y) < 1:
                    new_th_cell_mask[Cell_labels == (i + 1)] = 255

    th_cell_mask = np.uint8(np.multiply(new_th_cell_mask, separate_line))

    if not advanced_seg:
        # Draw the contours into the image
        cell_contours, hierarchy = cv2.findContours(th_cell_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cell_mask_contour, cell_contours, -1, (192, 0, 192), 6)

        if not save_sign:
            fig = plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(cell_mask_ORG)
            plt.subplot(1, 2, 2)
            plt.imshow(cell_mask_contour)
            # plt.show()
        else:
            cv2.imwrite(save_path + Sample_name + '_c1_Mask.jpg', cell_mask_contour)

    return th_cell_mask, separate_line

def Advanced_Cell_Mask_Regenerate(new_mask, mos_distance_set, th_cell_mask, cell_mask_ORG, cell_mask_contour, save_sign, save_path, Sample_name):

    new_mask_cell = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8), iterations=2)
    dist_transform_mos = cv2.distanceTransform(new_mask_cell, cv2.DIST_L2, 5)
    localMax_mos = peak_local_max(dist_transform_mos, min_distance=mos_distance_set, labels=new_mask_cell)
    localMax_mos_mask = np.zeros(dist_transform_mos.shape, dtype=bool)
    localMax_mos_mask[tuple(localMax_mos.T)] = True
    Cell_mos_markers = ndimage.label(localMax_mos_mask, structure=np.ones((3, 3)))[0]
    Cell_mos_labels = watershed(-dist_transform_mos, Cell_mos_markers, mask=th_cell_mask)

    # Extract the new separate lines
    new_cell_mask = np.zeros(th_cell_mask.shape, dtype=np.uint8)
    cell_mask_label = label(th_cell_mask)
    cell_mos_unique = np.unique(Cell_mos_labels)
    tmp_mask = np.zeros(Cell_mos_labels.shape, dtype='uint8')
    for c_i in cell_mos_unique:
        if c_i > 0:
            tmp_mask[Cell_mos_labels == c_i] = 255
            tmp_mask_D = cv2.dilate(tmp_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
            tmp_edge = tmp_mask_D - tmp_mask
            new_cell_mask = new_cell_mask + tmp_edge

    new_separate_line = np.multiply(th_cell_mask, new_cell_mask / 255)
    new_separate_line[new_separate_line > 0] = 255

    # Select which line should be removed

    new_separate_line_lab = np.multiply(new_separate_line / 255, cell_mask_label)
    new_separate_line_lab_unique = np.unique(new_separate_line_lab)
    regions = regionprops(cell_mask_label)
    for sep_i in new_separate_line_lab_unique:
        if sep_i > 0:
            tmp_reg = regions[int(sep_i) - 1]
            area = tmp_reg.area
            perimeter = tmp_reg.perimeter
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity > 0.8:
                new_separate_line_lab[new_separate_line_lab == sep_i] = 0

    advanced_separate_line = np.zeros(new_separate_line_lab.shape, dtype=np.uint8)
    advanced_separate_line[new_separate_line_lab > 0] = 255
    advanced_separate_line = 255 - advanced_separate_line

    # Generate the advanced cell mask
    th_cell_mask = np.multiply(th_cell_mask, advanced_separate_line / 255)
    th_cell_mask = np.uint8(th_cell_mask)

    new_mask = np.multiply(new_mask, advanced_separate_line / 255)
    new_mask = np.uint8(new_mask)

    cell_contours, hierarchy = cv2.findContours(th_cell_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(cell_mask_contour, cell_contours, -1, (192, 0, 192), 6)

    if not save_sign:
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(cell_mask_ORG)
        plt.subplot(1, 2, 2)
        plt.imshow(cell_mask_contour)
        # plt.show()
    else:
        cv2.imwrite(save_path + Sample_name + '_c1_Mask.jpg', cell_mask_contour)

    return th_cell_mask, new_mask, advanced_separate_line


def Mito_Mask_Generate(mos_mask, th_cell_mask, separate_line, advanced_seg, save_sign, mos_distance_set, cell_mask_ORG, cell_mask_contour, save_path, Sample_name):
    # Generate the initial mos mask and assign the cell labels for removing non-target mos
    binary_threshold, image_result = cv2.threshold(mos_mask, 1, 255, cv2.THRESH_BINARY)
    ret_mos, mos_markers_all = cv2.connectedComponents(image_result)

    overlap_mask = image_result / 255 + th_cell_mask / 255
    overlap_mask_selected = np.multiply(mos_markers_all, th_cell_mask / 255)

    new_mask = np.zeros(image_result.shape, dtype=np.uint8)
    for L_i in range(ret_mos):
        if np.max(overlap_mask[mos_markers_all == L_i]) > 1:
            mos_label = np.where(mos_markers_all == L_i)
            selected_mos_label = np.where(overlap_mask_selected == L_i)
            if len(selected_mos_label[0]) > 0:
                if len(selected_mos_label[0]) >= 0.75 * len(mos_label[0]):
                    new_mask[mos_markers_all == L_i] = 255

    new_mask = np.multiply(new_mask, separate_line)
    new_mask = np.uint8(new_mask)

    if advanced_seg:
        th_cell_mask, new_mask, advanced_separate_line = Advanced_Cell_Mask_Regenerate(new_mask, mos_distance_set, th_cell_mask, cell_mask_ORG, cell_mask_contour, save_sign, save_path, Sample_name)

    # Separate mos into individual cell ================================================================================

    cell_mask_label = label(th_cell_mask)
    th_cell_mask_new = np.multiply(cell_mask_label, new_mask / 255)
    new_ret_mos, new_mos_mask_label = cv2.connectedComponents(new_mask)
    sep_new_mask = np.zeros(new_mask.shape, dtype=np.uint8)

    for mos_i in range(new_ret_mos):
        tmp_mos = th_cell_mask_new[new_mos_mask_label == (mos_i + 1)]
        tmp_mos_label = np.unique(tmp_mos)
        if len(tmp_mos_label) > 1:
            all_count = []
            for s_m_i in tmp_mos_label:
                tmp_part = np.where(tmp_mos == s_m_i)
                all_count.append(len(tmp_part[0]))
            max_count_ind = np.where(all_count == np.max(all_count))
            tmp_mos_label = tmp_mos_label[max_count_ind[0]]
        sep_new_mask[new_mos_mask_label == (mos_i + 1)] = tmp_mos_label

    label_num = np.unique(sep_new_mask)

    return image_result, new_mask, sep_new_mask, label_num

def Fragment_Analysis(new_mask, Frag_Area_set, sep_new_mask, label_num, mos_mask_contour, save_sign, save_path, Sample_name, image_result):
    frag_label_img = label(new_mask)
    frag_regions = regionprops(frag_label_img)

    new_mask_Green = np.zeros(new_mask.shape, dtype=np.uint8)

    # Use the size of mos to identify the fragments
    for f_i, f_element in enumerate(frag_regions):
        if f_element.area > Frag_Area_set:
            new_mask_Green[frag_label_img == (f_i + 1)] = 255
    # fig=plt.figure()
    # plt.imshow(new_mask_Green)
    # plt.show()

    for r_i in label_num:
        if r_i > 0:
            tmp_check_unfrag = new_mask_Green[sep_new_mask == r_i]
            tmp_check_unfrag_true = np.where(tmp_check_unfrag == 255)
            if len(np.unique(np.array(tmp_check_unfrag_true))) == 0:
                sep_new_mask[sep_new_mask == r_i] = 0

    label_num = np.unique(sep_new_mask)

    text_location = []
    for new_i in range(1, len(label_num)):
        tmp_new_single_mask = (sep_new_mask == label_num[new_i])
        tmp_new_single_mask = label(tmp_new_single_mask)
        tmp_new_single_region = regionprops(tmp_new_single_mask)
        tmp_min_top = 0
        tmp_min_left = 0
        for sm_i, sm_element in enumerate(tmp_new_single_region):
            minr, minc, maxr, maxc = sm_element.bbox
            if sm_i == 0:
                tmp_min_top = minc
                tmp_min_left = minr
                tmp_max_top = maxc
                tmp_max_left = maxr
            else:
                if tmp_max_top < maxc:
                    tmp_max_top = maxc
                if tmp_max_left < maxr:
                    tmp_max_left = maxr
                if tmp_min_top > minc:
                    tmp_min_top = minc
                if tmp_min_left > minr:
                    tmp_min_left = minr
                    
        if sep_new_mask[tmp_min_top, tmp_min_left] != 0:
            text_location.append([tmp_max_top, tmp_min_left])
        else:
            text_location.append([tmp_min_top, tmp_min_left])

    # draw the contour of the mos in each cell and others ==============================================================

    mos_mask_contour_mark = mos_mask_contour.copy()
    mos_mask_contour_sep = mos_mask_contour.copy()
    mos_mask_contour_unFrag = mos_mask_contour.copy()

    colors = plt.cm.jet(np.linspace(0,1, 256))
    colors = colors[:, :3] * 255
    color_index = np.arange(colors.shape[0])
    random.shuffle(color_index)
    random.shuffle(color_index)
    random.shuffle(color_index)
    color_setting = []
    for la_i in range(len(label_num)):
        if la_i > 0:
            color = colors[color_index[la_i]].tolist()
            color_setting.append(color)
            tmp_mos_mask = np.zeros(new_mask.shape, dtype=np.uint8)
            tmp_mos_mask[sep_new_mask == label_num[la_i]] = 255
            tmp_mos_contours, tmp_mos_hierarchy = cv2.findContours(tmp_mos_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mos_mask_contour_mark, tmp_mos_contours, -1, color=color, thickness=6)

    for t_i in range(1, len(label_num)):
        cv2.putText(mos_mask_contour_mark, str(t_i), text_location[t_i-1], 0, 3, [255, 255, 255], 5)

    if not save_sign:
        fig = plt.figure()
        plt.imshow(mos_mask_contour_mark)
    else:
        cv2.imwrite(save_path + Sample_name + '_c3_Mask.jpg', mos_mask_contour_mark)

    # Fragmented =======================================================================================================
    new_mask[sep_new_mask == 0] = 0
    frag_label_img = label(new_mask)
    frag_regions = regionprops(frag_label_img)

    new_mask_Green = np.zeros(new_mask.shape, dtype=np.uint8)
    new_mask_Red = np.zeros(new_mask.shape, dtype=np.uint8)

    # Use the size of mos to identify the fragments
    for f_i, f_element in enumerate(frag_regions):
        if f_element.area > Frag_Area_set:
            new_mask_Green[frag_label_img == (f_i+1)] = 255
        else:
            new_mask_Red[frag_label_img == (f_i+1)] = 255

    frag_g_contours, frag_g_hierarchy = cv2.findContours(new_mask_Green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    frag_r_contours, frag_r_hierarchy = cv2.findContours(new_mask_Red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(mos_mask_contour_sep, frag_g_contours, -1, (0, 128, 128), 3)
    cv2.drawContours(mos_mask_contour_sep, frag_r_contours, -1, (0, 0, 255), 3)
    cv2.drawContours(mos_mask_contour_unFrag, frag_g_contours, -1, (128, 128, 128), 3)

    for t_i in range(1, len(label_num)):
        cv2.putText(mos_mask_contour_sep, str(t_i), text_location[t_i-1], 0, 3, [255, 255, 255], 5)
        cv2.putText(mos_mask_contour_unFrag, str(t_i), text_location[t_i - 1], 0, 3, [255, 255, 255], 5)

    if not save_sign:
        fig = plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(image_result)
        plt.subplot(1, 3, 2)
        plt.imshow(mos_mask_contour_sep)
        plt.subplot(1, 3, 3)
        plt.imshow(mos_mask_contour_unFrag)
    else:
        cv2.imwrite(save_path + Sample_name + '_c1_Fragment_Mask.jpg', mos_mask_contour_sep)
        cv2.imwrite(save_path + Sample_name + '_c1_unFragment_skeleton_Mask.jpg', mos_mask_contour_unFrag)

    return new_mask_Green, new_mask_Red, color_setting

def Frag_Analysis_Plot(mos_mask, sep_new_mask, new_mask_Red, label_num, color_setting, save_sign, save_path, Sample_name):
    frag_int = mos_mask[new_mask_Red == 255]
    sep_cell_only_frag = np.multiply(sep_new_mask, new_mask_Red/255)

    frag_area_per_cell = []
    frag_number_per_cell = []
    frag_int_per_cell = []
    for cell_ind in label_num:
        tmp_frag = np.zeros(sep_new_mask.shape, dtype=np.uint8)
        if cell_ind > 0:
            tmp_frag[sep_cell_only_frag == cell_ind] = 255
            tmp_frag_label = label(tmp_frag)
            tmp_frag_unique_label = np.unique(tmp_frag_label)
            frag_number_per_cell.append(len(tmp_frag_unique_label)-1)

            if (len(tmp_frag_unique_label)-1) > 0:
                tmp_frag_element = regionprops(tmp_frag_label)
                tmp_single_cell_frag_area = []
                tmp_single_cell_frag_int = []
                for s_f_i, single_frag_element in enumerate(tmp_frag_element):
                    small_fragment_int = mos_mask[tmp_frag_label == (s_f_i+1)]
                    tmp_single_cell_frag_int.append(np.mean(small_fragment_int))
                    tmp_single_cell_frag_area.append(single_frag_element.area)
                frag_area_per_cell.append(np.mean(np.array(tmp_single_cell_frag_area)))
                frag_int_per_cell.append(np.mean(np.array(tmp_single_cell_frag_int)))

            else:
                frag_area_per_cell.append(0)
                frag_int_per_cell.append(0)


    frag_area_per_cell = np.array(frag_area_per_cell)
    frag_number_per_cell = np.array(frag_number_per_cell)
    frag_int_per_cell = np.array(frag_int_per_cell)
    c_x = np.arange(1, len(frag_number_per_cell)+1)

    color_setting = np.array(color_setting)
    color_setting = color_setting/255
    alpha = np.ones(color_setting.shape[0])*0.7
    color_setting = np.concatenate((color_setting, alpha[:, np.newaxis]), axis=1)

    fig, ax = plt.subplots(3, 1, figsize=(8, 9))
    fig.suptitle(Sample_name, fontsize=11)
    ax[0].bar(c_x, frag_number_per_cell, width=0.5, color=color_setting, edgecolor='k')
    ax[0].set_ylabel('Number of Fragment', fontsize=14)
    ax[0].set_xticks(c_x)
    ax[0].spines[['right', 'top']].set_visible(False)

    ax[1].bar(c_x, frag_area_per_cell, width=0.5, color=color_setting, edgecolor='k')
    ax[1].set_ylabel('Area size of Fragment', fontsize=14)
    ax[1].set_xticks(c_x)
    ax[1].spines[['right', 'top']].set_visible(False)

    ax[2].bar(c_x, frag_int_per_cell, width=0.5, color=color_setting, edgecolor='k')
    ax[2].set_ylabel('Intensity of Fragment', fontsize=14)
    ax[2].set_xlabel('Cell', fontsize=14)
    ax[2].set_xticks(c_x)
    ax[2].spines[['right', 'top']].set_visible(False)

    plt.tight_layout()

    if save_sign:
        plt.savefig(save_path + Sample_name + '_Fragment_analysis.png')
        plt.close(fig)

    return frag_int, frag_area_per_cell, frag_number_per_cell, frag_int_per_cell, color_setting

    # Plot the unfragmentation analysis ================================================================================

def UnFrag_Analysis_Plot(mos_mask, sep_new_mask, new_mask_Green, label_num, frag_number_per_cell, color_setting, save_sign, save_path, Sample_name):

    unfrag_int = mos_mask[new_mask_Green == 255]
    sep_cell_only_unfrag = np.multiply(sep_new_mask, new_mask_Green/255)

    unfrag_area_per_cell = []
    unfrag_number_per_cell = []
    unfrag_int_per_cell = []
    for cell_ind in label_num:
        tmp_unfrag = np.zeros(sep_new_mask.shape, dtype=np.uint8)
        if cell_ind > 0:
            tmp_unfrag[sep_cell_only_unfrag == cell_ind] = 255
            tmp_unfrag_label = label(tmp_unfrag)
            tmp_unfrag_unique_label = np.unique(tmp_unfrag_label)
            unfrag_number_per_cell.append(len(tmp_unfrag_unique_label)-1)

            if (len(tmp_unfrag_unique_label)-1) > 0:
                tmp_unfrag_element = regionprops(tmp_unfrag_label)
                tmp_single_cell_unfrag_area = []
                tmp_single_cell_unfrag_int=[]
                for s_uf_i, single_unfrag_element in enumerate(tmp_unfrag_element):
                    small_fragment_int = mos_mask[tmp_unfrag_label == (s_uf_i+1)]
                    tmp_single_cell_unfrag_int.append(np.mean(small_fragment_int))
                    tmp_single_cell_unfrag_area.append(single_unfrag_element.area)
                unfrag_area_per_cell.append(np.mean(np.array(tmp_single_cell_unfrag_area)))
                unfrag_int_per_cell.append(np.mean(np.array(tmp_single_cell_unfrag_int)))
            else:
                unfrag_area_per_cell.append(0)
                unfrag_int_per_cell.append(0)

    unfrag_area_per_cell = np.array(unfrag_area_per_cell)
    unfrag_number_per_cell = np.array(unfrag_number_per_cell)
    unfrag_int_per_cell = np.array(unfrag_int_per_cell)
    un_c_x = np.arange(1, len(unfrag_number_per_cell)+1)
    c_x = np.arange(1, len(frag_number_per_cell)+1)

    fig, ax = plt.subplots(3, 1, figsize=(8, 9))
    fig.suptitle(Sample_name, fontsize=11)
    ax[0].bar(un_c_x, unfrag_number_per_cell, width=0.5, color=color_setting, edgecolor='k')
    ax[0].set_ylabel('Number of Unfragment', fontsize=14)
    ax[0].set_xticks(c_x)
    ax[0].spines[['right', 'top']].set_visible(False)

    ax[1].bar(un_c_x, unfrag_area_per_cell, width=0.5, color=color_setting, edgecolor='k')
    ax[1].set_ylabel('Area size of Unfragment', fontsize=14)
    ax[1].set_xticks(c_x)
    ax[1].spines[['right', 'top']].set_visible(False)

    ax[2].bar(un_c_x, unfrag_int_per_cell, width=0.5, color=color_setting, edgecolor='k')
    ax[2].set_ylabel('Intensity of Unfragment', fontsize=14)
    ax[2].set_xlabel('Cell', fontsize=14)
    ax[2].set_xticks(c_x)
    ax[2].spines[['right', 'top']].set_visible(False)

    plt.tight_layout()

    if save_sign:
        plt.savefig(save_path + Sample_name + '_Unfragment_analysis.png')
        plt.close(fig)

    return unfrag_int, unfrag_area_per_cell, unfrag_number_per_cell, unfrag_int_per_cell

def Intensity_Comparison(frag_int, unfrag_int, save_sign, save_path, Sample_name):
    all_intensity = [np.mean(frag_int), np.mean(unfrag_int)]
    all_x = np.arange(len(all_intensity))
    all_color = ['r', 'g']

    fig = plt.figure(figsize=(3, 3))
    plt.bar(all_x, all_intensity, width=0.5, color=all_color, edgecolor='k')
    plt.ylabel('Intensity', fontsize=14)
    plt.xticks(all_x, ['Fragment', 'Unfragment'], rotation=20, fontsize=12)
    plt.tight_layout()

    if save_sign:
        plt.savefig(save_path + Sample_name + '_Intensity_comparison.png')
        plt.close(fig)


def main(all_data_path, update_progress, **kwargs):
    """

    Input:
        all_data_path: the data path
        kwargs:
            save_sign: boolean, Show(False) or Save figures(True), default=True
            advanced_seg: boolean, Need to use advanced segmentation(True) or not(False). In general, keeping this parameter as default is good enough.
                          default=False
            element_region_size = Integer, the treshold value to remove small area, default=2500
            Frag_Area_set: Integer, the treshold value to select which belongs to Fragment, default=1500
            Suitable_Nuclei_distance_set: Integer, how far the nuclear centroid obtained by distance map can be considered for segmentation, default=80
            cell_distance_set: Integer, how far the cell obtained by distance map can be considered for segmentation, default=100
            mos_distance_set: Integer, how far the mito can be considered for segmentation, default=100
            cell_intensity_thresh: Integer, the threshold value of intensity for cell region identification, default=8

    Return: Save the results in the save path (Result Folder) if save_sign is True

    """

    save_sign = kwargs.get('save_sign', True)
    advanced_seg = kwargs.get('advanced_seg', False)
    element_region_size = kwargs.get('element_region_size', 2500)
    Frag_Area_set = kwargs.get('Frag_Area_set', 1500)
    Suitable_Nuclei_distance_set = kwargs.get('nuclei_distance_set', 80)
    cell_distance_set = kwargs.get('cell_distance_set', 100)
    mos_distance_set = kwargs.get('mos_distance_set', 100)
    cell_intensity_thresh = kwargs.get('cell_intensity_thresh', 8)

    Sample_list = os.listdir(all_data_path)
    for s_i in range(len(Sample_list)):

        data_path = all_data_path + Sample_list[s_i] + '/'
        save_path = 'Results/' + Sample_list[s_i] + '/'
        if not os.path.isdir('Results/'):
            os.mkdir('Results/')
        if save_sign:
            if not os.path.isdir(save_path):
                os.mkdir(save_path)

        matching_files_nuclei_org, matching_files_nuclei_color = Get_Image_Data(data_path, Sample_list[s_i], 4)
        nuclei_mask = cv2.imread(matching_files_nuclei_org[0])
        nuclei_mask_contour = cv2.imread(matching_files_nuclei_color[0])
        assert nuclei_mask is not None, "file for nuclei could not be read, check with os.path.exists()"

        matching_files_cell_org, matching_files_cell_color = Get_Image_Data(data_path, Sample_list[s_i], 1)
        cell_mask = cv2.imread(matching_files_cell_org[0])
        cell_mask_contour = cv2.imread(matching_files_cell_color[0])
        assert cell_mask is not None, "file for Neutrophil could not be read, check with os.path.exists()"

        # mos_mask = cv2.imread(data_path + 'Airyscan_63x_Tom20-488_Splenic neutrophils_1739_TG_4_c3_ORG.jpg')
        matching_files_mos_org, matching_files_mos_color = Get_Image_Data(data_path, Sample_list[s_i], 3)
        mos_mask = cv2.imread(matching_files_mos_org[0])
        mos_mask_contour = cv2.imread(matching_files_mos_color[0])
        assert mos_mask is not None, "file for Mitochondria could not be read, check with os.path.exists()"

        # Preprocessing the input image ================================================================================

        nuclei_mask_ORG = nuclei_mask.copy()
        nuclei_mask_ORG = cv2.cvtColor(nuclei_mask_ORG, cv2.COLOR_BGR2GRAY)
        cell_mask_ORG = cell_mask.copy()
        cell_mask_ORG = cv2.cvtColor(cell_mask_ORG, cv2.COLOR_BGR2GRAY)

        # Use the HSV to enhance brightness for initial segmentation (Only for cells)
        cell_mask = Brightness_Enhance(cell_mask)

        nuclei_mask = cv2.cvtColor(nuclei_mask, cv2.COLOR_BGR2GRAY)
        cell_mask = cv2.cvtColor(cell_mask, cv2.COLOR_BGR2GRAY)
        mos_mask = cv2.cvtColor(mos_mask, cv2.COLOR_BGR2GRAY)

        # print('Processing Cell Marker of {} ============================================================================'.format(Sample_list[s_i]))
        update_progress('Processing Cell Marker of {} ------'.format(Sample_list[s_i]))

        th_cell_mask, separate_line = Cell_Mask_Generate(cell_mask, cell_distance_set, nuclei_mask, Suitable_Nuclei_distance_set, element_region_size, cell_intensity_thresh, cell_mask_ORG, cell_mask_contour, advanced_seg, save_sign, save_path, Sample_list[s_i])

        # Selected the mos regions =====================================================================================

        # print('Processing MOS Marker of {} ============================================================================='.format(Sample_list[s_i]))
        update_progress('Processing Mito Marker of {} ------'.format(Sample_list[s_i]))

        image_result, new_mask, sep_new_mask, label_num = Mito_Mask_Generate(mos_mask, th_cell_mask, separate_line, advanced_seg, save_sign, mos_distance_set, cell_mask_ORG, cell_mask_contour, save_path, Sample_list[s_i])

        # Use UnFragmented to remove some unused fragments =============================================================

        new_mask_Green, new_mask_Red, color_setting = Fragment_Analysis(new_mask, Frag_Area_set, sep_new_mask, label_num, mos_mask_contour, save_sign, save_path, Sample_list[s_i], image_result)

        # Plot the fragmentation analysis ==============================================================================

        frag_int, frag_area_per_cell, frag_number_per_cell, frag_int_per_cell, color_setting = Frag_Analysis_Plot(mos_mask, sep_new_mask, new_mask_Red, label_num, color_setting, save_sign, save_path, Sample_list[s_i])
        unfrag_int, unfrag_area_per_cell, unfrag_number_per_cell, unfrag_int_per_cell = UnFrag_Analysis_Plot(mos_mask, sep_new_mask, new_mask_Green, label_num, frag_number_per_cell, color_setting, save_sign, save_path, Sample_list[s_i])

        # Compare intensity of Fragment and unFragment =================================================================

        Intensity_Comparison(frag_int, unfrag_int, save_sign, save_path, Sample_list[s_i])

        # Save all the information to CSV file =========================================================================

        if save_sign:
            cell_index = np.arange(1, len(label_num))
            df = pd.DataFrame({'Cell Number': cell_index,
                               'UnFrag_Number': unfrag_number_per_cell,
                               'UnFrag_Area': unfrag_area_per_cell,
                               'UnFrag_Intensity': unfrag_int_per_cell,
                               'Frag_Number': frag_number_per_cell,
                               'Frag_Area': frag_area_per_cell,
                               'Frag_Intensity': frag_int_per_cell})
            df.to_csv(save_path + Sample_list[s_i] + '_analysis.csv', index=False)
        else:
            plt.show()