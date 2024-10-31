# Main Function

# TH Mike Song October 2024

from Untilities import *


# Input ================================================================================================================
all_data_path = 'Data/'

# Main =================================================================================================================
"""
This workflow is used to analyze the cell components. Please make sure that the iuput image with different channels (staining) 
should be included in the same folder.

Input:
    all_data_path: the data path
    kwargs:
        save_sign: boolean, Show(False) or Save figures(True), default=True
        advanced_seg: boolean, Need to use advanced segmentation(True) or not(False). In general, keeping this parameter as default is good enough.
                      default=False
        element_region_size = Integer, the treshold value to remove small area, default=2500
        Frag_Area_set: Integer, the treshold value to select which belongs to Fragment, default=1500
        cell_distance_set: Integer, how far the cell can be considered for segmentation, default=50
        mos_distance_set: Integer, how far the mito can be considered for segmentation, default=100
        cell_intensity_thresh: Integer, the threshold value of intensity for cell region identification, default=8

Return: Save the results in the save path (Result Folder) if save_sign is True. The results include the analysis and segmentation of cell components.

"""


if __name__ == "__main__":
    main(all_data_path)
