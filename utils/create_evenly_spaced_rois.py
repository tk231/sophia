import numpy


def create_rois_in_centre(inarray: numpy.ndarray, img_id: str or None, roi_size: tuple = (30, 30), num_rois: int = 9,
                          margin_percentage: float = 0.3):
    """
    Create 9 evenly spaced ROIs of size 30x30 within the central 30% margins of a given numpy array.

    Parameters:
    - inarray (numpy.ndarray): The input array of dimensions (t, y, x).
    - roi_size (tuple): The size of each ROI (height, width).
    - num_rois (int): The number of ROIs to create.
    - margin_percentage (float): The percentage of margins to exclude from the edges.

    Returns:
    - list of numpy.ndarray: A list of ROIs.
    """
    (t, y, x) = inarray.shape
    roi_height, roi_width = roi_size

    # Calculate the central region boundaries
    x_margin = int(margin_percentage * x)
    y_margin = int(margin_percentage * y)
    central_x_start = x_margin
    central_x_end = x - x_margin
    central_y_start = y_margin
    central_y_end = y - y_margin

    # Calculate the number of ROIs along x and y dimension
    num_rois_per_dim = int(numpy.sqrt(num_rois))

    # Calculate the spacing between ROIs within the central region
    central_width = central_x_end - central_x_start
    central_height = central_y_end - central_y_start

    # Create spacing
    x_spacing = (central_width - num_rois_per_dim * roi_height) // (num_rois_per_dim + 1)
    y_spacing = (central_height - num_rois_per_dim * roi_width) // (num_rois_per_dim + 1)

    rois = []

    # Check if it will fit
    if (x_spacing <= 0) or (y_spacing <= 0):
        print(f"{img_id}: {num_rois} ROIs of {roi_width}x{roi_height} (w x h) do not fit into image of size {x}x{y}!")
        pass

    roi_dict = {}

    counter = 1

    for i in range(num_rois_per_dim):
        for j in range(num_rois_per_dim):
            roi_dict[f'ROI{counter}'] = {}
            start_x = central_x_start + x_spacing + i * (roi_height + x_spacing)
            start_y = central_y_start + y_spacing + j * (roi_width + y_spacing)
            roi_dict[f'ROI{counter}']['roi'] = inarray[:, start_y:start_y + roi_width, start_x:start_x + roi_height]
            roi_dict[f'ROI{counter}'][f'x_start_end'] = (start_x, start_x + roi_height)
            roi_dict[f'ROI{counter}'][f'y_start_end'] = (start_y, start_y + roi_width)
            counter += 1

    return roi_dict
