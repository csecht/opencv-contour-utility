import cv2
import sys

MY_OS = sys.platform[:3]

# Assign trackbar names based on OS b/c of name length limitations.
if MY_OS == 'lin':
    TBNAME = {
    '_contrast': f'{"Contrast/gain/alpha (100X):" : <40}',
    '_bright': f"{'Brightness/bias/beta, (-127):' : <40}",
    '_morph_op': ("Reduce noise morphology operator: "
                 f'{"0 open, 1 hitmiss, 2 close, 3 gradient" : <45}'),
    '_morph_shape': ("Reduce noise morphology shape: "
                    f'{"0 rectangle, 1 cross, 2 ellipse" : <48}'),
    '_noise_k': f'{"Reduce noise, kernel size (odd only):" : <40}',
    '_noise_i': f'{"Reduce noise, iterations:" : <46}',
    '_border': ("Border type:  "
               f'{"0 default, 1 reflect, 2 replicate, 3 isolated" : <46}'),
    '_filter': ("Filter type:  "
               f'{"0 box, 1 bilateral, 2 Gaussian, 3 median" : <46}'),
    '_kernel_size': f'{"Filter kernel size (odd only):" : <43}',
    '_ratio': f'{"Edges, max t-hold ratio (10x):" : <50}',
    '_thresh_min': f'{"Edges, lower t-hold:" : <50}',
    '_contour_mode': f'{"Find contour mode: 0 external, 1 list" : <40}',
    '_contour_method': f'{"Find contour method: 1 none, 2 simple" : <40}',
    '_contour_type': f'{"Contour size type: 0 area, 1 arc length" : <40}',
    '_contour_min': f'{"Contour size minimum (pixels):" : <30}',
    '_thresh_type': f'{"Thresholding type: 0 Otsu, 1 Triangle" : <40}',
    }
elif MY_OS == 'dar':
    TBNAME = {
    '_contrast': 'Alpha (100X):',
    '_bright': 'Beta (-127):',
    '_morph_op': 'Morph operator:',
    '_morph_shape': 'Morph shape:',
    '_noise_k': 'Noise redux, k:',
    '_noise_i': ' ...iterations:',
    '_border': 'Border type:',
    '_filter': 'Filter type:',
    '_kernel_size': 'Filter k size:',
    '_ratio': 'Edge th ratio:',
    '_thresh_min': '  ..lower thresh:',
    '_contour_mode': 'Find C mode:',
    '_contour_method': 'Find C method',
    '_contour_type': 'Contour type:',
    '_contour_min': 'Contour size min:',
    '_thresh_type': 'Threshold type',
    }
else:  # is Windows; names limited to 10 characters.
    TBNAME = {
    '_contrast': 'Alpha 100X',
    '_bright': 'Beta, -127',
    '_morph_op': 'Morph op:',
    '_morph_shape': '   shape:',
    '_noise_k': 'de-noise k',
    '_noise_i': '...iter:',
    '_border': 'Border:',
    '_filter': 'Filter:',
    '_kernel_size': 'Filter, k',
    '_ratio': 'Th ratio:',
    '_thresh_min': 'Th min:',
    '_contour_mode': 'Cnt mode:',
    '_contour_method': 'Cnt method',
    '_contour_type': 'Contour:',
    '_contour_min': 'Cnt size:',
    '_thresh_type': 'T-hold type',
    }

# Set ranges for trackbars used to adjust contrast and brightness for
#  the cv2.convertScaleAbs method.
ALPHA_MAX = 400
BETA_MAX = 254  # Provides a range of [-127 -- 127].

CV_BORDER = {
    # Key is the trackbar value.
    # Value of cv2.BORDER_* returns an integer, as commented...
    0: cv2.BORDER_REFLECT_101,  # 4  is same as cv2.BORDER_DEFAULT.
    1: cv2.BORDER_REFLECT,  # 2
    2: cv2.BORDER_REPLICATE,  # 1
    3: cv2.BORDER_ISOLATED,  # 16
}

BORDER_NAME = {
    # Key is the integer value returned from the cv2.BORDER_* constant.
    # Value is what to display as descriptive text.
    1: 'cv2.BORDER_REPLICATE',
    2: 'cv2.BORDER_REFLECT',
    4: 'cv2.BORDER_DEFAULT',  # Is same as cv2.BORDER_REFLECT_101'.
    16: 'cv2.BORDER_ISOLATED',
}

FILTER = {
    # Key is the trackbar value.
    # Value is what to display as descriptive text.
    0: 'cv2.blur',  # is default, a box filter.
    1: 'cv2.bilateralFilter',
    2: 'cv2.GaussianBlur',
    3: 'cv2.medianBlur'
}

TH_TYPE = {
    # Key is the integer value returned from the cv2.THRESH_* constant.
    # Value is what to display as descriptive text.
    # Note: Can mimic inverse types by adjusting alpha and beta channels.
    0: 'cv2.THRESH_BINARY',
    1: 'cv2.THRESH_BINARY_INVERSE',
    8: 'cv2.THRESH_OTSU',
    9: 'cv2.THRESH_OTSU_INVERSE',
    16: 'cv2.THRESH_TRIANGLE',
    17: 'cv2.THRESH_TRIANGLE_INVERSE',
}

CV_MORPHOP = {
    # Key is the trackbar value.
    # Value of cv2.MORPH_* returns an integer.
    0: cv2.MORPH_OPEN,  # 2
    1: cv2.MORPH_HITMISS,  # 7
    2: cv2.MORPH_CLOSE,  # 3
    3: cv2.MORPH_GRADIENT  # 4
}

MORPH_TYPE = {
    # Key is the integer value returned from the cv2.MORPH_* constant.
    # Value is what to display as descriptive text.
    2: 'cv2.MORPH_OPEN',  # default
    3: 'cv2.MORPH_CLOSE',
    4: 'cv2.MORPH_GRADIENT',
    7: 'cv2.MORPH_HITMISS',
}

MORPH_SHAPE = {
    # Key is the trackbar value; also is the constant's integer value.
    # Value is what to display as descriptive text.
    0: 'cv2.MORPH_RECT',  # (default)
    1: 'cv2.MORPH_CROSS',
    2: 'cv2.MORPH_ELLIPSE',
}

CONTOUR_MODE = {
    # Key is the trackbar value; also is the constant's integer value
    #   from the cv2.RETR__* constant.
    # Value is what to display as descriptive text.
    # RETR_EXTERNAL, RETR_LIST, RETR_CCOMP, RETR_TREE, RETR_FLOODFILL
    0: 'cv2.RETR_EXTERNAL',
    1: 'cv2.RETR_LIST',
}

CONTOUR_METHOD = {
    # Key is the trackbar value; also is the constant's integer value
    #   from the cv2.CHAIN_APPROX_* constant.
    # Value is what to display as descriptive text.
    # CHAIN_APPROX_SIMPLE, CHAIN_APPROX_NONE
    1: 'cv2.CHAIN_APPROX_NONE',
    2: 'cv2.CHAIN_APPROX_SIMPLE',
}

CONTOUR_TYPE = {
    # Key is the trackbar value.
    # Value is what to display as descriptive text.
    0: 'cv2.contourArea',
    1: 'cv2.arcLength',
}

# 	cv::HersheyFonts {
#   cv::FONT_HERSHEY_SIMPLEX = 0,
#   cv::FONT_HERSHEY_PLAIN = 1,
#   cv::FONT_HERSHEY_DUPLEX = 2,
#   cv::FONT_HERSHEY_COMPLEX = 3,
#   cv::FONT_HERSHEY_TRIPLEX = 4,
#   cv::FONT_HERSHEY_COMPLEX_SMALL = 5,
#   cv::FONT_HERSHEY_SCRIPT_SIMPLEX = 6,
#   cv::FONT_HERSHEY_SCRIPT_COMPLEX = 7,
#   cv::FONT_ITALIC = 16
# }
# https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/enum_cv_HersheyFonts.html
FONT_TYPE = cv2.FONT_HERSHEY_SIMPLEX

# Settings window text constants used in utils.text_array():
TEXT_THICKNESS = 1
TEXT_COLOR = 180, 180, 180  # light gray for a dark gray background

if MY_OS == 'lin':
    TEXT_SCALER = 0.5
elif MY_OS == 'dar':
    TEXT_SCALER = 0.4
else:  # is Windows
    TEXT_SCALER = 0.6

# Scaling factors for contour texts, empirically determined,
#  to use in manage_input().
LINE_SCALE = 1e-03
FONT_SCALE = 7.7e-04
CENTER_XSCALE = 0.035
