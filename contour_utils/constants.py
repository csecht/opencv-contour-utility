import cv2

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

CONTOUR = {
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
TEXT_SCALER = 0.5
TEXT_THICKNESS = 1
TEXT_COLOR = 180, 180, 180  # light gray for a dark gray background

# Scaling factors, empirically determined, for use in manage_input().
LINE_SCALE = 1e-03
FONT_SCALE = 7.7e-04
CENTER_XSCALE = 0.035
