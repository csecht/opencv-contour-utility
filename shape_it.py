#!/usr/bin/env python3
"""Use OpenCV to explore image processing parameters involved in
identifying objects of specific shape. Parameter values are adjusted
with slide bars.

USAGE Example command lines, from within the image-processor-main folder:
python3 -m shape_it --help
python3 -m shape_it --about
python3 -m shape_it --input images/sample4.jpg
python3 -m shape_it -i images/sample4.jpg

Quit program with Esc or Q key or from command line with Ctrl-C.
Save any selected window with Ctrl-S or from rt-click menu.
Save settings and contour image with the Save/Print slide bar.

Requires Python3.7 or later and the packages opencv-python and numpy.
Developed in Python 3.8-3.9.
"""

# Copyright (C) 2022 C.S. Echt, under GNU General Public License

# Standard library imports
import math
import sys
import threading

from pathlib import Path
import numpy as np

# Third party imports
try:
    import cv2
except (ImportError, ModuleNotFoundError) as err:
    print('*** The opencv-python package was not found'
          ' or needs an update:\n\n'
          'To install: from the current folder, run this command'
          ' for the Python package installer (PIP):\n'
          '   python3 -m pip install opencv-python\n\n'
          'A package may already be installed, but needs an update;\n'
          '   python3 -m pip install -U opencv-python\n\n'
          f'Error message:\n{err}')
    sys.exit(1)

# Local application imports
from contour_modules import (vcheck,
                             utils,
                             constants as const
                             )


class ProcessImage:
    """
    A suite of methods for applying various OpenCV image processing
    functions involved in on identifying objects in an image file using
    thresholding.
    Methods:
        manage_input
        setup_trackbars
        alpha_selector
        beta_selector
        morphology_op_selector
        noise_redux_iter_selector
        noise_redux_shape_selector
        noise_redux_kernel_selector
        border_selector
        filter_type_selector
        filter_kernel_selector
        thresh_type_selector
        contour_mode_selector
        contour_method_selector
        contour_limit_selector
        epsilon_selector
        num_vertices_selector
        mindist_selector
        param1_selector
        param2_selector
        minradius_selector
        maxradius_selector
        save_with_click
        adjust_contrast
        reduce_noise
        filter_image
        contour_threshold
        find_circles
        select_shape
        contour_shapes
        show_settings
    """
    __slots__ = ('input_img', 'gray_img', 'contoured_img', 'contrasted_img',
                 'reduced_noise_img', 'filtered_img', 'th_img', 'shaped_img',
                 'thresh', 'alpha', 'beta', 'input_contrast_sd',
                 'curr_contrast_sd', 'noise_iter', 'morph_op', 'morph_shape',
                 'filter_selection', 'sigma_color', 'sigma_space', 'sigma_x',
                 'sigma_y', 'border_type', 'th_type', 'computed_threshold',
                 'num_th_contours_all', 'num_th_contours_select',
                 'contour_type', 'contour_limit',
                 'polygon', 'num_shapes', 'e_factor',
                 'circles_mindist', 'circles_param1', 'circles_param2',
                 'circles_min_radius', 'circles_max_radius', 'circle_img2use',
                 'line_thickness',
                 'noise_kernel', 'filter_kernel',
                 'contour_mode', 'contour_method',
                 'num_vertices', 'contoured_txt', 'contour_tb_win',
                 'shaped_txt', 'shape_tb_win',
                 )

    def __init__(self):

        # The np.ndarray arrays for images and ndarrays to be processed.
        self.input_img = None
        self.gray_img = None
        self.contoured_img = None
        self.contrasted_img = None
        self.reduced_noise_img = None
        self.filtered_img = None
        self.th_img = None
        self.shaped_img = None
        self.thresh = None
        # self.stub_kernel = np.ones((5, 5), 'uint8')

        # Image processing parameters.
        self.alpha = 1.0
        self.beta = 0
        self.input_contrast_sd = 0
        self.curr_contrast_sd = 0
        self.noise_iter = 0
        self.morph_op = 2  # default cv2.MORPH_OPEN.
        self.morph_shape = 2  # default cv2.MORPH_ELLIPSE.
        self.filter_selection = ''
        self.sigma_color = 1
        self.sigma_space = 1
        self.sigma_x = 1
        self.sigma_y = 1
        self.border_type = 4  # cv2.BORDER_DEFAULT == cv2.BORDER_REFLECT_101
        self.th_type = 8  # cv2.threshold type cv2.THRESH_OTSU.
        self.computed_threshold = 0
        self.num_th_contours_all = 0
        self.num_th_contours_select = 0
        self.contour_type = ''
        self.contour_limit = 0

        # Shape finding variables.
        self.polygon = ''
        self.num_shapes = 0
        self.e_factor = 0
        self.circles_mindist = 1
        self.circles_param1 = 1
        self.circles_param2 = 1
        self.circles_min_radius = 1
        self.circles_max_radius = 1
        self.circle_img2use = 0

        self.line_thickness = 0

        # Need to set starting values for variables set by some trackbars
        # for faster program startup.
        self.noise_kernel = (3, 3)
        self.filter_kernel = (3, 3)
        self.contour_mode = 0  # cv2.RETR_EXTERNAL
        self.contour_method = 2  # cv2.CHAIN_APPROX_SIMPLE
        self.num_vertices = 3

        self.contoured_txt = ''
        self.contour_tb_win = ''
        self.shaped_txt = ''
        self.shape_tb_win = ''

        self.manage_input()
        self.setup_trackbars()

    def manage_input(self):
        """
        Read the image file specified in the --input command line option and
        assign variable values accordingly. Shows input cv2 image and the
        grayscale.

        Returns: None
        """

        # utils.args_handler() has verified the image path, so read from it.
        self.input_img = cv2.imread(arguments['input'])
        # self.gray_img = cv2.cvtColor(self.input_img, cv2.COLOR_BGR2GRAY)
        self.gray_img = cv2.imread(arguments['input'], cv2.IMREAD_GRAYSCALE)

        # Ideas for scaling: https://stackoverflow.com/questions/52846474/
        #   how-to-resize-text-for-cv2-puttext-according-to-the-image-size-in-opencv-python
        size2scale = min(self.input_img.shape[0], self.input_img.shape[1])
        self.line_thickness = math.ceil(size2scale * const.LINE_SCALE * arguments['scale'])

        # Display starting images. Use WINDOW_GUI_NORMAL to fit any size
        #   image on screen and allow manual resizing of window.
        cv2.namedWindow(const.WIN_NAME['input+gray'],
                        flags=cv2.WINDOW_GUI_NORMAL)

        # Need to scale only images to display, not those to be processed.
        #   Default --scale arg is 1.0, so no scaling when option not used.
        input_img_scaled = utils.scale_img(self.input_img, arguments['scale'])
        gray_img_scaled = utils.scale_img(self.gray_img, arguments['scale'])
        side_by_side = cv2.hconcat(
            [input_img_scaled, cv2.cvtColor(gray_img_scaled, cv2.COLOR_GRAY2RGB)])
        cv2.imshow(const.WIN_NAME['input+gray'], side_by_side)

    def setup_trackbars(self) -> None:
        """
        All trackbars that go in a separate window of image processing
        settings.

        Returns: None
        """
        # Define names used in namedWindow().
        #   Names are also used in show_settings().
        # Set up two separate windows; one for threshold and contour
        # trackbars and reporting and one for polygon trackbars and reporting.
        if utils.MY_OS in 'lin, win':
            self.contour_tb_win = "Threshold & contour settings (dbl-click text to save)"
            self.shape_tb_win = 'Shape approximation settings (dbl-click text to save)'
        else:  # is macOS
            self.contour_tb_win = "Threshold & contour settings (rt-click text to save)"
            self.shape_tb_win = 'Shape approximation settings (rt-click text to save)'

        # Move the control window away from the processing windows.
        # Force each window positions to make them visible on startup.
        if utils.MY_OS == 'lin':
            cv2.namedWindow(self.contour_tb_win, flags=cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(self.contour_tb_win, 1000, 95)
        elif utils.MY_OS == 'dar':
            cv2.namedWindow(self.contour_tb_win)
            cv2.moveWindow(self.contour_tb_win, 500, 35)
        else:  # is Windows
            cv2.namedWindow(self.contour_tb_win, flags=cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow(self.contour_tb_win, 500, 500)

        cv2.setMouseCallback(self.contour_tb_win,
                             self.save_with_click)

        cv2.createTrackbar(const.TBNAME['_contrast'],
                           self.contour_tb_win,
                           100,
                           const.ALPHA_MAX,
                           self.alpha_selector)
        cv2.createTrackbar(const.TBNAME['_bright'],
                           self.contour_tb_win,
                           127,
                           const.BETA_MAX,
                           self.beta_selector)
        cv2.createTrackbar(const.TBNAME['_morph_op'],
                           self.contour_tb_win,
                           0,
                           3,
                           self.morphology_op_selector)
        cv2.createTrackbar(const.TBNAME['_morph_shape'],
                           self.contour_tb_win,
                           0,
                           2,
                           self.noise_redux_shape_selector)
        cv2.createTrackbar(const.TBNAME['_noise_k'],
                           self.contour_tb_win,
                           3,
                           20,
                           self.noise_redux_kernel_selector)
        cv2.createTrackbar(const.TBNAME['_noise_i'],
                           self.contour_tb_win,
                           1,
                           5,
                           self.noise_redux_iter_selector)
        cv2.setTrackbarMin(const.TBNAME['_noise_i'], self.contour_tb_win, 1)
        cv2.createTrackbar(const.TBNAME['_border'],
                           self.contour_tb_win,
                           0,
                           3,
                           self.border_selector)
        cv2.createTrackbar(const.TBNAME['_filter'],
                           self.contour_tb_win,
                           0,
                           3,
                           self.filter_type_selector)
        cv2.createTrackbar(const.TBNAME['_kernel_size'],
                           self.contour_tb_win,
                           3,
                           50,
                           self.filter_kernel_selector)
        cv2.setTrackbarMin(const.TBNAME['_kernel_size'], self.contour_tb_win, 1)
        cv2.createTrackbar(const.TBNAME['_thresh_type'],
                           self.contour_tb_win,
                           0,
                           1,
                           self.thresh_type_selector)
        cv2.createTrackbar(const.TBNAME['_contour_type'],
                           self.contour_tb_win,
                           1,
                           1,
                           self.contour_type_selector)
        cv2.createTrackbar(const.TBNAME['_contour_mode'],
                           self.contour_tb_win,
                           0,
                           1,
                           self.contour_mode_selector)
        cv2.createTrackbar(const.TBNAME['_contour_method'],
                           self.contour_tb_win,
                           2,
                           2,
                           self.contour_method_selector)
        cv2.setTrackbarMin(const.TBNAME['_contour_method'], self.contour_tb_win, 1)
        cv2.createTrackbar(const.TBNAME['_contour_min'],
                           self.contour_tb_win,
                           100,
                           1000,
                           self.contour_limit_selector)
        cv2.setTrackbarMin(const.TBNAME['_contour_min'], self.contour_tb_win, 1)

        # Place namedWindow for shapes here so that the contrast_img is
        #   first created.
        if utils.MY_OS == 'lin':
            cv2.namedWindow(self.shape_tb_win, flags=cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(self.shape_tb_win, 500, 500)
        elif utils.MY_OS == 'dar':
            cv2.namedWindow(self.shape_tb_win)
            cv2.moveWindow(self.shape_tb_win, 400, 100)
        else:  # is Windows
            cv2.namedWindow(self.shape_tb_win, flags=cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow(self.shape_tb_win, 400, 200)

        cv2.setMouseCallback(self.shape_tb_win,
                             self.save_with_click)

        # Trackbar to set num_vertices in select_shape().
        cv2.createTrackbar(const.TBNAME['_shape'],
                           self.shape_tb_win,
                           3,
                           11,
                           self.num_vertices_selector)
        cv2.setTrackbarMin(const.TBNAME['_shape'], self.shape_tb_win, 3)

        # Trackbar for cv2.approxPolyDP parameter.
        cv2.createTrackbar(const.TBNAME['_epsilon'],
                           self.shape_tb_win,
                           3,
                           18,
                           self.epsilon_selector)
        cv2.setTrackbarMin(const.TBNAME['_epsilon'], self.shape_tb_win, 1)

        # Trackbars for cv2.HoughCircles parameters:
        cv2.createTrackbar(const.TBNAME['_circle_it'],
                           self.shape_tb_win,
                           1,
                           1,
                           self.circle_img_selector)

        cv2.createTrackbar(const.TBNAME['_mindist'],
                           self.shape_tb_win,
                           1,
                           20,
                           self.mindist_selector)  # make value 10X.
        cv2.setTrackbarMin(const.TBNAME['_mindist'], self.shape_tb_win, 1)
        cv2.createTrackbar(const.TBNAME['_param1'],
                           self.shape_tb_win,
                           3,
                           5,
                           self.param1_selector)  # make value 100X.
        cv2.setTrackbarMin(const.TBNAME['_param1'], self.shape_tb_win, 1)
        cv2.createTrackbar(const.TBNAME['_param2'],
                           self.shape_tb_win,
                           9,
                           10,
                           self.param2_selector)  # make value 1/10X.
        cv2.setTrackbarMin(const.TBNAME['_param2'], self.shape_tb_win, 5)
        cv2.createTrackbar(const.TBNAME['_minradius'],
                           self.shape_tb_win,
                           2,
                           20,
                           self.minradius_selector)  # make value 10X.
        cv2.setTrackbarMin(const.TBNAME['_minradius'], self.shape_tb_win, 1)
        cv2.createTrackbar(const.TBNAME['_maxradius'],
                           self.shape_tb_win,
                           50,
                           100,
                           self.maxradius_selector)  # make value 10X.
        cv2.setTrackbarMin(const.TBNAME['_maxradius'], self.shape_tb_win, 1)

    def alpha_selector(self, a_val) -> None:
        """
        The "Contrast/gain/alpha" trackbar controller the provides the
        alpha parameter, as float, for cv2.convertScaleAbs() used to
        adjust image contrast.
        Called from setup_trackbars().
        Calls adjust_contrast() and contour_threshold().

        Args:
            a_val: The integer value passed from trackbar.

        Returns: None
        """
        # Info: https://stackoverflow.com/questions/39308030/
        #   how-do-i-increase-the-contrast-of-an-image-in-python-opencv
        self.alpha = a_val / 100
        self.adjust_contrast()
        self.contour_threshold()

    def beta_selector(self, b_val) -> None:
        """
        The "Brightness/bias/beta" trackbar controller that provides the
        beta parameter for cv2.convertScaleAbs() used to adjust image
        brightness.
        Called from setup_trackbars().
        Calls adjust_contrast() and contour_threshold().

        Args:
            b_val: The integer value passed from trackbar.

        Returns: None
        """
        # Info: https://stackoverflow.com/questions/39308030/
        #   how-do-i-increase-the-contrast-of-an-image-in-python-opencv

        self.beta = b_val - 127
        self.adjust_contrast()
        self.contour_threshold()

    def morphology_op_selector(self, op_val) -> None:
        """
        The "Reduce noise" trackbar controller for cv2.morphologyEx
        morphology operation cv2 constant value of the CV_MORPHOP
        dictionary.
        Called from setup_trackbars().
        Calls adjust_contrast() and contour_threshold().

        Args:
            op_val: The integer value passed from trackbar.

        Returns: None
        """
        self.morph_op = const.CV_MORPHOP[op_val]
        self.adjust_contrast()
        self.contour_threshold()

    def noise_redux_iter_selector(self, i_val) -> None:
        """
        The "Reduce noise" trackbar controller for cv2.morphologyEx
        iterations.
        Limits ksize tuple to integers greater than zero.
        Called from setup_trackbars().
        Calls adjust_contrast() and contour_threshold().

        Args:
            i_val: The integer value passed from trackbar.

        Returns: None
        """
        self.noise_iter = i_val
        self.adjust_contrast()
        self.contour_threshold()

    def noise_redux_shape_selector(self, s_val) -> None:
        """
        The "Reduce noise morphology shape" controller. Defines the
        shape parameter of cv2.getStructuringElement.
        The trackbar integer value corresponds to the cv2.MORPH_* constant
        integer.
        Called from setup_trackbars().
        Calls adjust_contrast() and contour_threshold().


        Args:
            s_val: The integer value passed from trackbar.

        Returns: none

        """
        self.morph_shape = s_val
        self.adjust_contrast()
        self.contour_threshold()

    def noise_redux_kernel_selector(self, k_val) -> None:
        """
        The "Reduce noise" trackbar controller for cv2.morphologyEx
        kernel size.
        Limits ksize tuple to odd integers to prevent shifting of the
        image.
        Called from setup_trackbars().
        Calls adjust_contrast() and contour_threshold().

        Args:
            k_val: The integer value passed from trackbar.

        Returns: None
        """
        val_k = k_val + 1 if k_val % 2 == 0 else k_val
        self.noise_kernel = (val_k, val_k)
        self.adjust_contrast()
        self.contour_threshold()

    def border_selector(self, bd_val):
        """
        The "Border type" trackbar controller to select a border type
        cv2 constant value of the CV_BORDER dictionary.
        Called from setup_trackbars().
        Calls contour_threshold().

        Args:
            bd_val: The integer value passed from trackbar.

        Returns:
        """
        self.border_type = const.CV_BORDER[bd_val]
        self.contour_threshold()

    def filter_type_selector(self, f_val) -> None:
        """
        The "Filter type" trackbar controller to select the filter used
        to blur the grayscale image.
        Called from setup_trackbars(). Calls contour_threshold().

        Args:
            f_val: The integer value passed from trackbar.

        Returns: None

        """
        self.filter_selection = const.FILTER[f_val]
        self.contour_threshold()

    def filter_kernel_selector(self, k_val) -> None:
        """
        The "Filter kernel" trackbar controller to assigns tuple kernel
        size to a particular filter type in filter_image(). Restricts all
        filter kernels to odd integers.
        Called from setup_trackbars(). Calls contour_threshold()

        Args:
            k_val: The integer value passed from trackbar.

        Returns: None
        """

        # cv2.GaussianBlur and cv2.medianBlur need to have odd kernels,
        #   but cv2.blur and cv2.bilateralFilter will shift image between
        #   even and odd kernels so just make everything odd.
        val_k = k_val + 1 if k_val % 2 == 0 else k_val
        self.filter_kernel = val_k, val_k
        self.contour_threshold()

    def thresh_type_selector(self, t_val) -> None:
        """
        The "Thresholding type" trackbar controller that assigns the
        cv2.threshold thresh parameter as a cv2.THRESH_* constant.

        Args:
            t_val: The integer value passed from trackbar.

        Returns: None
        """
        # The cv2.THRESH_OTSU integer constant is 8,
        #   cv2.THRESH_TRIANGLE is 16.
        self.th_type = cv2.THRESH_OTSU if t_val == 0 else cv2.THRESH_TRIANGLE
        self.contour_threshold()

    def contour_type_selector(self, ct_val) -> None:
        """
        The "Contour type" trackbar controller that assigns the
        contour type (area or arc length) for selecting contours.
        Called from setup_trackbars(). Calls contour_threshold().

        Args:
            ct_val: The integer value passed from trackbar.

        Returns: None
        """
        self.contour_type = const.CONTOUR_TYPE[ct_val]
        self.contour_threshold()

    def contour_mode_selector(self, mode_val):
        """
        The "contour find mode" trackbar controller that assigns the
        mode keyword parameter for cv2.findContours().
        Called from setup_trackbars(). Calls contour_threshold().

        Args:
            mode_val: The integer value passed from trackbar.

        Returns: None
        """

        # This simple assignment works b/c the value for the
        #  cv2.RETR__* constant matches that for any trackbar value
        #  (0 or 1).
        self.contour_mode = mode_val
        self.contour_threshold()

    def contour_method_selector(self, meth_val):
        """
        The "contour find method" trackbar controller that assigns the
        method keyword parameter for cv2.findContours().
        Called from setup_trackbars(). Calls contour_threshold().

        Args:
            meth_val: The integer value passed from trackbar.

        Returns: None
        """
        # This simple assignment works b/c the value for the
        #  cv2.CHAIN_APPROX_* constant matches that for any
        #  trackbar value; no need for a lookup.
        #  cv2.CHAIN_APPROX_NONE -> 1, cv2.CHAIN_APPROX_SIMPLE -> 2
        self.contour_method = meth_val
        self.contour_threshold()

    def contour_limit_selector(self, cl_val) -> None:
        """
        The "Contour size limit" trackbar controller that assigns the
        contour type (area or arc length) for selecting contours.
        Called from setup_trackbars(). Calls contour_threshold().

        Args:
            cl_val: The integer value passed from trackbar.

        Returns: None
        """
        self.contour_limit = cl_val
        self.contour_threshold()

    def epsilon_selector(self, e_val) -> None:
        """
        The "contour length" Trackbar controller that assigns the
        reduction factor of contour length to calculate epsilon keyword
        in cv2.approxPolyDP(). It provides the percentage of contour
        length to use as epsilon.

        Args:
            e_val: The integer value passed from trackbar.

        Returns: None
        """
        self.e_factor = e_val / 100 / 3
        self.contour_threshold()

    def num_vertices_selector(self, sh_val) -> None:
        """
        The "shape" trackbar controller that assigns the number of
        vertices to be found in a polygon from cv2.approxPolyDP().
        Called from setup_trackbars(). Calls contour_threshold().

        Args:
            sh_val: The integer value passed from trackbar.

        Returns: None
        """
        self.num_vertices = sh_val
        self.contour_threshold()

    def mindist_selector(self, mind_val) -> None:
        """
        The "Min distance" trackbar controller that assigns the minimum
        distance between circles found by cv2.HoughCircles().
        Called from setup_trackbars(). Calls contour_threshold().

        Args:
            mind_val: The integer value passed from trackbar.

        Returns: None
        """

        self.circles_mindist = mind_val * 10
        self.contour_threshold()

    def param1_selector(self, p1_val) -> None:
        """
        The "param1" trackbar controller that assigns that parameter
        value for cv2.HoughCircles().
        Called from setup_trackbars(). Calls contour_threshold().

        Args:
            p1_val: The integer value passed from trackbar.

        Returns: None
        """

        self.circles_param1 = p1_val * 100
        self.contour_threshold()

    def param2_selector(self, p2_val) -> None:
        """
        The "param2" trackbar controller that assigns that parameter
        value for cv2.HoughCircles().
        Called from setup_trackbars(). Calls contour_threshold().

        Args:
            p2_val: The integer value passed from trackbar.

        Returns: None
        """
        # NOTE: in opencv_python 4.7, but not 4.6, the cv2.HoughCircles
        # method param cv2.HOUGH_GRADIENT_ALT does not accept 1.0.
        self.circles_param2 = p2_val / 10
        self.contour_threshold()

    def minradius_selector(self, minr_val) -> None:
        """
        The "min radius" trackbar controller that assigns the minimum
        radius of circles found by cv2.HoughCircles().
        Called from setup_trackbars(). Calls contour_threshold().

        Args:
            minr_val: The integer value passed from trackbar.

        Returns: None
        """

        self.circles_min_radius = minr_val * 10
        self.contour_threshold()

    def maxradius_selector(self, maxr_val) -> None:
        """
        The "max radius" trackbar controller that assigns the maximum
        radius of circles found by cv2.HoughCircles().
        Called from setup_trackbars(). Calls contour_threshold().

        Args:
            maxr_val: The integer value passed from trackbar.

        Returns: None
        """

        self.circles_max_radius = maxr_val * 10
        self.contour_threshold()

    def circle_img_selector(self, c_val) -> None:
        """
        The "Find circles" trackbar controller that assigns the switch
        for which image array used by cv2.HoughCircles().
        Called from setup_trackbars(). Calls contour_threshold();
        alternately calls find_circles() if *c_val* is 11 (circles).

        Args:
            c_val: The integer value passed from trackbar.

        Returns: None
        """
        # Note: these strings need to match those used for conditions in
        #  find_circles().
        if c_val == 0:
            self.circle_img2use = 'threshold image'
        else:
            self.circle_img2use = 'filtered image'

        # Best to bypass the extra work done by contour_threshold() and
        #   go directly to find_circles() when available.
        if c_val == 11:  # is 'circle'
            self.find_circles()
        else:
            self.contour_threshold()

    def save_with_click(self, event, *args):
        """
        Click event in the namedWindow calls module that saves the image
        and settings.
        Calls utils.save_img_and_settings.
        Called by cv2.setMouseCallback event.

        Args:
            event: The implicit mouse event.
            args: Implicit return values from setMouseCallback(); not used.

        Returns: *event* as a formality.
        """

        #  For mouse buttons, double click doesn't work in macOS;
        #    rt-click does Frame menu in Linux, hence different actions.
        if utils.MY_OS in 'lin, win':
            mouse_event = cv2.EVENT_LBUTTONDBLCLK
        else:  # is macOS
            mouse_event = cv2.EVENT_RBUTTONDOWN

        if event == mouse_event:
            utils.save_img_and_settings(img2save=self.contoured_img,
                                        txt2save=self.contoured_txt,
                                        caller=f'{Path(__file__).stem}')
            utils.save_img_and_settings(self.shaped_img,
                                        self.shaped_txt,
                                        f'{Path(__file__).stem}')

        return event

    def adjust_contrast(self) -> None:
        """
        Adjust contrast of the hard-coded self.gray_img image.
        Updates contrast via alpha and beta Trackbar values.
        Calls reduce_noise(). Called from 6 *_selector() methods.

        Returns: None
        """

        # Source concepts:
        # https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
        # https://stackoverflow.com/questions/39308030/
        #   how-do-i-increase-the-contrast-of-an-image-in-python-opencv

        self.input_contrast_sd = int(self.gray_img.std())

        self.contrasted_img = cv2.convertScaleAbs(src=self.gray_img,
                                                  alpha=self.alpha,
                                                  beta=self.beta)
        self.curr_contrast_sd = int(self.contrasted_img.std())

        cv2.namedWindow(const.WIN_NAME['contrast+redux'],
                        flags=cv2.WINDOW_GUI_NORMAL)
        contrasted_scaled = utils.scale_img(self.contrasted_img, arguments['scale'])
        reduced_noise_scaled = utils.scale_img(self.reduce_noise(), arguments['scale'])
        side_by_side = cv2.hconcat([contrasted_scaled, reduced_noise_scaled])
        cv2.imshow(const.WIN_NAME['contrast+redux'], side_by_side)

    def reduce_noise(self) -> np.ndarray:
        """
        Reduce noise in grayscale image with erode and dilate actions of
        cv2.morphologyEx.
        Uses cv2.getStructuringElement params shape=self.morph_shape,
        ksize=self.noise_kernel.
        Uses cv2.morphologyEx params op=self.morph_op,
        kernel=<local structuring element>, iterations=self.noise_iter,
        borderType=self.border_type.
        Called from adjust_contrast().

        Returns: The array defined in adjust_contrast(), self.contrasted_img,
                with noise reduction.
        """

        # See: https://docs.opencv2.org/3.0-beta/modules/imgproc/doc/filtering.html
        #  on page, see: cv2.getStructuringElement(shape, ksize[, anchor])
        # see: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        element = cv2.getStructuringElement(shape=self.morph_shape,
                                            ksize=self.noise_kernel)

        # Use morphologyEx as a shortcut for erosion followed by dilation.
        #   MORPH_OPEN is useful to remove noise and small features.
        #   MORPH_HITMISS helps to separate close objects by shrinking them.
        # Read https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
        # https://theailearner.com/tag/cv2-morphologyex/

        self.reduced_noise_img = cv2.morphologyEx(src=self.contrasted_img,
                                                  op=self.morph_op,
                                                  kernel=element,
                                                  iterations=self.noise_iter,
                                                  borderType=self.border_type)
        return self.reduced_noise_img

    def filter_image(self) -> np.ndarray:
        """
        Applies a filter selection to blur the image for Canny edge
        detection or threshold contouring.
        Called from contour_threshold().

        Returns: The filtered (blurred) image array processed by
                 reduce_noise().
        """

        # Bilateral parameters:
        # https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html
        # from doc: Sigma values: For simplicity, you can set the 2 sigma
        #  values to be the same. If they are small (< 10), the filter
        #  will not have much effect, whereas if they are large (> 150),
        #  they will have a very strong effect, making the image look "cartoonish".
        # NOTE: The larger the sigma the greater the effect of kernel size d.
        self.sigma_color = int(np.std(self.reduced_noise_img))
        self.sigma_space = self.sigma_color

        # Gaussian parameters:
        # see: https://theailearner.com/tag/cv2-gaussianblur/
        self.sigma_x = int(self.reduced_noise_img.std())
        # NOTE: The larger the sigma the greater the effect of kernel size d.
        # sigmaY=0 also uses sigmaX. Matches Space to d if d>0.
        self.sigma_y = self.sigma_x

        # Apply a filter to blur edges:
        if self.filter_selection == 'cv2.bilateralFilter':
            self.filtered_img = cv2.bilateralFilter(src=self.reduced_noise_img,
                                                    # d=-1 or 0, is very CPU intensive.
                                                    d=self.filter_kernel[0],
                                                    sigmaColor=self.sigma_color,
                                                    sigmaSpace=self.sigma_space,
                                                    borderType=self.border_type)
        elif self.filter_selection == 'cv2.GaussianBlur':
            # see: https://dsp.stackexchange.com/questions/32273/
            #  how-to-get-rid-of-ripples-from-a-gradient-image-of-a-smoothed-image
            self.filtered_img = cv2.GaussianBlur(src=self.reduced_noise_img,
                                                 ksize=self.filter_kernel,
                                                 sigmaX=self.sigma_x,
                                                 sigmaY=self.sigma_y,
                                                 borderType=self.border_type)
        elif self.filter_selection == 'cv2.medianBlur':
            self.filtered_img = cv2.medianBlur(src=self.reduced_noise_img,
                                               ksize=self.filter_kernel[0])
        else:
            self.filtered_img = cv2.blur(src=self.reduced_noise_img,
                                         ksize=self.filter_kernel,
                                         borderType=self.border_type)

        cv2.namedWindow(const.WIN_NAME['filtered'],
                        flags=cv2.WINDOW_GUI_NORMAL)
        filtered_img_scaled = utils.scale_img(self.filtered_img, arguments['scale'])
        cv2.imshow(const.WIN_NAME['filtered'], filtered_img_scaled)

        return self.filtered_img

    def contour_threshold(self) -> None:
        """
        Identify object contours with cv2.threshold() and
        cv2.drawContours(). Threshold types limited to Otsu and Triangle.
        Called from all *_selector methods. Calls filter_image().

        Returns: None
        """
        # Note from doc: Currently, the Otsu's and Triangle methods
        #   are implemented only for 8-bit single-channel images.
        # OTSU & TRIANGLE computes thresh value, hence thresh=0 is replaced
        #   with the self.computed_threshold;
        #   for other cv2.THRESH_*, thresh needs to be manually provided.
        # Convert values above thresh (maxval) to white.
        self.computed_threshold, self.th_img = cv2.threshold(self.filter_image(),
                                                             thresh=0,
                                                             maxval=255,
                                                             type=self.th_type)

        found_contours, hierarchy = cv2.findContours(image=self.th_img,
                                                     mode=self.contour_mode,
                                                     method=self.contour_method)

        # Set values to exclude edge contours that may take in
        #  contrasting borders on the image; an arbitrary 80% exclusion limit.
        max_area = self.gray_img.shape[0] * self.gray_img.shape[1] * 0.64
        max_length = self.gray_img.shape[0] * 0.8

        # 'contour_type' values are from "Contour size type" trackbar.
        if self.contour_type == 'cv2.contourArea':
            selected_contours = [_c for _c in found_contours
                                 if max_area > cv2.contourArea(_c) >= self.contour_limit]
        else:  # is cv2.arcLength; aka "perimeter"
            selected_contours = [
                _c for _c in found_contours
                if max_length > cv2.arcLength(_c, closed=False) >= self.contour_limit
            ]

        # Used only for reporting.
        self.num_th_contours_all = len(found_contours)
        self.num_th_contours_select = len(selected_contours)

        self.contoured_img = self.input_img.copy()
        drawn_contours = cv2.drawContours(self.contoured_img,
                                          contours=selected_contours,
                                          contourIdx=-1,  # all contours.
                                          color=const.CBLIND_COLOR_CV['yellow'],
                                          thickness=self.line_thickness * 2,
                                          lineType=cv2.LINE_AA)

        cv2.namedWindow(const.WIN_NAME['th+contours'],
                        flags=cv2.WINDOW_GUI_NORMAL)
        th_img_scaled = utils.scale_img(self.th_img, arguments['scale'])
        contours_scaled = utils.scale_img(drawn_contours, arguments['scale'])
        side_by_side = cv2.hconcat(
            [cv2.cvtColor(th_img_scaled, cv2.COLOR_GRAY2RGB), contours_scaled])
        cv2.imshow(const.WIN_NAME['th+contours'], side_by_side)

        self.select_shape(selected_contours)

    def select_shape(self, selected_contour_list: list) -> None:
        """
        Filter contoured objects of a specific approximated shape.
        Called from contour_threshold().
        Calls contour_shapes() with selected polygon contours.

        Args:
            selected_contour_list: List of selected contours from cv2.findContours.

        Returns: None
        """
        # inspiration: Adrian Rosebrock
        #  https://pyimagesearch.com/2016/02/08/opencv-shape-detection/

        self.polygon = 'None found'
        selected_polygon_contours = []

        # Note: num_vertices value is assigned by num_vertices_selector().
        # Shape trackbar value of 11 is for circles.
        if self.num_vertices == 11:
            self.polygon = 'circle'
            self.find_circles()
        else:
            for _c in selected_contour_list:
                len_contour = cv2.arcLength(_c, True)
                approx_poly = cv2.approxPolyDP(curve=_c,
                                               epsilon=self.e_factor * len_contour,
                                               closed=True)
                if len(approx_poly) == self.num_vertices == 3:
                    selected_polygon_contours.append(_c)
                elif len(approx_poly) == self.num_vertices == 4:
                    # Compute the bounding box of the contour and use the
                    #   bounding box to compute the aspect ratio.
                    # (_x, _y, _w, _h) = cv2.boundingRect(approx_poly)
                    # _ar = _w / float(_h)
                    # A square will have an aspect ratio that is approximately
                    #   equal to one, otherwise, the shape is a rectangle.
                    # self.polygon = "square" if 0.95 <= _ar <= 1.05 else "rectangle"
                    # self.polygon = 'rectangle'
                    selected_polygon_contours.append(_c)
                elif len(approx_poly) == self.num_vertices == 5:
                    selected_polygon_contours.append(_c)
                elif len(approx_poly) == self.num_vertices == 6:
                    selected_polygon_contours.append(_c)
                elif len(approx_poly) == self.num_vertices == 7:
                    selected_polygon_contours.append(_c)
                elif len(approx_poly) == self.num_vertices == 8: # and cv2.isContourConvex(_c):
                    selected_polygon_contours.append(_c)
                elif len(approx_poly) == self.num_vertices == 9:
                    selected_polygon_contours.append(_c)
                elif len(approx_poly) == self.num_vertices == 10 and not cv2.isContourConvex(_c):
                    selected_polygon_contours.append(_c)

            self.polygon = const.SHAPE_NAME[self.num_vertices]

            # num_shapes is used only for reporting.
            self.num_shapes = len(selected_polygon_contours)

            self.contour_shapes(selected_polygon_contours)

    def contour_shapes(self, contours: list) -> None:
        """
        Draw *contours* around detected polygon or circle.
        Calls show_settings(). Called from select_shape()

        Args:
            contours: Contour list of polygons or circles.

        Returns: None
        """
        self.shaped_img = self.input_img.copy()
        cv2.namedWindow(const.WIN_NAME['shape'],
                        flags=cv2.WINDOW_GUI_NORMAL)

        if contours:
            for _c in contours:
                # Compute the center of the contour, as a circle.
                # (_x, _y), radius = cv2.minEnclosingCircle(_c)
                # center = (int(_x), int(_y))

                # NOTE: When no recognized polygon contour is sent here,
                #  no window will pop up with drawn contours; the settings
                #  window will indicate polygon selected as "None found".
                #  Only when the polygon trackbar finds a match will the
                #  window pop up.

                cv2.drawContours(self.shaped_img,
                                 contours=[_c],
                                 contourIdx=-1,
                                 color=const.CBLIND_COLOR_CV['yellow'],
                                 thickness=self.line_thickness * 2,
                                 lineType=cv2.LINE_AA
                                 )

                # Show the input image with outline of selected polygon.
                shapes_scaled = utils.scale_img(self.shaped_img, arguments['scale'])
                cv2.imshow(const.WIN_NAME['shape'], shapes_scaled)
        else:  # contours parameter is None, b/c selected_polygon_contours = [].
            input_img_scaled = utils.scale_img(self.input_img, arguments['scale'])
            cv2.imshow(const.WIN_NAME['shape'], input_img_scaled)

        # Now update the settings text with current values.
        self.show_settings()

    def find_circles(self):
        """
        Implements the cv2.HOUGH_GRADIENT_ALT method of cv2.HoughCircles()
        to approximate circles in a filtered/blured threshold image, then
        displays them on the input image.
        Called from select_shape(). Calls utils.text_array().

        Returns: An array of HoughCircles contours.
        """
        self.shaped_img = self.input_img.copy()

        # Note: these strings need to match those used in circle_img_selector().
        #   defined in contour_threshold().
        if self.circle_img2use == 'threshold image':
            _, circle_this_img = cv2.threshold(self.filter_image(),
                                               thresh=0,
                                               maxval=255,
                                               type=self.th_type)

            # Here HoughCircles works on the threshold image, not found
            #  contours, so need to replace selected threshold contours image
            #  with a blank image so the user knows that contour trackbars
            #  do nothing to find circles.
            circle_this_img_scaled = utils.scale_img(circle_this_img, arguments['scale'])
            side_by_side = cv2.hconcat([circle_this_img_scaled,
                                        np.ones(circle_this_img_scaled.shape,
                                                dtype='uint8')])
            cv2.imshow(const.WIN_NAME['th+contours'], side_by_side)

        else:  # is "filtered image"
            circle_this_img = self.filtered_img
            # Here HoughCircles works on the filtered image, not threshold,
            #  so replace the threshold and contour images with a message.
            text_msg = ("Circles are now being found\n "
                        "   using the filtered image,\n"
                        "   not the threshold image.")
            cv2.imshow(const.WIN_NAME['th+contours'],
                       utils.text_array((220, 350), text_msg))

        # source: https://www.geeksforgeeks.org/circle-detection-using-opencv-python/
        # https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
        # Apply Hough transform on the filtered (blured) image.
        # Docs general recommendations for HOUGH_GRADIENT_ALT with good image contrast:
        #    param1=300, param2=0.9, minRadius=20, maxRadius=400
        found_circles = cv2.HoughCircles(image=circle_this_img,
                                         method=cv2.HOUGH_GRADIENT_ALT,
                                         dp=1.5,
                                         minDist=self.circles_mindist,
                                         param1=self.circles_param1,
                                         param2=self.circles_param2,
                                         minRadius=self.circles_min_radius,
                                         maxRadius=self.circles_max_radius)

        if found_circles is not None:
            # Convert the circle parameters to integers to get the right data type.
            found_circles = np.uint16(np.round(found_circles))

            self.num_shapes = len(found_circles[0, :])

            for _circle in found_circles[0, :]:
                _x, _y, _r = _circle

                # Draw the circumference of the found circle.
                cv2.circle(self.shaped_img,
                           center=(_x, _y),
                           radius=_r,
                           color=const.CBLIND_COLOR_CV['yellow'],
                           thickness=self.line_thickness * 2,
                           lineType=cv2.LINE_AA
                           )
                # Draw its center.
                cv2.circle(self.shaped_img,
                           center=(_x, _y),
                           radius=4,
                           color=const.CBLIND_COLOR_CV['yellow'],
                           thickness=self.line_thickness * 2,
                           lineType=cv2.LINE_AA
                           )

                # Show found circles marked on the input image.
                circles_scaled = utils.scale_img(self.shaped_img, arguments['scale'])
                cv2.imshow(const.WIN_NAME['shape'], circles_scaled)
        else:
            shapes_scaled = utils.scale_img(self.input_img, arguments['scale'])
            cv2.imshow(const.WIN_NAME['shape'], shapes_scaled)

        # Now update the settings text with current values.
        self.show_settings()

    def show_settings(self) -> None:
        """
        Display name of file and processing parameters in contour_tb_win
        window. Displays real-time changes to parameter values.
        Calls module utils.text_array() in contour_modules directory.
        Called from select_shape().

        Returns: None
        """

        # Need extra text to list parameter values for bilateral and gaussian FILTER.
        if self.filter_selection == 'cv2.bilateralFilter':
            filter_sigmas = (f'd={self.filter_kernel[0]},'
                             f' sigmaColor={self.sigma_color},'
                             f' sigmaSpace={self.sigma_space}')
        elif self.filter_selection == 'cv2.GaussianBlur':
            filter_sigmas = (f'sigmaX={self.sigma_x},'
                             f' sigmaY={self.sigma_y}')
        else:
            filter_sigmas = ''

        epsilon_pct = round(self.e_factor * 100, 2)

        # Text is formatted for clarity in window, terminal, and saved file.
        the_text = (
            f'Image: {arguments["input"]} (alpha SD: {self.input_contrast_sd})\n'
            f'{"Contrast:".ljust(20)}convertScaleAbs alpha={self.alpha},'
            f' beta={self.beta}\n'
            f'{" ".ljust(20)}(adjusted alpha SD {self.curr_contrast_sd})\n'
            f'{"Noise reduction:".ljust(20)}cv2.getStructuringElement ksize={self.noise_kernel},\n'
            f'{" ".ljust(20)}cv2.getStructuringElement shape={const.MORPH_SHAPE[self.morph_shape]}\n'
            f'{" ".ljust(20)}cv2.morphologyEx iterations={self.noise_iter}\n'
            f'{" ".ljust(20)}cv2.morphologyEx op={const.MORPH_TYPE[self.morph_op]},\n'
            f'{" ".ljust(20)}cv2.morphologyEx borderType={const.BORDER_NAME[self.border_type]}\n'
            f'{"Filter:".ljust(20)}{self.filter_selection} ksize={self.filter_kernel}\n'
            f'{" ".ljust(20)}borderType={const.BORDER_NAME[self.border_type]}\n'
            f'{" ".ljust(20)}{filter_sigmas}\n'  # is blank line for box and median.
            f'{"cv2.threshold".ljust(20)}type={const.TH_TYPE[self.th_type]},'
            f' thresh={int(self.computed_threshold)}, maxval=255\n'
            f'{"cv2.findContours".ljust(20)}mode={const.CONTOUR_MODE[self.contour_mode]}\n'
            f'{" ".ljust(20)}method={const.CONTOUR_METHOD[self.contour_method]}\n'
            f'{"Contour size type:".ljust(20)}{self.contour_type}\n'
            f'{"Contour size min.:".ljust(20)}{self.contour_limit} pixels\n'
            f'{"Contours selected:".ljust(20)}{self.num_th_contours_select}'
            f' (from {self.num_th_contours_all} total)'
        )

        shape_txt = (
            f'{"cv2.approxPolyDP".ljust(20)}epsilon={epsilon_pct}% contour length\n'
            f'{" ".ljust(20)}closed=True\n'
            f'{"cv2.HoughCircles".ljust(20)}image={self.circle_img2use}\n'
            f'{" ".ljust(20)}method=cv2.HOUGH_GRADIENT_ALT\n'
            f'{" ".ljust(20)}dp=1.5\n'
            f'{" ".ljust(20)}minDist={self.circles_mindist}\n'
            f'{" ".ljust(20)}param1={self.circles_param1}\n'
            f'{" ".ljust(20)}param2={self.circles_param2}\n'
            f'{" ".ljust(20)}minRadius={self.circles_min_radius}\n'
            f'{" ".ljust(20)}maxRadius={self.circles_max_radius}\n'
            f'{"Shape selected:".ljust(20)}{self.polygon}, found: {self.num_shapes}\n'
        )

        # Put text into contoured_txt for printing and saving to file.
        self.contoured_txt = the_text
        self.shaped_txt = shape_txt

        # Need to set the dimensions of the settings area to fit all text.
        #   Font style parameters are set in constants.py module.
        if utils.MY_OS == 'lin':
            settings_img = utils.text_array((420, 650), the_text)
        elif utils.MY_OS == 'dar':
            settings_img = utils.text_array((350, 620), the_text)
        else:  # is Windows
            settings_img = utils.text_array((820, 1200), the_text)

        if utils.MY_OS == 'lin':
            shape_settings_img = utils.text_array((280, 460), shape_txt)
        elif utils.MY_OS == 'dar':
            shape_settings_img = utils.text_array((240, 430), shape_txt)
        else:  # is Windows
            shape_settings_img = utils.text_array((490, 700), shape_txt)

        cv2.imshow(self.contour_tb_win, settings_img)
        cv2.imshow(self.shape_tb_win, shape_settings_img)


if __name__ == "__main__":
    # Program exits here, with msg, if system platform or Python version
    #  check fails.
    utils.check_platform()
    vcheck.minversion('3.7')

    # All checks are good, so grab as a 'global' the dictionary for
    #   command line argument values.
    arguments = utils.args_handler()

    PI = ProcessImage()
    print(f'{Path(__file__).name} is now running...')

    # Set infinite loop with sigint handler to monitor "quit"
    #  keystrokes.
    quit_thread = threading.Thread(target=utils.quit_keys(), daemon=True)
    quit_thread.start()
