#!/usr/bin/env python3
"""Use OpenCV to explore image processing parameters involved in
identifying objects and drawing contours. Parameter values are adjusted
with slide bars.

USAGE Example command lines, from within the image-processor-main folder:
python3 -m thresh_it --help
python3 -m thresh_it --about
python3 -m thresh_it --input images/sample1.jpg
python3 -m thresh_it -i images/sample2.jpg

Quit program with Esc or Q key or from command line with Ctrl-C.
Save any selected window with Ctrl-S or from rt-click menu.
Save settings and contour image with the Save/Print slide bar.

Requires Python3.7 or later and the packages opencv-python and numpy.
Developed in Python 3.8-3.9.
"""

# Copyright (C) 2022-2023 C.S. Echt, under GNU General Public License

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
        contour_type_selector
        contour_mode_selector
        contour_method_selector
        contour_limit_selector
        save_with_click
        adjust_contrast
        reduce_noise
        filter_image
        contour_threshold
        circle_the_contours
        show_settings
    """

    __slots__ = ('alpha', 'beta', 'border_type', 'computed_threshold',
                 'contour_limit', 'contour_type', 'contrasted_img',
                 'curr_contrast_sd', 'reduced_noise_img',
                 'filter_kernel', 'filter_selection',
                 'gray_img', 'morph_op', 'morph_shape',
                 'noise_iter', 'noise_kernel', 'num_th_contours_all',
                 'num_th_contours_select', 'input_contrast_sd', 'input_img',
                 'contoured_img', 'contoured_txt', 'contour_tb_win',
                 'sigma_color', 'sigma_space', 'sigma_x', 'sigma_y',
                 'th_type', 'thresh',
                 'font_scale', 'line_thickness', 'center_xoffset',
                 'contour_mode', 'contour_method', 'show_hull',
                 )

    def __init__(self):

        # The np.ndarray arrays for images to be processed.
        self.input_img = None
        self.gray_img = None
        self.contoured_img = None
        self.contrasted_img = None
        self.reduced_noise_img = None
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
        self.show_hull = ''

        self.font_scale = 0
        self.line_thickness = 0
        self.center_xoffset = 0

        # Need to set starting values for variables set by some trackbars
        # for faster program startup.
        self.noise_kernel = (3, 3)
        self.filter_kernel = (3, 3)
        self.contour_mode = 0  # cv2.RETR_EXTERNAL
        self.contour_method = 2  # cv2.CHAIN_APPROX_SIMPLE

        self.contoured_txt = ''
        self.contour_tb_win = ''

        self.manage_input()
        self.setup_trackbars()

    def manage_input(self) -> None:
        """
        Read the image file specified in the --input command line option and
        assign variable values accordingly. Shows input cv2 image and the
        grayscale.

        Returns: None
        """

        # utils.args_handler() has verified the image path, so read from it.
        self.input_img = cv2.imread(arguments['input'])
        self.gray_img = cv2.imread(arguments['input'], cv2.IMREAD_GRAYSCALE)

        # Ideas for scaling: https://stackoverflow.com/questions/52846474/
        #   how-to-resize-text-for-cv2-puttext-according-to-the-image-size-in-opencv-python
        size2scale = min(self.input_img.shape[0], self.input_img.shape[1])
        self.font_scale = size2scale * const.FONT_SCALE
        self.font_scale = max(self.font_scale, 0.5)
        self.line_thickness = math.ceil(size2scale * const.LINE_SCALE * arguments['scale'])
        self.center_xoffset = math.ceil(size2scale * const.CENTER_XSCALE * arguments['scale'])

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
        self.contour_tb_win = "Threshold & contour settings (dbl-click text to save)"

        # Move the control window away from the processing windows.
        # Linux Ubuntu: Place window near right edge of screen.
        cv2.namedWindow(self.contour_tb_win, flags=cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(self.contour_tb_win, 1000, 35)

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
                           0,
                           1,
                           self.contour_type_selector)
        cv2.createTrackbar(const.TBNAME['_contour_mode'],
                           self.contour_tb_win,
                           0,
                           1,
                           self.contour_mode_selector)
        cv2.createTrackbar(const.TBNAME['_contour_method'],
                           self.contour_tb_win,
                           1,
                           2,
                           self.contour_method_selector)
        cv2.setTrackbarMin(const.TBNAME['_contour_method'], self.contour_tb_win, 1)
        cv2.createTrackbar(const.TBNAME['_hull'],
                           self.contour_tb_win,
                           0,
                           1,
                           self.hull_selector)
        cv2.createTrackbar(const.TBNAME['_contour_min'],
                           self.contour_tb_win,
                           100,
                           1000,
                           self.contour_limit_selector)
        cv2.setTrackbarMin(const.TBNAME['_contour_min'], self.contour_tb_win, 1)

    def alpha_selector(self, a_val) -> None:
        """
        The "Contrast/gain/alpha" trackbar controller the provides the
        alpha parameter, as float, for cv2.convertScaleAbs() used to
        adjust image contrast.
        Called from setup_trackbars(). Calls contour_threshold(),
        reduce_noise(), and contour_threshold().

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
        Called from setup_trackbars(). Calls contour_threshold(),
        reduce_noise(), and contour_threshold().

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
        Called from setup_trackbars(). Calls contour_threshold(),
        reduce_noise(), and contour_threshold().

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
        Called from setup_trackbars(). Calls contour_threshold(),
        reduce_noise(), and contour_threshold().

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
        image. Called from setup_trackbars(). Calls contour_threshold(),
        reduce_noise(), and contour_threshold().

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
        Called from setup_trackbars(). Calls contour_threshold(),
        reduce_noise(), and contour_threshold().

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
        # The Ostu integer constant is 8, triangle is 16.
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
        #  trackbar value. Reassign 0 value to 1.
        # meth_val = 1 if meth_val == 0 else meth_val
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

    def hull_selector(self, h_val) -> None:
        """
        The "Show hull" trackbar controller that determines whether to
        draw cv2.convexHull on selected contours.
        Called from setup_trackbars(). Calls contour_threshold().

        Args:
            h_val: The integer value passed from trackbar.

        Returns: None
        """

        self.show_hull = h_val
        self.contour_threshold()

    def save_with_click(self, event, *args):
        """
        Click on the namedWindow calls module that saves the image and
        settings.
        Calls utils.save_img_and_settings.
        Called by cv2.setMouseCallback event.

        Args:
            event: The implicit mouse event.
            *args: Return values from setMouseCallback(); not used here.

        Returns: *event* as a formality.
        """

        if event == cv2.EVENT_LBUTTONDBLCLK:
            utils.save_img_and_settings(img2save=self.contoured_img,
                                        txt2save=self.contoured_txt,
                                        caller=f'{Path(__file__).stem}')

        return event

    def adjust_contrast(self) -> None:
        """
        Adjust contrast of the hard-coded self.gray_img image.
        Updates contrast via alpha and beta Trackbar values.

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

        contrasted_scaled = utils.scale_img(self.contrasted_img, arguments['scale'])
        reduced_noise_scaled = utils.scale_img(self.reduce_noise(), arguments['scale'])

        cv2.namedWindow(const.WIN_NAME['contrast+redux'],
                        flags=cv2.WINDOW_GUI_NORMAL)
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
            filtered_img = cv2.bilateralFilter(src=self.reduced_noise_img,
                                               # d=-1 or 0, is very CPU intensive.
                                               d=self.filter_kernel[0],
                                               sigmaColor=self.sigma_color,
                                               sigmaSpace=self.sigma_space,
                                               borderType=self.border_type)
        elif self.filter_selection == 'cv2.GaussianBlur':
            # see: https://dsp.stackexchange.com/questions/32273/
            #  how-to-get-rid-of-ripples-from-a-gradient-image-of-a-smoothed-image
            filtered_img = cv2.GaussianBlur(src=self.reduced_noise_img,
                                            ksize=self.filter_kernel,
                                            sigmaX=self.sigma_x,
                                            sigmaY=self.sigma_y,
                                            borderType=self.border_type)
        elif self.filter_selection == 'cv2.medianBlur':
            filtered_img = cv2.medianBlur(src=self.reduced_noise_img,
                                          ksize=self.filter_kernel[0])
        elif self.filter_selection == 'cv2.blur':
            filtered_img = cv2.blur(src=self.reduced_noise_img,
                                    ksize=self.filter_kernel,
                                    borderType=self.border_type)
        else:
            filtered_img = cv2.blur(src=self.reduced_noise_img,
                                    ksize=self.filter_kernel,
                                    borderType=self.border_type)

        cv2.namedWindow(const.WIN_NAME['filtered'],
                        flags=cv2.WINDOW_GUI_NORMAL)
        filtered_img_scaled = utils.scale_img(filtered_img, arguments['scale'])
        cv2.imshow(const.WIN_NAME['filtered'], filtered_img_scaled)

        return filtered_img

    def contour_threshold(self) -> None:
        """
        Identify object contours with cv2.threshold() and
        cv2.drawContours(). Threshold types limited to Otsu and Triangle.

        Returns: None
        """
        # https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
        # Check out CLOSED contours and convexHull:
        # https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
        # https://towardsdatascience.com/clahe-and-thresholding-in-python-3bf690303e40

        # Thresholding with OTSU works best with a blurring filter applied to
        #   image, like Gaussian or Bilateral
        # see: https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
        # https://theailearner.com/tag/cv2-thresh_otsu/  # good
        # see: https://stackabuse.com/opencv-thresholding-in-python-with-cv2threshold/
        # see: https://pyimagesearch.com/2021/04/28/opencv-thresholding-cv2-threshold/
        # see: https://towardsdatascience.com/image-segmentation-part-1-6e0ddb5f7f8a
        # see: https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html

        # Note from doc: Currently, the Otsu's and Triangle methods
        #   are implemented only for 8-bit single-channel images.
        # OTSU & TRIANGLE computes thresh value, hence thresh=0 is replaced
        #   with the self.computed_threshold;
        #   for other cv2.THRESH_*, thresh needs to be manually provided.
        # Convert values above thresh to white.
        self.computed_threshold, th_img = cv2.threshold(src=self.filter_image(),
                                                        thresh=0,
                                                        maxval=255,
                                                        type=self.th_type)

        found_contours, hierarchy = cv2.findContours(image=th_img,
                                                     mode=self.contour_mode,
                                                     method=self.contour_method)

        # Set values to exclude threshold contours that may include
        #  contrasting borders on the image; an arbitrary 90% length
        #  limit, 81% area limit.
        max_area = self.gray_img.shape[0] * self.gray_img.shape[1] * 0.81
        max_length = max(self.gray_img.shape[0], self.gray_img.shape[1]) * 0.9

        # 'contour_type' values are from "Contour size type" trackbar.
        if self.contour_type == 'cv2.contourArea':
            selected_contours = [
                _c for _c in found_contours
                if max_area > cv2.contourArea(_c) >= self.contour_limit]
        else:  # type is cv2.arcLength; aka "perimeter"
            selected_contours = [
                _c for _c in found_contours
                if max_length > cv2.arcLength(_c, closed=False) >= self.contour_limit]

        # Used only for reporting.
        self.num_th_contours_all = len(found_contours)
        self.num_th_contours_select = len(selected_contours)

        self.contoured_img = self.input_img.copy()

        # Draw hulls around selected contours when hull area is more than
        #   10% of contour area. This prevents obfuscation of drawn lines
        #   when hulls and contours are similar. 10% limit is arbitrary.
        if self.show_hull:
            hull_list = []
            for i, _ in enumerate(selected_contours):
                hull = cv2.convexHull(selected_contours[i])
                if cv2.contourArea(hull) >= cv2.contourArea(selected_contours[i]) * 1.1:
                    hull_list.append(hull)

            cv2.drawContours(self.contoured_img,
                             contours=hull_list,
                             contourIdx=-1,  # all hulls.
                             color=const.CBLIND_COLOR_CV['sky blue'],
                             thickness=self.line_thickness * 3,
                             lineType=cv2.LINE_AA)

        drawn_contours = cv2.drawContours(self.contoured_img,
                                          contours=selected_contours,
                                          contourIdx=-1,  # all contours.
                                          color=const.CBLIND_COLOR_CV['yellow'],
                                          thickness=self.line_thickness * 2,
                                          lineType=cv2.LINE_AA)

        cv2.namedWindow(const.WIN_NAME['th+contours'],
                        flags=cv2.WINDOW_GUI_NORMAL)
        th_img_scaled = utils.scale_img(th_img, arguments['scale'])
        contours_scaled = utils.scale_img(drawn_contours, arguments['scale'])
        side_by_side = cv2.hconcat(
            [cv2.cvtColor(th_img_scaled, cv2.COLOR_GRAY2RGB), contours_scaled])
        cv2.imshow(const.WIN_NAME['th+contours'], side_by_side)

        self.circle_the_contours(selected_contours)
        self.show_settings()

    def circle_the_contours(self, contour_list: list) -> None:
        """
        Draws a circles around contoured objects. Objects are expected
        to be oblong so that circle diameter can represent object length.
        Args:
            contour_list: List of selected contours from cv2.findContours.

        Returns: None
        """

        self.contoured_img = self.input_img.copy()

        for _c in contour_list:
            (_x, _y), radius = cv2.minEnclosingCircle(_c)
            center = (int(_x), int(_y))
            radius = int(radius)
            cv2.circle(self.contoured_img,
                       center=center,
                       radius=radius,
                       color=const.CBLIND_COLOR_CV['yellow'],
                       thickness=self.line_thickness * 2,
                       lineType=cv2.LINE_AA)

            # Display pixel diameter of each circled contour.
            #  Draw a filled black circle to use for text background.
            cv2.circle(self.contoured_img,
                       center=center,
                       radius=int(radius * 0.5),
                       color=(0, 0, 0),
                       thickness=-1,
                       lineType=cv2.LINE_AA)

            cv2.putText(img=self.contoured_img,
                        text=f'{radius * 2}px',
                        # Center text in the enclosing circle, scaled by px size.
                        org=(center[0] - self.center_xoffset, center[1] + 5),
                        fontFace=const.FONT_TYPE,
                        fontScale=self.font_scale,
                        color=const.CBLIND_COLOR_CV['yellow'],
                        thickness=self.line_thickness,
                        lineType=cv2.LINE_AA)  # LINE_AA is anti-aliased

        # cv2.mEC returns circled radius of contour as last element.
        # dia_list = [cv2.minEnclosingCircle(_c)[-1] * 2 for _c in selected_contour_list]
        # mean_size = round(mean(dia_list), 1) if dia_list else 0
        # print('mean threshold dia', mean_size)

        cv2.namedWindow(const.WIN_NAME['id_objects'],
                        flags=cv2.WINDOW_GUI_NORMAL)
        cv2.imshow(const.WIN_NAME['id_objects'], self.contoured_img)

    def show_settings(self) -> None:
        """
        Display name of file and processing parameters in contour_tb_win
        window. Displays real-time changes to parameter values.
        Calls module utils.text_array() in contour_modules directory.

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

        # Put text into contoured_txt for printing and saving to file.
        self.contoured_txt = the_text

        # Need to set the dimensions of the settings area to fit all text.
        #   Font style parameters are set in constants.py module.
        settings_img = utils.text_array((420, 620), the_text)
        cv2.imshow(self.contour_tb_win, settings_img)


if __name__ == "__main__":
    # Program exits here if system platform or Python version check fails.
    vcheck.minversion('3.7')

    # All checks are good, so grab as a 'global' the dictionary for
    #   command line argument values.
    arguments = utils.args_handler()

    PI = ProcessImage()
    print(f'{Path(__file__).name} is now running...\n',
          'Quit program with Esc or Q key, or Ctrl-C from Terminal.\n')

    # Set infinite loop with sigint handler to monitor "quit"
    #  keystrokes.
    quit_thread = threading.Thread(target=utils.quit_keys(), daemon=True)
    quit_thread.start()
