#!/usr/bin/env python3
"""Use OpenCV to explore image processing parameters involved in
histogram equalization with CLAHE. Parameter values are adjusted
with slide bars. Color images are converted to grayscale prior to
equalization. Intended as an optional image pre-processor for
thresh_it.py and edge_it.py to identify object contours.

USAGE Example command lines, from within the image-processor-main folder:
python3 -m equalize_it --help
python3 -m equalize_it --about
python3 -m equalize_it --input images/sample2.jpg

If not working as expected, try one of the alternative GUI implementations,
equalize_tk.py (tkinter) or equalize_qt.py (PyQt5)

Quit program with Esc or Q key; may need to first select a window other
than Histograms. Or quit from command line with Ctrl-C.

Requires Python3.7 or later and the packages opencv-python and numpy.
Developed in Python 3.8-3.9.
"""
# Copyright (C) 2022 C.S. Echt, under GNU General Public License

# Standard library imports.
import sys
import threading

from pathlib import Path

# Third party imports.
try:
    import cv2
    import matplotlib
    import numpy as np

    from matplotlib import pyplot as plt
except (ImportError, ModuleNotFoundError) as import_err:
    print('*** OpenCV, Matplotlib or Numpy  was not found or needs an update:\n\n'
          'To install: from the current folder, run this command'
          ' for the Python package installer (PIP):\n'
          '   python3 -m pip install -r requirements.txt\n\n'
          'Alternative command formats (system dependent):\n'
          '   py -m pip install -r requirements.txt (Windows)\n'
          '   pip install -r requirements.txt\n\n'
          'A package may already be installed, but needs an update;\n'
          '   this may be the case when the error message (below) is a bit cryptic\n'
          '   Example update command:\n'
          '   python3 -m pip install -U matplotlib\n'
          'On Linux, if tkinter is the problem, then you may need to run:\n'
          '   sudo apt-get install python3-tk\n'
          '   See also: https://tkdocs.com/tutorial/install.html \n\n'
          f'Error message:\n{import_err}')
    sys.exit(1)

# Local application imports
from contour_modules import (vcheck,
                             utils,
                             constants as const,
                             )


class ProcessImage:
    """
    A suite of methods for applying OpenCV histogram equalization with
    CLAHE.

    Methods:
        apply_clahe
        clip_selector
        manage_input
        save_with_click
        setup_trackbars
        show_histograms
        show_settings
        tile_selector

    """
    __slots__ = ('clahe_img', 'clahe_mean', 'clahe_sd', 'clip_limit',
                 'flat_gray_array', 'gray_img', 'input_img', 'input_mean',
                 'input_sd', 'settings_txt',
                 'settings_win', 'tile_size',
                 )

    def __init__(self):

        # The np.ndarray arrays for images to be processed.
        self.input_img = None
        self.gray_img = None
        self.flat_gray_array = None
        self.clahe_img = None

        # Image processing parameters amd metrics.
        self.clip_limit = 2.0  # Default trackbar value.
        self.tile_size = (8, 8)  # Default trackbar value.
        self.input_sd = 0
        self.input_mean = 0
        self.clahe_sd = 0
        self.clahe_mean = 0

        self.settings_txt = ''
        self.settings_win = ''

        # Need to get_backend for Windows.
        matplotlib.get_backend()
        plt.ion()
        plt.style.use(('bmh', 'fast'))

        self.manage_input()
        self.setup_trackbars()
        # NOTE that setup_trackbars sets starting pos other than zero,
        #  which creates a call event for apply_clahe()

    def manage_input(self):
        """
        Reads input images, creates grayscale image and its flattened
        array, adjusts displayed image size, displays input and grayscale
        side-by-side in one window.

        Returns: None
        """

        # utils.args_handler() has verified image path, so read from it.
        self.input_img = cv2.imread(arguments['input'])
        self.gray_img = cv2.cvtColor(self.input_img, cv2.COLOR_BGR2GRAY)
        self.flat_gray_array = self.gray_img.ravel()

        cv2.namedWindow(const.WIN_NAME['input+gray'],
                        flags=cv2.WINDOW_GUI_NORMAL)

        # NOTE: In Windows, w/o scaling, window may be expanded to full screen
        #   if system is set to remember window positions.
        if utils.MY_OS == 'win':
            cv2.resizeWindow(const.WIN_NAME['input+gray'], 1000, 500)

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

        if utils.MY_OS in 'lin, win':
            self.settings_win = "cv2.createCLAHE settings (dbl-click text to save)"
        else:  # is macOS
            self.settings_win = "cv2.createCLAHE settings (rt-click text to save)"

        # Move the control window away from the processing windows.
        if utils.MY_OS == 'lin':
            cv2.namedWindow(self.settings_win, flags=cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(self.settings_win, 800, 35)
        elif utils.MY_OS == 'dar':
            cv2.namedWindow(self.settings_win, flags=cv2.WINDOW_GUI_NORMAL)
            cv2.moveWindow(self.settings_win, 600, 15)
        else:  # is Windows
            # Need to compensate for WINDOW_AUTOSIZE not working in Windows10.
            cv2.namedWindow(self.settings_win, flags=cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow(self.settings_win, 900, 500)

        cv2.setMouseCallback(self.settings_win,
                             self.save_with_click)

        if utils.MY_OS == 'lin':
            clip_tb_name = 'Clip limit\n10X'
            tile_tb_name = 'Tile size (N, N)\n'
        elif utils.MY_OS == 'win':  # is WindowsS, limited to 10 characters
            clip_tb_name = 'Clip, 10X'
            tile_tb_name = 'Tile size'
        else:
            clip_tb_name = 'Clip limit, (10X)'
            tile_tb_name = 'Tile size, (N, N)'

        # Set trackbar minimum to 1 b/c can't use a zero value in selectors.
        cv2.createTrackbar(clip_tb_name,
                           self.settings_win,
                           20,
                           50,
                           self.clip_selector)
        cv2.setTrackbarMin(clip_tb_name,
                           self.settings_win,
                           1)

        cv2.createTrackbar(tile_tb_name,
                           self.settings_win,
                           8,
                           300,
                           self.tile_selector)
        cv2.setTrackbarMin(tile_tb_name,
                           self.settings_win,
                           1)

    def save_with_click(self, event, *args):
        """
        Double-click on the namedWindow calls module that saves the image
        and settings.
        Calls utils.save_img_and_settings.
        Called by cv2.setMouseCallback event.

        Args:
            event: The implicit mouse event.
            *args: Return values from setMouseCallback(); not used here.

        Returns: *event* as a formality.

        """
        if utils.MY_OS in 'lin, win':
            mouse_event = cv2.EVENT_LBUTTONDBLCLK
        else:
            mouse_event = cv2.EVENT_RBUTTONDOWN

        if event == mouse_event:
            utils.save_img_and_settings(self.clahe_img,
                                        self.settings_txt,
                                        f'{Path(__file__).stem}')
        return event

    def clip_selector(self, c_val) -> None:
        """
        The "CLAHE clip limit (10X)" trackbar handler. Limits tile_size
        to greater than zero.

        Args:
            c_val: The integer value passed from trackbar.
        Returns: None
        """

        self.clip_limit = c_val / 10

        self.apply_clahe()

    def tile_selector(self, t_val) -> None:
        """
        The "CLAHE tile size" trackbar handler. Limits tile_size
        to greater than zero.

        Args:
            t_val: The integer value passed from trackbar.
        Returns: None
        """

        self.tile_size = t_val, t_val

        self.apply_clahe()

    def apply_clahe(self) -> None:
        """
        Applies CLAHE adjustments to image and calculates pixel values
        for reporting.

        Returns: None
        """

        clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                tileGridSize=self.tile_size,
                                )
        self.clahe_img = clahe.apply(self.gray_img)

        self.input_sd = int(self.gray_img.std())
        self.input_mean = int(self.gray_img.mean())
        self.clahe_sd = int(self.clahe_img.std())
        self.clahe_mean = int(self.clahe_img.mean())

        # This order of calls places the CLAHE image window on top.
        self.show_histograms(self.clahe_img.ravel())

        self.show_settings()

        cv2.namedWindow(const.WIN_NAME['clahe'],
                        flags=cv2.WINDOW_GUI_NORMAL)
        clahe_img_scaled = utils.scale_img(self.clahe_img, arguments['scale'])
        cv2.imshow(const.WIN_NAME['clahe'], clahe_img_scaled)

    def show_histograms(self, live_histo: np.ndarray) -> None:
        """
        Updates CLAHE adjusted histogram plot with Matplotlib from
        trackbar changes. Called from apply_clahe().

        Args: A flattened (1-D) ndarray of the image so that its
            histogram can be displayed.

        Returns: None
        """
        # Need to clear prior histograms before drawing new ones.
        #  Redrawing both histograms is inefficient and slow, but it works.
        plt.cla()

        # hist() returns tuple of (counts(n), bins(edges), patches(artists)
        # histtype='step' draws a line, 'stepfilled' fills under the line;
        #   both are patches. Polygon artists provide faster rendering
        #   than the default 'bar', which is a BarContainer object of
        #   Rectangle artists.
        # For input img, use 'step' and pre-flattened ndarray for better
        #   performance when overlaying a static histogram.
        plt.hist(self.flat_gray_array,
                 bins=255,
                 range=[0, 256],
                 color='black',
                 alpha=1,
                 histtype='step',
                 label='Input, grayscale'
                 )

        plt.hist(live_histo,
                 bins=255,
                 range=[0, 256],
                 color='orange',
                 alpha=1,
                 histtype='stepfilled',
                 label='CLAHE adjusted'
                 )

        plt.title('Histograms')
        plt.xlabel('Pixel value')
        plt.ylabel('Pixel count')
        plt.legend()

    def show_settings(self) -> None:
        """
        Display name of file and processing parameters in contour_tb_win
        window. Displays real-time parameter changes.
        Calls module utils.text_array() in contour_modules directory.

        Returns: None
        """

        the_text = (
            f'Input image: {arguments["input"]}\n'
            f'Input grayscale pixel value: mean {self.input_mean},'
            f' stdev {self.input_sd}\n'
            f'cv2.createCLAHE cliplimit={self.clip_limit}, tileGridSize{self.tile_size}\n'
            f'CLAHE grayscale pixel value: mean {self.clahe_mean},'
            f' stdev {self.clahe_sd}'
        )

        # Put text into contoured_txt for printing and saving to file.
        self.settings_txt = the_text

        # Need to set the dimensions of the settings area to fit all text.
        #   Font style parameters are set in constants.py module.
        settings_img = utils.text_array((150, 500), the_text)

        cv2.imshow(self.settings_win, settings_img)


if __name__ == "__main__":
    # Program exits here if system platform or Python version check fails.
    utils.check_platform()
    vcheck.minversion('3.7')

    # All checks are good, so grab as a 'global' the dictionary of
    #   command line argument values.
    arguments = utils.args_handler()

    # Need to not set up tk canvas to display Histograms b/c
    #  generates a fatal memory allocation error. It has something
    #  to do with the start_event_loop function.
    PI = ProcessImage()
    print(f'{Path(__file__).name} is now running...')

    # Set infinite loop with sigint handler to monitor "quit"
    #  keystrokes.
    quit_thread = threading.Thread(
        target= utils.quit_keys(), daemon=True)

    quit_thread.start()
