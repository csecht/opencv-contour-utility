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

Quit program with Esc or Q key when a window other than Histograms is selected.
Or quit from command line with Ctrl-C.
Save settings and CLAHE image with the Save slide bar.

Requires Python3.7 or later and the packages opencv-python and numpy.
Developed in Python 3.8-3.9.
"""
# Copyright (C) 2022 C.S. Echt, under GNU General Public License

# Standard library imports.
import sys
from pathlib import Path
from datetime import datetime

# Third party imports.
try:
    import cv2
    from matplotlib import pyplot as plt
except (ImportError, ModuleNotFoundError) as import_err:
    print('*** OpenCV or Matplotlib was not found or needs an update:\n\n'
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
from contour_utils import (vcheck, utils)


class ProcessImage:
    __slots__ = ('clahe_img', 'clahe_mean', 'clahe_sd', 'clip_limit',
                 'flat_gray_array', 'gray_img', 'orig_img', 'orig_mean',
                 'orig_sd', 'settings_txt',
                 'settings_win', 'tile_size',
                 'save_tb_name', 'clahe_hist')

    def __init__(self):

        # The np.ndarray arrays for images to be processed.
        self.orig_img = None
        self.gray_img = None
        self.flat_gray_array = None
        self.clahe_img = None

        # Image processing parameters amd metrics.
        self.clip_limit = 2.0  # Default trackbar value.
        self.tile_size = (8, 8)  # Default trackbar value.
        self.orig_sd = 0
        self.orig_mean = 0
        self.clahe_sd = 0
        self.clahe_mean = 0

        self.settings_txt = ''
        self.settings_win = ''
        self.save_tb_name = ''

        plt.ion()
        self.manage_input()
        self.setup_trackbars()
        # NOTE that setup_trackbars sets starting pos other than zero,
        #  which creates a call event for set_clahe()

    def manage_input(self):
        """
        Reads input images, creates grayscale image and its flattened
        array, adjusts displayed image size, displays input and grayscale
        side-by-side in one window.

        Returns: None
        """

        # utils.args_handler() has verified image path, so read from it.
        self.orig_img = cv2.imread(arguments['input'])
        self.gray_img = cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2GRAY)

        win_name = 'Input <- | -> Grayscale for processing'
        cv2.namedWindow(win_name,
                        flags=cv2.WINDOW_KEEPRATIO)

        # Need to match shapes of the two cv image arrays.
        side_by_side = cv2.hconcat(
            [self.orig_img, cv2.cvtColor(self.gray_img, cv2.COLOR_GRAY2RGB)])

        cv2.imshow(win_name, side_by_side)

    def setup_trackbars(self) -> None:
        """
        All trackbars that go in a separate window of image processing
        settings.

        Returns: None
        """

        self.settings_win = "Image and cv2.createCLAHE settings"
        self.save_tb_name = 'Save, click at 0'
        cv2.namedWindow(self.settings_win, flags=cv2.WINDOW_AUTOSIZE)

        # Move the control window away from the processing windows.
        # Place window at right edge of screen by using an excessive x-coordinate.
        if utils.MY_OS == 'lin':
            cv2.namedWindow(self.settings_win, flags=cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(self.settings_win, 800, 35)
        elif utils.MY_OS == 'dar':
            cv2.namedWindow(self.settings_win, flags=cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(self.settings_win, 600, 15)
        else:  # is Windows
            cv2.namedWindow(self.settings_win, flags=cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow(self.settings_win, 500, 500)
            cv2.resizeWindow(self.settings_win, 500, 500)

        cv2.createTrackbar('Clip limit (10X)',
                           self.settings_win,
                           20,
                           50,
                           self.clip_selector)
        cv2.setTrackbarMin('Clip limit (10X)',
                           self.settings_win,
                           1)

        cv2.createTrackbar('Tile size (N, N)',
                           self.settings_win,
                           8,
                           200,
                           self.tile_selector)
        cv2.setTrackbarMin('Tile size (N, N)',
                           self.settings_win,
                           1)

        cv2.createTrackbar(self.save_tb_name,
                           self.settings_win,
                           2,
                           50,
                           self.save_selector)

    def clip_selector(self, c_val) -> None:
        """
        The "CLAHE clip limit (10X)" trackbar handler. Limits tile_size
        to greater than zero.

        Args:
            c_val: The integer value passed from trackbar.
        Returns: None
        """

        if c_val == 0:
            self.clip_limit = 0.1
            print('CLAHE clip limit of zero was reset to 0.1')
        else:
            self.clip_limit = c_val / 10

        self.set_clahe()

    def tile_selector(self, t_val) -> None:
        """
        The "CLAHE tile size" trackbar handler. Limits tile_size
        to greater than zero.

        Args:
            t_val: The integer value passed from trackbar.
        Returns: None
        """

        if t_val == 0:
            self.tile_size = 1, 1
            print('CLAHE tile size of zero was reset to (1, 1)')
        else:
            self.tile_size = t_val, t_val

        self.set_clahe()

    def save_selector(self, s_val) -> None:
        """
        The 'Save' trackbar handler.
        Args:
            s_val: The integer value passed from trackbar.

        Returns: None

        """
        # Need a pause to prevent multiple Trackbar event calls.
        if s_val == 0:
            plt.pause(0.5)
            self.save_img_and_settings()
            cv2.setTrackbarPos(self.save_tb_name,
                               self.settings_win,
                               1)

    def set_clahe(self) -> None:
        """
        Applies CLAHE adjustments to image and calculates pixel values
        for reporting.

        Returns: None
        """

        clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                tileGridSize=self.tile_size,
                                )
        self.clahe_img = clahe.apply(self.gray_img)

        self.orig_sd = int(self.gray_img.std())
        self.orig_mean = int(self.gray_img.mean())
        self.clahe_sd = int(self.clahe_img.std())
        self.clahe_mean = int(self.clahe_img.mean())

        # A hack to avoid having two settings text windows appear.
        self.show_settings()

        win_name = 'CLAHE adjusted'
        cv2.namedWindow(win_name,
                        flags=cv2.WINDOW_GUI_NORMAL)
        cv2.moveWindow(win_name, 300, 50)
        cv2.imshow(win_name, self.clahe_img)

        self.show_clahe_histogram()
        # show_clahe_histogram() calls show_input_histogram()

    def show_input_histogram(self) -> None:
        """
        Allows a one-time rendering of the input histogram, thus
        providing a faster response for updating the CLAHE histogram
        with Trackbar changes.
        Called from __init__().

        Returns: None
        """

        # hist() returns tuple of (counts(n), bins(edges), patches(artists)
        # histtype='step' draws a line, 'stepfilled' fills under the line;
        #   both are patches.Polygon artists that provide faster rendering
        #   than the default 'bar', which is a BarContainer object of
        #   Rectangle artists.

        plt.hist(self.gray_img.ravel(),
                      bins=255,
                      range=[0, 256],
                      color='gray',
                      alpha=0.4,
                      histtype='stepfilled',
                      label='Input grayscale'
                      )

    def show_clahe_histogram(self) -> None:
        """
        Updates CLAHE adjusted histogram plot with Matplotlib from
        trackbar changes. Called from set_clahe().
        Calls show_input_histogram()

        Returns: None
        """

        plt.cla()

        self.show_input_histogram()

        # Need to clear prior histograms before drawing new ones.
        plt.hist(self.clahe_img.ravel(),
                      bins=255,
                      range=[0, 256],
                      color='orange',
                      alpha=0.6,
                      histtype='stepfilled',  # 'step' draws a line.
                      label='CLAHE adjusted'
                      )
        plt.title('Histograms')
        plt.xlabel('Pixel value')
        plt.ylabel('Pixel count')

        plt.legend()


    def show_settings(self) -> None:
        """
        Display name of file and processing parameters in settings_win
        window. Displays real-time parameter changes.
        Calls module utils.text_array() in contour_utils directory.

        Returns: None
        """

        the_text = (
            f'Input image: {arguments["input"]}\n'
            f'Input grayscale pixel value: mean {self.orig_mean},'
            f' stdev {self.orig_sd}\n'
            f'cv2.createCLAHE cliplimit={self.clip_limit}, tileGridSize{self.tile_size}\n'
            f'CLAHE grayscale pixel value: mean {self.clahe_mean},'
            f' stdev {self.clahe_sd}'
        )

        # Put text into settings_txt for printing and saving to file.
        self.settings_txt = the_text

        # Need to set the dimensions of the settings area to fit all text.
        #   Font style parameters are set in constants.py module.
        settings_img = utils.text_array((150, 500), the_text)

        cv2.imshow(self.settings_win, settings_img)

    def save_img_and_settings(self) -> None:
        """
        Print to terminal/console and to file the currently selected
        trackbar and calculated image processing values.
        Save current result image. Called from save_selector().

        Returns: None
        """

        curr_time = datetime.now().strftime('%I%M%S')
        time2print = datetime.now().strftime('%I:%M:%S%p')

        image_file = utils.args_handler()['input']

        # For JPEG file format the supported parameter is cv2.IMWRITE_JPEG_QUALITY
        # with a possible value between 0 and 100, the default value being 95.
        # The higher value produces a better quality image file.
        #
        # For PNG file format the supported imwrite parameter is
        # cv2.IMWRITE_PNG_COMPRESSION with a possible value between 0 and 9,
        # the default being 3. The higher value does high compression of the
        # image resulting in a smaller file size but a longer compression time.

        img_ext = Path(Path(image_file).suffix)
        img_stem = Path(Path(image_file).stem)
        cv2.imwrite(f'{img_stem}_equalize_{curr_time}{img_ext}', self.clahe_img)

        settings2save = (f'\n\nTime saved: {time2print}\n'
                         'Settings for image:'
                         f' {img_stem}_equalize_{curr_time}{img_ext}\n'
                         + self.settings_txt)

        # Use this Path function for saving individual settings files:
        # Path(f'{img_stem}_clahe_settings{curr_time}.txt').write_text(settings2save)

        # Use this for appending multiple settings to single file:
        with Path(f'{img_stem}_equalize_settings.txt').open('a') as fp:
            fp.write(settings2save)

        print(f'\nCLAHE image and its settings were saved to files.'
              f'{settings2save}')


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
    utils.quit_keys()

    # Set infinite loop with sigint handler to monitor "quit" keystrokes.
    utils.quit_keys()
