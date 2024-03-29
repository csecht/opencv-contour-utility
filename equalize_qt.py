#!/usr/bin/env python3
"""
A more responsive version of equalize_it.py that uses two histogram
subplots on a PyQt5 canvas but updates just the CLAHE histogram.
Uses a PyQt5 GUI, which may require installation: pip3 install -U pyqt5.
Currently, fully functional only on macOS and Windows systems.

Works on Linux only if using opencv-python versions from 4.1.1.26 to
4.2.0.34. Later versions throw a qt.qpa.plugin for 'xcb' error.
Recommend running equalize_tk.py or equalize_it.py on Linux.

USAGE Example command lines, from within the image-processor-main folder:
python3 -m equalize_qt --help
python3 -m equalize_qt --about
python3 -m equalize_qt --input images/sample2.jpg

Quit program with Esc or Q key; may need to first select a different
window. Or quit from command line with Ctrl-C.

Requires Python3.7 or later and the packages opencv-python and numpy.
Developed in Python 3.8-3.9.
"""
# Copyright (C) 2022-2023 C.S. Echt, under GNU General Public License

# Standard library imports.
import sys

# Third party imports.
import cv2
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Local application imports
from contour_modules import (vcheck, utils, constants as const)


# noinspection PyUnresolvedReferences
class PlotWindow(QDialog):

    def __init__(self):
        super().__init__()

        # a figure and axes instance to plot on
        plt.style.use(('bmh', 'fast'))
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            nrows=2,
            num='Histograms',  # Provide a window title to replace 'Figure 1'.
            sharex='all',
            sharey='all',
            clear=True
        )
        plt.ion()

        # this is the Canvas Widget that
        # displays the 'figure'. It takes the
        # 'figure' instance as a parameter to __init__
        self.canvas = FigureCanvas(self.fig)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        toolbar = NavigationToolbar(self.canvas, self)

        # Button connected to 'quit_now' method
        self.button = QPushButton('Quit')
        self.button.clicked.connect(self.quit_now)

        # Source for using pyqt5:
        #  https://www.geeksforgeeks.org/how-to-embed-matplotlib-graph-in-pyqt5/
        # creating a Vertical Box layout
        layout = QVBoxLayout()

        # adding toolbar to the layout
        layout.addWidget(toolbar)

        # adding canvas to the layout
        layout.addWidget(self.canvas)

        # adding push button to the layout
        layout.addWidget(self.button)

        # setting layout to the main window
        self.setLayout(layout)

        # The np.ndarray arrays for images to be processed.
        self.input_img = None
        self.gray_img = None
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

        self.manage_input()
        self.setup_trackbars()
        self.show_input_histogram()

    @staticmethod
    def quit_now():
        cv2.destroyAllWindows()
        sys.exit('\n*** User quit the program. ***\n')

    def manage_input(self):
        """
        Reads input images, creates grayscale image and its flattened
        array, adjusts displayed image size, displays input and grayscale
        side-by-side in one window.

        Returns: None
        """

        # utils.args_handler() has verified image path, so read from it.
        self.input_img = cv2.imread(arguments['input'])
        self.gray_img = cv2.imread(arguments['input'], cv2.IMREAD_GRAYSCALE)

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

        self.settings_win = "cv2.createCLAHE settings (dbl-click text to save)"

        # Move the control window away from the processing windows.
        # Place window at right edge of screen by using an excessive x-coordinate.
        cv2.namedWindow(self.settings_win, flags=cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(self.settings_win, 800, 35)

        cv2.setMouseCallback(self.settings_win,
                             self.save_with_click)

        clip_tb_name = 'Clip limit\n10X'
        tile_tb_name = 'Tile size (N, N)\n'

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
                           200,
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

        if event == cv2.EVENT_LBUTTONDBLCLK:
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

        self.show_clahe_histogram()
        self.show_settings()

        cv2.namedWindow(const.WIN_NAME['clahe'],
                        flags=cv2.WINDOW_GUI_NORMAL)
        clahe_img_scaled = utils.scale_img(self.clahe_img, arguments['scale'])
        cv2.imshow(const.WIN_NAME['clahe'], clahe_img_scaled)

    def show_input_histogram(self) -> None:
        """
        Allows a one-time rendering of the input histogram, thus
        providing a faster response for updating the histogram Figure
        with CLAHE Trackbar changes.
        Called from __init__().

        Returns: None
        """

        self.ax1.clear()

        # hist() returns tuple of (counts(n), bins(edges), patches(artists)
        # histtype='step' draws a line, 'stepfilled' fills under the line;
        #   both are patches.Polygon artists that provide faster rendering
        #   than the default 'bar', which is a BarContainer object of
        #   Rectangle artists.
        # Need to match these parameters with those for ax2.hist().
        self.ax1.hist(self.gray_img.ravel(),
                      bins=255,
                      range=[0, 256],
                      color='blue',
                      alpha=0.4,
                      histtype='stepfilled',
                      )
        self.ax1.set_ylabel("Pixel count")
        self.ax1.set_title('Input (grayscale)')

    def show_clahe_histogram(self) -> None:
        """
        Updates CLAHE adjusted histogram plot with Matplotlib from
        trackbar changes. Called from apply_clahe().

        Returns: None
        """

        # Need to clear prior histograms before drawing new ones.
        self.ax2.clear()

        self.ax2.hist(self.clahe_img.ravel(),
                      bins=255,
                      range=[0, 256],
                      color='orange',
                      histtype='stepfilled',  # 'step' draws a line.
                      # linewidth=1.2
                      )
        self.ax2.set_title('CLAHE adjusted')
        self.ax2.set_xlabel("Pixel value")
        self.ax2.set_ylabel("Pixel count")

        # From: https://stackoverflow.com/questions/28269157/
        #  plotting-in-a-non-blocking-way-with-matplotlib
        # and, https://github.com/matplotlib/matplotlib/issues/11131
        # Note that start_event_loop is needed for live updates of clahe histograms.
        # self.fig.canvas.start_event_loop(0.1)

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


if __name__ == '__main__':
    # Program exits here if system platform or Python version check fails.
    vcheck.minversion('3.7')

    # All checks are good, so grab as a 'global' the dictionary of
    #   command line argument values.
    arguments = utils.args_handler()

    # creating a pyqt5 application
    app = QApplication(sys.argv)

    # creating a window object
    plot_win = PlotWindow()

    # showing the histogram plotting window
    plot_win.show()

    # loop
    sys.exit(app.exec_())
