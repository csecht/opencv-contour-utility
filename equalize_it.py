#!/usr/bin/env python3
"""Use OpenCV to explore image processing parameters involved in
histogram equalization with CLAHE. Parameter values are adjusted
with slide bars. Color images are converted to grayscale prior to
equalization. Intended as an optional image pre-processor for
thresh_it.py and edge_it.py to identify object contours.

USAGE Example command lines, from within the image-processor-main folder:
python3 -m equalize_it --help
python3 -m equalize_it --about
python3 -m equalize_it --input images/sample2.jpg --resize 0.5

Quit program with Return/Enter key. Quit from command line with Ctrl-C.
Save any selected window with Ctrl-S or from rt-click menu.
Save settings and CLAHE image with the Save/Print slide bar.

Requires Python3.7 or later and the packages opencv-python and numpy.
Developed in Python 3.8-3.9.
"""
# Copyright (C) 2022 C.S. Echt, under GNU General Public License

# Standard library imports.
import sys
from pathlib import Path

# Third party imports.
try:
    import cv2
    import numpy as np
    import matplotlib
    import matplotlib.backends.backend_tkagg as backend
    import tkinter as tk
    from matplotlib import pyplot as plt
    from matplotlib.widgets import Button
except (ImportError, ModuleNotFoundError) as import_err:
    print('*** OpenCV, Numpy, Matplotlib or tkinter (tk/tcl) was not found or needs an update:\n\n'
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
                 'fig', 'ax1', 'ax2')

    def __init__(self):

        # The np.ndarray arrays for images to be processed.
        self.orig_img = None
        self.gray_img = None
        self.flat_gray_array = None
        self.clahe_img = None

        # Matplotlib plotting with live updates.
        plt.style.use(('bmh', 'fast'))
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            nrows=2,
            num='Histograms',  # Provide a window title to replace 'Figure 1'.
            sharex='all',
            sharey='all',
            clear=True
        )
        # Note that plt.ion() needs to be called
        # AFTER subplots() is called,
        #   otherwise a "Segmentation fault (core dumped)" error is raised.
        # plt.ion() is used with fig.canvas.start_event_loop(0.1);
        #   not needed if fig.canvas.draw_idle() is used.
        plt.ion()

        # Image processing parameters amd metrics.
        self.clip_limit = 2.0  # Default trackbar value.
        self.tile_size = (8, 8)  # Default trackbar value.
        self.orig_sd = 0
        self.orig_mean = 0
        self.clahe_sd = 0
        self.clahe_mean = 0

        self.settings_txt = ''
        self.settings_win = ''

        self.manage_input()
        self.setup_canvas_window()
        self.show_input_histogram()
        self.setup_trackbars()

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

        # Display starting images here so that their imshow is called only once.
        win_name = '<- Input | Grayscale for processing'
        cv2.namedWindow(win_name,
                        flags=cv2.WINDOW_KEEPRATIO)
        side_by_side = cv2.hconcat(
            [self.orig_img, cv2.cvtColor(self.gray_img, cv2.COLOR_GRAY2RGB)])
        cv2.imshow(win_name, side_by_side)

    @staticmethod
    def setup_canvas_window() -> None:
        """
        A tkinter window for the Matplotlib figure canvas.
        """

        # canvas_window is the Tk mainloop defined in if __name__ == "__main__".
        canvas_window.title('Histograms')
        canvas_window.resizable(False, False)

        canvas_window.bind_all('<Escape>', utils.quit_keys)
        canvas_window.bind('<Control-q>', utils.quit_keys)

        canvas = backend.FigureCanvasTkAgg(plt.gcf(), canvas_window)
        toolbar = backend.NavigationToolbar2Tk(canvas, canvas_window)

        # Need to remove the useless subplots navigation button.
        # Source: https://stackoverflow.com/questions/59155873/
        #   how-to-remove-toolbar-button-from-navigationtoolbar2tk-figurecanvastkagg
        # Remove all tools from toolbar because the Histograms window is
        # non-responsive while in event_loop.
        for child in toolbar.children:
            toolbar.children[child].pack_forget()

        # Note: Toolbar and X icon do not execute until program quits or
        #   until a Trackbar event occurs. Something to do with
        #   self.fig.canvas.start_event_loop(0.1)?
        # def no_exit_on_x():
        #     print('The Histogram window will close when you quit the program (Esc or Q).')
        # canvas_window.protocol('WM_DELETE_WINDOW', no_exit_on_x)

        # Now display remaining widgets in canvas_window.
        # NOTE: toolbar must be gridded BEFORE canvas to prevent
        #   FigureCanvasTkAgg from preempting window geometry with its pack().
        toolbar.grid(row=1, column=0,
                     padx=5, pady=(0, 5),  # Put a border around toolbar.
                     sticky=tk.NSEW,
                     )
        canvas.get_tk_widget().grid(row=0, column=0,
                                    ipady=10, ipadx=10,
                                    padx=5, pady=5,  # Put a border around plot.
                                    sticky=tk.NSEW,
                                    )

    def setup_trackbars(self) -> None:
        """
        All trackbars that go in a separate window of image processing
        settings.

        Returns: None
        """

        self.settings_win = "Image and cv2.threshold settings"
        cv2.namedWindow(self.settings_win, flags=cv2.WINDOW_AUTOSIZE)
        # Move the control window away from the processing windows.
        # Place window at right edge of screen by using an excessive x-coordinate.
        if utils.MY_OS in 'lin, dar':
            cv2.namedWindow(self.settings_win, flags=cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(self.settings_win, 4000, 35)
        else:  # is Windows
            # TODO: FIX poor fit of trackbars and text img in settings_win.
            cv2.namedWindow(self.settings_win, flags=cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow(self.settings_win, 500, 500)

        cv2.createTrackbar('CLAHE clip limit (10X)',
                           self.settings_win,
                           20,
                           50,
                           self.clip_selector)
        cv2.createTrackbar('CLAHE tile size (N, N)',
                           self.settings_win,
                           8,
                           200,
                           self.tile_selector)
        cv2.createTrackbar('Slide to save CLAHE image and settings (0 <- -> 1):',
                           self.settings_win,
                           0,
                           1,
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

    def save_selector(self, event=None) -> None:
        """
        The 'save' trackbar handler.
        Args:
            event: Implicit event from movement of 'Save' trackbar.

        Returns: None

        """
        utils.save_img_and_settings(self.clahe_img,
                                    self.settings_txt,
                                    'clahe')

        return event  # Null use of *event* parameter; a formality.

    def set_clahe(self, startup=None) -> None:
        """
        Applies CLAHE adjustments to image and calculates pixel values
        for reporting.
        Args:
            startup: Flag for when call is from program startup.

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
        if not startup:
            self.show_settings()

        win_name = 'CLAHE adjusted'
        cv2.namedWindow(win_name,
                        flags=cv2.WINDOW_KEEPRATIO)
        cv2.imshow(win_name, self.clahe_img)

        self.show_clahe_histogram()

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
        # Need to match these parameters with those for ax2.hist().
        self.ax1.hist(self.gray_img.ravel(),
                      bins=255,
                      range=[0, 256],
                      color='blue',
                      histtype='stepfilled',
                      )
        self.ax1.set_ylabel("Pixel count")
        self.ax1.set_title('Input (grayscale)')

    def show_clahe_histogram(self) -> None:
        """
        Updates CLAHE adjusted histogram plot with Matplotlib from
        trackbar changes. Called from set_clahe().

        Returns: None
        """

        # Need to clear prior histograms before drawing new ones.
        self.ax2.clear()
        self.ax2.hist(self.clahe_img.ravel(),
                      bins=255,
                      range=[0, 256],
                      color='orange',
                      histtype='stepfilled',  # 'step' draws a line.
                      )
        self.ax2.set_title('CLAHE adjusted')
        self.ax2.set_xlabel("Pixel value")
        self.ax2.set_ylabel("Pixel count")
        # From: https://stackoverflow.com/questions/28269157/
        #  plotting-in-a-non-blocking-way-with-matplotlib
        # and, https://github.com/matplotlib/matplotlib/issues/11131
        # Note that start_event_loop is needed for live updates of clahe histograms.
        self.fig.canvas.start_event_loop(0.1)
        # Note that plt.ion() is not needed if draw_idle is used here.
        #  Using it in addition to plt.ion() is okay; no obvious effect.
        # self.fig.canvas.draw_idle()

    def show_settings(self) -> None:
        """
        Display name of file and processing parameters in self.settings_win
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


if __name__ == "__main__":
    # Program exits here if system platform or Python version check fails.
    utils.check_platform()
    vcheck.minversion('3.7')

    # All checks are good, so grab as a 'global' the dictionary of
    #   command line argument values.
    arguments = utils.args_handler()

    # Run the Matplotlib histogram plots in a tkinter window.
    canvas_window = tk.Tk()

    PI = ProcessImage()
    print(f'{Path(__file__).name} is now running...')

    # Set infinite loop with sigint handler to monitor "quit" keystrokes.
    utils.quit_keys()

    try:
        canvas_window.mainloop()
    except KeyboardInterrupt:
        print("\n*** User quit the program from Terminal/Console ***\n")
