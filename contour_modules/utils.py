"""
General housekeeping utilities.
Functions:
check_platform - Exits if not Linux, Windows, or MacOS.
args_handler - Handles command line arguments; returns dict of args.
scale_img - Resize displayed images.
save_img_and_settings - Saves files of result image and its settings.
text_array - Generates an image array of text.
valid_path_to - Get correct path to program's files.
quit_keys -  Error-free and informative exit from the program.
"""
# Copyright (C) 2022 C.S. Echt, under GNU General Public License'

# Standard library imports.
import argparse
import platform
import signal
import sys

from datetime import datetime

# noinspection PyCompatibility
from __main__ import __doc__
from pathlib import Path

# Third party imports.
import cv2
import numpy as np

# Local application imports.
import contour_modules
from contour_modules import constants as const

MY_OS = sys.platform[:3]


def check_platform() -> None:
    if MY_OS == 'dar':
        print('Developed in macOS 13; earlier versions may not work.\n')

    # Need to account for scaling in Windows10 and earlier releases.
    if MY_OS == 'win':
        from ctypes import windll

        if platform.release() < '10':
            windll.user32.SetProcessDPIAware()
        else:
            windll.shcore.SetProcessDpiAwareness(1)

        print('NOTE: Windows window sizing has issues.\n'
              'Manual window resizing may be needed.\n')

    print('Quit program with Esc or Q key, or Ctrl-C from Terminal.\n')


def args_handler() -> dict:
    """
    Handle command line arguments.

    Returns: Dictionary of argument values.
    """

    parser = argparse.ArgumentParser(description='Explore Image Processing Parameters.')
    parser.add_argument('--about',
                        help='Provide description, version, GNU license',
                        action='store_true',
                        default=False)
    parser.add_argument('--input', '-i',
                        help='Path to input image.',
                        default='images/sample1.jpg',
                        metavar='PATH/FILE')
    parser.add_argument('--scale', '-s',
                        help='Factor to change displayed image size (default: 1.0).',
                        default=1.0,
                        type=float,
                        required=False,
                        metavar='X')
    args = parser.parse_args()

    about_text = (f'{__doc__}\n'
                  f'{"Author:".ljust(13)}{contour_modules.__author__}\n'
                  f'{"Version:".ljust(13)}{contour_modules.__version__}\n'
                  f'{"Status:".ljust(13)}{contour_modules.__status__}\n'
                  f'{"URL:".ljust(13)}{contour_modules.URL}\n'
                  f'{contour_modules.__copyright__}'
                  f'{contour_modules.__license__}\n'
                  )

    if args.about:
        print('====================== ABOUT START ====================')
        print(about_text)
        print('====================== ABOUT END ====================')

        sys.exit(0)

    if not Path.exists(valid_path_to(args.input)):
        print('Could not open the image (check spelling and path):', args.input)
        sys.exit()

    if args.scale <= 0:
        args.scale = 1
        print('--scale X: X must be greater than zero. Resetting to 1.')

    input_arg = 0
    for i in ('--input', '--i', '-i'):
        if i in sys.argv[:]:
            input_arg += 1
    if input_arg == 0:
        print('Without the --input argument, the default input image file is: '
              f'{valid_path_to("images/sample1.jpg")}')

    arguments = {
        'about': args.about,
        'input': args.input,
        'scale': args.scale,
    }
    return arguments


def valid_path_to(relative_path: str) -> Path:
    """
    Get correct path to program's directory/file structure
    depending on whether program invocation is a standalone app or
    the command line. Works with symlinks. Allows command line
    using any path; does not need to be from parent directory.
    _MEIPASS var is used by distribution programs from
    PyInstaller --onefile; e.g. for images dir.

    :param relative_path: Program's local dir/file name, as string.
    :return: Absolute path as pathlib Path object.
    """
    # Modified from: https://stackoverflow.com/questions/7674790/
    #    bundling-data-files-with-pyinstaller-onefile and PyInstaller manual.
    if getattr(sys, 'frozen', False):  # hasattr(sys, '_MEIPASS'):
        base_path = getattr(sys, '_MEIPASS', Path(Path(__file__).resolve()).parent)
        valid_path = Path(base_path) / relative_path
    else:
        valid_path = Path(Path(__file__).parent, f'../{relative_path}').resolve()
    return valid_path


def text_array(text_shape: iter, do_text: str) -> np.ndarray:
    """
    Generate an array image of text to display in a cv2 window.
    Usage example:
      text_img = utils.text_array((150, 500), my_text)
      cv2.imshow(my_text_window, text_img)

    Args:
        text_shape: Pixel height, width for text image dimensions,
                    as tuple or list.
        do_text: Formatted text string to display in the image.

    Returns: An image array with the formatted text.
    """

    text_img = np.ones(text_shape, dtype='uint8')

    # Convert the ones array to an image with gray16 (41,41,41) bg.
    text_img[:] = np.ones(text_shape) * 41 / 255.0

    # Display currently selected settings.
    # To display multiple lines with putText():
    # https://stackoverflow.com/questions/27647424/
    #   opencv-puttext-new-line-character
    # https://stackoverflow.com/questions/28394017/
    #   how-to-insert-multiple-lines-of-text-into-frame-image/54234703#54234703
    text_size, _ = cv2.getTextSize(do_text,
                                   const.FONT_TYPE,
                                   const.TEXT_SCALER,
                                   const.TEXT_THICKNESS)
    line_height = text_size[1] + 9

    _x, y0 = (5, 50)
    for i, line in enumerate(do_text.split("\n")):
        _y = y0 + i * line_height
        cv2.putText(img=text_img,
                    text=line,
                    org=(_x, _y),
                    fontFace=const.FONT_TYPE,
                    fontScale=const.TEXT_SCALER,
                    color=const.TEXT_COLOR,
                    thickness=const.TEXT_THICKNESS,
                    lineType=cv2.LINE_AA)  # LINE_AA is anti-aliased

    return text_img


def save_img_and_settings(img2save: np.ndarray,
                          txt2save: str,
                          caller: str) -> None:
    """
    Print to terminal/console and to file the currently selected
    trackbar and calculated image processing values.
    Save current result image. Called from save_selector().

    Args:
        img2save: The current resulting image array.
        txt2save: The current image processing settings.
        caller: Descriptive name of the calling app to insert in the
                file names, e.g. 'clahe', 'threshold'.

    Returns: None
    """

    curr_time = datetime.now().strftime('%I%M%S')
    time2print = datetime.now().strftime('%I:%M:%S%p')

    # For JPEG file format the supported parameter is cv2.IMWRITE_JPEG_QUALITY
    # with a possible value between 0 and 100, the default value being 95.
    # The higher value produces a better quality image file.
    #
    # For PNG file format the supported imwrite parameter is
    # cv2.IMWRITE_PNG_COMPRESSION with a possible value between 0 and 9,
    # the default being 3. The higher value does high compression of the
    # image resulting in a smaller file size but a longer compression time.

    img_ext = Path(Path(args_handler()['input']).suffix)
    img_stem = Path(Path(args_handler()['input']).stem)
    first_word = txt2save.split()[0]
    print('first word', first_word)

    # Note: What's happening here is that separate files are saved for
    #   the contoured and shape images, while the one settings text file
    #   is appended to with both sets of text; condition is needed only
    #   when the shape_it.py file is the caller.
    #   BECAUSE first_word match is hard coded, need to keep the
    #   same first word in the contour settings text in all modules
    #   that use it.
    if first_word == 'Image:':  # text is from contoured_txt
        file_name = f'{img_stem}_{caller}_contoured_{curr_time}{img_ext}'
        cv2.imwrite(file_name, img2save)
    else:  # text is from shaped_txt
        file_name = f'{img_stem}_{caller}_shaped_{curr_time}{img_ext}'
        cv2.imwrite(file_name, img2save)

    settings2save = (f'\n\nTime saved: {time2print}\n'
                     'Settings for image:'
                     f' {img_stem}_{caller}_{curr_time}{img_ext}\n'
                     f'{txt2save}')

    # Use this Path function for saving individual settings files:
    # Path(f'{img_stem}_clahe_settings{curr_time}.txt').write_text(settings2save)

    # Use this for appending multiple settings to single file:
    with Path(f'{img_stem}_{caller}_settings.txt').open('a') as fp:
        fp.write(settings2save)

    print(f'Result image and its settings were saved to files.'
          f'{settings2save}')


def scale_img(img: np.ndarray, scale: float) -> np.ndarray:
    """
    Change size of displayed images from original (input) size.
    Intended mainly for when input image is too large to fit on screen.

    Args:
        img: A numpy.ndarray of image to be scaled.
        scale: The multiplication factor to grow or shrink the
                displayed image. Defined from cmd line arg '--scale'.
                Default from argparse is 1.0.

    Returns: A scaled np.ndarray object; if *scale* is 1, then no change.
    """

    # Is redundant with check of --scale value in args_handler().
    scale = 1 if scale == 0 else scale

    # Provide the best interpolation method for slight improvement of
    #  resized image depending on whether it is down- or up-scaled.
    interpolate = cv2.INTER_AREA if scale < 0 else cv2.INTER_CUBIC

    scaled_image = cv2.resize(src=img,
                              dsize=None,
                              fx=scale, fy=scale,
                              interpolation=interpolate)
    return scaled_image


def quit_keys() -> None:
    """
    Error-free and informative exit from the python-opencv program.
    Program runs until the Q or Esc key is pressed, or until Ctrl-C from
    the Terminal is pressed.
    Called from if __name__ == "__main__".

    Returns: None
    """

    def sigint_handler(signum, frame):
        cv2.destroyAllWindows()
        sys.exit('\n*** User quit from Terminal/Console. ***\n')

    # source: https://stackoverflow.com/questions/57690899/
    #   how-cv2-waitkey1-0xff-ordq-works
    while True:
        # Need to allow exit from the Terminal with Ctrl-q.
        signal.signal(signal.SIGINT, sigint_handler)

        key = cv2.waitKey(1)

        # Shuts down opencv and terminates the Python process when a
        # specific key is pressed from an active window.
        # 27 is the Esc key ASCII code in decimal.
        # 113 is the letter 'q' ASCII code in decimal.
        quit_codes = (27, 113)
        if key in quit_codes:
            cv2.destroyAllWindows()
            sys.exit('\n*** User quit the program. ***\n')
