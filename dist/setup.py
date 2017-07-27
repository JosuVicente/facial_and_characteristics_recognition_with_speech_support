from cx_Freeze import setup, Executable

import os
os.environ['TCL_LIBRARY'] = "C:\\Users\\Josu\\Anaconda3\\envs\\tfm\\tcl\\tcl8.6"
os.environ['TK_LIBRARY'] = "C:\\Users\\Josu\\Anaconda3\\envs\\tfm\\tcl\\tk8.6"

base = None


executables = [Executable("test.py", base=base)]

packages = ["idna",'numpy.core._methods', 'numpy.lib.format']
options = {
    'build_exe': {

        'packages':packages,
    },

}


setup(
    name = "tfm",
    options = options,
    version = "0.1",
    description = 'description',
    executables = executables
)

