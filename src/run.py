if __name__ == "__main__":
    import gui
    gui.Application().run()

# this command will allow you to compile thos code to an executable
"""
python3 -m nuitka --disable-console --standalone --enable-plugin=tk-inter --enable-plugin=numpy  --enable-plugin=pyqt5 src/run
"""
