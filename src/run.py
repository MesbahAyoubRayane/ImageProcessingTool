if __name__ == "__main__":
    import gui
    gui.Application().run()

# this command will allow you to compile thos code to an executable
"""
python3 -m nuitka --disable-console --standalone --onefile --enable-plugin=tk-inter src/run.py
"""
