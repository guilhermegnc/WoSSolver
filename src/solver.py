#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Words on Stream - Anagram Generator
"""
import tkinter as tk
import logging
from ui import WordsStreamGUI

def main():
    """Main function to run the application."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    root = tk.Tk()
    app = WordsStreamGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()