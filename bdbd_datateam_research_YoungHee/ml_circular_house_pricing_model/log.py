# -*- coding: utf-8 -*-

from datetime import datetime
import tempfile
import os
from datetime import datetime
import sys
import inspect

class Log:

    name = "fundbloc"
    log_dir = None
    log_path = None
    cur_date = None
    to_file = False

    def __init__(self, _to_file=False, _name=None, _dir=None):
        self.to_file = _to_file
        if self.to_file == False:
            return

        if _dir != None:
            self.log_dir = _dir
        else:
            self.log_dir = tempfile.mkdtemp()

        if _name != None:
            self.name = _name

        cur_date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.log_path = self.log_dir + "/" + self.name + "-logs-" + cur_date + ".log"

    def write_file(self, text):
        if self.to_file == False:
            return

        if os.path.isdir(self.log_dir) == False:
            os.makedirs(self.log_dir)

        _type = 'w'
        if os.path.isfile(self.log_path):
            _type = 'a'

        f = open(self.log_path, _type)
        f.write(text+"\n")
        f.close()

    def get_func_name(self):
        module = str(inspect.stack()[3][1])
        module = module.split('/')[-1]
        module = module.split('.')[0]
        return module+"."+sys._getframe(3).f_code.co_name+"()"

    def write(self, category, text):
        on_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = "["+on_date+"] "+category+" ["+self.get_func_name()+"] "+text
        print(text)
        self.write_file(text)
        
    def debug(self, text):
        self.write("DEBUG", text)

    def info(self, text):
        self.write("INFO", text)

    def warn(self, text):
        self.write("WARNING", text)

    def error(self, text):
        self.write("ERROR", text)
