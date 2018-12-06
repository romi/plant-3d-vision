#!/usr/bin/env python3
from lettucethink import fsdb
from optparse import OptionParser
from scanner import localdirs
import numpy as np

if __name__ == "__main__":
   usage = "usage: %prog [options] SCAN_ID"
   parser = OptionParser(usage=usage)
   parser.add_option("-i", "--id",
           dest="id",
           help ="scan id")

   (options, args) = parser.parse_args()
   options=vars(options)

   path = localdirs.db_dir
   #id="2018-11-26_15-05-57"
   db = fsdb.DB(path)
   scan = db.get_scan(options["id"])
   fileset = scan.get_fileset("images")
   fs=fileset.get_files()

   posefile = open("/tmp/poses.txt", mode = 'w') 

   for i,file in enumerate(fs):
       p = file.get_metadata("pose")
       s = file.get_id()+".jpg " + str(p[0]) + " " + str(p[1]) + " " + str(p[2])+"\n"
       posefile.write(s)
   posefile.close()
