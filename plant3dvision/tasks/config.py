#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import luigi

class PointCloudColorConfig(luigi.Config):
    colors = luigi.DictParameter(default = {
        "stem" : [1.0, 0.0, 0.0],
        "flower" : [1.0, 1.0, 0.0],
        "fruit" : [1.0, 0.0, 1.0],
        "pedicel" : [1.0, 1.0, 1.0],
        "leaf" : [0.0, 1.0, 0.0],
    })
    
