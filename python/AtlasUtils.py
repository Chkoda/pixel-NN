import os
base = os.path.dirname(os.path.realpath(__file__))

import ROOT 
ROOT.module._root.gROOT.LoadMacro(base + "/AtlasUtils.C") 


