import os

import ROOT

path = os.path.dirname(os.path.realpath(__file__))

ROOT.gROOT.SetBatch()
ROOT.gROOT.LoadMacro(path + "/AtlasStyle.C")
ROOT.gROOT.LoadMacro(path + "/AtlasUtils.C")
ROOT.SetAtlasStyle()

def draw_atlas_label(preliminary, offset=0.0):
    ROOT.ATLAS_LABEL(0.2, 0.88)
    txt = ROOT.TLatex()
    txt.SetNDC()
    txt.DrawText(
        0.33 + offset,
        0.88,
        'Simulation {}'.format(
            'Preliminary' if preliminary else 'Internal'
        )
    )
