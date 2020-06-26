#!/bin/env python2
""" Program to produce the pixel clustering NN performance plots """
import argparse
import array
import collections
import importlib
import itertools
import logging
import os
import shutil
import sys

import numpy as np
import ROOT
import root_numpy

from PixelNN import figures


LOG = logging.getLogger(os.path.basename(__file__))
NPARTICLES = []
LAYERS = ['ibl', 'barrel', 'endcap']
COLORS = [ROOT.kRed, ROOT.kBlack, ROOT.kBlue]
MARKERS = [21, 8, 23]
DIRECTIONS = ['X', 'Y']
VARIABLES = ['residuals']
CONDITIONALS = [
    'eta',
    'phi',
    'cluster_size',
    'cluster_size_X',
    'cluster_size_Y'
]

PALETTE_SET = False


def _set_palette():
    """ Set the custom color palette

    It is the default InvertedDarkBodyRadiator palette with white as
    the first color.
    See https://root.cern.ch/doc/master/TColor_8cxx_source.html#l02473
    """
    global PALETTE_SET  # pylint: disable=global-statement
    if not PALETTE_SET:
        stops = array.array('d', [
            0.0000, 0.1250, 0.2500,
            0.3750, 0.5000, 0.6250,
            0.7500, 0.8750, 1.0000
        ])
        red = array.array('d', [
            1.0, 234./255., 237./255.,
            230./255., 212./255., 156./255.,
            99./255., 45./255., 0./255.
        ])
        green = array.array('d', [
            1.0, 238./255.,
            238./255., 168./255.,
            101./255., 45./255.,
            0./255., 0./255., 0./255.
        ])
        blue = array.array('d', [
            1.0, 95./255., 11./255.,
            8./255., 9./255., 3./255.,
            1./255., 1./255., 0./255.
        ])
        ROOT.TColor.CreateGradientColorTable(
            9,
            stops,
            red,
            green,
            blue,
            255,
            1.0
        )
        PALETTE_SET = True


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('input')
    args.add_argument('output')
    args.add_argument('--force', action='store_true')
    args.add_argument('--loglevel', choices=['DEBUG', 'INFO'], default='INFO')
    args.add_argument('--preliminary', action='store_true')
    return args.parse_args()


def _get_histograms(path):
    tfile = ROOT.TFile(path, 'READ')
    ROOT.SetOwnership(tfile, False)
    hdict = collections.OrderedDict()
    for name in [br.GetName() for br in tfile.GetListOfKeys()]:
        hist = tfile.Get(name)
        hdict[name] = hist
    return hdict


def _get_nparticles(path):
    tfile = ROOT.TFile(path, 'READ')
    hdict = collections.OrderedDict()
    nparts = []
    for name in [br.GetName() for br in tfile.GetListOfKeys()]:
        for n in [1, 2, 3]:
            if n not in nparts and '_{}_'.format(n) in name:
                nparts.append(n)
    return nparts

def _get_if_pulls(path):
    tfile = ROOT.TFile(path, 'READ')
    for name in [br.GetName() for br in tfile.GetListOfKeys()]:
        if 'pull_' in name:
            return True
    return False


def _fit(thist, err=False):
    mu = thist.GetMean();
    sig = thist.GetStdDev()
    fit = thist.Fit('gaus', 'Q0S', '', mu - 3 * sig, mu + 3 * sig)
    # fit = thist.GetFunction('gaus')
    # return (
    #     fit.GetParameter('Constant'),
    #     fit.GetParameter('Mean'),
    #     fit.GetParameter('Sigma')
    # )
    results = [fit.Parameter(i) for i in range(3)]

    if not err:
        return results

    errors = [fit.ParError(i) for i in range(3)]
    return results, errors


def _fwhm(thist, err=False):
    bin1 = thist.FindFirstBinAbove(thist.GetMaximum() * 0.5)
    bin2 = thist.FindLastBinAbove(thist.GetMaximum() * 0.5)
    fwhm = thist.GetBinCenter(bin2) - thist.GetBinCenter(bin1)
    if not err:
        return fwhm

    e1 = thist.GetBinWidth(bin2) * 0.5
    e2 = thist.GetBinWidth(bin1) * 0.5
    unc = np.sqrt(e1*e1 + e2*e2)
    return fwhm, unc


def _layer_name(layer):
    return layer.upper() if layer == 'ibl' else layer[0].upper() + layer[1:]


def _plot_1d(hsdict, variable, nparticle, direction, preliminary):
    # pylint: disable=too-many-locals

    name = '_'.join([variable, str(nparticle), direction])

    canvas = ROOT.TCanvas('canvas_' + name, '', 0, 0, 800, 600)
    legend = ROOT.TLegend(0.61, 0.6, 0.91, 0.85)
    legend.SetBorderSize(0)
    stack = ROOT.THStack('stk_' + name, '')

    gausses = []
    for layer, color, marker in zip(LAYERS, COLORS, MARKERS):
        LOG.debug((name, layer))
        key = name + '_' + layer
        if key not in hsdict:
            continue
        hist = hsdict[key]
        if layer == 'endcap' and nparticle == 1:
            hist.Rebin(4)
            hist.Scale(0.5 / hist.Integral())
        else:
            hist.Rebin(2)
            hist.Scale(1.0 / hist.Integral())
            width = hist.GetBinWidth(1)
        hist.SetLineColor(color)
        hist.SetMarkerColor(color)
        hist.SetMarkerStyle(marker)

        if 'pull' in variable:
            hist.SetLineStyle(2)
            const, mean, sigma = _fit(hist)
            gausses.append(ROOT.TF1('f'+layer, '[0]*TMath::Gaus(x, [1], [2])', -5, 5))
            gausses[-1].SetLineColor(color)
            gausses[-1].SetLineStyle(2)
            gausses[-1].SetParameter(0, const)
            gausses[-1].SetParameter(1, mean)
            gausses[-1].SetParameter(2, sigma)
        else:
            mean = hist.GetMean()
            sigma = _fwhm(hist)

        stack.Add(hist)

        legend.AddEntry(
            hist,
            '#splitline{%s clusters}'
            '{#mu = %.2f %s, %s = %.2f %s}' % (
                _layer_name(layer),
                mean,
                'mm' if 'residuals' in variable else '',
                "fwhm" if 'residuals' in variable else "#sigma",
                sigma,
                'mm' if 'residuals' in variable else '',
            ),
            'LP'
        )

    stack.SetTitle(
        ';Truth hit {v} {bu};Particle density / {w} {wu}'.format(
            v=variable.replace('corr_', '').rstrip('s'),
            bu='[mm]' if 'residuals' in variable else '',
            w=width,
            wu='mm' if 'residuals' in variable else ''
        )
    )

    if 'pull' in variable:
        stack.Draw('p nostack')
        for g in gausses:
            g.Draw('same')
    else:
        stack.Draw('hist nostack')
        stack.Draw('p same nostack')

    scaley = 1.3
    if nparticle > 1:
        scaley = 1.7

    if 'pull' in variable:
        rangex = 5.0
    else:
        if direction == 'X':
            if nparticle == 1:
                rangex = 0.04
            else:
                rangex = 0.05
        else:
            rangex = 0.4

    print 'SETTING RANGE AT ' + str(rangex)
    stack.GetXaxis().SetRangeUser(-rangex, rangex)
    stack.SetMaximum(0.25)
    figures.draw_atlas_label(preliminary)
    legend.Draw()

    txt = ROOT.TLatex()
    txt.SetNDC()
    txt.SetTextSize(txt.GetTextSize() * 0.75)
    txt.DrawLatex(0.2, 0.79, 'PYTHIA8 dijet, 1.8 < p_{T}^{jet} < 2.5 TeV')
    txt.DrawText(
        0.2,
        0.74,
        '{}-particle{} clusters'.format(nparticle, '' if nparticle==1 else 's')
    )
    txt.DrawText(0.2, 0.69, 'local {} direction'.format(direction.lower()))

    canvas.SaveAs(name + '.pdf')


def _plot_1d_hists(hsdict, preliminary=False):

    prod = itertools.product(VARIABLES, NPARTICLES, DIRECTIONS)
    for var, npart, direc in prod:
        _plot_1d(
            hsdict=hsdict,
            variable=var,
            nparticle=npart,
            direction=direc,
            preliminary=preliminary
        )


def _plot_2d(hsdict, variable, nparticle, layer, preliminary):

    name = '_'.join([variable, str(nparticle), '2D', layer])
    key = name
    if key not in hsdict:
        return
    th2 = hsdict[name]

    _set_palette()
    ROOT.gStyle.SetNumberContours(255)

    canvas = ROOT.TCanvas('canvas_' + name, '', 0, 0, 800, 600)

    th2.Rebin2D(2, 2)

    th2.GetZaxis().SetTitle(
        "Particles / {x} x {y} {u}".format(
            x=th2.GetXaxis().GetBinWidth(1),
            y=th2.GetYaxis().GetBinWidth(1),
            u='mm^{2}' if 'residual' in variable else ''
        )
    )
    th2.GetZaxis().SetLabelSize(0.04)
    ROOT.TGaxis.SetMaxDigits(4)

    if 'residual' in variable:
        th2.GetXaxis().SetRangeUser(-0.05, 0.05)
        th2.GetYaxis().SetRangeUser(-0.5, 0.5)
    else:
        th2.GetXaxis().SetRangeUser(-5, 5)
        th2.GetYaxis().SetRangeUser(-5, 5)

    th2.GetXaxis().SetLabelSize(0.04)
    th2.GetYaxis().SetLabelSize(0.04)

    th2.SetTitle(
        ';Truth hit local x {v} {u};Truth hit local y {v} {u}'.format(
            v=variable.replace('corr_', '').rstrip('s'),
            u='[mm]' if 'residual' in variable else ''
        )
    )

    th2.Draw('COLZ')

    figures.draw_atlas_label(preliminary)

    txt = ROOT.TLatex()
    txt.SetNDC()
    txt.SetTextSize(txt.GetTextSize() * 0.75)
    txt.DrawLatex(0.2, 0.82, 'PYTHIA8 dijet, 1.8 < p_{T}^{jet} < 2.5 TeV')
    txt.DrawText(
        0.2,
        0.77,
        '{}-particle{} clusters'.format(nparticle, '' if nparticle==1 else 's')
    )

    txt.DrawText(
        0.2,
        0.72,
        layer.upper() if layer == 'ibl' else layer[0].upper() + layer[1:]
    )

    canvas.SaveAs(name + '.pdf')


def _plot_2d_hists(hsdict, preliminary):
    oldmargin = ROOT.gStyle.GetPadRightMargin()
    ROOT.gStyle.SetPadRightMargin(0.15)

    prod = itertools.product(VARIABLES, NPARTICLES, LAYERS)
    for var, npart, lyr in prod:
        _plot_2d(
            hsdict=hsdict,
            variable=var,
            nparticle=npart,
            layer=lyr,
            preliminary=preliminary
        )

    ROOT.gStyle.SetPadRightMargin(oldmargin)


def _varlabel(var):
    if var == 'eta':
        return 'Cluster global #eta', ''
    if var == 'phi':
        return 'Cluster global #phi', ''
    if var == 'cluster_size':
        return 'Cluster total size', ''
    if var == 'cluster_size_X':
        return 'Cluster size in local X direction', ''
    if var == 'cluster_size_Y':
        return 'Cluster size in local Y direction', ''


def _get_range(cond):
    if cond == 'eta':
        return (-2.5, 2.5)
    elif cond == 'phi':
        return (-3, 3)
    elif cond == 'cluster_size':
        return (1, 20)
    else:
        return (1, 8)

def _get_bins(cond):
    if cond == 'eta':
        bins = [-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]
    elif cond == 'phi':
        bins = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
    elif cond == 'cluster_size':
        bins = [1, 5, 10, 15, 20]
    else:
        bins = [1, 2, 3, 4, 5, 6, 7, 8]

    return bins


def _set_range(hist, variable, direction, cond):

    if 'pull' in variable:
        hist.SetMinimum(0)
        hist.SetMaximum(2.5)
    elif direction == 'X' or '1X' in variable:
        hist.SetMinimum(0)
        hist.SetMaximum(0.08)
    else:
        hist.SetMinimum(0)
        hist.SetMaximum(0.8)

def _rebin(bins, binvals, binerrs, cond):

    bins = np.array(bins)
    binvals = np.array(binvals)
    binerrs = np.array(binerrs)

    newbins = np.array(_get_bins(cond))
    newbinvals = np.zeros(newbins.shape[0] - 1)
    newbinerrs = np.zeros(newbins.shape[0] - 1)

    # ibins: array of indices such that elements where ibins == i fall
    # in bin with low-edge newbins[i-1]
    # so, for example: bins = [-2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, ...]
    # and newbins = [-2.5, -2, ...]
    # ibins: [1, 1, 1, 1, 1, 2, ...]
    ibins = np.digitize(bins[:-1], newbins)

    for i in range(1, newbins.shape[0]):
        # selector for elements of newbin[i-1]
        isel = np.where(ibins == i)[0]
        newbinvals[i-1] = np.mean(binvals[isel])
        errs = binerrs[isel]
        newbinerrs[i-1] = (1.0 / errs.shape[0]) * np.sqrt(np.sum(errs * errs))

    return newbins, newbinvals, newbinerrs



def _plot_2d_cond(hsdict, variable, cond, direction, prelim):

    if direction is not None:
        name = '_'.join([variable, direction, '2D', cond])
    else:
        name = '_'.join([variable, '2D', cond])

    canvas = ROOT.TCanvas("canvas_" + name, '', 0, 0, 800, 600)
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.85)
    legend.SetBorderSize(0)

    colors = {1: ROOT.kBlack, 2: ROOT.kRed, 3: ROOT.kBlue}

    for nparticle in [1, 2, 3]:

        # First, build an inclusive histogram with all the layers
        hist_incl = None
        for lyr in LAYERS:
            if direction is not None:
                hname = '_'.join([variable, str(nparticle), direction, '2D', cond, lyr])
            else:
                hname = '_'.join([variable, str(nparticle), '2D', cond, lyr])
            if hname in hsdict:
                if hist_incl is None:
                    hist_incl = hsdict[hname]
                else:
                    hist_incl.Add(hsdict[hname])

        if hist_incl is None:
            continue

        # Now, iterate through bins of the conditional variables in
        # the right range
        bins = []
        binvals = []
        binerrs = []
        vmin, vmax = _get_range(cond)
        for i in range(hist_incl.GetXaxis().FindBin(vmin), hist_incl.GetXaxis().FindBin(vmax)):
            proj = hist_incl.ProjectionY("proj%d%d" % (i,nparticle), i, i)
            if 'residuals' in variable:
                sigma, dsigma = _fwhm(proj, err=True)
            else:
                (_,_,sigma), (_,_,dsigma) = _fit(proj, err=True)
            binvals.append(sigma)
            binerrs.append(dsigma)
            bins.append(hist_incl.GetXaxis().GetBinLowEdge(i))
        bins.append(hist_incl.GetXaxis().GetBinLowEdge(i+1))

        bins, binvals, binerrs = _rebin(bins, binvals, binerrs, cond)

        hist = ROOT.TH1D(
            'h_%d_%s_%s_%s' % (nparticle, direction, cond, variable),
            '',
            len(bins)-1,
            array.array('d', bins)
        )
        ROOT.SetOwnership(hist, False)
        root_numpy.fill_hist(hist, bins[:-1], binvals)
        for i in range(1, bins.shape[0]):
            hist.SetBinError(i, binerrs[i-1])
        hist.SetMarkerColor(colors[nparticle])
        hist.SetLineColor(colors[nparticle])
        hist.SetMarkerStyle(MARKERS[nparticle-1])

        _set_range(hist, variable, direction, cond)

        if nparticle == 1:
            _set_range(hist, variable, direction, cond)
            hist.SetTitle(';{};{}'.format(
                _varlabel(cond)[0],
                'Residual fwhm [mm]' if 'residuals' in variable else 'Pull #sigma',
            ))

        if cond == 'cluster_size':
            hist.GetXaxis().SetNdivisions(4, False)
            for k,v in {1: 1, 2: 5, 3: 10, 4: 15, 5: 20}.iteritems():
                hist.GetXaxis().ChangeLabel(k, -1, -1, -1, -1, -1, str(v))

        if cond in ['cluster_size_X', 'cluster_size_Y']:
            hist.GetXaxis().SetNdivisions(7, False)
            hist.GetXaxis().CenterLabels()

        hist.Draw(('' if nparticle == 1 else 'same') + ' P E')

        legend.AddEntry(hist, "{}-particle{} clusters".format(nparticle, 's' if nparticle > 1 else ''))

    legend.Draw()
    figures.draw_atlas_label(prelim)
    txt = ROOT.TLatex()
    txt.SetNDC()
    txt.SetTextSize(txt.GetTextSize() * 0.75)
    txt.DrawLatex(0.2, 0.82, 'PYTHIA8 dijet, 1.8 < p_{T}^{jet} < 2.5 TeV')
    if direction is not None:
        txt.DrawText(0.2, 0.77, 'Local {} direction'.format(direction.lower()))

    # TODO add text for residuals1{X,Y} plots

    canvas.SaveAs(name + '.pdf')



def _plot_2d_cond_hists(hsdict, preliminary):

    ROOT.gStyle.SetErrorX(0.5)

    prod = itertools.product(
        VARIABLES,
        CONDITIONALS,
        DIRECTIONS,
    )

    for var, cond, direc in prod:
        _plot_2d_cond(
            hsdict=hsdict,
            variable=var,
            cond=cond,
            #nparticle=npart,
            direction=direc,
            prelim=preliminary
        )

########################################################################
# Main


def _main():
    args = _get_args()
    logging.basicConfig(level=args.loglevel)
    LOG.info('input: %s', args.input)
    LOG.info('output: %s', args.output)

    global NPARTICLES
    NPARTICLES = _get_nparticles(args.input)
    LOG.info('nparticles: %s', NPARTICLES)
    if _get_if_pulls(args.input):
        VARIABLES.append('pull')
    hists = _get_histograms(args.input)
    try:
        os.makedirs(args.output)
    except os.error:
        pass
    os.chdir(args.output)
    for histname, thist in hists.iteritems():
        LOG.debug('%s: %s', histname, str(thist))
    _plot_1d_hists(hists, preliminary=args.preliminary)
    _plot_2d_hists(hists, preliminary=args.preliminary)
    _plot_2d_cond_hists(hists, preliminary=args.preliminary)
    return 0


if __name__ == '__main__':
    sys.exit(_main())
