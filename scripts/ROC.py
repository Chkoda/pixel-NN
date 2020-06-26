#!/bin/env python2
""" Create the number NN performance graphs """
# pylint: disable=no-member
import array
import argparse
import collections
import importlib
import itertools
import time

import numpy as np
import pylatex as latex
import ROOT
import root_numpy
import sklearn.metrics

from PixelNN import figures


LCONDS = [
    ('IBL', '((NN_barrelEC == 0) && (NN_layer == 0))'),
    ('Barrel', '((NN_barrelEC == 0) && (NN_layer > 0))'),
    ('Endcap', '(NN_barrelEC != 0)')
]

LCONDS_ITK = [
    ('Barrel', '(NN_barrelEC == 0)'),
    ('Endcap', '(NN_barrelEC != 0)')
]


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--input', required=True)
    args.add_argument('--nclusters', default=10000000, type=int)
    args.add_argument('--preliminary', action='store_true')
    args.add_argument('--ITk', action='store_true')
    return args.parse_args()


def _load_data(path, nclusters, raw=False):
    data = collections.OrderedDict()
    for name, cond in LCONDS:
        data[name] = root_numpy.root2array(
            path,
            treename='NNoutput',
            branches=[
                "Output_number",
                "Output_number_true",
                "Output_number_estimated",
                "globalEta",
                "globalPhi",
                "cluster_size",
                "cluster_size_X",
                "cluster_size_Y"
            ],
            selection=cond,
            stop=nclusters
        )
        data[name]['Output_number_true'][
            np.where(data[name]['Output_number_true'] > 3)
        ] = 3
    return data

def _set_graph_style(graph, layer):
    graph.SetLineWidth(2)
    if layer == 'IBL':
        graph.SetLineStyle(1)
        graph.SetLineColor(ROOT.kRed)
    elif layer == 'Barrel':
        graph.SetLineStyle(2)
        graph.SetLineColor(ROOT.kBlack)
    else:
        graph.SetLineStyle(9)
        graph.SetLineColor(ROOT.kBlue)


def _roc_graph(data, classes, prelim=False):
    # pylint: disable=too-many-locals
    pos, neg = classes

    canvas = ROOT.TCanvas('c_{}vs{}.format(pos, neg)', '', 0, 0, 800, 600)
    canvas.SetLogx()

    graphs = ROOT.TMultiGraph()
    leg = ROOT.TLegend(0.63, 0.7, 0.9, 0.88)

    for layer in data:
        if pos == 3:
            pos_sel = data[layer]['Output_number_true'] >= pos
        else:
            pos_sel = data[layer]['Output_number_true'] == pos

        if neg == 3:
            neg_sel = data[layer]['Output_number_true'] >= neg
        else:
            neg_sel = data[layer]['Output_number_true'] == neg

        isel = np.where(
            np.logical_or(
                pos_sel,
                neg_sel,
            )
        )[0]

        fpr, tpr, _ = sklearn.metrics.roc_curve(
            data[layer]['Output_number_true'][isel],
            data[layer]['Output_number'][isel][:, pos - 1],
            pos_label=pos
        )
        auc = sklearn.metrics.auc(fpr, tpr)

        graph = ROOT.TGraph(fpr.size)
        _set_graph_style(graph, layer)
        root_numpy.fill_graph(graph, np.column_stack((fpr, tpr)))
        graphs.Add(graph)
        leg.AddEntry(graph, '{}, AUC: {:.2f}'.format(layer, auc), 'L')

    graphs.SetTitle(
        ';Pr(Estimated: {pos}-particle{ps} | True: {neg}-particle{ns});Pr(Estimated: {pos}-particle{ps} | True: {pos}-particle{ps})'.format(  # noqa
            pos=pos,
            neg=neg,
            ps=('' if pos == 1 else 's'),
            ns=('' if neg == 1 else 's')
        )
    )
    graphs.SetMaximum(1.5)
    graphs.Draw('AL')

    line = ROOT.TF1("line", "x", 0, 1)
    line.SetLineStyle(3)
    line.SetLineColor(ROOT.kGray + 2)
    line.Draw("same")
    leg.AddEntry(line, 'Random, AUC: 0.50', 'L')

    leg.SetTextSize(leg.GetTextSize() * 0.65)
    leg.SetBorderSize(0)
    leg.Draw()

    figures.draw_atlas_label(prelim)
    txt = ROOT.TLatex()
    txt.SetNDC()
    txt.SetTextSize(txt.GetTextSize() * 0.75)
    txt.DrawLatex(0.2, 0.82, 'PYTHIA8 dijet, 1.8 < p_{T}^{jet} < 2.5 TeV')

    canvas.SaveAs('ROC_{}vs{}.pdf'.format(pos, neg))


def _confusion_matrices(data):

    for layer in data:
        matrix = np.zeros((2,3))
        for i_true in [1, 2, 3]:
            subdata = data[layer][np.where(data[layer]['Output_number_true'] == i_true)]
            for i_nn in [1, 2, 3]:
                nclas = np.count_nonzero(subdata['Output_number_estimated'] == i_nn)
                if i_nn == 1:
                    matrix[0, i_true - 1] = float(nclas) / subdata.shape[0]
                else:
                    matrix[1, i_true - 1] += float(nclas) / subdata.shape[0]

        table = latex.Tabular('r r c c c')
        table.add_row('', '', '', latex.utils.bold('True'), '')
        table.add_row(
            '',
            '',
            latex.utils.bold('1'),
            latex.utils.bold('2'),
            latex.utils.bold('3')
        )
        table.add_hline()
        table.append(latex.Command('addlinespace', options='3mm'))
        for i in range(2):
            if i == 0:
                cell0 = latex.MultiRow(
                    # 3,
                    2,
                    data=latex.Command(
                        'rotatebox',
                        ['90', latex.utils.bold('Network')],
                        'origin=c'
                    )
                )
                n = '1'
            else:
                cell0 = ''
                n = '2/3'
            row = [cell0, latex.utils.bold(n)]
            for j in range(3):
                row.append('{:.2f}'.format(matrix[i, j]))
            table.add_row(row)
            if i == 0:
                table.append(latex.Command('addlinespace', options='4mm'))
        path = 'confusion_{}.tex'.format(layer)
        with open(path, 'w') as wfile:
            wfile.write(table.dumps())
            print '  -> wrote ' + path


def _get_bins(cond):
    if cond == 'globalEta':
        return np.array(
            [-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]
        )
    if cond == 'globalPhi':
        return np.array(
            [-3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
        )
    if cond == 'cluster_size':
        return np.array(
            [1, 5, 10, 15, 20]
        )
    if cond == 'cluster_size_X':
        return np.arange(1, 9)
    if cond == 'cluster_size_Y':
        return _get_bins('cluster_size_X')


def _get_xlabel(cond):
    if cond == 'globalEta':
        return 'Cluster global #eta'
    if cond == 'globalPhi':
        return 'Cluster global #phi'
    if cond == 'cluster_size':
        return 'Cluster total size'
    if cond == 'cluster_size_X':
        return 'Cluster size in local x direction'
    if cond == 'cluster_size_Y':
        return 'Cluster size in local y direction'

def _calc_rate(data, true, est):
    subdata = data[np.where(data['Output_number_true'] == true)]
    if subdata.shape[0] == 0:
        return 0, 0

    nclas = 0
    for n in est:
        nclas += np.count_nonzero(subdata['Output_number_estimated'] == n)

    m = float(nclas)
    N = float(subdata.shape[0])
    e = m / N
    de = np.sqrt(e * (1 - e) / N)
    return e, de


def _tpr_fnr(data, pos, cond, preliminary=False):

    ROOT.gStyle.SetErrorX(0.5)

    colors = {1: ROOT.kBlack, 2: ROOT.kRed, 3: ROOT.kBlue}
    markers = {1: 8, 2: 21,  3: 23}

    oldmargin = ROOT.gStyle.GetPadRightMargin()
    ROOT.gStyle.SetPadRightMargin(0.15)

    if pos == 1:
        fnrs = [2, 3]
    elif pos == 2:
        fnrs = [1, 3]
    else:
        fnrs = [1, 2]

    # define bins in eta
    bins = _get_bins(cond)

    bin_data_1 = np.zeros(len(bins) - 1)
    bin_data_2or3 = np.zeros(len(bins) - 1)
    e_bin_data_1 = np.zeros(len(bins) - 1)
    e_bin_data_2or3 = np.zeros(len(bins) - 1)

    for i in range(1, len(bins)):

        data_1 = np.zeros(3)
        data_2or3 = np.zeros(3)
        e_data_1 = np.zeros(3)
        e_data_2or3 = np.zeros(3)
        stats = np.zeros(3)
        for j, layer in enumerate(data.keys()):
            # get the indices corresponding to eta bins
            i_bins = np.digitize(data[layer][cond], bins)
            bin_data = data[layer][np.where(i_bins == i)]
            data_1[j], e_data_1[j] = _calc_rate(bin_data, true=pos, est=[1])
            data_2or3[j], e_data_2or3[j] = _calc_rate(bin_data, true=pos, est=[2,3])
            stats[j] = bin_data.shape[0]

        if np.all(stats == 0):
            weights = 0
        else:
            weights = stats / float(np.sum(stats))
        bin_data_1[i-1] = np.sum(weights * data_1)
        bin_data_2or3[i-1] = np.sum(weights * data_2or3)
        e_bin_data_1[i-1] = np.sqrt(np.sum(weights*weights*e_data_1*e_data_1))
        e_bin_data_2or3[i-1] = np.sqrt(np.sum(weights*weights*e_data_2or3*e_data_2or3))

    hist_1 = ROOT.TH1F(
        'h',
        '',
        len(bins) - 1,
        array.array('f', bins)
    )
    hist_2or3 = ROOT.TH1F(
        'ha',
        '',
        len(bins) - 1,
        array.array('f', bins)
    )

    root_numpy.fill_hist(hist_1, bins[:-1], weights=bin_data_1)
    root_numpy.fill_hist(hist_2or3, bins[:-1], weights=bin_data_2or3)
    for i in range(1, bins.shape[0]):
        hist_1.SetBinError(i, e_bin_data_1[i-1])
        hist_2or3.SetBinError(i, e_bin_data_2or3[i-1])

    cnv = ROOT.TCanvas('c', '', 0, 0, 800, 600)
    # cnv.SetLogy()

    hist_1.SetTitle(';{};Pr(Estimated | true: {} particle)'.format(
        _get_xlabel(cond),
        pos,
    ))

    xmax_1 = 1.6

    hist_1.SetMaximum(xmax_1)
    hist_1.SetMinimum(0.0)
    hist_1.SetLineColor(colors[1])
    hist_1.SetMarkerColor(colors[1])
    hist_1.SetMarkerStyle(markers[1])
    hist_1.Draw('p e')
    cnv.Update()

    hist_2or3.SetLineColor(colors[2])
    hist_2or3.SetMarkerColor(colors[2])
    hist_2or3.SetMarkerStyle(markers[2])
    hist_2or3.Draw('p e same')

    if cond == 'cluster_size':
        hist_1.GetXaxis().SetNdivisions(4, False)
        for k,v in {1: 1, 2: 5, 3: 10, 4: 15, 5: 20}.iteritems():
            hist_1.GetXaxis().ChangeLabel(k, -1, -1, -1, -1, -1, str(v))
    if cond in ['cluster_size_X', 'cluster_size_Y']:
        hist_1.GetXaxis().SetNdivisions(7, False)
        hist_1.GetXaxis().CenterLabels()

    figures.draw_atlas_label(preliminary)
    txt = ROOT.TLatex()
    txt.SetNDC()
    txt.SetTextSize(txt.GetTextSize() * 0.75)
    txt.DrawLatex(0.2, 0.82, 'PYTHIA8 dijet, 1.8 < p_{T}^{jet} < 2.5 TeV')
    txt.DrawText(
        0.2,
        0.77,
        '{}-particle{} clusters'.format(pos, '' if pos == 1 else 's')
    )

    leg = ROOT.TLegend(0.6, 0.7, 0.84, 0.82)
    leg.SetBorderSize(0)
    leg.AddEntry(hist_1, 'Estimated = 1', 'PL')
    leg.AddEntry(hist_2or3, 'Estimated = 2 or 3', 'PL')
    leg.Draw()

    cnv.SaveAs('tpr_fnr_{}_{}.pdf'.format(pos, cond))

    ROOT.gStyle.SetPadRightMargin(oldmargin)

def _do_raw_1d(data, prelim=False):

    colors = [ROOT.kBlack, ROOT.kRed, ROOT.kBlue]

    for npart, layer in itertools.product(range(1,4), [n for n, _ in LCONDS]):
        ldata = data[layer]['Output_number'][
            np.where(data[layer]['Output_number_true'] == npart)
        ]
        ldata /= np.sum(ldata, axis=1).reshape((ldata.shape[0], 1))

        canvas = ROOT.TCanvas(
            'c_raw_{}_{}'.format(npart, layer),
            '',
            0,
            0,
            800,
            600
        )
        canvas.SetLogy()

        leg = ROOT.TLegend(0.7, 0.7, 0.9, 0.85)
        leg.SetBorderSize(0)

        for i_out in range(3):
            hist = ROOT.TH1D(
                'h_raw_{}_{}_{}'.format(npart, layer, i_out),
                '',
                30,
                0,
                1
            )
            ROOT.SetOwnership(hist, False)
            root_numpy.fill_hist(hist, ldata[:, i_out])
            hist.Scale(1.0 / hist.Integral())
            hist.SetLineColor(colors[i_out])
            hist.SetMaximum(10)
            hist.Draw(('' if i_out == 0 else 'same') + 'HIST')

            leg.AddEntry(hist, '{}-particle bin'.format(i_out + 1), 'L')

        leg.Draw()
        figures.draw_atlas_label(preliminary=prelim)
        txt = ROOT.TLatex()
        txt.SetNDC()
        txt.SetTextSize(txt.GetTextSize() * 0.75)
        txt.DrawLatex(0.2, 0.82, 'PYTHIA8 dijet, 1.8 < p_{T}^{jet} < 2.5 TeV')
        txt.DrawText(
            0.2,
            0.77,
            '{}-particle clusters'.format(npart)
        )
        txt.DrawText(0.2, 0.72, layer)

        canvas.SaveAs('number_raw_{}_{}.pdf'.format(npart, layer))


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


def _do_raw_2d(data, x=2, y=3, prelim=False):

    x_min, x_max = 0, 1
    y_min, y_max = 0, 1.4
    z_min, z_max = 1e-6, 1
    x_n_bins = 50
    y_n_bins = int(x_n_bins * (y_max - y_min) / (x_max - x_min))

    oldmargin = ROOT.gStyle.GetPadRightMargin()
    ROOT.gStyle.SetPadRightMargin(0.15)
    _set_palette()
    ROOT.gStyle.SetNumberContours(255)

    for npart in range(1, 4):

        canvas = ROOT.TCanvas('c', '', 0, 0, 800, 600)
        canvas.SetLogz()

        hist = ROOT.TH2D(
            'h_raw_2D_{}'.format(npart),
            '',
            x_n_bins,
            x_min,
            x_max,
            y_n_bins,
            y_min,
            y_max
        )
        hist.GetYaxis().SetRangeUser(0, y_max)
        hist.GetZaxis().SetLabelSize(0.04)
        hist.GetZaxis().SetTitle(
            "Cluster density / {x} x {y}".format(
                x=hist.GetXaxis().GetBinWidth(1),
                y=hist.GetYaxis().GetBinWidth(1),
            )
        )
        hist.SetTitle(
            ';{x}-particle score;{y}-particle score'.format(
                x=x if x !=3 else '#geq {}'.format(x),
                y=y if y !=3 else '#geq {}'.format(y),
            )
        )
        ROOT.TGaxis.SetMaxDigits(4)

        for layer in [n for n, _ in LCONDS]:
            layer_data = data[layer]['Output_number'][
                np.where(data[layer]['Output_number_true'] == npart)
            ]
            root_numpy.fill_hist(hist, layer_data[:,[x-1,y-1]])

        hist.Scale(1.0 / hist.Integral())
        hist.SetMinimum(z_min)
        hist.SetMaximum(z_max)
        hist.Draw('COLZ')

        # Plot the classification region
        if x == 2 and y == 3:
            gr1 = ROOT.TGraph(4, array.array('d', [0, 0, 0.6, 0.6]), array.array('d', [0, 0.2, 0.2, 0]))
            gr1.SetFillColorAlpha(ROOT.kBlack, 0.2)
            gr1.SetLineWidth(0)
            gr1.Draw('F')

            # Add legend
            leg = ROOT.TLegend(0.5, 0.7, 0.8, 0.8)
            leg.SetTextSize(0.02)
            leg.SetBorderSize(0)
            leg.AddEntry(gr1, "1-particle classification region", "F")
            leg.Draw()

        # Add text
        figures.draw_atlas_label(preliminary=prelim)
        txt = ROOT.TLatex()
        txt.SetNDC()
        txt.SetTextSize(txt.GetTextSize() * 0.75)
        txt.DrawLatex(0.2, 0.82, 'PYTHIA8 dijet, 1.8 < p_{T}^{jet} < 2.5 TeV')
        txt.DrawText(
            0.2,
            0.77,
            '{}-particle clusters'.format(npart)
        )

        canvas.SaveAs('number_raw_{}_{}_2D_{}.pdf'.format(x, y, npart))

    ROOT.gStyle.SetPadRightMargin(oldmargin)


def _do_tpr_fnr(data, prelim=False):
    for cond in ['globalEta', 'globalPhi', 'cluster_size', 'cluster_size_X', 'cluster_size_Y']:
        _tpr_fnr(data, 1, cond, prelim)
        _tpr_fnr(data, 2, cond, prelim)
        _tpr_fnr(data, 3, cond, prelim)

def _do_rocs(data, prelim=False):
    _roc_graph(data, (3, 2), prelim)
    _roc_graph(data, (3, 1), prelim)
    _roc_graph(data, (2, 3), prelim)
    _roc_graph(data, (2, 1), prelim)
    _roc_graph(data, (1, 2), prelim)
    _roc_graph(data, (1, 3), prelim)


def _main():

    t_0 = time.time()
    args = _get_args()
    print '==> Loading data from ' + args.input
    if args.ITk:
        global LCONDS
        LCONDS = LCONDS_ITK
    out = _load_data(args.input, args.nclusters)
    print '==> Drawing raw output'
    _do_raw_1d(out, args.preliminary)
    _do_raw_2d(out, 1, 2, args.preliminary)
    _do_raw_2d(out, 1, 3, args.preliminary)
    _do_raw_2d(out, 2, 3, args.preliminary)
    print '==> Drawing the ROC curves'
    _do_rocs(out, args.preliminary)
    print '==> Computing the confusion matrices'
    _confusion_matrices(out)
    print '==> Drawing true positive rate / false negative rate curves'
    _do_tpr_fnr(out, args.preliminary)
    print '==> Completed in {:.2f}s'.format(time.time() - t_0)

if __name__ == '__main__':
    _main()
