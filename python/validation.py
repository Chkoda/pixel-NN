import itertools
import multiprocessing.pool
import ROOT

ROOT.gROOT.SetBatch(True)

def templates(do_pulls=False):
    templates = [
        ('residuals_{n}_{d}', '(Output_positions_{d} - Output_true_{d})'),
        ('residuals_{n}_2D', '(Output_positions_Y - Output_true_Y):(Output_positions_X - Output_true_X)'),
        ('residuals_{n}_{d}_2D_eta', '(Output_positions_{d} - Output_true_{d}):globalEta'),
        ('residuals_{n}_{d}_2D_phi', '(Output_positions_{d} - Output_true_{d}):globalPhi'),
        ('residuals_{n}_{d}_2D_cluster_size', '(Output_positions_{d} - Output_true_{d}):cluster_size'),
        ('residuals_{n}_{d}_2D_cluster_size_X', '(Output_positions_{d} - Output_true_{d}):cluster_size_X'),
        ('residuals_{n}_{d}_2D_cluster_size_Y', '(Output_positions_{d} - Output_true_{d}):cluster_size_Y'),
    ]
    if do_pulls:
        templates += [
            ('pull_{n}_{d}', '(Output_positions_{d} - Output_true_{d})/Output_uncert_{d}'),
            ('pull_{n}_2D', '(Output_positions_Y - Output_true_Y)/Output_uncert_Y:(Output_positions_X - Output_true_X)/Output_uncert_X'),
            ('pull_{n}_{d}_2D_eta', '(Output_positions_{d} - Output_true_{d})/Output_uncert_{d}:globalEta'),
            ('pull_{n}_{d}_2D_phi', '(Output_positions_{d} - Output_true_{d})/Output_uncert_{d}:globalPhi'),
            ('pull_{n}_{d}_2D_cluster_size', '(Output_positions_{d} - Output_true_{d})/Output_uncert_{d}:cluster_size'),
            ('pull_{n}_{d}_2D_cluster_size_X', '(Output_positions_{d} - Output_true_{d})/Output_uncert_{d}:cluster_size_X'),
            ('pull_{n}_{d}_2D_cluster_size_Y', '(Output_positions_{d} - Output_true_{d})/Output_uncert_{d}:cluster_size_Y'),
        ]
    return templates

directions = ['X', 'Y']
layers = [
    ('ibl', '((NN_layer==0)&&(NN_barrelEC==0))'),
    ('barrel', '((NN_layer>0)&&(NN_barrelEC==0))'),
    ('endcap', '(NN_barrelEC!=0)'),
]

layers_ITk = [
    ('barrel', '(NN_barrelEC==0)'),
    ('endcap', '(NN_barrelEC!=0)')
]

done = set()

def gen_variables(npart, layers=layers, do_pulls=False):
    global done
    it = itertools.product(templates(do_pulls), [npart], directions, layers)
    for t, n, d, (ln,lc) in it:
        h, v = t[:2]
        name = h.format(n=n,d=d) + '_' + ln
        if name in done:
            continue
        else:
            done.add(name)
        var = v.format(i=(n-1), d=d)
        cond = '(Output_number_true=={})'.format(n)
        if lc is not None:
            cond += ('&&' + lc)
        if len(t) == 3:
            cond += ('&&' + t[2])
        yield name, var, cond

def get_range(var, pixdims):

    def _range(var):
        if '_uncert_' in var:
            return 100, -6, 6
        if 'positions_X' in var:
            return 100, -pixdims[0], pixdims[0]
        if 'positions_Y' in var:
            return 100, -pixdims[1], pixdims[1]
        if var == 'globalEta':
            return 100, -5, 5
        if var == 'globalPhi':
            return 100, -4, 4
        if var == 'cluster_size':
            return 50, 0, 50
        if var in ['cluster_size_X', 'cluster_size_Y']:
            return 8, 0, 8

    fields = var.split(':')
    if len(fields) == 1:
        r = _range(fields[0])
        return '({},{},{})'.format(*r)
    else:
        r = _range(fields[1]) + _range(fields[0])
        return '({},{},{},{},{},{})'.format(*r)

def make_hist(tree, pixdims, name, var, cond):
    print name

    if '_2D' in name:
        varg = '{}>>{}{}'.format(var,name,get_range(var, pixdims))
    else:
        varg = '{}>>{}{}'.format(var,name,get_range(var, pixdims))
    tree.Draw(varg, cond)
    hist = ROOT.gROOT.Get(name)
    return hist


def make_all_hists(tree, npart):
    for vars in gen_variables(npart):
        make_hist(tree, *vars)

def make_histograms(input_paths, output_path, nparticle, IBL=True,
                    pixdims=(0.05,0.5), do_pulls=False):
    tree = ROOT.TChain('NNoutput')
    for path in input_paths:
        tree.Add(path)
    outfile = ROOT.TFile(output_path, 'RECREATE')
    for vars in gen_variables(nparticle, layers if IBL else layers_ITk, do_pulls):
        make_hist(tree, pixdims, *vars)
    outfile.Write()
