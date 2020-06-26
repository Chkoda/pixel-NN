def get_branches(nntype):
    x_branches = []
    y_branches = []

    for i in range(7*7):
        x_branches.append('NN_matrix{}'.format(i))
    for i in range(7):
        x_branches.append('NN_pitches{}'.format(i))
    x_branches += ['NN_layer', 'NN_barrelEC', 'NN_phi', 'NN_theta']

    if nntype == 'number':
        y_branches = ['NN_nparticles1', 'NN_nparticles2', 'NN_nparticles3']

    elif nntype.startswith('pos'):
        for i in range(int(nntype[-1])):
            y_branches.append('NN_position_id_X_{}'.format(i))
            y_branches.append('NN_position_id_Y_{}'.format(i))

    else: # error
        npart = int(nntype[-2])
        direc = nntype[-1].upper()

        for i in range(npart):
            x_branches.append('NN_position_pred_id_X_{}'.format(i))
            x_branches.append('NN_position_pred_id_Y_{}'.format(i))

        if npart == 1:
            nbins = 30
        elif npart == 2:
            nbins = 25
        else:
            nbins = 20

        for i in range(npart):
            for j in range(nbins):
                y_branches.append('NN_error_{}_{}_{}'.format(direc, i, j))

    return x_branches, y_branches
