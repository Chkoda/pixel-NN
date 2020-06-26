#define ValidationLoop_cxx
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

#include <pixel-NN/PixelNNUtils.h>
#include <pixel-NN/ValidationLoop.h>


std::vector<double>
ValidationLoop::NN_input_vector()
{
    std::vector<double> input;
    input.push_back(NN_matrix0);
    input.push_back(NN_matrix1);
    input.push_back(NN_matrix2);
    input.push_back(NN_matrix3);
    input.push_back(NN_matrix4);
    input.push_back(NN_matrix5);
    input.push_back(NN_matrix6);
    input.push_back(NN_matrix7);
    input.push_back(NN_matrix8);
    input.push_back(NN_matrix9);
    input.push_back(NN_matrix10);
    input.push_back(NN_matrix11);
    input.push_back(NN_matrix12);
    input.push_back(NN_matrix13);
    input.push_back(NN_matrix14);
    input.push_back(NN_matrix15);
    input.push_back(NN_matrix16);
    input.push_back(NN_matrix17);
    input.push_back(NN_matrix18);
    input.push_back(NN_matrix19);
    input.push_back(NN_matrix20);
    input.push_back(NN_matrix21);
    input.push_back(NN_matrix22);
    input.push_back(NN_matrix23);
    input.push_back(NN_matrix24);
    input.push_back(NN_matrix25);
    input.push_back(NN_matrix26);
    input.push_back(NN_matrix27);
    input.push_back(NN_matrix28);
    input.push_back(NN_matrix29);
    input.push_back(NN_matrix30);
    input.push_back(NN_matrix31);
    input.push_back(NN_matrix32);
    input.push_back(NN_matrix33);
    input.push_back(NN_matrix34);
    input.push_back(NN_matrix35);
    input.push_back(NN_matrix36);
    input.push_back(NN_matrix37);
    input.push_back(NN_matrix38);
    input.push_back(NN_matrix39);
    input.push_back(NN_matrix40);
    input.push_back(NN_matrix41);
    input.push_back(NN_matrix42);
    input.push_back(NN_matrix43);
    input.push_back(NN_matrix44);
    input.push_back(NN_matrix45);
    input.push_back(NN_matrix46);
    input.push_back(NN_matrix47);
    input.push_back(NN_matrix48);
    input.push_back(NN_pitches0);
    input.push_back(NN_pitches1);
    input.push_back(NN_pitches2);
    input.push_back(NN_pitches3);
    input.push_back(NN_pitches4);
    input.push_back(NN_pitches5);
    input.push_back(NN_pitches6);
    input.push_back(NN_layer);
    input.push_back(NN_barrelEC);
    input.push_back(NN_phi);
    input.push_back(NN_theta);
    return input;
}

std::vector<double>
ValidationLoop::collect_pitches_Y()
{
	std::vector<double> pitches;
	pitches.push_back(NN_pitches0);
	pitches.push_back(NN_pitches1);
	pitches.push_back(NN_pitches2);
	pitches.push_back(NN_pitches3);
	pitches.push_back(NN_pitches4);
	pitches.push_back(NN_pitches5);
	pitches.push_back(NN_pitches6);
	return pitches;
}


std::vector<double>
ValidationLoop::collect_positions()
{
	std::vector<double> pos;

	pos.push_back(NN_position_id_X_0);
	pos.push_back(NN_position_id_Y_0);

	if (NN_nparticles2 == 1 || NN_nparticles3 == 1) {
		pos.push_back(NN_position_id_X_1);
		pos.push_back(NN_position_id_Y_1);
	}

	if (NN_nparticles3 == 1) {
		pos.push_back(NN_position_id_X_2);
		pos.push_back(NN_position_id_Y_2);
	}

	return pos;
}

#include <iostream>
void ValidationLoop::Loop()
{
    if (fChain == 0) return;

    Long64_t nentries = fChain->GetEntriesFast();

    Long64_t nbytes = 0, nb = 0;
    for (Long64_t jentry=0; jentry<nentries;jentry++) {
    	Long64_t ientry = LoadTree(jentry);
    	if (ientry < 0) break;
    	nb = fChain->GetEntry(jentry);   nbytes += nb;

    	std::vector<double> NNinput = NN_input_vector();
    	std::vector<double> number_output =
    	    NN_number->calculateNormalized(NNinput);

	PixelNN::normalize_inplace(number_output);
	Output_nparticles1 = number_output.at(0);
	Output_nparticles2 = number_output.at(1);
	Output_nparticles3 = number_output.at(2);
	Output_nparticles = PixelNN::estimate_number(number_output);

	std::vector<double> pos_output;

	if (NN_nparticles1 == 1)
		pos_output = NN_pos1->calculateNormalized(NNinput);
	else if (NN_nparticles2 == 1)
		pos_output = NN_pos2->calculateNormalized(NNinput);
	else if (NN_nparticles3 == 1 && NN_nparticles_excess == 0)
		pos_output = NN_pos3->calculateNormalized(NNinput);
	else {
		Output_estimated_positions->clear();
		Output_true_positions->clear();
		Output_uncertainty_X->clear();
		Output_uncertainty_Y->clear();
		continue;
	}
	*Output_estimated_positions_raw = pos_output;


	std::vector<double> pitches = collect_pitches_Y();
	*Output_estimated_positions =
		PixelNN::hit_positions(pos_output,
				       NN_localPhiPixelIndexWeightedPosition,
				       NN_localEtaPixelIndexWeightedPosition,
				       NN_sizeY,
				       pitches);

	std::vector<double> true_pos = collect_positions();
	*Output_true_positions_raw = true_pos;
	*Output_true_positions =
		PixelNN::hit_positions(true_pos,
				       NN_localPhiPixelIndexWeightedPosition,
				       NN_localEtaPixelIndexWeightedPosition,
				       NN_sizeY,
				       pitches);

	NNinput.insert(NNinput.end(), pos_output.begin(), pos_output.end());
	std::vector<double> errorx_output;
	std::vector<double> errory_output;
	int np;
	if (NN_nparticles1 == 1) {
		errorx_output = NN_error1x->calculateNormalized(NNinput);
		errory_output = NN_error1y->calculateNormalized(NNinput);
		np = 1;
	} else if (NN_nparticles2 == 1) {
		errorx_output = NN_error2x->calculateNormalized(NNinput);
		errory_output = NN_error2y->calculateNormalized(NNinput);
		np = 2;
	} else if (NN_nparticles3 == 1) {
		errorx_output = NN_error3x->calculateNormalized(NNinput);
		errory_output = NN_error3y->calculateNormalized(NNinput);
		np = 3;
	}

	*Output_uncertainty_X =
		PixelNN::hit_position_uncertainty(errorx_output,
						    PixelNN::Direction::X,
						    np);

	*Output_uncertainty_Y =
		PixelNN::hit_position_uncertainty(errory_output,
						  PixelNN::Direction::Y,
						  np);

	new_tree->Fill();
    }
    output_file->Write(nullptr, TObject::kWriteDelete);
}
