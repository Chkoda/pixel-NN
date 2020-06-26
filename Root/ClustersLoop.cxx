#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <sstream>
#include <utility>
#include <vector>
#include <TFile.h>
#include <AthContainers/AuxElement.h>
#include <xAODEventInfo/EventInfo.h>
#include <xAODRootAccess/Init.h>
#include <xAODRootAccess/TEvent.h>
#include <xAODTracking/TrackMeasurementValidation.h>
#include <pixel-NN/ClustersLoop.h>
#include <pixel-NN/PixelNNUtils.h>

ClustersLoop::ClustersLoop(const std::string& name, ISvcLocator* pSvcLocator) :
	AthAnalysisAlgorithm(name, pSvcLocator)
{
	declareProperty("NNtype", NNtype = NUMBER);
	declareProperty("inclusive", inclusive = false);
	declareProperty("nclusters_1", nclusters_1 = -1);
	declareProperty("nclusters_2", nclusters_2 = -1);
	declareProperty("nclusters_3", nclusters_3 = -1);
	declareProperty("nclusters_3_inclusive", nclusters_3_inclusive = true);
	declareProperty("shuffle", shuffle = 0);
	declareProperty("TTrainedNetworks_path", TTrainedNetworks_path = "");
	declareProperty("max_eta", max_eta = 2.5);
	declareProperty("max_index", max_index = 5);
}

StatusCode ClustersLoop::initialize ()
{
	cnt_clusters_1 = 0;
	cnt_clusters_2 = 0;
	cnt_clusters_3 = 0;

	outtree = new TTree("NNinput", "NNinput");
	CHECK(histSvc()->regTree("/OUTTREE/NNinput", outtree));

	ttnn = nullptr;
	if (TTrainedNetworks_path != "") {
		TFile *tf = TFile::Open(TTrainedNetworks_path.c_str(), "READ");
		if (NNtype == POS1)
			ttnn = (TTrainedNetwork*)tf->Get("ImpactPoints1P");
		else if (NNtype == POS2)
			ttnn = (TTrainedNetwork*)tf->Get("ImpactPoints2P");
		else if (NNtype == POS3)
			ttnn = (TTrainedNetwork*)tf->Get("ImpactPoints3P");
	}
	do_error = (ttnn != nullptr);

	outtree->Branch("RunNumber", &out_RunNumber);
	outtree->Branch("EventNumber", &out_EventNumber);
	outtree->Branch("ClusterNumber", &out_ClusterNumber);
	outtree->Branch("NN_sizeX", &out_sizeX);
	outtree->Branch("NN_sizeY", &out_sizeY);
	outtree->Branch("NN_localEtaPixelIndexWeightedPosition",
			&out_localEtaPixelIndexWeightedPosition);
	outtree->Branch("NN_localPhiPixelIndexWeightedPosition",
			&out_localPhiPixelIndexWeightedPosition);
	outtree->Branch("NN_layer", &out_layer);
	outtree->Branch("NN_barrelEC", &out_barrelEC);
	outtree->Branch("NN_etaModule", &out_etaModule);
	outtree->Branch("NN_phi", &out_phi);
	outtree->Branch("NN_theta", &out_theta);
	outtree->Branch("globalX", &out_globalX);
	outtree->Branch("globalY", &out_globalY);
	outtree->Branch("globalZ", &out_globalZ);
	outtree->Branch("globalEta", &out_globalEta);
	outtree->Branch("globalPhi", &out_globalPhi);
	outtree->Branch("cluster_size", &out_cluster_size);
	outtree->Branch("cluster_size_X", &out_cluster_size_X);
	outtree->Branch("cluster_size_Y", &out_cluster_size_Y);

	return StatusCode::SUCCESS;
}

StatusCode ClustersLoop::firstExecute()
{
	m_first = true;

	const xAOD::TrackMeasurementValidationContainer *clusters;
	CHECK(evtStore()->retrieve(clusters, "PixelClusters"));

	const Cluster *cl = clusters->at(0);
	out_sizeX = cl->auxdata<int>("NN_sizeX");
	out_sizeY = cl->auxdata<int>("NN_sizeY");
	out_matrix.resize(out_sizeX * out_sizeY);

	double *matrixPtr = out_matrix.data();
	char matrixbranch[strlen("NN_matrix???") + 1];
	for (int i = 0; i < out_sizeX*out_sizeY; i++) {
		std::sprintf(matrixbranch, "NN_matrix%d", i);
		outtree->Branch(matrixbranch, matrixPtr + i);
	}

	out_pitches.resize(out_sizeY);
	double *pitchesPtr = out_pitches.data();
	char pitchesbranch[strlen("NN_pitches???") + 1];
	for (int i = 0; i < out_sizeY; i++) {
		std::sprintf(pitchesbranch, "NN_pitches%d", i);
		outtree->Branch(pitchesbranch, pitchesPtr + i);
	}

	if (inclusive || NNtype == NUMBER) {
		outtree->Branch("NN_nparticles1", &out_nparticles1);
		outtree->Branch("NN_nparticles2", &out_nparticles2);
		outtree->Branch("NN_nparticles3", &out_nparticles3);
		outtree->Branch("NN_nparticles_excess", &out_nparticles_excess);
	}
	if (inclusive || (NNtype >= POS1 && NNtype <= POS3)) {
	    outtree->Branch("NN_position_id_X_0",
			    &out_position_id_X_0);
	    outtree->Branch("NN_position_id_Y_0",
			    &out_position_id_Y_0);
	    outtree->Branch("NN_position_X_0",
			    &out_position_X_0);
	    outtree->Branch("NN_position_Y_0",
			    &out_position_Y_0);
	    if (do_error) {
		    outtree->Branch("NN_position_pred_id_X_0",
				    &out_position_pred_id_X_0);
		    outtree->Branch("NN_position_pred_id_Y_0",
				    &out_position_pred_id_Y_0);
		    outtree->Branch("NN_position_pred_X_0",
				    &out_position_pred_X_0);
		    outtree->Branch("NN_position_pred_Y_0",
				    &out_position_pred_Y_0);
	    }
	}
	if (inclusive || (NNtype >= POS2 && NNtype <= POS3)) {
	    outtree->Branch("NN_position_id_X_1",
			    &out_position_id_X_1);
	    outtree->Branch("NN_position_id_Y_1",
			    &out_position_id_Y_1);
	    outtree->Branch("NN_position_X_1",
			    &out_position_X_1);
	    outtree->Branch("NN_position_Y_1",
			    &out_position_Y_1);
	    if (do_error) {
		    outtree->Branch("NN_position_pred_id_X_1",
				    &out_position_pred_id_X_1);
		    outtree->Branch("NN_position_pred_id_Y_1",
				    &out_position_pred_id_Y_1);
		    outtree->Branch("NN_position_pred_X_1",
				    &out_position_pred_X_1);
		    outtree->Branch("NN_position_pred_Y_1",
				    &out_position_pred_Y_1);
	    }
	}
	if (inclusive || NNtype == POS3) {
	    outtree->Branch("NN_position_id_X_2",
			    &out_position_id_X_2);
	    outtree->Branch("NN_position_id_Y_2",
			    &out_position_id_Y_2);
	    outtree->Branch("NN_position_X_2",
			    &out_position_X_2);
	    outtree->Branch("NN_position_Y_2",
			    &out_position_Y_2);
	    if (do_error) {
		    outtree->Branch("NN_position_pred_id_X_2",
				    &out_position_pred_id_X_2);
		    outtree->Branch("NN_position_pred_id_Y_2",
				    &out_position_pred_id_Y_2);
		    outtree->Branch("NN_position_pred_X_2",
				    &out_position_pred_X_2);
		    outtree->Branch("NN_position_pred_Y_2",
				    &out_position_pred_Y_2);
	    }
	}

	if (has_evaluated_NN_info(*cl))
		branch_evaluated_NN_info();

	if (do_error) {
		int nbins, npart;
		if (NNtype == POS1) {
			npart = 1;
			nbins = 30;
		} else if (NNtype == POS2) {
			npart = 2;
			nbins = 25;
		} else if (NNtype == POS3) {
			npart = 3;
			nbins = 20;
		} else {
			return StatusCode::FAILURE;
		}
		for (int i = 0; i < npart; i++) {
			out_residualsX.push_back(std::vector<double>(nbins));
			out_residualsY.push_back(std::vector<double>(nbins));
			for (int j = 0; j < nbins; j++) {
				std::ostringstream keyX;
				keyX << "NN_error_X_" << i << "_" << j;
				outtree->Branch(
					keyX.str().c_str(),
					&out_residualsX.at(i).at(j));
				std::ostringstream keyY;
				keyY << "NN_error_Y_" << i << "_" << j;
				outtree->Branch(
					keyY.str().c_str(),
					&out_residualsY.at(i).at(j));
			}
		}

	}

	return StatusCode::SUCCESS;
}

StatusCode ClustersLoop::execute ()
{
	const xAOD::EventInfo *eventInfo = 0;
	CHECK(evtStore()->retrieve(eventInfo, "EventInfo"));

	out_RunNumber = eventInfo->runNumber();
	out_EventNumber = eventInfo->eventNumber();

	const xAOD::TrackMeasurementValidationContainer *clusters;
	CHECK(evtStore()->retrieve(clusters, "PixelClusters"));
	clustersLoop(clusters);

	return StatusCode::SUCCESS;
}

#define accessor(n,t,v) SG::AuxElement::ConstAccessor< t > n (v)

accessor(a_skip_cluster, bool, "skip_cluster");
accessor(a_sizeX, int, "NN_sizeX");
accessor(a_sizeY, int, "NN_sizeY");
accessor(a_posX, std::vector<float>, "NN_positions_indexX");
accessor(a_posY, std::vector<float>, "NN_positions_indexY");
accessor(a_posX_mm, std::vector<float>, "NN_positionsX");
accessor(a_posY_mm, std::vector<float>, "NN_positionsY");
accessor(a_localEtaPixelIndexWeightedPosition, float,
	 "NN_localEtaPixelIndexWeightedPosition");
accessor(a_localPhiPixelIndexWeightedPosition, float,
	 "NN_localPhiPixelIndexWeightedPosition");
accessor(a_layer, int, "layer");
accessor(a_bec, int, "bec");
accessor(a_etaModule, int, "eta_module");
accessor(a_matrix, std::vector<float>, "NN_matrixOfCharge");
accessor(a_pitches, std::vector<float>, "NN_vectorOfPitchesY");
accessor(a_theta, std::vector<float>,  "NN_theta");
accessor(a_phi, std::vector<float>,  "NN_phi");
accessor(a_globalX, float, "globalX");
accessor(a_globalY, float, "globalY");
accessor(a_globalZ, float, "globalZ");

#undef accessor

typedef std::vector<std::pair<float,float> > Positions;

bool pairComp(std::pair<float,float> a, std::pair<float,float> b)
{
	if (a.first == b.first)
		return a.second < b.second;
	else
		return a.first < b.first;
}

Positions sortedPositions(std::vector<float>& xs, std::vector<float>& ys)
{
	Positions ps;
	for (size_t i = 0; i < xs.size(); i++)
		ps.push_back(std::make_pair(xs.at(i), ys.at(i)));
	std::sort(ps.begin(), ps.end(), pairComp);
	return ps;
}

bool ClustersLoop::cluster_type_needed(size_t npart)
{
	if (npart == 0)
		return false;

	if (npart == 1 && nclusters_1 > -1) {
		if (cnt_clusters_1 >= nclusters_1)
			return false;
		cnt_clusters_1++;
	}

	if (npart == 2 && nclusters_2 > -1) {
		if (cnt_clusters_2 >= nclusters_2)
			return false;
		cnt_clusters_2++;
	}

	if (npart == 3 && nclusters_3 > -1) {
		if (cnt_clusters_3 >= nclusters_3)
			return false;
		cnt_clusters_3++;
	}

	if (npart > 3) {
		if (!nclusters_3_inclusive)
			return false;
		if (nclusters_3 > -1) {
			if (cnt_clusters_3 >= nclusters_3)
				return false;
		}
		cnt_clusters_3++;
	}

	return true;
}

void ClustersLoop::clustersLoop(const xAOD::TrackMeasurementValidationContainer* clusters)
{
	if (clusters->size() == 0)
		return;

	out_ClusterNumber = 0;
	for (auto c : *clusters) {

		if (m_first) {
			m_first = false;
			m_has_flag = a_skip_cluster.isAvailable(*c);
			std::cout << "skip_cluster flag available: "
				  << (m_has_flag? "True" : "False")
				  << std::endl;
		}

		if (m_has_flag && a_skip_cluster(*c))
			continue;

		/* Fetch all cluster observables */
		const std::vector<float> matrix = a_matrix(*c);
		std::vector<float> posX = a_posX(*c);
		std::vector<float> posY = a_posY(*c);
		std::vector<float> posX_mm = a_posX_mm(*c);
		std::vector<float> posY_mm = a_posY_mm(*c);
		const std::vector<float> pitches = a_pitches(*c);
		out_localEtaPixelIndexWeightedPosition =
			a_localEtaPixelIndexWeightedPosition(*c);
		out_localPhiPixelIndexWeightedPosition =
			a_localPhiPixelIndexWeightedPosition(*c);
		out_layer = a_layer(*c);
		out_barrelEC = a_bec(*c);
		out_etaModule = a_etaModule(*c);
		std::vector<float> theta = a_theta(*c);
		std::vector<float> phi = a_phi(*c);
		int sizeX = a_sizeX(*c);
		int sizeY = a_sizeY(*c);
		out_globalX = a_globalX(*c);
		out_globalY = a_globalY(*c);
		out_globalZ = a_globalZ(*c);
		out_globalEta =
			TVector3(out_globalX, out_globalY, out_globalZ).Eta();
		out_globalPhi =
			TVector3(out_globalX, out_globalY, out_globalZ).Phi();
		out_cluster_size = PixelNN::cluster_size(out_matrix);
		out_cluster_size_X = PixelNN::cluster_size_X(out_matrix,
							     out_sizeX,
							     out_sizeY);
		out_cluster_size_Y = PixelNN::cluster_size_Y(out_matrix,
							     out_sizeX,
							     out_sizeY);

		/* check if good cluster */
		// matrix size
		if (matrix.size() == 0)
			continue;
		if ((int)matrix.size() != sizeX * sizeY)
			continue;
		if (matrix.size() != out_matrix.size())
			continue;
		// NN_sizeX value
		if (sizeX == -100)
			continue;
		// BEC
		if (abs(out_barrelEC) > 2)
			continue;

		if (fabs(out_globalEta) > max_eta)
			continue;

		/* Fill charge matrix */
		for (size_t i = 0; i < matrix.size(); i++)
			out_matrix.at(i) = matrix.at(i);

		/* Fill vector of pitches */
		for (size_t i = 0; i < pitches.size(); i++)
			out_pitches.at(i) = pitches.at(i);

		/* Determine number of particles */
		out_nparticles1 = posX.size() == 1;
		out_nparticles2 = posX.size() == 2;
		out_nparticles3 = posX.size() >= 3;
		out_nparticles_excess = posX.size() > 3;

		/* Sort and store the positions */
		out_position_id_X_0 = 0;
		out_position_id_Y_0 = 0;
		out_position_id_X_1 = 0;
		out_position_id_Y_1 = 0;
		out_position_id_X_2 = 0;
		out_position_id_Y_2 = 0;
		Positions ps = sortedPositions(posX, posY);
		Positions ps_mm = sortedPositions(posX_mm, posY_mm);
		if (ps.size() >= 1) {
			out_position_id_X_0 = ps.at(0).first;
			out_position_id_Y_0 = ps.at(0).second;
			out_position_X_0 = ps_mm.at(0).first;
			out_position_Y_0 = ps_mm.at(0).second;
		}
		if (ps.size() >= 2) {
			out_position_id_X_1 = ps.at(1).first;
			out_position_id_Y_1 = ps.at(1).second;
			out_position_X_1 = ps_mm.at(1).first;
			out_position_Y_1 = ps_mm.at(1).second;
		}
		if (ps.size() >= 3) {
			out_position_id_X_2 = ps.at(2).first;
			out_position_id_Y_2 = ps.at(2).second;
			out_position_X_2 = ps_mm.at(2).first;
			out_position_Y_2 = ps_mm.at(2).second;
		}
		if ((fabs(out_position_id_X_0) > max_index) ||
		    (fabs(out_position_id_Y_0) > max_index) ||
		    (fabs(out_position_id_X_1) > max_index) ||
		    (fabs(out_position_id_Y_1) > max_index) ||
		    (fabs(out_position_id_X_2) > max_index) ||
		    (fabs(out_position_id_Y_2) > max_index))
			continue;

		if (std::isinf(out_position_id_X_0) ||
		    std::isnan(out_position_id_X_0) ||
		    std::isinf(out_position_id_Y_0) ||
		    std::isnan(out_position_id_Y_0) ||
		    std::isinf(out_position_id_X_1) ||
		    std::isnan(out_position_id_X_1) ||
		    std::isinf(out_position_id_Y_1) ||
		    std::isnan(out_position_id_Y_1) ||
		    std::isinf(out_position_id_X_2) ||
		    std::isnan(out_position_id_X_2) ||
		    std::isinf(out_position_id_Y_2) ||
		    std::isnan(out_position_id_Y_2))
			continue;

		    /* Loop over angles */
		for (size_t i = 0; i < theta.size(); i++) {
			out_theta = theta.at(i);
			if (out_theta == 0 || std::isnan(out_theta))
				continue;

			if (!cluster_type_needed(posX.size()))
				continue;

			out_phi   = phi.at(i);
			if (out_barrelEC == 2) {
				out_theta *= -1;
				out_phi   *= -1;
			}

			if (do_error)
				fill_error_NN_info();

			if (has_evaluated_NN_info(*c))
				fill_evaluated_NN_info(*c, i);

			out_ClusterNumber += 1;
			outtree->Fill();

		}
	}
}

void
fill_amplitude(double res, std::vector<double>& vect, double max)
{
	double sf = 1.0 / (2.0 * max);
	double epsilon = std::numeric_limits<double>::epsilon();
	int  amp = (res + max) * sf * ((vect.size() - 1) - epsilon);
	if (amp < 0)
		amp = 0;
	if ((size_t)amp > (vect.size() - 1))
		amp = vect.size() - 1;

	for (size_t i = 0; i < vect.size(); i++) {
		vect.at(i) = (double)(i == (size_t)amp);
	}
}

void
ClustersLoop::fill_error_NN_info()
{
	std::vector<double> input;
	for (double d : out_matrix)
		input.push_back(d);
	for (double d: out_pitches)
		input.push_back(d);
	input.push_back(out_layer);
	input.push_back(out_barrelEC);
	input.push_back(out_phi);
	input.push_back(out_theta);

	std::vector<double> pos_pred_id =
		ttnn->calculateNormalized(input);
	std::vector<double> pos_pred = PixelNN::hit_positions(
		pos_pred_id,
		out_localPhiPixelIndexWeightedPosition,
		out_localEtaPixelIndexWeightedPosition,
		out_sizeY,
		out_pitches);

	std::vector<double> pos_truth;

	if (NNtype >= POS1) {
		out_position_pred_id_X_0 = pos_pred_id.at(0);
		out_position_pred_id_Y_0 = pos_pred_id.at(1);
		out_position_pred_X_0 = pos_pred.at(0);
		out_position_pred_Y_0 = pos_pred.at(1);
		pos_truth.push_back(out_position_X_0);
		pos_truth.push_back(out_position_Y_0);
	}
	if (NNtype >= POS2) {
		out_position_pred_id_X_1 = pos_pred_id.at(2);
		out_position_pred_id_Y_1 = pos_pred_id.at(3);
		out_position_pred_X_1 = pos_pred.at(2);
		out_position_pred_Y_1 = pos_pred.at(3);
		pos_truth.push_back(out_position_X_1);
		pos_truth.push_back(out_position_Y_1);
	}
	if (NNtype >= POS3) {
		out_position_pred_id_X_2 = pos_pred_id.at(4);
		out_position_pred_id_Y_2 = pos_pred_id.at(5);
		out_position_pred_X_2 = pos_pred.at(4);
		out_position_pred_Y_2 = pos_pred.at(5);
		pos_truth.push_back(out_position_X_2);
		pos_truth.push_back(out_position_Y_2);
	}

	for (size_t i = 0; i < pos_pred.size() / 2; i++) {
		// X
		double x_pred = pos_pred.at(i*2);
		double x_truth = pos_truth.at(i*2);
		double max = (i == 0)? 0.03 : 0.05;
		fill_amplitude(x_pred - x_truth, out_residualsX.at(i), max);
		/* Y */
		double y_pred = pos_pred.at(i*2 + 1);
		double y_truth = pos_truth.at(i*2 +  1);
		max = (i == 0)? 0.3 : 0.5;
		fill_amplitude(y_pred - y_truth, out_residualsY.at(i), max);
	}

}

bool
ClustersLoop::has_evaluated_NN_info(const Cluster& c)
{
	return c.isAvailable<std::vector<std::vector<double>>>("Output_number");
}

void
ClustersLoop::branch_evaluated_NN_info()
{
	outtree->Branch("Output_number", &Output_number);
	outtree->Branch("Output_number_estimated", &Output_number_estimated);
	outtree->Branch("Output_number_true", &Output_number_true);
	outtree->Branch("Output_positions_X", &Output_positions_X);
	outtree->Branch("Output_positions_Y", &Output_positions_Y);
	outtree->Branch("Output_uncert_X", &Output_uncert_X);
	outtree->Branch("Output_uncert_Y", &Output_uncert_Y);
	outtree->Branch("Output_true_X", &Output_true_X);
	outtree->Branch("Output_true_Y", &Output_true_Y);
	outtree->Branch("Output_corr_positions_X", &Output_corr_positions_X);
	outtree->Branch("Output_corr_positions_Y", &Output_corr_positions_Y);
	outtree->Branch("Output_corr_uncert_X", &Output_corr_uncert_X);
	outtree->Branch("Output_corr_uncert_Y", &Output_corr_uncert_Y);
	outtree->Branch("Output_corr_true_X", &Output_corr_true_X);
	outtree->Branch("Output_corr_true_Y", &Output_corr_true_Y);
}

#define GET_VV(n) Output_ ## n =					\
		cluster.auxdata<std::vector<std::vector<double>>>	\
		("Output_" #n).at(i)

void ClustersLoop::fill_evaluated_NN_info(const Cluster& cluster, size_t i)
{

	GET_VV(number);
	PixelNN::normalize_inplace(Output_number);
	Output_number_estimated = PixelNN::estimate_number(Output_number);

	GET_VV(positions_X);
	GET_VV(positions_Y);
	GET_VV(uncert_X);
	GET_VV(uncert_Y);
	GET_VV(true_X);
	GET_VV(true_Y);
	GET_VV(corr_positions_X);
	GET_VV(corr_positions_Y);
	GET_VV(corr_uncert_X);
	GET_VV(corr_uncert_Y);
	GET_VV(corr_true_X);
	GET_VV(corr_true_Y);

	Output_number_true = a_posX(cluster).size();
}

TTree* shuffled(TTree* tree, int seed)
{
	TTree *newtree = tree->CloneTree(0);
	std::vector<Long64_t> indices(tree->GetEntries());
	std::iota(indices.begin(), indices.end(), 0);
	std::shuffle(
		indices.begin(),
		indices.end(),
		std::default_random_engine(seed)
		);
	tree->LoadBaskets(2328673567232LL);
	for (Long64_t& i : indices) {
		tree->GetEntry(i);
		newtree->Fill();
	}
	return newtree;
}

StatusCode ClustersLoop::finalize()
{
	if (shuffle > 0) {
		TTree *newtree = shuffled(outtree, shuffle);
		CHECK(histSvc()->deReg(outtree));
		CHECK(histSvc()->regTree("/OUTTREE/NNinput", newtree));
		delete outtree;
		outtree = newtree;
	}

	return StatusCode::SUCCESS;
}

#undef GET_VV
