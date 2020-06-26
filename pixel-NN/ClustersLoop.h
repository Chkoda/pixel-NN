#ifndef pixel_NN_dataset_ClustersLoop_H
#define pixel_NN_dataset_ClustersLoop_H

#include <map>
#include <string>
#include <vector>
#include <TTree.h>
#include <TVector3.h>
#include <AthAnalysisBaseComps/AthAnalysisAlgorithm.h>
#include <TrkNeuralNetworkUtils/TTrainedNetwork.h>
#include <xAODTracking/TrackMeasurementValidation.h>
#include <xAODTracking/TrackMeasurementValidationContainer.h>

#define NUMBER  0
#define POS1    1
#define POS2    2
#define POS3    3
#define ERRORX1 4
#define ERRORY1 5
#define ERRORX2 6
#define ERRORY2 7
#define ERRORX3 8
#define ERRORY3 9

typedef xAOD::TrackMeasurementValidation Cluster;

class ClustersLoop : public AthAnalysisAlgorithm
{
public:
	ClustersLoop(const std::string& name, ISvcLocator* pSvcLocator);
	~ClustersLoop() {};

	// set in driver script
	std::string outputName;
	int NNtype;
	bool inclusive;
	int nclusters_1;
	int nclusters_2;
	int nclusters_3;
	bool nclusters_3_inclusive;
	int shuffle;
	std::string TTrainedNetworks_path;
	double max_eta;
	double max_index;

	// counters TODO init elswehere
	int cnt_clusters_1; //!
	int cnt_clusters_2; //!
	int cnt_clusters_3; //!

	bool do_error;
	TTrainedNetwork *ttnn;

	TTree *outtree;  //!

	// output variables
	double out_RunNumber; //!
	double out_EventNumber; //!
	double out_ClusterNumber; //!
	double out_sizeX; //!
	double out_sizeY; //!
	double out_localEtaPixelIndexWeightedPosition; //!
	double out_localPhiPixelIndexWeightedPosition; //!
	double out_layer; //!
	double out_barrelEC; //!
	double out_etaModule; //!
	double out_phi; //!
	double out_theta; //!
	double out_globalX; //!
	double out_globalY; //!
	double out_globalZ; //!
	double out_globalEta; //!
	double out_globalPhi; //!
	std::vector<double> out_matrix; //!
	std::vector<double> out_pitches; //!
	double out_cluster_size; //!
	double out_cluster_size_X; //!
	double out_cluster_size_Y; //!

	// TODO (eventually put all of this in vectors)
	double out_nparticles1; //!
	double out_nparticles2; //!
	double out_nparticles3; //!
	double out_nparticles_excess; //!
	double out_position_id_X_0; //!
	double out_position_id_Y_0; //!
	double out_position_id_X_1; //!
	double out_position_id_Y_1; //!
	double out_position_id_X_2; //!
	double out_position_id_Y_2; //!
	double out_position_X_0; //!
	double out_position_Y_0; //!
	double out_position_X_1; //!
	double out_position_Y_1; //!
	double out_position_X_2; //!
	double out_position_Y_2; //!
	double out_position_pred_id_X_0; //!
	double out_position_pred_id_Y_0; //!
	double out_position_pred_id_X_1; //!
	double out_position_pred_id_Y_1; //!
	double out_position_pred_id_X_2; //!
	double out_position_pred_id_Y_2; //!
	double out_position_pred_X_0; //!
	double out_position_pred_Y_0; //!
	double out_position_pred_X_1; //!
	double out_position_pred_Y_1; //!
	double out_position_pred_X_2; //!
	double out_position_pred_Y_2; //!

	std::vector<std::vector<double>> out_residualsX;
	std::vector<std::vector<double>> out_residualsY;

	std::vector<double> Output_number; //!
	double Output_number_estimated; //!
	double Output_number_true; //!
	std::vector<double> Output_positions_X; //!
	std::vector<double> Output_positions_Y; //!
	std::vector<double> Output_uncert_X; //!
	std::vector<double> Output_uncert_Y; //!
	std::vector<double> Output_true_X; //!
	std::vector<double> Output_true_Y; //!
	std::vector<double> Output_corr_positions_X; //!
	std::vector<double> Output_corr_positions_Y; //!
	std::vector<double> Output_corr_uncert_X; //!
	std::vector<double> Output_corr_uncert_Y; //!
	std::vector<double> Output_corr_true_X; //!
	std::vector<double> Output_corr_true_Y; //!

	bool m_first; //!
	bool m_has_flag; //!

	// AthAnalysisAlgorithm methods
	virtual StatusCode initialize();
	virtual StatusCode firstExecute();
	virtual StatusCode execute();
	virtual StatusCode finalize();

	// Internal methods
	bool cluster_type_needed(size_t multiplicity);
	void clustersLoop(const xAOD::TrackMeasurementValidationContainer*);
	bool has_evaluated_NN_info(const Cluster& cluster);
	void branch_evaluated_NN_info();
	void fill_evaluated_NN_info(const Cluster& cluster, size_t i);
	void fill_error_NN_info();
};

#endif
