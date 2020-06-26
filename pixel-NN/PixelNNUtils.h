// -*- mode: c++ -*-

#include <utility>
#include <vector>

namespace PixelNN {

    enum Direction { X, Y };

    void
    normalize_inplace(std::vector<double>& dvec);

    std::pair<double, double>
    split_probabilities(std::vector<double>& nn_output);

    int
    estimate_number(std::vector<double>& nn_output,
    		    double threshold_2p = 0.6,
    		    double threshold_3p = 0.2);


    int
    estimate_number(std::pair<double, double>& probs,
		    double threshold_2p = 0.6,
		    double threshold_3p = 0.2);

    std::vector<double>
    hit_positions(std::vector<double>& nn_output,
		  double center_pos_X,
		  double center_pos_Y,
		  double size_Y,
		  std::vector<double>& pitches_Y);

    double
    correctedX(double center_pos, double pos_pixel);

    double
    correctedY(double center_pos,
	       double pos_pixels,
	       double size_Y,
	       std::vector<double>& pitches);

    std::vector<double>
    hit_position_uncertainty(std::vector<double>& nn_output,
			     Direction d,
			     int nparticles);

    int cluster_size(std::vector<double> &charge);
    int cluster_size_X(std::vector<double> &charges, int sizeX, int sizeY);
    int cluster_size_Y(std::vector<double> &charges, int sizeX, int sizeY);
}

