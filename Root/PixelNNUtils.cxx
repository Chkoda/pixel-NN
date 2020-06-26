#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <pixel-NN/PixelNNUtils.h>

// TODO: test this
void
PixelNN::normalize_inplace(std::vector<double>& dvec)
{
    double acc = 0;
    for (double d : dvec)
	acc += d;
    for (double &d : dvec)
	d /= acc;
}

std::pair<double, double>
PixelNN::split_probabilities(std::vector<double>& nn_output)
{
    if (nn_output.size() != 3)
	throw std::length_error("expected length-3 vector");

    double tot = 0;
    for (double p : nn_output)
	tot += p;

    return std::make_pair(nn_output.at(1) / tot, nn_output.at(2) / tot);
}

int
PixelNN::estimate_number(std::vector<double>& nn_output,
			 double threshold_2p,
			 double threshold_3p)
{
    std::pair<double, double> probs = PixelNN::split_probabilities(nn_output);
    return estimate_number(probs, threshold_2p, threshold_3p);
}

int
PixelNN::estimate_number(std::pair<double,double>& probs,
			 double threshold_2p,
			 double threshold_3p)
{
    if (probs.second > threshold_3p)
	return 3;
    if (probs.first > threshold_2p)
	return 2;
    else
	return 1;
}

std::vector<double>
PixelNN::hit_positions(std::vector<double>& nn_output,
		       double center_pos_X,
		       double center_pos_Y,
		       double size_Y,
		       std::vector<double>& pitches_Y)
{
    std::vector<double> corrected(nn_output.size());
    for (size_t i = 0; i < nn_output.size(); i += 2) {
	corrected.at(i) = correctedX(center_pos_X, nn_output.at(i));
	corrected.at(i + 1) = correctedY(center_pos_Y,
					 nn_output.at(i + 1),
					 size_Y,
					 pitches_Y);

    }

    return corrected;
}

double
PixelNN::correctedX(double center_pos,
		    double pos_pixels)
{
  double pitch = 0.05;
  return center_pos + pos_pixels * pitch;
}
#include <iostream>

double
PixelNN::correctedY(double center_pos,
		    double pos_pixels,
		    double size_Y,
		    std::vector<double>& pitches)
{
    double p = pos_pixels + (size_Y - 1) / 2.0;
    double p_Y = -100;
    double p_center = -100;
    double p_actual = 0;

    for (int i = 0; i < size_Y; i++) {
	if (p >= i && p <= (i + 1))
	    p_Y = p_actual + (p - i + 0.5) * pitches.at(i);
	if (i == (size_Y - 1) / 2)
	    p_center = p_actual + 0.5 * pitches.at(i);
	p_actual += pitches.at(i);
    }

    return center_pos + p_Y - p_center;
}

std::vector<double>
PixelNN::hit_position_uncertainty(std::vector<double>& nn_output,
				  Direction d,
				  int nparticles)
{
    std::vector<double> output_rms;

    if (nparticles < 1 || nparticles > 3)
	throw "nparticles < 0 || nparticles > 3";

    double maximum;
    if (d == Direction::X)
	maximum = (nparticles == 1)? 0.03 : 0.05;
    else
	maximum = (nparticles == 1)? 0.3 : 0.4;

    double minimum = -maximum;
    int dist_size = (int)nn_output.size() / nparticles;

    for (int i = 0; i < nparticles; i++)
    {
	double acc = 0;
	for (int u = 0; u < dist_size; u++)
	    acc += nn_output[i * dist_size + u];
	double rms = 0;
	for (int u = 0; u < dist_size; u++) {
	    rms += nn_output[i * dist_size + u] / acc * std::pow(minimum + (maximum - minimum)/(double)(dist_size - 2) * (u - 1./2.), 2);
	}
	rms = sqrt(rms);

	//now recompute between -3*RMSx and +3*RMSx
	double interval = 3 * rms;

	int min_bin = (int)(1+ (-interval - minimum) / (maximum - minimum) * (double)(dist_size - 2));
	int max_bin = (int)(1 +( interval - minimum) / (maximum - minimum) * (double)(dist_size - 2));

	if (max_bin > dist_size - 1)
	    max_bin = dist_size - 1;
	if (min_bin < 0)
	    min_bin = 0;

	rms = 0;
	for (int u = min_bin; u < max_bin + 1; u++)
	    rms += nn_output[i * dist_size + u] / acc * std::pow(minimum + (maximum - minimum)/(double)(dist_size - 2) * (u - 1./2.), 2);

	rms = sqrt(rms);
	output_rms.push_back(rms);
    }
    return output_rms;
}

int PixelNN::cluster_size(std::vector<double> &charges)
{
	int acc = 0;
	for (double c: charges)
		acc += (c > 0);
	return acc;
}

int PixelNN::cluster_size_X(std::vector<double> &charges, int sizeX, int sizeY)
{
	std::vector<int> non_empty;
	for (int x = 0; x < sizeX; x++) {
		bool has_charge = false;
		for (int y = 0; y < sizeY; y++) {
			if (charges.at(x * sizeX + y) > 0) {
				has_charge = true;
				break;
			}
		}
		if (has_charge)
			non_empty.push_back(x);
	}
	if (non_empty.size() == 0)
		return 0;
	auto minmax = std::minmax_element(non_empty.begin(), non_empty.end());
	return minmax.second - minmax.first + 1;
}

int PixelNN::cluster_size_Y(std::vector<double> &charges, int sizeX, int sizeY)
{
	std::vector<int> non_empty;
	for (int y = 0; y < sizeY; y++) {
		bool has_charge = false;
		for (int x = 0; x < sizeX; x++) {
			if (charges.at(x * sizeX + y) > 0) {
				has_charge = true;
				break;
			}
		}
		if (has_charge)
			non_empty.push_back(y);
	}
	if (non_empty.size() == 0)
		return 0;
	auto minmax = std::minmax_element(non_empty.begin(), non_empty.end());
	return minmax.second - minmax.first + 1;
}
