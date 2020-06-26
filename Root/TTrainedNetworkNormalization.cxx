#include "pixel-NN/TTrainedNetworkNormalization.h"

std::vector<double> TTrainedNetworkNormalization::scales()
{
	std::vector<double> s;
	for (TTrainedNetwork::Input inpt : net->getInputs())
		s.push_back(inpt.scale);
	return s;
}

std::vector<double> TTrainedNetworkNormalization::offsets()
{
	std::vector<double> s;
	for (TTrainedNetwork::Input inpt : net->getInputs())
		s.push_back(inpt.offset);
	return s;
}
