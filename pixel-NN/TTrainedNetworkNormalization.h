#include <vector>
#include <TObject.h>
#include <TrkNeuralNetworkUtils/TTrainedNetwork.h>

class TTrainedNetworkNormalization {
public:
	TTrainedNetworkNormalization(TTrainedNetwork* net) :
		net(net)
		{};
	std::vector<double> scales();
	std::vector<double> offsets();
private:
	TTrainedNetwork *net;
};
