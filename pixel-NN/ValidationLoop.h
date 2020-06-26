// -*- mode: c++ -*-
//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Mon Jun  5 09:15:12 2017 by ROOT version 6.08/00
// from TTree NNinput/NNinput
// found on file: ../../run/submitDir/data-NNinput/mc16_valid.361027.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ7W.recon.DAOD_IDTIDE.e3668_s2995_r8618_tid09730060_00.root
//////////////////////////////////////////////////////////

#ifndef ValidationLoop_h
#define ValidationLoop_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

#include <TrkNeuralNetworkUtils/TTrainedNetwork.h>

// Header file for the classes stored in the TTree if any.

class ValidationLoop {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   Double_t        RunNumber;
   Double_t        EventNumber;
   Double_t        ClusterNumber;
   Double_t        NN_sizeX;
   Double_t        NN_sizeY;
   Double_t        NN_localEtaPixelIndexWeightedPosition;
   Double_t        NN_localPhiPixelIndexWeightedPosition;
   Double_t        NN_layer;
   Double_t        NN_barrelEC;
   Double_t        NN_etaModule;
   Double_t        NN_phi;
   Double_t        NN_theta;
   Double_t        globalX;
   Double_t        globalY;
   Double_t        globalZ;
   Double_t        globalEta;
   Double_t        NN_matrix0;
   Double_t        NN_matrix1;
   Double_t        NN_matrix2;
   Double_t        NN_matrix3;
   Double_t        NN_matrix4;
   Double_t        NN_matrix5;
   Double_t        NN_matrix6;
   Double_t        NN_matrix7;
   Double_t        NN_matrix8;
   Double_t        NN_matrix9;
   Double_t        NN_matrix10;
   Double_t        NN_matrix11;
   Double_t        NN_matrix12;
   Double_t        NN_matrix13;
   Double_t        NN_matrix14;
   Double_t        NN_matrix15;
   Double_t        NN_matrix16;
   Double_t        NN_matrix17;
   Double_t        NN_matrix18;
   Double_t        NN_matrix19;
   Double_t        NN_matrix20;
   Double_t        NN_matrix21;
   Double_t        NN_matrix22;
   Double_t        NN_matrix23;
   Double_t        NN_matrix24;
   Double_t        NN_matrix25;
   Double_t        NN_matrix26;
   Double_t        NN_matrix27;
   Double_t        NN_matrix28;
   Double_t        NN_matrix29;
   Double_t        NN_matrix30;
   Double_t        NN_matrix31;
   Double_t        NN_matrix32;
   Double_t        NN_matrix33;
   Double_t        NN_matrix34;
   Double_t        NN_matrix35;
   Double_t        NN_matrix36;
   Double_t        NN_matrix37;
   Double_t        NN_matrix38;
   Double_t        NN_matrix39;
   Double_t        NN_matrix40;
   Double_t        NN_matrix41;
   Double_t        NN_matrix42;
   Double_t        NN_matrix43;
   Double_t        NN_matrix44;
   Double_t        NN_matrix45;
   Double_t        NN_matrix46;
   Double_t        NN_matrix47;
   Double_t        NN_matrix48;
   Double_t        NN_pitches0;
   Double_t        NN_pitches1;
   Double_t        NN_pitches2;
   Double_t        NN_pitches3;
   Double_t        NN_pitches4;
   Double_t        NN_pitches5;
   Double_t        NN_pitches6;
   Double_t        NN_nparticles1;
   Double_t        NN_nparticles2;
   Double_t        NN_nparticles3;
    Double_t        NN_nparticles_excess;
    Double_t        NN_position_id_X_0;
    Double_t        NN_position_id_Y_0;
    Double_t        NN_position_id_X_1;
    Double_t        NN_position_id_Y_1;
    Double_t        NN_position_id_X_2;
    Double_t        NN_position_id_Y_2;


   // List of branches
   TBranch        *b_RunNumber;   //!
   TBranch        *b_EventNumber;   //!
   TBranch        *b_ClusterNumber;   //!
   TBranch        *b_NN_sizeX;   //!
   TBranch        *b_NN_sizeY;   //!
   TBranch        *b_NN_localEtaPixelIndexWeightedPosition;   //!
   TBranch        *b_NN_localPhiPixelIndexWeightedPosition;   //!
   TBranch        *b_NN_layer;   //!
   TBranch        *b_NN_barrelEC;   //!
   TBranch        *b_NN_etaModule;   //!
   TBranch        *b_NN_phi;   //!
   TBranch        *b_NN_theta;   //!
   TBranch        *b_globalX;   //!
   TBranch        *b_globalY;   //!
   TBranch        *b_globalZ;   //!
   TBranch        *b_globalEta;   //!
   TBranch        *b_NN_matrix0;   //!
   TBranch        *b_NN_matrix1;   //!
   TBranch        *b_NN_matrix2;   //!
   TBranch        *b_NN_matrix3;   //!
   TBranch        *b_NN_matrix4;   //!
   TBranch        *b_NN_matrix5;   //!
   TBranch        *b_NN_matrix6;   //!
   TBranch        *b_NN_matrix7;   //!
   TBranch        *b_NN_matrix8;   //!
   TBranch        *b_NN_matrix9;   //!
   TBranch        *b_NN_matrix10;   //!
   TBranch        *b_NN_matrix11;   //!
   TBranch        *b_NN_matrix12;   //!
   TBranch        *b_NN_matrix13;   //!
   TBranch        *b_NN_matrix14;   //!
   TBranch        *b_NN_matrix15;   //!
   TBranch        *b_NN_matrix16;   //!
   TBranch        *b_NN_matrix17;   //!
   TBranch        *b_NN_matrix18;   //!
   TBranch        *b_NN_matrix19;   //!
   TBranch        *b_NN_matrix20;   //!
   TBranch        *b_NN_matrix21;   //!
   TBranch        *b_NN_matrix22;   //!
   TBranch        *b_NN_matrix23;   //!
   TBranch        *b_NN_matrix24;   //!
   TBranch        *b_NN_matrix25;   //!
   TBranch        *b_NN_matrix26;   //!
   TBranch        *b_NN_matrix27;   //!
   TBranch        *b_NN_matrix28;   //!
   TBranch        *b_NN_matrix29;   //!
   TBranch        *b_NN_matrix30;   //!
   TBranch        *b_NN_matrix31;   //!
   TBranch        *b_NN_matrix32;   //!
   TBranch        *b_NN_matrix33;   //!
   TBranch        *b_NN_matrix34;   //!
   TBranch        *b_NN_matrix35;   //!
   TBranch        *b_NN_matrix36;   //!
   TBranch        *b_NN_matrix37;   //!
   TBranch        *b_NN_matrix38;   //!
   TBranch        *b_NN_matrix39;   //!
   TBranch        *b_NN_matrix40;   //!
   TBranch        *b_NN_matrix41;   //!
   TBranch        *b_NN_matrix42;   //!
   TBranch        *b_NN_matrix43;   //!
   TBranch        *b_NN_matrix44;   //!
   TBranch        *b_NN_matrix45;   //!
   TBranch        *b_NN_matrix46;   //!
   TBranch        *b_NN_matrix47;   //!
   TBranch        *b_NN_matrix48;   //!
   TBranch        *b_NN_pitches0;   //!
   TBranch        *b_NN_pitches1;   //!
   TBranch        *b_NN_pitches2;   //!
   TBranch        *b_NN_pitches3;   //!
   TBranch        *b_NN_pitches4;   //!
   TBranch        *b_NN_pitches5;   //!
   TBranch        *b_NN_pitches6;   //!
   TBranch        *b_NN_nparticles1;   //!
   TBranch        *b_NN_nparticles2;   //!
   TBranch        *b_NN_nparticles3;   //!
    TBranch        *b_NN_nparticles_excess;   //!
    TBranch        *b_NN_position_id_X_0;   //!
    TBranch        *b_NN_position_id_Y_0;   //!
    TBranch        *b_NN_position_id_X_1;   //!
    TBranch        *b_NN_position_id_Y_1;   //!
    TBranch        *b_NN_position_id_X_2;   //!
    TBranch        *b_NN_position_id_Y_2;   //!


    ValidationLoop(const char*, const char*, const char*);
	virtual ~ValidationLoop();
    virtual Int_t    Cut(Long64_t entry);
    virtual Int_t    GetEntry(Long64_t entry);
    virtual Long64_t LoadTree(Long64_t entry);
    virtual void     Init(TTree *tree);
    virtual void     Loop();
    virtual Bool_t   Notify();
    virtual void     Show(Long64_t entry = -1);

    std::vector<double> NN_input_vector();
    std::vector<double> collect_pitches_Y();
    std::vector<double> collect_positions();

    TFile *input_file;
    TFile *NN_file;
    TFile *output_file;

    TTrainedNetwork *NN_number;
    TTrainedNetwork *NN_pos1;
    TTrainedNetwork *NN_pos2;
    TTrainedNetwork *NN_pos3;
    TTrainedNetwork *NN_error1x;
    TTrainedNetwork *NN_error1y;
    TTrainedNetwork *NN_error2x;
    TTrainedNetwork *NN_error2y;
    TTrainedNetwork *NN_error3x;
    TTrainedNetwork *NN_error3y;

    TTree *new_tree;

    double Output_nparticles;
    double Output_nparticles1;
    double Output_nparticles2;
    double Output_nparticles3;

    std::vector<double> *Output_estimated_positions_raw;
    std::vector<double> *Output_estimated_positions;
    std::vector<double> *Output_true_positions_raw;
    std::vector<double> *Output_true_positions;
    std::vector<double> *Output_uncertainty_X;
    std::vector<double> *Output_uncertainty_Y;
};

#endif

#ifdef ValidationLoop_cxx
ValidationLoop::ValidationLoop(const char *input_path,
			       const char *NN_path,
			       const char *output_path) :
    fChain(0),
    input_file(TFile::Open(input_path)),
    NN_file(TFile::Open(NN_path)),
    output_file(TFile::Open(output_path, "RECREATE"))
{
    TTree *tree = static_cast<TTree*>(input_file->Get("NNinput"));
    Init(tree);

    NN_number = static_cast<TTrainedNetwork*>(NN_file->Get("NumberParticles"));
    NN_pos1 = static_cast<TTrainedNetwork*>(NN_file->Get("ImpactPoints1P"));
    NN_pos2 = static_cast<TTrainedNetwork*>(NN_file->Get("ImpactPoints2P"));
    NN_pos3 = static_cast<TTrainedNetwork*>(NN_file->Get("ImpactPoints3P"));
    NN_error1x = static_cast<TTrainedNetwork*>
	(NN_file->Get("ImpactPointErrorsX1"));
    NN_error1y = static_cast<TTrainedNetwork*>
	(NN_file->Get("ImpactPointErrorsY1"));
    NN_error2x = static_cast<TTrainedNetwork*>
	(NN_file->Get("ImpactPointErrorsX2"));
    NN_error2y = static_cast<TTrainedNetwork*>
	(NN_file->Get("ImpactPointErrorsY2"));
    NN_error3x = static_cast<TTrainedNetwork*>
	(NN_file->Get("ImpactPointErrorsX3"));
    NN_error3y = static_cast<TTrainedNetwork*>
	(NN_file->Get("ImpactPointErrorsY3"));

    if (!NN_number || !NN_pos1 || !NN_pos2 || !NN_pos3 ||
	!NN_error1x || !NN_error1y || !NN_error2x || !NN_error2y ||
	!NN_error3x || !NN_error3y)
	throw "Missing NN";

    Output_estimated_positions_raw = new std::vector<double>();
    Output_estimated_positions = new std::vector<double>();
    Output_true_positions_raw = new std::vector<double>();
    Output_true_positions = new std::vector<double>();
    Output_uncertainty_X = new std::vector<double>();
    Output_uncertainty_Y = new std::vector<double>();

    output_file->cd();

    new_tree = tree->CloneTree(0);
    new_tree->SetName("NNValidation");
    new_tree->Branch("Output_nparticles", &Output_nparticles);
    new_tree->Branch("Output_nparticles1", &Output_nparticles1);
    new_tree->Branch("Output_nparticles2", &Output_nparticles2);
    new_tree->Branch("Output_estimated_positions_raw",
		     &Output_estimated_positions_raw);
    new_tree->Branch("Output_estimated_positions", &Output_estimated_positions);
    new_tree->Branch("Output_true_positions_raw", &Output_true_positions_raw);
    new_tree->Branch("Output_true_positions", &Output_true_positions);
    new_tree->Branch("Output_uncertainty_X", &Output_uncertainty_X);
    new_tree->Branch("Output_uncertainty_Y", &Output_uncertainty_Y);
}

ValidationLoop::~ValidationLoop()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t ValidationLoop::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t ValidationLoop::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void ValidationLoop::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("RunNumber", &RunNumber, &b_RunNumber);
   fChain->SetBranchAddress("EventNumber", &EventNumber, &b_EventNumber);
   fChain->SetBranchAddress("ClusterNumber", &ClusterNumber, &b_ClusterNumber);
   fChain->SetBranchAddress("NN_sizeX", &NN_sizeX, &b_NN_sizeX);
   fChain->SetBranchAddress("NN_sizeY", &NN_sizeY, &b_NN_sizeY);
   fChain->SetBranchAddress("NN_localEtaPixelIndexWeightedPosition", &NN_localEtaPixelIndexWeightedPosition, &b_NN_localEtaPixelIndexWeightedPosition);
   fChain->SetBranchAddress("NN_localPhiPixelIndexWeightedPosition", &NN_localPhiPixelIndexWeightedPosition, &b_NN_localPhiPixelIndexWeightedPosition);
   fChain->SetBranchAddress("NN_layer", &NN_layer, &b_NN_layer);
   fChain->SetBranchAddress("NN_barrelEC", &NN_barrelEC, &b_NN_barrelEC);
   fChain->SetBranchAddress("NN_etaModule", &NN_etaModule, &b_NN_etaModule);
   fChain->SetBranchAddress("NN_phi", &NN_phi, &b_NN_phi);
   fChain->SetBranchAddress("NN_theta", &NN_theta, &b_NN_theta);
   fChain->SetBranchAddress("globalX", &globalX, &b_globalX);
   fChain->SetBranchAddress("globalY", &globalY, &b_globalY);
   fChain->SetBranchAddress("globalZ", &globalZ, &b_globalZ);
   fChain->SetBranchAddress("globalEta", &globalEta, &b_globalEta);
   fChain->SetBranchAddress("NN_matrix0", &NN_matrix0, &b_NN_matrix0);
   fChain->SetBranchAddress("NN_matrix1", &NN_matrix1, &b_NN_matrix1);
   fChain->SetBranchAddress("NN_matrix2", &NN_matrix2, &b_NN_matrix2);
   fChain->SetBranchAddress("NN_matrix3", &NN_matrix3, &b_NN_matrix3);
   fChain->SetBranchAddress("NN_matrix4", &NN_matrix4, &b_NN_matrix4);
   fChain->SetBranchAddress("NN_matrix5", &NN_matrix5, &b_NN_matrix5);
   fChain->SetBranchAddress("NN_matrix6", &NN_matrix6, &b_NN_matrix6);
   fChain->SetBranchAddress("NN_matrix7", &NN_matrix7, &b_NN_matrix7);
   fChain->SetBranchAddress("NN_matrix8", &NN_matrix8, &b_NN_matrix8);
   fChain->SetBranchAddress("NN_matrix9", &NN_matrix9, &b_NN_matrix9);
   fChain->SetBranchAddress("NN_matrix10", &NN_matrix10, &b_NN_matrix10);
   fChain->SetBranchAddress("NN_matrix11", &NN_matrix11, &b_NN_matrix11);
   fChain->SetBranchAddress("NN_matrix12", &NN_matrix12, &b_NN_matrix12);
   fChain->SetBranchAddress("NN_matrix13", &NN_matrix13, &b_NN_matrix13);
   fChain->SetBranchAddress("NN_matrix14", &NN_matrix14, &b_NN_matrix14);
   fChain->SetBranchAddress("NN_matrix15", &NN_matrix15, &b_NN_matrix15);
   fChain->SetBranchAddress("NN_matrix16", &NN_matrix16, &b_NN_matrix16);
   fChain->SetBranchAddress("NN_matrix17", &NN_matrix17, &b_NN_matrix17);
   fChain->SetBranchAddress("NN_matrix18", &NN_matrix18, &b_NN_matrix18);
   fChain->SetBranchAddress("NN_matrix19", &NN_matrix19, &b_NN_matrix19);
   fChain->SetBranchAddress("NN_matrix20", &NN_matrix20, &b_NN_matrix20);
   fChain->SetBranchAddress("NN_matrix21", &NN_matrix21, &b_NN_matrix21);
   fChain->SetBranchAddress("NN_matrix22", &NN_matrix22, &b_NN_matrix22);
   fChain->SetBranchAddress("NN_matrix23", &NN_matrix23, &b_NN_matrix23);
   fChain->SetBranchAddress("NN_matrix24", &NN_matrix24, &b_NN_matrix24);
   fChain->SetBranchAddress("NN_matrix25", &NN_matrix25, &b_NN_matrix25);
   fChain->SetBranchAddress("NN_matrix26", &NN_matrix26, &b_NN_matrix26);
   fChain->SetBranchAddress("NN_matrix27", &NN_matrix27, &b_NN_matrix27);
   fChain->SetBranchAddress("NN_matrix28", &NN_matrix28, &b_NN_matrix28);
   fChain->SetBranchAddress("NN_matrix29", &NN_matrix29, &b_NN_matrix29);
   fChain->SetBranchAddress("NN_matrix30", &NN_matrix30, &b_NN_matrix30);
   fChain->SetBranchAddress("NN_matrix31", &NN_matrix31, &b_NN_matrix31);
   fChain->SetBranchAddress("NN_matrix32", &NN_matrix32, &b_NN_matrix32);
   fChain->SetBranchAddress("NN_matrix33", &NN_matrix33, &b_NN_matrix33);
   fChain->SetBranchAddress("NN_matrix34", &NN_matrix34, &b_NN_matrix34);
   fChain->SetBranchAddress("NN_matrix35", &NN_matrix35, &b_NN_matrix35);
   fChain->SetBranchAddress("NN_matrix36", &NN_matrix36, &b_NN_matrix36);
   fChain->SetBranchAddress("NN_matrix37", &NN_matrix37, &b_NN_matrix37);
   fChain->SetBranchAddress("NN_matrix38", &NN_matrix38, &b_NN_matrix38);
   fChain->SetBranchAddress("NN_matrix39", &NN_matrix39, &b_NN_matrix39);
   fChain->SetBranchAddress("NN_matrix40", &NN_matrix40, &b_NN_matrix40);
   fChain->SetBranchAddress("NN_matrix41", &NN_matrix41, &b_NN_matrix41);
   fChain->SetBranchAddress("NN_matrix42", &NN_matrix42, &b_NN_matrix42);
   fChain->SetBranchAddress("NN_matrix43", &NN_matrix43, &b_NN_matrix43);
   fChain->SetBranchAddress("NN_matrix44", &NN_matrix44, &b_NN_matrix44);
   fChain->SetBranchAddress("NN_matrix45", &NN_matrix45, &b_NN_matrix45);
   fChain->SetBranchAddress("NN_matrix46", &NN_matrix46, &b_NN_matrix46);
   fChain->SetBranchAddress("NN_matrix47", &NN_matrix47, &b_NN_matrix47);
   fChain->SetBranchAddress("NN_matrix48", &NN_matrix48, &b_NN_matrix48);
   fChain->SetBranchAddress("NN_pitches0", &NN_pitches0, &b_NN_pitches0);
   fChain->SetBranchAddress("NN_pitches1", &NN_pitches1, &b_NN_pitches1);
   fChain->SetBranchAddress("NN_pitches2", &NN_pitches2, &b_NN_pitches2);
   fChain->SetBranchAddress("NN_pitches3", &NN_pitches3, &b_NN_pitches3);
   fChain->SetBranchAddress("NN_pitches4", &NN_pitches4, &b_NN_pitches4);
   fChain->SetBranchAddress("NN_pitches5", &NN_pitches5, &b_NN_pitches5);
   fChain->SetBranchAddress("NN_pitches6", &NN_pitches6, &b_NN_pitches6);
   fChain->SetBranchAddress("NN_nparticles1", &NN_nparticles1, &b_NN_nparticles1);
   fChain->SetBranchAddress("NN_nparticles2", &NN_nparticles2, &b_NN_nparticles2);
   fChain->SetBranchAddress("NN_nparticles3", &NN_nparticles3, &b_NN_nparticles3);
   fChain->SetBranchAddress("NN_nparticles_excess", &NN_nparticles_excess, &b_NN_nparticles_excess);
   fChain->SetBranchAddress("NN_position_id_X_0", &NN_position_id_X_0, &b_NN_position_id_X_0);
   fChain->SetBranchAddress("NN_position_id_Y_0", &NN_position_id_Y_0, &b_NN_position_id_Y_0);
   fChain->SetBranchAddress("NN_position_id_X_1", &NN_position_id_X_1, &b_NN_position_id_X_1);
   fChain->SetBranchAddress("NN_position_id_Y_1", &NN_position_id_Y_1, &b_NN_position_id_Y_1);
   fChain->SetBranchAddress("NN_position_id_X_2", &NN_position_id_X_2, &b_NN_position_id_X_2);
   fChain->SetBranchAddress("NN_position_id_Y_2", &NN_position_id_Y_2, &b_NN_position_id_Y_2);

   Notify();
}

Bool_t ValidationLoop::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void ValidationLoop::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t ValidationLoop::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef ValidationLoop_cxx
