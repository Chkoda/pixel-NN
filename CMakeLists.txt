atlas_subdir(PixelNN)

find_package(ROOT COMPONENTS MathCore RIO Core Tree Hist Matrix)

atlas_depends_on_subdirs(
  PUBLIC
  Control/AthAnalysisBaseComps
  Tracking/TrkNeuralNetworkUtils
  xAOD/xAODTracking

  PRIVATE
  Control/AthContainers
  Control/xAODRootAccess
  xAOD/xAODEventInfo
  GaudiKernel
  )

atlas_add_root_dictionary(PixelNNLib
  PixelNNLibRootDict
  ROOT_HEADERS
  pixel-NN/TTrainedNetworkNormalization.h
  Root/LinkDef.h
  )

atlas_add_library(PixelNNLib
  Root/*.cxx
  ${PixelNNLibRootDict}

  PUBLIC_HEADERS
  pixel-NN

  INCLUDE_DIRS
  ${ROOT_INCLUDE_DIRS}

  LINK_LIBRARIES
  ${ROOT_LIBRARIES}
  AthAnalysisBaseCompsLib
  TrkNeuralNetworkUtilsLib
  xAODTracking

  PRIVATE_LINK_LIBRARIES
  AthContainers
  xAODEventInfo
  xAODRootAccess
  GaudiKernel
  )

atlas_add_component(PixelNN
  Root/components/*.cxx
  LINK_LIBRARIES
  GaudiKernel
  PixelNNLib
  )

atlas_install_python_modules(python/*)
atlas_install_scripts(scripts/*)
atlas_install_joboptions(joboptions/*)
atlas_install_runtime(share/*)
