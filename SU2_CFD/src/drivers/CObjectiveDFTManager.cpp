/*!
 * \brief Implementation of the objective DFT manager.
 */

#include "../../include/drivers/CObjectiveDFTManager.hpp"

#include "../../../Common/include/CConfig.hpp"
#include "../../include/output/tools/CWindowingTools.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

CObjectiveDFTManager::CObjectiveDFTManager(const CConfig* config)
  : WindowStartIter(config->GetStartWindowIteration()),
    WindowSize(config->GetIter_Avg_Objective()),
    WindowEndIter(WindowSize == 0 ? WindowStartIter : WindowStartIter + WindowSize - 1),
    Harmonic(config->GetObjectiveDFTHarmonic()),
    WindowKind(config->GetKindWindow()),
    ObjHistory(WindowSize, 0.0),
    Alpha(WindowSize, 0.0),
    Beta(WindowSize, 0.0),
    SamplesWritten(WindowSize, false),
    ReHatJ(0.0),
    ImHatJ(0.0),
    Amplitude(0.0),
    Phase(0.0),
    Finalized(false),
    SamplesCollected(0),
    OutputFilename(config->GetObjectiveDFT_FileName()) {}

bool CObjectiveDFTManager::IsInWindow(unsigned long iter) const {
  if (WindowSize == 0) return false;
  return (iter >= WindowStartIter) && (iter <= WindowEndIter);
}

void CObjectiveDFTManager::AddSample(unsigned long iter, su2double value) {
  if (!IsInWindow(iter) || WindowSize == 0) return;

  const auto localIdx = static_cast<size_t>(iter - WindowStartIter);
  if (localIdx >= ObjHistory.size()) return;

  ObjHistory[localIdx] = SU2_TYPE::GetValue(value);
  if (!SamplesWritten[localIdx]) {
    SamplesWritten[localIdx] = true;
    SamplesCollected++;
  }
}

void CObjectiveDFTManager::FinalizeDFT() {
  if (Finalized || !HasCompleteWindow()) return;
  ComputeDFTAndAlpha();
  Finalized = true;
}

su2double CObjectiveDFTManager::GetKernel(OBJFUNC_TEMPORAL_MODE mode, unsigned long iter) const {
  if (!Finalized || !IsInWindow(iter) || Alpha.empty()) return 0.0;
  const auto localIdx = static_cast<size_t>(iter - WindowStartIter);
  if (localIdx >= Alpha.size()) return 0.0;

  switch (mode) {
    case OBJFUNC_TEMPORAL_MODE::DFT_AMPLITUDE:
      return Alpha[localIdx];
    case OBJFUNC_TEMPORAL_MODE::DFT_PHASE:
      if (localIdx < Beta.size()) return Beta[localIdx];
      return 0.0;
    default:
      return 0.0;
  }
}

void CObjectiveDFTManager::ComputeDFTAndAlpha() {
  if (WindowSize == 0) return;

  const auto nSamples = static_cast<double>(WindowSize);
  const auto windowWidthMinusOne = (WindowSize > 0) ? WindowSize - 1 : 0;
  double weightSum = 0.0;

  ReHatJ = 0.0;
  ImHatJ = 0.0;

  for (unsigned long k = 0; k < WindowSize; ++k) {
    const double value = ObjHistory[k];
    const double weight = SU2_TYPE::GetValue(CWindowingTools::GetWndWeight(WindowKind, k, windowWidthMinusOne));
    const double phase = (WindowSize > 0)
                                ? -2.0 * SU2_TYPE::GetValue(PI_NUMBER) * static_cast<double>(Harmonic) *
                                      static_cast<double>(k) / nSamples
                                : 0.0;
    const double c = cos(phase);
    const double s = sin(phase);
    ReHatJ += weight * value * c;
    ImHatJ += weight * value * s;
    weightSum += weight;
  }

  if (weightSum <= std::numeric_limits<double>::min()) {
    Amplitude = 0.0;
    Phase = 0.0;
    Alpha.assign(WindowSize, 0.0);
    Beta.assign(WindowSize, 0.0);
    return;
  }

  const double denom2 = ReHatJ * ReHatJ + ImHatJ * ImHatJ;
  const double rawMagnitude = sqrt(denom2);

  const double normalization = 2.0 / weightSum;
  Amplitude = normalization * rawMagnitude;

  Alpha.assign(WindowSize, 0.0);
  Beta.assign(WindowSize, 0.0);

  if (rawMagnitude <= std::numeric_limits<double>::min() ||
      denom2 <= std::numeric_limits<double>::min()) {
    Phase = 0.0;
    return;
  }

  Phase = atan2(ImHatJ, ReHatJ);

  for (unsigned long k = 0; k < WindowSize; ++k) {
    const double weight = SU2_TYPE::GetValue(CWindowingTools::GetWndWeight(WindowKind, k, windowWidthMinusOne));
    const double phase = -2.0 * SU2_TYPE::GetValue(PI_NUMBER) * static_cast<double>(Harmonic) *
                         static_cast<double>(k) / nSamples;
    const double c = cos(phase);
    const double s = sin(phase);
    const double numeratorAmp = ReHatJ * c + ImHatJ * s;
    const double numeratorPhase = ReHatJ * s - ImHatJ * c;
    Alpha[k] = normalization * weight * numeratorAmp / rawMagnitude;
    Beta[k] = weight * numeratorPhase / denom2;
  }
}

void CObjectiveDFTManager::WriteAmplitudeFile() const {
  if (!Finalized || OutputFilename.empty()) return;

  std::ofstream file(OutputFilename.c_str(), std::ios::out);
  if (!file.is_open()) return;

  file << "# Window start iteration: " << WindowStartIter << "\n";
  file << "# Window size: " << WindowSize << "\n";
  file << "# Harmonic index: " << Harmonic << "\n";
  file << std::scientific << std::setprecision(12);
  file << "DFT_AMPLITUDE " << Amplitude << std::endl;
  file << "DFT_PHASE " << Phase << std::endl;
  if (!Alpha.empty()) {
    for (unsigned long k = 0; k < WindowSize; ++k) {
      const auto iter = WindowStartIter + k;
      file << "ALPHA " << iter << " " << Alpha[k] << std::endl;
    }
  }
  if (!Beta.empty()) {
    for (unsigned long k = 0; k < WindowSize; ++k) {
      const auto iter = WindowStartIter + k;
      file << "BETA " << iter << " " << Beta[k] << std::endl;
    }
  }
}

bool CObjectiveDFTManager::LoadAlphaFromFile() {
  if (OutputFilename.empty() || WindowSize == 0) return false;

  std::ifstream file(OutputFilename.c_str(), std::ios::in);
  if (!file.is_open()) return false;

  std::string line;
  bool alphaLoaded = false;
  bool amplitudeLoaded = false;
  bool betaLoaded = false;
  bool phaseLoaded = false;
  double amplitudeValue = 0.0;
  double phaseValue = 0.0;

  std::vector<su2double> loadedAlpha(WindowSize, 0.0);
  std::vector<su2double> loadedBeta(WindowSize, 0.0);
  std::vector<bool> loadedFlags(WindowSize, false);

  while (std::getline(file, line)) {
    std::string trimmed = line;
    trimmed.erase(0, trimmed.find_first_not_of(" \t\r\n"));
    if (trimmed.empty() || trimmed[0] == '#') continue;

    std::istringstream iss(trimmed);
    std::string tag;
    iss >> tag;
    if (tag == "DFT_AMPLITUDE") {
      iss >> amplitudeValue;
      amplitudeLoaded = true;
    } else if (tag == "DFT_PHASE") {
      iss >> phaseValue;
      phaseLoaded = true;
    } else if (tag == "ALPHA") {
      unsigned long iter;
      double value;
      iss >> iter >> value;
      if (iter >= WindowStartIter && iter < WindowStartIter + WindowSize) {
        const auto idx = static_cast<size_t>(iter - WindowStartIter);
        loadedAlpha[idx] = value;
        loadedFlags[idx] = true;
        alphaLoaded = true;
      }
    } else if (tag == "BETA") {
      unsigned long iter;
      double value;
      iss >> iter >> value;
      if (iter >= WindowStartIter && iter < WindowStartIter + WindowSize) {
        const auto idx = static_cast<size_t>(iter - WindowStartIter);
        loadedBeta[idx] = value;
        loadedFlags[idx] = true;
        betaLoaded = true;
      }
    }
  }

  if (!alphaLoaded && !betaLoaded) return false;

  Alpha = loadedAlpha;
  Beta = loadedBeta;
  SamplesWritten.assign(WindowSize, true);
  SamplesCollected = WindowSize;
  Finalized = true;
  if (amplitudeLoaded) {
    Amplitude = amplitudeValue;
  }
  if (phaseLoaded) {
    Phase = phaseValue;
  }
  return true;
}
