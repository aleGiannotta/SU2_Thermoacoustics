/*!
 * \file CObjectiveDFTManager.hpp
 * \brief Helper for collecting windowed objective histories and computing DFT-based sensitivities.
 * \author Alessandro Giannotta
 */

#pragma once

#include <vector>
#include <string>
#include <cstddef>
#include "../definition_structure.hpp"
#include "../../../Common/include/option_structure.hpp"

class CConfig;

/*!
 * \brief Manages the storage of a windowed objective history and computes the derivatives
 *        of a single-harmonic DFT magnitude and phase with respect to the instantaneous samples.
 */
class CObjectiveDFTManager {
public:
  explicit CObjectiveDFTManager(const CConfig* config);

  /*! \brief Store an instantaneous objective sample if it belongs to the configured window. */
  void AddSample(unsigned long iter, su2double value);

  /*! \brief Trigger the DFT evaluation and derivative computation once the window is filled. */
  void FinalizeDFT();

  bool IsInWindow(unsigned long iter) const;
  bool IsFinalized() const { return Finalized; }
  bool HasCompleteWindow() const { return SamplesCollected == WindowSize && WindowSize > 0; }

  unsigned long GetWindowStart() const { return WindowStartIter; }
  unsigned long GetWindowEnd() const { return WindowEndIter; }
  unsigned long GetWindowSize() const { return WindowSize; }

  /*! \brief Retrieve the derivative kernel associated with the provided iteration
   *         for the selected temporal objective (amplitude or phase).
   */
  su2double GetKernel(OBJFUNC_TEMPORAL_MODE mode, unsigned long iter) const;

  su2double GetAmplitude() const { return Amplitude; }
  su2double GetPhase() const { return Phase; }
  void WriteAmplitudeFile() const;
  bool LoadAlphaFromFile();

private:
  const unsigned long WindowStartIter;
  const unsigned long WindowSize;
  const unsigned long WindowEndIter;
  const unsigned long Harmonic;
  const WINDOW_FUNCTION WindowKind;

  std::vector<double> ObjHistory;
  std::vector<su2double> Alpha;
  std::vector<su2double> Beta;
  std::vector<bool> SamplesWritten;

  double ReHatJ;
  double ImHatJ;
  double Amplitude;
  double Phase;
  bool Finalized;
  unsigned long SamplesCollected;
  std::string OutputFilename;

  void ComputeDFTAndAlpha();
};
