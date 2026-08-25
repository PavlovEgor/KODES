#ifndef KODES_SETTINGS
#define KODES_SETTINGS

#pragma once

#include <string>

#include "basicTypes.cuh"
#include "Config.cuh"
#include "LaunchConfig.cuh"
#include "IntegratorControls.cuh"
#include "balancerTable.cuh"

// One JSON file holding everything a run is free to choose, so that changing
// the method, the balancer or a tolerance does not mean recompiling.
//
// See examples/seulex5/seulex5.json for a complete file. Every
// entry has a default, so a file naming nothing but the method is valid.
//
//   {
//       "method":   "seulex",       // methodTable:   seulex | euler
//       "balancer": "stiffness",    // balancerTable: temperature | rhsNorm
//                                   //                stiffness | none
//       "device": {
//           "share":           "best",   // best | half, see deviceShares
//           "threadsPerBlock": 256,
//
//           // sizing the run by hand instead - both are needed together and
//           // they override "share"
//           "concurrentSystems": 8192,
//           "batchSize":         1000000
//       },
//
//       "controls": {
//           "absTol":   1e-10,      // absolute tolerance
//           "relTol":   1e-1,       // relative tolerance
//           "maxSteps": 10000,      // steps one system may take per solve
//           "Treact":   0.0,        // systems at or below this are left alone
//
//           "safeScale":     0.9,   // the step size controller, only used by
//           "alphaIncrease": 0.2,   // a method that takes trial steps
//           "alphaDecrease": 0.25,
//           "minScale":      0.2,
//           "maxScale":      10.0
//       },
//
//       "run": {
//           "ensembleSize":    24576,   // systems to integrate
//           "endTime":         10.0,    // how far to integrate them
//           "initialTimeStep": 10.0     // first trial step of each thread slot
//       }
//   }
//
// This is host-only C++ around rapidjson, which is why it is a source list of
// its own in cmake/kodes.cmake: a caller that gets its settings from somewhere
// else - the OpenFOAM chemistry model reads an OpenFOAM dictionary - passes the
// same names and numbers to the same functions and never links this.

namespace kodes
{

class Settings
{
    Config config_;

    std::string path_;

    std::string method_;
    std::string balancer_;

    static Config openOrExit(const std::string& path);

public:

    // Reads and validates `path`. Fails with the list of known names if the
    // method or the balancer named is not one of them, so a typo is caught
    // before anything has been allocated on the device.
    explicit Settings(const std::string& path);

    const std::string& path() const { return path_; }

    // The name of the entry in methodTable
    const std::string& method() const { return method_; }

    // The name of the entry in balancerTable, or "none"
    const std::string& balancer() const { return balancer_; }

    bool balanceBatches() const { return balancer_ != KODES_NO_BALANCER; }

    // The request planLaunch resolves against the device
    LaunchConfig launchRequest() const;

    IntegratorControls controls() const;

    label ensembleSize() const;

    scalar endTime() const;

    scalar initialTimeStep() const;

    // Everything the run was told, in the order the file lists it
    void print() const;
};

}

#endif
