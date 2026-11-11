#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"

#include "library_api.h"

// A benchmark of a prebuilt KODES on one case of tests/cases: the states of
// <case>/<mechanism>.csv repeated to an ensemble, integrated over the case's
// deltaT with its tolerances, as many times as asked, each time from the same
// states. Prints the plan and the wall clock of every solve(), and writes the
// final states when asked to.
//
//     benchmark [options]           2D_flat_burner, libkodes_grimech.so from
//                                   the library path, seulex, temperature
//
// Through library_api.h, as kodes4foam calls KODES: a solve() is a whole step
// of the chemistry, host->device and back included. And by name, as the state
// files are written: the library says which specie is which component.

namespace
{

const char* const kUsage =
"usage: benchmark [options]\n"
"  --case CASE       a case of tests/cases, or the path of one\n"
"                                                     (default: 2D_flat_burner)\n"
"  --mechanism NAME  the case's <NAME>.csv            (default: the case's first)\n"
"  --library PATH    libkodes_<mechanism>.so to open  (default: from the library path)\n"
"  --method NAME     seulex | euler | rosenbrock23 | rosenbrock34 | rodas23 |\n"
"                    rodas34 | bdf                    (default: seulex)\n"
"  --balancer NAME   temperature | rhsNorm | stiffness | none\n"
"                                                     (default: temperature)\n"
"  --ensemble N      systems per solve(), the states repeated (default: 131072)\n"
"  --order ORDER     shuffled | as-is: shuffled spreads the copies over the\n"
"                    ensemble with a fixed seed; as-is keeps the file's order,\n"
"                    sorted by temperature for 2D_flat_burner (default: shuffled)\n"
"  --repeats N       timed solves after one warm-up   (default: 5)\n"
"  --deltaT S        instead of the case's\n"
"  --first-step S    the trial step every solve starts from (default: deltaT)\n"
"  --share NAME      best | half                      (default: best)\n"
"  --threads N       per block                        (default: 256)\n"
"  --Treact K        systems at or below are left alone (default: 0)\n"
"  --output FILE     the final state of every state of the file, as a state file\n";

void fail(const std::string& message)
{
    fprintf(stderr, "benchmark: %s\n", message.c_str());
    exit(EXIT_FAILURE);
}

struct Options
{
    std::string caseDir = KODES_TESTS_DIR "/cases/2D_flat_burner";
    std::string mechanism;
    std::string library;
    std::string method = "seulex";
    std::string balancer = "temperature";
    int ensemble = 131072;
    std::string order = "shuffled";
    int repeats = 5;
    double deltaT = 0;
    double firstStep = 0;
    std::string share = "best";
    int threads = 256;
    double Treact = 0;
    std::string output;
};

Options parseOptions(int argc, char** argv)
{
    Options options;

    for (int i = 1; i < argc; ++i)
    {
        const std::string name = argv[i];

        if (name == "-h" || name == "--help")
        {
            printf("%s", kUsage);
            exit(EXIT_SUCCESS);
        }

        if (i + 1 >= argc)
        {
            fail("no value for " + name + "\n" + kUsage);
        }

        const char* value = argv[++i];

        if (name == "--case")
        {
            // a bare name is one of tests/cases
            options.caseDir = strchr(value, '/') ? value : KODES_TESTS_DIR "/cases/" + std::string(value);
        }
        else if (name == "--mechanism") options.mechanism = value;
        else if (name == "--library") options.library = value;
        else if (name == "--method") options.method = value;
        else if (name == "--balancer") options.balancer = value;
        else if (name == "--ensemble") options.ensemble = atoi(value);
        else if (name == "--order") options.order = value;
        else if (name == "--repeats") options.repeats = atoi(value);
        else if (name == "--deltaT") options.deltaT = atof(value);
        else if (name == "--first-step") options.firstStep = atof(value);
        else if (name == "--share") options.share = value;
        else if (name == "--threads") options.threads = atoi(value);
        else if (name == "--Treact") options.Treact = atof(value);
        else if (name == "--output") options.output = value;
        else fail("unknown option " + name + "\n" + kUsage);
    }

    if (options.ensemble <= 0 || options.repeats <= 0 || options.threads <= 0)
    {
        fail("--ensemble, --repeats and --threads must be positive");
    }

    if (options.order != "shuffled" && options.order != "as-is")
    {
        fail("--order is shuffled or as-is, not " + options.order);
    }

    return options;
}

// What tests/cases/<case>/case.json says about running it
struct Case
{
    std::string name;
    std::vector<std::string> mechanisms;
    double deltaT;
    double absTol;
    double relTol;
};

Case readCase(const std::string& dir)
{
    const std::string path = dir + "/case.json";

    std::ifstream file(path);

    if (!file)
    {
        fail("cannot open " + path);
    }

    rapidjson::IStreamWrapper stream(file);
    rapidjson::Document json;
    json.ParseStream(stream);

    if (json.HasParseError() || !json.IsObject())
    {
        fail(path + " is not a JSON object");
    }

    const auto number = [&](const rapidjson::Value& object, const char* key)
    {
        if (!object.IsObject() || !object.HasMember(key) || !object[key].IsNumber())
        {
            fail(path + " has no number " + key);
        }
        return object[key].GetDouble();
    };

    Case c;

    c.name = dir.substr(dir.find_last_of('/') + 1);
    c.deltaT = number(json, "deltaT");

    if (!json.HasMember("controls"))
    {
        fail(path + " has no controls");
    }

    c.absTol = number(json["controls"], "absTol");
    c.relTol = number(json["controls"], "relTol");

    if (json.HasMember("mechanisms") && json["mechanisms"].IsArray())
    {
        for (const auto& m : json["mechanisms"].GetArray())
        {
            c.mechanisms.push_back(m.GetString());
        }
    }

    return c;
}

// A state file, see tests/README.md: label,T,p, then mass fractions by name
struct States
{
    std::string comment;
    std::vector<std::string> species;
    std::vector<std::string> labels;
    std::vector<double> T;
    std::vector<double> p;
    std::vector<std::vector<double>> Y;   // Y[state][column of species]
};

std::vector<std::string> split(const std::string& line)
{
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;

    while (std::getline(stream, field, ','))
    {
        fields.push_back(field);
    }
    return fields;
}

States readStates(const std::string& path)
{
    std::ifstream file(path);

    if (!file)
    {
        fail("cannot open " + path);
    }

    States states;
    std::string line;
    bool header = true;
    int lineNumber = 0;

    while (std::getline(file, line))
    {
        ++lineNumber;

        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }

        if (line.empty())
        {
            continue;
        }

        if (line[0] == '#')
        {
            states.comment += line.substr(1) + "\n";
            continue;
        }

        const std::vector<std::string> fields = split(line);

        if (header)
        {
            if (fields.size() < 3 || fields[0] != "label" || fields[1] != "T" || fields[2] != "p")
            {
                fail(path + ": the header must start with label,T,p");
            }
            states.species.assign(fields.begin() + 3, fields.end());
            header = false;
            continue;
        }

        if (fields.size() != states.species.size() + 3)
        {
            fail(path + ":" + std::to_string(lineNumber) + ": " + std::to_string(fields.size()) + " fields, the header has " + std::to_string(states.species.size() + 3));
        }

        states.labels.push_back(fields[0]);
        states.T.push_back(std::stod(fields[1]));
        states.p.push_back(std::stod(fields[2]));

        std::vector<double> Y;
        for (size_t i = 3; i < fields.size(); ++i)
        {
            Y.push_back(std::stod(fields[i]));
        }
        states.Y.push_back(Y);
    }

    if (states.labels.empty())
    {
        fail(path + " holds no states");
    }

    return states;
}

void* findSymbol(void* handle, const char* name)
{
    void* symbol = dlsym(handle, name);

    if (!symbol)
    {
        fail(std::string("not a KODES library, no symbol ") + name);
    }
    return symbol;
}

const kodes::Library& openLibrary(const std::string& path)
{
    // RTLD_LOCAL: nothing of one KODES library is visible to the next. It
    // stays open until the program ends, see examples/library.
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);

    if (!handle)
    {
        fail(dlerror());
    }

    const int version = reinterpret_cast<kodes::LibraryApiVersionFunction>
    (
        findSymbol(handle, kodes::kLibraryApiVersionSymbol)
    )();

    if (version != kodes::kLibraryApiVersion)
    {
        fail(path + " has interface version " + std::to_string(version) + ", this program was built for " + std::to_string(kodes::kLibraryApiVersion));
    }

    return *reinterpret_cast<kodes::LibraryFunction>
    (
        findSymbol(handle, kodes::kLibrarySymbol)
    )();
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    return n % 2 ? values[n/2] : 0.5*(values[n/2 - 1] + values[n/2]);
}

}

int main(int argc, char** argv)
{
    const Options options = parseOptions(argc, argv);

    const Case c = readCase(options.caseDir);

    const std::string mechanism =
        !options.mechanism.empty() ? options.mechanism
      : !c.mechanisms.empty() ? c.mechanisms[0]
      : "";

    if (mechanism.empty())
    {
        fail(options.caseDir + "/case.json names no mechanism, give --mechanism");
    }

    const std::string statesPath = options.caseDir + "/" + mechanism + ".csv";
    const States states = readStates(statesPath);
    const int numOfStates = int(states.labels.size());

    const kodes::Library& library = openLibrary
    (
        options.library.empty() ? "libkodes_" + mechanism + ".so" : options.library
    );

    const int systemSize = library.systemSize();
    const int nSpecie = library.nSpecie();

    // T and every mass fraction but the last specie's
    if (systemSize != nSpecie)
    {
        fail("a library whose system has " + std::to_string(systemSize) + " components for " + std::to_string(nSpecie) + " species, a pyJac one expected");
    }

    if (library.parameterSize() != 1)
    {
        fail("a library whose system has " + std::to_string(library.parameterSize()) + " parameters, the pressure only expected");
    }

    // Where each column of the file goes: component 1 + its index for a
    // specie of the state, -1 for the specie the state leaves out
    std::vector<int> component(states.species.size());

    for (size_t column = 0; column < states.species.size(); ++column)
    {
        int index = -1;

        for (int i = 0; i < nSpecie; ++i)
        {
            if (states.species[column] == library.specie(i))
            {
                index = i;
            }
        }

        if (index < 0)
        {
            fail(statesPath + ": specie " + states.species[column] + " is not in " + library.mechanism());
        }

        component[column] = index < nSpecie - 1 ? index + 1 : -1;
    }

    // The states as the library takes them, component-major, state s of the
    // file once each
    std::vector<std::vector<double>> fileState(systemSize, std::vector<double>(numOfStates, 0));
    double worstSum = 0;

    for (int s = 0; s < numOfStates; ++s)
    {
        fileState[0][s] = states.T[s];

        double sum = 0;
        for (size_t column = 0; column < states.species.size(); ++column)
        {
            sum += states.Y[s][column];

            if (component[column] > 0)
            {
                fileState[component[column]][s] = states.Y[s][column];
            }
        }
        worstSum = std::max(worstSum, std::fabs(sum - 1));
    }

    // Which state of the file each system of the ensemble starts from: every
    // one as often as the next, and spread out unless asked not to be
    std::vector<int> source(options.ensemble);

    for (int system = 0; system < options.ensemble; ++system)
    {
        source[system] = system % numOfStates;
    }

    if (options.order == "shuffled")
    {
        std::shuffle(source.begin(), source.end(), std::mt19937(20261111));
    }

    std::vector<std::vector<double>> initial(systemSize, std::vector<double>(options.ensemble));
    std::vector<double> pressure(options.ensemble);

    for (int system = 0; system < options.ensemble; ++system)
    {
        for (int i = 0; i < systemSize; ++i)
        {
            initial[i][system] = fileState[i][source[system]];
        }
        pressure[system] = states.p[source[system]];
    }

    const double deltaT = options.deltaT > 0 ? options.deltaT : c.deltaT;
    const double firstStep = options.firstStep > 0 ? options.firstStep : deltaT;

    kodes::SolverSettings settings;

    settings.method = options.method.c_str();
    settings.balancer = options.balancer.c_str();
    settings.deviceShare = options.share.c_str();
    settings.threadsPerBlock = options.threads;
    settings.absTol = c.absTol;
    settings.relTol = c.relTol;
    settings.Treact = options.Treact;

    printf
    (
        "\n%s, %s: %d states of %s, sum of Y within %.1e of 1\n",
        c.name.c_str(), mechanism.c_str(), numOfStates, statesPath.c_str(), worstSum
    );
    printf
    (
        "KODES %s, %s: %d species, %d reactions\n",
        library.version(), library.mechanism(), nSpecie, library.nReaction()
    );
    printf
    (
        "%d systems, %s, deltaT %g s, first step %g s, absTol %g, relTol %g, Treact %g\n",
        options.ensemble, options.order.c_str(), deltaT, firstStep,
        settings.absTol, settings.relTol, settings.Treact
    );

    using Clock = std::chrono::steady_clock;

    const auto seconds = [](const Clock::time_point a, const Clock::time_point b)
    {
        return std::chrono::duration<double>(b - a).count();
    };

    const Clock::time_point setupStart = Clock::now();

    const std::unique_ptr<kodes::Solver> solver(library.newSolver(options.ensemble, settings));

    const double setupSeconds = seconds(setupStart, Clock::now());

    printf
    (
        "%s, balancer %s: %d blocks x %d threads, %d systems per batch, %d batch(es);"
        " set up in %.3f s\n\n",
        solver->method(), solver->balancer(), solver->blocks(), solver->threads(),
        solver->batchSize(), solver->numOfBatches(), setupSeconds
    );

    std::vector<std::vector<double>> state(systemSize);
    std::vector<double> parameter;

    std::vector<double*> statePointers(systemSize);
    double* parameterPointers[1];

    std::vector<double> times;
    std::vector<double> firstFinalT;

    for (int run = 0; run <= options.repeats; ++run)
    {
        // every solve from the same states and the same trial step
        for (int i = 0; i < systemSize; ++i)
        {
            state[i] = initial[i];
            statePointers[i] = state[i].data();
        }
        parameter = pressure;
        parameterPointers[0] = parameter.data();

        solver->setDeltaT(firstStep);

        const Clock::time_point start = Clock::now();

        solver->solve(deltaT, statePointers.data(), parameterPointers);

        const double t = seconds(start, Clock::now());

        printf
        (
            "%s %d: %.4f s, %.0f systems/s\n",
            run == 0 ? "warm-up" : "solve  ", run, t, options.ensemble/t
        );

        if (run > 0)
        {
            times.push_back(t);
        }

        if (run == 1)
        {
            firstFinalT = state[0];
        }
    }

    // What the last solve ended with: nothing that is not a number, the range
    // of temperature, and whether the repeats agree with each other
    int nonFinite = 0;
    double TMin = INFINITY, TMax = -INFINITY, dTMax = 0, repeatDiff = 0;

    for (int system = 0; system < options.ensemble; ++system)
    {
        for (int i = 0; i < systemSize; ++i)
        {
            nonFinite += !std::isfinite(state[i][system]);
        }

        const double T = state[0][system];

        TMin = std::min(TMin, T);
        TMax = std::max(TMax, T);
        dTMax = std::max(dTMax, std::fabs(T - initial[0][system]));
        repeatDiff = std::max(repeatDiff, std::fabs(T - firstFinalT[system]));
    }

    const double best = *std::min_element(times.begin(), times.end());
    const double typical = median(times);

    printf
    (
        "\n%s, %s, balancer %s, %d systems: median %.4f s, best %.4f s, %.0f systems/s at the median\n",
        c.name.c_str(), solver->method(), solver->balancer(), options.ensemble, typical, best,
        options.ensemble/typical
    );
    printf
    (
        "T after deltaT: %.1f..%.1f K, largest change %.2f K, repeats agree to %.1e K, %d non-finite values\n",
        TMin, TMax, dTMax, repeatDiff, nonFinite
    );

    if (!options.output.empty())
    {
        // The first system that started from each state of the file
        std::vector<int> systemOf(numOfStates, -1);

        for (int system = 0; system < options.ensemble; ++system)
        {
            if (systemOf[source[system]] < 0)
            {
                systemOf[source[system]] = system;
            }
        }

        FILE* out = fopen(options.output.c_str(), "w");

        if (!out)
        {
            fail("cannot write " + options.output);
        }

        fprintf
        (
            out,
            "# %s, %s after deltaT = %g s with KODES %s, %s, balancer %s, absTol %g, relTol %g\n",
            c.name.c_str(), mechanism.c_str(), deltaT, library.version(),
            solver->method(), solver->balancer(), settings.absTol, settings.relTol
        );

        fprintf(out, "label,T,p");
        for (int i = 0; i < nSpecie; ++i)
        {
            fprintf(out, ",%s", library.specie(i));
        }
        fprintf(out, "\n");

        int written = 0;

        for (int s = 0; s < numOfStates; ++s)
        {
            const int system = systemOf[s];

            // fewer systems than states: the rest were not integrated
            if (system < 0)
            {
                continue;
            }

            fprintf(out, "%s,%.17g,%.17g", states.labels[s].c_str(), state[0][system], pressure[system]);

            // the specie the state leaves out is what the others leave of 1
            double rest = 1;
            for (int i = 1; i < systemSize; ++i)
            {
                fprintf(out, ",%.17g", state[i][system]);
                rest -= state[i][system];
            }
            fprintf(out, ",%.17g\n", rest);

            ++written;
        }

        fclose(out);

        printf("%d final states written to %s\n", written, options.output.c_str());
    }

    printf("\n");

    return nonFinite == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
