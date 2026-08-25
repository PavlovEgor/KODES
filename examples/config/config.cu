#include "Config.cuh"

#include <cstdio>
#include <stdexcept>
#include <utility>

// Everything kodes::Config does, on the file beside this one.
//
//     ./config [config.json]
//
// Config is the plain JSON reader the library is built on; kodes::Settings is
// the layer above it that knows what a KODES run needs. Use Config directly
// when a program has settings of its own to read - the OpenFOAM chemistry
// model, by contrast, uses neither, because it reads an OpenFOAM dictionary.
//
// It is host-only C++ and touches nothing CUDA, so it links against one
// translation unit and no device code at all.
int main(int argc, char** argv)
{
    const char* path = argc > 1 ? argv[1] : "config.json";

    // 1) Opening. A missing file or malformed JSON throws, so that a program
    //    can say something useful about its own settings file rather than
    //    dying inside the parser.
    kodes::Config config = [path] ()
    {
        try
        {
            return kodes::Config(path);
        }
        catch (const std::exception& e)
        {
            fprintf(stderr, "cannot read settings: %s\n", e.what());
            std::exit(EXIT_FAILURE);
        }
    }();

    printf("read %s\n\n", path);

    // 2) The four getters. Every one takes the value to use when the key is
    //    absent, so there is no separate "does it exist" step for the common
    //    case and no way to read an uninitialised setting.
    printf("flat keys\n");
    printf("  name       = %s\n", config.getString("name", "unnamed").c_str());
    printf("  chemistry  = %s\n", config.getBool("chemistry", false) ? "on" : "off");
    printf("  maxSteps   = %d\n", config.getInt("maxSteps", 1000));
    printf("  absTol     = %g\n", config.getDouble("absTol", 1e-10));

    // 3) getDouble accepts any number, not only one written with a decimal
    //    point. "relTol": 1 parses as an integer, and a reader that insisted
    //    on doubles would hand back the default without saying so - the kind
    //    of mistake that is only ever noticed in the results.
    printf("\nintegers where a double is expected\n");
    printf("  relTol     = %g   (written as 1 in the file)\n", config.getDouble("relTol", 1e-1));
    printf("  endTime    = %g  (written as 10)\n", config.getDouble("endTime", 1.0));

    // 4) Dotted paths walk into nested objects, to any depth, so a settings
    //    file can be grouped instead of flat.
    printf("\nnested objects, by dotted path\n");
    printf("  device.share                  = %s\n",
        config.getString("device.share", "best").c_str());
    printf("  device.threadsPerBlock        = %d\n",
        config.getInt("device.threadsPerBlock", 256));
    printf("  device.limits.batchSize       = %d\n",
        config.getInt("device.limits.batchSize", 0));

    // 5) hasKey, for a setting whose absence means something other than a
    //    default - here, "size the run by hand", which needs both entries or
    //    neither.
    printf("\nhasKey: telling absent from defaulted\n");

    const bool hasConcurrent = config.hasKey("device.limits.concurrentSystems");
    const bool hasBatch = config.hasKey("device.limits.batchSize");

    printf("  concurrentSystems given = %s\n", hasConcurrent ? "yes" : "no");
    printf("  batchSize given         = %s\n", hasBatch ? "yes" : "no");
    printf("  sized by hand           = %s\n",
        (hasConcurrent && hasBatch) ? "yes" : "no");

    printf("  device.limits.nothingHere given = %s\n",
        config.hasKey("device.limits.nothingHere") ? "yes" : "no");

    // 6) A key of the wrong type reads as absent. That keeps a typo from
    //    becoming a crash, at the price of it becoming a default - so a
    //    setting that must not be defaulted is worth a hasKey of its own.
    printf("\nwrong type, or no key at all: the default, either way\n");
    printf("  notANumber as int   = %d   (it is the string \"twelve\")\n",
        config.getInt("notANumber", -1));
    printf("  missingKey as int   = %d\n", config.getInt("missingKey", -1));
    printf("  name.deeper         = %s   (name is not an object)\n",
        config.getString("name.deeper", "<default>").c_str());

    // 7) Config owns an open file and a parsed document, so it is movable and
    //    not copyable. Moving is how it is handed on, e.g. out of a factory
    //    function like the lambda above.
    kodes::Config moved = std::move(config);
    printf("\nafter a move, the new owner reads the same document\n");
    printf("  name       = %s\n", moved.getString("name", "unnamed").c_str());

    return 0;
}
