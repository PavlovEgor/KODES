
#ifndef INTEG_CONTROLS
#define INTEG_CONTROLS
namespace kodes
{
class IntegratorControls
{
public:
    scalar  absTol;
    scalar  relTol;
    label   maxSteps;

    // --- linear solver -----------------------------------------------------
    // Every stage of the seulex extrapolation solves with 1/dt I - J for a
    // different dt while J is held fixed, so the stage matrices differ only by
    // a multiple of the identity. With iterativeLinearSolver on, one LU
    // factorisation is shared by several stages and the remaining stages are
    // solved with Bi-CGStab preconditioned by it, see shiftedBiCGStab. Off
    // restores one factorisation per stage.
    bool    iterativeLinearSolver;

    // rms tolerance of the iterative solve, measured against the same per
    // component scale (absTol + relTol*|y|) as the integration error, so it is
    // a fraction of the tolerance the step controller works to
    scalar  linTol;

    // iteration budget before the solver gives up and the caller refactorises
    label   maxLinIters;

    // the frozen factorisation is refreshed when the stage shift 1/dt differs
    // from the one it was built with by more than this factor either way. The
    // preconditioned spectrum spans roughly [1, ratio], so this bounds the
    // iteration count
    scalar  maxShiftRatio;

    // Expected number of Bi-CGStab iterations, used to decide whether a stage
    // is better off with its own factorisation. An LU costs about size/3 back
    // substitutions, a directly solved stage costs that plus one back
    // substitution per sub step, an iteratively solved one costs 1 + 2*iters
    // back substitutions per sub step, so reuse pays off while
    //
    //     nSubSteps*expectedLinIters < size/3
    //
    // For a small system an LU is worth so few back substitutions that no
    // stage qualifies, which is the right answer. The profile print reports
    // the iterations actually taken per solve, so this can be tuned against a
    // real run
    label   expectedLinIters;

    __device__ __host__
    IntegratorControls
    (
        const scalar absTol = 1e-12,
        const scalar relTol = 1e-1,
        const label maxSteps = 10000,
        const bool iterativeLinearSolver = true,
        const scalar linTol = 1e-6,
        const label maxLinIters = 8,
        const scalar maxShiftRatio = 2.5,
        const label expectedLinIters = 3
    )
    :
        absTol(absTol),
        relTol(relTol),
        maxSteps(maxSteps),
        iterativeLinearSolver(iterativeLinearSolver),
        linTol(linTol),
        maxLinIters(maxLinIters),
        maxShiftRatio(maxShiftRatio),
        expectedLinIters(expectedLinIters)
    {}
};
}
#endif
