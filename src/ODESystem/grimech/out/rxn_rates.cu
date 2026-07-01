#include "rates/rates_include.cuh"
#include "rates.cuh"
__device__ void eval_rxn_rates (const double T, const double pres, const double * __restrict__ C, double * __restrict__ fwd_rxn_rates, double * __restrict__ rev_rxn_rates) {
  eval_rxn_rates_0(T, pres, C, fwd_rxn_rates, rev_rxn_rates);
  eval_rxn_rates_1(T, pres, C, fwd_rxn_rates, rev_rxn_rates);
} // end eval_rxn_rates

