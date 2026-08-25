#include "header.cuh"
#include "chem_utils.cuh"
#include "rates.cuh"
#include "gpu_memory.cuh"

#if defined(CONP)

__device__ void dydt (const double t, const double pres, const double * __restrict__ y, double * __restrict__ dy, const mechanism_memory * __restrict__ d_mem) {

  // species molar concentrations
  double * __restrict__ conc = d_mem->conc;
  double y_N;
  double mw_avg;
  double rho;
  eval_conc (y[INDEX(0)], pres, &y[GRID_DIM], &y_N, &mw_avg, &rho, conc);

  double * __restrict__ fwd_rates = d_mem->fwd_rates;
  double * __restrict__ rev_rates = d_mem->rev_rates;
  eval_rxn_rates (y[INDEX(0)], pres, conc, fwd_rates, rev_rates);

  // get pressure modifications to reaction rates
  double * __restrict__ pres_mod = d_mem->pres_mod;
  get_rxn_pres_mod (y[INDEX(0)], pres, conc, pres_mod);

  double * __restrict__ spec_rates = d_mem->spec_rates;
  // evaluate species molar net production rates
  eval_spec_rates (fwd_rates, rev_rates, pres_mod, spec_rates, &spec_rates[INDEX(52)]);
  // local array holding constant pressure specific heat
  double * __restrict__ cp = d_mem->cp;
  eval_cp (y[INDEX(0)], cp);

  // constant pressure mass-average specific heat
  double cp_avg = (cp[INDEX(0)] * y[INDEX(1)]) + (cp[INDEX(1)] * y[INDEX(2)])
              + (cp[INDEX(2)] * y[INDEX(3)]) + (cp[INDEX(3)] * y[INDEX(4)])
              + (cp[INDEX(4)] * y[INDEX(5)]) + (cp[INDEX(5)] * y[INDEX(6)])
              + (cp[INDEX(6)] * y[INDEX(7)]) + (cp[INDEX(7)] * y[INDEX(8)])
              + (cp[INDEX(8)] * y[INDEX(9)]) + (cp[INDEX(9)] * y[INDEX(10)])
              + (cp[INDEX(10)] * y[INDEX(11)]) + (cp[INDEX(11)] * y[INDEX(12)])
              + (cp[INDEX(12)] * y[INDEX(13)]) + (cp[INDEX(13)] * y[INDEX(14)])
              + (cp[INDEX(14)] * y[INDEX(15)]) + (cp[INDEX(15)] * y[INDEX(16)])
              + (cp[INDEX(16)] * y[INDEX(17)]) + (cp[INDEX(17)] * y[INDEX(18)])
              + (cp[INDEX(18)] * y[INDEX(19)]) + (cp[INDEX(19)] * y[INDEX(20)])
              + (cp[INDEX(20)] * y[INDEX(21)]) + (cp[INDEX(21)] * y[INDEX(22)])
              + (cp[INDEX(22)] * y[INDEX(23)]) + (cp[INDEX(23)] * y[INDEX(24)])
              + (cp[INDEX(24)] * y[INDEX(25)]) + (cp[INDEX(25)] * y[INDEX(26)])
              + (cp[INDEX(26)] * y[INDEX(27)]) + (cp[INDEX(27)] * y[INDEX(28)])
              + (cp[INDEX(28)] * y[INDEX(29)]) + (cp[INDEX(29)] * y[INDEX(30)])
              + (cp[INDEX(30)] * y[INDEX(31)]) + (cp[INDEX(31)] * y[INDEX(32)])
              + (cp[INDEX(32)] * y[INDEX(33)]) + (cp[INDEX(33)] * y[INDEX(34)])
              + (cp[INDEX(34)] * y[INDEX(35)]) + (cp[INDEX(35)] * y[INDEX(36)])
              + (cp[INDEX(36)] * y[INDEX(37)]) + (cp[INDEX(37)] * y[INDEX(38)])
              + (cp[INDEX(38)] * y[INDEX(39)]) + (cp[INDEX(39)] * y[INDEX(40)])
              + (cp[INDEX(40)] * y[INDEX(41)]) + (cp[INDEX(41)] * y[INDEX(42)])
              + (cp[INDEX(42)] * y[INDEX(43)]) + (cp[INDEX(43)] * y[INDEX(44)])
              + (cp[INDEX(44)] * y[INDEX(45)]) + (cp[INDEX(45)] * y[INDEX(46)])
              + (cp[INDEX(46)] * y[INDEX(47)]) + (cp[INDEX(47)] * y[INDEX(48)])
              + (cp[INDEX(48)] * y[INDEX(49)]) + (cp[INDEX(49)] * y[INDEX(50)])
              + (cp[INDEX(50)] * y[INDEX(51)]) + (cp[INDEX(51)] * y[INDEX(52)]) + (cp[INDEX(52)] * y_N);

  // local array for species enthalpies
  double * __restrict__ h = d_mem->h;
  eval_h(y[INDEX(0)], h);
  // rate of change of temperature
  dy[INDEX(0)] = (-1.0 / (rho * cp_avg)) * ((spec_rates[INDEX(0)] * h[INDEX(0)] * 2.0158800000000001e+00)
        + (spec_rates[INDEX(1)] * h[INDEX(1)] * 1.0079400000000001e+00)
        + (spec_rates[INDEX(2)] * h[INDEX(2)] * 1.5999400000000000e+01)
        + (spec_rates[INDEX(3)] * h[INDEX(3)] * 3.1998799999999999e+01)
        + (spec_rates[INDEX(4)] * h[INDEX(4)] * 1.7007339999999999e+01)
        + (spec_rates[INDEX(5)] * h[INDEX(5)] * 1.8015280000000001e+01)
        + (spec_rates[INDEX(6)] * h[INDEX(6)] * 3.3006740000000001e+01)
        + (spec_rates[INDEX(7)] * h[INDEX(7)] * 3.4014679999999998e+01)
        + (spec_rates[INDEX(8)] * h[INDEX(8)] * 1.2010999999999999e+01)
        + (spec_rates[INDEX(9)] * h[INDEX(9)] * 1.3018939999999999e+01)
        + (spec_rates[INDEX(10)] * h[INDEX(10)] * 1.4026879999999998e+01)
        + (spec_rates[INDEX(11)] * h[INDEX(11)] * 1.4026879999999998e+01)
        + (spec_rates[INDEX(12)] * h[INDEX(12)] * 1.5034820000000000e+01)
        + (spec_rates[INDEX(13)] * h[INDEX(13)] * 1.6042760000000001e+01)
        + (spec_rates[INDEX(14)] * h[INDEX(14)] * 2.8010399999999997e+01)
        + (spec_rates[INDEX(15)] * h[INDEX(15)] * 4.4009799999999998e+01)
        + (spec_rates[INDEX(16)] * h[INDEX(16)] * 2.9018339999999998e+01)
        + (spec_rates[INDEX(17)] * h[INDEX(17)] * 3.0026280000000000e+01)
        + (spec_rates[INDEX(18)] * h[INDEX(18)] * 3.1034219999999998e+01)
        + (spec_rates[INDEX(19)] * h[INDEX(19)] * 3.1034219999999998e+01)
        + (spec_rates[INDEX(20)] * h[INDEX(20)] * 3.2042160000000003e+01)
        + (spec_rates[INDEX(21)] * h[INDEX(21)] * 2.5029940000000000e+01)
        + (spec_rates[INDEX(22)] * h[INDEX(22)] * 2.6037879999999998e+01)
        + (spec_rates[INDEX(23)] * h[INDEX(23)] * 2.7045819999999999e+01)
        + (spec_rates[INDEX(24)] * h[INDEX(24)] * 2.8053759999999997e+01)
        + (spec_rates[INDEX(25)] * h[INDEX(25)] * 2.9061699999999998e+01)
        + (spec_rates[INDEX(26)] * h[INDEX(26)] * 3.0069640000000000e+01)
        + (spec_rates[INDEX(27)] * h[INDEX(27)] * 4.1029339999999998e+01)
        + (spec_rates[INDEX(28)] * h[INDEX(28)] * 4.2037279999999996e+01)
        + (spec_rates[INDEX(29)] * h[INDEX(29)] * 4.2037280000000003e+01)
        + (spec_rates[INDEX(30)] * h[INDEX(30)] * 1.4006740000000001e+01)
        + (spec_rates[INDEX(31)] * h[INDEX(31)] * 1.5014680000000000e+01)
        + (spec_rates[INDEX(32)] * h[INDEX(32)] * 1.6022620000000000e+01)
        + (spec_rates[INDEX(33)] * h[INDEX(33)] * 1.7030560000000001e+01)
        + (spec_rates[INDEX(34)] * h[INDEX(34)] * 2.9021420000000003e+01)
        + (spec_rates[INDEX(35)] * h[INDEX(35)] * 3.0006140000000002e+01)
        + (spec_rates[INDEX(36)] * h[INDEX(36)] * 4.6005539999999996e+01)
        + (spec_rates[INDEX(37)] * h[INDEX(37)] * 4.4012880000000003e+01)
        + (spec_rates[INDEX(38)] * h[INDEX(38)] * 3.1014080000000000e+01)
        + (spec_rates[INDEX(39)] * h[INDEX(39)] * 2.6017740000000000e+01)
        + (spec_rates[INDEX(40)] * h[INDEX(40)] * 2.7025680000000001e+01)
        + (spec_rates[INDEX(41)] * h[INDEX(41)] * 2.8033619999999999e+01)
        + (spec_rates[INDEX(42)] * h[INDEX(42)] * 4.1032419999999995e+01)
        + (spec_rates[INDEX(43)] * h[INDEX(43)] * 4.3025080000000003e+01)
        + (spec_rates[INDEX(44)] * h[INDEX(44)] * 4.3025080000000003e+01)
        + (spec_rates[INDEX(45)] * h[INDEX(45)] * 4.3025080000000003e+01)
        + (spec_rates[INDEX(46)] * h[INDEX(46)] * 4.2017139999999998e+01)
        + (spec_rates[INDEX(48)] * h[INDEX(48)] * 4.3088580000000000e+01)
        + (spec_rates[INDEX(49)] * h[INDEX(49)] * 4.4096519999999998e+01)
        + (spec_rates[INDEX(50)] * h[INDEX(50)] * 4.3045220000000000e+01)
        + (spec_rates[INDEX(51)] * h[INDEX(51)] * 4.4053159999999998e+01)
        + (spec_rates[INDEX(52)] * h[INDEX(52)] * 2.8013480000000001e+01));

  // calculate rate of change of species mass fractions
  dy[INDEX(1)] = spec_rates[INDEX(0)] * (2.0158800000000001e+00 / rho);
  dy[INDEX(2)] = spec_rates[INDEX(1)] * (1.0079400000000001e+00 / rho);
  dy[INDEX(3)] = spec_rates[INDEX(2)] * (1.5999400000000000e+01 / rho);
  dy[INDEX(4)] = spec_rates[INDEX(3)] * (3.1998799999999999e+01 / rho);
  dy[INDEX(5)] = spec_rates[INDEX(4)] * (1.7007339999999999e+01 / rho);
  dy[INDEX(6)] = spec_rates[INDEX(5)] * (1.8015280000000001e+01 / rho);
  dy[INDEX(7)] = spec_rates[INDEX(6)] * (3.3006740000000001e+01 / rho);
  dy[INDEX(8)] = spec_rates[INDEX(7)] * (3.4014679999999998e+01 / rho);
  dy[INDEX(9)] = spec_rates[INDEX(8)] * (1.2010999999999999e+01 / rho);
  dy[INDEX(10)] = spec_rates[INDEX(9)] * (1.3018939999999999e+01 / rho);
  dy[INDEX(11)] = spec_rates[INDEX(10)] * (1.4026879999999998e+01 / rho);
  dy[INDEX(12)] = spec_rates[INDEX(11)] * (1.4026879999999998e+01 / rho);
  dy[INDEX(13)] = spec_rates[INDEX(12)] * (1.5034820000000000e+01 / rho);
  dy[INDEX(14)] = spec_rates[INDEX(13)] * (1.6042760000000001e+01 / rho);
  dy[INDEX(15)] = spec_rates[INDEX(14)] * (2.8010399999999997e+01 / rho);
  dy[INDEX(16)] = spec_rates[INDEX(15)] * (4.4009799999999998e+01 / rho);
  dy[INDEX(17)] = spec_rates[INDEX(16)] * (2.9018339999999998e+01 / rho);
  dy[INDEX(18)] = spec_rates[INDEX(17)] * (3.0026280000000000e+01 / rho);
  dy[INDEX(19)] = spec_rates[INDEX(18)] * (3.1034219999999998e+01 / rho);
  dy[INDEX(20)] = spec_rates[INDEX(19)] * (3.1034219999999998e+01 / rho);
  dy[INDEX(21)] = spec_rates[INDEX(20)] * (3.2042160000000003e+01 / rho);
  dy[INDEX(22)] = spec_rates[INDEX(21)] * (2.5029940000000000e+01 / rho);
  dy[INDEX(23)] = spec_rates[INDEX(22)] * (2.6037879999999998e+01 / rho);
  dy[INDEX(24)] = spec_rates[INDEX(23)] * (2.7045819999999999e+01 / rho);
  dy[INDEX(25)] = spec_rates[INDEX(24)] * (2.8053759999999997e+01 / rho);
  dy[INDEX(26)] = spec_rates[INDEX(25)] * (2.9061699999999998e+01 / rho);
  dy[INDEX(27)] = spec_rates[INDEX(26)] * (3.0069640000000000e+01 / rho);
  dy[INDEX(28)] = spec_rates[INDEX(27)] * (4.1029339999999998e+01 / rho);
  dy[INDEX(29)] = spec_rates[INDEX(28)] * (4.2037279999999996e+01 / rho);
  dy[INDEX(30)] = spec_rates[INDEX(29)] * (4.2037280000000003e+01 / rho);
  dy[INDEX(31)] = spec_rates[INDEX(30)] * (1.4006740000000001e+01 / rho);
  dy[INDEX(32)] = spec_rates[INDEX(31)] * (1.5014680000000000e+01 / rho);
  dy[INDEX(33)] = spec_rates[INDEX(32)] * (1.6022620000000000e+01 / rho);
  dy[INDEX(34)] = spec_rates[INDEX(33)] * (1.7030560000000001e+01 / rho);
  dy[INDEX(35)] = spec_rates[INDEX(34)] * (2.9021420000000003e+01 / rho);
  dy[INDEX(36)] = spec_rates[INDEX(35)] * (3.0006140000000002e+01 / rho);
  dy[INDEX(37)] = spec_rates[INDEX(36)] * (4.6005539999999996e+01 / rho);
  dy[INDEX(38)] = spec_rates[INDEX(37)] * (4.4012880000000003e+01 / rho);
  dy[INDEX(39)] = spec_rates[INDEX(38)] * (3.1014080000000000e+01 / rho);
  dy[INDEX(40)] = spec_rates[INDEX(39)] * (2.6017740000000000e+01 / rho);
  dy[INDEX(41)] = spec_rates[INDEX(40)] * (2.7025680000000001e+01 / rho);
  dy[INDEX(42)] = spec_rates[INDEX(41)] * (2.8033619999999999e+01 / rho);
  dy[INDEX(43)] = spec_rates[INDEX(42)] * (4.1032419999999995e+01 / rho);
  dy[INDEX(44)] = spec_rates[INDEX(43)] * (4.3025080000000003e+01 / rho);
  dy[INDEX(45)] = spec_rates[INDEX(44)] * (4.3025080000000003e+01 / rho);
  dy[INDEX(46)] = spec_rates[INDEX(45)] * (4.3025080000000003e+01 / rho);
  dy[INDEX(47)] = spec_rates[INDEX(46)] * (4.2017139999999998e+01 / rho);
  dy[INDEX(48)] = spec_rates[INDEX(47)] * (3.9948000000000000e+01 / rho);
  dy[INDEX(49)] = spec_rates[INDEX(48)] * (4.3088580000000000e+01 / rho);
  dy[INDEX(50)] = spec_rates[INDEX(49)] * (4.4096519999999998e+01 / rho);
  dy[INDEX(51)] = spec_rates[INDEX(50)] * (4.3045220000000000e+01 / rho);
  dy[INDEX(52)] = spec_rates[INDEX(51)] * (4.4053159999999998e+01 / rho);

} // end dydt

#elif defined(CONV)

__device__ void dydt (const double t, const double rho, const double * __restrict__ y, double * __restrict__ dy, const mechanism_memory * __restrict__ d_mem) {

  // species molar concentrations
  double * __restrict__ conc = d_mem->conc;
  double y_N;
  double mw_avg;
  double pres;
  eval_conc_rho (y[INDEX(0)], rho, &y[GRID_DIM], &y_N, &mw_avg, &pres, conc);

  double * __restrict__ fwd_rates = d_mem->fwd_rates;
  double * __restrict__ rev_rates = d_mem->rev_rates;
  eval_rxn_rates (y[INDEX(0)], pres, conc, fwd_rates, rev_rates);

  // get pressure modifications to reaction rates
  double * __restrict__ pres_mod = d_mem->pres_mod;
  get_rxn_pres_mod (y[INDEX(0)], pres, conc, pres_mod);

  double * __restrict__ spec_rates = d_mem->spec_rates;

  // evaluate species molar net production rates
  double dy_N;  eval_spec_rates (fwd_rates, rev_rates, pres_mod, &dy[GRID_DIM], &dy_N);

  double * __restrict__ cv = d_mem->cp;
  eval_cv(y[INDEX(0)], cv);

  // constant volume mass-average specific heat
  double cv_avg = (cv[INDEX(0)] * y[INDEX(1)]) + (cv[INDEX(1)] * y[INDEX(2)])
              + (cv[INDEX(2)] * y[INDEX(3)]) + (cv[INDEX(3)] * y[INDEX(4)])
              + (cv[INDEX(4)] * y[INDEX(5)]) + (cv[INDEX(5)] * y[INDEX(6)])
              + (cv[INDEX(6)] * y[INDEX(7)]) + (cv[INDEX(7)] * y[INDEX(8)])
              + (cv[INDEX(8)] * y[INDEX(9)]) + (cv[INDEX(9)] * y[INDEX(10)])
              + (cv[INDEX(10)] * y[INDEX(11)]) + (cv[INDEX(11)] * y[INDEX(12)])
              + (cv[INDEX(12)] * y[INDEX(13)]) + (cv[INDEX(13)] * y[INDEX(14)])
              + (cv[INDEX(14)] * y[INDEX(15)]) + (cv[INDEX(15)] * y[INDEX(16)])
              + (cv[INDEX(16)] * y[INDEX(17)]) + (cv[INDEX(17)] * y[INDEX(18)])
              + (cv[INDEX(18)] * y[INDEX(19)]) + (cv[INDEX(19)] * y[INDEX(20)])
              + (cv[INDEX(20)] * y[INDEX(21)]) + (cv[INDEX(21)] * y[INDEX(22)])
              + (cv[INDEX(22)] * y[INDEX(23)]) + (cv[INDEX(23)] * y[INDEX(24)])
              + (cv[INDEX(24)] * y[INDEX(25)]) + (cv[INDEX(25)] * y[INDEX(26)])
              + (cv[INDEX(26)] * y[INDEX(27)]) + (cv[INDEX(27)] * y[INDEX(28)])
              + (cv[INDEX(28)] * y[INDEX(29)]) + (cv[INDEX(29)] * y[INDEX(30)])
              + (cv[INDEX(30)] * y[INDEX(31)]) + (cv[INDEX(31)] * y[INDEX(32)])
              + (cv[INDEX(32)] * y[INDEX(33)]) + (cv[INDEX(33)] * y[INDEX(34)])
              + (cv[INDEX(34)] * y[INDEX(35)]) + (cv[INDEX(35)] * y[INDEX(36)])
              + (cv[INDEX(36)] * y[INDEX(37)]) + (cv[INDEX(37)] * y[INDEX(38)])
              + (cv[INDEX(38)] * y[INDEX(39)]) + (cv[INDEX(39)] * y[INDEX(40)])
              + (cv[INDEX(40)] * y[INDEX(41)]) + (cv[INDEX(41)] * y[INDEX(42)])
              + (cv[INDEX(42)] * y[INDEX(43)]) + (cv[INDEX(43)] * y[INDEX(44)])
              + (cv[INDEX(44)] * y[INDEX(45)]) + (cv[INDEX(45)] * y[INDEX(46)])
              + (cv[INDEX(46)] * y[INDEX(47)]) + (cv[INDEX(47)] * y[INDEX(48)])
              + (cv[INDEX(48)] * y[INDEX(49)]) + (cv[INDEX(49)] * y[INDEX(50)])
              + (cv[INDEX(50)] * y[INDEX(51)]) + (cv[INDEX(51)] * y[INDEX(52)]) + (cv[INDEX(52)] * y_N);

  // local array for species internal energies
  double * __restrict__ u = d_mem->h;
  eval_u (y[INDEX(0)], u);

  // rate of change of temperature
  dy[INDEX(0)] = (-1.0 / (rho * cv_avg)) * ((spec_rates[INDEX(0)] * u[INDEX(0)] * 2.0158800000000001e+00)
        + (spec_rates[INDEX(1)] * u[INDEX(1)] * 1.0079400000000001e+00)
        + (spec_rates[INDEX(2)] * u[INDEX(2)] * 1.5999400000000000e+01)
        + (spec_rates[INDEX(3)] * u[INDEX(3)] * 3.1998799999999999e+01)
        + (spec_rates[INDEX(4)] * u[INDEX(4)] * 1.7007339999999999e+01)
        + (spec_rates[INDEX(5)] * u[INDEX(5)] * 1.8015280000000001e+01)
        + (spec_rates[INDEX(6)] * u[INDEX(6)] * 3.3006740000000001e+01)
        + (spec_rates[INDEX(7)] * u[INDEX(7)] * 3.4014679999999998e+01)
        + (spec_rates[INDEX(8)] * u[INDEX(8)] * 1.2010999999999999e+01)
        + (spec_rates[INDEX(9)] * u[INDEX(9)] * 1.3018939999999999e+01)
        + (spec_rates[INDEX(10)] * u[INDEX(10)] * 1.4026879999999998e+01)
        + (spec_rates[INDEX(11)] * u[INDEX(11)] * 1.4026879999999998e+01)
        + (spec_rates[INDEX(12)] * u[INDEX(12)] * 1.5034820000000000e+01)
        + (spec_rates[INDEX(13)] * u[INDEX(13)] * 1.6042760000000001e+01)
        + (spec_rates[INDEX(14)] * u[INDEX(14)] * 2.8010399999999997e+01)
        + (spec_rates[INDEX(15)] * u[INDEX(15)] * 4.4009799999999998e+01)
        + (spec_rates[INDEX(16)] * u[INDEX(16)] * 2.9018339999999998e+01)
        + (spec_rates[INDEX(17)] * u[INDEX(17)] * 3.0026280000000000e+01)
        + (spec_rates[INDEX(18)] * u[INDEX(18)] * 3.1034219999999998e+01)
        + (spec_rates[INDEX(19)] * u[INDEX(19)] * 3.1034219999999998e+01)
        + (spec_rates[INDEX(20)] * u[INDEX(20)] * 3.2042160000000003e+01)
        + (spec_rates[INDEX(21)] * u[INDEX(21)] * 2.5029940000000000e+01)
        + (spec_rates[INDEX(22)] * u[INDEX(22)] * 2.6037879999999998e+01)
        + (spec_rates[INDEX(23)] * u[INDEX(23)] * 2.7045819999999999e+01)
        + (spec_rates[INDEX(24)] * u[INDEX(24)] * 2.8053759999999997e+01)
        + (spec_rates[INDEX(25)] * u[INDEX(25)] * 2.9061699999999998e+01)
        + (spec_rates[INDEX(26)] * u[INDEX(26)] * 3.0069640000000000e+01)
        + (spec_rates[INDEX(27)] * u[INDEX(27)] * 4.1029339999999998e+01)
        + (spec_rates[INDEX(28)] * u[INDEX(28)] * 4.2037279999999996e+01)
        + (spec_rates[INDEX(29)] * u[INDEX(29)] * 4.2037280000000003e+01)
        + (spec_rates[INDEX(30)] * u[INDEX(30)] * 1.4006740000000001e+01)
        + (spec_rates[INDEX(31)] * u[INDEX(31)] * 1.5014680000000000e+01)
        + (spec_rates[INDEX(32)] * u[INDEX(32)] * 1.6022620000000000e+01)
        + (spec_rates[INDEX(33)] * u[INDEX(33)] * 1.7030560000000001e+01)
        + (spec_rates[INDEX(34)] * u[INDEX(34)] * 2.9021420000000003e+01)
        + (spec_rates[INDEX(35)] * u[INDEX(35)] * 3.0006140000000002e+01)
        + (spec_rates[INDEX(36)] * u[INDEX(36)] * 4.6005539999999996e+01)
        + (spec_rates[INDEX(37)] * u[INDEX(37)] * 4.4012880000000003e+01)
        + (spec_rates[INDEX(38)] * u[INDEX(38)] * 3.1014080000000000e+01)
        + (spec_rates[INDEX(39)] * u[INDEX(39)] * 2.6017740000000000e+01)
        + (spec_rates[INDEX(40)] * u[INDEX(40)] * 2.7025680000000001e+01)
        + (spec_rates[INDEX(41)] * u[INDEX(41)] * 2.8033619999999999e+01)
        + (spec_rates[INDEX(42)] * u[INDEX(42)] * 4.1032419999999995e+01)
        + (spec_rates[INDEX(43)] * u[INDEX(43)] * 4.3025080000000003e+01)
        + (spec_rates[INDEX(44)] * u[INDEX(44)] * 4.3025080000000003e+01)
        + (spec_rates[INDEX(45)] * u[INDEX(45)] * 4.3025080000000003e+01)
        + (spec_rates[INDEX(46)] * u[INDEX(46)] * 4.2017139999999998e+01)
        + (spec_rates[INDEX(48)] * u[INDEX(48)] * 4.3088580000000000e+01)
        + (spec_rates[INDEX(49)] * u[INDEX(49)] * 4.4096519999999998e+01)
        + (spec_rates[INDEX(50)] * u[INDEX(50)] * 4.3045220000000000e+01)
        + (spec_rates[INDEX(51)] * u[INDEX(51)] * 4.4053159999999998e+01)
        + (spec_rates[INDEX(52)] * u[INDEX(52)] * 2.8013480000000001e+01));

  // calculate rate of change of species mass fractions
  dy[INDEX(1)] = spec_rates[INDEX(0)] * (2.0158800000000001e+00 / rho);
  dy[INDEX(2)] = spec_rates[INDEX(1)] * (1.0079400000000001e+00 / rho);
  dy[INDEX(3)] = spec_rates[INDEX(2)] * (1.5999400000000000e+01 / rho);
  dy[INDEX(4)] = spec_rates[INDEX(3)] * (3.1998799999999999e+01 / rho);
  dy[INDEX(5)] = spec_rates[INDEX(4)] * (1.7007339999999999e+01 / rho);
  dy[INDEX(6)] = spec_rates[INDEX(5)] * (1.8015280000000001e+01 / rho);
  dy[INDEX(7)] = spec_rates[INDEX(6)] * (3.3006740000000001e+01 / rho);
  dy[INDEX(8)] = spec_rates[INDEX(7)] * (3.4014679999999998e+01 / rho);
  dy[INDEX(9)] = spec_rates[INDEX(8)] * (1.2010999999999999e+01 / rho);
  dy[INDEX(10)] = spec_rates[INDEX(9)] * (1.3018939999999999e+01 / rho);
  dy[INDEX(11)] = spec_rates[INDEX(10)] * (1.4026879999999998e+01 / rho);
  dy[INDEX(12)] = spec_rates[INDEX(11)] * (1.4026879999999998e+01 / rho);
  dy[INDEX(13)] = spec_rates[INDEX(12)] * (1.5034820000000000e+01 / rho);
  dy[INDEX(14)] = spec_rates[INDEX(13)] * (1.6042760000000001e+01 / rho);
  dy[INDEX(15)] = spec_rates[INDEX(14)] * (2.8010399999999997e+01 / rho);
  dy[INDEX(16)] = spec_rates[INDEX(15)] * (4.4009799999999998e+01 / rho);
  dy[INDEX(17)] = spec_rates[INDEX(16)] * (2.9018339999999998e+01 / rho);
  dy[INDEX(18)] = spec_rates[INDEX(17)] * (3.0026280000000000e+01 / rho);
  dy[INDEX(19)] = spec_rates[INDEX(18)] * (3.1034219999999998e+01 / rho);
  dy[INDEX(20)] = spec_rates[INDEX(19)] * (3.1034219999999998e+01 / rho);
  dy[INDEX(21)] = spec_rates[INDEX(20)] * (3.2042160000000003e+01 / rho);
  dy[INDEX(22)] = spec_rates[INDEX(21)] * (2.5029940000000000e+01 / rho);
  dy[INDEX(23)] = spec_rates[INDEX(22)] * (2.6037879999999998e+01 / rho);
  dy[INDEX(24)] = spec_rates[INDEX(23)] * (2.7045819999999999e+01 / rho);
  dy[INDEX(25)] = spec_rates[INDEX(24)] * (2.8053759999999997e+01 / rho);
  dy[INDEX(26)] = spec_rates[INDEX(25)] * (2.9061699999999998e+01 / rho);
  dy[INDEX(27)] = spec_rates[INDEX(26)] * (3.0069640000000000e+01 / rho);
  dy[INDEX(28)] = spec_rates[INDEX(27)] * (4.1029339999999998e+01 / rho);
  dy[INDEX(29)] = spec_rates[INDEX(28)] * (4.2037279999999996e+01 / rho);
  dy[INDEX(30)] = spec_rates[INDEX(29)] * (4.2037280000000003e+01 / rho);
  dy[INDEX(31)] = spec_rates[INDEX(30)] * (1.4006740000000001e+01 / rho);
  dy[INDEX(32)] = spec_rates[INDEX(31)] * (1.5014680000000000e+01 / rho);
  dy[INDEX(33)] = spec_rates[INDEX(32)] * (1.6022620000000000e+01 / rho);
  dy[INDEX(34)] = spec_rates[INDEX(33)] * (1.7030560000000001e+01 / rho);
  dy[INDEX(35)] = spec_rates[INDEX(34)] * (2.9021420000000003e+01 / rho);
  dy[INDEX(36)] = spec_rates[INDEX(35)] * (3.0006140000000002e+01 / rho);
  dy[INDEX(37)] = spec_rates[INDEX(36)] * (4.6005539999999996e+01 / rho);
  dy[INDEX(38)] = spec_rates[INDEX(37)] * (4.4012880000000003e+01 / rho);
  dy[INDEX(39)] = spec_rates[INDEX(38)] * (3.1014080000000000e+01 / rho);
  dy[INDEX(40)] = spec_rates[INDEX(39)] * (2.6017740000000000e+01 / rho);
  dy[INDEX(41)] = spec_rates[INDEX(40)] * (2.7025680000000001e+01 / rho);
  dy[INDEX(42)] = spec_rates[INDEX(41)] * (2.8033619999999999e+01 / rho);
  dy[INDEX(43)] = spec_rates[INDEX(42)] * (4.1032419999999995e+01 / rho);
  dy[INDEX(44)] = spec_rates[INDEX(43)] * (4.3025080000000003e+01 / rho);
  dy[INDEX(45)] = spec_rates[INDEX(44)] * (4.3025080000000003e+01 / rho);
  dy[INDEX(46)] = spec_rates[INDEX(45)] * (4.3025080000000003e+01 / rho);
  dy[INDEX(47)] = spec_rates[INDEX(46)] * (4.2017139999999998e+01 / rho);
  dy[INDEX(48)] = spec_rates[INDEX(47)] * (3.9948000000000000e+01 / rho);
  dy[INDEX(49)] = spec_rates[INDEX(48)] * (4.3088580000000000e+01 / rho);
  dy[INDEX(50)] = spec_rates[INDEX(49)] * (4.4096519999999998e+01 / rho);
  dy[INDEX(51)] = spec_rates[INDEX(50)] * (4.3045220000000000e+01 / rho);
  dy[INDEX(52)] = spec_rates[INDEX(51)] * (4.4053159999999998e+01 / rho);

} // end dydt

#endif
