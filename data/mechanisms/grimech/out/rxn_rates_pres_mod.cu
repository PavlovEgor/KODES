#include <math.h>
#include "header.cuh"
#include "rates.cuh"

__device__ void get_rxn_pres_mod (const double T, const double pres, const double * __restrict__ C, double * __restrict__ pres_mod) {
  extern volatile __shared__ double shared_temp[];
  // third body variable declaration
  register double thd;

  // pressure dependence variable declarations
  register double k0;
  register double kinf;
  register double Pr;

  // troe variable declarations
  register double logFcent;
  register double A;
  register double B;

  register double logT = log(T);
  register double m = pres / (8.31446210e+03 * T);

  // reaction 0;
  shared_temp[threadIdx.x + 3 * blockDim.x] = C[INDEX(0)];
  shared_temp[threadIdx.x + 2 * blockDim.x] = C[INDEX(5)];
  shared_temp[threadIdx.x + 1 * blockDim.x] = C[INDEX(13)];
  shared_temp[threadIdx.x] = C[INDEX(14)];
  pres_mod[INDEX(0)] = m + 1.4 * shared_temp[threadIdx.x + 3 * blockDim.x] + 14.4 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.75 * shared_temp[threadIdx.x] + 2.6 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.17000000000000004 * C[INDEX(47)];

  // reaction 1;
  pres_mod[INDEX(1)] = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];

  // reaction 11;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * C[INDEX(3)] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 2.5 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.5 * C[INDEX(47)];
  k0 = exp(2.0215768003273094e+01 - (1.5096586945774882e+03 / T));
  kinf = exp(1.6705882315860439e+01 - (1.2001786621891031e+03 / T));
  Pr = k0 * thd / kinf;
  pres_mod[INDEX(2)] =  Pr / (1.0 + Pr);

  // reaction 32;
  pres_mod[INDEX(3)] = m - 1.0 * C[INDEX(3)] - 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] - 0.25 * shared_temp[threadIdx.x] + 0.5 * C[INDEX(15)] + 0.5 * C[INDEX(26)] - 1.0 * C[INDEX(52)] - 1.0 * C[INDEX(47)];

  // reaction 38;
  pres_mod[INDEX(4)] = m - 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] - 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] - 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.37 * C[INDEX(47)];

  // reaction 42;
  pres_mod[INDEX(5)] = m - 0.27 * shared_temp[threadIdx.x + 3 * blockDim.x] + 2.65 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 2.0 * C[INDEX(26)] - 0.62 * C[INDEX(47)];

  // reaction 49;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];
  k0 = exp(4.6090922573034192e+01 - 2.76 * logT - (8.0515130377466039e+02 / T));
  kinf = 600000000000.0;
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(4.38000000e-01 * exp(-T / 9.10000000e+01) + 5.62000000e-01 * exp(-T / 5.83600000e+03) + exp(-8.55200000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(6)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 51;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 2.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];
  k0 = exp(6.3132971828612241e+01 - 4.76 * logT - (1.2278557382563572e+03 / T));
  kinf = exp(3.0262909956065194e+01 - 0.534 * logT - (2.6972568676451124e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.17000000e-01 * exp(-T / 7.40000000e+01) + 7.83000000e-01 * exp(-T / 2.94100000e+03) + exp(-6.96400000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(7)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 53;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];
  k0 = exp(4.2350749824532706e+01 - 2.57 * logT - (2.1386831506514417e+02 / T));
  kinf = exp(2.0809443533187462e+01 + 0.48 * logT - (-1.3083708686338232e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.17600000e-01 * exp(-T / 2.71000000e+02) + 7.82400000e-01 * exp(-T / 2.75500000e+03) + exp(-6.57000000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(8)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 55;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)];
  k0 = exp(6.0106229318315691e+01 - 4.82 * logT - (3.2860237585303325e+03 / T));
  kinf = exp(2.0107079697522593e+01 + 0.454 * logT - (1.8115904334929860e+03 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.81300000e-01 * exp(-T / 1.03000000e+02) + 7.18700000e-01 * exp(-T / 1.29100000e+03) + exp(-4.16000000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(9)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 56;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)];
  k0 = exp(5.6050499592221364e+01 - 4.8 * logT - (2.7979007806169448e+03 / T));
  kinf = exp(2.0107079697522593e+01 + 0.454 * logT - (1.3083708686338232e+03 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.42000000e-01 * exp(-T / 9.40000000e+01) + 7.58000000e-01 * exp(-T / 1.55500000e+03) + exp(-4.20000000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(10)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 58;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)];
  k0 = exp(5.9037099382212084e+01 - 4.65 * logT - (2.5563553894845468e+03 / T));
  kinf = exp(2.0776806603874441e+01 + 0.5 * logT - (4.3276882577887996e+01 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(4.00000000e-01 * exp(-T / 1.00000000e+02) + 6.00000000e-01 * exp(-T / 9.00000000e+04) + exp(-1.00000000e+04 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(11)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 62;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)];
  k0 = exp(8.2129493702929153e+01 - 7.44 * logT - (7.0853314732170111e+03 / T));
  kinf = exp(2.1611157094298868e+01 + 0.515 * logT - (2.5160978242958137e+01 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(3.00000000e-01 * exp(-T / 1.00000000e+02) + 7.00000000e-01 * exp(-T / 9.00000000e+04) + exp(-1.00000000e+04 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(12)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 69;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];
  k0 = exp(6.3491553350821555e+01 - 4.8 * logT - (9.5611717323240919e+02 / T));
  kinf = exp(3.2236191301916641e+01 - 1.0 * logT);
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(3.53600000e-01 * exp(-T / 1.32000000e+02) + 6.46400000e-01 * exp(-T / 1.31500000e+03) + exp(-5.56600000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(13)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 70;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];
  k0 = exp(7.9622894228529887e+01 - 7.27 * logT - (3.6332452582831552e+03 / T));
  kinf = exp(2.2446032434687513e+01 - (1.2077269556619906e+03 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.49300000e-01 * exp(-T / 9.85000000e+01) + 7.50700000e-01 * exp(-T / 1.30200000e+03) + exp(-4.16700000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(14)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 71;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];
  k0 = exp(5.5598514468478307e+01 - 3.86 * logT - (1.6706889553324204e+03 / T));
  kinf = exp(2.2528270532924488e+01 + 0.27 * logT - (1.4090147816056557e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.18000000e-01 * exp(-T / 2.07500000e+02) + 7.82000000e-01 * exp(-T / 2.66300000e+03) + exp(-6.09500000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(15)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 73;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];
  k0 = exp(8.2382237724019660e+01 - 7.62 * logT - (3.5074403670683641e+03 / T));
  kinf = exp(2.0107079697522593e+01 + 0.454 * logT - (9.1585960804367619e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.47000000e-02 * exp(-T / 2.10000000e+02) + 9.75300000e-01 * exp(-T / 9.84000000e+02) + exp(-4.37400000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(16)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 75;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];
  k0 = exp(8.1278612893528006e+01 - 7.08 * logT - (3.3640227910835029e+03 / T));
  kinf = exp(3.3886771157681913e+01 - 0.99 * logT - (7.9508691247747720e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(1.57800000e-01 * exp(-T / 1.25000000e+02) + 8.42200000e-01 * exp(-T / 2.21900000e+03) + exp(-6.88200000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(17)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 82;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];
  k0 = exp(4.9977627770478051e+01 - 3.42 * logT - (4.2446570295870377e+04 / T));
  kinf = exp(1.0668955394675699e+01 + 1.5 * logT - (4.0056277362789355e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(6.80000000e-02 * exp(-T / 1.97000000e+02) + 9.32000000e-01 * exp(-T / 1.54000000e+03) + exp(-1.03000000e+04 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(18)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 84;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];
  k0 = exp(2.8463930238863654e+01 - 0.9 * logT - (-8.5547326026057669e+02 / T));
  kinf = exp(2.5027330930150580e+01 - 0.37 * logT);
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.65400000e-01 * exp(-T / 9.40000000e+01) + 7.34600000e-01 * exp(-T / 1.75600000e+03) + exp(-5.18200000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(19)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 94;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)];
  k0 = exp(7.0463847150941262e+01 - 5.92 * logT - (1.5801094336577710e+03 / T));
  kinf = exp(3.5564817990743961e+01 - 1.43 * logT - (6.6928202126268650e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(5.88000000e-01 * exp(-T / 1.95000000e+02) + 4.12000000e-01 * exp(-T / 5.90000000e+03) + exp(-6.39400000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(20)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 130;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];
  k0 = exp(5.1646413239482754e+01 - 3.74 * logT - (9.7423307756733914e+02 / T));
  kinf = 50000000000.0;
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(4.24300000e-01 * exp(-T / 2.37000000e+02) + 5.75700000e-01 * exp(-T / 1.65200000e+03) + exp(-5.06900000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(21)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 139;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];
  k0 = exp(6.3159338704452978e+01 - 5.11 * logT - (3.5703428126757599e+03 / T));
  kinf = exp(2.0512544805630757e+01 + 0.5 * logT - (2.2695202375148242e+03 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(4.09300000e-01 * exp(-T / 2.75000000e+02) + 5.90700000e-01 * exp(-T / 1.22600000e+03) + exp(-5.18500000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(22)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 146;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)];
  k0 = exp(7.4313994752651325e+01 - 6.36 * logT - (2.5362266068901804e+03 / T));
  kinf = exp(3.3808965229979151e+01 - 1.16 * logT - (5.7618640176374140e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(3.97300000e-01 * exp(-T / 2.08000000e+02) + 6.02700000e-01 * exp(-T / 3.92200000e+03) + exp(-1.01800000e+04 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(23)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 157;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];
  k0 = exp(8.1814253686413721e+01 - 7.03 * logT - (1.3898924381410075e+03 / T));
  kinf = exp(3.1846107295846778e+01 - 1.18 * logT - (3.2910559541789246e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(3.81000000e-01 * exp(-T / 7.32000000e+01) + 6.19000000e-01 * exp(-T / 1.18000000e+03) + exp(-9.99900000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(24)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 166;
  pres_mod[INDEX(25)] = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] - 1.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)];

  // reaction 173;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];
  k0 = exp(1.1098150931075307e+02 - 9.3 * logT - (4.9214873443226119e+04 / T));
  kinf = exp(2.9710462657608385e+01 + 0.44 * logT - (4.3664361642829550e+04 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(2.65500000e-01 * exp(-T / 1.80000000e+02) + 7.34500000e-01 * exp(-T / 1.03500000e+03) + exp(-5.41700000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(26)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 184;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.375 * C[INDEX(47)];
  k0 = exp(2.7180035492518574e+01 - (2.8502356153622979e+04 / T));
  kinf = exp(2.5093978711720020e+01 - (2.8190360023410296e+04 / T));
  Pr = k0 * thd / kinf;
  pres_mod[INDEX(27)] =  Pr / (1.0 + Pr);

  // reaction 186;
  pres_mod[INDEX(28)] = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];

  // reaction 204;
  pres_mod[INDEX(29)] = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];

  // reaction 211;
  pres_mod[INDEX(30)] = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];

  // reaction 226;
  pres_mod[INDEX(31)] = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];

  // reaction 229;
  pres_mod[INDEX(32)] = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];

  // reaction 236;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];
  k0 = exp(4.6388174096502127e+01 - 3.4 * logT - (9.5611717323240919e+02 / T));
  kinf = 33000000000.0;
  Pr = k0 * thd / kinf;
  pres_mod[INDEX(33)] =  Pr / (1.0 + Pr);

  // reaction 240;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)];
  k0 = exp(4.4011481031354357e+01 - 3.16 * logT - (3.7238247799578045e+02 / T));
  kinf = exp(2.1854667948437513e+01 + 0.15 * logT);
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(3.33000000e-01 * exp(-T / 2.35000000e+02) + 6.67000000e-01 * exp(-T / 2.11700000e+03) + exp(-4.53600000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(34)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 268;
  pres_mod[INDEX(35)] = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];

  // reaction 288;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];
  k0 = exp(4.5321890694949374e+01 - 2.8 * logT - (2.9689954326690599e+02 / T));
  kinf = exp(2.1401299379696308e+01 + 0.43 * logT - (-1.8619123899789022e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(4.22000000e-01 * exp(-T / 1.22000000e+02) + 5.78000000e-01 * exp(-T / 2.53500000e+03) + exp(-9.36500000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(36)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 303;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];
  k0 = exp(8.2904991918650921e+01 - 7.63 * logT - (1.9394082029672131e+03 / T));
  kinf = exp(2.0002747459590335e+01 + 0.422 * logT - (-8.8315033632783059e+02 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(5.35000000e-01 * exp(-T / 2.01000000e+02) + 4.65000000e-01 * exp(-T / 1.77300000e+03) + exp(-5.33300000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(37)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 311;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];
  k0 = exp(1.5757273495848671e+02 - 16.82 * logT - (6.5745636148849617e+03 / T));
  kinf = 9430000000.0;
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(8.47300000e-01 * exp(-T / 2.91000000e+02) + 1.52700000e-01 * exp(-T / 2.74200000e+03) + exp(-7.74800000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(38)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 317;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];
  k0 = exp(1.3234596258932871e+02 - 14.6 * logT - (9.1434994934909864e+03 / T));
  kinf = exp(7.8438486381524717e+00 + 1.6 * logT - (2.8683515196972276e+03 / T));
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(8.10600000e-01 * exp(-T / 2.77000000e+02) + 1.89400000e-01 * exp(-T / 8.74800000e+03) + exp(-7.89100000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(39)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

  // reaction 319;
  thd = m + 1.0 * shared_temp[threadIdx.x + 3 * blockDim.x] + 5.0 * shared_temp[threadIdx.x + 2 * blockDim.x] + 1.0 * shared_temp[threadIdx.x + 1 * blockDim.x] + 0.5 * shared_temp[threadIdx.x] + 1.0 * C[INDEX(15)] + 2.0 * C[INDEX(26)] - 0.30000000000000004 * C[INDEX(47)];
  k0 = exp(1.2812831981076212e+02 - 13.545 * logT - (5.7150645981055113e+03 / T));
  kinf = 36130000000.0;
  Pr = k0 * thd / kinf;
  logFcent = log10( fmax(6.85000000e-01 * exp(-T / 3.69000000e+02) + 3.15000000e-01 * exp(-T / 3.28500000e+03) + exp(-6.66700000e+03 / T), 1.0e-300));
  A = log10(fmax(Pr, 1.0e-300)) - 0.67 * logFcent - 0.4;
  B = 0.806 - 1.1762 * logFcent - 0.14 * log10(fmax(Pr, 1.0e-300));
  pres_mod[INDEX(40)] = exp10(logFcent / (1.0 + A * A / (B * B))) * Pr / (1.0 + Pr);

} // end get_rxn_pres_mod

