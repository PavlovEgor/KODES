#include "header.cuh"
#include "rates.cuh"

__device__ void eval_spec_rates (const double * __restrict__ fwd_rates, const double * __restrict__ rev_rates, const double * __restrict__ pres_mod, double * __restrict__ sp_rates, double * __restrict__ dy_N) {
  extern volatile __shared__ double shared_temp[];
  //rxn 0
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] = -2.0 * (fwd_rates[INDEX(0)] - rev_rates[INDEX(0)]) * pres_mod[INDEX(0)];
  //sp 3
  shared_temp[threadIdx.x + 2 * blockDim.x] = (fwd_rates[INDEX(0)] - rev_rates[INDEX(0)]) * pres_mod[INDEX(0)];

  //rxn 1
  //sp 1
  shared_temp[threadIdx.x] = -(fwd_rates[INDEX(1)] - rev_rates[INDEX(1)]) * pres_mod[INDEX(1)];
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(1)] - rev_rates[INDEX(1)]) * pres_mod[INDEX(1)];
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] = (fwd_rates[INDEX(1)] - rev_rates[INDEX(1)]) * pres_mod[INDEX(1)];

  //rxn 2
  //sp 0
  sp_rates[INDEX(0)] = -(fwd_rates[INDEX(2)] - rev_rates[INDEX(2)]);
  //sp 1
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(2)] - rev_rates[INDEX(2)]);
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(2)] - rev_rates[INDEX(2)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(2)] - rev_rates[INDEX(2)]);

  //rxn 3
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(3)] - rev_rates[INDEX(3)]);
  //sp 3
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(3)] - rev_rates[INDEX(3)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(3)] - rev_rates[INDEX(3)]);
  //sp 6
  sp_rates[INDEX(6)] = -(fwd_rates[INDEX(3)] - rev_rates[INDEX(3)]);

  //rxn 4
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(4)] - rev_rates[INDEX(4)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(4)] - rev_rates[INDEX(4)]);
  //sp 6
  sp_rates[INDEX(6)] += (fwd_rates[INDEX(4)] - rev_rates[INDEX(4)]);
  //sp 7
  sp_rates[INDEX(7)] = -(fwd_rates[INDEX(4)] - rev_rates[INDEX(4)]);

  //rxn 5
  //sp 9
  sp_rates[INDEX(9)] = -(fwd_rates[INDEX(5)] - rev_rates[INDEX(5)]);
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(5)] - rev_rates[INDEX(5)]);
  //sp 1
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(5)] - rev_rates[INDEX(5)]);
  //sp 14
  sp_rates[INDEX(14)] = (fwd_rates[INDEX(5)] - rev_rates[INDEX(5)]);

  //rxn 6
  //sp 16
  sp_rates[INDEX(16)] = (fwd_rates[INDEX(6)] - rev_rates[INDEX(6)]);
  //sp 1
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(6)] - rev_rates[INDEX(6)]);
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(6)] - rev_rates[INDEX(6)]);
  //sp 10
  sp_rates[INDEX(10)] = -(fwd_rates[INDEX(6)] - rev_rates[INDEX(6)]);

  //rxn 7
  //sp 0
  sp_rates[INDEX(0)] += (fwd_rates[INDEX(7)] - rev_rates[INDEX(7)]);
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(7)] - rev_rates[INDEX(7)]);
  //sp 11
  sp_rates[INDEX(11)] = -(fwd_rates[INDEX(7)] - rev_rates[INDEX(7)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(7)] - rev_rates[INDEX(7)]);

  //rxn 8
  //sp 16
  sp_rates[INDEX(16)] += (fwd_rates[INDEX(8)] - rev_rates[INDEX(8)]);
  //sp 1
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(8)] - rev_rates[INDEX(8)]);
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(8)] - rev_rates[INDEX(8)]);
  //sp 11
  sp_rates[INDEX(11)] -= (fwd_rates[INDEX(8)] - rev_rates[INDEX(8)]);

  //rxn 9
  //sp 1
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(9)] - rev_rates[INDEX(9)]);
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(9)] - rev_rates[INDEX(9)]);
  //sp 12
  sp_rates[INDEX(12)] = -(fwd_rates[INDEX(9)] - rev_rates[INDEX(9)]);
  //sp 17
  sp_rates[INDEX(17)] = (fwd_rates[INDEX(9)] - rev_rates[INDEX(9)]);

  //rxn 10
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(10)] - rev_rates[INDEX(10)]);
  //sp 12
  sp_rates[INDEX(12)] += (fwd_rates[INDEX(10)] - rev_rates[INDEX(10)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(10)] - rev_rates[INDEX(10)]);
  //sp 13
  sp_rates[INDEX(13)] = -(fwd_rates[INDEX(10)] - rev_rates[INDEX(10)]);

  //rxn 11
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(11)] - rev_rates[INDEX(11)]) * pres_mod[INDEX(2)];
  //sp 14
  sp_rates[INDEX(14)] -= (fwd_rates[INDEX(11)] - rev_rates[INDEX(11)]) * pres_mod[INDEX(2)];
  //sp 15
  sp_rates[INDEX(15)] = (fwd_rates[INDEX(11)] - rev_rates[INDEX(11)]) * pres_mod[INDEX(2)];

  //rxn 12
  sp_rates[INDEX(3)] = shared_temp[threadIdx.x + 2 * blockDim.x];
  //sp 16
  shared_temp[threadIdx.x + 2 * blockDim.x] = -(fwd_rates[INDEX(12)] - rev_rates[INDEX(12)]);
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(12)] - rev_rates[INDEX(12)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(12)] - rev_rates[INDEX(12)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(12)] - rev_rates[INDEX(12)]);

  //rxn 13
  //sp 16
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(13)] - rev_rates[INDEX(13)]);
  //sp 1
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(13)] - rev_rates[INDEX(13)]);
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(13)] - rev_rates[INDEX(13)]);
  //sp 15
  sp_rates[INDEX(15)] += (fwd_rates[INDEX(13)] - rev_rates[INDEX(13)]);

  //rxn 14
  //sp 16
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(14)] - rev_rates[INDEX(14)]);
  //sp 17
  sp_rates[INDEX(17)] -= (fwd_rates[INDEX(14)] - rev_rates[INDEX(14)]);
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(14)] - rev_rates[INDEX(14)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(14)] - rev_rates[INDEX(14)]);

  //rxn 15
  //sp 17
  sp_rates[INDEX(17)] += (fwd_rates[INDEX(15)] - rev_rates[INDEX(15)]);
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(15)] - rev_rates[INDEX(15)]);
  //sp 18
  sp_rates[INDEX(18)] = -(fwd_rates[INDEX(15)] - rev_rates[INDEX(15)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(15)] - rev_rates[INDEX(15)]);

  //rxn 16
  //sp 17
  sp_rates[INDEX(17)] += (fwd_rates[INDEX(16)] - rev_rates[INDEX(16)]);
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(16)] - rev_rates[INDEX(16)]);
  //sp 19
  sp_rates[INDEX(19)] = -(fwd_rates[INDEX(16)] - rev_rates[INDEX(16)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(16)] - rev_rates[INDEX(16)]);

  //rxn 17
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(17)] - rev_rates[INDEX(17)]);
  //sp 18
  sp_rates[INDEX(18)] += (fwd_rates[INDEX(17)] - rev_rates[INDEX(17)]);
  //sp 20
  sp_rates[INDEX(20)] = -(fwd_rates[INDEX(17)] - rev_rates[INDEX(17)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(17)] - rev_rates[INDEX(17)]);

  //rxn 18
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(18)] - rev_rates[INDEX(18)]);
  //sp 19
  sp_rates[INDEX(19)] += (fwd_rates[INDEX(18)] - rev_rates[INDEX(18)]);
  //sp 20
  sp_rates[INDEX(20)] -= (fwd_rates[INDEX(18)] - rev_rates[INDEX(18)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(18)] - rev_rates[INDEX(18)]);

  //rxn 19
  //sp 9
  sp_rates[INDEX(9)] += (fwd_rates[INDEX(19)] - rev_rates[INDEX(19)]);
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(19)] - rev_rates[INDEX(19)]);
  //sp 21
  sp_rates[INDEX(21)] = -(fwd_rates[INDEX(19)] - rev_rates[INDEX(19)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(19)] - rev_rates[INDEX(19)]);

  //rxn 20
  sp_rates[INDEX(16)] += shared_temp[threadIdx.x + 2 * blockDim.x];
  //sp 1
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(20)] - rev_rates[INDEX(20)]);
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(20)] - rev_rates[INDEX(20)]);
  //sp 27
  sp_rates[INDEX(27)] = (fwd_rates[INDEX(20)] - rev_rates[INDEX(20)]);
  //sp 22
  shared_temp[threadIdx.x + 2 * blockDim.x] = -(fwd_rates[INDEX(20)] - rev_rates[INDEX(20)]);

  //rxn 21
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(21)] - rev_rates[INDEX(21)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(21)] - rev_rates[INDEX(21)]);
  //sp 21
  sp_rates[INDEX(21)] += (fwd_rates[INDEX(21)] - rev_rates[INDEX(21)]);
  //sp 22
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(21)] - rev_rates[INDEX(21)]);

  //rxn 22
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(22)] - rev_rates[INDEX(22)]);
  //sp 10
  sp_rates[INDEX(10)] += (fwd_rates[INDEX(22)] - rev_rates[INDEX(22)]);
  //sp 22
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(22)] - rev_rates[INDEX(22)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(22)] - rev_rates[INDEX(22)]);

  //rxn 23
  //sp 1
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(23)] - rev_rates[INDEX(23)]);
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(23)] - rev_rates[INDEX(23)]);
  //sp 28
  sp_rates[INDEX(28)] = (fwd_rates[INDEX(23)] - rev_rates[INDEX(23)]);
  //sp 23
  sp_rates[INDEX(23)] = -(fwd_rates[INDEX(23)] - rev_rates[INDEX(23)]);

  //rxn 24
  //sp 24
  sp_rates[INDEX(24)] = -(fwd_rates[INDEX(24)] - rev_rates[INDEX(24)]);
  //sp 16
  sp_rates[INDEX(16)] += (fwd_rates[INDEX(24)] - rev_rates[INDEX(24)]);
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(24)] - rev_rates[INDEX(24)]);
  //sp 12
  sp_rates[INDEX(12)] += (fwd_rates[INDEX(24)] - rev_rates[INDEX(24)]);

  //rxn 25
  //sp 25
  sp_rates[INDEX(25)] = -(fwd_rates[INDEX(25)] - rev_rates[INDEX(25)]);
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(25)] - rev_rates[INDEX(25)]);
  //sp 12
  sp_rates[INDEX(12)] += (fwd_rates[INDEX(25)] - rev_rates[INDEX(25)]);
  //sp 17
  sp_rates[INDEX(17)] += (fwd_rates[INDEX(25)] - rev_rates[INDEX(25)]);

  //rxn 26
  //sp 25
  sp_rates[INDEX(25)] += (fwd_rates[INDEX(26)] - rev_rates[INDEX(26)]);
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(26)] - rev_rates[INDEX(26)]);
  //sp 26
  sp_rates[INDEX(26)] = -(fwd_rates[INDEX(26)] - rev_rates[INDEX(26)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(26)] - rev_rates[INDEX(26)]);

  //rxn 27
  //sp 1
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(27)] - rev_rates[INDEX(27)]);
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(27)] - rev_rates[INDEX(27)]);
  //sp 27
  sp_rates[INDEX(27)] -= (fwd_rates[INDEX(27)] - rev_rates[INDEX(27)]);
  //sp 14
  sp_rates[INDEX(14)] += 2.0 * (fwd_rates[INDEX(27)] - rev_rates[INDEX(27)]);

  //rxn 28
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(28)] - rev_rates[INDEX(28)]);
  //sp 27
  sp_rates[INDEX(27)] += (fwd_rates[INDEX(28)] - rev_rates[INDEX(28)]);
  //sp 28
  sp_rates[INDEX(28)] -= (fwd_rates[INDEX(28)] - rev_rates[INDEX(28)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(28)] - rev_rates[INDEX(28)]);

  //rxn 29
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(29)] - rev_rates[INDEX(29)]);
  //sp 10
  sp_rates[INDEX(10)] += (fwd_rates[INDEX(29)] - rev_rates[INDEX(29)]);
  //sp 28
  sp_rates[INDEX(28)] -= (fwd_rates[INDEX(29)] - rev_rates[INDEX(29)]);
  //sp 15
  sp_rates[INDEX(15)] += (fwd_rates[INDEX(29)] - rev_rates[INDEX(29)]);

  //rxn 30
  sp_rates[INDEX(22)] = shared_temp[threadIdx.x + 2 * blockDim.x];
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(30)] - rev_rates[INDEX(30)]);
  //sp 3
  shared_temp[threadIdx.x + 2 * blockDim.x] = -(fwd_rates[INDEX(30)] - rev_rates[INDEX(30)]);
  //sp 14
  sp_rates[INDEX(14)] -= (fwd_rates[INDEX(30)] - rev_rates[INDEX(30)]);
  //sp 15
  sp_rates[INDEX(15)] += (fwd_rates[INDEX(30)] - rev_rates[INDEX(30)]);

  //rxn 31
  sp_rates[INDEX(1)] = shared_temp[threadIdx.x];
  //sp 16
  sp_rates[INDEX(16)] += (fwd_rates[INDEX(31)] - rev_rates[INDEX(31)]);
  //sp 17
  sp_rates[INDEX(17)] -= (fwd_rates[INDEX(31)] - rev_rates[INDEX(31)]);
  //sp 3
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(31)] - rev_rates[INDEX(31)]);
  //sp 6
  shared_temp[threadIdx.x] = (fwd_rates[INDEX(31)] - rev_rates[INDEX(31)]);

  //rxn 32
  sp_rates[INDEX(4)] = shared_temp[threadIdx.x + 1 * blockDim.x];
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] = -(fwd_rates[INDEX(32)] - rev_rates[INDEX(32)]) * pres_mod[INDEX(3)];
  //sp 3
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(32)] - rev_rates[INDEX(32)]) * pres_mod[INDEX(3)];
  //sp 6
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(32)] - rev_rates[INDEX(32)]) * pres_mod[INDEX(3)];

  //rxn 33
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(33)] - rev_rates[INDEX(33)]);
  //sp 3
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(33)] - rev_rates[INDEX(33)]);
  //sp 6
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(33)] - rev_rates[INDEX(33)]);

  //rxn 34
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(34)] - rev_rates[INDEX(34)]);
  //sp 3
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(34)] - rev_rates[INDEX(34)]);
  //sp 6
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(34)] - rev_rates[INDEX(34)]);

  //rxn 35
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(35)] - rev_rates[INDEX(35)]);
  //sp 3
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(35)] - rev_rates[INDEX(35)]);
  //sp 6
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(35)] - rev_rates[INDEX(35)]);

  //rxn 36
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(36)] - rev_rates[INDEX(36)]);
  //sp 3
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(36)] - rev_rates[INDEX(36)]);
  //sp 6
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(36)] - rev_rates[INDEX(36)]);

  //rxn 37
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(37)] - rev_rates[INDEX(37)]);
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(37)] - rev_rates[INDEX(37)]);
  //sp 3
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(37)] - rev_rates[INDEX(37)]);
  //sp 4
  sp_rates[INDEX(4)] += (fwd_rates[INDEX(37)] - rev_rates[INDEX(37)]);

  //rxn 38
  sp_rates[INDEX(6)] += shared_temp[threadIdx.x];
  //sp 0
  shared_temp[threadIdx.x] = (fwd_rates[INDEX(38)] - rev_rates[INDEX(38)]) * pres_mod[INDEX(4)];
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= 2.0 * (fwd_rates[INDEX(38)] - rev_rates[INDEX(38)]) * pres_mod[INDEX(4)];

  //rxn 39
  //sp 0
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(39)] - rev_rates[INDEX(39)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= 2.0 * (fwd_rates[INDEX(39)] - rev_rates[INDEX(39)]);

  //rxn 40
  //sp 0
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(40)] - rev_rates[INDEX(40)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= 2.0 * (fwd_rates[INDEX(40)] - rev_rates[INDEX(40)]);

  //rxn 41
  //sp 0
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(41)] - rev_rates[INDEX(41)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= 2.0 * (fwd_rates[INDEX(41)] - rev_rates[INDEX(41)]);

  //rxn 42
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(42)] - rev_rates[INDEX(42)]) * pres_mod[INDEX(5)];
  //sp 4
  sp_rates[INDEX(4)] -= (fwd_rates[INDEX(42)] - rev_rates[INDEX(42)]) * pres_mod[INDEX(5)];
  //sp 5
  sp_rates[INDEX(5)] = (fwd_rates[INDEX(42)] - rev_rates[INDEX(42)]) * pres_mod[INDEX(5)];

  //rxn 43
  sp_rates[INDEX(3)] += shared_temp[threadIdx.x + 2 * blockDim.x];
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(43)] - rev_rates[INDEX(43)]);
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(43)] - rev_rates[INDEX(43)]);
  //sp 5
  sp_rates[INDEX(5)] += (fwd_rates[INDEX(43)] - rev_rates[INDEX(43)]);
  //sp 6
  shared_temp[threadIdx.x + 2 * blockDim.x] = -(fwd_rates[INDEX(43)] - rev_rates[INDEX(43)]);

  //rxn 44
  //sp 0
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(44)] - rev_rates[INDEX(44)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(44)] - rev_rates[INDEX(44)]);
  //sp 3
  sp_rates[INDEX(3)] += (fwd_rates[INDEX(44)] - rev_rates[INDEX(44)]);
  //sp 6
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(44)] - rev_rates[INDEX(44)]);

  //rxn 45
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(45)] - rev_rates[INDEX(45)]);
  //sp 4
  sp_rates[INDEX(4)] += 2.0 * (fwd_rates[INDEX(45)] - rev_rates[INDEX(45)]);
  //sp 6
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(45)] - rev_rates[INDEX(45)]);

  //rxn 46
  //sp 0
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(46)] - rev_rates[INDEX(46)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(46)] - rev_rates[INDEX(46)]);
  //sp 6
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(46)] - rev_rates[INDEX(46)]);
  //sp 7
  sp_rates[INDEX(7)] -= (fwd_rates[INDEX(46)] - rev_rates[INDEX(46)]);

  //rxn 47
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(47)] - rev_rates[INDEX(47)]);
  //sp 4
  sp_rates[INDEX(4)] += (fwd_rates[INDEX(47)] - rev_rates[INDEX(47)]);
  //sp 5
  sp_rates[INDEX(5)] += (fwd_rates[INDEX(47)] - rev_rates[INDEX(47)]);
  //sp 7
  sp_rates[INDEX(7)] -= (fwd_rates[INDEX(47)] - rev_rates[INDEX(47)]);

  //rxn 48
  //sp 8
  sp_rates[INDEX(8)] = (fwd_rates[INDEX(48)] - rev_rates[INDEX(48)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(48)] - rev_rates[INDEX(48)]);
  //sp 9
  sp_rates[INDEX(9)] -= (fwd_rates[INDEX(48)] - rev_rates[INDEX(48)]);
  //sp 0
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(48)] - rev_rates[INDEX(48)]);

  //rxn 49
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(49)] - rev_rates[INDEX(49)]) * pres_mod[INDEX(6)];
  //sp 10
  sp_rates[INDEX(10)] -= (fwd_rates[INDEX(49)] - rev_rates[INDEX(49)]) * pres_mod[INDEX(6)];
  //sp 12
  sp_rates[INDEX(12)] += (fwd_rates[INDEX(49)] - rev_rates[INDEX(49)]) * pres_mod[INDEX(6)];

  //rxn 50
  //sp 0
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(50)] - rev_rates[INDEX(50)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(50)] - rev_rates[INDEX(50)]);
  //sp 11
  sp_rates[INDEX(11)] -= (fwd_rates[INDEX(50)] - rev_rates[INDEX(50)]);
  //sp 9
  sp_rates[INDEX(9)] += (fwd_rates[INDEX(50)] - rev_rates[INDEX(50)]);

  //rxn 51
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(51)] - rev_rates[INDEX(51)]) * pres_mod[INDEX(7)];
  //sp 12
  sp_rates[INDEX(12)] -= (fwd_rates[INDEX(51)] - rev_rates[INDEX(51)]) * pres_mod[INDEX(7)];
  //sp 13
  sp_rates[INDEX(13)] += (fwd_rates[INDEX(51)] - rev_rates[INDEX(51)]) * pres_mod[INDEX(7)];

  //rxn 52
  //sp 0
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(52)] - rev_rates[INDEX(52)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(52)] - rev_rates[INDEX(52)]);
  //sp 12
  sp_rates[INDEX(12)] += (fwd_rates[INDEX(52)] - rev_rates[INDEX(52)]);
  //sp 13
  sp_rates[INDEX(13)] -= (fwd_rates[INDEX(52)] - rev_rates[INDEX(52)]);

  //rxn 53
  //sp 16
  sp_rates[INDEX(16)] -= (fwd_rates[INDEX(53)] - rev_rates[INDEX(53)]) * pres_mod[INDEX(8)];
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(53)] - rev_rates[INDEX(53)]) * pres_mod[INDEX(8)];
  //sp 17
  sp_rates[INDEX(17)] += (fwd_rates[INDEX(53)] - rev_rates[INDEX(53)]) * pres_mod[INDEX(8)];

  //rxn 54
  //sp 16
  sp_rates[INDEX(16)] -= (fwd_rates[INDEX(54)] - rev_rates[INDEX(54)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(54)] - rev_rates[INDEX(54)]);
  //sp 0
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(54)] - rev_rates[INDEX(54)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(54)] - rev_rates[INDEX(54)]);

  //rxn 55
  sp_rates[INDEX(2)] = shared_temp[threadIdx.x + 3 * blockDim.x];
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(55)] - rev_rates[INDEX(55)]) * pres_mod[INDEX(9)];
  //sp 18
  sp_rates[INDEX(18)] += (fwd_rates[INDEX(55)] - rev_rates[INDEX(55)]) * pres_mod[INDEX(9)];
  //sp 17
  shared_temp[threadIdx.x + 3 * blockDim.x] = -(fwd_rates[INDEX(55)] - rev_rates[INDEX(55)]) * pres_mod[INDEX(9)];

  //rxn 56
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(56)] - rev_rates[INDEX(56)]) * pres_mod[INDEX(10)];
  //sp 19
  sp_rates[INDEX(19)] += (fwd_rates[INDEX(56)] - rev_rates[INDEX(56)]) * pres_mod[INDEX(10)];
  //sp 17
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(56)] - rev_rates[INDEX(56)]) * pres_mod[INDEX(10)];

  //rxn 57
  //sp 16
  sp_rates[INDEX(16)] += (fwd_rates[INDEX(57)] - rev_rates[INDEX(57)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(57)] - rev_rates[INDEX(57)]);
  //sp 17
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(57)] - rev_rates[INDEX(57)]);
  //sp 0
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(57)] - rev_rates[INDEX(57)]);

  //rxn 58
  sp_rates[INDEX(6)] += shared_temp[threadIdx.x + 2 * blockDim.x];
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(58)] - rev_rates[INDEX(58)]) * pres_mod[INDEX(11)];
  //sp 18
  shared_temp[threadIdx.x + 2 * blockDim.x] = -(fwd_rates[INDEX(58)] - rev_rates[INDEX(58)]) * pres_mod[INDEX(11)];
  //sp 20
  sp_rates[INDEX(20)] += (fwd_rates[INDEX(58)] - rev_rates[INDEX(58)]) * pres_mod[INDEX(11)];

  //rxn 59
  //sp 0
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(59)] - rev_rates[INDEX(59)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(59)] - rev_rates[INDEX(59)]);
  //sp 18
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(59)] - rev_rates[INDEX(59)]);
  //sp 17
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(59)] - rev_rates[INDEX(59)]);

  //rxn 60
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(60)] - rev_rates[INDEX(60)]);
  //sp 18
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(60)] - rev_rates[INDEX(60)]);
  //sp 4
  sp_rates[INDEX(4)] += (fwd_rates[INDEX(60)] - rev_rates[INDEX(60)]);
  //sp 12
  sp_rates[INDEX(12)] += (fwd_rates[INDEX(60)] - rev_rates[INDEX(60)]);

  //rxn 61
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(61)] - rev_rates[INDEX(61)]);
  //sp 18
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(61)] - rev_rates[INDEX(61)]);
  //sp 11
  sp_rates[INDEX(11)] += (fwd_rates[INDEX(61)] - rev_rates[INDEX(61)]);
  //sp 5
  sp_rates[INDEX(5)] += (fwd_rates[INDEX(61)] - rev_rates[INDEX(61)]);

  //rxn 62
  sp_rates[INDEX(0)] += shared_temp[threadIdx.x];
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(62)] - rev_rates[INDEX(62)]) * pres_mod[INDEX(12)];
  //sp 19
  shared_temp[threadIdx.x] = -(fwd_rates[INDEX(62)] - rev_rates[INDEX(62)]) * pres_mod[INDEX(12)];
  //sp 20
  sp_rates[INDEX(20)] += (fwd_rates[INDEX(62)] - rev_rates[INDEX(62)]) * pres_mod[INDEX(12)];

  //rxn 63
  //sp 18
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(63)] - rev_rates[INDEX(63)]);
  //sp 19
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(63)] - rev_rates[INDEX(63)]);

  //rxn 64
  //sp 0
  sp_rates[INDEX(0)] += (fwd_rates[INDEX(64)] - rev_rates[INDEX(64)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(64)] - rev_rates[INDEX(64)]);
  //sp 19
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(64)] - rev_rates[INDEX(64)]);
  //sp 17
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(64)] - rev_rates[INDEX(64)]);

  //rxn 65
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(65)] - rev_rates[INDEX(65)]);
  //sp 19
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(65)] - rev_rates[INDEX(65)]);
  //sp 4
  sp_rates[INDEX(4)] += (fwd_rates[INDEX(65)] - rev_rates[INDEX(65)]);
  //sp 12
  sp_rates[INDEX(12)] += (fwd_rates[INDEX(65)] - rev_rates[INDEX(65)]);

  //rxn 66
  //sp 11
  sp_rates[INDEX(11)] += (fwd_rates[INDEX(66)] - rev_rates[INDEX(66)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(66)] - rev_rates[INDEX(66)]);
  //sp 19
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(66)] - rev_rates[INDEX(66)]);
  //sp 5
  sp_rates[INDEX(5)] += (fwd_rates[INDEX(66)] - rev_rates[INDEX(66)]);

  //rxn 67
  //sp 0
  sp_rates[INDEX(0)] += (fwd_rates[INDEX(67)] - rev_rates[INDEX(67)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(67)] - rev_rates[INDEX(67)]);
  //sp 18
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(67)] - rev_rates[INDEX(67)]);
  //sp 20
  sp_rates[INDEX(20)] -= (fwd_rates[INDEX(67)] - rev_rates[INDEX(67)]);

  //rxn 68
  //sp 0
  sp_rates[INDEX(0)] += (fwd_rates[INDEX(68)] - rev_rates[INDEX(68)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(68)] - rev_rates[INDEX(68)]);
  //sp 19
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(68)] - rev_rates[INDEX(68)]);
  //sp 20
  sp_rates[INDEX(20)] -= (fwd_rates[INDEX(68)] - rev_rates[INDEX(68)]);

  //rxn 69
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(69)] - rev_rates[INDEX(69)]) * pres_mod[INDEX(13)];
  //sp 21
  sp_rates[INDEX(21)] -= (fwd_rates[INDEX(69)] - rev_rates[INDEX(69)]) * pres_mod[INDEX(13)];
  //sp 22
  sp_rates[INDEX(22)] += (fwd_rates[INDEX(69)] - rev_rates[INDEX(69)]) * pres_mod[INDEX(13)];

  //rxn 70
  sp_rates[INDEX(17)] += shared_temp[threadIdx.x + 3 * blockDim.x];
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(70)] - rev_rates[INDEX(70)]) * pres_mod[INDEX(14)];
  //sp 22
  sp_rates[INDEX(22)] -= (fwd_rates[INDEX(70)] - rev_rates[INDEX(70)]) * pres_mod[INDEX(14)];
  //sp 23
  shared_temp[threadIdx.x + 3 * blockDim.x] = (fwd_rates[INDEX(70)] - rev_rates[INDEX(70)]) * pres_mod[INDEX(14)];

  //rxn 71
  //sp 24
  sp_rates[INDEX(24)] += (fwd_rates[INDEX(71)] - rev_rates[INDEX(71)]) * pres_mod[INDEX(15)];
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(71)] - rev_rates[INDEX(71)]) * pres_mod[INDEX(15)];
  //sp 23
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(71)] - rev_rates[INDEX(71)]) * pres_mod[INDEX(15)];

  //rxn 72
  //sp 0
  sp_rates[INDEX(0)] += (fwd_rates[INDEX(72)] - rev_rates[INDEX(72)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(72)] - rev_rates[INDEX(72)]);
  //sp 22
  sp_rates[INDEX(22)] += (fwd_rates[INDEX(72)] - rev_rates[INDEX(72)]);
  //sp 23
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(72)] - rev_rates[INDEX(72)]);

  //rxn 73
  //sp 24
  sp_rates[INDEX(24)] -= (fwd_rates[INDEX(73)] - rev_rates[INDEX(73)]) * pres_mod[INDEX(16)];
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(73)] - rev_rates[INDEX(73)]) * pres_mod[INDEX(16)];
  //sp 25
  sp_rates[INDEX(25)] += (fwd_rates[INDEX(73)] - rev_rates[INDEX(73)]) * pres_mod[INDEX(16)];

  //rxn 74
  //sp 24
  sp_rates[INDEX(24)] -= (fwd_rates[INDEX(74)] - rev_rates[INDEX(74)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(74)] - rev_rates[INDEX(74)]);
  //sp 0
  sp_rates[INDEX(0)] += (fwd_rates[INDEX(74)] - rev_rates[INDEX(74)]);
  //sp 23
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(74)] - rev_rates[INDEX(74)]);

  //rxn 75
  sp_rates[INDEX(18)] += shared_temp[threadIdx.x + 2 * blockDim.x];
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(75)] - rev_rates[INDEX(75)]) * pres_mod[INDEX(17)];
  //sp 26
  sp_rates[INDEX(26)] += (fwd_rates[INDEX(75)] - rev_rates[INDEX(75)]) * pres_mod[INDEX(17)];
  //sp 25
  shared_temp[threadIdx.x + 2 * blockDim.x] = -(fwd_rates[INDEX(75)] - rev_rates[INDEX(75)]) * pres_mod[INDEX(17)];

  //rxn 76
  //sp 0
  sp_rates[INDEX(0)] += (fwd_rates[INDEX(76)] - rev_rates[INDEX(76)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(76)] - rev_rates[INDEX(76)]);
  //sp 25
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(76)] - rev_rates[INDEX(76)]);
  //sp 24
  sp_rates[INDEX(24)] += (fwd_rates[INDEX(76)] - rev_rates[INDEX(76)]);

  //rxn 77
  //sp 0
  sp_rates[INDEX(0)] += (fwd_rates[INDEX(77)] - rev_rates[INDEX(77)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(77)] - rev_rates[INDEX(77)]);
  //sp 26
  sp_rates[INDEX(26)] -= (fwd_rates[INDEX(77)] - rev_rates[INDEX(77)]);
  //sp 25
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(77)] - rev_rates[INDEX(77)]);

  //rxn 78
  //sp 11
  sp_rates[INDEX(11)] += (fwd_rates[INDEX(78)] - rev_rates[INDEX(78)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(78)] - rev_rates[INDEX(78)]);
  //sp 27
  sp_rates[INDEX(27)] -= (fwd_rates[INDEX(78)] - rev_rates[INDEX(78)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(78)] - rev_rates[INDEX(78)]);

  //rxn 79
  sp_rates[INDEX(19)] += shared_temp[threadIdx.x];
  //sp 0
  sp_rates[INDEX(0)] += (fwd_rates[INDEX(79)] - rev_rates[INDEX(79)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(79)] - rev_rates[INDEX(79)]);
  //sp 27
  sp_rates[INDEX(27)] += (fwd_rates[INDEX(79)] - rev_rates[INDEX(79)]);
  //sp 28
  shared_temp[threadIdx.x] = -(fwd_rates[INDEX(79)] - rev_rates[INDEX(79)]);

  //rxn 80
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(80)] - rev_rates[INDEX(80)]);
  //sp 28
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(80)] - rev_rates[INDEX(80)]);
  //sp 12
  sp_rates[INDEX(12)] += (fwd_rates[INDEX(80)] - rev_rates[INDEX(80)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(80)] - rev_rates[INDEX(80)]);

  //rxn 81
  //sp 28
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(81)] - rev_rates[INDEX(81)]);
  //sp 29
  sp_rates[INDEX(29)] = -(fwd_rates[INDEX(81)] - rev_rates[INDEX(81)]);

  //rxn 82
  //sp 0
  sp_rates[INDEX(0)] -= (fwd_rates[INDEX(82)] - rev_rates[INDEX(82)]) * pres_mod[INDEX(18)];
  //sp 17
  sp_rates[INDEX(17)] += (fwd_rates[INDEX(82)] - rev_rates[INDEX(82)]) * pres_mod[INDEX(18)];
  //sp 14
  sp_rates[INDEX(14)] -= (fwd_rates[INDEX(82)] - rev_rates[INDEX(82)]) * pres_mod[INDEX(18)];

  //rxn 83
  sp_rates[INDEX(23)] += shared_temp[threadIdx.x + 3 * blockDim.x];
  //sp 0
  sp_rates[INDEX(0)] -= (fwd_rates[INDEX(83)] - rev_rates[INDEX(83)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(83)] - rev_rates[INDEX(83)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] = -(fwd_rates[INDEX(83)] - rev_rates[INDEX(83)]);
  //sp 5
  sp_rates[INDEX(5)] += (fwd_rates[INDEX(83)] - rev_rates[INDEX(83)]);

  //rxn 84
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= 2.0 * (fwd_rates[INDEX(84)] - rev_rates[INDEX(84)]) * pres_mod[INDEX(19)];
  //sp 7
  sp_rates[INDEX(7)] += (fwd_rates[INDEX(84)] - rev_rates[INDEX(84)]) * pres_mod[INDEX(19)];

  //rxn 85
  sp_rates[INDEX(25)] += shared_temp[threadIdx.x + 2 * blockDim.x];
  //sp 2
  sp_rates[INDEX(2)] += (fwd_rates[INDEX(85)] - rev_rates[INDEX(85)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= 2.0 * (fwd_rates[INDEX(85)] - rev_rates[INDEX(85)]);
  //sp 5
  shared_temp[threadIdx.x + 2 * blockDim.x] = (fwd_rates[INDEX(85)] - rev_rates[INDEX(85)]);

  //rxn 86
  sp_rates[INDEX(28)] += shared_temp[threadIdx.x];
  //sp 3
  sp_rates[INDEX(3)] += (fwd_rates[INDEX(86)] - rev_rates[INDEX(86)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(86)] - rev_rates[INDEX(86)]);
  //sp 5
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(86)] - rev_rates[INDEX(86)]);
  //sp 6
  shared_temp[threadIdx.x] = -(fwd_rates[INDEX(86)] - rev_rates[INDEX(86)]);

  //rxn 87
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(87)] - rev_rates[INDEX(87)]);
  //sp 5
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(87)] - rev_rates[INDEX(87)]);
  //sp 6
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(87)] - rev_rates[INDEX(87)]);
  //sp 7
  sp_rates[INDEX(7)] -= (fwd_rates[INDEX(87)] - rev_rates[INDEX(87)]);

  //rxn 88
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(88)] - rev_rates[INDEX(88)]);
  //sp 5
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(88)] - rev_rates[INDEX(88)]);
  //sp 6
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(88)] - rev_rates[INDEX(88)]);
  //sp 7
  sp_rates[INDEX(7)] -= (fwd_rates[INDEX(88)] - rev_rates[INDEX(88)]);

  //rxn 89
  //sp 8
  sp_rates[INDEX(8)] -= (fwd_rates[INDEX(89)] - rev_rates[INDEX(89)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(89)] - rev_rates[INDEX(89)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(89)] - rev_rates[INDEX(89)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(89)] - rev_rates[INDEX(89)]);

  //rxn 90
  //sp 16
  sp_rates[INDEX(16)] += (fwd_rates[INDEX(90)] - rev_rates[INDEX(90)]);
  //sp 9
  sp_rates[INDEX(9)] -= (fwd_rates[INDEX(90)] - rev_rates[INDEX(90)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(90)] - rev_rates[INDEX(90)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(90)] - rev_rates[INDEX(90)]);

  //rxn 91
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(91)] - rev_rates[INDEX(91)]);
  //sp 10
  sp_rates[INDEX(10)] -= (fwd_rates[INDEX(91)] - rev_rates[INDEX(91)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(91)] - rev_rates[INDEX(91)]);
  //sp 17
  sp_rates[INDEX(17)] += (fwd_rates[INDEX(91)] - rev_rates[INDEX(91)]);

  //rxn 92
  //sp 9
  sp_rates[INDEX(9)] += (fwd_rates[INDEX(92)] - rev_rates[INDEX(92)]);
  //sp 10
  sp_rates[INDEX(10)] -= (fwd_rates[INDEX(92)] - rev_rates[INDEX(92)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(92)] - rev_rates[INDEX(92)]);
  //sp 5
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(92)] - rev_rates[INDEX(92)]);

  //rxn 93
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(93)] - rev_rates[INDEX(93)]);
  //sp 11
  sp_rates[INDEX(11)] -= (fwd_rates[INDEX(93)] - rev_rates[INDEX(93)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(93)] - rev_rates[INDEX(93)]);
  //sp 17
  sp_rates[INDEX(17)] += (fwd_rates[INDEX(93)] - rev_rates[INDEX(93)]);

  //rxn 94
  sp_rates[INDEX(6)] += shared_temp[threadIdx.x];
  //sp 20
  sp_rates[INDEX(20)] += (fwd_rates[INDEX(94)] - rev_rates[INDEX(94)]) * pres_mod[INDEX(20)];
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(94)] - rev_rates[INDEX(94)]) * pres_mod[INDEX(20)];
  //sp 12
  shared_temp[threadIdx.x] = -(fwd_rates[INDEX(94)] - rev_rates[INDEX(94)]) * pres_mod[INDEX(20)];

  //rxn 95
  //sp 10
  sp_rates[INDEX(10)] += (fwd_rates[INDEX(95)] - rev_rates[INDEX(95)]);
  //sp 5
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(95)] - rev_rates[INDEX(95)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(95)] - rev_rates[INDEX(95)]);
  //sp 12
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(95)] - rev_rates[INDEX(95)]);

  //rxn 96
  //sp 5
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(96)] - rev_rates[INDEX(96)]);
  //sp 11
  sp_rates[INDEX(11)] += (fwd_rates[INDEX(96)] - rev_rates[INDEX(96)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(96)] - rev_rates[INDEX(96)]);
  //sp 12
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(96)] - rev_rates[INDEX(96)]);

  //rxn 97
  //sp 12
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(97)] - rev_rates[INDEX(97)]);
  //sp 5
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(97)] - rev_rates[INDEX(97)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(97)] - rev_rates[INDEX(97)]);
  //sp 13
  sp_rates[INDEX(13)] -= (fwd_rates[INDEX(97)] - rev_rates[INDEX(97)]);

  //rxn 98
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(98)] - rev_rates[INDEX(98)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(98)] - rev_rates[INDEX(98)]);
  //sp 14
  sp_rates[INDEX(14)] -= (fwd_rates[INDEX(98)] - rev_rates[INDEX(98)]);
  //sp 15
  sp_rates[INDEX(15)] += (fwd_rates[INDEX(98)] - rev_rates[INDEX(98)]);

  //rxn 99
  //sp 16
  sp_rates[INDEX(16)] -= (fwd_rates[INDEX(99)] - rev_rates[INDEX(99)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(99)] - rev_rates[INDEX(99)]);
  //sp 5
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(99)] - rev_rates[INDEX(99)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(99)] - rev_rates[INDEX(99)]);

  //rxn 100
  sp_rates[INDEX(12)] += shared_temp[threadIdx.x];
  //sp 16
  sp_rates[INDEX(16)] += (fwd_rates[INDEX(100)] - rev_rates[INDEX(100)]);
  //sp 17
  shared_temp[threadIdx.x] = -(fwd_rates[INDEX(100)] - rev_rates[INDEX(100)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(100)] - rev_rates[INDEX(100)]);
  //sp 5
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(100)] - rev_rates[INDEX(100)]);

  //rxn 101
  //sp 17
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(101)] - rev_rates[INDEX(101)]);
  //sp 18
  sp_rates[INDEX(18)] -= (fwd_rates[INDEX(101)] - rev_rates[INDEX(101)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(101)] - rev_rates[INDEX(101)]);
  //sp 5
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(101)] - rev_rates[INDEX(101)]);

  //rxn 102
  //sp 17
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(102)] - rev_rates[INDEX(102)]);
  //sp 19
  sp_rates[INDEX(19)] -= (fwd_rates[INDEX(102)] - rev_rates[INDEX(102)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(102)] - rev_rates[INDEX(102)]);
  //sp 5
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(102)] - rev_rates[INDEX(102)]);

  //rxn 103
  //sp 18
  sp_rates[INDEX(18)] += (fwd_rates[INDEX(103)] - rev_rates[INDEX(103)]);
  //sp 5
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(103)] - rev_rates[INDEX(103)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(103)] - rev_rates[INDEX(103)]);
  //sp 20
  sp_rates[INDEX(20)] -= (fwd_rates[INDEX(103)] - rev_rates[INDEX(103)]);

  //rxn 104
  //sp 5
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(104)] - rev_rates[INDEX(104)]);
  //sp 19
  sp_rates[INDEX(19)] += (fwd_rates[INDEX(104)] - rev_rates[INDEX(104)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(104)] - rev_rates[INDEX(104)]);
  //sp 20
  sp_rates[INDEX(20)] -= (fwd_rates[INDEX(104)] - rev_rates[INDEX(104)]);

  //rxn 105
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(105)] - rev_rates[INDEX(105)]);
  //sp 27
  sp_rates[INDEX(27)] += (fwd_rates[INDEX(105)] - rev_rates[INDEX(105)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(105)] - rev_rates[INDEX(105)]);
  //sp 21
  sp_rates[INDEX(21)] -= (fwd_rates[INDEX(105)] - rev_rates[INDEX(105)]);

  //rxn 106
  sp_rates[INDEX(17)] += shared_temp[threadIdx.x];
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(106)] - rev_rates[INDEX(106)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(106)] - rev_rates[INDEX(106)]);
  //sp 28
  sp_rates[INDEX(28)] += (fwd_rates[INDEX(106)] - rev_rates[INDEX(106)]);
  //sp 22
  shared_temp[threadIdx.x] = -(fwd_rates[INDEX(106)] - rev_rates[INDEX(106)]);

  //rxn 107
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(107)] - rev_rates[INDEX(107)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(107)] - rev_rates[INDEX(107)]);
  //sp 29
  sp_rates[INDEX(29)] += (fwd_rates[INDEX(107)] - rev_rates[INDEX(107)]);
  //sp 22
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(107)] - rev_rates[INDEX(107)]);

  //rxn 108
  //sp 5
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(108)] - rev_rates[INDEX(108)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(108)] - rev_rates[INDEX(108)]);
  //sp 21
  sp_rates[INDEX(21)] += (fwd_rates[INDEX(108)] - rev_rates[INDEX(108)]);
  //sp 22
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(108)] - rev_rates[INDEX(108)]);

  //rxn 109
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(109)] - rev_rates[INDEX(109)]);
  //sp 12
  sp_rates[INDEX(12)] += (fwd_rates[INDEX(109)] - rev_rates[INDEX(109)]);
  //sp 22
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(109)] - rev_rates[INDEX(109)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(109)] - rev_rates[INDEX(109)]);

  //rxn 110
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(110)] - rev_rates[INDEX(110)]);
  //sp 5
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(110)] - rev_rates[INDEX(110)]);
  //sp 22
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(110)] - rev_rates[INDEX(110)]);
  //sp 23
  sp_rates[INDEX(23)] -= (fwd_rates[INDEX(110)] - rev_rates[INDEX(110)]);

  //rxn 111
  //sp 24
  sp_rates[INDEX(24)] -= (fwd_rates[INDEX(111)] - rev_rates[INDEX(111)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(111)] - rev_rates[INDEX(111)]);
  //sp 5
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(111)] - rev_rates[INDEX(111)]);
  //sp 23
  sp_rates[INDEX(23)] += (fwd_rates[INDEX(111)] - rev_rates[INDEX(111)]);

  //rxn 112
  //sp 25
  sp_rates[INDEX(25)] += (fwd_rates[INDEX(112)] - rev_rates[INDEX(112)]);
  //sp 26
  sp_rates[INDEX(26)] -= (fwd_rates[INDEX(112)] - rev_rates[INDEX(112)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(112)] - rev_rates[INDEX(112)]);
  //sp 5
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(112)] - rev_rates[INDEX(112)]);

  //rxn 113
  //sp 5
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(113)] - rev_rates[INDEX(113)]);
  //sp 27
  sp_rates[INDEX(27)] += (fwd_rates[INDEX(113)] - rev_rates[INDEX(113)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(113)] - rev_rates[INDEX(113)]);
  //sp 28
  sp_rates[INDEX(28)] -= (fwd_rates[INDEX(113)] - rev_rates[INDEX(113)]);

  //rxn 114
  sp_rates[INDEX(1)] += shared_temp[threadIdx.x + 1 * blockDim.x];
  //sp 3
  sp_rates[INDEX(3)] += (fwd_rates[INDEX(114)] - rev_rates[INDEX(114)]);
  //sp 6
  shared_temp[threadIdx.x + 1 * blockDim.x] = -2.0 * (fwd_rates[INDEX(114)] - rev_rates[INDEX(114)]);
  //sp 7
  sp_rates[INDEX(7)] += (fwd_rates[INDEX(114)] - rev_rates[INDEX(114)]);

  //rxn 115
  //sp 3
  sp_rates[INDEX(3)] += (fwd_rates[INDEX(115)] - rev_rates[INDEX(115)]);
  //sp 6
  shared_temp[threadIdx.x + 1 * blockDim.x] -= 2.0 * (fwd_rates[INDEX(115)] - rev_rates[INDEX(115)]);
  //sp 7
  sp_rates[INDEX(7)] += (fwd_rates[INDEX(115)] - rev_rates[INDEX(115)]);

  //rxn 116
  //sp 17
  sp_rates[INDEX(17)] += (fwd_rates[INDEX(116)] - rev_rates[INDEX(116)]);
  //sp 10
  sp_rates[INDEX(10)] -= (fwd_rates[INDEX(116)] - rev_rates[INDEX(116)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(116)] - rev_rates[INDEX(116)]);
  //sp 6
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(116)] - rev_rates[INDEX(116)]);

  //rxn 117
  //sp 3
  sp_rates[INDEX(3)] += (fwd_rates[INDEX(117)] - rev_rates[INDEX(117)]);
  //sp 12
  sp_rates[INDEX(12)] -= (fwd_rates[INDEX(117)] - rev_rates[INDEX(117)]);
  //sp 13
  sp_rates[INDEX(13)] += (fwd_rates[INDEX(117)] - rev_rates[INDEX(117)]);
  //sp 6
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(117)] - rev_rates[INDEX(117)]);

  //rxn 118
  //sp 19
  sp_rates[INDEX(19)] += (fwd_rates[INDEX(118)] - rev_rates[INDEX(118)]);
  //sp 12
  sp_rates[INDEX(12)] -= (fwd_rates[INDEX(118)] - rev_rates[INDEX(118)]);
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(118)] - rev_rates[INDEX(118)]);
  //sp 6
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(118)] - rev_rates[INDEX(118)]);

  //rxn 119
  //sp 4
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(119)] - rev_rates[INDEX(119)]);
  //sp 15
  sp_rates[INDEX(15)] += (fwd_rates[INDEX(119)] - rev_rates[INDEX(119)]);
  //sp 6
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(119)] - rev_rates[INDEX(119)]);
  //sp 14
  sp_rates[INDEX(14)] -= (fwd_rates[INDEX(119)] - rev_rates[INDEX(119)]);

  //rxn 120
  //sp 16
  sp_rates[INDEX(16)] += (fwd_rates[INDEX(120)] - rev_rates[INDEX(120)]);
  //sp 17
  sp_rates[INDEX(17)] -= (fwd_rates[INDEX(120)] - rev_rates[INDEX(120)]);
  //sp 6
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(120)] - rev_rates[INDEX(120)]);
  //sp 7
  sp_rates[INDEX(7)] += (fwd_rates[INDEX(120)] - rev_rates[INDEX(120)]);

  //rxn 121
  sp_rates[INDEX(22)] += shared_temp[threadIdx.x];
  //sp 8
  shared_temp[threadIdx.x] = -(fwd_rates[INDEX(121)] - rev_rates[INDEX(121)]);
  //sp 2
  sp_rates[INDEX(2)] += (fwd_rates[INDEX(121)] - rev_rates[INDEX(121)]);
  //sp 3
  sp_rates[INDEX(3)] -= (fwd_rates[INDEX(121)] - rev_rates[INDEX(121)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(121)] - rev_rates[INDEX(121)]);

  //rxn 122
  //sp 8
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(122)] - rev_rates[INDEX(122)]);
  //sp 1
  sp_rates[INDEX(1)] += (fwd_rates[INDEX(122)] - rev_rates[INDEX(122)]);
  //sp 10
  sp_rates[INDEX(10)] -= (fwd_rates[INDEX(122)] - rev_rates[INDEX(122)]);
  //sp 21
  sp_rates[INDEX(21)] += (fwd_rates[INDEX(122)] - rev_rates[INDEX(122)]);

  //rxn 123
  //sp 8
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(123)] - rev_rates[INDEX(123)]);
  //sp 1
  sp_rates[INDEX(1)] += (fwd_rates[INDEX(123)] - rev_rates[INDEX(123)]);
  //sp 12
  sp_rates[INDEX(12)] -= (fwd_rates[INDEX(123)] - rev_rates[INDEX(123)]);
  //sp 22
  sp_rates[INDEX(22)] += (fwd_rates[INDEX(123)] - rev_rates[INDEX(123)]);

  //rxn 124
  sp_rates[INDEX(5)] += shared_temp[threadIdx.x + 2 * blockDim.x];
  //sp 16
  sp_rates[INDEX(16)] += (fwd_rates[INDEX(124)] - rev_rates[INDEX(124)]);
  //sp 9
  shared_temp[threadIdx.x + 2 * blockDim.x] = -(fwd_rates[INDEX(124)] - rev_rates[INDEX(124)]);
  //sp 2
  sp_rates[INDEX(2)] += (fwd_rates[INDEX(124)] - rev_rates[INDEX(124)]);
  //sp 3
  sp_rates[INDEX(3)] -= (fwd_rates[INDEX(124)] - rev_rates[INDEX(124)]);

  //rxn 125
  sp_rates[INDEX(4)] += shared_temp[threadIdx.x + 3 * blockDim.x];
  //sp 0
  sp_rates[INDEX(0)] -= (fwd_rates[INDEX(125)] - rev_rates[INDEX(125)]);
  //sp 9
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(125)] - rev_rates[INDEX(125)]);
  //sp 10
  sp_rates[INDEX(10)] += (fwd_rates[INDEX(125)] - rev_rates[INDEX(125)]);
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] = (fwd_rates[INDEX(125)] - rev_rates[INDEX(125)]);

  //rxn 126
  //sp 9
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(126)] - rev_rates[INDEX(126)]);
  //sp 5
  sp_rates[INDEX(5)] -= (fwd_rates[INDEX(126)] - rev_rates[INDEX(126)]);
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(126)] - rev_rates[INDEX(126)]);
  //sp 17
  sp_rates[INDEX(17)] += (fwd_rates[INDEX(126)] - rev_rates[INDEX(126)]);

  //rxn 127
  //sp 9
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(127)] - rev_rates[INDEX(127)]);
  //sp 10
  sp_rates[INDEX(10)] -= (fwd_rates[INDEX(127)] - rev_rates[INDEX(127)]);
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(127)] - rev_rates[INDEX(127)]);
  //sp 22
  sp_rates[INDEX(22)] += (fwd_rates[INDEX(127)] - rev_rates[INDEX(127)]);

  //rxn 128
  //sp 9
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(128)] - rev_rates[INDEX(128)]);
  //sp 12
  sp_rates[INDEX(12)] -= (fwd_rates[INDEX(128)] - rev_rates[INDEX(128)]);
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(128)] - rev_rates[INDEX(128)]);
  //sp 23
  sp_rates[INDEX(23)] += (fwd_rates[INDEX(128)] - rev_rates[INDEX(128)]);

  //rxn 129
  //sp 24
  sp_rates[INDEX(24)] += (fwd_rates[INDEX(129)] - rev_rates[INDEX(129)]);
  //sp 9
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(129)] - rev_rates[INDEX(129)]);
  //sp 13
  sp_rates[INDEX(13)] -= (fwd_rates[INDEX(129)] - rev_rates[INDEX(129)]);
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(129)] - rev_rates[INDEX(129)]);

  //rxn 130
  //sp 9
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(130)] - rev_rates[INDEX(130)]) * pres_mod[INDEX(21)];
  //sp 27
  sp_rates[INDEX(27)] += (fwd_rates[INDEX(130)] - rev_rates[INDEX(130)]) * pres_mod[INDEX(21)];
  //sp 14
  sp_rates[INDEX(14)] -= (fwd_rates[INDEX(130)] - rev_rates[INDEX(130)]) * pres_mod[INDEX(21)];

  //rxn 131
  //sp 16
  sp_rates[INDEX(16)] += (fwd_rates[INDEX(131)] - rev_rates[INDEX(131)]);
  //sp 9
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(131)] - rev_rates[INDEX(131)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(131)] - rev_rates[INDEX(131)]);
  //sp 15
  sp_rates[INDEX(15)] -= (fwd_rates[INDEX(131)] - rev_rates[INDEX(131)]);

  //rxn 132
  //sp 9
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(132)] - rev_rates[INDEX(132)]);
  //sp 28
  sp_rates[INDEX(28)] += (fwd_rates[INDEX(132)] - rev_rates[INDEX(132)]);
  //sp 17
  sp_rates[INDEX(17)] -= (fwd_rates[INDEX(132)] - rev_rates[INDEX(132)]);
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(132)] - rev_rates[INDEX(132)]);

  //rxn 133
  //sp 9
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(133)] - rev_rates[INDEX(133)]);
  //sp 27
  sp_rates[INDEX(27)] -= (fwd_rates[INDEX(133)] - rev_rates[INDEX(133)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(133)] - rev_rates[INDEX(133)]);
  //sp 22
  sp_rates[INDEX(22)] += (fwd_rates[INDEX(133)] - rev_rates[INDEX(133)]);

  //rxn 134
  sp_rates[INDEX(6)] += shared_temp[threadIdx.x + 1 * blockDim.x];
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] += fwd_rates[INDEX(134)];
  //sp 3
  sp_rates[INDEX(3)] -= fwd_rates[INDEX(134)];
  //sp 4
  sp_rates[INDEX(4)] += fwd_rates[INDEX(134)];
  //sp 10
  shared_temp[threadIdx.x + 1 * blockDim.x] = -fwd_rates[INDEX(134)];
  //sp 14
  sp_rates[INDEX(14)] += fwd_rates[INDEX(134)];

  //rxn 135
  //sp 0
  sp_rates[INDEX(0)] -= (fwd_rates[INDEX(135)] - rev_rates[INDEX(134)]);
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(135)] - rev_rates[INDEX(134)]);
  //sp 10
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(135)] - rev_rates[INDEX(134)]);
  //sp 12
  sp_rates[INDEX(12)] += (fwd_rates[INDEX(135)] - rev_rates[INDEX(134)]);

  //rxn 136
  //sp 0
  sp_rates[INDEX(0)] += (fwd_rates[INDEX(136)] - rev_rates[INDEX(135)]);
  //sp 10
  shared_temp[threadIdx.x + 1 * blockDim.x] -= 2.0 * (fwd_rates[INDEX(136)] - rev_rates[INDEX(135)]);
  //sp 22
  sp_rates[INDEX(22)] += (fwd_rates[INDEX(136)] - rev_rates[INDEX(135)]);

  //rxn 137
  //sp 24
  sp_rates[INDEX(24)] += (fwd_rates[INDEX(137)] - rev_rates[INDEX(136)]);
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(137)] - rev_rates[INDEX(136)]);
  //sp 10
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(137)] - rev_rates[INDEX(136)]);
  //sp 12
  sp_rates[INDEX(12)] -= (fwd_rates[INDEX(137)] - rev_rates[INDEX(136)]);

  //rxn 138
  //sp 10
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(138)] - rev_rates[INDEX(137)]);
  //sp 12
  sp_rates[INDEX(12)] += 2.0 * (fwd_rates[INDEX(138)] - rev_rates[INDEX(137)]);
  //sp 13
  sp_rates[INDEX(13)] -= (fwd_rates[INDEX(138)] - rev_rates[INDEX(137)]);

  //rxn 139
  //sp 10
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(139)] - rev_rates[INDEX(138)]) * pres_mod[INDEX(22)];
  //sp 28
  sp_rates[INDEX(28)] += (fwd_rates[INDEX(139)] - rev_rates[INDEX(138)]) * pres_mod[INDEX(22)];
  //sp 14
  sp_rates[INDEX(14)] -= (fwd_rates[INDEX(139)] - rev_rates[INDEX(138)]) * pres_mod[INDEX(22)];

  //rxn 140
  //sp 10
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(140)] - rev_rates[INDEX(139)]);
  //sp 27
  sp_rates[INDEX(27)] -= (fwd_rates[INDEX(140)] - rev_rates[INDEX(139)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(140)] - rev_rates[INDEX(139)]);
  //sp 23
  sp_rates[INDEX(23)] += (fwd_rates[INDEX(140)] - rev_rates[INDEX(139)]);

  //rxn 141
  sp_rates[INDEX(8)] += shared_temp[threadIdx.x];
  //sp 10
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(141)] - rev_rates[INDEX(140)]);
  //sp 11
  shared_temp[threadIdx.x] = -(fwd_rates[INDEX(141)] - rev_rates[INDEX(140)]);

  //rxn 142
  //sp 10
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(142)] - rev_rates[INDEX(141)]);
  //sp 11
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(142)] - rev_rates[INDEX(141)]);

  //rxn 143
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(143)] - rev_rates[INDEX(142)]);
  //sp 3
  sp_rates[INDEX(3)] -= (fwd_rates[INDEX(143)] - rev_rates[INDEX(142)]);
  //sp 4
  sp_rates[INDEX(4)] += (fwd_rates[INDEX(143)] - rev_rates[INDEX(142)]);
  //sp 11
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(143)] - rev_rates[INDEX(142)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(143)] - rev_rates[INDEX(142)]);

  //rxn 144
  //sp 3
  sp_rates[INDEX(3)] -= (fwd_rates[INDEX(144)] - rev_rates[INDEX(143)]);
  //sp 11
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(144)] - rev_rates[INDEX(143)]);
  //sp 5
  sp_rates[INDEX(5)] += (fwd_rates[INDEX(144)] - rev_rates[INDEX(143)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(144)] - rev_rates[INDEX(143)]);

  //rxn 145
  //sp 0
  sp_rates[INDEX(0)] -= (fwd_rates[INDEX(145)] - rev_rates[INDEX(144)]);
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(145)] - rev_rates[INDEX(144)]);
  //sp 11
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(145)] - rev_rates[INDEX(144)]);
  //sp 12
  sp_rates[INDEX(12)] += (fwd_rates[INDEX(145)] - rev_rates[INDEX(144)]);

  //rxn 146
  //sp 11
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(146)] - rev_rates[INDEX(145)]) * pres_mod[INDEX(23)];
  //sp 20
  sp_rates[INDEX(20)] += (fwd_rates[INDEX(146)] - rev_rates[INDEX(145)]) * pres_mod[INDEX(23)];
  //sp 5
  sp_rates[INDEX(5)] -= (fwd_rates[INDEX(146)] - rev_rates[INDEX(145)]) * pres_mod[INDEX(23)];

  //rxn 147
  //sp 10
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(147)] - rev_rates[INDEX(146)]);
  //sp 11
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(147)] - rev_rates[INDEX(146)]);

  //rxn 148
  //sp 24
  sp_rates[INDEX(24)] += (fwd_rates[INDEX(148)] - rev_rates[INDEX(147)]);
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(148)] - rev_rates[INDEX(147)]);
  //sp 11
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(148)] - rev_rates[INDEX(147)]);
  //sp 12
  sp_rates[INDEX(12)] -= (fwd_rates[INDEX(148)] - rev_rates[INDEX(147)]);

  //rxn 149
  //sp 11
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(149)] - rev_rates[INDEX(148)]);
  //sp 12
  sp_rates[INDEX(12)] += 2.0 * (fwd_rates[INDEX(149)] - rev_rates[INDEX(148)]);
  //sp 13
  sp_rates[INDEX(13)] -= (fwd_rates[INDEX(149)] - rev_rates[INDEX(148)]);

  //rxn 150
  //sp 10
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(150)] - rev_rates[INDEX(149)]);
  //sp 11
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(150)] - rev_rates[INDEX(149)]);

  //rxn 151
  //sp 10
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(151)] - rev_rates[INDEX(150)]);
  //sp 11
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(151)] - rev_rates[INDEX(150)]);

  //rxn 152
  //sp 17
  sp_rates[INDEX(17)] += (fwd_rates[INDEX(152)] - rev_rates[INDEX(151)]);
  //sp 11
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(152)] - rev_rates[INDEX(151)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(152)] - rev_rates[INDEX(151)]);
  //sp 15
  sp_rates[INDEX(15)] -= (fwd_rates[INDEX(152)] - rev_rates[INDEX(151)]);

  //rxn 153
  sp_rates[INDEX(9)] += shared_temp[threadIdx.x + 2 * blockDim.x];
  //sp 25
  sp_rates[INDEX(25)] += (fwd_rates[INDEX(153)] - rev_rates[INDEX(152)]);
  //sp 26
  sp_rates[INDEX(26)] -= (fwd_rates[INDEX(153)] - rev_rates[INDEX(152)]);
  //sp 11
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(153)] - rev_rates[INDEX(152)]);
  //sp 12
  shared_temp[threadIdx.x + 2 * blockDim.x] = (fwd_rates[INDEX(153)] - rev_rates[INDEX(152)]);

  //rxn 154
  //sp 19
  sp_rates[INDEX(19)] += (fwd_rates[INDEX(154)] - rev_rates[INDEX(153)]);
  //sp 2
  sp_rates[INDEX(2)] += (fwd_rates[INDEX(154)] - rev_rates[INDEX(153)]);
  //sp 3
  sp_rates[INDEX(3)] -= (fwd_rates[INDEX(154)] - rev_rates[INDEX(153)]);
  //sp 12
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(154)] - rev_rates[INDEX(153)]);

  //rxn 155
  //sp 17
  sp_rates[INDEX(17)] += (fwd_rates[INDEX(155)] - rev_rates[INDEX(154)]);
  //sp 3
  sp_rates[INDEX(3)] -= (fwd_rates[INDEX(155)] - rev_rates[INDEX(154)]);
  //sp 12
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(155)] - rev_rates[INDEX(154)]);
  //sp 4
  sp_rates[INDEX(4)] += (fwd_rates[INDEX(155)] - rev_rates[INDEX(154)]);

  //rxn 156
  //sp 12
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(156)] - rev_rates[INDEX(155)]);
  //sp 13
  sp_rates[INDEX(13)] += (fwd_rates[INDEX(156)] - rev_rates[INDEX(155)]);
  //sp 6
  sp_rates[INDEX(6)] += (fwd_rates[INDEX(156)] - rev_rates[INDEX(155)]);
  //sp 7
  sp_rates[INDEX(7)] -= (fwd_rates[INDEX(156)] - rev_rates[INDEX(155)]);

  //rxn 157
  //sp 26
  sp_rates[INDEX(26)] += (fwd_rates[INDEX(157)] - rev_rates[INDEX(156)]) * pres_mod[INDEX(24)];
  //sp 12
  shared_temp[threadIdx.x + 2 * blockDim.x] -= 2.0 * (fwd_rates[INDEX(157)] - rev_rates[INDEX(156)]) * pres_mod[INDEX(24)];

  //rxn 158
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(158)] - rev_rates[INDEX(157)]);
  //sp 12
  shared_temp[threadIdx.x + 2 * blockDim.x] -= 2.0 * (fwd_rates[INDEX(158)] - rev_rates[INDEX(157)]);
  //sp 25
  sp_rates[INDEX(25)] += (fwd_rates[INDEX(158)] - rev_rates[INDEX(157)]);

  //rxn 159
  sp_rates[INDEX(10)] += shared_temp[threadIdx.x + 1 * blockDim.x];
  //sp 16
  sp_rates[INDEX(16)] -= (fwd_rates[INDEX(159)] - rev_rates[INDEX(158)]);
  //sp 12
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(159)] - rev_rates[INDEX(158)]);
  //sp 13
  shared_temp[threadIdx.x + 1 * blockDim.x] = (fwd_rates[INDEX(159)] - rev_rates[INDEX(158)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(159)] - rev_rates[INDEX(158)]);

  //rxn 160
  //sp 16
  sp_rates[INDEX(16)] += (fwd_rates[INDEX(160)] - rev_rates[INDEX(159)]);
  //sp 17
  sp_rates[INDEX(17)] -= (fwd_rates[INDEX(160)] - rev_rates[INDEX(159)]);
  //sp 12
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(160)] - rev_rates[INDEX(159)]);
  //sp 13
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(160)] - rev_rates[INDEX(159)]);

  //rxn 161
  //sp 18
  sp_rates[INDEX(18)] += (fwd_rates[INDEX(161)] - rev_rates[INDEX(160)]);
  //sp 13
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(161)] - rev_rates[INDEX(160)]);
  //sp 12
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(161)] - rev_rates[INDEX(160)]);
  //sp 20
  sp_rates[INDEX(20)] -= (fwd_rates[INDEX(161)] - rev_rates[INDEX(160)]);

  //rxn 162
  //sp 13
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(162)] - rev_rates[INDEX(161)]);
  //sp 19
  sp_rates[INDEX(19)] += (fwd_rates[INDEX(162)] - rev_rates[INDEX(161)]);
  //sp 12
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(162)] - rev_rates[INDEX(161)]);
  //sp 20
  sp_rates[INDEX(20)] -= (fwd_rates[INDEX(162)] - rev_rates[INDEX(161)]);

  //rxn 163
  //sp 24
  sp_rates[INDEX(24)] -= (fwd_rates[INDEX(163)] - rev_rates[INDEX(162)]);
  //sp 12
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(163)] - rev_rates[INDEX(162)]);
  //sp 13
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(163)] - rev_rates[INDEX(162)]);
  //sp 23
  sp_rates[INDEX(23)] += (fwd_rates[INDEX(163)] - rev_rates[INDEX(162)]);

  //rxn 164
  //sp 25
  sp_rates[INDEX(25)] += (fwd_rates[INDEX(164)] - rev_rates[INDEX(163)]);
  //sp 26
  sp_rates[INDEX(26)] -= (fwd_rates[INDEX(164)] - rev_rates[INDEX(163)]);
  //sp 12
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(164)] - rev_rates[INDEX(163)]);
  //sp 13
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(164)] - rev_rates[INDEX(163)]);

  //rxn 165
  sp_rates[INDEX(11)] += shared_temp[threadIdx.x];
  //sp 16
  shared_temp[threadIdx.x] = -(fwd_rates[INDEX(165)] - rev_rates[INDEX(164)]);
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(165)] - rev_rates[INDEX(164)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(165)] - rev_rates[INDEX(164)]);

  //rxn 166
  //sp 16
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(166)] - rev_rates[INDEX(165)]) * pres_mod[INDEX(25)];
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(166)] - rev_rates[INDEX(165)]) * pres_mod[INDEX(25)];
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(166)] - rev_rates[INDEX(165)]) * pres_mod[INDEX(25)];

  //rxn 167
  sp_rates[INDEX(12)] += shared_temp[threadIdx.x + 2 * blockDim.x];
  sp_rates[INDEX(13)] += shared_temp[threadIdx.x + 1 * blockDim.x];
  //sp 16
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(167)] - rev_rates[INDEX(166)]);
  //sp 3
  shared_temp[threadIdx.x + 2 * blockDim.x] = -(fwd_rates[INDEX(167)] - rev_rates[INDEX(166)]);
  //sp 6
  shared_temp[threadIdx.x + 1 * blockDim.x] = (fwd_rates[INDEX(167)] - rev_rates[INDEX(166)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(167)] - rev_rates[INDEX(166)]);

  //rxn 168
  //sp 17
  sp_rates[INDEX(17)] += (fwd_rates[INDEX(168)] - rev_rates[INDEX(167)]);
  //sp 18
  sp_rates[INDEX(18)] -= (fwd_rates[INDEX(168)] - rev_rates[INDEX(167)]);
  //sp 3
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(168)] - rev_rates[INDEX(167)]);
  //sp 6
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(168)] - rev_rates[INDEX(167)]);

  //rxn 169
  //sp 3
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(169)] - rev_rates[INDEX(168)]);
  //sp 17
  sp_rates[INDEX(17)] += (fwd_rates[INDEX(169)] - rev_rates[INDEX(168)]);
  //sp 19
  sp_rates[INDEX(19)] -= (fwd_rates[INDEX(169)] - rev_rates[INDEX(168)]);
  //sp 6
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(169)] - rev_rates[INDEX(168)]);

  //rxn 170
  //sp 16
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(170)] - rev_rates[INDEX(169)]);
  //sp 3
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(170)] - rev_rates[INDEX(169)]);
  //sp 21
  sp_rates[INDEX(21)] -= (fwd_rates[INDEX(170)] - rev_rates[INDEX(169)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(170)] - rev_rates[INDEX(169)]);

  //rxn 171
  //sp 0
  sp_rates[INDEX(0)] -= (fwd_rates[INDEX(171)] - rev_rates[INDEX(170)]);
  //sp 1
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(171)] - rev_rates[INDEX(170)]);
  //sp 21
  sp_rates[INDEX(21)] -= (fwd_rates[INDEX(171)] - rev_rates[INDEX(170)]);
  //sp 22
  sp_rates[INDEX(22)] += (fwd_rates[INDEX(171)] - rev_rates[INDEX(170)]);

  //rxn 172
  //sp 16
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(172)] - rev_rates[INDEX(171)]);
  //sp 17
  sp_rates[INDEX(17)] += (fwd_rates[INDEX(172)] - rev_rates[INDEX(171)]);
  //sp 3
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(172)] - rev_rates[INDEX(171)]);
  //sp 23
  sp_rates[INDEX(23)] -= (fwd_rates[INDEX(172)] - rev_rates[INDEX(171)]);

  //rxn 173
  //sp 24
  sp_rates[INDEX(24)] -= (fwd_rates[INDEX(173)] - rev_rates[INDEX(172)]) * pres_mod[INDEX(26)];
  //sp 0
  sp_rates[INDEX(0)] += (fwd_rates[INDEX(173)] - rev_rates[INDEX(172)]) * pres_mod[INDEX(26)];
  //sp 22
  sp_rates[INDEX(22)] += (fwd_rates[INDEX(173)] - rev_rates[INDEX(172)]) * pres_mod[INDEX(26)];

  //rxn 174
  //sp 24
  sp_rates[INDEX(24)] += (fwd_rates[INDEX(174)] - rev_rates[INDEX(173)]);
  //sp 25
  sp_rates[INDEX(25)] -= (fwd_rates[INDEX(174)] - rev_rates[INDEX(173)]);
  //sp 3
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(174)] - rev_rates[INDEX(173)]);
  //sp 6
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(174)] - rev_rates[INDEX(173)]);

  //rxn 175
  //sp 3
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(175)] - rev_rates[INDEX(174)]);
  //sp 27
  sp_rates[INDEX(27)] -= (fwd_rates[INDEX(175)] - rev_rates[INDEX(174)]);
  //sp 4
  sp_rates[INDEX(4)] += (fwd_rates[INDEX(175)] - rev_rates[INDEX(174)]);
  //sp 14
  sp_rates[INDEX(14)] += 2.0 * (fwd_rates[INDEX(175)] - rev_rates[INDEX(174)]);

  //rxn 176
  //sp 27
  sp_rates[INDEX(27)] -= 2.0 * (fwd_rates[INDEX(176)] - rev_rates[INDEX(175)]);
  //sp 14
  sp_rates[INDEX(14)] += 2.0 * (fwd_rates[INDEX(176)] - rev_rates[INDEX(175)]);
  //sp 22
  sp_rates[INDEX(22)] += (fwd_rates[INDEX(176)] - rev_rates[INDEX(175)]);

  //rxn 177
  sp_rates[INDEX(1)] += shared_temp[threadIdx.x + 3 * blockDim.x];
  sp_rates[INDEX(16)] += shared_temp[threadIdx.x];
  //sp 2
  sp_rates[INDEX(2)] += (fwd_rates[INDEX(177)] - rev_rates[INDEX(176)]);
  //sp 35
  shared_temp[threadIdx.x + 3 * blockDim.x] = -(fwd_rates[INDEX(177)] - rev_rates[INDEX(176)]);
  //sp 47
  (*dy_N) = (fwd_rates[INDEX(177)] - rev_rates[INDEX(176)]);
  //sp 30
  shared_temp[threadIdx.x] = -(fwd_rates[INDEX(177)] - rev_rates[INDEX(176)]);

  //rxn 178
  //sp 35
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(178)] - rev_rates[INDEX(177)]);
  //sp 2
  sp_rates[INDEX(2)] += (fwd_rates[INDEX(178)] - rev_rates[INDEX(177)]);
  //sp 3
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(178)] - rev_rates[INDEX(177)]);
  //sp 30
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(178)] - rev_rates[INDEX(177)]);

  //rxn 179
  //sp 1
  sp_rates[INDEX(1)] += (fwd_rates[INDEX(179)] - rev_rates[INDEX(178)]);
  //sp 35
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(179)] - rev_rates[INDEX(178)]);
  //sp 4
  sp_rates[INDEX(4)] -= (fwd_rates[INDEX(179)] - rev_rates[INDEX(178)]);
  //sp 30
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(179)] - rev_rates[INDEX(178)]);

  //rxn 180
  sp_rates[INDEX(6)] += shared_temp[threadIdx.x + 1 * blockDim.x];
  //sp 2
  sp_rates[INDEX(2)] -= (fwd_rates[INDEX(180)] - rev_rates[INDEX(179)]);
  //sp 3
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(180)] - rev_rates[INDEX(179)]);
  //sp 47
  (*dy_N) += (fwd_rates[INDEX(180)] - rev_rates[INDEX(179)]);
  //sp 37
  shared_temp[threadIdx.x + 1 * blockDim.x] = -(fwd_rates[INDEX(180)] - rev_rates[INDEX(179)]);

  //rxn 181
  //sp 2
  sp_rates[INDEX(2)] -= (fwd_rates[INDEX(181)] - rev_rates[INDEX(180)]);
  //sp 35
  shared_temp[threadIdx.x + 3 * blockDim.x] += 2.0 * (fwd_rates[INDEX(181)] - rev_rates[INDEX(180)]);
  //sp 37
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(181)] - rev_rates[INDEX(180)]);

  //rxn 182
  sp_rates[INDEX(30)] = shared_temp[threadIdx.x];
  //sp 1
  sp_rates[INDEX(1)] -= (fwd_rates[INDEX(182)] - rev_rates[INDEX(181)]);
  //sp 4
  sp_rates[INDEX(4)] += (fwd_rates[INDEX(182)] - rev_rates[INDEX(181)]);
  //sp 47
  shared_temp[threadIdx.x] = (fwd_rates[INDEX(182)] - rev_rates[INDEX(181)]);
  //sp 37
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(182)] - rev_rates[INDEX(181)]);

  //rxn 183
  //sp 4
  sp_rates[INDEX(4)] -= (fwd_rates[INDEX(183)] - rev_rates[INDEX(182)]);
  //sp 37
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(183)] - rev_rates[INDEX(182)]);
  //sp 47
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(183)] - rev_rates[INDEX(182)]);
  //sp 6
  sp_rates[INDEX(6)] += (fwd_rates[INDEX(183)] - rev_rates[INDEX(182)]);

  //rxn 184
  //sp 2
  sp_rates[INDEX(2)] += (fwd_rates[INDEX(184)] - rev_rates[INDEX(183)]) * pres_mod[INDEX(27)];
  //sp 47
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(184)] - rev_rates[INDEX(183)]) * pres_mod[INDEX(27)];
  //sp 37
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(184)] - rev_rates[INDEX(183)]) * pres_mod[INDEX(27)];

  //rxn 185
  sp_rates[INDEX(3)] += shared_temp[threadIdx.x + 2 * blockDim.x];
  //sp 35
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(185)] - rev_rates[INDEX(184)]);
  //sp 36
  shared_temp[threadIdx.x + 2 * blockDim.x] = (fwd_rates[INDEX(185)] - rev_rates[INDEX(184)]);
  //sp 4
  sp_rates[INDEX(4)] += (fwd_rates[INDEX(185)] - rev_rates[INDEX(184)]);
  //sp 6
  sp_rates[INDEX(6)] -= (fwd_rates[INDEX(185)] - rev_rates[INDEX(184)]);

  //rxn 186
  //sp 2
  sp_rates[INDEX(2)] -= (fwd_rates[INDEX(186)] - rev_rates[INDEX(185)]) * pres_mod[INDEX(28)];
  //sp 35
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(186)] - rev_rates[INDEX(185)]) * pres_mod[INDEX(28)];
  //sp 36
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(186)] - rev_rates[INDEX(185)]) * pres_mod[INDEX(28)];

  //rxn 187
  //sp 3
  sp_rates[INDEX(3)] += (fwd_rates[INDEX(187)] - rev_rates[INDEX(186)]);
  //sp 2
  sp_rates[INDEX(2)] -= (fwd_rates[INDEX(187)] - rev_rates[INDEX(186)]);
  //sp 35
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(187)] - rev_rates[INDEX(186)]);
  //sp 36
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(187)] - rev_rates[INDEX(186)]);

  //rxn 188
  sp_rates[INDEX(37)] = shared_temp[threadIdx.x + 1 * blockDim.x];
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] = -(fwd_rates[INDEX(188)] - rev_rates[INDEX(187)]);
  //sp 35
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(188)] - rev_rates[INDEX(187)]);
  //sp 36
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(188)] - rev_rates[INDEX(187)]);
  //sp 4
  sp_rates[INDEX(4)] += (fwd_rates[INDEX(188)] - rev_rates[INDEX(187)]);

  //rxn 189
  (*dy_N) += shared_temp[threadIdx.x];
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(189)] - rev_rates[INDEX(188)]);
  //sp 2
  sp_rates[INDEX(2)] -= (fwd_rates[INDEX(189)] - rev_rates[INDEX(188)]);
  //sp 35
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(189)] - rev_rates[INDEX(188)]);
  //sp 31
  shared_temp[threadIdx.x] = -(fwd_rates[INDEX(189)] - rev_rates[INDEX(188)]);

  //rxn 190
  //sp 0
  sp_rates[INDEX(0)] += (fwd_rates[INDEX(190)] - rev_rates[INDEX(189)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(190)] - rev_rates[INDEX(189)]);
  //sp 30
  sp_rates[INDEX(30)] += (fwd_rates[INDEX(190)] - rev_rates[INDEX(189)]);
  //sp 31
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(190)] - rev_rates[INDEX(189)]);

  //rxn 191
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(191)] - rev_rates[INDEX(190)]);
  //sp 4
  sp_rates[INDEX(4)] -= (fwd_rates[INDEX(191)] - rev_rates[INDEX(190)]);
  //sp 38
  sp_rates[INDEX(38)] = (fwd_rates[INDEX(191)] - rev_rates[INDEX(190)]);
  //sp 31
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(191)] - rev_rates[INDEX(190)]);

  //rxn 192
  //sp 4
  sp_rates[INDEX(4)] -= (fwd_rates[INDEX(192)] - rev_rates[INDEX(191)]);
  //sp 5
  sp_rates[INDEX(5)] += (fwd_rates[INDEX(192)] - rev_rates[INDEX(191)]);
  //sp 30
  sp_rates[INDEX(30)] += (fwd_rates[INDEX(192)] - rev_rates[INDEX(191)]);
  //sp 31
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(192)] - rev_rates[INDEX(191)]);

  //rxn 193
  //sp 2
  sp_rates[INDEX(2)] += (fwd_rates[INDEX(193)] - rev_rates[INDEX(192)]);
  //sp 3
  sp_rates[INDEX(3)] -= (fwd_rates[INDEX(193)] - rev_rates[INDEX(192)]);
  //sp 38
  sp_rates[INDEX(38)] += (fwd_rates[INDEX(193)] - rev_rates[INDEX(192)]);
  //sp 31
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(193)] - rev_rates[INDEX(192)]);

  //rxn 194
  //sp 35
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(194)] - rev_rates[INDEX(193)]);
  //sp 3
  sp_rates[INDEX(3)] -= (fwd_rates[INDEX(194)] - rev_rates[INDEX(193)]);
  //sp 4
  sp_rates[INDEX(4)] += (fwd_rates[INDEX(194)] - rev_rates[INDEX(193)]);
  //sp 31
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(194)] - rev_rates[INDEX(193)]);

  //rxn 195
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(195)] - rev_rates[INDEX(194)]);
  //sp 47
  (*dy_N) += (fwd_rates[INDEX(195)] - rev_rates[INDEX(194)]);
  //sp 30
  sp_rates[INDEX(30)] -= (fwd_rates[INDEX(195)] - rev_rates[INDEX(194)]);
  //sp 31
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(195)] - rev_rates[INDEX(194)]);

  //rxn 196
  //sp 0
  sp_rates[INDEX(0)] += (fwd_rates[INDEX(196)] - rev_rates[INDEX(195)]);
  //sp 5
  sp_rates[INDEX(5)] -= (fwd_rates[INDEX(196)] - rev_rates[INDEX(195)]);
  //sp 38
  sp_rates[INDEX(38)] += (fwd_rates[INDEX(196)] - rev_rates[INDEX(195)]);
  //sp 31
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(196)] - rev_rates[INDEX(195)]);

  //rxn 197
  //sp 35
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(197)] - rev_rates[INDEX(196)]);
  //sp 47
  (*dy_N) += (fwd_rates[INDEX(197)] - rev_rates[INDEX(196)]);
  //sp 4
  sp_rates[INDEX(4)] += (fwd_rates[INDEX(197)] - rev_rates[INDEX(196)]);
  //sp 31
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(197)] - rev_rates[INDEX(196)]);

  //rxn 198
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(198)] - rev_rates[INDEX(197)]);
  //sp 35
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(198)] - rev_rates[INDEX(197)]);
  //sp 37
  sp_rates[INDEX(37)] += (fwd_rates[INDEX(198)] - rev_rates[INDEX(197)]);
  //sp 31
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(198)] - rev_rates[INDEX(197)]);

  //rxn 199
  sp_rates[INDEX(36)] = shared_temp[threadIdx.x + 2 * blockDim.x];
  //sp 32
  shared_temp[threadIdx.x + 2 * blockDim.x] = -(fwd_rates[INDEX(199)] - rev_rates[INDEX(198)]);
  //sp 2
  sp_rates[INDEX(2)] -= (fwd_rates[INDEX(199)] - rev_rates[INDEX(198)]);
  //sp 4
  sp_rates[INDEX(4)] += (fwd_rates[INDEX(199)] - rev_rates[INDEX(198)]);
  //sp 31
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(199)] - rev_rates[INDEX(198)]);

  //rxn 200
  //sp 32
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(200)] - rev_rates[INDEX(199)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(200)] - rev_rates[INDEX(199)]);
  //sp 2
  sp_rates[INDEX(2)] -= (fwd_rates[INDEX(200)] - rev_rates[INDEX(199)]);
  //sp 38
  sp_rates[INDEX(38)] += (fwd_rates[INDEX(200)] - rev_rates[INDEX(199)]);

  //rxn 201
  //sp 32
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(201)] - rev_rates[INDEX(200)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(201)] - rev_rates[INDEX(200)]);
  //sp 0
  sp_rates[INDEX(0)] += (fwd_rates[INDEX(201)] - rev_rates[INDEX(200)]);
  //sp 31
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(201)] - rev_rates[INDEX(200)]);

  //rxn 202
  //sp 32
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(202)] - rev_rates[INDEX(201)]);
  //sp 4
  sp_rates[INDEX(4)] -= (fwd_rates[INDEX(202)] - rev_rates[INDEX(201)]);
  //sp 5
  sp_rates[INDEX(5)] += (fwd_rates[INDEX(202)] - rev_rates[INDEX(201)]);
  //sp 31
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(202)] - rev_rates[INDEX(201)]);

  //rxn 203
  sp_rates[INDEX(35)] = shared_temp[threadIdx.x + 3 * blockDim.x];
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(203)] - rev_rates[INDEX(202)]);
  //sp 34
  shared_temp[threadIdx.x + 3 * blockDim.x] = -(fwd_rates[INDEX(203)] - rev_rates[INDEX(202)]);
  //sp 47
  (*dy_N) += (fwd_rates[INDEX(203)] - rev_rates[INDEX(202)]);

  //rxn 204
  sp_rates[INDEX(31)] = shared_temp[threadIdx.x];
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(204)] - rev_rates[INDEX(203)]) * pres_mod[INDEX(29)];
  //sp 34
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(204)] - rev_rates[INDEX(203)]) * pres_mod[INDEX(29)];
  //sp 47
  shared_temp[threadIdx.x] = (fwd_rates[INDEX(204)] - rev_rates[INDEX(203)]) * pres_mod[INDEX(29)];

  //rxn 205
  //sp 34
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(205)] - rev_rates[INDEX(204)]);
  //sp 3
  sp_rates[INDEX(3)] -= (fwd_rates[INDEX(205)] - rev_rates[INDEX(204)]);
  //sp 47
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(205)] - rev_rates[INDEX(204)]);
  //sp 6
  sp_rates[INDEX(6)] += (fwd_rates[INDEX(205)] - rev_rates[INDEX(204)]);

  //rxn 206
  //sp 34
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(206)] - rev_rates[INDEX(205)]);
  //sp 2
  sp_rates[INDEX(2)] -= (fwd_rates[INDEX(206)] - rev_rates[INDEX(205)]);
  //sp 4
  sp_rates[INDEX(4)] += (fwd_rates[INDEX(206)] - rev_rates[INDEX(205)]);
  //sp 47
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(206)] - rev_rates[INDEX(205)]);

  //rxn 207
  //sp 35
  sp_rates[INDEX(35)] += (fwd_rates[INDEX(207)] - rev_rates[INDEX(206)]);
  //sp 34
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(207)] - rev_rates[INDEX(206)]);
  //sp 2
  sp_rates[INDEX(2)] -= (fwd_rates[INDEX(207)] - rev_rates[INDEX(206)]);
  //sp 31
  sp_rates[INDEX(31)] += (fwd_rates[INDEX(207)] - rev_rates[INDEX(206)]);

  //rxn 208
  //sp 0
  sp_rates[INDEX(0)] += (fwd_rates[INDEX(208)] - rev_rates[INDEX(207)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(208)] - rev_rates[INDEX(207)]);
  //sp 34
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(208)] - rev_rates[INDEX(207)]);
  //sp 47
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(208)] - rev_rates[INDEX(207)]);

  //rxn 209
  //sp 34
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(209)] - rev_rates[INDEX(208)]);
  //sp 4
  sp_rates[INDEX(4)] -= (fwd_rates[INDEX(209)] - rev_rates[INDEX(208)]);
  //sp 5
  sp_rates[INDEX(5)] += (fwd_rates[INDEX(209)] - rev_rates[INDEX(208)]);
  //sp 47
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(209)] - rev_rates[INDEX(208)]);

  //rxn 210
  //sp 34
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(210)] - rev_rates[INDEX(209)]);
  //sp 12
  sp_rates[INDEX(12)] -= (fwd_rates[INDEX(210)] - rev_rates[INDEX(209)]);
  //sp 13
  sp_rates[INDEX(13)] += (fwd_rates[INDEX(210)] - rev_rates[INDEX(209)]);
  //sp 47
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(210)] - rev_rates[INDEX(209)]);

  //rxn 211
  sp_rates[INDEX(32)] = shared_temp[threadIdx.x + 2 * blockDim.x];
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(211)] - rev_rates[INDEX(210)]) * pres_mod[INDEX(30)];
  //sp 35
  shared_temp[threadIdx.x + 2 * blockDim.x] = -(fwd_rates[INDEX(211)] - rev_rates[INDEX(210)]) * pres_mod[INDEX(30)];
  //sp 38
  sp_rates[INDEX(38)] += (fwd_rates[INDEX(211)] - rev_rates[INDEX(210)]) * pres_mod[INDEX(30)];

  //rxn 212
  sp_rates[INDEX(34)] = shared_temp[threadIdx.x + 3 * blockDim.x];
  //sp 2
  sp_rates[INDEX(2)] -= (fwd_rates[INDEX(212)] - rev_rates[INDEX(211)]);
  //sp 35
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(212)] - rev_rates[INDEX(211)]);
  //sp 4
  sp_rates[INDEX(4)] += (fwd_rates[INDEX(212)] - rev_rates[INDEX(211)]);
  //sp 38
  shared_temp[threadIdx.x + 3 * blockDim.x] = -(fwd_rates[INDEX(212)] - rev_rates[INDEX(211)]);

  //rxn 213
  //sp 0
  sp_rates[INDEX(0)] += (fwd_rates[INDEX(213)] - rev_rates[INDEX(212)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(213)] - rev_rates[INDEX(212)]);
  //sp 35
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(213)] - rev_rates[INDEX(212)]);
  //sp 38
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(213)] - rev_rates[INDEX(212)]);

  //rxn 214
  //sp 35
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(214)] - rev_rates[INDEX(213)]);
  //sp 4
  sp_rates[INDEX(4)] -= (fwd_rates[INDEX(214)] - rev_rates[INDEX(213)]);
  //sp 5
  sp_rates[INDEX(5)] += (fwd_rates[INDEX(214)] - rev_rates[INDEX(213)]);
  //sp 38
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(214)] - rev_rates[INDEX(213)]);

  //rxn 215
  //sp 35
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(215)] - rev_rates[INDEX(214)]);
  //sp 3
  sp_rates[INDEX(3)] -= (fwd_rates[INDEX(215)] - rev_rates[INDEX(214)]);
  //sp 38
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(215)] - rev_rates[INDEX(214)]);
  //sp 6
  sp_rates[INDEX(6)] += (fwd_rates[INDEX(215)] - rev_rates[INDEX(214)]);

  //rxn 216
  (*dy_N) += shared_temp[threadIdx.x];
  //sp 2
  sp_rates[INDEX(2)] -= (fwd_rates[INDEX(216)] - rev_rates[INDEX(215)]);
  //sp 30
  sp_rates[INDEX(30)] += (fwd_rates[INDEX(216)] - rev_rates[INDEX(215)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(216)] - rev_rates[INDEX(215)]);
  //sp 39
  shared_temp[threadIdx.x] = -(fwd_rates[INDEX(216)] - rev_rates[INDEX(215)]);

  //rxn 217
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(217)] - rev_rates[INDEX(216)]);
  //sp 4
  sp_rates[INDEX(4)] -= (fwd_rates[INDEX(217)] - rev_rates[INDEX(216)]);
  //sp 46
  sp_rates[INDEX(46)] = (fwd_rates[INDEX(217)] - rev_rates[INDEX(216)]);
  //sp 39
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(217)] - rev_rates[INDEX(216)]);

  //rxn 218
  //sp 40
  sp_rates[INDEX(40)] = (fwd_rates[INDEX(218)] - rev_rates[INDEX(217)]);
  //sp 4
  sp_rates[INDEX(4)] += (fwd_rates[INDEX(218)] - rev_rates[INDEX(217)]);
  //sp 5
  sp_rates[INDEX(5)] -= (fwd_rates[INDEX(218)] - rev_rates[INDEX(217)]);
  //sp 39
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(218)] - rev_rates[INDEX(217)]);

  //rxn 219
  //sp 2
  sp_rates[INDEX(2)] += (fwd_rates[INDEX(219)] - rev_rates[INDEX(218)]);
  //sp 3
  sp_rates[INDEX(3)] -= (fwd_rates[INDEX(219)] - rev_rates[INDEX(218)]);
  //sp 46
  sp_rates[INDEX(46)] += (fwd_rates[INDEX(219)] - rev_rates[INDEX(218)]);
  //sp 39
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(219)] - rev_rates[INDEX(218)]);

  //rxn 220
  //sp 0
  sp_rates[INDEX(0)] -= (fwd_rates[INDEX(220)] - rev_rates[INDEX(219)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(220)] - rev_rates[INDEX(219)]);
  //sp 40
  sp_rates[INDEX(40)] += (fwd_rates[INDEX(220)] - rev_rates[INDEX(219)]);
  //sp 39
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(220)] - rev_rates[INDEX(219)]);

  //rxn 221
  sp_rates[INDEX(38)] += shared_temp[threadIdx.x + 3 * blockDim.x];
  //sp 2
  sp_rates[INDEX(2)] -= (fwd_rates[INDEX(221)] - rev_rates[INDEX(220)]);
  //sp 35
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(221)] - rev_rates[INDEX(220)]);
  //sp 46
  shared_temp[threadIdx.x + 3 * blockDim.x] = -(fwd_rates[INDEX(221)] - rev_rates[INDEX(220)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(221)] - rev_rates[INDEX(220)]);

  //rxn 222
  sp_rates[INDEX(39)] = shared_temp[threadIdx.x];
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(222)] - rev_rates[INDEX(221)]);
  //sp 14
  shared_temp[threadIdx.x] = (fwd_rates[INDEX(222)] - rev_rates[INDEX(221)]);
  //sp 46
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(222)] - rev_rates[INDEX(221)]);
  //sp 31
  sp_rates[INDEX(31)] += (fwd_rates[INDEX(222)] - rev_rates[INDEX(221)]);

  //rxn 223
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(223)] - rev_rates[INDEX(222)]);
  //sp 35
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(223)] - rev_rates[INDEX(222)]);
  //sp 4
  sp_rates[INDEX(4)] -= (fwd_rates[INDEX(223)] - rev_rates[INDEX(222)]);
  //sp 46
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(223)] - rev_rates[INDEX(222)]);
  //sp 14
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(223)] - rev_rates[INDEX(222)]);

  //rxn 224
  //sp 47
  (*dy_N) += (fwd_rates[INDEX(224)] - rev_rates[INDEX(223)]);
  //sp 14
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(224)] - rev_rates[INDEX(223)]);
  //sp 46
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(224)] - rev_rates[INDEX(223)]);
  //sp 30
  sp_rates[INDEX(30)] -= (fwd_rates[INDEX(224)] - rev_rates[INDEX(223)]);

  //rxn 225
  //sp 35
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(225)] - rev_rates[INDEX(224)]);
  //sp 3
  sp_rates[INDEX(3)] -= (fwd_rates[INDEX(225)] - rev_rates[INDEX(224)]);
  //sp 46
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(225)] - rev_rates[INDEX(224)]);
  //sp 15
  sp_rates[INDEX(15)] += (fwd_rates[INDEX(225)] - rev_rates[INDEX(224)]);

  //rxn 226
  //sp 14
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(226)] - rev_rates[INDEX(225)]) * pres_mod[INDEX(31)];
  //sp 46
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(226)] - rev_rates[INDEX(225)]) * pres_mod[INDEX(31)];
  //sp 30
  sp_rates[INDEX(30)] += (fwd_rates[INDEX(226)] - rev_rates[INDEX(225)]) * pres_mod[INDEX(31)];

  //rxn 227
  //sp 35
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(227)] - rev_rates[INDEX(226)]);
  //sp 37
  sp_rates[INDEX(37)] += (fwd_rates[INDEX(227)] - rev_rates[INDEX(226)]);
  //sp 46
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(227)] - rev_rates[INDEX(226)]);
  //sp 14
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(227)] - rev_rates[INDEX(226)]);

  //rxn 228
  //sp 35
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(228)] - rev_rates[INDEX(227)]);
  //sp 47
  (*dy_N) += (fwd_rates[INDEX(228)] - rev_rates[INDEX(227)]);
  //sp 46
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(228)] - rev_rates[INDEX(227)]);
  //sp 15
  sp_rates[INDEX(15)] += (fwd_rates[INDEX(228)] - rev_rates[INDEX(227)]);

  //rxn 229
  sp_rates[INDEX(14)] += shared_temp[threadIdx.x];
  //sp 40
  shared_temp[threadIdx.x] = -(fwd_rates[INDEX(229)] - rev_rates[INDEX(228)]) * pres_mod[INDEX(32)];
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(229)] - rev_rates[INDEX(228)]) * pres_mod[INDEX(32)];
  //sp 39
  sp_rates[INDEX(39)] += (fwd_rates[INDEX(229)] - rev_rates[INDEX(228)]) * pres_mod[INDEX(32)];

  //rxn 230
  sp_rates[INDEX(35)] += shared_temp[threadIdx.x + 2 * blockDim.x];
  //sp 40
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(230)] - rev_rates[INDEX(229)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(230)] - rev_rates[INDEX(229)]);
  //sp 2
  shared_temp[threadIdx.x + 2 * blockDim.x] = -(fwd_rates[INDEX(230)] - rev_rates[INDEX(229)]);
  //sp 46
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(230)] - rev_rates[INDEX(229)]);

  //rxn 231
  //sp 40
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(231)] - rev_rates[INDEX(230)]);
  //sp 2
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(231)] - rev_rates[INDEX(230)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(231)] - rev_rates[INDEX(230)]);
  //sp 31
  sp_rates[INDEX(31)] += (fwd_rates[INDEX(231)] - rev_rates[INDEX(230)]);

  //rxn 232
  sp_rates[INDEX(1)] += shared_temp[threadIdx.x + 1 * blockDim.x];
  //sp 40
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(232)] - rev_rates[INDEX(231)]);
  //sp 2
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(232)] - rev_rates[INDEX(231)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] = (fwd_rates[INDEX(232)] - rev_rates[INDEX(231)]);
  //sp 39
  sp_rates[INDEX(39)] += (fwd_rates[INDEX(232)] - rev_rates[INDEX(231)]);

  //rxn 233
  //sp 40
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(233)] - rev_rates[INDEX(232)]);
  //sp 1
  sp_rates[INDEX(1)] += (fwd_rates[INDEX(233)] - rev_rates[INDEX(232)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(233)] - rev_rates[INDEX(232)]);
  //sp 44
  sp_rates[INDEX(44)] = (fwd_rates[INDEX(233)] - rev_rates[INDEX(232)]);

  //rxn 234
  //sp 40
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(234)] - rev_rates[INDEX(233)]);
  //sp 1
  sp_rates[INDEX(1)] += (fwd_rates[INDEX(234)] - rev_rates[INDEX(233)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(234)] - rev_rates[INDEX(233)]);
  //sp 45
  sp_rates[INDEX(45)] = (fwd_rates[INDEX(234)] - rev_rates[INDEX(233)]);

  //rxn 235
  //sp 40
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(235)] - rev_rates[INDEX(234)]);
  //sp 32
  sp_rates[INDEX(32)] += (fwd_rates[INDEX(235)] - rev_rates[INDEX(234)]);
  //sp 4
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(235)] - rev_rates[INDEX(234)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(235)] - rev_rates[INDEX(234)]);

  //rxn 236
  //sp 40
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(236)] - rev_rates[INDEX(235)]) * pres_mod[INDEX(33)];
  //sp 1
  sp_rates[INDEX(1)] -= (fwd_rates[INDEX(236)] - rev_rates[INDEX(235)]) * pres_mod[INDEX(33)];
  //sp 41
  sp_rates[INDEX(41)] = (fwd_rates[INDEX(236)] - rev_rates[INDEX(235)]) * pres_mod[INDEX(33)];

  //rxn 237
  sp_rates[INDEX(46)] += shared_temp[threadIdx.x + 3 * blockDim.x];
  sp_rates[INDEX(2)] += shared_temp[threadIdx.x + 2 * blockDim.x];
  //sp 41
  sp_rates[INDEX(41)] -= (fwd_rates[INDEX(237)] - rev_rates[INDEX(236)]);
  //sp 10
  sp_rates[INDEX(10)] += (fwd_rates[INDEX(237)] - rev_rates[INDEX(236)]);
  //sp 47
  shared_temp[threadIdx.x + 3 * blockDim.x] = (fwd_rates[INDEX(237)] - rev_rates[INDEX(236)]);
  //sp 30
  shared_temp[threadIdx.x + 2 * blockDim.x] = -(fwd_rates[INDEX(237)] - rev_rates[INDEX(236)]);

  //rxn 238
  //sp 8
  sp_rates[INDEX(8)] -= (fwd_rates[INDEX(238)] - rev_rates[INDEX(237)]);
  //sp 47
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(238)] - rev_rates[INDEX(237)]);
  //sp 30
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(238)] - rev_rates[INDEX(237)]);
  //sp 39
  sp_rates[INDEX(39)] += (fwd_rates[INDEX(238)] - rev_rates[INDEX(237)]);

  //rxn 239
  //sp 40
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(239)] - rev_rates[INDEX(238)]);
  //sp 9
  sp_rates[INDEX(9)] -= (fwd_rates[INDEX(239)] - rev_rates[INDEX(238)]);
  //sp 47
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(239)] - rev_rates[INDEX(238)]);
  //sp 30
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(239)] - rev_rates[INDEX(238)]);

  //rxn 240
  //sp 9
  sp_rates[INDEX(9)] -= (fwd_rates[INDEX(240)] - rev_rates[INDEX(239)]) * pres_mod[INDEX(34)];
  //sp 42
  sp_rates[INDEX(42)] = (fwd_rates[INDEX(240)] - rev_rates[INDEX(239)]) * pres_mod[INDEX(34)];
  //sp 47
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(240)] - rev_rates[INDEX(239)]) * pres_mod[INDEX(34)];

  //rxn 241
  //sp 40
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(241)] - rev_rates[INDEX(240)]);
  //sp 10
  sp_rates[INDEX(10)] -= (fwd_rates[INDEX(241)] - rev_rates[INDEX(240)]);
  //sp 47
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(241)] - rev_rates[INDEX(240)]);
  //sp 31
  sp_rates[INDEX(31)] += (fwd_rates[INDEX(241)] - rev_rates[INDEX(240)]);

  //rxn 242
  //sp 40
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(242)] - rev_rates[INDEX(241)]);
  //sp 11
  sp_rates[INDEX(11)] -= (fwd_rates[INDEX(242)] - rev_rates[INDEX(241)]);
  //sp 47
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(242)] - rev_rates[INDEX(241)]);
  //sp 31
  sp_rates[INDEX(31)] += (fwd_rates[INDEX(242)] - rev_rates[INDEX(241)]);

  //rxn 243
  sp_rates[INDEX(4)] += shared_temp[threadIdx.x + 1 * blockDim.x];
  //sp 8
  sp_rates[INDEX(8)] -= (fwd_rates[INDEX(243)] - rev_rates[INDEX(242)]);
  //sp 2
  sp_rates[INDEX(2)] += (fwd_rates[INDEX(243)] - rev_rates[INDEX(242)]);
  //sp 35
  shared_temp[threadIdx.x + 1 * blockDim.x] = -(fwd_rates[INDEX(243)] - rev_rates[INDEX(242)]);
  //sp 39
  sp_rates[INDEX(39)] += (fwd_rates[INDEX(243)] - rev_rates[INDEX(242)]);

  //rxn 244
  //sp 8
  sp_rates[INDEX(8)] -= (fwd_rates[INDEX(244)] - rev_rates[INDEX(243)]);
  //sp 35
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(244)] - rev_rates[INDEX(243)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(244)] - rev_rates[INDEX(243)]);
  //sp 30
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(244)] - rev_rates[INDEX(243)]);

  //rxn 245
  (*dy_N) += shared_temp[threadIdx.x + 3 * blockDim.x];
  //sp 40
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(245)] - rev_rates[INDEX(244)]);
  //sp 9
  shared_temp[threadIdx.x + 3 * blockDim.x] = -(fwd_rates[INDEX(245)] - rev_rates[INDEX(244)]);
  //sp 2
  sp_rates[INDEX(2)] += (fwd_rates[INDEX(245)] - rev_rates[INDEX(244)]);
  //sp 35
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(245)] - rev_rates[INDEX(244)]);

  //rxn 246
  //sp 46
  sp_rates[INDEX(46)] += (fwd_rates[INDEX(246)] - rev_rates[INDEX(245)]);
  //sp 9
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(246)] - rev_rates[INDEX(245)]);
  //sp 35
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(246)] - rev_rates[INDEX(245)]);
  //sp 1
  sp_rates[INDEX(1)] += (fwd_rates[INDEX(246)] - rev_rates[INDEX(245)]);

  //rxn 247
  //sp 16
  sp_rates[INDEX(16)] += (fwd_rates[INDEX(247)] - rev_rates[INDEX(246)]);
  //sp 9
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(247)] - rev_rates[INDEX(246)]);
  //sp 35
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(247)] - rev_rates[INDEX(246)]);
  //sp 30
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(247)] - rev_rates[INDEX(246)]);

  //rxn 248
  sp_rates[INDEX(40)] += shared_temp[threadIdx.x];
  //sp 1
  sp_rates[INDEX(1)] += (fwd_rates[INDEX(248)] - rev_rates[INDEX(247)]);
  //sp 10
  shared_temp[threadIdx.x] = -(fwd_rates[INDEX(248)] - rev_rates[INDEX(247)]);
  //sp 35
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(248)] - rev_rates[INDEX(247)]);
  //sp 45
  sp_rates[INDEX(45)] += (fwd_rates[INDEX(248)] - rev_rates[INDEX(247)]);

  //rxn 249
  //sp 40
  sp_rates[INDEX(40)] += (fwd_rates[INDEX(249)] - rev_rates[INDEX(248)]);
  //sp 10
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(249)] - rev_rates[INDEX(248)]);
  //sp 35
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(249)] - rev_rates[INDEX(248)]);
  //sp 4
  sp_rates[INDEX(4)] += (fwd_rates[INDEX(249)] - rev_rates[INDEX(248)]);

  //rxn 250
  //sp 1
  sp_rates[INDEX(1)] += (fwd_rates[INDEX(250)] - rev_rates[INDEX(249)]);
  //sp 10
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(250)] - rev_rates[INDEX(249)]);
  //sp 35
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(250)] - rev_rates[INDEX(249)]);
  //sp 43
  sp_rates[INDEX(43)] = (fwd_rates[INDEX(250)] - rev_rates[INDEX(249)]);

  //rxn 251
  sp_rates[INDEX(30)] += shared_temp[threadIdx.x + 2 * blockDim.x];
  //sp 35
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(251)] - rev_rates[INDEX(250)]);
  //sp 11
  shared_temp[threadIdx.x + 2 * blockDim.x] = -(fwd_rates[INDEX(251)] - rev_rates[INDEX(250)]);
  //sp 45
  sp_rates[INDEX(45)] += (fwd_rates[INDEX(251)] - rev_rates[INDEX(250)]);
  //sp 1
  sp_rates[INDEX(1)] += (fwd_rates[INDEX(251)] - rev_rates[INDEX(250)]);

  //rxn 252
  //sp 40
  sp_rates[INDEX(40)] += (fwd_rates[INDEX(252)] - rev_rates[INDEX(251)]);
  //sp 35
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(252)] - rev_rates[INDEX(251)]);
  //sp 11
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(252)] - rev_rates[INDEX(251)]);
  //sp 4
  sp_rates[INDEX(4)] += (fwd_rates[INDEX(252)] - rev_rates[INDEX(251)]);

  //rxn 253
  //sp 35
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(253)] - rev_rates[INDEX(252)]);
  //sp 11
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(253)] - rev_rates[INDEX(252)]);
  //sp 1
  sp_rates[INDEX(1)] += (fwd_rates[INDEX(253)] - rev_rates[INDEX(252)]);
  //sp 43
  sp_rates[INDEX(43)] += (fwd_rates[INDEX(253)] - rev_rates[INDEX(252)]);

  //rxn 254
  //sp 40
  sp_rates[INDEX(40)] += (fwd_rates[INDEX(254)] - rev_rates[INDEX(253)]);
  //sp 35
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(254)] - rev_rates[INDEX(253)]);
  //sp 12
  sp_rates[INDEX(12)] -= (fwd_rates[INDEX(254)] - rev_rates[INDEX(253)]);
  //sp 5
  sp_rates[INDEX(5)] += (fwd_rates[INDEX(254)] - rev_rates[INDEX(253)]);

  //rxn 255
  //sp 41
  sp_rates[INDEX(41)] += (fwd_rates[INDEX(255)] - rev_rates[INDEX(254)]);
  //sp 35
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(255)] - rev_rates[INDEX(254)]);
  //sp 12
  sp_rates[INDEX(12)] -= (fwd_rates[INDEX(255)] - rev_rates[INDEX(254)]);
  //sp 4
  sp_rates[INDEX(4)] += (fwd_rates[INDEX(255)] - rev_rates[INDEX(254)]);

  //rxn 256
  sp_rates[INDEX(9)] += shared_temp[threadIdx.x + 3 * blockDim.x];
  sp_rates[INDEX(10)] += shared_temp[threadIdx.x];
  //sp 1
  sp_rates[INDEX(1)] += (fwd_rates[INDEX(256)] - rev_rates[INDEX(255)]);
  //sp 2
  shared_temp[threadIdx.x] = -(fwd_rates[INDEX(256)] - rev_rates[INDEX(255)]);
  //sp 42
  shared_temp[threadIdx.x + 3 * blockDim.x] = -(fwd_rates[INDEX(256)] - rev_rates[INDEX(255)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(256)] - rev_rates[INDEX(255)]);
  //sp 47
  (*dy_N) += (fwd_rates[INDEX(256)] - rev_rates[INDEX(255)]);

  //rxn 257
  //sp 40
  sp_rates[INDEX(40)] += (fwd_rates[INDEX(257)] - rev_rates[INDEX(256)]);
  //sp 35
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(257)] - rev_rates[INDEX(256)]);
  //sp 42
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(257)] - rev_rates[INDEX(256)]);
  //sp 2
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(257)] - rev_rates[INDEX(256)]);

  //rxn 258
  sp_rates[INDEX(11)] += shared_temp[threadIdx.x + 2 * blockDim.x];
  //sp 2
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(258)] - rev_rates[INDEX(257)]);
  //sp 3
  sp_rates[INDEX(3)] -= (fwd_rates[INDEX(258)] - rev_rates[INDEX(257)]);
  //sp 42
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(258)] - rev_rates[INDEX(257)]);
  //sp 16
  sp_rates[INDEX(16)] += (fwd_rates[INDEX(258)] - rev_rates[INDEX(257)]);
  //sp 47
  shared_temp[threadIdx.x + 2 * blockDim.x] = (fwd_rates[INDEX(258)] - rev_rates[INDEX(257)]);

  //rxn 259
  //sp 1
  sp_rates[INDEX(1)] += (fwd_rates[INDEX(259)] - rev_rates[INDEX(258)]);
  //sp 4
  sp_rates[INDEX(4)] -= (fwd_rates[INDEX(259)] - rev_rates[INDEX(258)]);
  //sp 42
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(259)] - rev_rates[INDEX(258)]);
  //sp 16
  sp_rates[INDEX(16)] += (fwd_rates[INDEX(259)] - rev_rates[INDEX(258)]);
  //sp 47
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(259)] - rev_rates[INDEX(258)]);

  //rxn 260
  //sp 1
  sp_rates[INDEX(1)] -= (fwd_rates[INDEX(260)] - rev_rates[INDEX(259)]);
  //sp 42
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(260)] - rev_rates[INDEX(259)]);
  //sp 10
  sp_rates[INDEX(10)] += (fwd_rates[INDEX(260)] - rev_rates[INDEX(259)]);
  //sp 47
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(260)] - rev_rates[INDEX(259)]);

  //rxn 261
  sp_rates[INDEX(35)] += shared_temp[threadIdx.x + 1 * blockDim.x];
  //sp 2
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(261)] - rev_rates[INDEX(260)]);
  //sp 15
  sp_rates[INDEX(15)] += (fwd_rates[INDEX(261)] - rev_rates[INDEX(260)]);
  //sp 45
  shared_temp[threadIdx.x + 1 * blockDim.x] = -(fwd_rates[INDEX(261)] - rev_rates[INDEX(260)]);
  //sp 31
  sp_rates[INDEX(31)] += (fwd_rates[INDEX(261)] - rev_rates[INDEX(260)]);

  //rxn 262
  //sp 2
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(262)] - rev_rates[INDEX(261)]);
  //sp 45
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(262)] - rev_rates[INDEX(261)]);
  //sp 38
  sp_rates[INDEX(38)] += (fwd_rates[INDEX(262)] - rev_rates[INDEX(261)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(262)] - rev_rates[INDEX(261)]);

  //rxn 263
  //sp 2
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(263)] - rev_rates[INDEX(262)]);
  //sp 4
  sp_rates[INDEX(4)] += (fwd_rates[INDEX(263)] - rev_rates[INDEX(262)]);
  //sp 45
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(263)] - rev_rates[INDEX(262)]);
  //sp 46
  sp_rates[INDEX(46)] += (fwd_rates[INDEX(263)] - rev_rates[INDEX(262)]);

  //rxn 264
  //sp 32
  sp_rates[INDEX(32)] += (fwd_rates[INDEX(264)] - rev_rates[INDEX(263)]);
  //sp 1
  sp_rates[INDEX(1)] -= (fwd_rates[INDEX(264)] - rev_rates[INDEX(263)]);
  //sp 45
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(264)] - rev_rates[INDEX(263)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(264)] - rev_rates[INDEX(263)]);

  //rxn 265
  //sp 0
  sp_rates[INDEX(0)] += (fwd_rates[INDEX(265)] - rev_rates[INDEX(264)]);
  //sp 1
  sp_rates[INDEX(1)] -= (fwd_rates[INDEX(265)] - rev_rates[INDEX(264)]);
  //sp 45
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(265)] - rev_rates[INDEX(264)]);
  //sp 46
  sp_rates[INDEX(46)] += (fwd_rates[INDEX(265)] - rev_rates[INDEX(264)]);

  //rxn 266
  //sp 5
  sp_rates[INDEX(5)] += (fwd_rates[INDEX(266)] - rev_rates[INDEX(265)]);
  //sp 4
  sp_rates[INDEX(4)] -= (fwd_rates[INDEX(266)] - rev_rates[INDEX(265)]);
  //sp 45
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(266)] - rev_rates[INDEX(265)]);
  //sp 46
  sp_rates[INDEX(46)] += (fwd_rates[INDEX(266)] - rev_rates[INDEX(265)]);

  //rxn 267
  //sp 32
  sp_rates[INDEX(32)] += (fwd_rates[INDEX(267)] - rev_rates[INDEX(266)]);
  //sp 4
  sp_rates[INDEX(4)] -= (fwd_rates[INDEX(267)] - rev_rates[INDEX(266)]);
  //sp 45
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(267)] - rev_rates[INDEX(266)]);
  //sp 15
  sp_rates[INDEX(15)] += (fwd_rates[INDEX(267)] - rev_rates[INDEX(266)]);

  //rxn 268
  //sp 45
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(268)] - rev_rates[INDEX(267)]) * pres_mod[INDEX(35)];
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(268)] - rev_rates[INDEX(267)]) * pres_mod[INDEX(35)];
  //sp 31
  sp_rates[INDEX(31)] += (fwd_rates[INDEX(268)] - rev_rates[INDEX(267)]) * pres_mod[INDEX(35)];

  //rxn 269
  sp_rates[INDEX(42)] += shared_temp[threadIdx.x + 3 * blockDim.x];
  //sp 43
  shared_temp[threadIdx.x + 3 * blockDim.x] = -(fwd_rates[INDEX(269)] - rev_rates[INDEX(268)]);
  //sp 45
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(269)] - rev_rates[INDEX(268)]);

  //rxn 270
  //sp 40
  sp_rates[INDEX(40)] += (fwd_rates[INDEX(270)] - rev_rates[INDEX(269)]);
  //sp 1
  sp_rates[INDEX(1)] -= (fwd_rates[INDEX(270)] - rev_rates[INDEX(269)]);
  //sp 43
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(270)] - rev_rates[INDEX(269)]);
  //sp 4
  sp_rates[INDEX(4)] += (fwd_rates[INDEX(270)] - rev_rates[INDEX(269)]);

  //rxn 271
  //sp 32
  sp_rates[INDEX(32)] += (fwd_rates[INDEX(271)] - rev_rates[INDEX(270)]);
  //sp 1
  sp_rates[INDEX(1)] -= (fwd_rates[INDEX(271)] - rev_rates[INDEX(270)]);
  //sp 43
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(271)] - rev_rates[INDEX(270)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(271)] - rev_rates[INDEX(270)]);

  //rxn 272
  //sp 44
  sp_rates[INDEX(44)] -= (fwd_rates[INDEX(272)] - rev_rates[INDEX(271)]);
  //sp 45
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(272)] - rev_rates[INDEX(271)]);

  //rxn 273
  //sp 35
  sp_rates[INDEX(35)] -= (fwd_rates[INDEX(273)] - rev_rates[INDEX(272)]);
  //sp 27
  sp_rates[INDEX(27)] -= (fwd_rates[INDEX(273)] - rev_rates[INDEX(272)]);
  //sp 43
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(273)] - rev_rates[INDEX(272)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(273)] - rev_rates[INDEX(272)]);

  //rxn 274
  //sp 41
  sp_rates[INDEX(41)] += (fwd_rates[INDEX(274)] - rev_rates[INDEX(273)]);
  //sp 12
  sp_rates[INDEX(12)] -= (fwd_rates[INDEX(274)] - rev_rates[INDEX(273)]);
  //sp 30
  sp_rates[INDEX(30)] -= (fwd_rates[INDEX(274)] - rev_rates[INDEX(273)]);
  //sp 1
  sp_rates[INDEX(1)] += (fwd_rates[INDEX(274)] - rev_rates[INDEX(273)]);

  //rxn 275
  //sp 40
  sp_rates[INDEX(40)] += (fwd_rates[INDEX(275)] - rev_rates[INDEX(274)]);
  //sp 0
  sp_rates[INDEX(0)] += (fwd_rates[INDEX(275)] - rev_rates[INDEX(274)]);
  //sp 12
  sp_rates[INDEX(12)] -= (fwd_rates[INDEX(275)] - rev_rates[INDEX(274)]);
  //sp 30
  sp_rates[INDEX(30)] -= (fwd_rates[INDEX(275)] - rev_rates[INDEX(274)]);

  //rxn 276
  (*dy_N) += shared_temp[threadIdx.x + 2 * blockDim.x];
  sp_rates[INDEX(2)] += shared_temp[threadIdx.x];
  //sp 32
  shared_temp[threadIdx.x + 2 * blockDim.x] = (fwd_rates[INDEX(276)] - rev_rates[INDEX(275)]);
  //sp 33
  shared_temp[threadIdx.x] = -(fwd_rates[INDEX(276)] - rev_rates[INDEX(275)]);
  //sp 1
  sp_rates[INDEX(1)] -= (fwd_rates[INDEX(276)] - rev_rates[INDEX(275)]);
  //sp 0
  sp_rates[INDEX(0)] += (fwd_rates[INDEX(276)] - rev_rates[INDEX(275)]);

  //rxn 277
  //sp 32
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(277)] - rev_rates[INDEX(276)]);
  //sp 33
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(277)] - rev_rates[INDEX(276)]);
  //sp 4
  sp_rates[INDEX(4)] -= (fwd_rates[INDEX(277)] - rev_rates[INDEX(276)]);
  //sp 5
  sp_rates[INDEX(5)] += (fwd_rates[INDEX(277)] - rev_rates[INDEX(276)]);

  //rxn 278
  //sp 32
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(278)] - rev_rates[INDEX(277)]);
  //sp 33
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(278)] - rev_rates[INDEX(277)]);
  //sp 2
  sp_rates[INDEX(2)] -= (fwd_rates[INDEX(278)] - rev_rates[INDEX(277)]);
  //sp 4
  sp_rates[INDEX(4)] += (fwd_rates[INDEX(278)] - rev_rates[INDEX(277)]);

  //rxn 279
  //sp 15
  sp_rates[INDEX(15)] -= (fwd_rates[INDEX(279)] - rev_rates[INDEX(278)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(279)] - rev_rates[INDEX(278)]);
  //sp 38
  sp_rates[INDEX(38)] += (fwd_rates[INDEX(279)] - rev_rates[INDEX(278)]);
  //sp 31
  sp_rates[INDEX(31)] -= (fwd_rates[INDEX(279)] - rev_rates[INDEX(278)]);

  //rxn 280
  //sp 35
  sp_rates[INDEX(35)] += (fwd_rates[INDEX(280)] - rev_rates[INDEX(279)]);
  //sp 36
  sp_rates[INDEX(36)] -= (fwd_rates[INDEX(280)] - rev_rates[INDEX(279)]);
  //sp 46
  sp_rates[INDEX(46)] += (fwd_rates[INDEX(280)] - rev_rates[INDEX(279)]);
  //sp 39
  sp_rates[INDEX(39)] -= (fwd_rates[INDEX(280)] - rev_rates[INDEX(279)]);

  //rxn 281
  //sp 36
  sp_rates[INDEX(36)] -= (fwd_rates[INDEX(281)] - rev_rates[INDEX(280)]);
  //sp 37
  sp_rates[INDEX(37)] += (fwd_rates[INDEX(281)] - rev_rates[INDEX(280)]);
  //sp 46
  sp_rates[INDEX(46)] -= (fwd_rates[INDEX(281)] - rev_rates[INDEX(280)]);
  //sp 15
  sp_rates[INDEX(15)] += (fwd_rates[INDEX(281)] - rev_rates[INDEX(280)]);

  //rxn 282
  //sp 35
  sp_rates[INDEX(35)] += (fwd_rates[INDEX(282)] - rev_rates[INDEX(281)]);
  //sp 14
  sp_rates[INDEX(14)] += (fwd_rates[INDEX(282)] - rev_rates[INDEX(281)]);
  //sp 30
  sp_rates[INDEX(30)] -= (fwd_rates[INDEX(282)] - rev_rates[INDEX(281)]);
  //sp 15
  sp_rates[INDEX(15)] -= (fwd_rates[INDEX(282)] - rev_rates[INDEX(281)]);

  //rxn 283
  sp_rates[INDEX(45)] += shared_temp[threadIdx.x + 1 * blockDim.x];
  sp_rates[INDEX(43)] += shared_temp[threadIdx.x + 3 * blockDim.x];
  //sp 0
  sp_rates[INDEX(0)] += fwd_rates[INDEX(283)];
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] = fwd_rates[INDEX(283)];
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] = -fwd_rates[INDEX(283)];
  //sp 12
  sp_rates[INDEX(12)] -= fwd_rates[INDEX(283)];
  //sp 14
  sp_rates[INDEX(14)] += fwd_rates[INDEX(283)];

  //rxn 284
  //sp 24
  sp_rates[INDEX(24)] -= (fwd_rates[INDEX(284)] - rev_rates[INDEX(282)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(284)] - rev_rates[INDEX(282)]);
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(284)] - rev_rates[INDEX(282)]);
  //sp 51
  sp_rates[INDEX(50)] = (fwd_rates[INDEX(284)] - rev_rates[INDEX(282)]);

  //rxn 285
  //sp 25
  sp_rates[INDEX(25)] -= (fwd_rates[INDEX(285)] - rev_rates[INDEX(283)]);
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(285)] - rev_rates[INDEX(283)]);
  //sp 52
  sp_rates[INDEX(51)] = (fwd_rates[INDEX(285)] - rev_rates[INDEX(283)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] += (fwd_rates[INDEX(285)] - rev_rates[INDEX(283)]);

  //rxn 286
  //sp 3
  sp_rates[INDEX(3)] += (fwd_rates[INDEX(286)] - rev_rates[INDEX(284)]);
  //sp 4
  sp_rates[INDEX(4)] -= (fwd_rates[INDEX(286)] - rev_rates[INDEX(284)]);
  //sp 5
  sp_rates[INDEX(5)] += (fwd_rates[INDEX(286)] - rev_rates[INDEX(284)]);
  //sp 6
  sp_rates[INDEX(6)] -= (fwd_rates[INDEX(286)] - rev_rates[INDEX(284)]);

  //rxn 287
  //sp 0
  sp_rates[INDEX(0)] += fwd_rates[INDEX(287)];
  //sp 17
  sp_rates[INDEX(17)] += fwd_rates[INDEX(287)];
  //sp 4
  sp_rates[INDEX(4)] -= fwd_rates[INDEX(287)];
  //sp 12
  sp_rates[INDEX(12)] -= fwd_rates[INDEX(287)];

  //rxn 288
  //sp 0
  sp_rates[INDEX(0)] -= (fwd_rates[INDEX(288)] - rev_rates[INDEX(285)]) * pres_mod[INDEX(36)];
  //sp 9
  sp_rates[INDEX(9)] -= (fwd_rates[INDEX(288)] - rev_rates[INDEX(285)]) * pres_mod[INDEX(36)];
  //sp 12
  sp_rates[INDEX(12)] += (fwd_rates[INDEX(288)] - rev_rates[INDEX(285)]) * pres_mod[INDEX(36)];

  //rxn 289
  sp_rates[INDEX(32)] += shared_temp[threadIdx.x + 2 * blockDim.x];
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] += 2.0 * fwd_rates[INDEX(289)];
  //sp 10
  shared_temp[threadIdx.x + 2 * blockDim.x] = -fwd_rates[INDEX(289)];
  //sp 3
  sp_rates[INDEX(3)] -= fwd_rates[INDEX(289)];
  //sp 15
  sp_rates[INDEX(15)] += fwd_rates[INDEX(289)];

  //rxn 290
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(290)] - rev_rates[INDEX(286)]);
  //sp 17
  sp_rates[INDEX(17)] += (fwd_rates[INDEX(290)] - rev_rates[INDEX(286)]);
  //sp 10
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(290)] - rev_rates[INDEX(286)]);
  //sp 3
  sp_rates[INDEX(3)] -= (fwd_rates[INDEX(290)] - rev_rates[INDEX(286)]);

  //rxn 291
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] += 2.0 * fwd_rates[INDEX(291)];
  //sp 10
  shared_temp[threadIdx.x + 2 * blockDim.x] -= 2.0 * fwd_rates[INDEX(291)];
  //sp 22
  sp_rates[INDEX(22)] += fwd_rates[INDEX(291)];

  //rxn 292
  //sp 0
  sp_rates[INDEX(0)] += fwd_rates[INDEX(292)];
  //sp 17
  sp_rates[INDEX(17)] += fwd_rates[INDEX(292)];
  //sp 11
  sp_rates[INDEX(11)] -= fwd_rates[INDEX(292)];
  //sp 5
  sp_rates[INDEX(5)] -= fwd_rates[INDEX(292)];

  //rxn 293
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] += (fwd_rates[INDEX(293)] - rev_rates[INDEX(287)]);
  //sp 3
  sp_rates[INDEX(3)] -= (fwd_rates[INDEX(293)] - rev_rates[INDEX(287)]);
  //sp 51
  sp_rates[INDEX(50)] += (fwd_rates[INDEX(293)] - rev_rates[INDEX(287)]);
  //sp 23
  sp_rates[INDEX(23)] -= (fwd_rates[INDEX(293)] - rev_rates[INDEX(287)]);

  //rxn 294
  //sp 3
  sp_rates[INDEX(3)] -= (fwd_rates[INDEX(294)] - rev_rates[INDEX(288)]);
  //sp 22
  sp_rates[INDEX(22)] += (fwd_rates[INDEX(294)] - rev_rates[INDEX(288)]);
  //sp 6
  sp_rates[INDEX(6)] += (fwd_rates[INDEX(294)] - rev_rates[INDEX(288)]);
  //sp 23
  sp_rates[INDEX(23)] -= (fwd_rates[INDEX(294)] - rev_rates[INDEX(288)]);

  //rxn 295
  sp_rates[INDEX(33)] = shared_temp[threadIdx.x];
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(295)] - rev_rates[INDEX(289)]);
  //sp 52
  shared_temp[threadIdx.x] = -(fwd_rates[INDEX(295)] - rev_rates[INDEX(289)]);
  //sp 4
  sp_rates[INDEX(4)] += (fwd_rates[INDEX(295)] - rev_rates[INDEX(289)]);
  //sp 51
  sp_rates[INDEX(50)] += (fwd_rates[INDEX(295)] - rev_rates[INDEX(289)]);

  //rxn 296
  //sp 2
  shared_temp[threadIdx.x + 3 * blockDim.x] -= fwd_rates[INDEX(296)];
  //sp 4
  sp_rates[INDEX(4)] += fwd_rates[INDEX(296)];
  //sp 12
  sp_rates[INDEX(12)] += fwd_rates[INDEX(296)];
  //sp 14
  sp_rates[INDEX(14)] += fwd_rates[INDEX(296)];
  //sp 52
  shared_temp[threadIdx.x] -= fwd_rates[INDEX(296)];

  //rxn 297
  //sp 3
  sp_rates[INDEX(3)] -= fwd_rates[INDEX(297)];
  //sp 6
  sp_rates[INDEX(6)] += fwd_rates[INDEX(297)];
  //sp 12
  sp_rates[INDEX(12)] += fwd_rates[INDEX(297)];
  //sp 14
  sp_rates[INDEX(14)] += fwd_rates[INDEX(297)];
  //sp 52
  shared_temp[threadIdx.x] -= fwd_rates[INDEX(297)];

  //rxn 298
  //sp 0
  sp_rates[INDEX(0)] += (fwd_rates[INDEX(298)] - rev_rates[INDEX(290)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(298)] - rev_rates[INDEX(290)]);
  //sp 51
  sp_rates[INDEX(50)] += (fwd_rates[INDEX(298)] - rev_rates[INDEX(290)]);
  //sp 52
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(298)] - rev_rates[INDEX(290)]);

  //rxn 299
  sp_rates[INDEX(10)] += shared_temp[threadIdx.x + 2 * blockDim.x];
  sp_rates[INDEX(2)] += shared_temp[threadIdx.x + 3 * blockDim.x];
  //sp 0
  sp_rates[INDEX(0)] += fwd_rates[INDEX(299)];
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= fwd_rates[INDEX(299)];
  //sp 12
  shared_temp[threadIdx.x + 3 * blockDim.x] = fwd_rates[INDEX(299)];
  //sp 14
  shared_temp[threadIdx.x + 2 * blockDim.x] = fwd_rates[INDEX(299)];
  //sp 52
  shared_temp[threadIdx.x] -= fwd_rates[INDEX(299)];

  //rxn 300
  //sp 4
  sp_rates[INDEX(4)] -= fwd_rates[INDEX(300)];
  //sp 5
  sp_rates[INDEX(5)] += fwd_rates[INDEX(300)];
  //sp 12
  shared_temp[threadIdx.x + 3 * blockDim.x] += fwd_rates[INDEX(300)];
  //sp 14
  shared_temp[threadIdx.x + 2 * blockDim.x] += fwd_rates[INDEX(300)];
  //sp 52
  shared_temp[threadIdx.x] -= fwd_rates[INDEX(300)];

  //rxn 301
  //sp 6
  sp_rates[INDEX(6)] -= fwd_rates[INDEX(301)];
  //sp 7
  sp_rates[INDEX(7)] += fwd_rates[INDEX(301)];
  //sp 12
  shared_temp[threadIdx.x + 3 * blockDim.x] += fwd_rates[INDEX(301)];
  //sp 14
  shared_temp[threadIdx.x + 2 * blockDim.x] += fwd_rates[INDEX(301)];
  //sp 52
  shared_temp[threadIdx.x] -= fwd_rates[INDEX(301)];

  //rxn 302
  //sp 52
  shared_temp[threadIdx.x] -= fwd_rates[INDEX(302)];
  //sp 13
  sp_rates[INDEX(13)] += fwd_rates[INDEX(302)];
  //sp 14
  shared_temp[threadIdx.x + 2 * blockDim.x] += fwd_rates[INDEX(302)];

  //rxn 303
  sp_rates[INDEX(12)] += shared_temp[threadIdx.x + 3 * blockDim.x];
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(303)] - rev_rates[INDEX(291)]) * pres_mod[INDEX(37)];
  //sp 51
  shared_temp[threadIdx.x + 3 * blockDim.x] = (fwd_rates[INDEX(303)] - rev_rates[INDEX(291)]) * pres_mod[INDEX(37)];
  //sp 28
  sp_rates[INDEX(28)] -= (fwd_rates[INDEX(303)] - rev_rates[INDEX(291)]) * pres_mod[INDEX(37)];

  //rxn 304
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] += fwd_rates[INDEX(304)];
  //sp 2
  sp_rates[INDEX(2)] -= fwd_rates[INDEX(304)];
  //sp 10
  sp_rates[INDEX(10)] += fwd_rates[INDEX(304)];
  //sp 15
  sp_rates[INDEX(15)] += fwd_rates[INDEX(304)];
  //sp 51
  shared_temp[threadIdx.x + 3 * blockDim.x] -= fwd_rates[INDEX(304)];

  //rxn 305
  //sp 3
  sp_rates[INDEX(3)] -= fwd_rates[INDEX(305)];
  //sp 4
  sp_rates[INDEX(4)] += fwd_rates[INDEX(305)];
  //sp 14
  shared_temp[threadIdx.x + 2 * blockDim.x] += fwd_rates[INDEX(305)];
  //sp 17
  sp_rates[INDEX(17)] += fwd_rates[INDEX(305)];
  //sp 51
  shared_temp[threadIdx.x + 3 * blockDim.x] -= fwd_rates[INDEX(305)];

  //rxn 306
  //sp 16
  sp_rates[INDEX(16)] += 2.0 * fwd_rates[INDEX(306)];
  //sp 51
  shared_temp[threadIdx.x + 3 * blockDim.x] -= fwd_rates[INDEX(306)];
  //sp 3
  sp_rates[INDEX(3)] -= fwd_rates[INDEX(306)];
  //sp 4
  sp_rates[INDEX(4)] += fwd_rates[INDEX(306)];

  //rxn 307
  //sp 16
  sp_rates[INDEX(16)] += (fwd_rates[INDEX(307)] - rev_rates[INDEX(292)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(307)] - rev_rates[INDEX(292)]);
  //sp 51
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(307)] - rev_rates[INDEX(292)]);
  //sp 12
  sp_rates[INDEX(12)] += (fwd_rates[INDEX(307)] - rev_rates[INDEX(292)]);

  //rxn 308
  //sp 0
  sp_rates[INDEX(0)] += (fwd_rates[INDEX(308)] - rev_rates[INDEX(293)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(308)] - rev_rates[INDEX(293)]);
  //sp 51
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(308)] - rev_rates[INDEX(293)]);
  //sp 28
  sp_rates[INDEX(28)] += (fwd_rates[INDEX(308)] - rev_rates[INDEX(293)]);

  //rxn 309
  //sp 51
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(309)] - rev_rates[INDEX(294)]);
  //sp 28
  sp_rates[INDEX(28)] += (fwd_rates[INDEX(309)] - rev_rates[INDEX(294)]);
  //sp 4
  sp_rates[INDEX(4)] -= (fwd_rates[INDEX(309)] - rev_rates[INDEX(294)]);
  //sp 5
  sp_rates[INDEX(5)] += (fwd_rates[INDEX(309)] - rev_rates[INDEX(294)]);

  //rxn 310
  //sp 16
  sp_rates[INDEX(16)] += (fwd_rates[INDEX(310)] - rev_rates[INDEX(295)]);
  //sp 51
  shared_temp[threadIdx.x + 3 * blockDim.x] -= (fwd_rates[INDEX(310)] - rev_rates[INDEX(295)]);
  //sp 18
  sp_rates[INDEX(18)] += (fwd_rates[INDEX(310)] - rev_rates[INDEX(295)]);
  //sp 4
  sp_rates[INDEX(4)] -= (fwd_rates[INDEX(310)] - rev_rates[INDEX(295)]);

  //rxn 311
  sp_rates[INDEX(51)] += shared_temp[threadIdx.x];
  //sp 25
  sp_rates[INDEX(25)] -= (fwd_rates[INDEX(311)] - rev_rates[INDEX(296)]) * pres_mod[INDEX(38)];
  //sp 12
  sp_rates[INDEX(12)] -= (fwd_rates[INDEX(311)] - rev_rates[INDEX(296)]) * pres_mod[INDEX(38)];
  //sp 50
  shared_temp[threadIdx.x] = (fwd_rates[INDEX(311)] - rev_rates[INDEX(296)]) * pres_mod[INDEX(38)];

  //rxn 312
  sp_rates[INDEX(14)] += shared_temp[threadIdx.x + 2 * blockDim.x];
  //sp 49
  shared_temp[threadIdx.x + 2 * blockDim.x] = (fwd_rates[INDEX(312)] - rev_rates[INDEX(297)]);
  //sp 50
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(312)] - rev_rates[INDEX(297)]);
  //sp 2
  sp_rates[INDEX(2)] -= (fwd_rates[INDEX(312)] - rev_rates[INDEX(297)]);
  //sp 4
  sp_rates[INDEX(4)] += (fwd_rates[INDEX(312)] - rev_rates[INDEX(297)]);

  //rxn 313
  //sp 49
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(313)] - rev_rates[INDEX(298)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(313)] - rev_rates[INDEX(298)]);
  //sp 0
  sp_rates[INDEX(0)] += (fwd_rates[INDEX(313)] - rev_rates[INDEX(298)]);
  //sp 50
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(313)] - rev_rates[INDEX(298)]);

  //rxn 314
  //sp 49
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(314)] - rev_rates[INDEX(299)]);
  //sp 50
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(314)] - rev_rates[INDEX(299)]);
  //sp 4
  sp_rates[INDEX(4)] -= (fwd_rates[INDEX(314)] - rev_rates[INDEX(299)]);
  //sp 5
  sp_rates[INDEX(5)] += (fwd_rates[INDEX(314)] - rev_rates[INDEX(299)]);

  //rxn 315
  //sp 49
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(315)] - rev_rates[INDEX(300)]);
  //sp 50
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(315)] - rev_rates[INDEX(300)]);
  //sp 6
  sp_rates[INDEX(6)] += (fwd_rates[INDEX(315)] - rev_rates[INDEX(300)]);
  //sp 7
  sp_rates[INDEX(7)] -= (fwd_rates[INDEX(315)] - rev_rates[INDEX(300)]);

  //rxn 316
  //sp 49
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(316)] - rev_rates[INDEX(301)]);
  //sp 50
  shared_temp[threadIdx.x] -= (fwd_rates[INDEX(316)] - rev_rates[INDEX(301)]);
  //sp 12
  sp_rates[INDEX(12)] -= (fwd_rates[INDEX(316)] - rev_rates[INDEX(301)]);
  //sp 13
  sp_rates[INDEX(13)] += (fwd_rates[INDEX(316)] - rev_rates[INDEX(301)]);

  //rxn 317
  //sp 24
  sp_rates[INDEX(24)] -= (fwd_rates[INDEX(317)] - rev_rates[INDEX(302)]) * pres_mod[INDEX(39)];
  //sp 49
  shared_temp[threadIdx.x + 2 * blockDim.x] += (fwd_rates[INDEX(317)] - rev_rates[INDEX(302)]) * pres_mod[INDEX(39)];
  //sp 12
  sp_rates[INDEX(12)] -= (fwd_rates[INDEX(317)] - rev_rates[INDEX(302)]) * pres_mod[INDEX(39)];

  //rxn 318
  //sp 49
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(318)] - rev_rates[INDEX(303)]);
  //sp 25
  sp_rates[INDEX(25)] += (fwd_rates[INDEX(318)] - rev_rates[INDEX(303)]);
  //sp 2
  sp_rates[INDEX(2)] -= (fwd_rates[INDEX(318)] - rev_rates[INDEX(303)]);
  //sp 17
  sp_rates[INDEX(17)] += (fwd_rates[INDEX(318)] - rev_rates[INDEX(303)]);

  //rxn 319
  //sp 49
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(319)] - rev_rates[INDEX(304)]) * pres_mod[INDEX(40)];
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(319)] - rev_rates[INDEX(304)]) * pres_mod[INDEX(40)];
  //sp 50
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(319)] - rev_rates[INDEX(304)]) * pres_mod[INDEX(40)];

  //rxn 320
  //sp 49
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(320)] - rev_rates[INDEX(305)]);
  //sp 1
  shared_temp[threadIdx.x + 1 * blockDim.x] -= (fwd_rates[INDEX(320)] - rev_rates[INDEX(305)]);
  //sp 12
  sp_rates[INDEX(12)] += (fwd_rates[INDEX(320)] - rev_rates[INDEX(305)]);
  //sp 25
  sp_rates[INDEX(25)] += (fwd_rates[INDEX(320)] - rev_rates[INDEX(305)]);

  //rxn 321
  //sp 49
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(321)] - rev_rates[INDEX(306)]);
  //sp 25
  sp_rates[INDEX(25)] += (fwd_rates[INDEX(321)] - rev_rates[INDEX(306)]);
  //sp 18
  sp_rates[INDEX(18)] += (fwd_rates[INDEX(321)] - rev_rates[INDEX(306)]);
  //sp 4
  sp_rates[INDEX(4)] -= (fwd_rates[INDEX(321)] - rev_rates[INDEX(306)]);

  //rxn 322
  //sp 49
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(322)] - rev_rates[INDEX(307)]);
  //sp 50
  shared_temp[threadIdx.x] += (fwd_rates[INDEX(322)] - rev_rates[INDEX(307)]);
  //sp 3
  sp_rates[INDEX(3)] += (fwd_rates[INDEX(322)] - rev_rates[INDEX(307)]);
  //sp 6
  sp_rates[INDEX(6)] -= (fwd_rates[INDEX(322)] - rev_rates[INDEX(307)]);

  //rxn 323
  //sp 4
  sp_rates[INDEX(4)] += fwd_rates[INDEX(323)];
  //sp 6
  sp_rates[INDEX(6)] -= fwd_rates[INDEX(323)];
  //sp 49
  shared_temp[threadIdx.x + 2 * blockDim.x] -= fwd_rates[INDEX(323)];
  //sp 17
  sp_rates[INDEX(17)] += fwd_rates[INDEX(323)];
  //sp 25
  sp_rates[INDEX(25)] += fwd_rates[INDEX(323)];

  //rxn 324
  //sp 49
  shared_temp[threadIdx.x + 2 * blockDim.x] -= (fwd_rates[INDEX(324)] - rev_rates[INDEX(308)]);
  //sp 25
  sp_rates[INDEX(25)] += 2.0 * (fwd_rates[INDEX(324)] - rev_rates[INDEX(308)]);
  //sp 12
  sp_rates[INDEX(12)] -= (fwd_rates[INDEX(324)] - rev_rates[INDEX(308)]);

  //sp 48
  sp_rates[INDEX(47)] = 0.0;
  sp_rates[INDEX(1)] += shared_temp[threadIdx.x + 1 * blockDim.x];
  sp_rates[INDEX(50)] += shared_temp[threadIdx.x + 3 * blockDim.x];
  sp_rates[INDEX(49)] = shared_temp[threadIdx.x];
  sp_rates[INDEX(48)] = shared_temp[threadIdx.x + 2 * blockDim.x];
} // end eval_spec_rates

