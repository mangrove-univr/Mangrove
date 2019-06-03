#pragma once

#include "Inference/ResultCollector.hpp"
#include "TemplateConf.cuh"

namespace mangrove {

    template<int F_INDEX>
    struct RecursiveMatchRanges {

        // check: first op inv(second)
        inline static bool ApplyT (ResultCollector<numeric_t> & results, int first, int second) {

            using function = typename GET_INVARIANT<F_INDEX-1, NBtemplate>::type::unary;
            using cmp = typename GET_INVARIANT<F_INDEX-1, NBtemplate>::type::first;

            int *monotony = results.getMonotony();
            num2_t *unaryResult = static_cast<num2_t *>(results.getVectorResult<1>());

            if (monotony[F_INDEX-1] == MonotonyProperty::INCREASING ||
                monotony[F_INDEX-1] == MonotonyProperty::DECREASING) {

                numeric_t f_max_second = function()(unaryResult[second][1]);
                numeric_t f_min_second = function()(unaryResult[second][0]);

                if (!(
                      ((unaryResult[first][0] == unaryResult[first][1] &&
                        unaryResult[first][0] == f_min_second &&
                        f_min_second == f_max_second) )
                      ||
                      (std::is_same<cmp, equal<numeric_t>>::value &&
                          (unaryResult[first][0] != std::min(f_min_second, f_max_second) ||
                           unaryResult[first][1] != std::max(f_min_second, f_max_second)) )
                      ||
                      ((std::is_same<cmp, less<numeric_t>>::value  ||
                        std::is_same<cmp, notEqual<numeric_t>>::value) &&
                          (unaryResult[first][1] < std::min(f_min_second, f_max_second) ||
                           unaryResult[first][0] > std::max(f_min_second, f_max_second)) )
                      ||
                      (std::is_same<cmp, lessEq<numeric_t>>::value &&
                           (unaryResult[first][1] <= std::min(f_min_second, f_max_second) ||
                            unaryResult[first][0] >= std::max(f_min_second, f_max_second)) )
                    ))
                    return true;
            }
            else {
                return true;
            }

            return RecursiveMatchRanges<F_INDEX-1>::ApplyT(results, first, second);
        }
    };

    template<>
    struct RecursiveMatchRanges<0> {
        inline static bool ApplyT (ResultCollector<numeric_t> &, int , int) { return false; }
    };
}
