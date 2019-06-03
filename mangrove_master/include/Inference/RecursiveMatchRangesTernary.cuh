#pragma once

#include "Inference/ResultCollector.hpp"
#include "TemplateConf.cuh"

namespace mangrove {

    template<int F_INDEX>
    struct RecursiveMatchRangesTernary {

        // check: left op inv(right_1,right_2)
        inline static bool ApplyT (ResultCollector<numeric_t> & results, int left, int right_1, int right_2) {

            using invariant = typename GET_INVARIANT<F_INDEX-1, NTtemplate>::type;
            using cmp = typename GET_INVARIANT<F_INDEX-1, NTtemplate>::type::first;

            int *monotony = results.getTernaryMonotony();
            num2_t *unaryResult = static_cast<num2_t *>(results.getVectorResult<1>());

            if (monotony[F_INDEX-1] == MonotonyProperty::INCREASING ||
                monotony[F_INDEX-1] == MonotonyProperty::DECREASING) {

                numeric_t f_max_second = APPLY_FUN<1, invariant>::eval(unaryResult[right_1][1],
                                                                       unaryResult[right_2][1]);

                numeric_t f_min_second = APPLY_FUN<1, invariant>::eval(unaryResult[right_1][0],
                                                                       unaryResult[right_2][0]);

                if (!(
                      (std::is_same<cmp, equal<numeric_t>>::value &&
                          (unaryResult[left][0] < std::min(f_min_second, f_max_second) ||
                           unaryResult[left][1] > std::max(f_min_second, f_max_second)) )
                      ||
                      ((std::is_same<cmp, less<numeric_t>>::value  ||
                        std::is_same<cmp, notEqual<numeric_t>>::value) &&
                          (unaryResult[left][1] < std::min(f_min_second, f_max_second) ||
                           unaryResult[left][0] > std::max(f_min_second, f_max_second)) )
                      ||
                      (std::is_same<cmp, lessEq<numeric_t>>::value &&
                           (unaryResult[left][1] <= std::min(f_min_second, f_max_second) ||
                            unaryResult[left][0] >= std::max(f_min_second, f_max_second)) )
                    ))
                    return true;
            }
            else
            { return true; }

            return RecursiveMatchRangesTernary<F_INDEX-1>::ApplyT(results, left, right_1,  right_2);
        }
    };

    template<>
    struct RecursiveMatchRangesTernary<0> {
        inline static bool ApplyT (ResultCollector<numeric_t> &, int , int, int) { return false; }
    };
}
