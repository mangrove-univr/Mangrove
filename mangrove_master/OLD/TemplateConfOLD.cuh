#pragma once
#include "UserFunctions.cuh"
#include "config.cuh"

// templates checked by inference
using BooleanBinaryInv1 = BinaryBoolInvariants < equal<unsigned> >;
using BooleanBinaryInv2 = BinaryBoolInvariants < notEqualB<unsigned> >;

using BT_Inv1  = TernaryBoolInvariants < equal<unsigned>, AND<unsigned>       >;
using BT_Inv2  = TernaryBoolInvariants < equal<unsigned>, IMPLY<unsigned>     >;
using BT_Inv3  = TernaryBoolInvariants < equal<unsigned>, R_IMPLY<unsigned>   >;
using BT_Inv4  = TernaryBoolInvariants < equal<unsigned>, XOR<unsigned>       >;
using BT_Inv5  = TernaryBoolInvariants < equal<unsigned>, OR<unsigned>        >;
using BT_Inv6  = TernaryBoolInvariants < equal<unsigned>, NOR<unsigned>       >;
using BT_Inv7  = TernaryBoolInvariants < equal<unsigned>, XNOR<unsigned>      >;
using BT_Inv8  = TernaryBoolInvariants < equal<unsigned>, NOT_R_IMPLY<unsigned>>;
using BT_Inv9  = TernaryBoolInvariants < equal<unsigned>, NOT_IMPLY<unsigned> >;
using BT_Inv10 = TernaryBoolInvariants < equal<unsigned>, NAND<unsigned>      >;

using TernaryBoolTemplate = Template<BT_Inv1, BT_Inv2, BT_Inv3, BT_Inv4,
                                     BT_Inv5, BT_Inv6, BT_Inv7, BT_Inv8,
                                     BT_Inv9, BT_Inv10>;

//==============================================================================

// templates checked by inference)
using NB_Inv1 = BinaryNumericInvariants < equal<numeric_t> >;       // 1
using NB_Inv2 = BinaryNumericInvariants < notEqualN<numeric_t> >;   // 2
using NB_Inv3 = BinaryNumericInvariants < less<numeric_t> >;        // 3
using NB_Inv4 = BinaryNumericInvariants < lessEq<numeric_t> >;      // 4

// templates not checked by inference)
using NB_Inv5 = BinaryNumericInvariants < lessSqrt<numeric_t> >;    // 5
using NB_Inv6 = BinaryNumericInvariants < equalLog<numeric_t> >;    // 6
using NB_Inv7 = BinaryNumericInvariants < lessSucc<numeric_t> >;    // 7
using NB_Inv8 = BinaryNumericInvariants < equalTwice<numeric_t> >;  // 8

using BinaryNumericTemplate = Template<NB_Inv1, NB_Inv2, NB_Inv3,
                                       NB_Inv4, NB_Inv5, NB_Inv6,
                                       NB_Inv7, NB_Inv8>;

using NT_Inv1 = TernaryNumericInvariants < equal<numeric_t>, min<numeric_t> >;  // 1
using NT_Inv2 = TernaryNumericInvariants < equal<numeric_t>, max<numeric_t> >;  // 2
using NT_Inv3 = TernaryNumericInvariants < less<numeric_t>,  mul<numeric_t> >;  // 3
using NT_Inv4 = TernaryNumericInvariants < equal<numeric_t>, exp<numeric_t> >;  // 4
using NT_Inv5 = TernaryNumericInvariants < lessEq<numeric_t>, add<numeric_t> >; // 5

using TernaryNumericTemplate = Template<NT_Inv1, NT_Inv2, NT_Inv3,
                                        NT_Inv4, NT_Inv5>;

//------------------------------------------------------------------------------

using TemplateSET = TemplateSETstr<TernaryBoolTemplate, TernaryNumericTemplate,
                                   BinaryNumericTemplate>;
