/*
#pragma once
#include "UserFunctions.cuh"

using BB_Inv1 = BinaryBoolInvariants < equal<int> >;
using BB_Inv2 = BinaryBoolInvariants < notEqual<int> >;

using BT_Inv1  = TernaryBoolInvariants < equal<int>, AND<int> >;
using BT_Inv2  = TernaryBoolInvariants < equal<int>, XOR<int> >;
using BT_Inv3  = TernaryBoolInvariants < equal<int>, OR<int>  >;
//------------------------------------------------------------------------------

using BinaryBoolTemplate = Template<BB_Inv1, BB_Inv2>;
using TernaryBoolTemplate = Template<BT_Inv1, BT_Inv2, BT_Inv3>;

using TemplateSET = TemplateSETstr<BinaryBoolTemplate, TernaryBoolTemplate>;
*/
