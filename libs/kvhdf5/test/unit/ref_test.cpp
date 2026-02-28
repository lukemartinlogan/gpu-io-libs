#include <catch2/catch_test_macros.hpp>
#include "kvhdf5/ref.h"

using namespace kvhdf5;

TEST_CASE("Ref construction from reference", "[ref]") {
    int x = 42;
    Ref<int> r(x);
    REQUIRE(r.get() == &x);
    REQUIRE(*r == 42);
}

TEST_CASE("Ref arrow operator", "[ref]") {
    struct S { int val; };
    S s{99};
    Ref<S> r(s);
    REQUIRE(r->val == 99);
}

TEST_CASE("Ref reflects mutations", "[ref]") {
    int x = 1;
    Ref<int> r(x);
    x = 2;
    REQUIRE(*r == 2);
}

TEST_CASE("Ref const access", "[ref]") {
    const int x = 10;
    Ref<const int> r(x);
    REQUIRE(*r == 10);
}
