#include <catch2/catch_test_macros.hpp>
#include <utils/gpu_string.h>

using kvhdf5::gpu_string;
using kvhdf5::gpu_string_view;

// ============================================================
// gpu_string_view — Construction
// ============================================================

TEST_CASE("gpu_string_view: default constructor is empty", "[gpu_string_view]") {
    gpu_string_view v;
    CHECK(v.empty());
    CHECK(v.size() == 0);
    CHECK(v.data() == nullptr);
}

TEST_CASE("gpu_string_view: construct from string literal", "[gpu_string_view]") {
    gpu_string_view v("hello");
    CHECK(v.size() == 5);
    CHECK(v.data()[0] == 'h');
    CHECK(v.data()[4] == 'o');
}

TEST_CASE("gpu_string_view: construct from empty literal", "[gpu_string_view]") {
    gpu_string_view v("");
    CHECK(v.size() == 0);
    CHECK(v.empty());
    CHECK(v.data() != nullptr); // Points to the literal, just zero length
}

TEST_CASE("gpu_string_view: construct from pointer and length", "[gpu_string_view]") {
    const char* s = "abcdef";
    gpu_string_view v(s, 3);
    CHECK(v.size() == 3);
    CHECK(v[0] == 'a');
    CHECK(v[1] == 'b');
    CHECK(v[2] == 'c');
}

TEST_CASE("gpu_string_view: construct from pointer and length of zero", "[gpu_string_view]") {
    const char* s = "abc";
    gpu_string_view v(s, 0);
    CHECK(v.empty());
    CHECK(v.data() == s);
}

TEST_CASE("gpu_string_view: construct from null-terminated C string (runtime)", "[gpu_string_view]") {
    char buf[] = "runtime";
    const char* p = buf;
    // Force through the const char* constructor (not the array template)
    gpu_string_view v(p);
    CHECK(v.size() == 7);
    CHECK(v[0] == 'r');
}

TEST_CASE("gpu_string_view: construct from nullptr C string", "[gpu_string_view]") {
    const char* p = nullptr;
    gpu_string_view v(p);
    CHECK(v.empty());
    CHECK(v.data() == nullptr);
}

// ============================================================
// gpu_string_view — Accessors & Indexing
// ============================================================

TEST_CASE("gpu_string_view: indexing", "[gpu_string_view]") {
    gpu_string_view v("abcd");
    CHECK(v[0] == 'a');
    CHECK(v[1] == 'b');
    CHECK(v[2] == 'c');
    CHECK(v[3] == 'd');
}

TEST_CASE("gpu_string_view: data points to original memory", "[gpu_string_view]") {
    const char literal[] = "test";
    gpu_string_view v(literal);
    CHECK(v.data() == literal);
}

// ============================================================
// gpu_string_view — Iterators
// ============================================================

TEST_CASE("gpu_string_view: begin/end iteration", "[gpu_string_view]") {
    gpu_string_view v("abc");
    std::string collected(v.begin(), v.end());
    CHECK(collected == "abc");
}

TEST_CASE("gpu_string_view: range-for loop", "[gpu_string_view]") {
    gpu_string_view v("xyz");
    std::string collected;
    for (char c : v) collected += c;
    CHECK(collected == "xyz");
}

TEST_CASE("gpu_string_view: empty view has begin == end", "[gpu_string_view]") {
    gpu_string_view v;
    CHECK(v.begin() == v.end());
}

// ============================================================
// gpu_string_view — Equality
// ============================================================

TEST_CASE("gpu_string_view: equal views", "[gpu_string_view]") {
    gpu_string_view a("hello");
    gpu_string_view b("hello");
    CHECK(a == b);
    CHECK_FALSE(a != b);
}

TEST_CASE("gpu_string_view: different content", "[gpu_string_view]") {
    gpu_string_view a("hello");
    gpu_string_view b("world");
    CHECK(a != b);
    CHECK_FALSE(a == b);
}

TEST_CASE("gpu_string_view: different lengths", "[gpu_string_view]") {
    gpu_string_view a("abc");
    gpu_string_view b("abcd");
    CHECK(a != b);
    CHECK(b != a);
}

TEST_CASE("gpu_string_view: empty views are equal", "[gpu_string_view]") {
    gpu_string_view a;
    gpu_string_view b("");
    CHECK(a == b);
}

TEST_CASE("gpu_string_view: same prefix different lengths are not equal", "[gpu_string_view]") {
    gpu_string_view a("ab");
    gpu_string_view b("abc");
    CHECK(a != b);
}

// ============================================================
// gpu_string_view — Ordering
// ============================================================

TEST_CASE("gpu_string_view: lexicographic less-than", "[gpu_string_view]") {
    CHECK(gpu_string_view("abc") < gpu_string_view("abd"));
    CHECK(gpu_string_view("abc") < gpu_string_view("abcd"));
    CHECK(gpu_string_view("a") < gpu_string_view("b"));
    CHECK(gpu_string_view("") < gpu_string_view("a"));
}

TEST_CASE("gpu_string_view: not less-than when equal", "[gpu_string_view]") {
    CHECK_FALSE(gpu_string_view("abc") < gpu_string_view("abc"));
}

TEST_CASE("gpu_string_view: not less-than when greater", "[gpu_string_view]") {
    CHECK_FALSE(gpu_string_view("b") < gpu_string_view("a"));
    CHECK_FALSE(gpu_string_view("abcd") < gpu_string_view("abc"));
}

TEST_CASE("gpu_string_view: greater-than", "[gpu_string_view]") {
    CHECK(gpu_string_view("b") > gpu_string_view("a"));
    CHECK(gpu_string_view("abcd") > gpu_string_view("abc"));
    CHECK_FALSE(gpu_string_view("abc") > gpu_string_view("abc"));
}

TEST_CASE("gpu_string_view: less-than-or-equal", "[gpu_string_view]") {
    CHECK(gpu_string_view("abc") <= gpu_string_view("abc"));
    CHECK(gpu_string_view("abc") <= gpu_string_view("abd"));
    CHECK_FALSE(gpu_string_view("abd") <= gpu_string_view("abc"));
}

TEST_CASE("gpu_string_view: greater-than-or-equal", "[gpu_string_view]") {
    CHECK(gpu_string_view("abc") >= gpu_string_view("abc"));
    CHECK(gpu_string_view("abd") >= gpu_string_view("abc"));
    CHECK_FALSE(gpu_string_view("abc") >= gpu_string_view("abd"));
}

TEST_CASE("gpu_string_view: ordering with empty strings", "[gpu_string_view]") {
    gpu_string_view empty;
    gpu_string_view nonempty("a");

    CHECK(empty < nonempty);
    CHECK(empty <= nonempty);
    CHECK(nonempty > empty);
    CHECK(nonempty >= empty);
    CHECK(empty <= empty);
    CHECK(empty >= empty);
}

// ============================================================
// gpu_string_view — substr
// ============================================================

TEST_CASE("gpu_string_view: substr from beginning", "[gpu_string_view]") {
    gpu_string_view v("hello world");
    auto s = v.substr(0, 5);
    CHECK(s == gpu_string_view("hello"));
    CHECK(s.size() == 5);
}

TEST_CASE("gpu_string_view: substr from middle", "[gpu_string_view]") {
    gpu_string_view v("hello world");
    auto s = v.substr(6, 5);
    CHECK(s == gpu_string_view("world"));
}

TEST_CASE("gpu_string_view: substr to end (no count)", "[gpu_string_view]") {
    gpu_string_view v("hello world");
    auto s = v.substr(6);
    CHECK(s == gpu_string_view("world"));
    CHECK(s.size() == 5);
}

TEST_CASE("gpu_string_view: substr with count exceeding length", "[gpu_string_view]") {
    gpu_string_view v("abc");
    auto s = v.substr(1, 100);
    CHECK(s == gpu_string_view("bc"));
    CHECK(s.size() == 2);
}

TEST_CASE("gpu_string_view: substr at pos == length is empty", "[gpu_string_view]") {
    gpu_string_view v("abc");
    auto s = v.substr(3);
    CHECK(s.empty());
}

TEST_CASE("gpu_string_view: substr at pos beyond length is empty", "[gpu_string_view]") {
    gpu_string_view v("abc");
    auto s = v.substr(10);
    CHECK(s.empty());
}

TEST_CASE("gpu_string_view: substr of entire string", "[gpu_string_view]") {
    gpu_string_view v("hello");
    auto s = v.substr(0);
    CHECK(s == v);
}

TEST_CASE("gpu_string_view: substr single character", "[gpu_string_view]") {
    gpu_string_view v("abcde");
    auto s = v.substr(2, 1);
    CHECK(s.size() == 1);
    CHECK(s[0] == 'c');
}

// ============================================================
// gpu_string_view — Edge cases
// ============================================================

TEST_CASE("gpu_string_view: single character string", "[gpu_string_view]") {
    gpu_string_view v("x");
    CHECK(v.size() == 1);
    CHECK(v[0] == 'x');
    CHECK_FALSE(v.empty());
}

TEST_CASE("gpu_string_view: string with embedded concepts (spaces, special chars)", "[gpu_string_view]") {
    gpu_string_view v("hello\tworld\n!");
    CHECK(v.size() == 13);
    CHECK(v[5] == '\t');
    CHECK(v[11] == '\n');
}

TEST_CASE("gpu_string_view: binary data with embedded null", "[gpu_string_view]") {
    // Using pointer+length constructor, we can hold binary data
    const char data[] = {'a', '\0', 'b'};
    gpu_string_view v(data, 3);
    CHECK(v.size() == 3);
    CHECK(v[0] == 'a');
    CHECK(v[1] == '\0');
    CHECK(v[2] == 'b');
}

TEST_CASE("gpu_string_view: two views into same buffer are independent", "[gpu_string_view]") {
    const char* buf = "abcdef";
    gpu_string_view a(buf, 3);     // "abc"
    gpu_string_view b(buf + 3, 3); // "def"
    CHECK(a == gpu_string_view("abc"));
    CHECK(b == gpu_string_view("def"));
    CHECK(a != b);
    CHECK(a < b);
}

// ============================================================
// gpu_string — Construction
// ============================================================

TEST_CASE("gpu_string: default constructor is empty", "[gpu_string]") {
    gpu_string<> s;
    CHECK(s.empty());
    CHECK(s.size() == 0);
    CHECK(s.c_str()[0] == '\0');
}

TEST_CASE("gpu_string: construct from literal", "[gpu_string]") {
    gpu_string<> s("hello");
    CHECK(s.size() == 5);
    CHECK(s == gpu_string_view("hello"));
    CHECK(s.c_str()[5] == '\0'); // null terminated
}

TEST_CASE("gpu_string: construct from empty literal", "[gpu_string]") {
    gpu_string<> s("");
    CHECK(s.empty());
    CHECK(s.c_str()[0] == '\0');
}

TEST_CASE("gpu_string: construct from view", "[gpu_string]") {
    gpu_string_view v("test string");
    gpu_string<> s(v);
    CHECK(s == v);
    CHECK(s.size() == v.size());
}

TEST_CASE("gpu_string: construct from view truncates when too long", "[gpu_string]") {
    gpu_string_view v("this is a long string");
    gpu_string<5> s(v);
    CHECK(s.size() == 5);
    CHECK(s == gpu_string_view("this "));
    CHECK(s.c_str()[5] == '\0');
}

TEST_CASE("gpu_string: small capacity", "[gpu_string]") {
    gpu_string<3> s("abc");
    CHECK(s.size() == 3);
    CHECK(s == gpu_string_view("abc"));
}

// ============================================================
// gpu_string — Conversion to view
// ============================================================

TEST_CASE("gpu_string: implicit conversion to view", "[gpu_string]") {
    gpu_string<> s("hello");
    gpu_string_view v = s;
    CHECK(v.size() == 5);
    CHECK(v == gpu_string_view("hello"));
}

TEST_CASE("gpu_string: view points to string's internal buffer", "[gpu_string]") {
    gpu_string<> s("hello");
    gpu_string_view v = s;
    CHECK(v.data() == s.data());
}

// ============================================================
// gpu_string — Comparisons (through view delegation)
// ============================================================

TEST_CASE("gpu_string: equality with view", "[gpu_string]") {
    gpu_string<> s("hello");
    CHECK(s == gpu_string_view("hello"));
    CHECK_FALSE(s == gpu_string_view("world"));
    CHECK(s != gpu_string_view("world"));
    CHECK_FALSE(s != gpu_string_view("hello"));
}

TEST_CASE("gpu_string: equality between two gpu_strings", "[gpu_string]") {
    gpu_string<> a("hello");
    gpu_string<> b("hello");
    gpu_string<> c("world");
    CHECK(a == b);
    CHECK(a != c);
}

TEST_CASE("gpu_string: ordering with view", "[gpu_string]") {
    gpu_string<> s("bbb");
    CHECK(s < gpu_string_view("ccc"));
    CHECK(s > gpu_string_view("aaa"));
    CHECK(s <= gpu_string_view("bbb"));
    CHECK(s >= gpu_string_view("bbb"));
    CHECK(s <= gpu_string_view("ccc"));
    CHECK(s >= gpu_string_view("aaa"));
}

TEST_CASE("gpu_string: ordering between two gpu_strings", "[gpu_string]") {
    gpu_string<> a("abc");
    gpu_string<> b("abd");
    CHECK(a < b);
    CHECK(b > a);
}

// ============================================================
// gpu_string — Mutation
// ============================================================

TEST_CASE("gpu_string: append view", "[gpu_string]") {
    gpu_string<10> s("hello");
    bool ok = s.append(" world");  // Wait, " world" is 6 chars, total 11 > 10
    CHECK_FALSE(ok);
    CHECK(s == gpu_string_view("hello")); // Unchanged on failure

    ok = s.append("!");
    CHECK(ok);
    CHECK(s == gpu_string_view("hello!"));
}

TEST_CASE("gpu_string: append to exact capacity", "[gpu_string]") {
    gpu_string<8> s("hello");
    bool ok = s.append("!!!");
    CHECK(ok);
    CHECK(s.size() == 8);
    CHECK(s == gpu_string_view("hello!!!"));

    ok = s.append("x");
    CHECK_FALSE(ok);
}

TEST_CASE("gpu_string: append empty view", "[gpu_string]") {
    gpu_string<> s("hello");
    bool ok = s.append(gpu_string_view());
    CHECK(ok);
    CHECK(s == gpu_string_view("hello"));
}

TEST_CASE("gpu_string: push_back", "[gpu_string]") {
    gpu_string<3> s;
    CHECK(s.push_back('a'));
    CHECK(s.push_back('b'));
    CHECK(s.push_back('c'));
    CHECK_FALSE(s.push_back('d'));
    CHECK(s == gpu_string_view("abc"));
}

TEST_CASE("gpu_string: push_back maintains null terminator", "[gpu_string]") {
    gpu_string<5> s;
    s.push_back('x');
    CHECK(s.c_str()[1] == '\0');
    s.push_back('y');
    CHECK(s.c_str()[2] == '\0');
}

TEST_CASE("gpu_string: clear", "[gpu_string]") {
    gpu_string<> s("hello");
    s.clear();
    CHECK(s.empty());
    CHECK(s.size() == 0);
    CHECK(s.c_str()[0] == '\0');
}

TEST_CASE("gpu_string: clear then reuse", "[gpu_string]") {
    gpu_string<10> s("hello");
    s.clear();
    CHECK(s.append("world"));
    CHECK(s == gpu_string_view("world"));
}

// ============================================================
// gpu_string — Iterators
// ============================================================

TEST_CASE("gpu_string: begin/end", "[gpu_string]") {
    gpu_string<> s("abc");
    std::string collected(s.begin(), s.end());
    CHECK(collected == "abc");
}

TEST_CASE("gpu_string: cbegin/cend", "[gpu_string]") {
    gpu_string<> s("abc");
    std::string collected(s.cbegin(), s.cend());
    CHECK(collected == "abc");
}

TEST_CASE("gpu_string: range-for loop", "[gpu_string]") {
    gpu_string<> s("xyz");
    std::string collected;
    for (char c : s) collected += c;
    CHECK(collected == "xyz");
}

TEST_CASE("gpu_string: mutable indexing", "[gpu_string]") {
    gpu_string<> s("abc");
    s[1] = 'X';
    CHECK(s == gpu_string_view("aXc"));
}

// ============================================================
// gpu_string — max_size
// ============================================================

TEST_CASE("gpu_string: max_size reflects template parameter", "[gpu_string]") {
    CHECK(gpu_string<10>::max_size() == 10);
    CHECK(gpu_string<255>::max_size() == 255);
    CHECK(gpu_string<1>::max_size() == 1);
}
