#include <gtest/gtest.h>



GTEST_API_ int main(int argc, char** argv) {
    std::cout << "Running main() from test_main.cc\n";

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
