// SPDX-License-Identifier: AGPL-3.0-only

#include "../w4a16_gemv.cu"
#include "w4a16_gemv_tail_fixture_support.cuh"
#include "w4a16_gemv_tail_fixture_cases.cuh"

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr << "usage: fixture (--write-oracle|--check-oracle) PATH\n";
            return 2;
        }
        FixtureData data;
        if (std::string(argv[1]) == "--write-oracle") {
            std::ofstream output(argv[2]);
            if (!output) throw std::runtime_error("cannot create oracle");
            for (const auto& spec : CASES) {
                std::cout << "oracle " << spec.name << " N=" << spec.width << std::endl;
                write_record(output, spec, run_case(spec, spec.width, data));
            }
        } else if (std::string(argv[1]) == "--check-oracle") {
            const auto oracle = read_oracle(argv[2]);
            for (const auto& spec : CASES) {
                const auto found = oracle.find(spec.name);
                if (found == oracle.end()) throw std::runtime_error(std::string(spec.name) + ": missing oracle");
                for (unsigned int n = 1; n <= spec.width; ++n) {
                    std::cout << "check " << spec.name << " N=" << n << std::endl;
                    compare_prefix(spec, n, run_case(spec, n, data), found->second);
                }
            }
        } else {
            throw std::runtime_error("unknown mode");
        }
        std::cout << "PASS w4a16 barrier-tail fixture" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
